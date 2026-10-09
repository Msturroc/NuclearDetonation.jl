# Simulation wrapper for the GUI
# Wraps NuclearDetonation.jl API into a single callable function

using NuclearDetonation
using NuclearDetonation.Transport
using NCDatasets
using StaticArrays
using Random
using Dates

# --- Particle size helpers (from nancy_bomb_release.jl) ---

function snap_settling_velocity(d_um::Float64)
    snap_d = [2.2, 4.4, 8.6, 14.6, 22.8, 36.1, 56.5, 92.3, 173.2]
    snap_v = [0.2, 0.7, 2.5, 6.9, 15.9, 35.6, 71.2, 137.0, 277.3]
    log_d = log.(snap_d)
    log_v = log.(snap_v)
    ld = log(d_um)
    if ld <= log_d[1]
        slope = (log_v[2] - log_v[1]) / (log_d[2] - log_d[1])
        return exp(log_v[1] + slope * (ld - log_d[1]))
    elseif ld >= log_d[end]
        slope = (log_v[end] - log_v[end-1]) / (log_d[end] - log_d[end-1])
        return exp(log_v[end] + slope * (ld - log_d[end]))
    end
    i = searchsortedlast(log_d, ld)
    i = clamp(i, 1, length(log_d) - 1)
    frac = (ld - log_d[i]) / (log_d[i+1] - log_d[i])
    return exp(log_v[i] + frac * (log_v[i+1] - log_v[i]))
end

function generate_bimodal_bins(d_fine, sg_fine, d_coarse, sg_coarse; n_bins=15)
    log_d_min = min(log(d_fine) - 3*log(sg_fine), log(d_coarse) - 3*log(sg_coarse))
    log_d_max = max(log(d_fine) + 3*log(sg_fine), log(d_coarse) + 3*log(sg_coarse))
    log_d_min = max(log_d_min, log(1.0))
    log_d_max = min(log_d_max, log(500.0))
    d_centres = exp.(range(log_d_min, log_d_max, length=n_bins))
    [(d=d, v=snap_settling_velocity(d)) for d in d_centres]
end

function compute_bimodal_weights(d_fine, sg_fine, d_coarse, sg_coarse, frac_fine, bins)
    weights = Float64[]
    for bin in bins
        ld = log(bin.d)
        w_fine = exp(-0.5 * ((ld - log(d_fine)) / log(sg_fine))^2) / log(sg_fine)
        w_coarse = exp(-0.5 * ((ld - log(d_coarse)) / log(sg_coarse))^2) / log(sg_coarse)
        push!(weights, frac_fine * w_fine + (1.0 - frac_fine) * w_coarse)
    end
    weights ./= sum(weights)
    weights
end

function gaussian_smooth(field::Matrix{T}, sigma::Real; truncate::Real=4.0) where T
    radius = ceil(Int, sigma * truncate)
    kernel_1d = [exp(-0.5 * (x / sigma)^2) for x in -radius:radius]
    kernel_1d ./= sum(kernel_1d)
    nx, ny = size(field)
    temp = zeros(T, nx, ny)
    smoothed = zeros(T, nx, ny)
    for j in 1:ny, i in 1:nx
        val, weight = zero(T), zero(T)
        for k in -radius:radius
            ii = i + k
            if 1 <= ii <= nx
                w = kernel_1d[k + radius + 1]
                val += field[ii, j] * w; weight += w
            end
        end
        temp[i, j] = weight > 0 ? val / weight : zero(T)
    end
    for i in 1:nx, j in 1:ny
        val, weight = zero(T), zero(T)
        for k in -radius:radius
            jj = j + k
            if 1 <= jj <= ny
                w = kernel_1d[k + radius + 1]
                val += temp[i, jj] * w; weight += w
            end
        end
        smoothed[i, j] = weight > 0 ? val / weight : zero(T)
    end
    return smoothed
end

# Resolve a path under data/ at runtime. Source-tree layout uses pkgdir,
# bundled (PackageCompiler) layout puts data next to the executable.
function _resolve_bundled_path(relpath::String)
    pkg = pkgdir(NuclearDetonation)
    src_candidate = pkg === nothing ? "" : joinpath(pkg, "data", relpath)
    isfile(src_candidate) && return src_candidate
    bundled = joinpath(dirname(Sys.BINDIR), "data", relpath)
    isfile(bundled) && return bundled
    # Last resort: return source-tree candidate so the downstream error names a real path
    return isempty(src_candidate) ? bundled : src_candidate
end

# --- Weather data ---

const ACTIVE_DATASET = Ref{String}("")
const ERA5_STATE = Ref{Any}(nothing)   # MetData for the active built-in dataset
const ARL_STATE = Ref{Any}(nothing)    # MetData converted for the last ARL run

"""Weather data in the ERA5 NetCDF layout, with some time steps pre-read into memory."""
struct MetData
    files::Vector{String}
    times::Vector{Vector{Dates.DateTime}}   # met times in each file
    met_format::Any
    met_cache::Dict{Tuple{Int,Int}, Transport.MeteoFields}
    nx_met::Int
    ny_met::Int
    nk_met::Int
    lat_range::Vector{Float64}
    lon_range::Vector{Float64}
    tmpdir::String                          # converted ARL data to clean up ("" for ERA5)
end

# Built-in datasets: name → (file list, files to pre-read, label)
const DATASET_CONFIGS = Dict(
    "nancy" => (files_fn=nancy_era5_files, cache_start=5, cache_end=11, label="Nancy (NTS)"),
    "etex"  => (files_fn=etex_era5_files,  cache_start=5, cache_end=19, label="ETEX (Europe)"),
)

function load_met_data(files::Vector{String}; cache_files = eachindex(files), tmpdir = "",
                       report = (pct, msg) -> nothing)
    met_format = Transport.detect_met_format(files[1])
    dims, lat_range, lon_range = NCDataset(files[1]) do ds
        Transport.get_met_dimensions(met_format, ds),
        Float64.(ds["latitude"][:]), Float64.(ds["longitude"][:])
    end
    times = [NCDataset(ds -> Dates.DateTime.(ds["time"][:]), f) for f in files]

    met_cache = Dict{Tuple{Int,Int}, Transport.MeteoFields}()
    cached = collect(cache_files)
    for (k, file_idx) in enumerate(cached)
        report(round(Int, 100 * (k - 1) / length(cached)), "Reading weather data ($k/$(length(cached)))...")
        NCDataset(files[file_idx]) do ds
            n = length(times[file_idx])
            for t_idx in 1:n
                mf = Transport.MeteoFields(dims..., T=Float32)
                Transport.read_initial_met_fields!(met_format, mf, ds, t_idx, min(t_idx + 1, n))
                met_cache[(file_idx, t_idx)] = mf
            end
        end
    end
    return MetData(files, times, met_format, met_cache, dims..., lat_range, lon_range, tmpdir)
end

function preload_era5!(; dataset::String="nancy", progress_callback=nothing)
    report = something(progress_callback, (pct, msg) -> nothing)
    cfg = get(DATASET_CONFIGS, dataset, nothing)
    isnothing(cfg) && error("Unknown dataset: $dataset. Available: $(join(keys(DATASET_CONFIGS), ", "))")
    if ACTIVE_DATASET[] == dataset && !isnothing(ERA5_STATE[])
        report(100, "$(cfg.label) ERA5 data already loaded")
        return ERA5_STATE[]
    end

    report(0, "Loading $(cfg.label) ERA5 data...")
    files = cfg.files_fn()
    ERA5_STATE[] = load_met_data(files; cache_files = cfg.cache_start:min(cfg.cache_end, length(files)),
                                 report)
    ACTIVE_DATASET[] = dataset
    report(100, "$(cfg.label) ERA5 data ready")
    return ERA5_STATE[]
end

"""
    met_start(met, start) -> (file_idx, time_idx, offset_s)

Met window a run starting at `start` begins in, and how long after that window's
start the release happens. Windows only exist between time steps within a file,
so a start on a file's last step begins one window earlier.
"""
function met_start(met::MetData, start::Dates.DateTime)
    best = nothing
    for (f, ts) in enumerate(met.times), t in 1:length(ts)-1
        ts[t] <= start || continue
        (best === nothing || ts[t] > best[3]) && (best = (f, t, ts[t]))
    end
    best === nothing && error("Start time $start is before the weather data begins")
    f, t, t0 = best
    return f, t, Dates.value(start - t0) / 1000.0
end

function _met_fields(met::MetData, f::Int, t::Int)
    haskey(met.met_cache, (f, t)) && return met.met_cache[(f, t)]
    return NCDataset(met.files[f]) do ds
        n = length(met.times[f])
        mf = Transport.MeteoFields(met.nx_met, met.ny_met, met.nk_met, T=Float32)
        Transport.read_initial_met_fields!(met.met_format, mf, ds, t, min(t + 1, n))
        mf
    end
end

# --- Results ---

struct SimulationResult
    dose_grid::Matrix{Float64}
    lon_grid::StepRangeLen{Float64}
    lat_grid::StepRangeLen{Float64}
    max_dose::Float64
    deposition_log::Vector
    smooth_sigma::Float64
    domain::Any                 # SimulationDomain, kept for CSV export
    units::String               # "mSv/h" or "kBq/m²"
    components::Vector{String}  # nuclide name per component index
    release_offset_s::Float64   # model time of the release
end

# Isotope half-life lookup (hours). Names not listed here need an explicit
# half-life in the source-term entry.
const ISOTOPE_HALFLIVES = Dict{String,Float64}(
    "Cs-137"  => 30.17 * 365.25 * 24.0,   # 264,357 h
    "I-131"   => 8.02 * 24.0,              # 192.5 h
    "Sr-90"   => 28.9 * 365.25 * 24.0,     # 253,066 h
    "Cs-134"  => 2.062 * 365.25 * 24.0,
    "Co-60"   => 5.27 * 365.25 * 24.0,
    "I-133"   => 20.8,
    "Ru-103"  => 39.26 * 24.0,
    "Ru-106"  => 373.6 * 24.0,
    "Te-132"  => 3.20 * 24.0,
    "Xe-133"  => 5.25 * 24.0,
    "Pu-239"  => 24110 * 365.25 * 24.0,
    "Generic" => 0.0,                       # NoDecay
)

"""Half-life in hours for each isotope: `NaN` means the ISOTOPE_HALFLIVES preset, 0 no decay."""
_halflives(isotopes, halflives_hours) =
    [isnan(h) ? get(ISOTOPE_HALFLIVES, i, 0.0) : h for (i, h) in zip(isotopes, halflives_hours)]

_decay_params(halflives) =
    [hl > 0.0 ? Transport.DecayParams(kdecay=Transport.ExponentialDecay, halftime_hours=hl) :
                Transport.DecayParams(kdecay=Transport.NoDecay, halftime_hours=0.0)
     for hl in halflives]

# --- Release setups ---

const NANCY_YIELD_KT = 24.0
const EARTH_RADIUS_M = 6_371_000.0

"""
    cloud_height_scale(yield_kt)

Stabilised cloud height relative to the calibrated Nancy geometry (12.5 km top at
24 kt). Below 24 kt cloud height roughly doubles per tenfold yield (Glasstone &
Dolan §9.96, exponent 0.30); above it growth flattens as the cloud reaches the
stratosphere, to about 20 km at 1.2 Mt (exponent 0.12).
"""
cloud_height_scale(yield_kt) = (yield_kt / NANCY_YIELD_KT)^(yield_kt <= NANCY_YIELD_KT ? 0.30 : 0.12)

"""Cloud radius relative to Nancy, with the usual cube-root yield scaling."""
cloud_radius_scale(yield_kt) = cbrt(yield_kt / NANCY_YIELD_KT)

"""Bomb burst: the calibrated Nancy setup, with activity and cloud size scaled by yield."""
function _bomb_setup(yield_kt, n_particles, release_x, release_y)
    params = nancy_optimised_config()
    p, lf, ps = params.physics_scales, params.layer_fractions, params.particle_size_config
    total_activity = params.activity_Bq * (yield_kt / NANCY_YIELD_KT)

    # 3-layer NOAA release geometry
    hs, rs = cloud_height_scale(yield_kt), cloud_radius_scale(yield_kt)
    layers = [CylinderRelease(0.0, 3800.0hs, 537.0rs),
              CylinderRelease(3800.0hs, 6100.0hs, 1500.0rs),
              CylinderRelease(6100.0hs, 12500.0hs, 2500.0rs)]
    n_lower  = round(Int, n_particles * lf.lower)
    n_middle = round(Int, n_particles * lf.middle)
    counts = [n_lower, n_middle, n_particles - n_lower - n_middle]
    fractions = [lf.lower, lf.middle, lf.upper]
    sources = [ReleaseSource((release_x, release_y), layer, BombRelease(0.0),
                             [total_activity * f], max(n, 1))
               for (layer, f, n) in zip(layers, fractions, counts)]

    # Bimodal particle size distribution
    size_bins = generate_bimodal_bins(ps.d_median_fine_μm, ps.sigma_g_fine,
                                      ps.d_median_coarse_μm, ps.sigma_g_coarse)
    bin_weights = compute_bimodal_weights(ps.d_median_fine_μm, ps.sigma_g_fine,
                                          ps.d_median_coarse_μm, ps.sigma_g_coarse,
                                          ps.frac_fine, size_bins)

    hanna = HannaTurbulenceConfig{Float64}(
        sigma_scale=p.sigma_h_scale, sigma_scale_vertical=p.sigma_w_scale,
        tl_scale=p.tl_scale, use_cbl=true)
    dep = Transport.DepositionConfig{Float64}(
        apply_dry_deposition=true, apply_wet_deposition=false,
        use_simple_deposition=true, simple_deposition_velocity=0.002 * p.vd_scale,
        simple_surface_height=30.0 * p.surface_height_scale,
        mixing_height=1000.0 * p.mixing_height_scale,
        surface_roughness=0.1 * p.roughness_scale)

    return (; bomb = true, sources, components = ["MixedFP"], halflives = [0.0],
              size_bins, cum_weights = cumsum(bin_weights), vgrav_scale = p.vgrav_scale,
              hanna, dep, omega_scale = p.omega_scale, settling = true,
              release_top_m = 12500.0hs, smooth_sigma = p.smooth_sigma)
end

"""Point release (NPP stack): one source per nuclide, with the ETEX-calibrated transport."""
function _point_setup(isotopes, activities_tbq, halflives, stack_height_m, n_particles,
                      release_x, release_y)
    half_h = 5.0
    geometry = ColumnRelease(max(stack_height_m - half_h, 0.0), stack_height_m + half_h)
    n_per_iso = max(1, div(n_particles, length(isotopes)))
    # BombRelease puts each source's whole activity (Bq) into its particles;
    # release timing is handled by per-particle release times instead.
    sources = [ReleaseSource((release_x, release_y), geometry, BombRelease(0.0),
                             [a * 1e12], n_per_iso) for a in activities_tbq]

    ep = etex_optimised_config().physics_scales
    hanna = HannaTurbulenceConfig{Float64}(
        sigma_scale=ep.sigma_h_scale, sigma_scale_vertical=ep.sigma_w_scale,
        tl_scale=ep.tl_scale, use_cbl=true)
    dep = Transport.DepositionConfig{Float64}(
        apply_dry_deposition=true, apply_wet_deposition=false,
        use_simple_deposition=true, simple_deposition_velocity=0.002,
        mixing_height=1000.0 * ep.mixing_height_scale,
        surface_roughness=0.1 * ep.roughness_scale)

    return (; bomb = false, sources, components = isotopes, halflives,
              hanna, dep, omega_scale = ep.omega_scale, settling = false,
              release_top_m = stack_height_m + half_h, smooth_sigma = 2.0)
end

"""
Add the setup's particles to `state`, converting release heights to sigma with the
initial met fields. Returns the particle size config and per-particle release times.
"""
function _place_particles!(state, setup, rng, met, init_met, release_x, release_y;
                           release_start_s, release_duration_s)
    ncomp = length(setup.components)
    radii, densities, size_idx, release_times = Float64[], Float64[], Int[], Float64[]
    domain = state.domain
    for (isrc, src) in enumerate(setup.sources)
        pos_s, act_s, released = Transport.generate_release_particles(
            rng, src, 0, 1,
            ones(Float64, met.nx_met, met.ny_met), ones(Float64, met.nx_met, met.ny_met),
            domain.dx, domain.dy, domain.hlevel)
        (released && !isempty(pos_s)) || continue
        icomp = setup.bomb ? 1 : isrc
        for (pos, activity) in zip(pos_s, act_s)
            sigma_z = Transport.height_to_sigma_hybrid(release_x, release_y, pos[3], init_met, 0.0)
            mass = zeros(Float64, ncomp); mass[icomp] = activity
            Transport.add_particle!(state.ensemble,
                SVector{3,Float64}(pos[1], pos[2], sigma_z), SVector{3,Float64}(0.0, 0.0, 0.0),
                mass, 0.0, icomp=icomp)
            push!(release_times, release_start_s +
                  (release_duration_s > 0 ? rand(rng) * release_duration_s : 0.0))
            if setup.bomb
                idx = clamp(searchsortedfirst(setup.cum_weights, rand(rng)), 1, length(setup.size_bins))
                push!(radii, setup.size_bins[idx].d * 0.5e-6)
                push!(densities, 2500.0)
                push!(size_idx, idx)
                state.ensemble.particles[end].grv = Float32(
                    setup.size_bins[idx].v * 0.01 * setup.vgrav_scale)
            else
                push!(radii, 5.0 * 0.5e-6)
                push!(densities, 2000.0)
                push!(size_idx, 1)
            end
        end
    end
    psc = if setup.bomb
        ParticleSizeConfig(
            size_bins=[ParticleProperties(diameter_μm=b.d, density_gcm3=2.5) for b in setup.size_bins],
            particle_radii=radii, particle_densities=densities, particle_size_indices=size_idx,
            fixed_gravity_cm_s=[b.v * setup.vgrav_scale for b in setup.size_bins])
    else
        ParticleSizeConfig(size_bins=[ParticleProperties(diameter_μm=5.0, density_gcm3=2.0)],
            particle_radii=radii, particle_densities=densities, particle_size_indices=size_idx)
    end
    return psc, release_times
end

"""
Grid deposition events onto a regular lon/lat grid as activity per m², using the
true area of each grid row (cells shrink with cos(latitude)).
"""
function _deposition_density(deposition_log, domain, lon_grid, lat_grid, weight)
    nx, ny = length(lon_grid), length(lat_grid)
    field = zeros(nx, ny)
    for evt in deposition_log
        elat, elon = Transport.grid_to_latlon(domain, evt.x, evt.y)
        elon > 180.0 && (elon -= 360.0)
        i = searchsortedlast(lon_grid, elon)
        j = searchsortedlast(lat_grid, elat)
        (1 <= i <= nx && 1 <= j <= ny) && (field[i, j] += weight(evt))
    end
    dlon, dlat = step(lon_grid), step(lat_grid)
    for j in 1:ny
        dy = EARTH_RADIUS_M * deg2rad(dlat)
        dx = EARTH_RADIUS_M * deg2rad(dlon) * cosd(lat_grid[j] + dlat / 2)
        field[:, j] ./= dx * dy
    end
    return field
end

# --- Unified simulation entry point ---

# Runtime lookup — a const would bake the build machine's temp dir into the app
_trace_file() = joinpath(tempdir(), "nucdet_gui_particles_trace.csv")

"""Model time (s) the met files can cover from window `t0` of file `f0` onwards."""
function _available_run_s(met::MetData, f0, t0)
    total = 0.0
    for f in f0:length(met.times), t in (f == f0 ? t0 : 1):length(met.times[f])-1
        total += Dates.value(met.times[f][t+1] - met.times[f][t]) / 1000.0
    end
    return total
end

"""
    run_simulation_with_source(; weather_source, arl_dir, lat, lon, start_date, start_hour,
                               duration_hours, n_particles, release_mode, ...)

Run a bomb burst (`release_mode = "bomb"`, `yield_kt`) or a point release
(`"npp"`: `isotopes`, `activities_tbq`, `halflives_hours`, `stack_height_m`,
`release_duration_hours`) on the built-in ERA5 data or local ARL files.
`duration_hours` counts from the release. Bomb results are dose rate (mSv/h) at
H+duration; point releases give deposition (kBq/m²) at the end, decayed per nuclide.
"""
function run_simulation_with_source(;
    weather_source::String = "era5",
    arl_dir::String = "",
    lat::Float64, lon::Float64,
    start_date::String, start_hour::Int,
    duration_hours::Int, n_particles::Int,
    release_mode::String = "bomb",
    yield_kt::Float64 = 24.0,
    isotopes::Vector{String} = ["Cs-137"],
    activities_tbq::Vector{Float64} = [1.0],
    halflives_hours::Vector{Float64} = fill(NaN, length(isotopes)),
    stack_height_m::Float64 = 100.0,
    release_duration_hours::Float64 = 1.0,
    progress_callback = nothing,
)
    update!(pct, msg) = isnothing(progress_callback) || progress_callback(pct, msg)
    start = Dates.DateTime(Dates.Date(start_date)) + Dates.Hour(start_hour)

    if weather_source == "arl"
        update!(2, "Preparing ARL weather data...")
        met = prepare_arl_simulation!(arl_dir, lat, lon, start_date, start_hour, duration_hours;
                  progress_callback = (pct, msg) -> update!(2 + div(pct, 10), msg))
    else
        met = ERA5_STATE[]
        isnothing(met) && error("ERA5 data not loaded. Call preload_era5!() first.")
    end
    update!(15, "Setting up simulation domain...")

    f0, t0, offset_s = met_start(met, start)
    run_s = offset_s + duration_hours * 3600.0
    available_s = _available_run_s(met, f0, t0)
    run_s <= available_s ||
        error("The weather data only covers $(round(Int, (available_s - offset_s) / 3600)) h " *
              "after this start time — shorten the duration or start earlier")

    # Transport works in the met file's longitude convention. ARL subsets are
    # contiguous; if any longitude is negative, shift the lot by +360 so the range
    # increases monotonically and doesn't wrap around 0.
    lons = met.lon_range
    shift = weather_source == "arl" && any(<(0), lons) ? 360.0 : 0.0
    domain = Transport.SimulationDomain(
        lon_min = minimum(lons) + shift, lon_max = maximum(lons) + shift,
        lat_min = minimum(met.lat_range), lat_max = maximum(met.lat_range),
        z_min = 0.0, z_max = 35000.0,
        nx = met.nx_met, ny = met.ny_met, nz = met.nk_met,
        start_time = met.times[f0][t0],
        end_time = met.times[f0][t0] + Dates.Second(round(Int, run_s)),
    )
    release_x, release_y = Transport.latlon_to_grid(domain, lat, lon + shift)

    update!(18, "Generating particles...")
    if release_mode == "npp"
        halflives = _halflives(isotopes, halflives_hours)
        setup = _point_setup(isotopes, activities_tbq, halflives, stack_height_m,
                             n_particles, release_x, release_y)
        release_duration_s = release_duration_hours * 3600.0
    else
        setup = _bomb_setup(yield_kt, n_particles, release_x, release_y)
        release_duration_s = 0.0
    end
    decay_params = _decay_params(setup.halflives)
    state = Transport.initialize_simulation(domain, setup.sources, setup.components, decay_params;
                                            log_depositions=true)
    rng = Random.MersenneTwister(42)
    psc, release_times = _place_particles!(state, setup, rng, met, _met_fields(met, f0, t0),
                                           release_x, release_y;
                                           release_start_s = offset_s, release_duration_s)

    update!(25, "Running $(duration_hours)-hour simulation ($(length(state.ensemble.particles)) particles)...")
    num_cfg = ERA5NumericalConfig{Float64}(
        interpolation_order=Transport.LinearInterp, ode_solver_type=:Euler, fixed_dt=300.0,
        turbulence=Transport.OrnsteinUhlenbeck)
    out_cfg = OutputConfig(trace_frequency=TRACE_DISABLED, verbosity=VERBOSITY_QUIET, trace_enabled=false)
    sim_cfg = Transport.SimulationConfig{Float64}(
        saveat=collect(0.0:3600.0:run_s), verbose=false, max_duration=run_s,
        save_snapshots=true, dt_particle=300.0, use_reference_stepping=true,
        max_files=length(met.files), omega_scale=setup.omega_scale, output_config=out_cfg)

    snapshots = Transport.run_simulation!(state, met.files,
        particle_size_config=psc, deposition_config=setup.dep,
        hanna_config=setup.hanna, decay_params=decay_params, config=sim_cfg,
        numerical_config=num_cfg, advection_enabled=true, settling_enabled=setup.settling,
        dry_deposition_enabled=true, wet_deposition_enabled=false,
        release_height_m=setup.release_top_m, met_data_cache=met.met_cache,
        met_format_override=met.met_format,
        met_dimensions=(met.nx_met, met.ny_met, met.nk_met),
        cache_init_file_idx=f0, cache_init_time_idx=t0,
        sigma_already_initialized=true, release_times_s=release_times,
        trace_filename=_trace_file())

    units = setup.bomb ? "mSv/h" : "kBq/m²"
    store_animation_data!(snapshots, domain, Float64[];
                          heights_m_ascending=Float64.(domain.hlevel),
                          release_mode, units, release_offset_s=offset_s)
    update!(85, setup.bomb ? "Computing dose rates..." : "Computing deposition...")

    display_lons = [l > 180.0 ? l - 360.0 : l for l in lons]
    lon_grid = range(minimum(display_lons), maximum(display_lons), step=0.023)
    lat_grid = range(minimum(met.lat_range), maximum(met.lat_range), step=0.018)

    if setup.bomb
        # Dose rate at H+duration from deposited fission products (Way-Wigner t^-1.2)
        K_DOSE = 1.9e-6
        field = _deposition_density(state.deposition_log, domain, lon_grid, lat_grid, evt -> evt.mass)
        field .*= K_DOSE * Float64(duration_hours)^(-1.2)
    else
        # Deposition at the end of the run: decay each deposit from when it landed
        λ = [hl > 0 ? log(2) / (hl * 3600.0) : 0.0 for hl in setup.halflives]
        weight(evt) = evt.mass * exp(-λ[evt.component] * (run_s - evt.time))
        field = _deposition_density(state.deposition_log, domain, lon_grid, lat_grid, weight)
        field ./= 1000.0  # Bq/m² → kBq/m²
    end
    smoothed = gaussian_smooth(field, setup.smooth_sigma)

    update!(95, "Generating contours...")
    return SimulationResult(smoothed, lon_grid, lat_grid, maximum(smoothed), state.deposition_log,
                            setup.smooth_sigma, domain, units, setup.components, offset_s)
end

# --- ARL weather data loading ---

"""
    load_arl_metadata!(dir_path; progress_callback)

Scan ARL directory and return bounds/date info. Does NOT convert data yet.
"""
function load_arl_metadata!(dir_path::String; progress_callback=nothing)
    isnothing(progress_callback) || progress_callback(0, "Scanning ARL directory...")
    bounds = get_arl_bounds(dir_path)
    isnothing(progress_callback) || progress_callback(100, "Found $(bounds.n_files) ARL files")
    return bounds
end

"""
    prepare_arl_simulation!(dir_path, lat, lon, start_date, start_hour, duration_hours;
                            progress_callback) -> MetData

Convert the ARL region around the release to ERA5-compatible NetCDF covering the
run, and read it into memory. Replaces (and deletes) the previous conversion.
"""
function prepare_arl_simulation!(dir_path::String, lat::Float64, lon::Float64,
                                  start_date::String, start_hour::Int, duration_hours::Int;
                                  progress_callback=nothing)
    update!(pct, msg) = isnothing(progress_callback) || progress_callback(pct, msg)
    sim_start_dt = Dates.DateTime(Dates.Date(start_date)) + Dates.Hour(start_hour)

    previous = ARL_STATE[]
    if previous !== nothing && isdir(previous.tmpdir)
        rm(previous.tmpdir; recursive=true, force=true)
    end
    ARL_STATE[] = nothing

    update!(5, "Converting ARL data to simulation format...")
    # Scale subsetting radius with duration — particles can travel ~500km/day
    # Base: 5° lat / 10° lon for 12h, scale up for longer simulations
    duration_scale = max(1.0, duration_hours / 12.0)
    rlat = min(5.0 * duration_scale, 30.0)
    rlon = min(10.0 * duration_scale, 60.0)
    nc_files, meta = convert_arl_region(dir_path, lat, lon, sim_start_dt, duration_hours;
                                        radius_lat=rlat, radius_lon=rlon,
                                        progress_callback=progress_callback)

    update!(90, "Building met cache...")
    met = load_met_data(nc_files; tmpdir = meta.tmpdir)
    # Keep the ARL subset's own longitudes (may be negative) for the domain
    met = MetData(met.files, met.times, met.met_format, met.met_cache, met.nx_met, met.ny_met,
                  met.nk_met, Float64.(meta.lat_range), Float64.(meta.lon_range), met.tmpdir)
    ARL_STATE[] = met
    update!(100, "ARL data ready")
    return met
end

"""
    export_deposition_csv(result::SimulationResult) -> String

Deposition events as CSV: position, activity, nuclide and time since release.
"""
function export_deposition_csv(result::SimulationResult)
    io = IOBuffer()
    println(io, "latitude,longitude,deposition_Bq,nuclide,time_since_release_s")
    for evt in result.deposition_log
        lat, lon = Transport.grid_to_latlon(result.domain, evt.x, evt.y)
        lon > 180.0 && (lon -= 360.0)
        nuclide = get(result.components, evt.component, "")
        println(io, "$(round(lat, digits=5)),$(round(lon, digits=5)),$(evt.mass),$nuclide,",
                evt.time - result.release_offset_s)
    end
    return String(take!(io))
end
