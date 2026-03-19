# Simulation wrapper for the web GUI
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

# --- Pre-loaded ERA5 data (populated by preload_era5!) ---

const ERA5_STATE = Ref{Any}(nothing)
const ACTIVE_DATASET = Ref{String}("nancy")

struct ERA5Data
    files::Vector{String}
    met_format::Any
    met_cache::Dict{Tuple{Int,Int}, Transport.MeteoFields}
    nx_met::Int
    ny_met::Int
    nk_met::Int
    lat_range::Vector{Float64}
    lon_range::Vector{Float64}
end

# --- ARL weather data state ---

const ARL_STATE = Ref{Any}(nothing)

struct ARLData
    dir_path::String
    files::Vector{String}          # converted NetCDF file paths
    met_format::Any
    met_cache::Dict{Tuple{Int,Int}, Transport.MeteoFields}
    nx_met::Int
    ny_met::Int
    nk_met::Int
    lat_range::Vector{Float64}
    lon_range::Vector{Float64}
    date_min::Date
    date_max::Date
    tmpdir::String
end

# etex_era5_files() is now provided by NuclearDetonation.Transport (data_access.jl)

# Dataset configuration: name → (file_loader, cache_range, label)
const DATASET_CONFIGS = Dict{String,NamedTuple{(:files_fn, :cache_start, :cache_end, :label),
                             Tuple{Function, Int, Int, String}}}(
    "nancy" => (files_fn=nancy_era5_files, cache_start=5, cache_end=11, label="Nancy (NTS)"),
    "etex"  => (files_fn=etex_era5_files,  cache_start=5, cache_end=24, label="ETEX (Europe)"),
)

function preload_era5!(; dataset::String="nancy", progress_callback=nothing)
    # Skip if already loaded
    if ACTIVE_DATASET[] == dataset && !isnothing(ERA5_STATE[])
        isnothing(progress_callback) || progress_callback(100, "$(DATASET_CONFIGS[dataset].label) already loaded")
        return ERA5_STATE[]
    end

    cfg = get(DATASET_CONFIGS, dataset, nothing)
    isnothing(cfg) && error("Unknown dataset: $dataset. Available: $(join(keys(DATASET_CONFIGS), ", "))")

    isnothing(progress_callback) || progress_callback(0, "Loading $(cfg.label) ERA5 data...")
    era5_files = cfg.files_fn()

    met_format = Transport.detect_met_format(era5_files[1])
    nx_met, ny_met, nk_met = NCDataset(era5_files[1]) do ds
        Transport.get_met_dimensions(met_format, ds)
    end

    lat_range, lon_range = NCDataset(era5_files[1]) do ds
        Float64.(ds["latitude"][:]), Float64.(ds["longitude"][:])
    end

    isnothing(progress_callback) || progress_callback(30, "Caching met fields...")
    met_cache = Dict{Tuple{Int,Int}, Transport.MeteoFields}()
    cache_end = min(cfg.cache_end, length(era5_files))
    for file_idx in cfg.cache_start:cache_end
        NCDataset(era5_files[file_idx]) do ds
            times = Transport.get_time_variable(met_format, ds)
            for t_idx in 1:length(times)
                mf = Transport.MeteoFields(nx_met, ny_met, nk_met, T=Float32)
                t2 = t_idx < length(times) ? t_idx + 1 : t_idx
                Transport.read_initial_met_fields!(met_format, mf, ds, t_idx, t2)
                met_cache[(file_idx, t_idx)] = mf
            end
        end
    end

    isnothing(progress_callback) || progress_callback(100, "$(cfg.label) ERA5 data ready")
    ERA5_STATE[] = ERA5Data(era5_files, met_format, met_cache,
                            nx_met, ny_met, nk_met, lat_range, lon_range)
    ACTIVE_DATASET[] = dataset
    return ERA5_STATE[]
end

# --- Main simulation function ---

struct SimulationResult
    dose_grid::Matrix{Float64}
    lon_grid::StepRangeLen{Float64}
    lat_grid::StepRangeLen{Float64}
    max_dose::Float64
    deposition_log::Vector
    smooth_sigma::Float64
    domain::Any  # SimulationDomain, kept for CSV export
    units::String  # "mSv/h" or "kBq/m²"
end

# Isotope half-life lookup (hours)
const ISOTOPE_HALFLIVES = Dict{String,Float64}(
    "Cs-137"  => 30.17 * 365.25 * 24.0,   # 264,357 h
    "I-131"   => 8.02 * 24.0,              # 192.5 h
    "Sr-90"   => 28.9 * 365.25 * 24.0,     # 253,066 h
    "Generic" => 0.0,                       # NoDecay
)

function run_dispersion_simulation(;
    lat::Float64 = 37.0956,
    lon::Float64 = -116.1028,
    yield_kt::Float64 = 24.0,
    start_date::String = "1953-03-24",
    start_hour::Int = 13,
    duration_hours::Int = 12,
    n_particles::Int = 5000,
    progress_callback = nothing,
    release_mode::String = "bomb",
    activity_tbq::Float64 = 1.0,
    stack_height_m::Float64 = 100.0,
    isotope::String = "Cs-137",
    release_duration_hours::Float64 = 1.0,
)
    era5 = ERA5_STATE[]
    isnothing(era5) && error("ERA5 data not loaded. Call preload_era5!() first.")

    update!(pct, msg) = isnothing(progress_callback) || progress_callback(pct, msg)

    update!(5, "Setting up simulation domain...")

    # Parse start time
    date_parts = parse.(Int, split(start_date, "-"))
    start_dt = Dates.DateTime(date_parts[1], date_parts[2], date_parts[3], start_hour, 0)

    # Create domain from ERA5 grid extents
    domain = Transport.SimulationDomain(
        lon_min = minimum(era5.lon_range), lon_max = maximum(era5.lon_range),
        lat_min = minimum(era5.lat_range), lat_max = maximum(era5.lat_range),
        z_min = 0.0, z_max = 35000.0,
        nx = era5.nx_met, ny = era5.ny_met, nz = era5.nk_met,
        start_time = start_dt,
        end_time = start_dt + Dates.Hour(duration_hours),
    )

    release_x, release_y = Transport.latlon_to_grid(domain, lat, lon)
    update!(10, "Configuring physics...")

    rng = Random.MersenneTwister(42)
    init_met = era5.met_cache[(5, 1)]

    if release_mode == "npp"
        # --- NPP (Point Release) mode ---
        half_h = 5.0
        geometry = ColumnRelease(stack_height_m - half_h, stack_height_m + half_h)
        total_activity = activity_tbq * 1e12  # TBq → Bq

        # Use a dummy source for initialization (actual particles are pre-generated below)
        source = ReleaseSource(
            (release_x, release_y), geometry,
            ConstantRelease(), [total_activity], n_particles,
        )

        # Decay setup
        hl = get(ISOTOPE_HALFLIVES, isotope, 0.0)
        if hl > 0.0
            decay_params = [Transport.DecayParams(
                kdecay=Transport.ExponentialDecay, halftime_hours=hl)]
        else
            decay_params = [Transport.DecayParams(
                kdecay=Transport.NoDecay, halftime_hours=0.0)]
        end

        state = Transport.initialize_simulation(domain, [source], [isotope], decay_params;
                                                 log_depositions=true)

        # Pre-generate particles with staggered ages to simulate continuous release.
        # Each particle gets a random "birth time" within [0, release_duration].
        # Particles with age > 0 won't have been advected during the delay, but this
        # is the standard Lagrangian approximation for pre-seeded releases.
        rel_dur_s = max(release_duration_hours, 1.0/12.0) * 3600.0

        update!(15, "Generating particles ($(release_duration_hours)h release)...")

        particle_prop = ParticleProperties(diameter_μm=5.0, density_gcm3=2.0)
        npp_radii = Float64[]
        npp_densities = Float64[]
        npp_size_indices = Int[]

        pos_s, act_s, released_s = Transport.generate_release_particles(
            rng, source, 0, 1,
            ones(Float64, era5.nx_met, era5.ny_met),
            ones(Float64, era5.nx_met, era5.ny_met),
            domain.dx, domain.dy, domain.hlevel,
        )
        if released_s && !isempty(pos_s)
            for (k, (pos, activity)) in enumerate(zip(pos_s, act_s))
                sigma_z = Transport.height_to_sigma_hybrid(
                    release_x, release_y, pos[3], init_met, 0.0)
                # Stagger birth: particle k is "born" at a random time in the release window
                age = rand(rng) * rel_dur_s
                Transport.add_particle!(state.ensemble,
                    SVector{3,Float64}(pos[1], pos[2], sigma_z),
                    SVector{3,Float64}(0.0, 0.0, 0.0),
                    [activity], age, icomp=1)
                push!(npp_radii, 5.0 * 0.5e-6)
                push!(npp_densities, 2000.0)
                push!(npp_size_indices, 1)
            end
        end

        update!(25, "Running $(duration_hours)-hour simulation ($(length(state.ensemble.particles)) particles)...")

        psc = ParticleSizeConfig(size_bins=[particle_prop],
            particle_radii=npp_radii, particle_densities=npp_densities,
            particle_size_indices=npp_size_indices)
        hanna = HannaTurbulenceConfig{Float64}(use_cbl=true)
        dep = Transport.DepositionConfig{Float64}(
            apply_dry_deposition=true, apply_wet_deposition=false,
            use_simple_deposition=true, simple_deposition_velocity=0.002)

        num_cfg = ERA5NumericalConfig{Float64}(
            interpolation_order=Transport.LinearInterp, ode_solver_type=:Euler,
            fixed_dt=300.0, turbulence=Transport.OrnsteinUhlenbeck)

        out_cfg = OutputConfig(
            trace_frequency=TRACE_DISABLED, verbosity=VERBOSITY_QUIET, trace_enabled=false)

        sim_cfg = Transport.SimulationConfig{Float64}(
            saveat=[Float64(h) * 3600.0 for h in 0:duration_hours],
            verbose=false, max_duration=Float64(duration_hours) * 3600.0,
            save_snapshots=true, dt_particle=300.0, use_reference_stepping=true,
            max_files=length(era5.files), output_config=out_cfg)

        snapshots = Transport.run_simulation!(state, era5.files,
            particle_size_config=psc, deposition_config=dep,
            hanna_config=hanna, decay_params=decay_params, config=sim_cfg,
            numerical_config=num_cfg, advection_enabled=true, settling_enabled=false,
            dry_deposition_enabled=true, wet_deposition_enabled=false,
            release_height_m=stack_height_m + half_h, met_data_cache=era5.met_cache,
            met_format_override=era5.met_format,
            met_dimensions=(era5.nx_met, era5.ny_met, era5.nk_met),
            cache_init_file_idx=5, cache_init_time_idx=1,
            sigma_already_initialized=true)

        store_animation_data!(snapshots, domain, Float64[]; release_mode="npp", units="kBq/m²")
        update!(85, "Computing deposition...")

        # Build deposition field on fine grid (kBq/m²)
        display_lons = [l > 180.0 ? l - 360.0 : l for l in era5.lon_range]
        lon_grid = range(minimum(display_lons), maximum(display_lons), step=0.023)
        lat_grid = range(minimum(era5.lat_range), maximum(era5.lat_range), step=0.018)
        nx_out, ny_out = length(lon_grid), length(lat_grid)

        fine_dep = zeros(nx_out, ny_out)
        for evt in state.deposition_log
            elat, elon = Transport.grid_to_latlon(domain, evt.x, evt.y)
            elon > 180.0 && (elon -= 360.0)
            i = searchsortedlast(lon_grid, elon)
            j = searchsortedlast(lat_grid, elat)
            if 1 <= i <= nx_out && 1 <= j <= ny_out
                fine_dep[i, j] += evt.mass
            end
        end

        dlat = abs(lat_grid[2] - lat_grid[1])
        dlon = abs(lon_grid[2] - lon_grid[1])
        ref_lat = 0.5 * (first(lat_grid) + last(lat_grid))
        dy_m = dlat * 111_000.0
        dx_m = dlon * 111_000.0 * cosd(ref_lat)
        cell_area_m2 = dx_m * dy_m

        # Convert Bq → kBq/m²
        dep_kBq = fine_dep ./ (cell_area_m2 * 1000.0)
        dep_smooth = gaussian_smooth(dep_kBq, 2.0)
        max_val = maximum(dep_smooth)

        update!(95, "Generating contours...")
        return SimulationResult(dep_smooth, lon_grid, lat_grid, max_val,
                                state.deposition_log, 2.0, domain, "kBq/m²")
    end

    # --- Bomb mode (existing) ---

    # Get optimised parameters, scale activity by yield
    params = nancy_optimised_config()
    p = params.physics_scales
    lf = params.layer_fractions
    ps = params.particle_size_config
    total_activity = params.activity_Bq * (yield_kt / 24.0)

    # 3-layer NOAA release geometry
    layer_lower  = CylinderRelease(0.0, 3800.0, 537.0)
    layer_middle = CylinderRelease(3800.0, 6100.0, 1500.0)
    layer_upper  = CylinderRelease(6100.0, 12500.0, 2500.0)

    n_lower  = round(Int, n_particles * lf.lower)
    n_middle = round(Int, n_particles * lf.middle)
    n_upper  = n_particles - n_lower - n_middle

    sources = [
        ReleaseSource((release_x, release_y), layer_lower,
                       BombRelease(0.0), [total_activity * lf.lower], max(n_lower, 1)),
        ReleaseSource((release_x, release_y), layer_middle,
                       BombRelease(0.0), [total_activity * lf.middle], max(n_middle, 1)),
        ReleaseSource((release_x, release_y), layer_upper,
                       BombRelease(0.0), [total_activity * lf.upper], max(n_upper, 1)),
    ]

    update!(15, "Generating particles...")

    # Bimodal particle size distribution
    size_bins = generate_bimodal_bins(ps.d_median_fine_μm, ps.sigma_g_fine,
                                      ps.d_median_coarse_μm, ps.sigma_g_coarse)
    bin_weights = compute_bimodal_weights(ps.d_median_fine_μm, ps.sigma_g_fine,
                                          ps.d_median_coarse_μm, ps.sigma_g_coarse,
                                          ps.frac_fine, size_bins)

    # Initialize state
    decay_params = [Transport.DecayParams(kdecay=Transport.NoDecay, halftime_hours=0.0)]
    state = Transport.initialize_simulation(domain, sources, ["MixedFP"], decay_params;
                                             log_depositions=true)

    snap_bins = [ParticleProperties(diameter_μm=b.d, density_gcm3=2.5) for b in size_bins]
    particle_radii = Float64[]
    particle_densities = Float64[]
    particle_size_indices = Int[]
    fixed_gravity = [b.v * p.vgrav_scale for b in size_bins]
    cum_weights = cumsum(bin_weights)
    base_density = 2500.0

    for src in sources
        pos_s, act_s, released_s = Transport.generate_release_particles(
            rng, src, 0, 1,
            ones(Float64, era5.nx_met, era5.ny_met),
            ones(Float64, era5.nx_met, era5.ny_met),
            domain.dx, domain.dy, domain.hlevel,
        )
        if released_s && !isempty(pos_s)
            for (pos, activity) in zip(pos_s, act_s)
                sigma_z = Transport.height_to_sigma_hybrid(
                    release_x, release_y, pos[3], init_met, 0.0)
                Transport.add_particle!(state.ensemble,
                    SVector{3,Float64}(pos[1], pos[2], sigma_z),
                    SVector{3,Float64}(0.0, 0.0, 0.0),
                    [activity], 0.0, icomp=1)

                idx = clamp(searchsortedfirst(cum_weights, rand(rng)), 1, length(size_bins))
                push!(particle_radii, size_bins[idx].d * 0.5e-6)
                push!(particle_densities, base_density)
                push!(particle_size_indices, idx)

                np = length(state.ensemble.particles)
                state.ensemble.particles[np].grv = Float32(
                    size_bins[idx].v * 0.01 * p.vgrav_scale)
            end
        end
    end

    update!(25, "Running $(duration_hours)-hour simulation ($(length(state.ensemble.particles)) particles)...")

    # Configure physics
    psc = ParticleSizeConfig(size_bins=snap_bins, particle_radii=particle_radii,
        particle_densities=particle_densities, particle_size_indices=particle_size_indices,
        fixed_gravity_cm_s=fixed_gravity)

    hanna = HannaTurbulenceConfig{Float64}(
        sigma_scale=p.sigma_h_scale, sigma_scale_vertical=p.sigma_w_scale,
        tl_scale=p.tl_scale, use_cbl=true)

    dep = Transport.DepositionConfig{Float64}(
        apply_dry_deposition=true, apply_wet_deposition=false,
        use_simple_deposition=true, simple_deposition_velocity=0.002 * p.vd_scale,
        simple_surface_height=30.0 * p.surface_height_scale,
        mixing_height=1000.0 * p.mixing_height_scale,
        surface_roughness=0.1 * p.roughness_scale)

    num_cfg = ERA5NumericalConfig{Float64}(
        interpolation_order=Transport.LinearInterp, ode_solver_type=:Euler, fixed_dt=300.0,
        turbulence=Transport.OrnsteinUhlenbeck)

    out_cfg = OutputConfig(
        trace_frequency=TRACE_DISABLED,
        verbosity=VERBOSITY_QUIET,
        trace_enabled=false)

    sim_cfg = Transport.SimulationConfig{Float64}(
        saveat=[Float64(h) * 3600.0 for h in 0:duration_hours],
        verbose=false, max_duration=Float64(duration_hours) * 3600.0,
        save_snapshots=true, dt_particle=300.0, use_reference_stepping=true,
        max_files=length(era5.files), omega_scale=p.omega_scale,
        output_config=out_cfg)

    # Run simulation
    snapshots = Transport.run_simulation!(state, era5.files,
        particle_size_config=psc, deposition_config=dep,
        hanna_config=hanna, decay_params=decay_params, config=sim_cfg,
        numerical_config=num_cfg, advection_enabled=true, settling_enabled=true,
        dry_deposition_enabled=true, wet_deposition_enabled=false,
        release_height_m=12500.0, met_data_cache=era5.met_cache,
        met_format_override=era5.met_format,
        met_dimensions=(era5.nx_met, era5.ny_met, era5.nk_met),
        cache_init_file_idx=5, cache_init_time_idx=1,
        sigma_already_initialized=true)

    store_animation_data!(snapshots, domain, Float64[]; release_mode="bomb", units="mSv/h")
    update!(85, "Computing dose rates...")

    # Build dose rate field on fine grid
    display_lons = [l > 180.0 ? l - 360.0 : l for l in era5.lon_range]
    lon_grid = range(minimum(display_lons), maximum(display_lons), step=0.023)
    lat_grid = range(minimum(era5.lat_range), maximum(era5.lat_range), step=0.018)
    nx_out, ny_out = length(lon_grid), length(lat_grid)

    fine_dep = zeros(nx_out, ny_out)
    for evt in state.deposition_log
        elat, elon = Transport.grid_to_latlon(domain, evt.x, evt.y)
        elon > 180.0 && (elon -= 360.0)
        i = searchsortedlast(lon_grid, elon)
        j = searchsortedlast(lat_grid, elat)
        if 1 <= i <= nx_out && 1 <= j <= ny_out
            fine_dep[i, j] += evt.mass
        end
    end

    # Convert to dose rate (mSv/h at H+duration)
    dlat = abs(lat_grid[2] - lat_grid[1])
    dlon = abs(lon_grid[2] - lon_grid[1])
    ref_lat = 0.5 * (first(lat_grid) + last(lat_grid))
    dy_m = dlat * 111_000.0
    dx_m = dlon * 111_000.0 * cosd(ref_lat)
    cell_area_m2 = dx_m * dy_m

    K_DOSE = 1.9e-6
    decay_factor = Float64(duration_hours)^(-1.2)
    dose_factor = K_DOSE * decay_factor / cell_area_m2

    dose_mSvh = fine_dep .* dose_factor
    dose_smooth = gaussian_smooth(dose_mSvh, p.smooth_sigma)
    max_dose = maximum(dose_smooth)

    update!(95, "Generating contours...")

    return SimulationResult(dose_smooth, lon_grid, lat_grid, max_dose,
                            state.deposition_log, p.smooth_sigma, domain, "mSv/h")
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
    prepare_arl_simulation!(dir_path, lat, lon, start_date, duration_hours; progress_callback)

Convert ARL data to ERA5-compatible NetCDF and build met cache for simulation.
"""
function prepare_arl_simulation!(dir_path::String, lat::Float64, lon::Float64,
                                  start_date::String, start_hour::Int, duration_hours::Int;
                                  progress_callback=nothing)
    update!(pct, msg) = isnothing(progress_callback) || progress_callback(pct, msg)

    date_parts = parse.(Int, split(start_date, "-"))
    sim_start_dt = Dates.DateTime(date_parts[1], date_parts[2], date_parts[3], start_hour, 0)

    update!(5, "Converting ARL data to simulation format...")

    # Scale subsetting radius with duration — particles can travel ~500km/day
    # Base: 5° lat / 10° lon for 12h, scale up for longer simulations
    duration_scale = max(1.0, duration_hours / 12.0)
    rlat = min(5.0 * duration_scale, 30.0)
    rlon = min(10.0 * duration_scale, 60.0)

    # Convert ARL region to a single NetCDF file with exactly the needed hours
    nc_files, meta = convert_arl_region(dir_path, lat, lon, sim_start_dt, duration_hours;
                                         radius_lat=rlat, radius_lon=rlon,
                                         progress_callback=progress_callback)

    update!(90, "Building met cache...")

    # Use the existing ERA5Format reader on the converted files
    met_format = Transport.detect_met_format(nc_files[1])
    nx_met, ny_met, nk_met = NCDataset(nc_files[1]) do ds
        Transport.get_met_dimensions(met_format, ds)
    end

    # Build met cache from converted NetCDF files
    met_cache = Dict{Tuple{Int,Int}, Transport.MeteoFields}()
    for (file_idx, nc_file) in enumerate(nc_files)
        NCDataset(nc_file) do ds
            times = Transport.get_time_variable(met_format, ds)
            for t_idx in 1:length(times)
                mf = Transport.MeteoFields(nx_met, ny_met, nk_met, T=Float32)
                t2 = t_idx < length(times) ? t_idx + 1 : t_idx
                Transport.read_initial_met_fields!(met_format, mf, ds, t_idx, t2)
                met_cache[(file_idx, t_idx)] = mf
            end
        end
    end

    # Date range from the files actually used in this simulation
    date_parts_dm = parse.(Int, split(start_date, "-"))
    sim_start_date = Dates.Date(date_parts_dm[1], date_parts_dm[2], date_parts_dm[3])
    date_min = sim_start_date
    date_max = sim_start_date + Dates.Day(cld(duration_hours, 24))

    ARL_STATE[] = ARLData(dir_path, nc_files, met_format, met_cache,
                           nx_met, ny_met, nk_met,
                           meta.lat_range, meta.lon_range,
                           date_min, date_max, meta.tmpdir)

    update!(100, "ARL data ready")
    return ARL_STATE[]
end

# --- Unified simulation entry point ---

"""
    run_simulation_with_source(weather_source, arl_dir; kwargs...)

Run simulation using either ERA5 (built-in) or ARL weather data.
"""
function run_simulation_with_source(;
    weather_source::String = "era5",
    arl_dir::String = "",
    lat::Float64, lon::Float64,
    yield_kt::Float64 = 24.0,
    start_date::String, start_hour::Int,
    duration_hours::Int, n_particles::Int,
    release_mode::String = "bomb",
    activity_tbq::Float64 = 1.0,
    stack_height_m::Float64 = 100.0,
    isotope::String = "Cs-137",
    release_duration_hours::Float64 = 1.0,
    progress_callback = nothing,
)
    if weather_source == "arl"
        return run_arl_dispersion_simulation(;
            arl_dir, lat, lon, yield_kt, start_date, start_hour,
            duration_hours, n_particles, release_mode, activity_tbq,
            stack_height_m, isotope, release_duration_hours, progress_callback)
    else
        return run_dispersion_simulation(;
            lat, lon, yield_kt, start_date, start_hour,
            duration_hours, n_particles, release_mode, activity_tbq,
            stack_height_m, isotope, release_duration_hours, progress_callback)
    end
end

function run_arl_dispersion_simulation(;
    arl_dir::String,
    lat::Float64, lon::Float64,
    yield_kt::Float64 = 24.0,
    start_date::String = "2011-01-01",
    start_hour::Int = 0,
    duration_hours::Int = 12,
    n_particles::Int = 5000,
    release_mode::String = "bomb",
    activity_tbq::Float64 = 1.0,
    stack_height_m::Float64 = 100.0,
    isotope::String = "Cs-137",
    release_duration_hours::Float64 = 1.0,
    progress_callback = nothing,
)
    update!(pct, msg) = isnothing(progress_callback) || progress_callback(pct, msg)

    update!(2, "Preparing ARL weather data...")

    # Convert ARL data for this simulation region
    prepare_arl_simulation!(arl_dir, lat, lon, start_date, start_hour, duration_hours;
                             progress_callback = (pct, msg) -> update!(2 + div(pct, 10), msg))

    arl = ARL_STATE[]
    isnothing(arl) && error("ARL data not loaded")

    update!(15, "Setting up simulation domain...")

    # Parse start time
    date_parts = parse.(Int, split(start_date, "-"))
    start_dt = Dates.DateTime(date_parts[1], date_parts[2], date_parts[3], start_hour, 0)

    # Create domain from ARL grid extents
    # Transport uses 0-360 longitude internally.
    # ARL subsets are contiguous regions. If any longitude is negative,
    # shift everything by +360 so the range is monotonically increasing
    # and doesn't wrap around 0.
    raw_lons = Float64.(arl.lon_range)
    if any(l -> l < 0, raw_lons)
        arl_lons_360 = raw_lons .+ 360.0
    else
        arl_lons_360 = raw_lons
    end
    domain = Transport.SimulationDomain(
        lon_min = minimum(arl_lons_360), lon_max = maximum(arl_lons_360),
        lat_min = minimum(arl.lat_range), lat_max = maximum(arl.lat_range),
        z_min = 0.0, z_max = 35000.0,
        nx = arl.nx_met, ny = arl.ny_met, nz = arl.nk_met,
        start_time = start_dt,
        end_time = start_dt + Dates.Hour(duration_hours),
    )

    # Shift release longitude to match domain's 0-360 convention if needed
    # When ARL lons were shifted by +360, ALL longitudes must be shifted too,
    # not just negative ones (e.g. Paluel at lon=0.6 needs to become 360.6)
    release_lon = any(l -> l < 0, raw_lons) ? lon + 360.0 : lon
    release_x, release_y = Transport.latlon_to_grid(domain, lat, release_lon)
    update!(18, "Configuring physics...")

    rng = Random.MersenneTwister(42)

    # ARL NetCDF starts at the simulation start time, so init is always (1, 1)
    init_file_idx = 1
    init_time_idx = 1
    init_met = arl.met_cache[(init_file_idx, init_time_idx)]

    if release_mode == "bomb"
        # --- Bomb mode ---
        params = nancy_optimised_config()
        p = params.physics_scales
        lf = params.layer_fractions
        ps = params.particle_size_config
        total_activity = params.activity_Bq * (yield_kt / 24.0)

        layer_lower  = CylinderRelease(0.0, 3800.0, 537.0)
        layer_middle = CylinderRelease(3800.0, 6100.0, 1500.0)
        layer_upper  = CylinderRelease(6100.0, 12500.0, 2500.0)

        n_lower  = round(Int, n_particles * lf.lower)
        n_middle = round(Int, n_particles * lf.middle)
        n_upper  = n_particles - n_lower - n_middle

        sources = [
            ReleaseSource((release_x, release_y), layer_lower,
                           BombRelease(0.0), [total_activity * lf.lower], max(n_lower, 1)),
            ReleaseSource((release_x, release_y), layer_middle,
                           BombRelease(0.0), [total_activity * lf.middle], max(n_middle, 1)),
            ReleaseSource((release_x, release_y), layer_upper,
                           BombRelease(0.0), [total_activity * lf.upper], max(n_upper, 1)),
        ]

        decay_params = [Transport.DecayParams(kdecay=Transport.NoDecay, halftime_hours=0.0)]
        state = Transport.initialize_simulation(domain, sources, ["MixedFP"], decay_params;
                                                 log_depositions=true)

        size_bins = generate_bimodal_bins(ps.d_median_fine_μm, ps.sigma_g_fine,
                                          ps.d_median_coarse_μm, ps.sigma_g_coarse)
        bin_weights = compute_bimodal_weights(ps.d_median_fine_μm, ps.sigma_g_fine,
                                              ps.d_median_coarse_μm, ps.sigma_g_coarse,
                                              ps.frac_fine, size_bins)

        snap_bins = [ParticleProperties(diameter_μm=b.d, density_gcm3=2.5) for b in size_bins]
        particle_radii = Float64[]
        particle_densities = Float64[]
        particle_size_indices = Int[]
        fixed_gravity = [b.v * p.vgrav_scale for b in size_bins]
        cum_weights = cumsum(bin_weights)
        base_density = 2500.0

        for src in sources
            pos_s, act_s, released_s = Transport.generate_release_particles(
                rng, src, 0, 1,
                ones(Float64, arl.nx_met, arl.ny_met),
                ones(Float64, arl.nx_met, arl.ny_met),
                domain.dx, domain.dy, domain.hlevel,
            )
            if released_s && !isempty(pos_s)
                for (pos, activity) in zip(pos_s, act_s)
                    sigma_z = Transport.height_to_sigma_hybrid(
                        release_x, release_y, pos[3], init_met, 0.0)
                    Transport.add_particle!(state.ensemble,
                        SVector{3,Float64}(pos[1], pos[2], sigma_z),
                        SVector{3,Float64}(0.0, 0.0, 0.0),
                        [activity], 0.0, icomp=1)

                    idx = clamp(searchsortedfirst(cum_weights, rand(rng)), 1, length(size_bins))
                    push!(particle_radii, size_bins[idx].d * 0.5e-6)
                    push!(particle_densities, base_density)
                    push!(particle_size_indices, idx)

                    np = length(state.ensemble.particles)
                    state.ensemble.particles[np].grv = Float32(
                        size_bins[idx].v * 0.01 * p.vgrav_scale)
                end
            end
        end

        update!(25, "Running $(duration_hours)-hour simulation ($(length(state.ensemble.particles)) particles)...")

        psc = ParticleSizeConfig(size_bins=snap_bins, particle_radii=particle_radii,
            particle_densities=particle_densities, particle_size_indices=particle_size_indices,
            fixed_gravity_cm_s=fixed_gravity)

        hanna = HannaTurbulenceConfig{Float64}(
            sigma_scale=p.sigma_h_scale, sigma_scale_vertical=p.sigma_w_scale,
            tl_scale=p.tl_scale, use_cbl=true)

        dep = Transport.DepositionConfig{Float64}(
            apply_dry_deposition=true, apply_wet_deposition=false,
            use_simple_deposition=true, simple_deposition_velocity=0.002 * p.vd_scale,
            simple_surface_height=30.0 * p.surface_height_scale,
            mixing_height=1000.0 * p.mixing_height_scale,
            surface_roughness=0.1 * p.roughness_scale)

        num_cfg = ERA5NumericalConfig{Float64}(
            interpolation_order=Transport.LinearInterp, ode_solver_type=:Euler, fixed_dt=300.0,
            turbulence=Transport.OrnsteinUhlenbeck)

        out_cfg = OutputConfig(
            trace_frequency=TRACE_DISABLED, verbosity=VERBOSITY_QUIET, trace_enabled=false)

        sim_cfg = Transport.SimulationConfig{Float64}(
            saveat=[Float64(h) * 3600.0 for h in 0:duration_hours],
            verbose=false, max_duration=Float64(duration_hours) * 3600.0,
            save_snapshots=true, dt_particle=300.0, use_reference_stepping=true,
            max_files=length(arl.files), omega_scale=p.omega_scale,
            output_config=out_cfg)

        snapshots = Transport.run_simulation!(state, arl.files,
            particle_size_config=psc, deposition_config=dep,
            hanna_config=hanna, decay_params=decay_params, config=sim_cfg,
            numerical_config=num_cfg, advection_enabled=true, settling_enabled=true,
            dry_deposition_enabled=true, wet_deposition_enabled=false,
            release_height_m=12500.0, met_data_cache=arl.met_cache,
            met_format_override=arl.met_format,
            met_dimensions=(arl.nx_met, arl.ny_met, arl.nk_met),
            cache_init_file_idx=init_file_idx, cache_init_time_idx=init_time_idx,
            sigma_already_initialized=true)

        store_animation_data!(snapshots, domain, Float64[]; release_mode="bomb", units="mSv/h")
        update!(85, "Computing dose rates...")

        # Build dose rate field on fine grid, dynamically sized from domain
        lon_min_disp = minimum(arl.lon_range)
        lon_max_disp = maximum(arl.lon_range)
        lat_min_disp = minimum(arl.lat_range)
        lat_max_disp = maximum(arl.lat_range)
        lon_grid = range(lon_min_disp, lon_max_disp, step=0.023)
        lat_grid = range(lat_min_disp, lat_max_disp, step=0.018)
        nx_out, ny_out = length(lon_grid), length(lat_grid)

        fine_dep = zeros(nx_out, ny_out)
        for evt in state.deposition_log
            elat, elon = Transport.grid_to_latlon(domain, evt.x, evt.y)
            elon > 180.0 && (elon -= 360.0)
            i = searchsortedlast(lon_grid, elon)
            j = searchsortedlast(lat_grid, elat)
            if 1 <= i <= nx_out && 1 <= j <= ny_out
                fine_dep[i, j] += evt.mass
            end
        end

        dlat = abs(lat_grid[2] - lat_grid[1])
        dlon = abs(lon_grid[2] - lon_grid[1])
        ref_lat = 0.5 * (first(lat_grid) + last(lat_grid))
        dy_m = dlat * 111_000.0
        dx_m = dlon * 111_000.0 * cosd(ref_lat)
        cell_area_m2 = dx_m * dy_m

        K_DOSE = 1.9e-6
        decay_factor = Float64(duration_hours)^(-1.2)
        dose_factor = K_DOSE * decay_factor / cell_area_m2

        dose_mSvh = fine_dep .* dose_factor
        dose_smooth = gaussian_smooth(dose_mSvh, p.smooth_sigma)
        max_dose = maximum(dose_smooth)

        update!(95, "Generating contours...")
        return SimulationResult(dose_smooth, lon_grid, lat_grid, max_dose,
                                state.deposition_log, p.smooth_sigma, domain, "mSv/h")
    else
        # --- NPP (Point Release) mode ---
        half_h = 5.0
        geometry = ColumnRelease(stack_height_m - half_h, stack_height_m + half_h)
        total_activity = activity_tbq * 1e12  # TBq → Bq

        source = ReleaseSource(
            (release_x, release_y), geometry,
            ConstantRelease(), [total_activity], n_particles,
        )

        # Decay setup
        hl = get(ISOTOPE_HALFLIVES, isotope, 0.0)
        if hl > 0.0
            decay_params = [Transport.DecayParams(
                kdecay=Transport.ExponentialDecay, halftime_hours=hl)]
        else
            decay_params = [Transport.DecayParams(
                kdecay=Transport.NoDecay, halftime_hours=0.0)]
        end

        state = Transport.initialize_simulation(domain, [source], [isotope], decay_params;
                                                 log_depositions=true)

        rel_dur_s = max(release_duration_hours, 1.0/12.0) * 3600.0
        update!(18, "Generating particles ($(release_duration_hours)h release)...")

        particle_prop = ParticleProperties(diameter_μm=5.0, density_gcm3=2.0)
        npp_radii = Float64[]
        npp_densities = Float64[]
        npp_size_indices = Int[]

        pos_s, act_s, released_s = Transport.generate_release_particles(
            rng, source, 0, 1,
            ones(Float64, arl.nx_met, arl.ny_met),
            ones(Float64, arl.nx_met, arl.ny_met),
            domain.dx, domain.dy, domain.hlevel,
        )
        if released_s && !isempty(pos_s)
            for (pos, activity) in zip(pos_s, act_s)
                sigma_z = Transport.height_to_sigma_hybrid(
                    release_x, release_y, pos[3], init_met, 0.0)
                age = rand(rng) * rel_dur_s
                Transport.add_particle!(state.ensemble,
                    SVector{3,Float64}(pos[1], pos[2], sigma_z),
                    SVector{3,Float64}(0.0, 0.0, 0.0),
                    [activity], age, icomp=1)
                push!(npp_radii, 5.0 * 0.5e-6)
                push!(npp_densities, 2000.0)
                push!(npp_size_indices, 1)
            end
        end

        update!(25, "Running $(duration_hours)-hour simulation ($(length(state.ensemble.particles)) particles)...")

        psc = ParticleSizeConfig(size_bins=[particle_prop],
            particle_radii=npp_radii, particle_densities=npp_densities,
            particle_size_indices=npp_size_indices)
        hanna = HannaTurbulenceConfig{Float64}(use_cbl=true)
        dep = Transport.DepositionConfig{Float64}(
            apply_dry_deposition=true, apply_wet_deposition=false,
            use_simple_deposition=true, simple_deposition_velocity=0.002)

        num_cfg = ERA5NumericalConfig{Float64}(
            interpolation_order=Transport.LinearInterp, ode_solver_type=:Euler,
            fixed_dt=300.0, turbulence=Transport.OrnsteinUhlenbeck)

        out_cfg = OutputConfig(
            trace_frequency=TRACE_DISABLED, verbosity=VERBOSITY_QUIET, trace_enabled=false)

        sim_cfg = Transport.SimulationConfig{Float64}(
            saveat=[Float64(h) * 3600.0 for h in 0:duration_hours],
            verbose=false, max_duration=Float64(duration_hours) * 3600.0,
            save_snapshots=true, dt_particle=300.0, use_reference_stepping=true,
            max_files=length(arl.files), output_config=out_cfg)

        snapshots = Transport.run_simulation!(state, arl.files,
            particle_size_config=psc, deposition_config=dep,
            hanna_config=hanna, decay_params=decay_params, config=sim_cfg,
            numerical_config=num_cfg, advection_enabled=true, settling_enabled=false,
            dry_deposition_enabled=true, wet_deposition_enabled=false,
            release_height_m=stack_height_m + half_h, met_data_cache=arl.met_cache,
            met_format_override=arl.met_format,
            met_dimensions=(arl.nx_met, arl.ny_met, arl.nk_met),
            cache_init_file_idx=init_file_idx, cache_init_time_idx=init_time_idx,
            sigma_already_initialized=true)

        store_animation_data!(snapshots, domain, Float64[]; release_mode="npp", units="kBq/m²")
        update!(85, "Computing deposition...")

        # Build deposition field on fine grid (kBq/m²)
        lon_min_disp = minimum(arl.lon_range)
        lon_max_disp = maximum(arl.lon_range)
        lat_min_disp = minimum(arl.lat_range)
        lat_max_disp = maximum(arl.lat_range)
        lon_grid = range(lon_min_disp, lon_max_disp, step=0.023)
        lat_grid = range(lat_min_disp, lat_max_disp, step=0.018)
        nx_out, ny_out = length(lon_grid), length(lat_grid)

        fine_dep = zeros(nx_out, ny_out)
        for evt in state.deposition_log
            elat, elon = Transport.grid_to_latlon(domain, evt.x, evt.y)
            elon > 180.0 && (elon -= 360.0)
            i = searchsortedlast(lon_grid, elon)
            j = searchsortedlast(lat_grid, elat)
            if 1 <= i <= nx_out && 1 <= j <= ny_out
                fine_dep[i, j] += evt.mass
            end
        end

        dlat = abs(lat_grid[2] - lat_grid[1])
        dlon = abs(lon_grid[2] - lon_grid[1])
        ref_lat = 0.5 * (first(lat_grid) + last(lat_grid))
        dy_m = dlat * 111_000.0
        dx_m = dlon * 111_000.0 * cosd(ref_lat)
        cell_area_m2 = dx_m * dy_m

        # Convert Bq → kBq/m²
        dep_kBq = fine_dep ./ (cell_area_m2 * 1000.0)
        dep_smooth = gaussian_smooth(dep_kBq, 2.0)
        max_val = maximum(dep_smooth)

        update!(95, "Generating contours...")
        return SimulationResult(dep_smooth, lon_grid, lat_grid, max_val,
                                state.deposition_log, 2.0, domain, "kBq/m²")
    end
end
