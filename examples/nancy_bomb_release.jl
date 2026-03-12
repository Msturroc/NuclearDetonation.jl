#!/usr/bin/env julia
# Nancy Bomb Release Example
# ===========================
# Simulates the Upshot-Knothole Nancy nuclear test (24 kT, 24 March 1953)
# using ERA5 reanalysis data and Ornstein-Uhlenbeck turbulence.
#
# Runs a 12-hour dispersion simulation with the latest BIPOP-CMA-ES
# optimised parameters, then plots model-predicted dose rate contours.
#
# Requirements:
#   - ] add CairoMakie
#   - ERA5 data artifact (~96 MB, downloaded automatically on first run)
#
# Usage:
#   julia --project=.. nancy_bomb_release.jl

using NuclearDetonation
using NuclearDetonation.Transport
using CairoMakie
using NCDatasets
using StaticArrays
using Random
using Dates

println("="^70)
println("NANCY BOMB RELEASE — Ornstein-Uhlenbeck")
println("="^70)

# ============================================================================
# Particle size helpers (bimodal log-normal with SNAP settling)
# ============================================================================

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

# ============================================================================
# 1. Load ERA5 met data
# ============================================================================

println("\n1. Loading ERA5 met data...")
era5_files = nancy_era5_files()
println("   Found $(length(era5_files)) ERA5 files")

met_format = Transport.detect_met_format(era5_files[1])
nx_met, ny_met, nk_met = NCDataset(era5_files[1]) do ds
    Transport.get_met_dimensions(met_format, ds)
end

# Pre-cache met fields for files 5–11 (covers detonation time +12 h)
met_cache = Dict{Tuple{Int,Int}, Transport.MeteoFields}()
for file_idx in 5:min(11, length(era5_files))
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
println("   Loaded $(length(met_cache)) timesteps")

# ============================================================================
# 2. Set up domain and source
# ============================================================================

println("\n2. Setting up domain...")
lat_range, lon_range = NCDataset(era5_files[1]) do ds
    Float64.(ds["latitude"][:]), Float64.(ds["longitude"][:])
end

start_dt = Dates.DateTime(1953, 3, 24, 13, 0)
domain = Transport.SimulationDomain(
    lon_min = minimum(lon_range), lon_max = maximum(lon_range),
    lat_min = minimum(lat_range), lat_max = maximum(lat_range),
    z_min = 0.0, z_max = 35000.0, nx = nx_met, ny = ny_met, nz = nk_met,
    start_time = start_dt, end_time = start_dt + Dates.Hour(12),
)

release_x, release_y = Transport.latlon_to_grid(domain, 37.0956, -116.1028)
println("   Release: grid ($(round(release_x, digits=1)), $(round(release_y, digits=1)))")

# ============================================================================
# 3. Optimised parameters (BIPOP-CMA-ES OU, 78.2%)
# ============================================================================

params = nancy_optimised_config()
p = params.physics_scales
lf = params.layer_fractions
ps = params.particle_size_config

# NOAA 3-layer release geometry
layer_lower  = CylinderRelease(0.0, 3800.0, 537.0)
layer_middle = CylinderRelease(3800.0, 6100.0, 1500.0)
layer_upper  = CylinderRelease(6100.0, 12500.0, 2500.0)

n_particles = 10_000
total_activity = params.activity_Bq
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

println("   3-layer: lower=$(round(lf.lower*100, digits=1))% " *
        "middle=$(round(lf.middle*100, digits=1))% " *
        "upper=$(round(lf.upper*100, digits=1))%")

# ============================================================================
# 4. Bimodal particle size distribution
# ============================================================================

size_bins = generate_bimodal_bins(ps.d_median_fine_μm, ps.sigma_g_fine,
                                  ps.d_median_coarse_μm, ps.sigma_g_coarse)
bin_weights = compute_bimodal_weights(ps.d_median_fine_μm, ps.sigma_g_fine,
                                      ps.d_median_coarse_μm, ps.sigma_g_coarse,
                                      ps.frac_fine, size_bins)
println("\n3. Particle size: $(length(size_bins)) bins, " *
        "fine=$(round(ps.d_median_fine_μm, digits=1)) μm, " *
        "coarse=$(round(ps.d_median_coarse_μm, digits=1)) μm")

# ============================================================================
# 5. Initialise simulation state and generate particles
# ============================================================================

println("\n4. Initialising particles...")
decay_params = [Transport.DecayParams(kdecay=Transport.NoDecay, halftime_hours=0.0)]
state = Transport.initialize_simulation(domain, sources, ["MixedFP"], decay_params;
                                         log_depositions=true)

rng = Random.MersenneTwister(42)
init_met = met_cache[(5, 1)]

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
        ones(Float64, nx_met, ny_met), ones(Float64, ny_met, ny_met),
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

            # Set gravitational settling on particle
            np = length(state.ensemble.particles)
            state.ensemble.particles[np].grv = Float32(
                size_bins[idx].v * 0.01 * p.vgrav_scale)
        end
    end
end

println("   $(length(state.ensemble.particles)) particles generated")

# ============================================================================
# 6. Configure physics and run
# ============================================================================

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

sim_cfg = Transport.SimulationConfig{Float64}(
    saveat=[Float64(h) * 3600.0 for h in 1:12], verbose=false, max_duration=12.0 * 3600.0,
    save_snapshots=true, dt_particle=300.0, use_reference_stepping=true,
    max_files=7, omega_scale=p.omega_scale)

println("\n5. Running 12-hour simulation...")
Transport.run_simulation!(state, era5_files,
    particle_size_config=psc, deposition_config=dep,
    hanna_config=hanna, decay_params=decay_params, config=sim_cfg,
    numerical_config=num_cfg, advection_enabled=true, settling_enabled=true,
    dry_deposition_enabled=true, wet_deposition_enabled=false,
    release_height_m=12500.0, met_data_cache=met_cache,
    met_format_override=met_format, met_dimensions=(nx_met, ny_met, nk_met),
    cache_init_file_idx=5, cache_init_time_idx=1,
    sigma_already_initialized=true)

println("   Simulation complete")

# ============================================================================
# 7. Build dose rate field on fine grid
# ============================================================================

println("\n6. Building dose rate field...")

# Try to load observations; if unavailable, use hardcoded grid
obs_dir = joinpath(pkgdir(NuclearDetonation), "data", "nancy_observations")
has_observations = isdir(obs_dir) &&
    isfile(joinpath(obs_dir, "Nancy_doserate_contours.geojson")) &&
    isfile(joinpath(obs_dir, "Nancy_TOA.geojson"))

if has_observations
    nancy_obs = Transport.load_nancy_observations()
    lat_grid, lon_grid = Transport.suggest_grid(nancy_obs; resolution_km=2.0, buffer_fraction=0.5)
    println("   Observations loaded — will produce side-by-side plot")
else
    lon_grid = range(-117.5, -112.0, step=0.023)
    lat_grid = range(36.5, 41.0, step=0.018)
    nancy_obs = nothing
    println("   No observations found — standalone simulation mode")
end
nx_out, ny_out = length(lon_grid), length(lat_grid)

fine_dep = zeros(nx_out, ny_out)
for evt in state.deposition_log
    lat, lon = Transport.grid_to_latlon(domain, evt.x, evt.y)
    lon > 180.0 && (lon -= 360.0)
    i = searchsortedlast(lon_grid, lon)
    j = searchsortedlast(lat_grid, lat)
    if 1 <= i <= nx_out && 1 <= j <= ny_out
        fine_dep[i, j] += evt.mass
    end
end

# Convert deposition (Bq per cell) to dose rate (mR/h at H+12)
dlat = length(lat_grid) > 1 ? abs(lat_grid[2] - lat_grid[1]) : 0.018
dlon = length(lon_grid) > 1 ? abs(lon_grid[2] - lon_grid[1]) : 0.023
ref_lat = 0.5 * (first(lat_grid) + last(lat_grid))
dy_m = dlat * 111_000.0
dx_m = dlon * 111_000.0 * cosd(ref_lat)
cell_area_m2 = dx_m * dy_m

K_DOSE = 1.9e-6          # mSv/h per Bq/m² at H+1 (Glasstone & Dolan)
decay_12h = 12.0^(-1.2)  # Bomb decay factor to H+12
mSv_to_mR = 100.0        # 1 mSv/h ≈ 100 mR/h
dose_factor = K_DOSE * decay_12h * mSv_to_mR / cell_area_m2

dose_mRh = fine_dep .* dose_factor
dose_smooth = gaussian_smooth(dose_mRh, p.smooth_sigma)

max_dose = maximum(dose_smooth)
println("   Max dose rate: $(round(max_dose, digits=1)) mR/h")
println("   Cell area: $(round(cell_area_m2 / 1e6, digits=2)) km²")
println("   Grid: $(nx_out) x $(ny_out) cells")
println("   Deposition events: $(length(state.deposition_log))")

# Build cumulative hourly deposition snapshots and compute model TOA
println("\n   Computing model time-of-arrival...")
sorted_events = sort(state.deposition_log, by=e -> e.time)
model_toa = fill(NaN, nx_out, ny_out)

model_snapshots = Vector{Matrix{Float64}}()
snapshot_hours = Float64[]
for hour in 1:12
    hour_end_time = Float64(hour) * 3600.0
    cumulative_dep = zeros(nx_out, ny_out)
    for evt in sorted_events
        if evt.time <= hour_end_time
            lat, lon = Transport.grid_to_latlon(domain, evt.x, evt.y)
            lon > 180.0 && (lon -= 360.0)
            i = searchsortedlast(lon_grid, lon)
            j = searchsortedlast(lat_grid, lat)
            if 1 <= i <= nx_out && 1 <= j <= ny_out
                cumulative_dep[i, j] += evt.mass
            end
        end
    end
    push!(model_snapshots, cumulative_dep)
    push!(snapshot_hours, Float64(hour))
end

# Smooth cumulative snapshots with a larger kernel for TOA visualisation
toa_smooth_sigma = max(p.smooth_sigma * 5.0, 5.0)
model_snapshots_smooth = [gaussian_smooth(snap, toa_smooth_sigma) for snap in model_snapshots]

# Threshold-based first-arrival detection
max_dose_toa = maximum(model_snapshots_smooth[end])
threshold = max_dose_toa * 0.001
for i in 1:nx_out, j in 1:ny_out
    for (t_idx, snap) in enumerate(model_snapshots_smooth)
        if snap[i, j] >= threshold
            model_toa[i, j] = snapshot_hours[t_idx]
            break
        end
    end
end
println("   TOA coverage: $(sum(.!isnan.(model_toa))) cells with arrival times")

# ============================================================================
# 8. Plot dose rate contours and time of arrival
# ============================================================================

println("\n7. Creating figure...")

contour_levels = [0.4, 1.0, 4.0, 10.0, 40.0, 100.0]
contour_colors = [:blue, :cyan, :green, :yellow, :orange, :red]
toa_hours = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
toa_cmap = cgrad(:viridis, length(toa_hours), categorical=true)

if has_observations
    # --- 2x2: dose rate (top) + TOA (bottom), obs (left) vs model (right) ---
    obs_bounds = Transport.contour_bounds(nancy_obs.dose_rate_contours)
    lat_buf = 0.1 * (obs_bounds[2] - obs_bounds[1])
    lon_buf = 0.1 * (obs_bounds[4] - obs_bounds[3])
    ax_lon_min = obs_bounds[3] - lon_buf
    ax_lon_max = obs_bounds[4] + lon_buf
    ax_lat_min = obs_bounds[1] - lat_buf
    ax_lat_max = obs_bounds[2] + lat_buf

    fig = Figure(size=(1400, 1400), fontsize=14)

    # --- Row 1: Dose rate ---
    ax_obs = Axis(fig[1, 1],
        title = "Observed Dose Rate at H+12",
        xlabel = "Longitude (°)", ylabel = "Latitude (°)",
        limits = (ax_lon_min, ax_lon_max, ax_lat_min, ax_lat_max),
        aspect = DataAspect(),
    )

    for (level, col) in zip(contour_levels, contour_colors)
        for contour_obj in nancy_obs.dose_rate_contours
            contour_obj.dose_rate_mR_hr != level && continue
            for polygon in contour_obj.polygons
                lats = [pt[1] for pt in polygon]
                lons = [pt[2] for pt in polygon]
                lines!(ax_obs, lons, lats, color=col, linewidth=2.5)
            end
        end
    end
    scatter!(ax_obs, [-116.1028], [37.0956], marker=:star5, markersize=20, color=:black)

    ax_mod = Axis(fig[1, 2],
        title = "Model Dose Rate at H+12",
        xlabel = "Longitude (°)", ylabel = "Latitude (°)",
        limits = (ax_lon_min, ax_lon_max, ax_lat_min, ax_lat_max),
        aspect = DataAspect(),
    )

    for (level, col) in zip(contour_levels, contour_colors)
        contour!(ax_mod, collect(lon_grid), collect(lat_grid), dose_smooth,
            levels=[level], color=col, linewidth=2.5)
    end
    scatter!(ax_mod, [-116.1028], [37.0956], marker=:star5, markersize=20, color=:black)

    legend_elements = [LineElement(color=c, linewidth=3) for c in contour_colors]
    legend_labels = ["$(l) mR/h" for l in contour_levels]
    Legend(fig[2, :], legend_elements, legend_labels, "Dose Rate (H+12)",
        orientation=:horizontal, tellwidth=false, tellheight=true)

    # --- Row 2: Time of Arrival ---
    ax_toa_obs = Axis(fig[3, 1],
        title = "Observed Time of Arrival",
        xlabel = "Longitude (°)", ylabel = "Latitude (°)",
        limits = (ax_lon_min, ax_lon_max, ax_lat_min, ax_lat_max),
        aspect = DataAspect(),
    )

    for toa_c in nancy_obs.toa_contours
        ci = searchsortedlast(toa_hours, toa_c.hour)
        ci = clamp(ci, 1, length(toa_hours))
        col = toa_cmap[ci]
        for line in toa_c.lines
            lats = [pt[1] for pt in line]
            lons = [pt[2] for pt in line]
            lines!(ax_toa_obs, lons, lats, color=col, linewidth=2.5)
        end
    end
    scatter!(ax_toa_obs, [-116.1028], [37.0956], marker=:star5, markersize=20, color=:black)

    ax_toa_mod = Axis(fig[3, 2],
        title = "Model Time of Arrival",
        xlabel = "Longitude (°)", ylabel = "Latitude (°)",
        limits = (ax_lon_min, ax_lon_max, ax_lat_min, ax_lat_max),
        aspect = DataAspect(),
    )

    for (hi, h) in enumerate(toa_hours)
        contour!(ax_toa_mod, collect(lon_grid), collect(lat_grid), model_toa,
            levels=[h], color=toa_cmap[hi], linewidth=2.5)
    end
    scatter!(ax_toa_mod, [-116.1028], [37.0956], marker=:star5, markersize=20, color=:black)

    toa_legend_elements = [LineElement(color=toa_cmap[i], linewidth=3) for i in eachindex(toa_hours)]
    toa_legend_labels = ["H+$(Int(h))" for h in toa_hours]
    Legend(fig[4, :], toa_legend_elements, toa_legend_labels, "Time of Arrival",
        orientation=:horizontal, tellwidth=false, tellheight=true)

    Label(fig[0, :], "Nancy 24 kT — Observed vs Model", fontsize=18, font=:bold)
else
    # --- Standalone: model-only plot ---
    fig = Figure(size=(700, 800), fontsize=14)

    ax = Axis(fig[1, 1],
        title = "Nancy 24 kT — Model Dose Rate at H+12",
        xlabel = "Longitude (°)", ylabel = "Latitude (°)",
        limits = (first(lon_grid), last(lon_grid), first(lat_grid), last(lat_grid)),
        aspect = DataAspect(),
    )

    for (level, col) in zip(contour_levels, contour_colors)
        contour!(ax, collect(lon_grid), collect(lat_grid), dose_smooth,
            levels=[level], color=col, linewidth=2.5)
    end
    scatter!(ax, [-116.1028], [37.0956], marker=:star5, markersize=20, color=:black)

    legend_elements = [LineElement(color=c, linewidth=3) for c in contour_colors]
    legend_labels = ["$(l) mR/h" for l in contour_levels]
    Legend(fig[2, 1], legend_elements, legend_labels, "Dose Rate (H+12)",
        orientation=:horizontal, tellwidth=false, tellheight=true)
end

outfile = joinpath(@__DIR__, "nancy_bomb_release.png")
save(outfile, fig, px_per_unit=2)
println("\nSaved: $(outfile)")

# Save dose rate grid as NetCDF for consortium comparison
ncfile = joinpath(@__DIR__, "nancy_dosegrid_ou.nc")
NCDataset(ncfile, "c") do ds
    defDim(ds, "lon", length(lon_grid))
    defDim(ds, "lat", length(lat_grid))
    ds_lon = defVar(ds, "lon", Float64, ("lon",))
    ds_lat = defVar(ds, "lat", Float64, ("lat",))
    ds_dose = defVar(ds, "doserate", Float64, ("lon", "lat"))
    ds_lon[:] = collect(lon_grid)
    ds_lat[:] = collect(lat_grid)
    ds_dose[:,:] = dose_mRh  # raw (unsmoothed) mR/h at H+12
end
println("Saved dose grid: $(ncfile)")
