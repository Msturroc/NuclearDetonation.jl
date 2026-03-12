#!/usr/bin/env julia
# Smoky Bomb Release — Diagnostic Visualisation
# ================================================
# Simulates the Plumbbob Smoky nuclear test (44 kT, 31 August 1957)
# using ERA5 reanalysis data and Ornstein-Uhlenbeck turbulence with
# the best BIPOP-CMA-ES parameters from smoky_cmaes_ou_best.txt.
#
# If observation data is available, produces a side-by-side figure:
#   LEFT  — Observed dose rate contours (digitised from DASA-1251)
#   RIGHT — Model-predicted dose rate contours at H+12
# Otherwise, produces a standalone model-only plot.
#
# Requirements:
#   - ] add CairoMakie
#   - ERA5 data artifact (~61 MB, downloaded automatically on first run)
#     or local ERA5 data in ERA5_data/ (_snap.nc files from merge_era5_smoky.jl)
#
# Usage:
#   julia --project=../.. smoky_bomb_release.jl

using NuclearDetonation
using NuclearDetonation.Transport
using CairoMakie
using NCDatasets
using StaticArrays
using Random
using Dates
using Statistics

println("="^70)
println("SMOKY BOMB RELEASE — Ornstein-Uhlenbeck Diagnostic")
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

function generate_bimodal_bins(d_fine::Float64, sg_fine::Float64,
                               d_coarse::Float64, sg_coarse::Float64;
                               n_bins::Int=15)
    log_d_min = min(log(d_fine) - 3*log(sg_fine), log(d_coarse) - 3*log(sg_coarse))
    log_d_max = max(log(d_fine) + 3*log(sg_fine), log(d_coarse) + 3*log(sg_coarse))
    log_d_min = max(log_d_min, log(1.0))
    log_d_max = min(log_d_max, log(500.0))
    d_centres = exp.(range(log_d_min, log_d_max, length=n_bins))
    [(d=d, v=snap_settling_velocity(d)) for d in d_centres]
end

function compute_bimodal_weights(d_fine::Float64, sg_fine::Float64,
                                  d_coarse::Float64, sg_coarse::Float64,
                                  frac_fine::Float64, bins)
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
# 1. Load best parameters from checkpoint
# ============================================================================

println("\n1. Loading best CMA-ES parameters...")
best_file = joinpath(@__DIR__, "smoky_cmaes_ou_best.txt")
params = Dict{String,Float64}()
for line in eachline(best_file)
    startswith(line, "#") && continue
    parts = split(line, '\t')
    length(parts) == 2 && (params[parts[1]] = parse(Float64, parts[2]))
end

d_median_fine     = params["d_median_fine"]
sigma_g_fine      = params["sigma_g_fine"]
d_median_coarse   = params["d_median_coarse"]
sigma_g_coarse    = params["sigma_g_coarse"]
frac_fine         = params["frac_fine"]
frac_lower_raw    = params["frac_lower"]
frac_middle_raw   = params["frac_middle"]
sigma_w_scale     = params["sigma_w_scale"]
sigma_h_scale     = params["sigma_h_scale"]
h_diff_scale      = params["h_diff_scale"]
tl_scale          = params["tl_scale"]
vd_scale          = params["vd_scale"]
vgrav_scale       = params["vgrav_scale"]
omega_scale       = params["omega_scale"]
mixing_height_scale = params["mixing_height_scale"]
tmix_scale        = params["tmix_scale"]
surface_height_scale = params["surface_height_scale"]
roughness_scale   = params["roughness_scale"]
activity_scale    = params["activity_scale"]
smooth_sigma      = params["smooth_sigma"]
# Layer heights — use checkpoint values or DASA-1251 defaults (AGL = MSL - 1409m surface)
stem_top_m        = get(params, "stem_top_m", 1822.0)
cap_mid_m         = get(params, "cap_mid_m", 5541.0)
cloud_top_m       = get(params, "cloud_top_m", 9259.0)

# Normalise layer fractions
frac_upper_raw = max(1.0 - frac_lower_raw - frac_middle_raw, 0.05)
frac_total = frac_lower_raw + frac_middle_raw + frac_upper_raw
frac_lower  = frac_lower_raw / frac_total
frac_middle = frac_middle_raw / frac_total
frac_upper  = frac_upper_raw / frac_total

# Enforce layer height ordering
layer_heights_sorted = sort([stem_top_m, cap_mid_m, cloud_top_m])
stem_top_m  = layer_heights_sorted[1]
cap_mid_m   = layer_heights_sorted[2]
cloud_top_m = layer_heights_sorted[3]

println("   Loaded $(length(params)) parameters from $(basename(best_file))")
println("   Particle size: fine=$(round(d_median_fine, digits=1)) um, " *
        "coarse=$(round(d_median_coarse, digits=1)) um")
println("   Layer fractions: lower=$(round(frac_lower*100, digits=1))% " *
        "middle=$(round(frac_middle*100, digits=1))% " *
        "upper=$(round(frac_upper*100, digits=1))%")
println("   Layer heights: stem_top=$(round(stem_top_m, digits=0)) m, " *
        "cap_mid=$(round(cap_mid_m, digits=0)) m, " *
        "cloud_top=$(round(cloud_top_m, digits=0)) m")

# ============================================================================
# 2. Load ERA5 met data (local fallback or Zenodo artifact)
# ============================================================================

println("\n2. Loading ERA5 met data...")
local_era5_dir = joinpath(@__DIR__, "ERA5_data")
era5_files = if isdir(local_era5_dir) && !isempty(filter(f -> endswith(f, "_snap.nc"), readdir(local_era5_dir)))
    sort(filter(f -> endswith(f, "_snap.nc"),
                [joinpath(local_era5_dir, f) for f in readdir(local_era5_dir)]))
else
    smoky_era5_files()
end
println("   Found $(length(era5_files)) ERA5 files")

met_format = Transport.detect_met_format(era5_files[1])
nx_met, ny_met, nk_met = NCDataset(era5_files[1]) do ds
    Transport.get_met_dimensions(met_format, ds)
end

# Pre-cache met fields for files 5-11 (covers detonation time +12 h)
met_cache = Dict{Tuple{Int,Int}, Transport.MeteoFields}()
cache_start_file = 5
cache_end_file = 11
for file_idx in cache_start_file:min(cache_end_file, length(era5_files))
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
println("   Loaded $(length(met_cache)) timesteps into cache")

# ============================================================================
# 3. Set up domain and release geometry
# ============================================================================

println("\n3. Setting up domain...")
lat_range, lon_range = NCDataset(era5_files[1]) do ds
    Float64.(ds["latitude"][:]), Float64.(ds["longitude"][:])
end

start_dt = Dates.DateTime(1957, 8, 31, 12, 0)
domain = Transport.SimulationDomain(
    lon_min = minimum(lon_range), lon_max = maximum(lon_range),
    lat_min = minimum(lat_range), lat_max = maximum(lat_range),
    z_min = 0.0, z_max = 35000.0, nx = nx_met, ny = ny_met, nz = nk_met,
    start_time = start_dt, end_time = start_dt + Dates.Hour(12),
)

release_x, release_y = Transport.latlon_to_grid(domain, 37.177, -116.046)
println("   Release: grid ($(round(release_x, digits=1)), $(round(release_y, digits=1)))")

# 3-layer release geometry from tuneable heights (DASA-1251 defaults)
layer_lower  = CylinderRelease(0.0, stem_top_m, 0.2 * stem_top_m)
layer_middle = CylinderRelease(stem_top_m, cap_mid_m, 0.25 * (cap_mid_m - stem_top_m))
layer_upper  = CylinderRelease(cap_mid_m, cloud_top_m, 0.25 * (cloud_top_m - cap_mid_m))
release_height_m = cloud_top_m

n_particles = 10_000
total_activity = activity_scale * 1.0e15
n_lower  = max(round(Int, n_particles * frac_lower), 1)
n_middle = max(round(Int, n_particles * frac_middle), 1)
n_upper  = max(n_particles - n_lower - n_middle, 1)

sources = [
    ReleaseSource((release_x, release_y), layer_lower,
                   BombRelease(0.0), [total_activity * frac_lower], max(n_lower, 1)),
    ReleaseSource((release_x, release_y), layer_middle,
                   BombRelease(0.0), [total_activity * frac_middle], max(n_middle, 1)),
    ReleaseSource((release_x, release_y), layer_upper,
                   BombRelease(0.0), [total_activity * frac_upper], max(n_upper, 1)),
]

println("   3-layer: lower=$(n_lower) middle=$(n_middle) upper=$(n_upper) particles")

# ============================================================================
# 4. Bimodal particle size distribution
# ============================================================================

size_bins = generate_bimodal_bins(d_median_fine, sigma_g_fine,
                                  d_median_coarse, sigma_g_coarse)
bin_weights = compute_bimodal_weights(d_median_fine, sigma_g_fine,
                                      d_median_coarse, sigma_g_coarse,
                                      frac_fine, size_bins)
println("\n4. Particle size: $(length(size_bins)) bins, " *
        "fine=$(round(d_median_fine, digits=1)) um (sg=$(round(sigma_g_fine, digits=2))), " *
        "coarse=$(round(d_median_coarse, digits=1)) um (sg=$(round(sigma_g_coarse, digits=2)))")

# ============================================================================
# 5. Initialise simulation state and generate particles
# ============================================================================

println("\n5. Initialising particles...")
decay_params = [Transport.DecayParams(kdecay=Transport.NoDecay, halftime_hours=0.0)]
state = Transport.initialize_simulation(domain, sources, ["MixedFP"], decay_params;
                                         log_depositions=true)

rng = Random.MersenneTwister(42)
init_met = met_cache[(cache_start_file, 1)]

snap_bins = [ParticleProperties(diameter_μm=b.d, density_gcm3=2.5) for b in size_bins]
particle_radii = Float64[]
particle_densities = Float64[]
particle_size_indices = Int[]
fixed_gravity = [b.v * vgrav_scale for b in size_bins]
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
                size_bins[idx].v * 0.01 * vgrav_scale)
        end
    end
end

println("   $(length(state.ensemble.particles)) particles generated")

# ============================================================================
# 6. Configure physics and run 12-hour simulation
# ============================================================================

psc = ParticleSizeConfig(size_bins=snap_bins, particle_radii=particle_radii,
    particle_densities=particle_densities, particle_size_indices=particle_size_indices,
    fixed_gravity_cm_s=fixed_gravity)

hanna = HannaTurbulenceConfig{Float64}(
    sigma_scale=sigma_h_scale, sigma_scale_vertical=sigma_w_scale,
    tl_scale=tl_scale, use_cbl=true)

dep = Transport.DepositionConfig{Float64}(
    apply_dry_deposition=true, apply_wet_deposition=false,
    use_simple_deposition=true, simple_deposition_velocity=0.002 * vd_scale,
    simple_surface_height=30.0 * surface_height_scale,
    mixing_height=1000.0 * mixing_height_scale,
    surface_roughness=0.1 * roughness_scale)

num_cfg = ERA5NumericalConfig{Float64}(
    interpolation_order=Transport.LinearInterp, ode_solver_type=:Euler, fixed_dt=300.0,
    turbulence=Transport.OrnsteinUhlenbeck)

sim_cfg = Transport.SimulationConfig{Float64}(
    saveat=[Float64(h) * 3600.0 for h in 1:12], verbose=false, max_duration=12.0 * 3600.0,
    save_snapshots=true, dt_particle=300.0, use_reference_stepping=true,
    max_files=cache_end_file - cache_start_file + 1, omega_scale=omega_scale)

println("\n6. Running 12-hour simulation...")
Transport.run_simulation!(state, era5_files,
    particle_size_config=psc, deposition_config=dep,
    hanna_config=hanna, decay_params=decay_params, config=sim_cfg,
    numerical_config=num_cfg, advection_enabled=true, settling_enabled=true,
    dry_deposition_enabled=true, wet_deposition_enabled=false,
    release_height_m=release_height_m, met_data_cache=met_cache,
    met_format_override=met_format, met_dimensions=(nx_met, ny_met, nk_met),
    cache_init_file_idx=cache_start_file, cache_init_time_idx=1,
    sigma_already_initialized=true)

println("   Simulation complete")

# ============================================================================
# 7. Build dose rate field on fine grid
# ============================================================================

println("\n7. Building dose rate field...")

# Try to load observations; if unavailable, use hardcoded grid
obs_dir = joinpath(pkgdir(NuclearDetonation), "data", "smoky_observations")
has_observations = isdir(obs_dir) &&
    isfile(joinpath(obs_dir, "Smoky_doserate_contours.geojson")) &&
    isfile(joinpath(obs_dir, "Smoky_TOA.geojson"))

if has_observations
    smoky_obs = Transport.load_smoky_observations()
    lat_grid, lon_grid = Transport.suggest_grid(smoky_obs; resolution_km=2.0, buffer_fraction=0.5)
    println("   Observations loaded — will produce side-by-side plot")
else
    # Standalone grid covering NTS to ~500 km downwind (ESE)
    lon_grid = range(-117.5, -109.0, step=0.023)
    lat_grid = range(35.5, 41.0, step=0.018)
    smoky_obs = nothing
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

K_DOSE = 1.9e-6          # mSv/h per Bq/m^2 at H+1 (Glasstone & Dolan)
decay_12h = 12.0^(-1.2)  # Bomb decay factor to H+12
mSv_to_mR = 100.0        # 1 mSv/h ~ 100 mR/h
dose_factor = K_DOSE * decay_12h * mSv_to_mR / cell_area_m2

dose_mRh = fine_dep .* dose_factor
dose_smooth = gaussian_smooth(dose_mRh, smooth_sigma)

max_dose = maximum(dose_smooth)
println("   Max model dose rate: $(round(max_dose, digits=1)) mR/h")
println("   Cell area: $(round(cell_area_m2 / 1e6, digits=2)) km^2")
println("   Grid: $(nx_out) x $(ny_out) cells")
println("   Deposition events: $(length(state.deposition_log))")

# Build cumulative hourly deposition snapshots and compute model TOA
# (matching the CMA-ES scoring logic in smoky_cmaes_particle_size.jl)
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

# Smooth cumulative snapshots with a larger kernel than dose rate — individual
# hourly snapshots are much sparser than the total field, so need more
# smoothing to produce a continuous front for contouring.
toa_smooth_sigma = max(smooth_sigma * 5.0, 5.0)
model_snapshots_smooth = [gaussian_smooth(snap, toa_smooth_sigma) for snap in model_snapshots]

# Threshold = 0.1% of the final (H+12) smoothed peak — low enough to
# detect arrival in far-field cells where deposits are sparse.
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
# 8. Plot dose rate contours
# ============================================================================

println("\n8. Creating figure...")

contour_levels = [1.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
contour_colors = [:blue, :cyan, :green, :yellow, :orange, :red]
toa_hours = [1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0]
toa_cmap = cgrad(:viridis, length(toa_hours), categorical=true)

if has_observations
    # --- 2x2: dose rate (top) + TOA (bottom), obs (left) vs model (right) ---
    obs_bounds = Transport.contour_bounds(smoky_obs.dose_rate_contours)
    lat_buf = 0.1 * (obs_bounds[2] - obs_bounds[1])
    lon_buf = 0.1 * (obs_bounds[4] - obs_bounds[3])
    ax_lon_min = obs_bounds[3] - lon_buf
    ax_lon_max = obs_bounds[4] + lon_buf
    ax_lat_min = obs_bounds[1] - lat_buf
    ax_lat_max = obs_bounds[2] + lat_buf

    # Size figure to match the geographic aspect ratio (wide plume → wide figure)
    geo_width = (ax_lon_max - ax_lon_min) * cosd(0.5 * (ax_lat_min + ax_lat_max))
    geo_height = ax_lat_max - ax_lat_min
    panel_aspect = geo_width / geo_height  # ~2:1 for Smoky's ESE plume
    fig_width = 1400
    fig_height = round(Int, fig_width / (panel_aspect * 0.85))  # 0.85 accounts for legends/title
    fig = Figure(size=(fig_width, fig_height), fontsize=14)

    # --- Row 1: Dose rate ---
    ax_obs = Axis(fig[1, 1],
        title = "Observed Dose Rate at H+12",
        xlabel = "Longitude (°)", ylabel = "Latitude (°)",
        limits = (ax_lon_min, ax_lon_max, ax_lat_min, ax_lat_max),
        aspect = DataAspect(),
    )

    for (level, col) in zip(contour_levels, contour_colors)
        for contour_obj in smoky_obs.dose_rate_contours
            contour_obj.dose_rate_mR_hr != level && continue
            for polygon in contour_obj.polygons
                lats = [p[1] for p in polygon]
                lons = [p[2] for p in polygon]
                lines!(ax_obs, lons, lats, color=col, linewidth=2.5)
            end
        end
    end
    scatter!(ax_obs, [-116.046], [37.177], marker=:star5, markersize=20, color=:black)

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
    scatter!(ax_mod, [-116.046], [37.177], marker=:star5, markersize=20, color=:black)

    legend_elements = [LineElement(color=c, linewidth=3) for c in contour_colors]
    legend_labels = ["$(Int(l)) mR/h" for l in contour_levels]
    Legend(fig[2, :], legend_elements, legend_labels, "Dose Rate (H+12)",
        orientation=:horizontal, tellwidth=false, tellheight=true)

    # --- Row 2: Time of Arrival ---
    ax_toa_obs = Axis(fig[3, 1],
        title = "Observed Time of Arrival",
        xlabel = "Longitude (°)", ylabel = "Latitude (°)",
        limits = (ax_lon_min, ax_lon_max, ax_lat_min, ax_lat_max),
        aspect = DataAspect(),
    )

    for (k, toa_c) in enumerate(smoky_obs.toa_contours)
        # Colour by hour
        ci = searchsortedlast(toa_hours, toa_c.hour)
        ci = clamp(ci, 1, length(toa_hours))
        col = toa_cmap[ci]
        for line in toa_c.lines
            lats = [p[1] for p in line]
            lons = [p[2] for p in line]
            lines!(ax_toa_obs, lons, lats, color=col, linewidth=2.5)
        end
    end
    scatter!(ax_toa_obs, [-116.046], [37.177], marker=:star5, markersize=20, color=:black)

    ax_toa_mod = Axis(fig[3, 2],
        title = "Model Time of Arrival",
        xlabel = "Longitude (°)", ylabel = "Latitude (°)",
        limits = (ax_lon_min, ax_lon_max, ax_lat_min, ax_lat_max),
        aspect = DataAspect(),
    )

    # Plot each TOA level individually with explicit colours (matching observed side)
    for (hi, h) in enumerate(toa_hours)
        contour!(ax_toa_mod, collect(lon_grid), collect(lat_grid), model_toa,
            levels=[h], color=toa_cmap[hi], linewidth=2.5)
    end
    scatter!(ax_toa_mod, [-116.046], [37.177], marker=:star5, markersize=20, color=:black)

    toa_legend_elements = [LineElement(color=toa_cmap[i], linewidth=3) for i in eachindex(toa_hours)]
    toa_legend_labels = ["H+$(Int(h))" for h in toa_hours]
    Legend(fig[4, :], toa_legend_elements, toa_legend_labels, "Time of Arrival",
        orientation=:horizontal, tellwidth=false, tellheight=true)

    Label(fig[0, :], "Smoky 44 kT — Observed vs Model", fontsize=18, font=:bold)
else
    # --- Standalone: model-only plot ---
    fig = Figure(size=(700, 800), fontsize=14)

    ax_mod = Axis(fig[1, 1],
        title = "Smoky 44 kT — Model Dose Rate at H+12",
        xlabel = "Longitude (°)", ylabel = "Latitude (°)",
        limits = (first(lon_grid), last(lon_grid), first(lat_grid), last(lat_grid)),
        aspect = DataAspect(),
    )

    for (level, col) in zip(contour_levels, contour_colors)
        contour!(ax_mod, collect(lon_grid), collect(lat_grid), dose_smooth,
            levels=[level], color=col, linewidth=2.5)
    end
    scatter!(ax_mod, [-116.046], [37.177], marker=:star5, markersize=20, color=:black)

    legend_elements = [LineElement(color=c, linewidth=3) for c in contour_colors]
    legend_labels = ["$(Int(l)) mR/h" for l in contour_levels]
    Legend(fig[2, 1], legend_elements, legend_labels, "Dose Rate (H+12)",
        orientation=:horizontal, tellwidth=false, tellheight=true)
end

outfile = joinpath(@__DIR__, "smoky_bomb_release.png")
save(outfile, fig, px_per_unit=2)
println("\nSaved: $(outfile)")

# Compute per-threshold FMS scores
if has_observations
    println("\n9. Computing FMS scores...")
    obs_masks = Transport.rasterise_all_contours(smoky_obs.dose_rate_contours,
        collect(Float64, lat_grid), collect(Float64, lon_grid))
    fms_scores = Float64[]
    for (dose_rate, obs_mask) in obs_masks
        model_mask = dose_smooth .>= dose_rate
        inter = Float64(sum(model_mask .& obs_mask))
        uni = Float64(sum(model_mask .| obs_mask))
        fms = uni > 0 ? inter / uni : 0.0
        push!(fms_scores, fms)
        println("   $(dose_rate) mR/h: FMS = $(round(fms * 100, digits=1))%")
    end
    geo_mean = exp(mean(log(max(s, 0.005)) for s in fms_scores))
    println("   Geometric mean FMS: $(round(geo_mean * 100, digits=1))%")
end

# Print diagnostic summary
println("\n" * "="^70)
println("DIAGNOSTIC SUMMARY")
println("="^70)
println("Max model dose rate: $(round(max_dose, digits=1)) mR/h")
println("\nParameters at/near bounds (v5 bounds):")
println("  d_median_fine  = $(round(d_median_fine, digits=4))  (bounds=[10, 100], $(round((d_median_fine-10.0)/(100.0-10.0)*100, digits=1))% of range)")
println("  frac_fine      = $(round(frac_fine, digits=4))  (bounds=[0.05, 0.70], $(round((frac_fine-0.05)/(0.70-0.05)*100, digits=1))% of range)")
println("  frac_lower     = $(round(frac_lower, digits=4))  (bounds=[0.01, 0.50], $(round((frac_lower-0.01)/(0.50-0.01)*100, digits=1))% of range)")
println("  omega_scale    = $(round(omega_scale, digits=4))  (bounds=[0.1, 3.0], $(round((omega_scale-0.1)/(3.0-0.1)*100, digits=1))% of range)")
println("\nTotal activity: $(round(activity_scale, digits=1)) x 10^15 Bq")
println("Smooth sigma: $(round(smooth_sigma, digits=2)) cells")
println("="^70)
