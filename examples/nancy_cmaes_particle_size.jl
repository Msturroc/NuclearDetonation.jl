#!/usr/bin/env julia
# Nancy BIPOP-CMA-ES v4 — 3-layer release structure
#
# Finds ONE best parameter set for a given turbulence scheme (RW or OU).
# Uses BIPOP restarts and per-generation CRN for noise handling.
# Scoring: 35% FMS + 20% shape (inertia ellipse matching) + 15% extent + 30% TOA
# Shape score penalises symmetric blobs by comparing aspect ratio + orientation
# of the model plume against the observed Nancy contours at each dose level.
#
# v4: 3 CylinderRelease sources matching NOAA 1984 observed debris layers:
#   Lower  (0–3,800 m, r=537 m)    — transported NNW
#   Middle (3,800–6,100 m, r=1,500 m) — transported NNE
#   Upper  (6,100–12,500 m, r=2,500 m) — transported NE at 50 mph
#   Mass fractions frac_lower + frac_middle optimised; frac_upper = remainder
#
# 20 parameters (v5: added activity_scale, smooth_sigma):
#   PARTICLE SIZE - BIMODAL (5):  d_fine, σ_fine, d_coarse, σ_coarse, frac_fine
#   LAYER FRACTIONS (2):          frac_lower, frac_middle
#   TURBULENCE (4):               sigma_w_scale, sigma_h_scale, h_diff_scale, tl_scale
#   PHYSICS (5):                  vd, vgrav, omega, mixing_height, tmix
#   DEPOSITION (2):               surface_height_scale, roughness_scale
#   CALIBRATION (2):              activity_scale (×1e15 Bq), smooth_sigma (Gaussian cells)
#
# Usage:
#   julia --threads=12 --project=.. nancy_cmaes_particle_size.jl RW    # RandomWalk
#   julia --threads=12 --project=.. nancy_cmaes_particle_size.jl OU    # Ornstein-Uhlenbeck
#
# Environment variables:
#   MAX_EVALS=6000       Total evaluation budget (default: 6000)
#   WARM_START=1         Start from APMC v7 best particle (default: 1)

using LinearAlgebra
using Random
using Statistics
using Printf
using Base.Threads
using NCDatasets
using StaticArrays
using Dates
using NuclearDetonation
using NuclearDetonation.Transport

# ============================================================================
# PARSE ARGUMENTS
# ============================================================================

const TURB_SCHEME = if length(ARGS) >= 1 && uppercase(ARGS[1]) == "OU"
    :OU
else
    :RW
end

const TURB_NAME = TURB_SCHEME == :OU ? "Ornstein-Uhlenbeck" : "RandomWalk"

println("="^70)
println("NANCY BIPOP-CMA-ES v4 — $(TURB_NAME) ($(nthreads()) threads)")
println("Pure optimiser: finds ONE best parameter set at d=20")
println("Noise handling: per-generation CRN (common random numbers)")
println("Restarts: BIPOP (alternating large/small population)")
println("v4: 3-layer release (NOAA 1984), tighter density/h_diff bounds")
println("="^70)

# ============================================================================
# PARAMETER BOUNDS — v4: 3-layer release, tighter density and h_diff
# ============================================================================

const PARAM_NAMES = [
    "d_median_fine", "sigma_g_fine", "d_median_coarse", "sigma_g_coarse", "frac_fine",
    "frac_lower", "frac_middle",
    "sigma_w_scale", "sigma_h_scale", "h_diff_scale", "tl_scale",
    "vd_scale", "vgrav_scale", "omega_scale", "mixing_height_scale", "tmix_scale",
    "surface_height_scale", "roughness_scale",
    "activity_scale", "smooth_sigma"
]

const LB = Float64[
    5.0, 1.1, 50.0, 1.1, 0.05,           # particle size: sigma_g min 1.1 (was 1.2)
    0.01, 0.01,                            # layer fractions: wider range
    0.01, 0.1, 0.05, 0.1,                # turbulence: sigma_w, sigma_h, h_diff, tl_scale
    0.1, 0.1, 0.1, 0.1, 0.1,             # physics: vd, vgrav, omega, mixing_height, tmix
    0.1, 0.1,                              # deposition: surface_height, roughness
    5.0, 0.5                               # calibration: activity ×[5,100]e15 Bq, smooth σ [0.5,5] cells
]

const UB = Float64[
    150.0, 3.5, 300.0, 3.5, 0.95,        # particle size: wider d_median and sigma_g
    0.60, 0.70,                            # layer fractions
    5.0, 8.0, 2.0, 5.0,                  # turbulence: sigma_w, sigma_h, h_diff, tl_scale≤5
    10.0, 5.0, 3.0, 5.0, 10.0,            # physics: vd≤10, vgrav≤5, omega≤3, mixing_height≤5, tmix≤10
    5.0, 5.0,                              # deposition: surface_height, roughness
    100.0, 5.0                             # calibration: activity ×[5,100]e15 Bq, smooth σ [0.5,5] cells
]

const N_DIM = length(LB)
const DOMAIN_WIDTH = UB .- LB

# Warm start fallback (used if no checkpoint file found)
const WARM_START_PARAMS = clamp.([
    51.9, 1.97, 184.0, 1.70, 0.66,        # particle size (from RW v7 best)
    0.15, 0.08,                            # layer fractions (mostly upper=0.77)
    3.93, 1.34, 1.41, 1.0,                  # turbulence (strong vertical, moderate h_diff, tl_scale=1)
    0.94, 1.28, 1.12, 2.28, 2.61,         # physics (from RW v7 best, density_scale removed)
    3.79, 2.38,                            # deposition (from RW v7 best)
    30.0, 2.10                             # calibration: activity=30e15, σ=2.1
], LB, UB)

# ============================================================================
# PARTICLE SIZE DISTRIBUTION FUNCTIONS
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
    log_d_centres = range(log_d_min, log_d_max, length=n_bins)
    d_centres = exp.(log_d_centres)
    [(d=d, v=snap_settling_velocity(d)) for d in d_centres]
end

function compute_bimodal_weights(d_fine::Float64, sg_fine::Float64,
                                  d_coarse::Float64, sg_coarse::Float64,
                                  frac_fine::Float64, bins)
    log_d_fine = log(d_fine)
    log_sg_fine = log(sg_fine)
    log_d_coarse = log(d_coarse)
    log_sg_coarse = log(sg_coarse)
    weights = Float64[]
    for bin in bins
        ld = log(bin.d)
        w_fine = exp(-0.5 * ((ld - log_d_fine) / log_sg_fine)^2) / log_sg_fine
        w_coarse = exp(-0.5 * ((ld - log_d_coarse) / log_sg_coarse)^2) / log_sg_coarse
        push!(weights, frac_fine * w_fine + (1.0 - frac_fine) * w_coarse)
    end
    weights ./= sum(weights)
    weights
end

# ============================================================================
# PRE-LOAD ALL DATA
# ============================================================================

println("\n1. Loading ERA5 met data...")
const ERA5_FILES = nancy_era5_files()

const MET_FORMAT = Transport.detect_met_format(ERA5_FILES[1])
const NX, NY, NK = NCDataset(ERA5_FILES[1]) do ds
    Transport.get_met_dimensions(MET_FORMAT, ds)
end

println("   Pre-loading met data (files 5-11)...")
const MET_CACHE = Dict{Tuple{Int,Int}, Transport.MeteoFields}()
const CACHE_START_FILE = 5
const CACHE_END_FILE = 11
for file_idx in CACHE_START_FILE:CACHE_END_FILE
    NCDataset(ERA5_FILES[file_idx]) do ds
        times = Transport.get_time_variable(MET_FORMAT, ds)
        for t_idx in 1:length(times)
            mf = Transport.MeteoFields(NX, NY, NK, T=Float32)
            if t_idx < length(times)
                Transport.read_initial_met_fields!(MET_FORMAT, mf, ds, t_idx, t_idx + 1)
            else
                Transport.read_initial_met_fields!(MET_FORMAT, mf, ds, t_idx, t_idx)
            end
            MET_CACHE[(file_idx, t_idx)] = mf
        end
    end
end
println("   Loaded $(length(MET_CACHE)) timesteps")

println("\n2. Setting up domain...")
const LAT_RANGE, LON_RANGE = NCDataset(ERA5_FILES[1]) do ds
    Float64.(ds["latitude"][:]), Float64.(ds["longitude"][:])
end

const START_DT = Dates.DateTime(1953, 3, 24, 13, 0)
const DOMAIN = Transport.SimulationDomain(
    lon_min = minimum(LON_RANGE), lon_max = maximum(LON_RANGE),
    lat_min = minimum(LAT_RANGE), lat_max = maximum(LAT_RANGE),
    z_min = 0.0, z_max = 35000.0, nx = NX, ny = NY, nz = NK,
    start_time = START_DT, end_time = START_DT + Dates.Hour(12)
)

const RELEASE_X, RELEASE_Y = Transport.latlon_to_grid(DOMAIN, 37.0956, -116.1028)
println("   Release: grid ($(round(RELEASE_X, digits=1)), $(round(RELEASE_Y, digits=1)))")

println("\n3. Loading Nancy observations...")
const NANCY_OBS = Transport.load_nancy_observations()

const LAT_GRID, LON_GRID = Transport.suggest_grid(NANCY_OBS; resolution_km=2.0, buffer_fraction=0.5)
const OBS_MASKS = Transport.rasterise_all_contours(NANCY_OBS.dose_rate_contours, LAT_GRID, LON_GRID)
println("   Fine obs grid: $(length(LAT_GRID))x$(length(LON_GRID)) (2km, 50% buffer)")

# Dose rate conversion: Bq/m² → mR/h at H+12
const CELL_AREA_M2 = let
    dlat = length(LAT_GRID) > 1 ? abs(LAT_GRID[2] - LAT_GRID[1]) : 0.018
    dlon = length(LON_GRID) > 1 ? abs(LON_GRID[2] - LON_GRID[1]) : 0.023
    ref_lat = 0.5 * (first(LAT_GRID) + last(LAT_GRID))
    (dlat * 111_000.0) * (dlon * 111_000.0 * cosd(ref_lat))
end
const DOSE_FACTOR = 1.9e-6 * 12.0^(-1.2) * 100.0 / CELL_AREA_M2  # K_DOSE * decay_12h * mSv→mR / area

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

const SOURCE_LAT = 37.0956
const SOURCE_LON = -116.1028
const OBS_MAX_DIST_KM = let
    max_dist = 0.0
    for (_, obs_mask) in OBS_MASKS
        for i in 1:length(LON_GRID)
            for j in 1:length(LAT_GRID)
                if obs_mask[i, j]
                    dlat = LAT_GRID[j] - SOURCE_LAT
                    dlon = (LON_GRID[i] - SOURCE_LON) * cosd(SOURCE_LAT)
                    dist_km = sqrt(dlat^2 + dlon^2) * 111.0
                    max_dist = max(max_dist, dist_km)
                end
            end
        end
    end
    max_dist
end
println("   Observed max plume extent: $(round(OBS_MAX_DIST_KM, digits=0)) km")

# Pre-compute observed inertia ellipse properties per contour level.
# Used for shape scoring: penalises models that produce blobs instead of
# the elongated plume shape seen in the Nancy observations.

"""
    inertia_ellipse(mask, lat_grid, lon_grid; min_cells=10)

Compute the inertia ellipse (aspect ratio + orientation) of a binary mask.
Coordinates converted to approximate km to avoid lat/lon distortion.
Returns (ar=aspect_ratio, angle=angle_rad) or nothing if too few cells.
"""
function inertia_ellipse(mask::AbstractMatrix, lat_grid, lon_grid; min_cells::Int=10)
    ref_lat = 0.5 * (first(lat_grid) + last(lat_grid))
    km_per_lon = cosd(ref_lat) * 111.0
    km_per_lat = 111.0

    xs = Float64[]
    ys = Float64[]
    for i in eachindex(lon_grid)
        for j in eachindex(lat_grid)
            if mask isa AbstractMatrix{Bool} ? mask[i, j] : mask[i, j] > 0
                push!(xs, (lon_grid[i] - lon_grid[1]) * km_per_lon)
                push!(ys, (lat_grid[j] - lat_grid[1]) * km_per_lat)
            end
        end
    end

    n = length(xs)
    if n < min_cells
        return nothing
    end

    cx = mean(xs)
    cy = mean(ys)

    # 2x2 inertia tensor
    Ixx = mean((xs .- cx).^2)
    Iyy = mean((ys .- cy).^2)
    Ixy = mean((xs .- cx) .* (ys .- cy))

    # Eigenvalues of [[Ixx, Ixy], [Ixy, Iyy]]
    T = Ixx + Iyy
    D = Ixx * Iyy - Ixy^2
    disc = max(T^2 / 4 - D, 0.0)
    lambda1 = T / 2 + sqrt(disc)
    lambda2 = T / 2 - sqrt(disc)
    lambda1 = max(lambda1, 1e-10)
    lambda2 = max(lambda2, 1e-10)

    ar = sqrt(lambda1 / lambda2)
    angle = 0.5 * atan(2 * Ixy, Ixx - Iyy)

    return (ar=ar, angle=angle)
end

# Pre-compute observed shapes (constant for the entire run)
const OBS_SHAPES = let
    shapes = Dict{Float64, @NamedTuple{ar::Float64, angle::Float64}}()
    for (dose_rate, obs_mask) in OBS_MASKS
        props = inertia_ellipse(obs_mask, LAT_GRID, LON_GRID)
        if !isnothing(props)
            shapes[dose_rate] = props
            println("   Contour $(dose_rate) mR/h: AR=$(round(props.ar, digits=1)), angle=$(round(rad2deg(props.angle), digits=0))°")
        end
    end
    shapes
end
println("   Computed observed shapes for $(length(OBS_SHAPES)) contour levels")

# 3-layer release geometry from NOAA 1984 analysis of Nancy debris cloud
# Each layer went in a different wind direction — no consortium member modelled this
const LAYER_LOWER  = Transport.CylinderRelease(0.0, 3800.0, 537.0)     # NNW transport
const LAYER_MIDDLE = Transport.CylinderRelease(3800.0, 6100.0, 1500.0) # NNE transport
const LAYER_UPPER  = Transport.CylinderRelease(6100.0, 12500.0, 2500.0) # NE at 50 mph
const RELEASE_HEIGHT_M = 12500.0  # top of upper layer
println("\n4. 3-layer release (NOAA 1984 observed debris structure):")
println("   Lower:  0 – 3,800 m,  r=537 m  (NNW transport)")
println("   Middle: 3,800 – 6,100 m,  r=1,500 m  (NNE transport)")
println("   Upper:  6,100 – 12,500 m,  r=2,500 m  (NE at 50 mph)")

println("\n5. All data pre-loaded")

# ============================================================================
# DISTANCE FUNCTION — with per-generation CRN
# ============================================================================

"""
    rho_core(params, turb_scheme, gen_seed)

Run SNAP simulation and return loss = 1 - combined_score.
`gen_seed` provides CRN: all candidates in a generation share the same
random particle-to-bin assignments, reducing ranking noise for CMA-ES.
Fresh seeds across generations prevent convergence to noise artefacts.
"""
function rho_core(params::Vector{Float64}, turb_scheme::Symbol, gen_seed::UInt64)
    # Unpack 20 parameters (v5: added activity_scale, smooth_sigma)
    d_median_fine     = params[1]
    sigma_g_fine      = params[2]
    d_median_coarse   = params[3]
    sigma_g_coarse    = params[4]
    frac_fine         = params[5]
    frac_lower        = params[6]
    frac_middle       = params[7]
    sigma_w_scale     = params[8]
    sigma_h_scale     = params[9]
    h_diff_scale      = params[10]
    tl_scale          = params[11]
    vd_scale          = params[12]
    vgrav_scale       = params[13]
    omega_scale       = params[14]
    mixing_height_scale = params[15]
    tmix_scale        = params[16]
    surface_height_scale = params[17]
    roughness_scale   = params[18]
    activity_scale    = params[19]
    smooth_sigma      = params[20]

    frac_upper = clamp(1.0 - frac_lower - frac_middle, 0.05, 1.0)

    # CRN: deterministic per generation, identical across candidates
    rng = Random.MersenneTwister(gen_seed)

    # Generate bimodal particle size distribution
    size_bins = generate_bimodal_bins(d_median_fine, sigma_g_fine,
                                      d_median_coarse, sigma_g_coarse; n_bins=15)
    bin_weights = compute_bimodal_weights(d_median_fine, sigma_g_fine,
                                          d_median_coarse, sigma_g_coarse,
                                          frac_fine, size_bins)

    # 3-layer release: particle counts proportional to mass fractions
    n_particles = 1000
    total_activity = activity_scale * 1.0e15  # activity_scale centred ~15 → 1.5e16 Bq
    n_lower  = round(Int, n_particles * frac_lower)
    n_middle = round(Int, n_particles * frac_middle)
    n_upper  = n_particles - n_lower - n_middle  # remainder to upper

    sources = [
        Transport.ReleaseSource((RELEASE_X, RELEASE_Y), LAYER_LOWER,
                           Transport.BombRelease(0.0), [total_activity * frac_lower], max(n_lower, 1)),
        Transport.ReleaseSource((RELEASE_X, RELEASE_Y), LAYER_MIDDLE,
                           Transport.BombRelease(0.0), [total_activity * frac_middle], max(n_middle, 1)),
        Transport.ReleaseSource((RELEASE_X, RELEASE_Y), LAYER_UPPER,
                           Transport.BombRelease(0.0), [total_activity * frac_upper], max(n_upper, 1)),
    ]
    decay_params = [Transport.DecayParams(kdecay=Transport.NoDecay, halftime_hours=0.0)]

    state = Transport.initialize_simulation(DOMAIN, sources, ["MixedFP"], decay_params;
                                        log_depositions=true)

    init_met = MET_CACHE[(CACHE_START_FILE, 1)]

    # Generate particles from all 3 layers
    positions_m = Tuple{Float64,Float64,Float64}[]
    activities = Float64[]
    for src in sources
        pos_s, act_s, released_s = Transport.generate_release_particles(
            rng, src, 0, 1,
            ones(Float64, NX, NY), ones(Float64, NY, NY),
            DOMAIN.dx, DOMAIN.dy, DOMAIN.hlevel
        )
        if released_s && !isempty(pos_s)
            append!(positions_m, pos_s)
            append!(activities, act_s)
        end
    end

    if isempty(positions_m)
        GC.gc(false)
        return (loss = 1.0, fms = 0.0, shape = 0.0, extent = 0.0, toa = 0.0, combined_old = 0.0)
    end

    n_part = length(positions_m)
    n_classes = length(size_bins)

    snap_size_bins = [Transport.ParticleProperties(diameter_μm=b.d, density_gcm3=2.5) for b in size_bins]
    particle_radii = Float64[]
    particle_densities = Float64[]
    particle_size_indices = Int[]
    fixed_gravity = [b.v * vgrav_scale for b in size_bins]

    cum_weights = cumsum(bin_weights)
    base_density = 2500.0

    # Assign size bins (CRN: rng consumption order deterministic)
    assigned_bins = Vector{Int}(undef, n_part)
    for i in 1:n_part
        r = rand(rng)
        idx = searchsortedfirst(cum_weights, r)
        assigned_bins[i] = clamp(idx, 1, n_classes)
    end

    # Add particles to ensemble with size properties (no vertical IC transforms — layers are fixed)
    for i in 1:n_part
        pos = positions_m[i]
        activity = activities[i]
        sigma_z = Transport.height_to_sigma_hybrid(RELEASE_X, RELEASE_Y, pos[3], init_met, 0.0)
        Transport.add_particle!(state.ensemble,
                           SVector{3,Float64}(pos[1], pos[2], sigma_z),
                           SVector{3,Float64}(0.0, 0.0, 0.0),
                           [activity], 0.0, icomp=1)

        idx = assigned_bins[i]
        push!(particle_radii, size_bins[idx].d * 0.5e-6)
        push!(particle_densities, base_density)
        push!(particle_size_indices, idx)
        state.ensemble.particles[i].grv = Float32(size_bins[idx].v * 0.01 * vgrav_scale)
    end

    particle_size_config = Transport.ParticleSizeConfig(
        size_bins=snap_size_bins, particle_radii=particle_radii,
        particle_densities=particle_densities,
        particle_size_indices=particle_size_indices,
        fixed_gravity_cm_s=fixed_gravity
    )

    hanna_config = Transport.HannaTurbulenceConfig{Float64}(
        sigma_scale = sigma_h_scale,
        sigma_scale_vertical = sigma_w_scale,
        tl_scale = tl_scale,
        use_cbl = true
    )

    base_tmix = 900.0 * tmix_scale
    diffusion_config = Transport.TurbulentDiffusionConfig{Float64}(
        apply_diffusion = true,
        tmix_h = base_tmix / max(h_diff_scale, 0.1),
        tmix_v = base_tmix,
        horizontal_a_bl = 0.5 * h_diff_scale,
        horizontal_a_above = 0.25 * h_diff_scale,
        hmax = 2500.0 * mixing_height_scale
    )

    deposition_config = Transport.DepositionConfig{Float64}(
        apply_dry_deposition = true,
        apply_wet_deposition = false,
        use_simple_deposition = true,
        simple_deposition_velocity = 0.002 * vd_scale,
        simple_surface_height = 30.0 * surface_height_scale,
        mixing_height = 1000.0 * mixing_height_scale,
        surface_roughness = 0.1 * roughness_scale
    )

    snapshot_times = [Float64(h) * 3600.0 for h in 1:12]

    numerical_config = Transport.ERA5NumericalConfig{Float64}(
        interpolation_order = Transport.LinearInterp,
        ode_solver_type = :Euler,
        fixed_dt = 300.0,
        turbulence = turb_scheme == :OU ? Transport.OrnsteinUhlenbeck : Transport.RandomWalk
    )

    sim_config = Transport.SimulationConfig{Float64}(
        saveat = snapshot_times,
        verbose = false,
        max_duration = 12.0 * 3600.0,
        save_snapshots = true,
        dt_particle = 300.0,
        use_fortran_stepping = true,
        max_files = CACHE_END_FILE - CACHE_START_FILE + 1,
        omega_scale = omega_scale
    )

    Transport.run_simulation!(state, ERA5_FILES,
        particle_size_config=particle_size_config,
        deposition_config=deposition_config,
        diffusion_config=diffusion_config,
        hanna_config=hanna_config,
        decay_params=decay_params,
        config=sim_config,
        numerical_config=numerical_config,
        advection_enabled=true,
        settling_enabled=true,
        dry_deposition_enabled=true,
        wet_deposition_enabled=false,
        release_height_m=RELEASE_HEIGHT_M,
        met_data_cache=MET_CACHE,
        met_format_override=MET_FORMAT,
        met_dimensions=(NX, NY, NK),
        cache_init_file_idx=CACHE_START_FILE,
        cache_init_time_idx=1,
        sigma_already_initialized=true
    )

    # Build hourly fine-grid deposition for TOA + final FMS
    nx_obs, ny_obs = length(LON_GRID), length(LAT_GRID)
    sorted_events = sort(state.deposition_log, by=e->e.time)

    model_snapshots = Vector{Matrix{Float64}}()
    snapshot_hours = Float64[]

    for hour in 1:12
        hour_end_time = Float64(hour) * 3600.0
        hourly_dep = zeros(nx_obs, ny_obs)
        for evt in sorted_events
            if evt.time <= hour_end_time
                lat, lon = Transport.grid_to_latlon(DOMAIN, evt.x, evt.y)
                if lon > 180.0
                    lon = lon - 360.0
                end
                i_fine = searchsortedlast(LON_GRID, lon)
                j_fine = searchsortedlast(LAT_GRID, lat)
                if 1 <= i_fine <= nx_obs && 1 <= j_fine <= ny_obs
                    hourly_dep[i_fine, j_fine] += evt.mass
                end
            end
        end
        push!(model_snapshots, hourly_dep)
        push!(snapshot_hours, Float64(hour))
    end

    final_dose = model_snapshots[end]
    total = sum(final_dose)
    if total <= 0
        GC.gc(false)
        return (loss = 1.0, fms = 0.0, shape = 0.0, extent = 0.0, toa = 0.0, combined_old = 0.0)
    end

    # Convert to dose rate (mR/h at H+12) and smooth for FMS
    final_dose_mRh = final_dose .* DOSE_FACTOR
    dose_smooth = gaussian_smooth(final_dose_mRh, smooth_sigma)

    # FMS scoring using ACTUAL dose rate thresholds + shape scoring
    fms_scores = Float64[]
    shape_scores = Float64[]
    for (dose_rate, obs_mask) in OBS_MASKS
        obs_area = sum(obs_mask)
        if obs_area == 0
            push!(fms_scores, 0.0)
            push!(shape_scores, 0.0)
            continue
        end
        model_mask = dose_smooth .>= dose_rate
        inter = Float64(sum(model_mask .& obs_mask))
        uni = Float64(sum(model_mask .| obs_mask))
        fms = uni > 0 ? inter / uni : 0.0
        push!(fms_scores, fms)

        # Shape score: compare inertia ellipse of model vs observed mask
        obs_shape = get(OBS_SHAPES, dose_rate, nothing)
        model_shape = if sum(model_mask) > 0
            inertia_ellipse(model_mask, LAT_GRID, LON_GRID)
        else
            nothing
        end
        if !isnothing(obs_shape) && !isnothing(model_shape)
            ar_score = min(model_shape.ar, obs_shape.ar) / max(model_shape.ar, obs_shape.ar)
            angle_diff = model_shape.angle - obs_shape.angle
            orient_score = cos(angle_diff)^2
            push!(shape_scores, 0.7 * ar_score + 0.3 * orient_score)
        else
            push!(shape_scores, 0.0)
        end
    end

    geo_mean_fms = exp(mean(log(max(s, 0.005)) for s in fms_scores))
    geo_mean_shape = exp(mean(log(max(s, 0.005)) for s in shape_scores))

    # Plume extent
    model_max_dist_km = 0.0
    for i in 1:nx_obs
        for j in 1:ny_obs
            if final_dose[i, j] > 0
                dlat = LAT_GRID[j] - SOURCE_LAT
                dlon = (LON_GRID[i] - SOURCE_LON) * cosd(SOURCE_LAT)
                dist_km = sqrt(dlat^2 + dlon^2) * 111.0
                model_max_dist_km = max(model_max_dist_km, dist_km)
            end
        end
    end
    extent_score = clamp(model_max_dist_km / OBS_MAX_DIST_KM, 0.0, 1.0)

    # TOA scoring
    model_snapshots_norm = Vector{Matrix{Float64}}()
    for snap in model_snapshots
        snap_total = sum(snap)
        push!(model_snapshots_norm, snap_total > 0 ? snap ./ snap_total : snap)
    end

    toa_result = Transport.compute_toa_score(model_snapshots_norm, snapshot_hours,
                                         NANCY_OBS.toa_contours, LAT_GRID, LON_GRID;
                                         threshold_fraction=0.01)
    toa_score = if isnothing(toa_result) || isinf(toa_result.mean_arrival_error_hours)
        0.0
    else
        max(0.0, 1.0 - toa_result.mean_arrival_error_hours / 6.0)
    end

    # New combined (used for CMA-ES ranking): 35% FMS + 20% shape + 15% extent + 30% TOA
    combined = 0.35 * geo_mean_fms + 0.20 * geo_mean_shape + 0.15 * extent_score + 0.30 * toa_score
    # Old combined (for apples-to-apples comparison with APMC v7/v9): 50% FMS + 20% extent + 30% TOA
    combined_old = 0.50 * geo_mean_fms + 0.20 * extent_score + 0.30 * toa_score

    GC.gc(false)
    return (loss = 1.0 - combined,
            fms = geo_mean_fms,
            shape = geo_mean_shape,
            extent = extent_score,
            toa = toa_score,
            combined_old = combined_old)
end

# ============================================================================
# CMA-ES IMPLEMENTATION
# ============================================================================

mutable struct CMAES
    N::Int              # dimension
    lambda::Int         # population size
    mu::Int             # number of parents
    weights::Vector{Float64}
    mueff::Float64
    cc::Float64
    cs::Float64
    c1::Float64
    cmu::Float64
    damps::Float64
    chiN::Float64
    xmean::Vector{Float64}
    sigma::Float64
    pc::Vector{Float64}
    ps::Vector{Float64}
    C::Matrix{Float64}
    B::Matrix{Float64}
    D::Vector{Float64}
    eigeneval::Int
    lb::Vector{Float64}
    ub::Vector{Float64}
    counteval::Int
    generation::Int
    best_ever_val::Float64
    best_ever_x::Vector{Float64}
    stagnation_counter::Int
end

"""
    CMAES(xstart, lb, ub; popsize=0, sigma_frac=0.3)

Initialise CMA-ES with automatic sigma and covariance scaling from bounds.
`sigma_frac` controls what fraction of each dimension's range is covered
by ±1σ initially (0.3 = initial search covers ~60% of each dimension).
The covariance diagonal is set so that all dimensions are explored equally
despite wildly different scales (e.g. d_median 5-250 μm vs sigma_g 1.2-3.0).
"""
function CMAES(xstart::Vector{Float64};
               lb::Vector{Float64}, ub::Vector{Float64},
               popsize::Int=0, sigma_frac::Float64=0.3)
    N = length(xstart)
    lambda = popsize > 0 ? popsize : 4 + floor(Int, 3 * log(N))
    mu = lambda ÷ 2
    raw_weights = [log(lambda/2 + 0.5) - log(i) for i in 1:mu]
    weights = raw_weights ./ sum(raw_weights)
    mueff = sum(weights)^2 / sum(weights.^2)

    cc = (4 + mueff/N) / (N + 4 + 2*mueff/N)
    cs = (mueff + 2) / (N + mueff + 5)
    c1 = 2 / ((N + 1.3)^2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((N + 2)^2 + mueff))
    damps = 2 * mueff / lambda + 0.3 + cs
    chiN = sqrt(N) * (1 - 1/(4*N) + 1/(21*N^2))

    xmean = copy(xstart)
    pc = zeros(N)
    ps = zeros(N)

    # Auto-scale: set sigma=1 and encode per-dimension scales in C.
    # C_ii = (sigma_frac * range_i)^2 so that sigma * sqrt(C_ii) = sigma_frac * range_i.
    # This means ±3σ covers ±3*sigma_frac ≈ 90% of each dimension from the start.
    ranges = ub .- lb
    initial_stds = sigma_frac .* ranges
    C = diagm(initial_stds.^2)
    B = Matrix{Float64}(I, N, N)
    D_vec = copy(initial_stds)
    sigma = 1.0  # all scaling is in C

    CMAES(N, lambda, mu, weights, mueff, cc, cs, c1, cmu, damps, chiN,
          xmean, sigma, pc, ps, C, B, D_vec, 0, lb, ub, 0, 0,
          Inf, copy(xstart), 0)
end

function update_eigensystem!(es::CMAES)
    es.C .= (es.C .+ es.C') ./ 2
    F = eigen(Symmetric(es.C))
    es.D .= sqrt.(max.(F.values, 1e-20))
    es.B .= F.vectors
    es.eigeneval = es.counteval
end

function ask(es::CMAES)::Vector{Vector{Float64}}
    if es.counteval - es.eigeneval > es.lambda / (es.c1 + es.cmu) / es.N / 10
        update_eigensystem!(es)
    end
    candidates = Vector{Vector{Float64}}(undef, es.lambda)
    for k in 1:es.lambda
        z = randn(es.N)
        y = es.B * (es.D .* z)
        x = es.xmean .+ es.sigma .* y
        x .= clamp.(x, es.lb, es.ub)
        candidates[k] = x
    end
    return candidates
end

function tell!(es::CMAES, candidates::Vector{Vector{Float64}}, fitvals::Vector{Float64})
    es.counteval += es.lambda
    es.generation += 1

    idx = sortperm(fitvals)
    xold = copy(es.xmean)

    # Update mean
    es.xmean .= sum(es.weights[i] .* candidates[idx[i]] for i in 1:es.mu)

    # Update evolution paths
    y = (es.xmean .- xold) ./ es.sigma
    z = es.B' * y
    z ./= (es.D .+ 1e-20)
    Cinvsqrt_y = es.B * z

    csn = sqrt(es.cs * (2 - es.cs) * es.mueff)
    es.ps .= (1 - es.cs) .* es.ps .+ csn .* Cinvsqrt_y
    pslen = norm(es.ps)

    threshold = (1.4 + 2/(es.N + 1)) * es.chiN * sqrt(1 - (1 - es.cs)^(2 * es.counteval / es.lambda))
    hsig = pslen < threshold ? 1.0 : 0.0

    ccn = sqrt(es.cc * (2 - es.cc) * es.mueff)
    es.pc .= (1 - es.cc) .* es.pc .+ hsig * ccn .* y

    # Update covariance matrix
    c1a = es.c1 * (1 - (1 - hsig^2) * es.cc * (2 - es.cc))
    rank_mu = zeros(es.N, es.N)
    for i in 1:es.mu
        yi = (candidates[idx[i]] .- xold) ./ es.sigma
        rank_mu .+= es.weights[i] .* (yi * yi')
    end
    es.C .= (1 - c1a - es.cmu * sum(es.weights)) .* es.C
    es.C .+= es.c1 .* (es.pc * es.pc')
    es.C .+= es.cmu .* rank_mu

    # Update step size
    es.sigma *= exp(min(1.0, (es.cs / es.damps) * (pslen / es.chiN - 1) / 2))

    # Track best-ever
    if fitvals[idx[1]] < es.best_ever_val
        es.best_ever_val = fitvals[idx[1]]
        es.best_ever_x = copy(candidates[idx[1]])
        es.stagnation_counter = 0
    else
        es.stagnation_counter += 1
    end

    return fitvals[idx[1]], candidates[idx[1]]
end

function should_restart(es::CMAES)::Tuple{Bool, String}
    if any(isnan.(es.C)) || any(isinf.(es.C))
        return true, "numerical error in C"
    end
    if isnan(es.sigma) || isinf(es.sigma)
        return true, "numerical error in sigma"
    end

    # Step size too small
    if es.sigma * maximum(es.D) < 1e-12
        return true, "sigma * max(D) < 1e-12"
    end

    # Condition number too large
    cond_C = maximum(es.D)^2 / (minimum(es.D)^2 + 1e-20)
    if cond_C > 1e14
        return true, @sprintf("condition number %.1e > 1e14", cond_C)
    end

    # Stagnation: no improvement for 100 + 30*N/lambda generations
    stag_limit = 100 + ceil(Int, 30 * es.N / es.lambda)
    if es.stagnation_counter > stag_limit
        return true, @sprintf("no improvement for %d generations", es.stagnation_counter)
    end

    # All principal axes too small relative to domain
    max_std = es.sigma * maximum(es.D)
    min_domain = minimum(es.ub .- es.lb)
    if max_std < 1e-8 * min_domain
        return true, "search radius negligible vs domain"
    end

    return false, ""
end

# ============================================================================
# PARALLEL EVALUATION WITH PER-GENERATION CRN
# ============================================================================

struct EvalResult
    loss::Float64
    fms::Float64
    shape::Float64
    extent::Float64
    toa::Float64
    combined_old::Float64
end

const FAILED_EVAL = EvalResult(1.0, 0.0, 0.0, 0.0, 0.0, 0.0)

function evaluate_generation(candidates::Vector{Vector{Float64}},
                              turb_scheme::Symbol, gen_seed::UInt64)
    n = length(candidates)
    results = Vector{EvalResult}(undef, n)
    Threads.@threads for i in 1:n
        try
            r = rho_core(candidates[i], turb_scheme, gen_seed)
            results[i] = EvalResult(r.loss, r.fms, r.shape, r.extent, r.toa, r.combined_old)
        catch e
            @warn "Evaluation failed for candidate $i" exception=e
            results[i] = FAILED_EVAL
        end
    end
    return results
end

# ============================================================================
# BIPOP-CMA-ES MAIN LOOP
# ============================================================================

const MAX_EVALS = parse(Int, get(ENV, "MAX_EVALS", "6000"))
const USE_WARM_START = parse(Int, get(ENV, "WARM_START", "1")) == 1

const DEFAULT_LAMBDA = 4 + floor(Int, 3 * log(N_DIM))  # 12 for d=18
const SIGMA_FRAC = 0.3  # ±1σ covers 30% of each dimension's range

# Starting point: prefer checkpoint file, fall back to hardcoded warm start
function load_checkpoint_params(turb_suffix::String)
    ckpt = joinpath(@__DIR__, "nancy_cmaes_$(turb_suffix)_best.txt")
    isfile(ckpt) || return nothing
    params = Dict{String, Float64}()
    for line in eachline(ckpt)
        startswith(line, "#") && continue
        parts = split(line, "\t", limit=2)
        length(parts) == 2 || continue
        params[strip(parts[1])] = parse(Float64, strip(parts[2]))
    end
    x = Float64[]
    for pname in PARAM_NAMES
        haskey(params, pname) || return nothing
        push!(x, params[pname])
    end
    return clamp.(x, LB, UB)
end

x0 = if USE_WARM_START
    turb_suffix = lowercase(string(TURB_SCHEME))
    ckpt_params = load_checkpoint_params(turb_suffix)
    if ckpt_params !== nothing
        println("\n   Warm start from checkpoint: nancy_cmaes_$(turb_suffix)_best.txt")
        ckpt_params
    else
        println("\n   Warm start from hardcoded APMC v7 best particle")
        copy(WARM_START_PARAMS)
    end
else
    println("\n   Starting from centre of domain")
    (LB .+ UB) ./ 2.0
end

println("\n" * "="^70)
println("BIPOP-CMA-ES Configuration:")
println("  Turbulence: $(TURB_NAME)")
println("  Dimension: $(N_DIM)")
println("  Default lambda: $(DEFAULT_LAMBDA)")
println("  Auto-sigma: C diagonal scaled from bounds (sigma_frac=$(SIGMA_FRAC))")
println("  Budget: $(MAX_EVALS) evaluations")
println("  Threads: $(nthreads())")
println("  Noise handling: per-generation CRN")
println("  Warm start: $(USE_WARM_START)")
println("="^70)

# BIPOP state
global_best_val = Inf
global_best_x = copy(x0)
global_best_diag = FAILED_EVAL
total_evals = 0
budget_large = 0   # evals spent on large-population restarts
budget_small = 0   # evals spent on small-population restarts
restart_count = 0
large_lambda = DEFAULT_LAMBDA  # doubles each large restart

results_file = joinpath(@__DIR__, "nancy_cmaes_$(lowercase(string(TURB_SCHEME)))_results.txt")
checkpoint_file = joinpath(@__DIR__, "nancy_cmaes_$(lowercase(string(TURB_SCHEME)))_best.txt")

t_start = time()

while total_evals < MAX_EVALS
    global total_evals, global_best_val, global_best_x, global_best_diag
    global restart_count, large_lambda, budget_large, budget_small

    # Decide restart type and sigma_frac
    if restart_count == 0
        # First run: default lambda, standard sigma, counts toward large budget
        run_lambda = DEFAULT_LAMBDA
        run_type = :large
        run_sigma_frac = SIGMA_FRAC
        run_x0 = copy(x0)
    elseif budget_large <= budget_small
        # Large restart: double lambda, full exploration from global best
        large_lambda = min(large_lambda * 2, MAX_EVALS ÷ 10)
        run_lambda = large_lambda
        run_type = :large
        run_sigma_frac = SIGMA_FRAC
        run_x0 = copy(global_best_x)
    else
        # Small restart: default lambda, tighter random sigma, perturbed start
        run_lambda = DEFAULT_LAMBDA
        run_type = :small
        # Random sigma_frac in [0.01, 0.3] (log-uniform)
        run_sigma_frac = SIGMA_FRAC * 10.0^(-2.0 * rand())
        # Random starting point biased toward global best
        mix = 0.3 + 0.4 * rand()
        run_x0 = mix .* global_best_x .+ (1.0 - mix) .* (LB .+ rand(N_DIM) .* DOMAIN_WIDTH)
        run_x0 .= clamp.(run_x0, LB, UB)
    end

    remaining = MAX_EVALS - total_evals
    if remaining < run_lambda
        break
    end

    restart_count += 1
    run_evals = 0

    println("\n" * "-"^50)
    println("RESTART #$(restart_count) ($(run_type), lambda=$(run_lambda), sigma_frac=$(round(run_sigma_frac, digits=4)))")
    println("-"^50)

    es = CMAES(run_x0; lb=LB, ub=UB, popsize=run_lambda, sigma_frac=run_sigma_frac)
    es.best_ever_val = global_best_val
    es.best_ever_x = copy(global_best_x)

    while total_evals + run_lambda <= MAX_EVALS
        # Draw fresh generation seed for CRN
        gen_seed = rand(UInt64)

        candidates = ask(es)
        eval_results = evaluate_generation(candidates, TURB_SCHEME, gen_seed)
        fitvals = [r.loss for r in eval_results]

        gen_best_val, gen_best_x = tell!(es, candidates, fitvals)
        total_evals += run_lambda
        run_evals += run_lambda

        # Find the best result this generation for diagnostics
        gen_best_idx = argmin(fitvals)
        gen_best_r = eval_results[gen_best_idx]

        # Update global best
        improved = false
        if gen_best_val < global_best_val
            global_best_val = gen_best_val
            global_best_x = copy(gen_best_x)
            global_best_diag = gen_best_r
            improved = true

            # Save checkpoint
            open(checkpoint_file, "w") do f
                for (j, pname) in enumerate(PARAM_NAMES)
                    println(f, "$(pname)\t$(global_best_x[j])")
                end
                println(f, "# loss\t$(global_best_val)")
                println(f, "# score_new\t$(1.0 - global_best_val)")
                println(f, "# score_old\t$(round(global_best_diag.combined_old * 100, digits=2))%")
                println(f, "# fms\t$(global_best_diag.fms)")
                println(f, "# shape\t$(global_best_diag.shape)")
                println(f, "# extent\t$(global_best_diag.extent)")
                println(f, "# toa\t$(global_best_diag.toa)")
            end
        end

        marker = improved ? " ***" : ""
        elapsed = time() - t_start
        # Print both old-style score (comparable to APMC) and component breakdown
        @printf("  Gen %3d [%5d/%d] FMS=%.2f shp=%.2f ext=%.2f toa=%.2f | old=%.1f%% new=%.1f%% | σ=%.3f [%.0fs]%s\n",
                es.generation, total_evals, MAX_EVALS,
                gen_best_r.fms, gen_best_r.shape, gen_best_r.extent, gen_best_r.toa,
                gen_best_r.combined_old * 100,
                (1.0 - gen_best_val) * 100,
                es.sigma, elapsed, marker)
        flush(stdout)

        # Check restart conditions
        do_restart, reason = should_restart(es)
        if do_restart
            println("  -> Restart triggered: $(reason)")
            break
        end
    end

    # Update BIPOP budget tracking
    if run_type == :large
        budget_large += run_evals
    else
        budget_small += run_evals
    end

    println("  Run used $(run_evals) evals ($(run_type)). Budget: large=$(budget_large), small=$(budget_small)")
end

t_elapsed = time() - t_start

# ============================================================================
# RESULTS
# ============================================================================

println("\n" * "="^70)
println("BIPOP-CMA-ES COMPLETE — $(TURB_NAME)")
println("="^70)
println("Total evaluations: $(total_evals)")
println("Restarts: $(restart_count)")
println("Wall time: $(round(t_elapsed/60, digits=1)) minutes")
println("Best loss: $(round(global_best_val, digits=6))")
println("\nScore breakdown (best particle):")
println("  FMS (geo mean):   $(round(global_best_diag.fms, digits=4))")
println("  Shape:            $(round(global_best_diag.shape, digits=4))")
println("  Extent:           $(round(global_best_diag.extent, digits=4))")
println("  TOA:              $(round(global_best_diag.toa, digits=4))")
println("  Old combined (50%FMS+20%ext+30%TOA): $(round(global_best_diag.combined_old * 100, digits=2))%  <- compare to APMC")
println("  New combined (35%FMS+20%shp+15%ext+30%TOA): $(round((1.0 - global_best_val) * 100, digits=2))%")

println("\nBest parameters:")
for (j, pname) in enumerate(PARAM_NAMES)
    println("  $(rpad(pname, 24)) $(round(global_best_x[j], digits=4))")
end

# Interpret particle size
println("\nParticle size distribution:")
println("  Fine mode:   $(round(global_best_x[1], digits=1)) μm (sigma_g=$(round(global_best_x[2], digits=2)))")
println("  Coarse mode: $(round(global_best_x[3], digits=1)) μm (sigma_g=$(round(global_best_x[4], digits=2)))")
println("  Fine fraction: $(round(global_best_x[5]*100, digits=1))%")
frac_l = global_best_x[6]
frac_m = global_best_x[7]
frac_u = clamp(1.0 - frac_l - frac_m, 0.05, 1.0)
println("\nLayer mass fractions:")
println("  Lower  (0–3,800 m):     $(round(frac_l*100, digits=1))%")
println("  Middle (3,800–6,100 m): $(round(frac_m*100, digits=1))%")
println("  Upper  (6,100–12,500 m): $(round(frac_u*100, digits=1))%")
println("\nCalibration:")
println("  Total activity: $(round(global_best_x[19], digits=1))×10¹⁵ Bq = $(round(global_best_x[19]*1e15, sigdigits=3)) Bq")
println("  Smooth sigma:   $(round(global_best_x[20], digits=2)) cells")

# Save full results
open(results_file, "w") do f
    println(f, "Nancy BIPOP-CMA-ES Results — $(TURB_NAME)")
    println(f, "="^60)
    println(f, "Total evaluations: $(total_evals)")
    println(f, "Restarts: $(restart_count)")
    println(f, "Wall time: $(round(t_elapsed/60, digits=1)) minutes")
    println(f, "Best loss: $(global_best_val)")
    println(f, "Best score: $(round((1.0 - global_best_val) * 100, digits=2))%")
    println(f, "\nBest parameters:")
    for (j, pname) in enumerate(PARAM_NAMES)
        println(f, "  $(pname): $(global_best_x[j])")
    end
    println(f, "\nParticle size distribution:")
    println(f, "  Fine: $(round(global_best_x[1], digits=1)) μm (σ_g=$(round(global_best_x[2], digits=2)))")
    println(f, "  Coarse: $(round(global_best_x[3], digits=1)) μm (σ_g=$(round(global_best_x[4], digits=2)))")
    println(f, "  Fine fraction: $(round(global_best_x[5]*100, digits=1))%")
    println(f, "\nBounds:")
    for (j, pname) in enumerate(PARAM_NAMES)
        println(f, "  $(pname): [$(LB[j]), $(UB[j])]")
    end
end

println("\nResults saved to $(basename(results_file))")
println("Checkpoint saved to $(basename(checkpoint_file))")
