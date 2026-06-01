#!/usr/bin/env julia
# Standalone GPU Nancy Bomb Release Prototype
# ============================================
# A simplified standalone version of the Nancy 24 kT bomb release test that
# runs on CPU and GPU and produces identical deposition fields (up to Float32
# non-associativity). Uses the optimised particle-size, layer-fraction, and
# turbulence-scale parameters from nancy_optimised_config().
#
# Physics implemented:
#   - Trilinear advection on a single ERA5 wind snapshot (lon, lat, level)
#   - Per-particle gravitational settling (Stokes-like, size-dependent)
#   - Ornstein-Uhlenbeck turbulence with constant σ (not Hanna)
#   - Simple ground impact → particle logs its mass to a 2D deposition grid
#
# Intentionally simplified — does NOT match nancy_bomb_release.jl exactly
# because that uses Hanna turbulence, sigma coordinates, time-varying met
# fields, and a full Tsit5 solver. The point is to verify CPU == GPU and
# benchmark the speedup for optimisation workflows.
#
# Outputs (in this same directory):
#   - gpu_nancy_comparison.png : 2x2 plot of CPU vs GPU dose rate contours
#                                 plus an absolute-difference map.
#
# Run: julia --threads=auto --project=/home/marc/NuclearDetonation.jl \
#        gpu_nancy_simulation.jl

using CUDA
using NCDatasets
using Printf
using Random
using Statistics
using CairoMakie

# ============================================================================
# Nancy-optimised parameters (copied from nancy_optimised_config — see
# /home/marc/NuclearDetonation.jl/src/transport/defaults.jl)
# ============================================================================

const D_MEDIAN_FINE_UM   = 127.552f0
const SIGMA_G_FINE       = 2.669f0
const D_MEDIAN_COARSE_UM = 141.861f0
const SIGMA_G_COARSE     = 2.523f0
const FRAC_FINE          = 0.8652f0

const LAYER_LOWER_FRAC   = 0.05617f0
const LAYER_MIDDLE_FRAC  = 0.35074f0
const LAYER_UPPER_FRAC   = 1.0f0 - LAYER_LOWER_FRAC - LAYER_MIDDLE_FRAC

# Physics scaling factors
const SIGMA_W_SCALE      = 4.028f0
const SIGMA_H_SCALE      = 2.220f0
const TL_SCALE           = 4.458f0
const VGRAV_SCALE        = 0.5532f0

const ACTIVITY_BQ        = 48.418f15

# Nancy test site
const NANCY_LON  = -116.1028f0
const NANCY_LAT  =  37.0956f0

# Release layer altitudes (m)
const LAYER_LOWER_TOP   = 3_800.0f0
const LAYER_MIDDLE_TOP  = 6_100.0f0
const LAYER_UPPER_TOP   = 12_500.0f0

# OU turbulence — base σ and timescale (scaled by optimisation factors)
const BASE_SIGMA_H = 0.5f0   # m/s
const BASE_SIGMA_W = 0.2f0   # m/s
const BASE_TL_S    = 100.0f0 # s

const SIGMA_H_MPS = BASE_SIGMA_H * SIGMA_H_SCALE
const SIGMA_W_MPS = BASE_SIGMA_W * SIGMA_W_SCALE
const TL_H_S      = BASE_TL_S * TL_SCALE
const TL_W_S      = BASE_TL_S * TL_SCALE * 0.3f0

# Simulation parameters
const DT           = 300.0f0       # 5 min
const NSTEPS       = 144           # 12 h
const N_PARTICLES  = 10_000

# ============================================================================
# Load Nancy ERA5 (single snapshot — keeps prototype simple, matches both runs)
# ============================================================================

function load_nancy_era5()
    era5_dir = "/home/marc/.julia/artifacts"
    # Find the artifact dir containing Nancy files
    nancy_dir = nothing
    for root in readdir(era5_dir, join=true)
        cand = joinpath(root, "nancy_era5_data")
        if isdir(cand); nancy_dir = cand; break; end
    end
    if nancy_dir === nothing
        error("Could not find nancy_era5_data artifact under $era5_dir")
    end
    files = sort(filter(f -> endswith(f, ".nc"), readdir(nancy_dir, join=true)))
    # Use file covering 13:00 UTC 24 March 1953 (file index 5 in SNAP ordering)
    f = files[5]
    NCDataset(f) do ds
        lon = Float32.(Array(ds["longitude"][:]))
        lat = Float32.(Array(ds["latitude"][:]))
        ap  = Float32.(Array(ds["ap"][:]))
        b   = Float32.(Array(ds["b"][:]))
        # Use first time snapshot
        u = Float32.(Array(ds["x_wind_ml"][:,:,:,1]))
        v = Float32.(Array(ds["y_wind_ml"][:,:,:,1]))
        w = Float32.(Array(ds["omega_ml"][:,:,:,1]))
        psfc = Float32.(Array(ds["surface_air_pressure"][:,:,1]))
        return (; lon, lat, ap, b, u, v, w, psfc, file=f)
    end
end

# Convert model level index to altitude (rough — uses std atmosphere + ap/b)
function level_to_altitude(k::Int, ap::Vector{Float32}, b::Vector{Float32}, psfc::Float32)
    # ap and b are hybrid coefficients: p(k) = ap(k) + b(k)*psfc
    p = ap[k] + b[k] * psfc
    # Rough inverse of std atmosphere
    return Float32(-8000.0 * log(p / psfc))
end

function altitude_to_level(alt::Float32, ap::Vector{Float32}, b::Vector{Float32}, psfc::Float32)
    # Find which level matches the given altitude (rough inverse)
    nk = length(ap)
    best_k = 1
    best_diff = Inf32
    for k in 1:nk
        a = level_to_altitude(k, ap, b, psfc)
        d = abs(a - alt)
        if d < best_diff
            best_diff = d
            best_k = k
        end
    end
    return Float32(best_k)
end

# ============================================================================
# Grid metadata (scalar-friendly, passed to kernels)
# ============================================================================

struct Grid
    lon_min::Float32
    lat_min::Float32
    dx::Float32
    dy::Float32
    nx::Int32
    ny::Int32
    nz::Int32
    m_per_deg_lat::Float32
    m_per_deg_lon::Float32
    dep_lon_min::Float32
    dep_lat_min::Float32
    dep_dx::Float32
    dep_dy::Float32
    dep_nx::Int32
    dep_ny::Int32
end

function make_grid(era5; dep_nx=120, dep_ny=120, buf_deg=3.5f0)
    lon = era5.lon; lat = era5.lat
    nx = Int32(length(lon)); ny = Int32(length(lat)); nz = Int32(size(era5.u, 3))
    dx = abs(lon[2] - lon[1])
    dy = abs(lat[2] - lat[1])
    mid_lat = NANCY_LAT

    dep_lon_min = NANCY_LON - buf_deg
    dep_lat_min = NANCY_LAT - buf_deg * 0.5f0
    dep_lon_max = NANCY_LON + buf_deg * 1.5f0
    dep_lat_max = NANCY_LAT + buf_deg
    dep_dx = (dep_lon_max - dep_lon_min) / Float32(dep_nx)
    dep_dy = (dep_lat_max - dep_lat_min) / Float32(dep_ny)

    Grid(
        Float32(minimum(lon)), Float32(minimum(lat)),
        Float32(dx), Float32(dy), nx, ny, nz,
        111_000.0f0, 111_000.0f0 * cosd(mid_lat),
        dep_lon_min, dep_lat_min, dep_dx, dep_dy,
        Int32(dep_nx), Int32(dep_ny),
    )
end

# ============================================================================
# Bimodal particle size distribution (matches nancy_bomb_release.jl logic)
# ============================================================================

function snap_settling_velocity(d_um::Float32)
    snap_d = Float32[2.2, 4.4, 8.6, 14.6, 22.8, 36.1, 56.5, 92.3, 173.2]
    snap_v = Float32[0.2, 0.7, 2.5, 6.9, 15.9, 35.6, 71.2, 137.0, 277.3]
    ld = log(d_um)
    if ld <= log(snap_d[1])
        slope = (log(snap_v[2]) - log(snap_v[1])) / (log(snap_d[2]) - log(snap_d[1]))
        return exp(log(snap_v[1]) + slope * (ld - log(snap_d[1])))
    elseif ld >= log(snap_d[end])
        slope = (log(snap_v[end]) - log(snap_v[end-1])) / (log(snap_d[end]) - log(snap_d[end-1]))
        return exp(log(snap_v[end]) + slope * (ld - log(snap_d[end])))
    end
    i = 1
    for k in 1:length(snap_d)-1
        if log(snap_d[k]) <= ld <= log(snap_d[k+1]); i = k; break; end
    end
    frac = (ld - log(snap_d[i])) / (log(snap_d[i+1]) - log(snap_d[i]))
    return exp(log(snap_v[i]) + frac * (log(snap_v[i+1]) - log(snap_v[i])))
end

function sample_bimodal_diameter(rng)
    if rand(rng, Float32) < FRAC_FINE
        d = D_MEDIAN_FINE_UM * exp(randn(rng, Float32) * log(SIGMA_G_FINE))
    else
        d = D_MEDIAN_COARSE_UM * exp(randn(rng, Float32) * log(SIGMA_G_COARSE))
    end
    return clamp(Float32(d), 1.0f0, 500.0f0)
end

# ============================================================================
# Initial particle release (3-layer cylinder sampling)
# ============================================================================

function generate_particles(rng, n::Int)
    lons = Vector{Float32}(undef, n)
    lats = Vector{Float32}(undef, n)
    alts = Vector{Float32}(undef, n)
    v_grav = Vector{Float32}(undef, n)
    mass = Vector{Float32}(undef, n)

    activity_per_particle = Float32(ACTIVITY_BQ / n)

    for i in 1:n
        u = rand(rng, Float32)
        if u < LAYER_LOWER_FRAC
            alt_lo, alt_hi, radius = 0.0f0, LAYER_LOWER_TOP, 537.0f0
        elseif u < LAYER_LOWER_FRAC + LAYER_MIDDLE_FRAC
            alt_lo, alt_hi, radius = LAYER_LOWER_TOP, LAYER_MIDDLE_TOP, 1500.0f0
        else
            alt_lo, alt_hi, radius = LAYER_MIDDLE_TOP, LAYER_UPPER_TOP, 2500.0f0
        end

        r = radius * sqrt(rand(rng, Float32))
        θ = 2f0 * Float32(π) * rand(rng, Float32)
        dx_m = r * cos(θ)
        dy_m = r * sin(θ)

        lons[i] = NANCY_LON + dx_m / (111_000.0f0 * cosd(NANCY_LAT))
        lats[i] = NANCY_LAT + dy_m / 111_000.0f0
        alts[i] = alt_lo + (alt_hi - alt_lo) * rand(rng, Float32)

        d_um = sample_bimodal_diameter(rng)
        v_grav[i] = snap_settling_velocity(d_um) * 0.01f0 * VGRAV_SCALE  # cm/s → m/s
        mass[i] = activity_per_particle
    end

    return lons, lats, alts, v_grav, mass
end

# ============================================================================
# Trilinear interpolation (host + device versions, identical arithmetic)
# ============================================================================

@inline function trilinear_host(field, lon::Float32, lat::Float32, zlev::Float32, g::Grid)
    fx = (lon - g.lon_min) / g.dx
    fy = (lat - g.lat_min) / g.dy
    fz = zlev
    i0 = clamp(floor(Int, fx) + 1, 1, Int(g.nx) - 1)
    j0 = clamp(floor(Int, fy) + 1, 1, Int(g.ny) - 1)
    k0 = clamp(floor(Int, fz) + 1, 1, Int(g.nz) - 1)
    tx = clamp(fx - (i0 - 1), 0.0f0, 1.0f0)
    ty = clamp(fy - (j0 - 1), 0.0f0, 1.0f0)
    tz = clamp(fz - (k0 - 1), 0.0f0, 1.0f0)
    @inbounds begin
        c000 = field[i0,   j0,   k0  ]; c100 = field[i0+1, j0,   k0  ]
        c010 = field[i0,   j0+1, k0  ]; c110 = field[i0+1, j0+1, k0  ]
        c001 = field[i0,   j0,   k0+1]; c101 = field[i0+1, j0,   k0+1]
        c011 = field[i0,   j0+1, k0+1]; c111 = field[i0+1, j0+1, k0+1]
    end
    c00 = c000 * (1 - tx) + c100 * tx
    c10 = c010 * (1 - tx) + c110 * tx
    c01 = c001 * (1 - tx) + c101 * tx
    c11 = c011 * (1 - tx) + c111 * tx
    c0  = c00  * (1 - ty) + c10  * ty
    c1  = c01  * (1 - ty) + c11  * ty
    return c0 * (1 - tz) + c1 * tz
end

@inline function trilinear_dev(field, lon::Float32, lat::Float32, zlev::Float32, g::Grid)
    fx = (lon - g.lon_min) / g.dx
    fy = (lat - g.lat_min) / g.dy
    fz = zlev
    i0 = clamp(unsafe_trunc(Int32, fx) + Int32(1), Int32(1), g.nx - Int32(1))
    j0 = clamp(unsafe_trunc(Int32, fy) + Int32(1), Int32(1), g.ny - Int32(1))
    k0 = clamp(unsafe_trunc(Int32, fz) + Int32(1), Int32(1), g.nz - Int32(1))
    tx = clamp(fx - Float32(i0 - 1), 0.0f0, 1.0f0)
    ty = clamp(fy - Float32(j0 - 1), 0.0f0, 1.0f0)
    tz = clamp(fz - Float32(k0 - 1), 0.0f0, 1.0f0)
    @inbounds begin
        c000 = field[i0,   j0,   k0  ]; c100 = field[i0+1, j0,   k0  ]
        c010 = field[i0,   j0+1, k0  ]; c110 = field[i0+1, j0+1, k0  ]
        c001 = field[i0,   j0,   k0+1]; c101 = field[i0+1, j0,   k0+1]
        c011 = field[i0,   j0+1, k0+1]; c111 = field[i0+1, j0+1, k0+1]
    end
    c00 = c000 * (1 - tx) + c100 * tx
    c10 = c010 * (1 - tx) + c110 * tx
    c01 = c001 * (1 - tx) + c101 * tx
    c11 = c011 * (1 - tx) + c111 * tx
    c0  = c00  * (1 - ty) + c10  * ty
    c1  = c01  * (1 - ty) + c11  * ty
    return c0 * (1 - tz) + c1 * tz
end

# ============================================================================
# CPU step: advection + OU + gravity + ground deposition
# ============================================================================

function step_cpu!(lons, lats, alts, utb, vtb, wtb, alive, deposition,
                   mass, v_grav,
                   u_field, v_field, w_field,
                   ap, b_coef, psfc,
                   noise_lon, noise_lat, noise_w,
                   g::Grid, α_h::Float32, α_w::Float32,
                   β_h::Float32, β_w::Float32,
                   step::Int, dt::Float32)
    N = length(lons)
    for p in 1:N
        alive[p] == 0 && continue
        @inbounds begin
            lon = lons[p]; lat = lats[p]; alt = alts[p]
            ut  = utb[p];  vt  = vtb[p];  wt  = wtb[p]

            # Map altitude to level index for interpolation
            zlev = altitude_to_level(alt, ap, b_coef, psfc)

            u_w = trilinear_host(u_field, lon, lat, zlev, g)
            v_w = trilinear_host(v_field, lon, lat, zlev, g)
            w_w = trilinear_host(w_field, lon, lat, zlev, g)

            # OU update
            ut_new = α_h * ut + β_h * noise_lon[p, step]
            vt_new = α_h * vt + β_h * noise_lat[p, step]
            wt_new = α_w * wt + β_w * noise_w[p,  step]

            # Convert omega (Pa/s) to m/s via standard atmosphere (rough)
            w_mps = -w_w * 0.08f0   # rough factor Pa/s → m/s

            eff_u = u_w + ut_new
            eff_v = v_w + vt_new
            eff_w = w_mps + wt_new - v_grav[p]

            lons[p] = lon + eff_u * dt / g.m_per_deg_lon
            lats[p] = lat + eff_v * dt / g.m_per_deg_lat
            new_alt = alt + eff_w * dt

            if new_alt <= 0.0f0
                # Deposit on the ground
                i = floor(Int, (lons[p] - g.dep_lon_min) / g.dep_dx) + 1
                j = floor(Int, (lats[p] - g.dep_lat_min) / g.dep_dy) + 1
                if 1 <= i <= Int(g.dep_nx) && 1 <= j <= Int(g.dep_ny)
                    deposition[i, j] += mass[p]
                end
                alive[p] = Int32(0)
            else
                alts[p] = new_alt
                utb[p] = ut_new; vtb[p] = vt_new; wtb[p] = wt_new
            end
        end
    end
end

# ============================================================================
# GPU kernel: one thread per particle
# ============================================================================

function step_kernel!(lons, lats, alts, utb, vtb, wtb, alive, deposition,
                      mass, v_grav,
                      u_field, v_field, w_field,
                      level_alts,
                      noise_lon, noise_lat, noise_w,
                      g::Grid, α_h::Float32, α_w::Float32,
                      β_h::Float32, β_w::Float32,
                      step::Int32, dt::Float32)
    p = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    p > length(lons) && return nothing
    @inbounds if alive[p] == Int32(1)
        lon = lons[p]; lat = lats[p]; alt = alts[p]
        ut = utb[p]; vt = vtb[p]; wt = wtb[p]

        # Altitude → level index via precomputed lookup
        # Find closest level (linear search, OK for 137 levels)
        zlev = Float32(1)
        best_diff = abs(level_alts[Int32(1)] - alt)
        @inbounds for k in Int32(2):g.nz
            d = abs(level_alts[k] - alt)
            if d < best_diff
                best_diff = d
                zlev = Float32(k)
            end
        end

        u_w = trilinear_dev(u_field, lon, lat, zlev, g)
        v_w = trilinear_dev(v_field, lon, lat, zlev, g)
        w_w = trilinear_dev(w_field, lon, lat, zlev, g)

        ut_new = α_h * ut + β_h * noise_lon[p, step]
        vt_new = α_h * vt + β_h * noise_lat[p, step]
        wt_new = α_w * wt + β_w * noise_w[p,  step]

        w_mps = -w_w * 0.08f0

        eff_u = u_w + ut_new
        eff_v = v_w + vt_new
        eff_w = w_mps + wt_new - v_grav[p]

        new_lon = lon + eff_u * dt / g.m_per_deg_lon
        new_lat = lat + eff_v * dt / g.m_per_deg_lat
        new_alt = alt + eff_w * dt

        if new_alt <= 0.0f0
            i = unsafe_trunc(Int32, (new_lon - g.dep_lon_min) / g.dep_dx) + Int32(1)
            j = unsafe_trunc(Int32, (new_lat - g.dep_lat_min) / g.dep_dy) + Int32(1)
            if Int32(1) <= i <= g.dep_nx && Int32(1) <= j <= g.dep_ny
                CUDA.@atomic deposition[i, j] += mass[p]
            end
            alive[p] = Int32(0)
            lons[p] = new_lon; lats[p] = new_lat
        else
            lons[p] = new_lon; lats[p] = new_lat; alts[p] = new_alt
            utb[p] = ut_new; vtb[p] = vt_new; wtb[p] = wt_new
        end
    end
    return nothing
end

# ============================================================================
# Dose rate computation (matches nancy_bomb_release.jl)
# ============================================================================

function deposition_to_dose_rate(dep::Matrix{Float32}, g::Grid)
    # Cell area in m²
    mid_lat = 0.5f0 * (g.dep_lat_min + g.dep_lat_min + Float32(g.dep_ny) * g.dep_dy)
    dy_m = g.dep_dy * 111_000.0f0
    dx_m = g.dep_dx * 111_000.0f0 * cosd(mid_lat)
    cell_area_m2 = dx_m * dy_m

    K_DOSE = 1.9f-6       # mSv/h per Bq/m² at H+1
    decay_12h = 12.0f0^(-1.2f0)
    mSv_to_mR = 100.0f0
    factor = K_DOSE * decay_12h * mSv_to_mR / cell_area_m2

    return dep .* factor, cell_area_m2
end

# ============================================================================
# Main
# ============================================================================

function main()
    println("="^72)
    println("STANDALONE GPU NANCY — CPU vs GPU benchmark")
    println("="^72)

    println("\nLoading Nancy ERA5 snapshot...")
    era5 = load_nancy_era5()
    println("  File: ", basename(era5.file))
    @printf "  Grid: %d×%d×%d\n" size(era5.u,1) size(era5.u,2) size(era5.u,3)

    g = make_grid(era5)
    @printf "  Particle grid: %.2f..%.2f lon, %.2f..%.2f lat  (%d×%d cells)\n" (g.dep_lon_min) (g.dep_lon_min + Float32(g.dep_nx)*g.dep_dx) (g.dep_lat_min) (g.dep_lat_min + Float32(g.dep_ny)*g.dep_dy) g.dep_nx g.dep_ny

    println("  Threads.nthreads() = $(Threads.nthreads())")
    println("  CUDA device: ", CUDA.name(CUDA.device()))

    # Representative surface pressure at Nancy
    psfc = Float32(mean(era5.psfc))

    # Precompute altitude per level for kernel
    level_alts = Float32[level_to_altitude(k, era5.ap, era5.b, psfc) for k in 1:length(era5.ap)]

    α_h = exp(-DT / TL_H_S)
    α_w = exp(-DT / TL_W_S)
    β_h = sqrt(1 - α_h^2) * SIGMA_H_MPS
    β_w = sqrt(1 - α_w^2) * SIGMA_W_MPS

    rng = MersenneTwister(42)

    # ---- Initial particles (shared between CPU and GPU) ----
    println("\nGenerating $N_PARTICLES particles...")
    lon0, lat0, alt0, v_grav, mass = generate_particles(rng, N_PARTICLES)
    @printf "  Altitude range: %.0f – %.0f m\n" extrema(alt0)...
    @printf "  Mean gravity settling: %.3f m/s\n" mean(v_grav)

    # ---- Pre-generated noise shared by CPU and GPU ----
    println("  Pre-generating noise tensor ($(N_PARTICLES)×3×$NSTEPS Float32)...")
    noise_lon = randn(rng, Float32, N_PARTICLES, NSTEPS)
    noise_lat = randn(rng, Float32, N_PARTICLES, NSTEPS)
    noise_w   = randn(rng, Float32, N_PARTICLES, NSTEPS)

    # ==================== CPU RUN ====================
    println("\n── CPU run ──")
    lons_c = copy(lon0); lats_c = copy(lat0); alts_c = copy(alt0)
    utb_c  = zeros(Float32, N_PARTICLES); vtb_c  = zeros(Float32, N_PARTICLES); wtb_c = zeros(Float32, N_PARTICLES)
    alive_c = ones(Int32, N_PARTICLES)
    dep_c  = zeros(Float32, Int(g.dep_nx), Int(g.dep_ny))

    t0 = time()
    for step in 1:NSTEPS
        step_cpu!(lons_c, lats_c, alts_c, utb_c, vtb_c, wtb_c, alive_c, dep_c,
                  mass, v_grav, era5.u, era5.v, era5.w,
                  era5.ap, era5.b, psfc,
                  noise_lon, noise_lat, noise_w,
                  g, α_h, α_w, β_h, β_w, step, DT)
    end
    t_cpu = time() - t0
    n_alive_c = sum(alive_c)
    n_deposited_c = N_PARTICLES - n_alive_c
    @printf "  Wall time: %.2f s\n" t_cpu
    @printf "  Deposited: %d / %d  (%.1f%%)\n" n_deposited_c N_PARTICLES (100 * n_deposited_c / N_PARTICLES)
    @printf "  Deposition grid: total=%.3e max=%.3e Bq\n" sum(dep_c) maximum(dep_c)

    # ==================== GPU RUN ====================
    println("\n── GPU run ──")

    lons_g = CuArray(lon0); lats_g = CuArray(lat0); alts_g = CuArray(alt0)
    utb_g = CUDA.zeros(Float32, N_PARTICLES); vtb_g = CUDA.zeros(Float32, N_PARTICLES); wtb_g = CUDA.zeros(Float32, N_PARTICLES)
    alive_g = CuArray(ones(Int32, N_PARTICLES))
    dep_g = CUDA.zeros(Float32, Int(g.dep_nx), Int(g.dep_ny))
    mass_d = CuArray(mass); v_grav_d = CuArray(v_grav)
    u_d = CuArray(era5.u); v_d = CuArray(era5.v); w_d = CuArray(era5.w)
    level_alts_d = CuArray(level_alts)
    noise_lon_d = CuArray(noise_lon); noise_lat_d = CuArray(noise_lat); noise_w_d = CuArray(noise_w)

    # Warm-up
    threads = 256
    blocks = cld(N_PARTICLES, threads)
    @cuda threads=threads blocks=blocks step_kernel!(
        lons_g, lats_g, alts_g, utb_g, vtb_g, wtb_g, alive_g, dep_g,
        mass_d, v_grav_d, u_d, v_d, w_d, level_alts_d,
        noise_lon_d, noise_lat_d, noise_w_d,
        g, α_h, α_w, β_h, β_w, Int32(1), DT)
    CUDA.synchronize()

    # Reset state after warm-up
    copyto!(lons_g, lon0); copyto!(lats_g, lat0); copyto!(alts_g, alt0)
    fill!(utb_g, 0f0); fill!(vtb_g, 0f0); fill!(wtb_g, 0f0)
    fill!(alive_g, Int32(1)); fill!(dep_g, 0f0)

    CUDA.synchronize()
    t0 = time()
    for step in 1:NSTEPS
        @cuda threads=threads blocks=blocks step_kernel!(
            lons_g, lats_g, alts_g, utb_g, vtb_g, wtb_g, alive_g, dep_g,
            mass_d, v_grav_d, u_d, v_d, w_d, level_alts_d,
            noise_lon_d, noise_lat_d, noise_w_d,
            g, α_h, α_w, β_h, β_w, Int32(step), DT)
    end
    CUDA.synchronize()
    t_gpu = time() - t0
    n_alive_g = Int(sum(Array(alive_g)))
    n_deposited_g = N_PARTICLES - n_alive_g
    dep_g_host = Array(dep_g)
    @printf "  Wall time: %.2f s\n" t_gpu
    @printf "  Deposited: %d / %d  (%.1f%%)\n" n_deposited_g N_PARTICLES (100 * n_deposited_g / N_PARTICLES)
    @printf "  Deposition grid: total=%.3e max=%.3e Bq\n" sum(dep_g_host) maximum(dep_g_host)

    # ==================== COMPARISON ====================
    println("\n── Comparison ──")
    speedup = t_cpu / t_gpu
    @printf "  Speedup CPU→GPU: %.1f×\n" speedup
    @printf "  Deposition grid max |Δ| = %.3e Bq   rel = %.2e\n" (maximum(abs.(dep_c .- dep_g_host))) (maximum(abs.(dep_c .- dep_g_host)) / max(maximum(dep_c), 1f-30))

    # Compare final particle positions of still-airborne particles
    lons_g_h = Array(lons_g); lats_g_h = Array(lats_g); alts_g_h = Array(alts_g)
    both_alive = (alive_c .== 1) .& (Array(alive_g) .== 1)
    if any(both_alive)
        err_lon = abs.(lons_c[both_alive] .- lons_g_h[both_alive])
        err_lat = abs.(lats_c[both_alive] .- lats_g_h[both_alive])
        err_alt = abs.(alts_c[both_alive] .- alts_g_h[both_alive])
        @printf "  Airborne position error (both alive, n=%d):\n" sum(both_alive)
        @printf "    max |Δlon|=%.2e deg  max |Δlat|=%.2e deg  max |Δalt|=%.2f m\n" maximum(err_lon) maximum(err_lat) maximum(err_alt)
    else
        println("  All particles deposited in both runs")
    end

    # ==================== DOSE RATE + PLOT ====================
    println("\nComputing dose rate and generating comparison plot...")
    dose_c, cell_area = deposition_to_dose_rate(dep_c, g)
    dose_g, _         = deposition_to_dose_rate(dep_g_host, g)

    dep_lons = [g.dep_lon_min + (i - 0.5f0) * g.dep_dx for i in 1:Int(g.dep_nx)]
    dep_lats = [g.dep_lat_min + (j - 0.5f0) * g.dep_dy for j in 1:Int(g.dep_ny)]

    dose_max = max(maximum(dose_c), maximum(dose_g))
    @printf "  Max dose rate: CPU=%.2f GPU=%.2f mR/h\n" maximum(dose_c) maximum(dose_g)

    levels = [0.4f0, 1.0f0, 4.0f0, 10.0f0, 40.0f0, 100.0f0]
    colors = [:blue, :cyan, :green, :yellow, :orange, :red]

    fig = Figure(size = (1400, 800), fontsize = 14)

    ax1 = Axis(fig[1,1], title="CPU — dose rate H+12 (mR/h)", xlabel="Longitude", ylabel="Latitude", aspect=DataAspect())
    ax2 = Axis(fig[1,2], title="GPU — dose rate H+12 (mR/h)", xlabel="Longitude", ylabel="Latitude", aspect=DataAspect())
    ax3 = Axis(fig[1,3], title="Overlay (CPU solid, GPU dashed)", xlabel="Longitude", ylabel="Latitude", aspect=DataAspect())

    for (lv, col) in zip(levels, colors)
        contour!(ax1, dep_lons, dep_lats, dose_c, levels=[lv], color=col, linewidth=2)
        contour!(ax2, dep_lons, dep_lats, dose_g, levels=[lv], color=col, linewidth=2)
        contour!(ax3, dep_lons, dep_lats, dose_c, levels=[lv], color=col, linewidth=2)
        contour!(ax3, dep_lons, dep_lats, dose_g, levels=[lv], color=col, linewidth=1.5, linestyle=:dash)
    end
    for ax in (ax1, ax2, ax3)
        scatter!(ax, [NANCY_LON], [NANCY_LAT], marker=:star5, color=:black, markersize=15)
    end

    legend_elems = [LineElement(color=c, linewidth=3) for c in colors]
    legend_labels = ["$(l) mR/h" for l in levels]
    Legend(fig[2, 1:3], legend_elems, legend_labels, "Dose Rate",
           orientation=:horizontal, tellwidth=false, tellheight=true)

    # Timing + stats summary
    Label(fig[0, 1:3],
          @sprintf("Standalone GPU Nancy — %d particles, %d steps (12 h). CPU %.2f s → GPU %.2f s (%.1f×)",
                   N_PARTICLES, NSTEPS, t_cpu, t_gpu, speedup),
          fontsize=16, font=:bold)

    outpath = joinpath(@__DIR__, "gpu_nancy_comparison.png")
    save(outpath, fig, px_per_unit=2)
    println("  Saved: $outpath")

    println("\n" * "="^72)
    println("Done")
    println("="^72)
end

main()
