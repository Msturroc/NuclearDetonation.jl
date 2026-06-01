#!/usr/bin/env julia
# GPU Particle Transport Prototype — Full physics benchmark
# ==========================================================
# Loads a real ERA5 model-level file and runs a 12-hour transport simulation
# of N particles on both CPU and GPU. Physics: advection + gravitational
# settling + Ornstein-Uhlenbeck turbulence. Noise is pre-generated on the
# host and shared between CPU and GPU runs so the comparison is bitwise
# (up to Float32 non-associativity).
#
# Reports for each N:
#   - CPU and GPU wall times (12 h sim, 144 timesteps)
#   - Max and mean particle-position error
#   - Plume centroid and spread (sanity check)
#
# Standalone — no NuclearDetonation.jl dependency.
#
# Run: julia --threads=auto --project=/home/marc/NuclearDetonation.jl \
#        gpu_advection_prototype.jl

using CUDA
using BenchmarkTools
using NCDatasets
using Printf
using Random
using Statistics

# ============================================================================
# Load real ERA5 winds
# ============================================================================

const ERA5_FILE = "/home/marc/NuclearDetonation.jl/examples/smoky_example/ERA5_data/era5_19570901_03-05_ml.nc"

function load_era5()
    NCDataset(ERA5_FILE) do ds
        lon  = Float32.(Array(ds["longitude"][:]))
        lat  = Float32.(Array(ds["latitude"][:]))
        u3 = Float32.(Array(ds["u"][:,:,:,1]))
        v3 = Float32.(Array(ds["v"][:,:,:,1]))
        w3 = Float32.(Array(ds["w"][:,:,:,1]))
        return (; lon, lat, u=u3, v=v3, w=w3)
    end
end

# ============================================================================
# Grid metadata — passed to kernels as scalars via a plain struct
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
end

function make_grid(era5)
    lon = era5.lon; lat = era5.lat
    nx = Int32(length(lon)); ny = Int32(length(lat)); nz = Int32(size(era5.u, 3))
    dx = abs(lon[2] - lon[1])
    dy = abs(lat[2] - lat[1])
    mid_lat = 0.5f0 * (lat[1] + lat[end])
    Grid(
        Float32(minimum(lon)),
        Float32(minimum(lat)),
        Float32(dx),
        Float32(dy),
        nx, ny, nz,
        111_000.0f0,
        111_000.0f0 * cosd(mid_lat),
    )
end

# ============================================================================
# Physics parameters (match rough SNAP / NuclearDetonation defaults)
# ============================================================================

struct PhysicsParams
    sigma_u::Float32   # horizontal turbulent velocity stddev (m/s)
    sigma_v::Float32
    sigma_w::Float32   # vertical turbulent velocity stddev (m/s)
    tau_u::Float32     # horizontal Lagrangian timescale (s)
    tau_v::Float32
    tau_w::Float32     # vertical Lagrangian timescale (s)
    w_grav::Float32    # constant gravitational settling velocity (m/s)
end

const PHYSICS = PhysicsParams(
    1.5f0, 1.5f0, 0.3f0,     # σ_u, σ_v, σ_w
    300.0f0, 300.0f0, 100.0f0, # τ (seconds)
    0.05f0,                    # gravity: 5 cm/s (fine particle)
)

# ============================================================================
# Trilinear interpolation (host + device versions)
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
        c000 = field[i0,   j0,   k0  ]
        c100 = field[i0+1, j0,   k0  ]
        c010 = field[i0,   j0+1, k0  ]
        c110 = field[i0+1, j0+1, k0  ]
        c001 = field[i0,   j0,   k0+1]
        c101 = field[i0+1, j0,   k0+1]
        c011 = field[i0,   j0+1, k0+1]
        c111 = field[i0+1, j0+1, k0+1]
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
        c000 = field[i0,   j0,   k0  ]
        c100 = field[i0+1, j0,   k0  ]
        c010 = field[i0,   j0+1, k0  ]
        c110 = field[i0+1, j0+1, k0  ]
        c001 = field[i0,   j0,   k0+1]
        c101 = field[i0+1, j0,   k0+1]
        c011 = field[i0,   j0+1, k0+1]
        c111 = field[i0+1, j0+1, k0+1]
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
# CPU integration: advection + gravity + OU turbulence
# Noise tensor shape (N, 3, nsteps) — read once per (particle, step)
# ============================================================================

function integrate_cpu!(lons, lats, zs, utb, vtb, wtb,
                        u_field, v_field, w_field,
                        noise,
                        g::Grid, phys::PhysicsParams, dt, nsteps)

    α_u = exp(-dt / phys.tau_u)
    α_v = exp(-dt / phys.tau_v)
    α_w = exp(-dt / phys.tau_w)
    β_u = sqrt(1 - α_u^2) * phys.sigma_u
    β_v = sqrt(1 - α_v^2) * phys.sigma_v
    β_w = sqrt(1 - α_w^2) * phys.sigma_w

    # Convert w_grav (m/s) → model-level units per second.
    # The demo sticks with the scale factor used for vertical wind.
    w_grav_lev = phys.w_grav * 0.001f0

    for step in 1:nsteps
        Threads.@threads for p in eachindex(lons)
            @inbounds begin
                lon = lons[p]; lat = lats[p]; zlev = zs[p]
                ut  = utb[p]; vt  = vtb[p]; wt  = wtb[p]

                u = trilinear_host(u_field, lon, lat, zlev, g)
                v = trilinear_host(v_field, lon, lat, zlev, g)
                w = trilinear_host(w_field, lon, lat, zlev, g)

                # OU update
                ut_new = α_u * ut + β_u * noise[p, 1, step]
                vt_new = α_v * vt + β_v * noise[p, 2, step]
                wt_new = α_w * wt + β_w * noise[p, 3, step]

                # Combined velocity: mean wind + turbulent perturbation
                eff_u = u + ut_new
                eff_v = v + vt_new
                eff_w = w + wt_new

                lons[p] = lon + eff_u * dt / g.m_per_deg_lon
                lats[p] = lat + eff_v * dt / g.m_per_deg_lat
                new_z = zlev + eff_w * dt * 0.001f0 - w_grav_lev * dt
                zs[p] = clamp(new_z, 0.0f0, Float32(g.nz - 1))

                utb[p] = ut_new; vtb[p] = vt_new; wtb[p] = wt_new
            end
        end
    end
end

# ============================================================================
# GPU kernel: one particle per thread, one step per kernel call
# ============================================================================

function transport_kernel!(lons, lats, zs, utb, vtb, wtb,
                           u_field, v_field, w_field,
                           noise, step::Int32,
                           α_u::Float32, α_v::Float32, α_w::Float32,
                           β_u::Float32, β_v::Float32, β_w::Float32,
                           w_grav_lev::Float32,
                           g::Grid, dt::Float32)
    p = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    p > length(lons) && return nothing

    @inbounds begin
        lon = lons[p]; lat = lats[p]; zlev = zs[p]
        ut  = utb[p];  vt  = vtb[p];  wt  = wtb[p]

        u = trilinear_dev(u_field, lon, lat, zlev, g)
        v = trilinear_dev(v_field, lon, lat, zlev, g)
        w = trilinear_dev(w_field, lon, lat, zlev, g)

        ut_new = α_u * ut + β_u * noise[p, Int32(1), step]
        vt_new = α_v * vt + β_v * noise[p, Int32(2), step]
        wt_new = α_w * wt + β_w * noise[p, Int32(3), step]

        eff_u = u + ut_new
        eff_v = v + vt_new
        eff_w = w + wt_new

        lons[p] = lon + eff_u * dt / g.m_per_deg_lon
        lats[p] = lat + eff_v * dt / g.m_per_deg_lat
        new_z = zlev + eff_w * dt * 0.001f0 - w_grav_lev * dt
        zs[p] = max(0.0f0, min(Float32(g.nz - 1), new_z))

        utb[p] = ut_new; vtb[p] = vt_new; wtb[p] = wt_new
    end
    return nothing
end

function integrate_gpu!(lons_d, lats_d, zs_d, utb_d, vtb_d, wtb_d,
                        u_d, v_d, w_d, noise_d, g::Grid, phys::PhysicsParams,
                        dt, nsteps)
    α_u = exp(-dt / phys.tau_u)
    α_v = exp(-dt / phys.tau_v)
    α_w = exp(-dt / phys.tau_w)
    β_u = sqrt(1 - α_u^2) * phys.sigma_u
    β_v = sqrt(1 - α_v^2) * phys.sigma_v
    β_w = sqrt(1 - α_w^2) * phys.sigma_w
    w_grav_lev = phys.w_grav * 0.001f0

    n = length(lons_d)
    threads = 256
    blocks = cld(n, threads)
    for step in 1:nsteps
        @cuda threads=threads blocks=blocks transport_kernel!(
            lons_d, lats_d, zs_d, utb_d, vtb_d, wtb_d,
            u_d, v_d, w_d, noise_d, Int32(step),
            α_u, α_v, α_w, β_u, β_v, β_w, w_grav_lev, g, Float32(dt))
    end
    CUDA.synchronize()
end

# ============================================================================
# Main
# ============================================================================

function main()
    println("="^72)
    println("GPU TRANSPORT PROTOTYPE — Real ERA5 + OU turbulence + settling")
    println("="^72)

    println("\nLoading ERA5 winds from $(basename(ERA5_FILE))")
    era5 = load_era5()
    g = make_grid(era5)
    @printf "  Grid: %d×%d×%d  (lon %.2f..%.2f, lat %.2f..%.2f)\n" g.nx g.ny g.nz (g.lon_min) (g.lon_min + (g.nx-1)*g.dx) (g.lat_min) (g.lat_min + (g.ny-1)*g.dy)
    @printf "  |u| range: %.2f – %.2f m/s\n" extrema(era5.u)...
    println("  Threads.nthreads() = $(Threads.nthreads())")
    println("  CUDA device: ", CUDA.name(CUDA.device()))

    u_d = CuArray(era5.u); v_d = CuArray(era5.v); w_d = CuArray(era5.w)

    dt = 300.0f0
    nsteps = 144  # 12 h
    println("\nSimulation: dt=$(dt)s, $(nsteps) steps ($(Int(nsteps*dt/3600)) h)")
    @printf "Physics: σ_u=%.1f σ_v=%.1f σ_w=%.2f (m/s)   τ_u=%.0f τ_w=%.0f (s)   w_grav=%.2f m/s\n" (PHYSICS.sigma_u) (PHYSICS.sigma_v) (PHYSICS.sigma_w) (PHYSICS.tau_u) (PHYSICS.tau_w) (PHYSICS.w_grav)

    rng = MersenneTwister(42)

    particle_counts = [1000, 2500, 10_000]
    for N in particle_counts
        println("\n── N = $N particles ──")

        mid_lon = 0.5f0 * (g.lon_min + g.lon_min + (g.nx-1)*g.dx)
        mid_lat = 0.5f0 * (g.lat_min + g.lat_min + (g.ny-1)*g.dy)
        lon0 = mid_lon .+ 0.5f0 .* (rand(rng, Float32, N) .- 0.5f0)
        lat0 = mid_lat .+ 0.5f0 .* (rand(rng, Float32, N) .- 0.5f0)
        z0   = 100.0f0 .+ 10.0f0 .* rand(rng, Float32, N)

        # Pre-generate noise tensor (N, 3, nsteps), shared by CPU and GPU.
        noise = randn(rng, Float32, N, 3, nsteps)
        noise_d = CuArray(noise)

        # ---- CPU run ----
        lons_c = copy(lon0); lats_c = copy(lat0); zs_c = copy(z0)
        utb_c = zeros(Float32, N); vtb_c = zeros(Float32, N); wtb_c = zeros(Float32, N)
        t_cpu = @belapsed integrate_cpu!($lons_c, $lats_c, $zs_c,
                                         $utb_c, $vtb_c, $wtb_c,
                                         $(era5.u), $(era5.v), $(era5.w),
                                         $noise, $g, $PHYSICS, $dt, $nsteps) samples=3 evals=1 setup=(
            $lons_c .= $lon0; $lats_c .= $lat0; $zs_c .= $z0;
            $utb_c .= 0; $vtb_c .= 0; $wtb_c .= 0)

        # ---- GPU run ----
        lons_g = CuArray(lon0); lats_g = CuArray(lat0); zs_g = CuArray(z0)
        utb_g = CUDA.zeros(Float32, N); vtb_g = CUDA.zeros(Float32, N); wtb_g = CUDA.zeros(Float32, N)

        # Warm-up / JIT
        integrate_gpu!(lons_g, lats_g, zs_g, utb_g, vtb_g, wtb_g,
                       u_d, v_d, w_d, noise_d, g, PHYSICS, dt, 1)
        CUDA.synchronize()

        t_gpu = @belapsed (integrate_gpu!($lons_g, $lats_g, $zs_g,
                                           $utb_g, $vtb_g, $wtb_g,
                                           $u_d, $v_d, $w_d, $noise_d,
                                           $g, $PHYSICS, $dt, $nsteps);
                           CUDA.synchronize()) samples=3 evals=1 setup=(
            copyto!($lons_g, $lon0); copyto!($lats_g, $lat0); copyto!($zs_g, $z0);
            fill!($utb_g, 0); fill!($vtb_g, 0); fill!($wtb_g, 0))

        # ---- Accuracy comparison on final positions ----
        lons_g_host = Array(lons_g); lats_g_host = Array(lats_g); zs_g_host = Array(zs_g)
        err_lon = abs.(lons_c .- lons_g_host)
        err_lat = abs.(lats_c .- lats_g_host)
        max_err_deg = max(maximum(err_lon), maximum(err_lat))
        mean_err_deg = 0.5f0 * (sum(err_lon)/N + sum(err_lat)/N)
        max_err_m = max_err_deg * g.m_per_deg_lon
        mean_err_m = mean_err_deg * g.m_per_deg_lon

        # Plume statistics sanity check
        plume_centroid = (mean(lons_c), mean(lats_c))
        plume_spread = (std(lons_c), std(lats_c))

        speedup = t_cpu / t_gpu
        @printf "  CPU: %8.2f ms   GPU: %8.2f ms   speedup: %6.1f×\n" (t_cpu*1000) (t_gpu*1000) speedup
        @printf "  Final-position error: max=%.2e m   mean=%.2e m\n" max_err_m mean_err_m
        @printf "  Plume centroid (lon,lat): (%.3f, %.3f)   spread: (%.3f, %.3f) deg\n" plume_centroid[1] plume_centroid[2] plume_spread[1] plume_spread[2]
    end

    println("\n" * "="^72)
    println("Done")
    println("="^72)
end

main()
