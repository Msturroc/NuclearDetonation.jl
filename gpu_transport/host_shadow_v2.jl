#!/usr/bin/env julia
# Host shadow — Float32 CPU mirror of Transport.integrate_timestep!
# =================================================================
# Track B Phase B2: a pure-CPU Float32 implementation of the Nancy inner loop
# that mirrors the upstream `Transport.integrate_timestep!` body line-for-line
# but bypasses Interpolations.jl entirely, doing manual trilinear+temporal
# lookups against raw 4D Float32 met tapes. The point is to debug FP drift on
# CPU first, before porting the same arithmetic onto CUDA. Validated against
# `cpu_reference.jl` via FMS gate (≤0.005 over 6 dose-rate thresholds).
#
# Critical simplification (verified from upstream defaults):
#   - HannaTurbulenceConfig defaults: use_dynamic_L = false, flexpart_mode = false
#   - DepositionConfig default: monin_obukhov_length = 1e10  (neutral)
#   - cpu_reference.jl does not override either
#   ⇒ L = 1e10 always, stability_ratio = h/|L| ≈ 0 → ALWAYS NEUTRAL Hanna branch
#   ⇒ use_cbl branch (`L < 0`) never fires
#   ⇒ no CBL params, no flexpart_vertical_step, no apply_simple_convection
#
# What is mirrored:
#   1. Manual Float32 trilinear+temporal lookups against raw met tapes
#   2. Per-particle hybrid_profile build (length-nk Float32 column) reused
#      across both Heun half-steps and all `ifine` vertical substeps
#   3. Dry deposition exponential decay (always-on; Nancy has dry_active=true)
#   4. Heun 2-stage advection with settling vg_sigma from layer thickness
#   5. Hanna NEUTRAL branch only (sigu, sigv, sigw, dsigwdz, tlu, tlv, tlw)
#   6. Horizontal OU (always)
#   7. Vertical sub-stepping (ifine=5) with OU + drift correction (original mode)
#   8. Reflect/clamp at boundaries; bound check
#   9. Hourly cumulative deposition snapshots
#
# Public API:
#   run_host_shadow(params::Vector{Float64}, gen_seed::UInt64; rng_seed=gen_seed)
#     → (deposition_grid, hourly_dep, n_active_final)
#
# Assumes the upstream `nancy_cmaes_particle_size.jl` has been included (with
# MAX_EVALS=0 to suppress its loop) so MET_CACHE / DOMAIN / NX / NY / NK /
# RELEASE_X / RELEASE_Y / LAYER_LOWER / LAYER_MIDDLE / LAYER_UPPER /
# CACHE_START_FILE / CACHE_END_FILE / LON_GRID / LAT_GRID /
# generate_bimodal_bins / compute_bimodal_weights are bound at module scope.

using Random
using StaticArrays
using NuclearDetonation
using NuclearDetonation.Transport

# ============================================================================
# Float32 wind tape — one met window's worth of raw 4D met data, permuted to
# ascending sigma order (matches what create_wind_interpolants does internally)
# ============================================================================
struct WindTapeF32
    u::Array{Float32,4}        # (nx, ny, nk, 2)
    v::Array{Float32,4}
    w::Array{Float32,4}
    t::Array{Float32,4}
    p::Array{Float32,4}        # 3D pressure (hPa)
    hlevel::Array{Float32,4}   # geopotential height at midpoint (m)
    ps::Array{Float32,3}       # (nx, ny, 2)  surface pressure (hPa)
    hbl::Array{Float32,3}      # (nx, ny, 2)  PBL height (m)
    z_grid::Vector{Float32}    # length-nk sigma levels, ASCENDING
    t1::Float32
    t2::Float32
    nx::Int
    ny::Int
    nk::Int
end

function build_wind_tape(mf::Transport.MeteoFields, t1::Real, t2::Real)
    nx, ny, nk = mf.nx, mf.ny, mf.nk
    z = copy(mf.vlevel)
    perm = sortperm(z)
    z_sorted = Float32.(z[perm])
    # break ties up
    for i in 2:nk
        if z_sorted[i] <= z_sorted[i-1]
            z_sorted[i] = z_sorted[i-1] + Float32(eps(Float32) * 10)
        end
    end

    permuted = perm != collect(1:nk)
    pull3 = (a1, a2) -> begin
        out = Array{Float32,4}(undef, nx, ny, nk, 2)
        if permuted
            @views begin
                out[:, :, :, 1] .= a1[:, :, perm]
                out[:, :, :, 2] .= a2[:, :, perm]
            end
        else
            out[:, :, :, 1] .= a1
            out[:, :, :, 2] .= a2
        end
        replace!(v -> isnan(v) ? 0.0f0 : v, out)
        out
    end

    u = pull3(mf.u1, mf.u2)
    v = pull3(mf.v1, mf.v2)
    w = pull3(mf.w1, mf.w2)
    t = pull3(mf.t1, mf.t2)
    p = pull3(mf.p1, mf.p2)
    h = pull3(mf.hlevel1, mf.hlevel2)
    # Mirror the package's NaN substitutions (create_wind_interpolants):
    replace!(val -> val == 0.0f0 || isnan(val) ? 1013.25f0 : val, p)
    replace!(val -> isnan(val) || val < 100.0f0 ? 288.15f0 : val, t)
    replace!(val -> isnan(val) ? 9999.0f0 : val, h)

    ps = Array{Float32,3}(undef, nx, ny, 2)
    ps[:, :, 1] .= mf.ps1
    ps[:, :, 2] .= mf.ps2
    replace!(v -> isnan(v) ? 1013.25f0 : v, ps)

    hbl = Array{Float32,3}(undef, nx, ny, 2)
    hbl[:, :, 1] .= mf.hbl1
    hbl[:, :, 2] .= mf.hbl2
    replace!(v -> isnan(v) ? 0.0f0 : v, hbl)

    return WindTapeF32(u, v, w, t, p, h, ps, hbl, z_sorted,
                       Float32(t1), Float32(t2), nx, ny, nk)
end

# ============================================================================
# Manual interpolation primitives (mirror Interpolations.Gridded(Linear()))
# ============================================================================
# Find the lower index for value `q` in a sorted ascending vector `grid`.
# Returns (i_lo, frac) so q ≈ grid[i_lo] + frac * (grid[i_lo+1] - grid[i_lo]),
# clamped to [1, length(grid)-1] for i_lo and [0, 1] for frac.
@inline function locate_f32(grid::Vector{Float32}, q::Float32)
    n = length(grid)
    if q <= grid[1]
        return 1, 0.0f0
    elseif q >= grid[n]
        return n - 1, 1.0f0
    end
    # binary search for the upper bound
    lo, hi = 1, n
    @inbounds while lo < hi - 1
        mid = (lo + hi) >> 1
        if grid[mid] <= q
            lo = mid
        else
            hi = mid
        end
    end
    g1 = @inbounds grid[lo]
    g2 = @inbounds grid[lo + 1]
    return lo, (q - g1) / (g2 - g1)
end

# Integer-grid (1:nx) helpers — much cheaper than searching, since x_grid = 1:nx
@inline function locate_int_f32(q::Float32, n::Int)
    if q <= 1.0f0
        return 1, 0.0f0
    elseif q >= Float32(n)
        return n - 1, 1.0f0
    end
    i = unsafe_trunc(Int, q)
    return i, q - Float32(i)
end

# 4D trilinear+temporal interp on (1:nx, 1:ny, z_grid, [t1, t2])
@inline function interp4d_f32(tape::Array{Float32,4}, x::Float32, y::Float32, σ::Float32, tfrac::Float32,
                              z_grid::Vector{Float32}, nx::Int, ny::Int)
    i, fx = locate_int_f32(x, nx)
    j, fy = locate_int_f32(y, ny)
    k, fz = locate_f32(z_grid, σ)
    g_fx = 1.0f0 - fx
    g_fy = 1.0f0 - fy
    g_fz = 1.0f0 - fz
    g_ft = 1.0f0 - tfrac

    @inbounds begin
        # bilinear at (z=k, t=1)
        a00 = tape[i,   j,   k,   1] * g_fx + tape[i+1, j,   k,   1] * fx
        a10 = tape[i,   j+1, k,   1] * g_fx + tape[i+1, j+1, k,   1] * fx
        b00 = tape[i,   j,   k+1, 1] * g_fx + tape[i+1, j,   k+1, 1] * fx
        b10 = tape[i,   j+1, k+1, 1] * g_fx + tape[i+1, j+1, k+1, 1] * fx
        c00 = tape[i,   j,   k,   2] * g_fx + tape[i+1, j,   k,   2] * fx
        c10 = tape[i,   j+1, k,   2] * g_fx + tape[i+1, j+1, k,   2] * fx
        d00 = tape[i,   j,   k+1, 2] * g_fx + tape[i+1, j,   k+1, 2] * fx
        d10 = tape[i,   j+1, k+1, 2] * g_fx + tape[i+1, j+1, k+1, 2] * fx
    end

    a0 = a00 * g_fy + a10 * fy   # k, t1
    b0 = b00 * g_fy + b10 * fy   # k+1, t1
    c0 = c00 * g_fy + c10 * fy   # k, t2
    d0 = d00 * g_fy + d10 * fy   # k+1, t2

    v_t1 = a0 * g_fz + b0 * fz
    v_t2 = c0 * g_fz + d0 * fz
    return v_t1 * g_ft + v_t2 * tfrac
end

# 3D bilinear+temporal interp for surface fields (ps, hbl)
@inline function interp3d_f32(tape::Array{Float32,3}, x::Float32, y::Float32, tfrac::Float32,
                              nx::Int, ny::Int)
    i, fx = locate_int_f32(x, nx)
    j, fy = locate_int_f32(y, ny)
    g_fx = 1.0f0 - fx
    g_fy = 1.0f0 - fy
    g_ft = 1.0f0 - tfrac
    @inbounds begin
        a1 = tape[i,   j,   1] * g_fx + tape[i+1, j,   1] * fx
        a2 = tape[i,   j+1, 1] * g_fx + tape[i+1, j+1, 1] * fx
        b1 = tape[i,   j,   2] * g_fx + tape[i+1, j,   2] * fx
        b2 = tape[i,   j+1, 2] * g_fx + tape[i+1, j+1, 2] * fx
    end
    v1 = a1 * g_fy + a2 * fy
    v2 = b1 * g_fy + b2 * fy
    return v1 * g_ft + v2 * tfrac
end

# Build a per-particle hybrid profile (length-nk Float32 column of heights)
@inline function build_profile_f32!(prof::Vector{Float32}, hlevel::Array{Float32,4},
                                    x::Float32, y::Float32, tfrac::Float32,
                                    nx::Int, ny::Int, nk::Int)
    i, fx = locate_int_f32(x, nx)
    j, fy = locate_int_f32(y, ny)
    g_fx = 1.0f0 - fx
    g_fy = 1.0f0 - fy
    g_ft = 1.0f0 - tfrac
    @inbounds for k in 1:nk
        a1 = hlevel[i, j, k, 1] * g_fx + hlevel[i+1, j, k, 1] * fx
        a2 = hlevel[i, j+1, k, 1] * g_fx + hlevel[i+1, j+1, k, 1] * fx
        b1 = hlevel[i, j, k, 2] * g_fx + hlevel[i+1, j, k, 2] * fx
        b2 = hlevel[i, j+1, k, 2] * g_fx + hlevel[i+1, j+1, k, 2] * fx
        v1 = a1 * g_fy + a2 * fy
        v2 = b1 * g_fy + b2 * fy
        prof[k] = v1 * g_ft + v2 * tfrac
    end
    return nothing
end

# Linear interp of a sigma → height profile column.
# σ_levels are ASCENDING. heights may be DESCENDING (low σ = high altitude),
# but we interpolate in σ-space, so the descending-ness only matters for the
# inverse function below.
@inline function height_from_sigma_f32(prof::Vector{Float32}, σ_levels::Vector{Float32}, σ::Float32)
    k, fz = locate_f32(σ_levels, σ)
    @inbounds h_lo = prof[k]
    @inbounds h_hi = prof[k + 1]
    h = h_lo + fz * (h_hi - h_lo)
    h_min, h_max = extrema(prof)
    return clamp(h, h_min, h_max)
end

@inline function sigma_from_height_f32(prof::Vector{Float32}, σ_levels::Vector{Float32},
                                       z_target::Float32, fallback::Float32)
    n = length(prof)
    # Determine direction
    @inbounds ascending = prof[1] < prof[n]
    # Linear search — n is small (~137 ERA5 levels). Avoid allocation.
    z_target = Float32(z_target)
    if ascending
        @inbounds h1 = prof[1]
        @inbounds h_end = prof[n]
        if z_target <= h1
            @inbounds return clamp(σ_levels[1], 0.0f0, 1.0f0)
        elseif z_target >= h_end
            @inbounds return clamp(σ_levels[n], 0.0f0, 1.0f0)
        end
        @inbounds for k in 1:(n-1)
            h_lo = prof[k]
            h_hi = prof[k+1]
            if h_lo <= z_target <= h_hi && h_hi > h_lo
                ratio = (z_target - h_lo) / (h_hi - h_lo)
                σ = σ_levels[k] + ratio * (σ_levels[k+1] - σ_levels[k])
                return clamp(σ, 0.0f0, 1.0f0)
            end
        end
        return clamp(fallback, 0.0f0, 1.0f0)
    else
        # heights descending — sigma ascending corresponds to descending altitude
        @inbounds h1 = prof[1]    # tallest
        @inbounds h_end = prof[n] # lowest
        if z_target >= h1
            @inbounds return clamp(σ_levels[1], 0.0f0, 1.0f0)
        elseif z_target <= h_end
            @inbounds return clamp(σ_levels[n], 0.0f0, 1.0f0)
        end
        @inbounds for k in 1:(n-1)
            h_hi = prof[k]      # higher altitude (smaller σ)
            h_lo = prof[k+1]    # lower altitude (larger σ)
            if h_lo <= z_target <= h_hi && h_hi > h_lo
                ratio = (h_hi - z_target) / (h_hi - h_lo)
                σ = σ_levels[k] + ratio * (σ_levels[k+1] - σ_levels[k])
                return clamp(σ, 0.0f0, 1.0f0)
            end
        end
        return clamp(fallback, 0.0f0, 1.0f0)
    end
end

# ============================================================================
# Hanna NEUTRAL — the only branch that fires for Nancy (L = 1e10)
# Returns (sigu, sigv, sigw, dsigwdz, tlu, tlv, tlw)
# Mirrors compute_neutral_turbulence in turbulence_hanna.jl:195
# ============================================================================
@inline function hanna_neutral_inline(z::Float32, ust::Float32,
                                      sigma_h_scale::Float32, sigma_w_scale::Float32,
                                      tl_scale::Float32)
    f_cor = 1.0f-4
    tlu_min = 10.0f0
    tlv_min = 10.0f0
    tlw_min = 30.0f0    # NOTE: package default is 30, not 10

    ust_safe = max(1.0f-4, ust)
    corr = z / ust_safe

    sigu = 1.0f-2 + 2.0f0 * ust_safe * exp(-3.0f0 * f_cor * corr)
    sigw_pre = 1.3f0 * ust_safe * exp(-2.0f0 * f_cor * corr)
    dsigwdz = -2.0f0 * f_cor * sigw_pre
    sigw = sigw_pre + 1.0f-2
    sigv = sigw

    tl = 0.5f0 * z / sigw / (1.0f0 + 1.5f-3 * corr)
    tlu = max(tlu_min, tl) * tl_scale
    tlv = max(tlv_min, tl) * tl_scale
    tlw = max(tlw_min, tl) * tl_scale

    if dsigwdz == 0.0f0
        dsigwdz = 1.0f-10
    end

    return (sigu * sigma_h_scale,
            sigv * sigma_h_scale,
            sigw * sigma_w_scale,
            dsigwdz * sigma_w_scale,
            tlu, tlv, tlw)
end

@inline function ou_step_inline(u_old::Float32, sigma::Float32, tl::Float32, dt::Float32, rnd::Float32)
    dt_over_tl = dt / tl
    if dt_over_tl < 0.5f0
        return muladd(sigma * sqrt(2.0f0 * dt_over_tl), rnd, (1.0f0 - dt_over_tl) * u_old)
    else
        r = exp(-dt_over_tl)
        return muladd(sigma * sqrt(1.0f0 - r * r), rnd, r * u_old)
    end
end

# ============================================================================
# Run-once geometry / domain constants pulled from upstream into a struct
# ============================================================================
struct ShadowGeom
    nx_met::Int
    ny_met::Int
    nk::Int
    nx_dom::Int
    ny_dom::Int
    grid_scale_x::Float32
    grid_scale_y::Float32
    map_ratio_x::Float32
    map_ratio_y::Float32
    z_max_m::Float32
    lat_reversed::Bool
    lon_min::Float32
    lon_max::Float32
    lat_min::Float32
    lat_max::Float32
end

function build_shadow_geom(domain::Transport.SimulationDomain, sample_tape::WindTapeF32)
    nx_met, ny_met, nk = sample_tape.nx, sample_tape.ny, sample_tape.nk
    nx_dom, ny_dom = domain.nx, domain.ny
    grid_scale_x = Float32((nx_met - 1) / (nx_dom - 1))
    grid_scale_y = Float32((ny_met - 1) / (ny_dom - 1))
    z_max_m = Float32(maximum(domain.hlevel))

    dlon_deg = (domain.lon_max - domain.lon_min) / (nx_dom - 1)
    dlat_deg = (domain.lat_max - domain.lat_min) / (ny_dom - 1)
    R_earth = 6.371f6
    dx_eq = R_earth * Float32(dlon_deg) * Float32(π) / 180.0f0
    dy_eq = R_earth * Float32(dlat_deg) * Float32(π) / 180.0f0
    map_ratio_x = 1.0f0 / dx_eq
    map_ratio_y = 1.0f0 / dy_eq

    # The package's `lat_reversed = (winds.y_grid[end] < winds.y_grid[1])`
    # always evaluates to FALSE because y_grid is just `1:ny`. The ERA5
    # reader pre-flips the data into S→N order before constructing
    # `MeteoFields`, so met arrays are already y=1 → lat_min, y=ny → lat_max.
    # We mirror that — never use the lat_reversed branch.
    lat_reversed = false

    return ShadowGeom(nx_met, ny_met, nk, nx_dom, ny_dom,
                      grid_scale_x, grid_scale_y, map_ratio_x, map_ratio_y,
                      z_max_m, lat_reversed,
                      Float32(domain.lon_min), Float32(domain.lon_max),
                      Float32(domain.lat_min), Float32(domain.lat_max))
end

# Domain (i, j) → met (x, y) coordinates
@inline function dom_to_met(geom::ShadowGeom, x_dom::Float32, y_dom::Float32)
    x_met = (x_dom - 1.0f0) * geom.grid_scale_x + 1.0f0
    y_met = if geom.lat_reversed
        Float32(geom.ny_met) - (y_dom - 1.0f0) * geom.grid_scale_y
    else
        (y_dom - 1.0f0) * geom.grid_scale_y + 1.0f0
    end
    return x_met, y_met
end

@inline function met_to_dom(geom::ShadowGeom, x_met::Float32, y_met::Float32)
    x_dom = (x_met - 1.0f0) / geom.grid_scale_x + 1.0f0
    y_dom = if geom.lat_reversed
        (Float32(geom.ny_met) - y_met) / geom.grid_scale_y + 1.0f0
    else
        (y_met - 1.0f0) / geom.grid_scale_y + 1.0f0
    end
    return x_dom, y_dom
end

@inline function lat_at_y(geom::ShadowGeom, y_met::Float32)
    lat_frac = (y_met - 1.0f0) / Float32(max(geom.ny_met - 1, 1))
    return geom.lat_min + lat_frac * (geom.lat_max - geom.lat_min)
end

# ============================================================================
# Initial particle generation — reuse the same code path as cpu_reference.jl,
# but lift positions/masses straight into Float32 buffers.
# ============================================================================
struct ShadowParticles
    n::Int
    xs::Vector{Float32}        # domain x
    ys::Vector{Float32}        # domain y
    σs::Vector{Float32}        # sigma
    u_turbs::Vector{Float32}
    v_turbs::Vector{Float32}
    w_turbs::Vector{Float32}
    masses::Vector{Float32}
    grav::Vector{Float32}      # m/s settling for each particle
    diameter::Vector{Float32}  # μm
    active::Vector{Bool}
end

function generate_shadow_particles(params::Vector{Float64}, gen_seed::UInt64)
    d_median_fine     = params[1]
    sigma_g_fine      = params[2]
    d_median_coarse   = params[3]
    sigma_g_coarse    = params[4]
    frac_fine         = params[5]
    frac_lower        = params[6]
    frac_middle       = params[7]
    vgrav_scale       = params[13]
    activity_scale    = params[19]

    frac_upper = clamp(1.0 - frac_lower - frac_middle, 0.05, 1.0)
    rng = Random.MersenneTwister(gen_seed)

    size_bins = generate_bimodal_bins(d_median_fine, sigma_g_fine,
                                      d_median_coarse, sigma_g_coarse;
                                      n_bins = 15)
    bin_weights = compute_bimodal_weights(d_median_fine, sigma_g_fine,
                                          d_median_coarse, sigma_g_coarse,
                                          frac_fine, size_bins)

    n_particles = 1000
    total_activity = activity_scale * 1.0e15
    n_lower  = round(Int, n_particles * frac_lower)
    n_middle = round(Int, n_particles * frac_middle)
    n_upper  = n_particles - n_lower - n_middle

    sources = [
        Transport.ReleaseSource((RELEASE_X, RELEASE_Y), LAYER_LOWER,
                                Transport.BombRelease(0.0),
                                [total_activity * frac_lower],  max(n_lower, 1)),
        Transport.ReleaseSource((RELEASE_X, RELEASE_Y), LAYER_MIDDLE,
                                Transport.BombRelease(0.0),
                                [total_activity * frac_middle], max(n_middle, 1)),
        Transport.ReleaseSource((RELEASE_X, RELEASE_Y), LAYER_UPPER,
                                Transport.BombRelease(0.0),
                                [total_activity * frac_upper],  max(n_upper, 1)),
    ]

    init_met = MET_CACHE[(CACHE_START_FILE, 1)]

    positions_m = Tuple{Float64,Float64,Float64}[]
    activities  = Float64[]
    for src in sources
        pos_s, act_s, released_s = Transport.generate_release_particles(
            rng, src, 0, 1,
            ones(Float64, NX, NY), ones(Float64, NY, NY),
            DOMAIN.dx, DOMAIN.dy, DOMAIN.hlevel)
        if released_s && !isempty(pos_s)
            append!(positions_m, pos_s)
            append!(activities,  act_s)
        end
    end

    n = length(positions_m)
    if n == 0
        return ShadowParticles(0,
            Float32[], Float32[], Float32[],
            Float32[], Float32[], Float32[],
            Float32[], Float32[], Float32[], Bool[])
    end

    cum_weights = cumsum(bin_weights)
    n_classes = length(size_bins)
    fixed_gravity_ms = [Float32(b.v * 0.01 * vgrav_scale) for b in size_bins]
    diameters_um     = [Float32(b.d) for b in size_bins]

    xs = Vector{Float32}(undef, n)
    ys = Vector{Float32}(undef, n)
    σs = Vector{Float32}(undef, n)
    u_turbs = zeros(Float32, n)
    v_turbs = zeros(Float32, n)
    w_turbs = zeros(Float32, n)
    masses  = Vector{Float32}(undef, n)
    grav    = Vector{Float32}(undef, n)
    diameter = Vector{Float32}(undef, n)
    active  = trues(n)

    for i in 1:n
        pos = positions_m[i]
        sigma_z = Transport.height_to_sigma_hybrid(RELEASE_X, RELEASE_Y, pos[3], init_met, 0.0)
        xs[i] = Float32(pos[1])
        ys[i] = Float32(pos[2])
        σs[i] = Float32(clamp(sigma_z, 0.0, 1.0))
        masses[i] = Float32(activities[i])

        r = rand(rng)
        idx = clamp(searchsortedfirst(cum_weights, r), 1, n_classes)
        grav[i] = fixed_gravity_ms[idx]
        diameter[i] = diameters_um[idx]
    end

    return ShadowParticles(n, xs, ys, σs, u_turbs, v_turbs, w_turbs,
                            masses, grav, diameter, active)
end

# ============================================================================
# Main host shadow driver
# ============================================================================
"""
    run_host_shadow(params::Vector{Float64}, gen_seed::UInt64; rng_seed=gen_seed)
        → (deposition_grid, hourly_dep, n_active_final)

Run a Float32 CPU shadow of Nancy through 12 simulated hours and return
deposition binned to the (LON_GRID, LAT_GRID) observation grid.

- `deposition_grid::Matrix{Float32}` shape (nx_obs, ny_obs) — final cumulative Bq
- `hourly_dep::Array{Float32,3}`     shape (nx_obs, ny_obs, 12) — cumulative through hour h
- `n_active_final::Int` — number of particles still alive at end
"""
function run_host_shadow(params::Vector{Float64}, gen_seed::UInt64; rng_seed::UInt64 = gen_seed)
    # --- Parameters (verbatim from rho_core) ---
    sigma_w_scale     = Float32(params[8])
    sigma_h_scale     = Float32(params[9])
    tl_scale          = Float32(params[11])
    vd_scale          = Float32(params[12])
    omega_scale       = Float32(params[14])
    mixing_height_scale  = Float32(params[15])  # unused (deposition_config.mixing_height; L is fixed)
    surface_height_scale = Float32(params[17])
    drag_coef         = 0.05f0  # default

    simple_dep_velocity = 0.002f0 * vd_scale       # m/s, surface-layer extra deposition velocity
    h_surface_m         = 30.0f0 * surface_height_scale

    # --- Particles ---
    pts = generate_shadow_particles(params, gen_seed)
    if pts.n == 0
        nx_obs, ny_obs = length(LON_GRID), length(LAT_GRID)
        return zeros(Float32, nx_obs, ny_obs), zeros(Float32, nx_obs, ny_obs, 12), 0
    end

    # --- Geometry from a sample tape ---
    sample_mf = MET_CACHE[(CACHE_START_FILE, 1)]
    sample_tape = build_wind_tape(sample_mf, 0.0, 3600.0)
    geom = build_shadow_geom(DOMAIN, sample_tape)

    # --- Output grid ---
    nx_obs, ny_obs = length(LON_GRID), length(LAT_GRID)
    dep_grid = zeros(Float32, nx_obs, ny_obs)
    hourly_dep = zeros(Float32, nx_obs, ny_obs, 12)
    lon_grid_f32 = Float32.(LON_GRID)
    lat_grid_f32 = Float32.(LAT_GRID)

    # --- Per-particle scratch buffers ---
    profile_buf = Vector{Float32}(undef, geom.nk)

    # --- RNG ---
    rng = Random.MersenneTwister(rng_seed)

    # --- Sim parameters (Nancy uses 300 s steps, 12 hours, ifine=5) ---
    dt = 300.0f0
    ifine = 5
    dt_sub = dt / Float32(ifine)
    max_duration = 12.0f0 * 3600.0f0
    n_steps = Int(round(max_duration / dt))   # 144

    # Sub-stepping per met window: cached path → 1 hour windows, 12 substeps each
    n_substeps_per_window = Int(round(3600.0f0 / dt))   # 12

    current_time = 0.0f0
    next_hour_idx = 1
    next_hour_time = 3600.0f0

    file_range_start = CACHE_START_FILE
    file_range_end = CACHE_END_FILE

    # =====================================================================
    # MAIN MET LOOP
    # =====================================================================
    for file_idx in file_range_start:file_range_end
        # Count windows for this file
        n_windows = 0
        for k in keys(MET_CACHE)
            if k[1] == file_idx
                n_windows = max(n_windows, k[2])
            end
        end
        n_windows = max(0, n_windows - 1)
        n_windows == 0 && continue

        for window_idx in 1:n_windows
            # Build Float32 tape for THIS window
            mf = MET_CACHE[(file_idx, window_idx)]
            tape = build_wind_tape(mf, 0.0, 3600.0)
            local_time = 0.0f0

            for sub_idx in 1:n_substeps_per_window
                if current_time >= max_duration
                    break
                end

                # tfrac for the current step in the [0, 3600] window
                t_eval_f = local_time / 3600.0f0
                t_end_f  = (local_time + dt) / 3600.0f0

                @inbounds for i in 1:pts.n
                    pts.active[i] || continue

                    # ---------- Particle current state ----------
                    x_dom = pts.xs[i]
                    y_dom = pts.ys[i]
                    σ_p = clamp(pts.σs[i], 0.0f0, 1.0f0)
                    x_met, y_met = dom_to_met(geom, x_dom, y_dom)

                    # Build hybrid profile at start position
                    build_profile_f32!(profile_buf, tape.hlevel, x_met, y_met,
                                       t_eval_f, geom.nx_met, geom.ny_met, geom.nk)
                    z_height = height_from_sigma_f32(profile_buf, tape.z_grid, σ_p)
                    z_sigma = sigma_from_height_f32(profile_buf, tape.z_grid, z_height, σ_p)
                    z_sigma = clamp(z_sigma, 0.0f0, 1.0f0)

                    # ---------- STEP 1: Dry deposition (always-on for Nancy) ----------
                    if z_sigma > 0.996f0
                        vg_ms = pts.grav[i]
                        vd_simple = simple_dep_velocity + vg_ms
                        k_dep = vd_simple / h_surface_m
                        decay_factor = exp(-k_dep * dt)

                        mass = pts.masses[i]
                        if mass > 0.0f0
                            new_mass = mass * decay_factor
                            deposited = mass - new_mass
                            pts.masses[i] = new_mass

                            # Bin deposition into observation grid
                            lon = geom.lon_min + (x_dom - 1.0f0) *
                                  (geom.lon_max - geom.lon_min) / Float32(geom.nx_dom - 1)
                            lat = geom.lat_min + (y_dom - 1.0f0) *
                                  (geom.lat_max - geom.lat_min) / Float32(geom.ny_dom - 1)
                            if lon > 180.0f0
                                lon -= 360.0f0
                            end
                            i_obs = searchsortedlast(lon_grid_f32, lon)
                            j_obs = searchsortedlast(lat_grid_f32, lat)
                            if 1 <= i_obs <= nx_obs && 1 <= j_obs <= ny_obs
                                dep_grid[i_obs, j_obs] += deposited
                            end
                        end

                        # Complete deposition for ≥20 μm at z_sigma ≥ 0.999
                        if z_sigma >= 0.999f0 && pts.diameter[i] >= 20.0f0
                            mass = pts.masses[i]
                            if mass > 0.0f0
                                lon = geom.lon_min + (x_dom - 1.0f0) *
                                      (geom.lon_max - geom.lon_min) / Float32(geom.nx_dom - 1)
                                lat = geom.lat_min + (y_dom - 1.0f0) *
                                      (geom.lat_max - geom.lat_min) / Float32(geom.ny_dom - 1)
                                if lon > 180.0f0
                                    lon -= 360.0f0
                                end
                                i_obs = searchsortedlast(lon_grid_f32, lon)
                                j_obs = searchsortedlast(lat_grid_f32, lat)
                                if 1 <= i_obs <= nx_obs && 1 <= j_obs <= ny_obs
                                    dep_grid[i_obs, j_obs] += mass
                                end
                            end
                            pts.masses[i] = 0.0f0
                            pts.active[i] = false
                            continue
                        end

                        if pts.masses[i] < 1.0f-10
                            pts.active[i] = false
                            continue
                        end
                    end

                    # ---------- STEP 2: Heun 2-stage advection ----------
                    # First evaluation at (x_met, y_met, σ, t_eval_f)
                    u1_w = interp4d_f32(tape.u, x_met, y_met, σ_p, t_eval_f, tape.z_grid, geom.nx_met, geom.ny_met)
                    v1_w = interp4d_f32(tape.v, x_met, y_met, σ_p, t_eval_f, tape.z_grid, geom.nx_met, geom.ny_met)
                    w1_w = interp4d_f32(tape.w, x_met, y_met, σ_p, t_eval_f, tape.z_grid, geom.nx_met, geom.ny_met) * omega_scale

                    # vg → σ tendency from layer thickness around σ_p
                    vg_ms = pts.grav[i]
                    # Avoid double settling at surface (dry deposition handles it)
                    vg_sigma1 = if σ_p > 0.996f0
                        0.0f0
                    else
                        z_clamped = clamp(σ_p, tape.z_grid[1] + Float32(eps(Float32)),
                                          tape.z_grid[end] - Float32(eps(Float32)))
                        idx_g, _ = locate_f32(tape.z_grid, z_clamped)
                        σ_up = tape.z_grid[idx_g]
                        σ_dn = tape.z_grid[idx_g + 1]
                        h_up = profile_buf[idx_g]
                        h_dn = profile_buf[idx_g + 1]
                        dsig = σ_dn - σ_up
                        dz = h_up - h_dn
                        if abs(dz) < Float32(eps(Float32))
                            vg_ms / geom.z_max_m
                        else
                            vg_ms * dsig / dz
                        end
                    end

                    lat_deg = lat_at_y(geom, y_met)
                    clat = max(cos(lat_deg * Float32(π) / 180.0f0), 0.01745f0)
                    xm_factor = 1.0f0 / clat

                    du1_x = u1_w * geom.map_ratio_x * xm_factor
                    du1_y = v1_w * geom.map_ratio_y
                    du1_z = w1_w + vg_sigma1

                    # Predictor in met coords
                    x_met_pred = x_met + du1_x * dt
                    y_met_pred = y_met + du1_y * dt
                    σ_pred = clamp(σ_p + du1_z * dt, 0.0f0, 1.0f0)

                    # Profile at predictor (NB: package rebuilds profile_local inside the RHS,
                    # but the same buffer is shared. We rebuild here for the eval2 settling.)
                    build_profile_f32!(profile_buf, tape.hlevel, x_met_pred, y_met_pred,
                                       t_end_f, geom.nx_met, geom.ny_met, geom.nk)

                    u2_w = interp4d_f32(tape.u, x_met_pred, y_met_pred, σ_pred, t_end_f, tape.z_grid, geom.nx_met, geom.ny_met)
                    v2_w = interp4d_f32(tape.v, x_met_pred, y_met_pred, σ_pred, t_end_f, tape.z_grid, geom.nx_met, geom.ny_met)
                    w2_w = interp4d_f32(tape.w, x_met_pred, y_met_pred, σ_pred, t_end_f, tape.z_grid, geom.nx_met, geom.ny_met) * omega_scale

                    vg_sigma2 = if σ_pred > 0.996f0
                        0.0f0
                    else
                        z_clamped = clamp(σ_pred, tape.z_grid[1] + Float32(eps(Float32)),
                                          tape.z_grid[end] - Float32(eps(Float32)))
                        idx_g, _ = locate_f32(tape.z_grid, z_clamped)
                        σ_up = tape.z_grid[idx_g]
                        σ_dn = tape.z_grid[idx_g + 1]
                        h_up = profile_buf[idx_g]
                        h_dn = profile_buf[idx_g + 1]
                        dsig = σ_dn - σ_up
                        dz = h_up - h_dn
                        if abs(dz) < Float32(eps(Float32))
                            vg_ms / geom.z_max_m
                        else
                            vg_ms * dsig / dz
                        end
                    end

                    lat_deg2 = lat_at_y(geom, y_met_pred)
                    clat2 = max(cos(lat_deg2 * Float32(π) / 180.0f0), 0.01745f0)
                    xm2 = 1.0f0 / clat2

                    du2_x = u2_w * geom.map_ratio_x * xm2
                    du2_y = v2_w * geom.map_ratio_y
                    du2_z = w2_w + vg_sigma2

                    half_dt = dt * 0.5f0
                    x_met_final = x_met + (du1_x + du2_x) * half_dt
                    y_met_final = y_met + (du1_y + du2_y) * half_dt
                    σ_after_adv = clamp(σ_p + (du1_z + du2_z) * half_dt, 0.0f0, 1.0f0)

                    # Convert final met coords back to domain
                    x_dom_final, y_dom_final = met_to_dom(geom, x_met_final, y_met_final)

                    # Profile at the final post-advection position (used by turbulence)
                    build_profile_f32!(profile_buf, tape.hlevel, x_met_final, y_met_final,
                                       t_end_f, geom.nx_met, geom.ny_met, geom.nk)

                    z_sigma_dep = clamp(σ_after_adv, 0.0f0, 1.0f0)

                    # Particle below ground after advection — surface deposit + kill
                    z_height_after = height_from_sigma_f32(profile_buf, tape.z_grid, z_sigma_dep)
                    if z_height_after < 0.0f0
                        mass = pts.masses[i]
                        if mass > 0.0f0
                            lon = geom.lon_min + (x_dom_final - 1.0f0) *
                                  (geom.lon_max - geom.lon_min) / Float32(geom.nx_dom - 1)
                            lat = geom.lat_min + (y_dom_final - 1.0f0) *
                                  (geom.lat_max - geom.lat_min) / Float32(geom.ny_dom - 1)
                            if lon > 180.0f0
                                lon -= 360.0f0
                            end
                            i_obs = searchsortedlast(lon_grid_f32, lon)
                            j_obs = searchsortedlast(lat_grid_f32, lat)
                            if 1 <= i_obs <= nx_obs && 1 <= j_obs <= ny_obs
                                dep_grid[i_obs, j_obs] += mass
                            end
                        end
                        pts.active[i] = false
                        continue
                    end

                    # ---------- STEP 3: Hanna turbulence (NEUTRAL only) ----------
                    # h, ust at current met position (after advection)
                    h_dynamic = interp3d_f32(tape.hbl, x_met_final, y_met_final, t_eval_f,
                                             geom.nx_met, geom.ny_met)
                    h_pbl = max(if h_dynamic > 0.0f0; h_dynamic else 1000.0f0 * mixing_height_scale end, 50.0f0)

                    u_surf = interp4d_f32(tape.u, x_met_final, y_met_final, 1.0f0, t_eval_f,
                                          tape.z_grid, geom.nx_met, geom.ny_met)
                    v_surf = interp4d_f32(tape.v, x_met_final, y_met_final, 1.0f0, t_eval_f,
                                          tape.z_grid, geom.nx_met, geom.ny_met)
                    u_mag = sqrt(u_surf * u_surf + v_surf * v_surf)
                    ust = max(drag_coef * u_mag, 0.01f0)

                    # Convert post-advection sigma to height for the turbulence loop
                    z_m_current = clamp(z_height_after, 0.0f0, geom.z_max_m)

                    sigu, sigv, sigw, dsigwdz, tlu, tlv, tlw =
                        hanna_neutral_inline(z_m_current, ust, sigma_h_scale, sigma_w_scale, tl_scale)

                    # Horizontal OU (always)
                    rnd_u = randn(rng, Float32)
                    rnd_v = randn(rng, Float32)
                    pts.u_turbs[i] = ou_step_inline(pts.u_turbs[i], sigu, tlu, dt, rnd_u)
                    pts.v_turbs[i] = ou_step_inline(pts.v_turbs[i], sigv, tlv, dt, rnd_v)

                    at_ground = z_sigma_dep >= 0.9999f0
                    z_sigma_current = z_sigma_dep

                    if at_ground
                        pts.w_turbs[i] = 0.0f0
                    else
                        # Density at current sigma
                        T_k = interp4d_f32(tape.t, x_met, y_met, z_sigma_current, t_eval_f,
                                           tape.z_grid, geom.nx_met, geom.ny_met)
                        ps_pa = interp3d_f32(tape.ps, x_met, y_met, t_eval_f,
                                             geom.nx_met, geom.ny_met) * 100.0f0
                        P_pa = ps_pa * z_sigma_current
                        R_air = 287.0f0
                        g = 9.81f0
                        rhoa = P_pa / (R_air * T_k)
                        rhograd = -rhoa * g / (R_air * T_k)

                        for i_sub in 1:ifine
                            sigu_s, sigv_s, sigw_s, dsigwdz_s, tlu_s, tlv_s, tlw_s = if i_sub == 1
                                (sigu, sigv, sigw, dsigwdz, tlu, tlv, tlw)
                            else
                                hanna_neutral_inline(z_m_current, ust,
                                                     sigma_h_scale, sigma_w_scale, tl_scale)
                            end

                            rnd_w = randn(rng, Float32)

                            # Original mode: OU then drift correction (no CBL since L = 1e10)
                            w_old = pts.w_turbs[i]
                            w_new = ou_step_inline(w_old, sigw_s, tlw_s, dt_sub, rnd_w)
                            pts.w_turbs[i] = w_new

                            w_drift_grad = sigw_s * dsigwdz_s
                            w_drift_skew = if abs(sigw_s) > 0.01f0
                                (w_new * w_new / sigw_s) * dsigwdz_s
                            else
                                0.0f0
                            end
                            w_drift_dens = if abs(rhoa) > 0.01f0
                                (sigw_s * sigw_s / rhoa) * rhograd
                            else
                                0.0f0
                            end
                            w_drift = w_drift_grad + w_drift_skew + w_drift_dens
                            w_total = w_new + w_drift

                            delz_m = w_total * dt_sub
                            z_m_new = z_m_current + delz_m
                            z_m_new = clamp(z_m_new, 0.0f0, geom.z_max_m)

                            z_sigma_new = sigma_from_height_f32(profile_buf, tape.z_grid, z_m_new, z_sigma_current)
                            z_sigma_current = clamp(z_sigma_new, 0.0f0, 1.0f0)

                            z_m_current = height_from_sigma_f32(profile_buf, tape.z_grid, z_sigma_current)
                            z_m_current = clamp(z_m_current, 0.0f0, geom.z_max_m)

                            if i_sub < ifine
                                T_k = interp4d_f32(tape.t, x_met, y_met, z_sigma_current, t_eval_f,
                                                   tape.z_grid, geom.nx_met, geom.ny_met)
                                ps_pa = interp3d_f32(tape.ps, x_met, y_met, t_eval_f,
                                                     geom.nx_met, geom.ny_met) * 100.0f0
                                P_pa = ps_pa * z_sigma_current
                                rhoa = P_pa / (R_air * T_k)
                                rhograd = -rhoa * g / (R_air * T_k)
                            end
                        end
                    end

                    # Apply turbulent horizontal displacements
                    x_dom_final += pts.u_turbs[i] * dt * geom.map_ratio_x
                    y_dom_final += pts.v_turbs[i] * dt * geom.map_ratio_y

                    z_sigma_final = z_sigma_current
                    if z_sigma_final >= 0.996f0
                        pts.w_turbs[i] = 0.0f0
                    end
                    z_sigma_final = clamp(z_sigma_final, 0.0f0, 1.0f0)

                    # Bounds check
                    if !(1.0f0 <= x_dom_final <= Float32(geom.nx_dom)) ||
                       !(1.0f0 <= y_dom_final <= Float32(geom.ny_dom))
                        pts.active[i] = false
                        continue
                    end

                    pts.xs[i] = x_dom_final
                    pts.ys[i] = y_dom_final
                    pts.σs[i] = z_sigma_final
                end

                current_time += dt
                local_time   += dt

                # Hourly snapshot
                while next_hour_idx <= 12 && current_time >= next_hour_time - 0.5f0
                    @inbounds for jj in 1:ny_obs, ii in 1:nx_obs
                        hourly_dep[ii, jj, next_hour_idx] = dep_grid[ii, jj]
                    end
                    next_hour_idx += 1
                    next_hour_time += 3600.0f0
                end
            end

            current_time >= max_duration && break
        end
        current_time >= max_duration && break
    end

    # Final snapshot guard
    while next_hour_idx <= 12
        @inbounds for jj in 1:ny_obs, ii in 1:nx_obs
            hourly_dep[ii, jj, next_hour_idx] = dep_grid[ii, jj]
        end
        next_hour_idx += 1
    end

    n_active = count(pts.active)
    return dep_grid, hourly_dep, n_active
end
