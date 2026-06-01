#!/usr/bin/env julia
# CUDA kernel — GPU port of host_shadow_v2.jl per-particle inner loop
# ====================================================================
# Mirrors `run_host_shadow` arithmetic Float32-for-Float32 so the per-cell
# diff stays within ~1e-3 rel and FMS gap vs reference stays at the chaotic
# noise floor (~0.03). One thread = one particle. Twelve kernel launches per
# 12-hour Nancy simulation (one per met window), with hourly snapshot of
# `dep_grid → hourly_dep[:, :, h]` between launches.
#
# Public API:
#   run_gpu_shadow(params, gen_seed; rng_seed=gen_seed)
#       → (deposition_grid::Matrix{Float32},
#          hourly_dep::Array{Float32,3},
#          n_active_final::Int)
#
# Depends on:
#   - met_upload.jl       (GpuMetWindow, load_nancy_gpu_windows, build_gpu_window)
#   - host_shadow_v2.jl   (generate_shadow_particles, ShadowParticles, ShadowGeom,
#                          build_shadow_geom, build_wind_tape)

using CUDA
using Random
using NuclearDetonation
using NuclearDetonation.Transport

# ============================================================================
# Device helpers — match host_shadow_v2.jl primitives byte-for-byte.
# ============================================================================
@inline function locate_int_dev(q::Float32, n::Int)
    if q <= 1.0f0
        return 1, 0.0f0
    elseif q >= Float32(n)
        return n - 1, 1.0f0
    end
    i = unsafe_trunc(Int32, q)
    return Int(i), q - Float32(i)
end

@inline function locate_dev(grid, q::Float32)
    n = length(grid)
    @inbounds if q <= grid[1]
        return 1, 0.0f0
    elseif q >= grid[n]
        return n - 1, 1.0f0
    end
    lo = 1
    hi = n
    @inbounds while lo < hi - 1
        mid = (lo + hi) >> 1
        if grid[mid] <= q
            lo = mid
        else
            hi = mid
        end
    end
    @inbounds g1 = grid[lo]
    @inbounds g2 = grid[lo + 1]
    return lo, (q - g1) / (g2 - g1)
end

@inline function interp4d_dev(tape, x::Float32, y::Float32, σ::Float32, tfrac::Float32,
                              z_grid, nx::Int, ny::Int)
    i, fx = locate_int_dev(x, nx)
    j, fy = locate_int_dev(y, ny)
    k, fz = locate_dev(z_grid, σ)
    g_fx = 1.0f0 - fx
    g_fy = 1.0f0 - fy
    g_fz = 1.0f0 - fz
    g_ft = 1.0f0 - tfrac

    @inbounds begin
        a00 = tape[i,   j,   k,   1] * g_fx + tape[i+1, j,   k,   1] * fx
        a10 = tape[i,   j+1, k,   1] * g_fx + tape[i+1, j+1, k,   1] * fx
        b00 = tape[i,   j,   k+1, 1] * g_fx + tape[i+1, j,   k+1, 1] * fx
        b10 = tape[i,   j+1, k+1, 1] * g_fx + tape[i+1, j+1, k+1, 1] * fx
        c00 = tape[i,   j,   k,   2] * g_fx + tape[i+1, j,   k,   2] * fx
        c10 = tape[i,   j+1, k,   2] * g_fx + tape[i+1, j+1, k,   2] * fx
        d00 = tape[i,   j,   k+1, 2] * g_fx + tape[i+1, j,   k+1, 2] * fx
        d10 = tape[i,   j+1, k+1, 2] * g_fx + tape[i+1, j+1, k+1, 2] * fx
    end
    a0 = a00 * g_fy + a10 * fy
    b0 = b00 * g_fy + b10 * fy
    c0 = c00 * g_fy + c10 * fy
    d0 = d00 * g_fy + d10 * fy
    v_t1 = a0 * g_fz + b0 * fz
    v_t2 = c0 * g_fz + d0 * fz
    return v_t1 * g_ft + v_t2 * tfrac
end

@inline function interp3d_dev(tape, x::Float32, y::Float32, tfrac::Float32, nx::Int, ny::Int)
    i, fx = locate_int_dev(x, nx)
    j, fy = locate_int_dev(y, ny)
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

# Inline version that returns h_lo, h_hi for a particular sigma at a given (x, y, t)
# — used for vg_sigma settling without a full profile rebuild.
@inline function hlevel_pair_dev(hlevel, x::Float32, y::Float32, σ::Float32, tfrac::Float32,
                                  z_grid, nx::Int, ny::Int)
    i, fx = locate_int_dev(x, nx)
    j, fy = locate_int_dev(y, ny)
    k, _ = locate_dev(z_grid, σ)
    g_fx = 1.0f0 - fx
    g_fy = 1.0f0 - fy
    g_ft = 1.0f0 - tfrac

    @inbounds begin
        # k
        a00 = hlevel[i,   j,   k,   1] * g_fx + hlevel[i+1, j,   k,   1] * fx
        a10 = hlevel[i,   j+1, k,   1] * g_fx + hlevel[i+1, j+1, k,   1] * fx
        c00 = hlevel[i,   j,   k,   2] * g_fx + hlevel[i+1, j,   k,   2] * fx
        c10 = hlevel[i,   j+1, k,   2] * g_fx + hlevel[i+1, j+1, k,   2] * fx
        h_up_t1 = a00 * g_fy + a10 * fy
        h_up_t2 = c00 * g_fy + c10 * fy
        h_up = h_up_t1 * g_ft + h_up_t2 * tfrac

        # k+1
        b00 = hlevel[i,   j,   k+1, 1] * g_fx + hlevel[i+1, j,   k+1, 1] * fx
        b10 = hlevel[i,   j+1, k+1, 1] * g_fx + hlevel[i+1, j+1, k+1, 1] * fx
        d00 = hlevel[i,   j,   k+1, 2] * g_fx + hlevel[i+1, j,   k+1, 2] * fx
        d10 = hlevel[i,   j+1, k+1, 2] * g_fx + hlevel[i+1, j+1, k+1, 2] * fx
        h_dn_t1 = b00 * g_fy + b10 * fy
        h_dn_t2 = d00 * g_fy + d10 * fy
        h_dn = h_dn_t1 * g_ft + h_dn_t2 * tfrac
    end
    return k, h_up, h_dn
end

# Hanna NEUTRAL (matches host_shadow_v2.jl::hanna_neutral_inline)
@inline function hanna_neutral_dev(z::Float32, ust::Float32,
                                   sigma_h::Float32, sigma_w::Float32, tl_scale::Float32)
    f_cor = 1.0f-4
    tlu_min = 10.0f0
    tlv_min = 10.0f0
    tlw_min = 30.0f0

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

    return (sigu * sigma_h, sigv * sigma_h, sigw * sigma_w,
            dsigwdz * sigma_w, tlu, tlv, tlw)
end

@inline function ou_step_dev(u_old::Float32, sigma::Float32, tl::Float32, dt::Float32, rnd::Float32)
    dt_over_tl = dt / tl
    if dt_over_tl < 0.5f0
        return muladd(sigma * sqrt(2.0f0 * dt_over_tl), rnd, (1.0f0 - dt_over_tl) * u_old)
    else
        r = exp(-dt_over_tl)
        return muladd(sigma * sqrt(1.0f0 - r * r), rnd, r * u_old)
    end
end

# Per-particle xorshift RNG → Float32 normal (Box-Muller) — keeps a UInt64
# state in registers; one randn call advances the state once.
@inline function xorshift64_dev(state::UInt64)
    x = state
    x = x ⊻ (x << 13)
    x = x ⊻ (x >> 7)
    x = x ⊻ (x << 17)
    return x
end

# Convert a UInt64 to a Float32 in (0, 1] (drop top bit for safety).
@inline function rand01_dev(state::UInt64)
    state = xorshift64_dev(state)
    # Take 24 high bits → Float32 mantissa
    u = (state >> 40) & 0x00FFFFFF
    return state, (Float32(u) + 1.0f0) * Float32(1.0 / 16777217.0)  # in (0, 1)
end

@inline function randn_dev(state::UInt64)
    state, u1 = rand01_dev(state)
    state, u2 = rand01_dev(state)
    r = sqrt(-2.0f0 * log(u1))
    θ = 6.2831855f0 * u2
    return state, r * cos(θ)
end

# Inline single-particle dry deposition write (atomic into dep_grid)
@inline function bin_dep_dev!(dep_grid, lon_grid, lat_grid, nx_obs::Int, ny_obs::Int,
                              x_dom::Float32, y_dom::Float32, mass::Float32,
                              lon_min::Float32, lon_max::Float32,
                              lat_min::Float32, lat_max::Float32,
                              nx_dom::Int, ny_dom::Int)
    lon = lon_min + (x_dom - 1.0f0) * (lon_max - lon_min) / Float32(nx_dom - 1)
    lat = lat_min + (y_dom - 1.0f0) * (lat_max - lat_min) / Float32(ny_dom - 1)
    if lon > 180.0f0
        lon -= 360.0f0
    end
    # searchsortedlast on a sorted Float32 grid (small nx_obs/ny_obs → linear)
    i = 0
    @inbounds for k in 1:nx_obs
        if lon_grid[k] <= lon
            i = k
        else
            break
        end
    end
    j = 0
    @inbounds for k in 1:ny_obs
        if lat_grid[k] <= lat
            j = k
        else
            break
        end
    end
    if 1 <= i <= nx_obs && 1 <= j <= ny_obs
        @inbounds CUDA.@atomic dep_grid[i, j] += mass
    end
    return nothing
end

# ============================================================================
# Kernel — process one met window (n_substeps × dt) for all particles
# ============================================================================
function transport_window_kernel!(
    xs, ys, σs, u_turbs, v_turbs, w_turbs, masses, active,
    grav_v, diameter,
    u_tape, v_tape, w_tape, t_tape, ps_tape, hbl_tape, hlevel_tape,
    z_grid,
    dep_grid, lon_grid, lat_grid,
    rng_states,
    # Geometry (boxed scalars)
    nx_met::Int32, ny_met::Int32, nk::Int32,
    nx_dom::Int32, ny_dom::Int32,
    nx_obs::Int32, ny_obs::Int32,
    grid_scale_x::Float32, grid_scale_y::Float32,
    map_ratio_x::Float32, map_ratio_y::Float32, z_max_m::Float32,
    lon_min::Float32, lon_max::Float32, lat_min::Float32, lat_max::Float32,
    # Param-derived constants
    sigma_h::Float32, sigma_w::Float32, tl_scale::Float32,
    omega_scale::Float32, drag_coef::Float32, mixing_height_scale::Float32,
    simple_dep_velocity::Float32, h_surface_m::Float32,
    # Time stepping
    dt::Float32, dt_sub::Float32, ifine::Int32, n_substeps::Int32,
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    n = length(xs)
    i > n && return

    @inbounds begin
        if !active[i]
            return
        end

        x_dom = xs[i]
        y_dom = ys[i]
        σ_p = clamp(σs[i], 0.0f0, 1.0f0)
        u_t = u_turbs[i]
        v_t = v_turbs[i]
        w_t = w_turbs[i]
        mass = masses[i]
        vg_ms = grav_v[i]
        diam_um = diameter[i]
        rng = rng_states[i]
    end

    # n_substeps × {1 main step}
    @inbounds for sub_idx in Int32(1):n_substeps
        local_time = Float32(sub_idx - Int32(1)) * dt
        t_eval_f = local_time / 3600.0f0
        t_end_f = (local_time + dt) / 3600.0f0

        # met coordinates (lat_reversed = false)
        x_met = (x_dom - 1.0f0) * grid_scale_x + 1.0f0
        y_met = (y_dom - 1.0f0) * grid_scale_y + 1.0f0

        # ---------- Dry deposition at start position ----------
        if σ_p > 0.996f0
            vd_simple = simple_dep_velocity + vg_ms
            k_dep = vd_simple / h_surface_m
            decay_factor = exp(-k_dep * dt)

            if mass > 0.0f0
                new_mass = mass * decay_factor
                deposited = mass - new_mass
                mass = new_mass
                bin_dep_dev!(dep_grid, lon_grid, lat_grid, Int(nx_obs), Int(ny_obs),
                             x_dom, y_dom, deposited,
                             lon_min, lon_max, lat_min, lat_max,
                             Int(nx_dom), Int(ny_dom))
            end

            if σ_p >= 0.999f0 && diam_um >= 20.0f0
                if mass > 0.0f0
                    bin_dep_dev!(dep_grid, lon_grid, lat_grid, Int(nx_obs), Int(ny_obs),
                                 x_dom, y_dom, mass,
                                 lon_min, lon_max, lat_min, lat_max,
                                 Int(nx_dom), Int(ny_dom))
                end
                @inbounds masses[i] = 0.0f0
                @inbounds active[i] = false
                @inbounds rng_states[i] = rng
                return
            end

            if mass < 1.0f-10
                @inbounds masses[i] = mass
                @inbounds active[i] = false
                @inbounds rng_states[i] = rng
                return
            end
        end

        # ---------- Heun stage 1 ----------
        u1_w = interp4d_dev(u_tape, x_met, y_met, σ_p, t_eval_f, z_grid, Int(nx_met), Int(ny_met))
        v1_w = interp4d_dev(v_tape, x_met, y_met, σ_p, t_eval_f, z_grid, Int(nx_met), Int(ny_met))
        w1_w = interp4d_dev(w_tape, x_met, y_met, σ_p, t_eval_f, z_grid, Int(nx_met), Int(ny_met)) * omega_scale

        vg_sigma1 = if σ_p > 0.996f0
            0.0f0
        else
            σ_clamp = clamp(σ_p, z_grid[1] + Float32(eps(Float32)),
                            z_grid[Int(nk)] - Float32(eps(Float32)))
            kg, h_up, h_dn = hlevel_pair_dev(hlevel_tape, x_met, y_met, σ_clamp, t_eval_f,
                                              z_grid, Int(nx_met), Int(ny_met))
            σ_up = z_grid[kg]
            σ_dn = z_grid[kg + 1]
            dsig = σ_dn - σ_up
            dz = h_up - h_dn
            if abs(dz) < Float32(eps(Float32))
                vg_ms / z_max_m
            else
                vg_ms * dsig / dz
            end
        end

        lat_frac = (y_met - 1.0f0) / Float32(max(Int(ny_met) - 1, 1))
        lat_deg = lat_min + lat_frac * (lat_max - lat_min)
        clat = max(cos(lat_deg * 0.017453292f0), 0.01745f0)
        xm_factor = 1.0f0 / clat

        du1_x = u1_w * map_ratio_x * xm_factor
        du1_y = v1_w * map_ratio_y
        du1_z = w1_w + vg_sigma1

        x_met_pred = x_met + du1_x * dt
        y_met_pred = y_met + du1_y * dt
        σ_pred = clamp(σ_p + du1_z * dt, 0.0f0, 1.0f0)

        # ---------- Heun stage 2 ----------
        u2_w = interp4d_dev(u_tape, x_met_pred, y_met_pred, σ_pred, t_end_f, z_grid, Int(nx_met), Int(ny_met))
        v2_w = interp4d_dev(v_tape, x_met_pred, y_met_pred, σ_pred, t_end_f, z_grid, Int(nx_met), Int(ny_met))
        w2_w = interp4d_dev(w_tape, x_met_pred, y_met_pred, σ_pred, t_end_f, z_grid, Int(nx_met), Int(ny_met)) * omega_scale

        vg_sigma2 = if σ_pred > 0.996f0
            0.0f0
        else
            σ_clamp = clamp(σ_pred, z_grid[1] + Float32(eps(Float32)),
                            z_grid[Int(nk)] - Float32(eps(Float32)))
            kg, h_up, h_dn = hlevel_pair_dev(hlevel_tape, x_met_pred, y_met_pred, σ_clamp, t_end_f,
                                              z_grid, Int(nx_met), Int(ny_met))
            σ_up = z_grid[kg]
            σ_dn = z_grid[kg + 1]
            dsig = σ_dn - σ_up
            dz = h_up - h_dn
            if abs(dz) < Float32(eps(Float32))
                vg_ms / z_max_m
            else
                vg_ms * dsig / dz
            end
        end

        lat_frac2 = (y_met_pred - 1.0f0) / Float32(max(Int(ny_met) - 1, 1))
        lat_deg2 = lat_min + lat_frac2 * (lat_max - lat_min)
        clat2 = max(cos(lat_deg2 * 0.017453292f0), 0.01745f0)
        xm2 = 1.0f0 / clat2

        du2_x = u2_w * map_ratio_x * xm2
        du2_y = v2_w * map_ratio_y
        du2_z = w2_w + vg_sigma2

        half_dt = dt * 0.5f0
        x_met_final = x_met + (du1_x + du2_x) * half_dt
        y_met_final = y_met + (du1_y + du2_y) * half_dt
        σ_after_adv = clamp(σ_p + (du1_z + du2_z) * half_dt, 0.0f0, 1.0f0)

        x_dom_final = (x_met_final - 1.0f0) / grid_scale_x + 1.0f0
        y_dom_final = (y_met_final - 1.0f0) / grid_scale_y + 1.0f0

        # ---------- Hanna turbulence ----------
        h_dynamic = interp3d_dev(hbl_tape, x_met_final, y_met_final, t_eval_f,
                                  Int(nx_met), Int(ny_met))
        h_pbl = max(if h_dynamic > 0.0f0
                        h_dynamic
                    else
                        1000.0f0 * mixing_height_scale
                    end, 50.0f0)

        u_surf = interp4d_dev(u_tape, x_met_final, y_met_final, 1.0f0, t_eval_f,
                               z_grid, Int(nx_met), Int(ny_met))
        v_surf = interp4d_dev(v_tape, x_met_final, y_met_final, 1.0f0, t_eval_f,
                               z_grid, Int(nx_met), Int(ny_met))
        u_mag = sqrt(u_surf * u_surf + v_surf * v_surf)
        ust = max(drag_coef * u_mag, 0.01f0)

        # post-advection sigma → height — use a per-particle profile rebuild
        # (one trilinear lookup per level, but inlined as height_from_sigma_dev)
        # For simplicity here we build only h_up/h_dn around σ_after_adv:
        kg2, h_up2, h_dn2 = hlevel_pair_dev(hlevel_tape, x_met_final, y_met_final,
                                             σ_after_adv, t_end_f, z_grid,
                                             Int(nx_met), Int(ny_met))
        σ_up2 = z_grid[kg2]
        σ_dn2 = z_grid[kg2 + 1]
        ratio = (σ_after_adv - σ_up2) / (σ_dn2 - σ_up2)
        z_m_current = h_up2 + ratio * (h_dn2 - h_up2)
        z_m_current = clamp(z_m_current, 0.0f0, z_max_m)

        if z_m_current < 0.0f0
            if mass > 0.0f0
                bin_dep_dev!(dep_grid, lon_grid, lat_grid, Int(nx_obs), Int(ny_obs),
                             x_dom_final, y_dom_final, mass,
                             lon_min, lon_max, lat_min, lat_max,
                             Int(nx_dom), Int(ny_dom))
            end
            @inbounds masses[i] = 0.0f0
            @inbounds active[i] = false
            @inbounds rng_states[i] = rng
            return
        end

        sigu, sigv, sigw, dsigwdz, tlu, tlv, tlw =
            hanna_neutral_dev(z_m_current, ust, sigma_h, sigma_w, tl_scale)

        rng, rnd_u = randn_dev(rng)
        rng, rnd_v = randn_dev(rng)
        u_t = ou_step_dev(u_t, sigu, tlu, dt, rnd_u)
        v_t = ou_step_dev(v_t, sigv, tlv, dt, rnd_v)

        z_sigma_dep = clamp(σ_after_adv, 0.0f0, 1.0f0)
        at_ground = z_sigma_dep >= 0.9999f0
        z_sigma_current = z_sigma_dep

        if at_ground
            w_t = 0.0f0
        else
            T_k = interp4d_dev(t_tape, x_met, y_met, z_sigma_current, t_eval_f,
                                z_grid, Int(nx_met), Int(ny_met))
            ps_pa = interp3d_dev(ps_tape, x_met, y_met, t_eval_f,
                                  Int(nx_met), Int(ny_met)) * 100.0f0
            P_pa = ps_pa * z_sigma_current
            R_air = 287.0f0
            g_grav = 9.81f0
            rhoa = P_pa / (R_air * T_k)
            rhograd = -rhoa * g_grav / (R_air * T_k)

            for i_sub in Int32(1):ifine
                sigu_s, sigv_s, sigw_s, dsigwdz_s, tlu_s, tlv_s, tlw_s = if i_sub == Int32(1)
                    (sigu, sigv, sigw, dsigwdz, tlu, tlv, tlw)
                else
                    hanna_neutral_dev(z_m_current, ust, sigma_h, sigma_w, tl_scale)
                end

                rng, rnd_w = randn_dev(rng)
                w_new = ou_step_dev(w_t, sigw_s, tlw_s, dt_sub, rnd_w)
                w_t = w_new

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
                z_m_new = clamp(z_m_new, 0.0f0, z_max_m)

                # height → sigma using the same h_up/h_dn pair around z_m_new
                # via direct linear interp on the local 2-level segment around the
                # current sigma. For accuracy we re-locate h_up/h_dn using the new
                # sigma estimate. Fast approx: invert linearly.
                kg3, h_up3, h_dn3 = hlevel_pair_dev(hlevel_tape, x_met_final, y_met_final,
                                                     z_sigma_current, t_end_f, z_grid,
                                                     Int(nx_met), Int(ny_met))
                σ_up3 = z_grid[kg3]
                σ_dn3 = z_grid[kg3 + 1]
                # linear: z_m = h_up3 + r * (h_dn3 - h_up3), invert for r
                dz3 = h_dn3 - h_up3
                if abs(dz3) > Float32(eps(Float32))
                    r3 = (z_m_new - h_up3) / dz3
                    z_sigma_new = σ_up3 + r3 * (σ_dn3 - σ_up3)
                else
                    z_sigma_new = z_sigma_current
                end
                z_sigma_current = clamp(z_sigma_new, 0.0f0, 1.0f0)

                # update z_m_current from new sigma
                kg4, h_up4, h_dn4 = hlevel_pair_dev(hlevel_tape, x_met_final, y_met_final,
                                                     z_sigma_current, t_end_f, z_grid,
                                                     Int(nx_met), Int(ny_met))
                σ_up4 = z_grid[kg4]
                σ_dn4 = z_grid[kg4 + 1]
                ratio4 = (z_sigma_current - σ_up4) / (σ_dn4 - σ_up4)
                z_m_current = h_up4 + ratio4 * (h_dn4 - h_up4)
                z_m_current = clamp(z_m_current, 0.0f0, z_max_m)

                if i_sub < ifine
                    T_k = interp4d_dev(t_tape, x_met, y_met, z_sigma_current, t_eval_f,
                                        z_grid, Int(nx_met), Int(ny_met))
                    ps_pa = interp3d_dev(ps_tape, x_met, y_met, t_eval_f,
                                          Int(nx_met), Int(ny_met)) * 100.0f0
                    P_pa = ps_pa * z_sigma_current
                    rhoa = P_pa / (R_air * T_k)
                    rhograd = -rhoa * g_grav / (R_air * T_k)
                end
            end
        end

        # Apply turbulent horizontal displacements
        x_dom_final += u_t * dt * map_ratio_x
        y_dom_final += v_t * dt * map_ratio_y
        z_sigma_final = z_sigma_current
        if z_sigma_final >= 0.996f0
            w_t = 0.0f0
        end

        if !(1.0f0 <= x_dom_final <= Float32(nx_dom)) ||
           !(1.0f0 <= y_dom_final <= Float32(ny_dom))
            @inbounds active[i] = false
            @inbounds rng_states[i] = rng
            return
        end

        x_dom = x_dom_final
        y_dom = y_dom_final
        σ_p = clamp(z_sigma_final, 0.0f0, 1.0f0)
    end

    @inbounds begin
        xs[i] = x_dom
        ys[i] = y_dom
        σs[i] = σ_p
        u_turbs[i] = u_t
        v_turbs[i] = v_t
        w_turbs[i] = w_t
        masses[i] = mass
        rng_states[i] = rng
    end
    return nothing
end

# ============================================================================
# Host-side launcher
# ============================================================================
"""
    run_gpu_shadow(params::Vector{Float64}, gen_seed::UInt64; rng_seed=gen_seed)
        → (deposition_grid::Matrix{Float32}, hourly_dep::Array{Float32,3}, n_active::Int)

Mirror of `run_host_shadow` on the GPU. Single-candidate, one thread per particle.
Twelve kernel launches per 12-hour Nancy simulation (one per met window) with
`dep_grid → hourly_dep[:,:,h]` snapshots between launches.
"""
function run_gpu_shadow(params::Vector{Float64}, gen_seed::UInt64;
                        rng_seed::UInt64 = gen_seed,
                        windows::Union{Nothing,Vector{GpuMetWindow}} = nothing)
    sigma_w_scale     = Float32(params[8])
    sigma_h_scale     = Float32(params[9])
    tl_scale          = Float32(params[11])
    vd_scale          = Float32(params[12])
    omega_scale       = Float32(params[14])
    mixing_height_scale  = Float32(params[15])
    surface_height_scale = Float32(params[17])
    drag_coef         = 0.05f0

    simple_dep_velocity = 0.002f0 * vd_scale
    h_surface_m         = 30.0f0 * surface_height_scale

    # Particles (host)
    pts = generate_shadow_particles(params, gen_seed)
    if pts.n == 0
        nx_obs, ny_obs = length(LON_GRID), length(LAT_GRID)
        return zeros(Float32, nx_obs, ny_obs), zeros(Float32, nx_obs, ny_obs, 12), 0
    end

    # Geometry from a sample tape (host side, used for grid_scale + map_ratio)
    sample_mf = MET_CACHE[(CACHE_START_FILE, 1)]
    sample_tape = build_wind_tape(sample_mf, 0.0, 3600.0)
    geom = build_shadow_geom(DOMAIN, sample_tape)

    # Met windows
    win_list = windows === nothing ? load_nancy_gpu_windows() : windows
    @assert length(win_list) >= 12 "Need at least 12 met windows for Nancy 12-hour run; got $(length(win_list))"

    # Output grid (host + device)
    nx_obs, ny_obs = length(LON_GRID), length(LAT_GRID)
    dep_dev = CUDA.zeros(Float32, nx_obs, ny_obs)
    hourly_host = zeros(Float32, nx_obs, ny_obs, 12)

    lon_dev = CuArray(Float32.(LON_GRID))
    lat_dev = CuArray(Float32.(LAT_GRID))

    # Particles → device
    xs_dev = CuArray(pts.xs)
    ys_dev = CuArray(pts.ys)
    σs_dev = CuArray(pts.σs)
    u_turbs_dev = CuArray(pts.u_turbs)
    v_turbs_dev = CuArray(pts.v_turbs)
    w_turbs_dev = CuArray(pts.w_turbs)
    masses_dev  = CuArray(pts.masses)
    grav_dev    = CuArray(pts.grav)
    diam_dev    = CuArray(pts.diameter)
    active_dev  = CuArray(pts.active)

    # RNG state — one UInt64 per particle, seeded from rng_seed
    rng_states = Vector{UInt64}(undef, pts.n)
    seed_rng = Random.MersenneTwister(rng_seed)
    for i in 1:pts.n
        # Combine rng_seed + particle id; xorshift can't accept zero
        s = rand(seed_rng, UInt64) | UInt64(0x1)
        rng_states[i] = s
    end
    rng_dev = CuArray(rng_states)

    dt = 300.0f0
    ifine = Int32(5)
    dt_sub = dt / Float32(ifine)
    n_substeps_per_window = Int32(12)   # 12 × 300s = 1 hour

    threads = 256
    blocks = cld(pts.n, threads)

    # Twelve launches — one per met window
    for h in 1:12
        win = win_list[h]
        @cuda threads=threads blocks=blocks transport_window_kernel!(
            xs_dev, ys_dev, σs_dev, u_turbs_dev, v_turbs_dev, w_turbs_dev,
            masses_dev, active_dev, grav_dev, diam_dev,
            win.u, win.v, win.w, win.t, win.ps, win.hbl, win.hlevel,
            win.z_grid,
            dep_dev, lon_dev, lat_dev,
            rng_dev,
            Int32(geom.nx_met), Int32(geom.ny_met), Int32(geom.nk),
            Int32(geom.nx_dom), Int32(geom.ny_dom),
            Int32(nx_obs), Int32(ny_obs),
            geom.grid_scale_x, geom.grid_scale_y,
            geom.map_ratio_x, geom.map_ratio_y, geom.z_max_m,
            geom.lon_min, geom.lon_max, geom.lat_min, geom.lat_max,
            sigma_h_scale, sigma_w_scale, tl_scale,
            omega_scale, drag_coef, mixing_height_scale,
            simple_dep_velocity, h_surface_m,
            dt, dt_sub, ifine, n_substeps_per_window,
        )
        CUDA.synchronize()
        # Snapshot to host (full Array copy, then assign view)
        snap = Array(dep_dev)
        @inbounds hourly_host[:, :, h] .= snap
    end

    final_grid = Array(dep_dev)
    n_active = Int(sum(Array(active_dev)))
    return final_grid, hourly_host, n_active
end
