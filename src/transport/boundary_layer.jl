# Boundary layer diagnostics
#
# Computes sigma/eta coordinate of boundary layer top and mixing height
# by evaluating Richardson numbers in hybrid vertical coordinates.

# Hybrid coefficient unit helper — coefficients are stored in hPa (converted in met_formats)
_hybrid_coeff_to_hpa(::Type{ERA5Format}, a::T) where T = a

# Height calculation sign helpers
# ERA5 data is reversed during loading → surface-to-top (k=1 surface, k=nk TOA)
#   → pih[k] > pih[k+1], use +1: zh[k+1] = zh[k] + T*(pih[k]-pih[k+1])/g
_height_sign_initial(::Type{ERA5Format}) = 1   # +1 for surface-to-top integration
_height_sign_loop(::Type{ERA5Format}) = 1      # +1 for surface-to-top integration

# Temperature → potential temperature conversion for boundary layer calculation
# ERA5 stores absolute temperature T → needs conversion to potential temperature θ
# Formula: θ = T * (1000/p)^(R/cp) = T * t2thetafac(p)
_t2theta(::Type{ERA5Format}, T_val::TT, p_hpa::TT, rcp::TT) where TT = T_val * (TT(1000.0) / p_hpa)^rcp

"""
    compute_boundary_layer!(::Type{F}, fields::MeteoFields{T}; ...) where {F<:MetFormat, T}

Compute boundary layer with met-format-specific unit handling.
ERA5 hybrid coefficients are in hPa for consistent pressure calculations.
"""
function compute_boundary_layer!(::Type{F}, fields::MeteoFields{T};
                                 time_level::Int=2,
                                 ric_factor::Real=1.8,
                                 psurf_ref::Real=1000.0) where {F<:MetFormat, T}
    _compute_boundary_layer_impl!(F, fields;
                                   time_level=time_level,
                                   ric_factor=ric_factor,
                                   psurf_ref=psurf_ref)
end

"""
    compute_boundary_layer!(fields::MeteoFields{T}; ...)

Default version for backwards compatibility (uses ERA5 behavior).
"""
function compute_boundary_layer!(fields::MeteoFields{T};
                                 time_level::Int=2,
                                 ric_factor::Real=1.8,
                                 psurf_ref::Real=1000.0) where T
    compute_boundary_layer!(ERA5Format, fields; time_level, ric_factor, psurf_ref)
end

"""
    _compute_boundary_layer_impl!(fmt_type, fields; ...)

Internal implementation with format-specific unit and sign handling.
Vertical indexing: surface-to-top (k=1 surface, k=nk TOA) — ERA5 data is reversed during loading.
Height calculation sign is handled via _height_sign_initial(F) and _height_sign_loop(F).
"""
function _compute_boundary_layer_impl!(::Type{F}, fields::MeteoFields{T};
                                        time_level::Int=2,
                                        ric_factor::Real=1.8,
                                        psurf_ref::Real=1000.0) where {F<:MetFormat, T}
    nx, ny, nk = fields.nx, fields.ny, fields.nk

    cp = T(1004.0)
    g = T(G_GRAVITY_M_S2)
    ginv = inv(g)
    r = T(R_SPECIFIC_J_KG_K)
    rcp = r / cp
    psurf_ref = T(psurf_ref)
    ric_factor = T(ric_factor)

    # Exner function
    exner(p::T) = cp * (p / T(1000.0))^rcp

    # Determine allowable pressure bounds for BL search
    pbl_top = T(600.0)
    pbl_bot = T(975.0)

    p = _hybrid_coeff_to_hpa(F, fields.ahalf[max(nk - 1, 1)]) + fields.bhalf[max(nk - 1, 1)] * psurf_ref
    if pbl_top < p
        pbl_top = p
    end
    p = _hybrid_coeff_to_hpa(F, fields.ahalf[min(2, nk)]) + fields.bhalf[min(2, nk)] * psurf_ref
    if pbl_bot > p
        pbl_bot = p
    end

    kbl_top = 2
    kbl_bot = 2
    nkk = min(nk - 2, nk)
    for k in 2:nkk
        p = _hybrid_coeff_to_hpa(F, fields.ahalf[k]) + fields.bhalf[k] * psurf_ref
        if p > pbl_top
            kbl_top = k + 1
        end
        if p >= pbl_bot
            kbl_bot = k
        end
    end
    kbl_top = clamp(kbl_top, 2, nk)
    kbl_bot = clamp(kbl_bot, 2, max(nk - 1, 2))

    # Convert pressure bounds to sigma/eta bounds using hybrid coefficients
    function interp_vhalf(k_low::Int, target_p::T)
        k_high = clamp(k_low + 1, 1, nk)
        p2 = _hybrid_coeff_to_hpa(F, fields.ahalf[k_high]) + fields.bhalf[k_high] * psurf_ref
        p1 = _hybrid_coeff_to_hpa(F, fields.ahalf[k_low]) + fields.bhalf[k_low] * psurf_ref
        denom = p2 - p1
        if abs(denom) < eps(T)
            return fields.vhalf[k_high]
        end
        return fields.vhalf[k_low] +
               (fields.vhalf[k_high] - fields.vhalf[k_low]) * (target_p - p1) / denom
    end

    vbl_top = interp_vhalf(kbl_top - 1, pbl_top)
    vbl_bot = interp_vhalf(kbl_bot, pbl_bot)

    # Workspace arrays (nk+1 to allow k+1 access)
    pih = zeros(T, nk + 1)
    pif = zeros(T, nk + 1)
    p_level = zeros(T, nk + 1)  # Pressure at full levels (hPa) for T→θ conversion
    zh  = zeros(T, nk + 1)
    zf  = zeros(T, nk + 1)
    thh = zeros(T, nk + 1)

    u_field = time_level == 1 ? fields.u1 : fields.u2
    v_field = time_level == 1 ? fields.v1 : fields.v2
    t_field = time_level == 1 ? fields.t1 : fields.t2
    ps_field = time_level == 1 ? fields.ps1 : fields.ps2
    bl_storage = time_level == 1 ? fields.bl1 : fields.bl2
    hbl_storage = time_level == 1 ? fields.hbl1 : fields.hbl2

    for j in 1:ny, i in 1:nx
        ps = ps_field[i, j]
        if ps <= 0
            # No valid surface pressure; keep previous values
            continue
        end

        u_surface = u_field[i, j, 1]
        v_surface = v_field[i, j, 1]
        u_field[i, j, 1] = zero(T)
        v_field[i, j, 1] = zero(T)

        # Initialise exner function and pressure at the first two levels
        for k in 1:2
            pih[k] = exner(_hybrid_coeff_to_hpa(F, fields.ahalf[k]) + fields.bhalf[k] * ps)
            p_level[k] = _hybrid_coeff_to_hpa(F, fields.alevel[k]) + fields.blevel[k] * ps
            pif[k] = exner(p_level[k])
        end

        zh[1] = zero(T)
        zf[1] = zero(T)
        # Format-specific sign: +1 for surface-to-top integration
        # Convert T→θ (reference does this via t2thetafac before BL computation)
        theta_2 = _t2theta(F, t_field[i, j, 2], p_level[2], rcp)
        zh[2] = zh[1] + _height_sign_initial(F) * theta_2 * (pih[1] - pih[2]) * ginv
        denom = pih[1] - pih[2]
        if abs(denom) > eps(T)
            zf[2] = zh[1] + (zh[2] - zh[1]) * (pih[1] - pif[2]) / denom
        else
            zf[2] = zh[2]
        end

        ktop = 0
        k = 1
        ri = zero(T)
        ric = zero(T)
        riu = zero(T)
        ricu = zero(T)

        while ((ktop == 0 || k < kbl_bot) && k < kbl_top)
            k += 1
            if k + 1 > nk
                break
            end

            pih[k + 1] = exner(_hybrid_coeff_to_hpa(F, fields.ahalf[k + 1]) + fields.bhalf[k + 1] * ps)
            p_level[k + 1] = _hybrid_coeff_to_hpa(F, fields.alevel[k + 1]) + fields.blevel[k + 1] * ps
            pif[k + 1] = exner(p_level[k + 1])

            # Convert T→θ (reference does this via t2thetafac before BL computation)
            theta_k = _t2theta(F, t_field[i, j, k], p_level[k], rcp)
            theta_kp1 = _t2theta(F, t_field[i, j, k + 1], p_level[k + 1], rcp)

            denom_full = pif[k] - pif[k + 1]
            thh[k] = theta_k
            if abs(denom_full) > eps(T)
                thh[k] += (theta_kp1 - theta_k) * (pif[k] - pih[k]) / denom_full
            end

            # Format-specific sign for height calculation (see _height_sign_loop helper)
            # Surface-to-top: pih[k] > pih[k+1], use +1 → heights increase with k
            zh[k + 1] = zh[k] + _height_sign_loop(F) * theta_kp1 * (pih[k] - pih[k + 1]) * ginv
            denom_half = pih[k] - pih[k + 1]
            if abs(denom_half) > eps(T)
                zf[k + 1] = zh[k] + (zh[k + 1] - zh[k]) *
                            (pih[k] - pif[k + 1]) / denom_half
            else
                zf[k + 1] = zh[k + 1]
            end

            if ktop == 0
                dz = zf[k + 1] - zf[k]
                dv2_min = T(1e-5) * dz * dz
                du = u_field[i, j, k + 1] - u_field[i, j, k]
                dv = v_field[i, j, k + 1] - v_field[i, j, k]
                dv2 = du * du + dv * dv
                if dv2 < dv2_min
                    dv2 = dv2_min
                end

                # Use potential temperature difference (dθ) for Richardson number
                dth = theta_kp1 - theta_k
                denom_ri = thh[k] * pih[k] * dv2
                ri_val = abs(denom_ri) > eps(T) ? cp * g * dth * dz / denom_ri : zero(T)
                # Use abs(dz) to avoid fractional power of negative number
                ric_val = T(0.115) * (abs(dz) * T(100.0))^T(0.175) * ric_factor


                if ri_val > ric_val
                    ktop = k
                else
                    riu = ri_val
                    ricu = ric_val
                end

                ri = ri_val
                ric = ric_val
            end
        end

        k_local = ktop
        if k_local == 0
            k_local = kbl_top
        end

        vbl = T(0)
        if fields.vhalf[k_local] >= vbl_bot
            vbl = vbl_bot
            k_local = kbl_bot
        elseif k_local == kbl_top && ri <= ric
            vbl = vbl_top
            k_local = kbl_top - 1
        else
            k_local = max(k_local - 1, 1)
            dri = ric - ri
            driu = ricu - riu
            denom_v = driu - dri
            frac = abs(denom_v) > eps(T) ? driu / denom_v : zero(T)
            vbl = fields.vhalf[k_local] +
                  (fields.vhalf[k_local + 1] - fields.vhalf[k_local]) * frac
            vbl = clamp(vbl, vbl_top, vbl_bot)
        end

        denom_vhalf = fields.vhalf[k_local] - fields.vhalf[k_local + 1]
        hbl = zh[k_local]
        if abs(denom_vhalf) > eps(T)
            hbl += (zh[k_local + 1] - zh[k_local]) *
                   (fields.vhalf[k_local] - vbl) / denom_vhalf
        end

        if !isfinite(hbl)
            hbl = zh[k_local]
        end
        if !isfinite(vbl)
            vbl = vbl_top
        end

        bl_storage[i, j] = vbl
        hbl_storage[i, j] = hbl

        u_field[i, j, 1] = u_surface
        v_field[i, j, 1] = v_surface
    end

    return nothing
end

export compute_boundary_layer!
