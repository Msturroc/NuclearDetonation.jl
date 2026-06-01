# Float32 port of NuclearDetonation.jl Hanna CBL turbulence parameterisation
# ===========================================================================
# Source: /home/marc/NuclearDetonation.jl/src/transport/turbulence_hanna.jl
#
# Provides two parallel implementations:
#   - `hanna_dev`:  CUDA device function, called from inside kernels
#   - `hanna_f32`:  pure-Float32 host function, used by the CPU shadow
#                   reference and the unit test
#
# Both return a 6-tuple `(sigu, sigv, sigw, tlu, tlv, tlw)` in Float32 with
# identical arithmetic order so that the GPU and shadow are bitwise equal.
#
# The Float64 NuclearDetonation routine also returns `dsigwdz` and a stability
# label; the GPU port omits those because the OU update uses only σ and τ.

# Hanna scaling configuration — held as a plain Float32 struct so it can be
# passed to device kernels by value (`isbits`).
struct HannaCfgF32
    f_coriolis::Float32
    sigma_scale::Float32
    sigma_scale_vertical::Float32
    tl_scale::Float32
    tlu_min::Float32
    tlv_min::Float32
    tlw_min::Float32
end

# Defaults match HannaTurbulenceConfig{Float64}() when no scales are passed.
function default_hanna_cfg(; sigma_h_scale::Real = 1.0,
                             sigma_w_scale::Real = 1.0,
                             tl_scale::Real      = 1.0)
    HannaCfgF32(
        Float32(1.0e-4),                # f_coriolis
        Float32(sigma_h_scale),         # sigma_scale (horizontal)
        Float32(sigma_w_scale),         # sigma_scale_vertical
        Float32(tl_scale),              # tl_scale
        Float32(10.0),                  # tlu_min
        Float32(10.0),                  # tlv_min
        Float32(10.0),                  # tlw_min
    )
end

# ---------------------------------------------------------------------------
# Branch 1: NEUTRAL — corresponds to compute_neutral_turbulence
# ---------------------------------------------------------------------------
@inline function hanna_neutral_f32(z::Float32, h::Float32, ust::Float32, cfg::HannaCfgF32)
    ust = max(1.0f-4, ust)
    corr = z / ust

    sigu = 1.0f-2 + 2.0f0 * ust * exp(-3.0f0 * cfg.f_coriolis * corr)
    sigw = 1.3f0 * ust * exp(-2.0f0 * cfg.f_coriolis * corr)
    sigw = sigw + 1.0f-2
    sigv = sigw

    tl = 0.5f0 * z / sigw / (1.0f0 + 1.5f-3 * corr)
    tlu = max(cfg.tlu_min, tl) * cfg.tl_scale
    tlv = max(cfg.tlv_min, tl) * cfg.tl_scale
    tlw = max(cfg.tlw_min, tl) * cfg.tl_scale

    sh = cfg.sigma_scale
    sv = cfg.sigma_scale_vertical
    return (sigu*sh, sigv*sh, sigw*sv, tlu, tlv, tlw)
end

# ---------------------------------------------------------------------------
# Branch 2: UNSTABLE — compute_unstable_turbulence (CBL)
# ---------------------------------------------------------------------------
@inline function hanna_unstable_f32(z::Float32, h::Float32, L::Float32,
                                    ust::Float32, wst::Float32, zeta::Float32,
                                    cfg::HannaCfgF32)
    zeta = max(zeta, 0.0f0)

    sigu = 1.0f-2 + ust * (12.0f0 - 0.5f0 * h / L)^0.33333f0
    sigv = sigu

    sigw_arg = max(
        1.2f0 * wst*wst * (1.0f0 - 0.9f0 * zeta) * zeta^0.66666f0
        + (1.8f0 - 1.4f0 * zeta) * ust*ust,
        0.0f0,
    )
    sigw = sqrt(sigw_arg) + 1.0f-2

    tlu = 0.15f0 * h / sigu
    tlv = tlu
    absL = abs(L)
    tlw = if z < absL
        0.1f0 * z / (sigw * (0.55f0 - 0.38f0 * abs(z / L)))
    elseif zeta < 0.1f0
        0.59f0 * z / sigw
    else
        0.15f0 * h / sigw * (1.0f0 - exp(-5.0f0 * zeta))
    end

    tlu = max(cfg.tlu_min, tlu) * cfg.tl_scale
    tlv = max(cfg.tlv_min, tlv) * cfg.tl_scale
    tlw = max(cfg.tlw_min, tlw) * cfg.tl_scale

    sh = cfg.sigma_scale
    sv = cfg.sigma_scale_vertical
    return (sigu*sh, sigv*sh, sigw*sv, tlu, tlv, tlw)
end

# ---------------------------------------------------------------------------
# Branch 3: STABLE — compute_stable_turbulence
# ---------------------------------------------------------------------------
@inline function hanna_stable_f32(z::Float32, h::Float32, L::Float32,
                                  ust::Float32, zeta::Float32, cfg::HannaCfgF32)
    sigu = 1.0f-2 + 2.0f0 * ust * (1.0f0 - zeta)
    sigv = 1.0f-2 + 1.3f0 * ust * (1.0f0 - zeta)
    sigw = sigv

    tlu = 0.15f0 * h / sigu * sqrt(zeta)
    tlv = 0.467f0 * tlu
    tlw = 0.1f0 * h / sigw * zeta^0.8f0

    tlu = max(cfg.tlu_min, tlu) * cfg.tl_scale
    tlv = max(cfg.tlv_min, tlv) * cfg.tl_scale
    tlw = max(cfg.tlw_min, tlw) * cfg.tl_scale

    sh = cfg.sigma_scale
    sv = cfg.sigma_scale_vertical
    return (sigu*sh, sigv*sh, sigw*sv, tlu, tlv, tlw)
end

# ---------------------------------------------------------------------------
# Top-level dispatcher — host (Float32 twin) and device versions are identical.
# ---------------------------------------------------------------------------
@inline function hanna_f32(z::Float32, h::Float32, L::Float32,
                           ust::Float32, wst::Float32, cfg::HannaCfgF32)
    zeta = z / h
    stability_ratio = h / abs(L)
    if stability_ratio < 1.0f0
        return hanna_neutral_f32(z, h, ust, cfg)
    elseif L < 0.0f0
        return hanna_unstable_f32(z, h, L, ust, wst, zeta, cfg)
    else
        return hanna_stable_f32(z, h, L, ust, zeta, cfg)
    end
end

# Device-side alias — CUDA.jl will JIT this for GPU. Same function body but
# kept as a separate symbol in case the device version diverges later (e.g.
# uses `CUDA.fast_exp` for raw fast-math). For now both call the @inline
# helpers above which compile cleanly to PTX.
@inline function hanna_dev(z::Float32, h::Float32, L::Float32,
                           ust::Float32, wst::Float32, cfg::HannaCfgF32)
    return hanna_f32(z, h, L, ust, wst, cfg)
end

# ---------------------------------------------------------------------------
# Ornstein-Uhlenbeck step (Float32) — port of `ornstein_uhlenbeck_step`.
# Branchy on dt/tl < 0.5 to match the Float64 reference exactly.
# ---------------------------------------------------------------------------
@inline function ou_step_f32(u_old::Float32, sigma::Float32, tl::Float32,
                             dt::Float32, rnd::Float32)
    dt_over_tl = dt / tl
    if dt_over_tl < 0.5f0
        return muladd(sigma * sqrt(2.0f0 * dt_over_tl), rnd,
                      (1.0f0 - dt_over_tl) * u_old)
    else
        r = exp(-dt_over_tl)
        return muladd(sigma * sqrt(1.0f0 - r*r), rnd, r * u_old)
    end
end
