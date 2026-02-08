# Hanna (1982) Turbulence Parameterisation
# References:
# - Hanna, S.R. (1982): Applications in Air Pollution Modelling
# - Stohl et al. (2005): FLEXPART version 6.2

using SpecialFunctions: erf

export HannaTurbulenceConfig
export compute_hanna_parameters, apply_hanna_turbulence!
export HannaTurbulenceParameters
export flexpart_vertical_step
export apply_simple_convection, estimate_obukhov_length
export CBLParameters, compute_cbl_parameters, compute_cbl_langevin_terms

"""
    HannaTurbulenceConfig{T<:Real}

Configuration for Hanna (1982) turbulence parameterization with Ornstein-Uhlenbeck process.

# Fields
- `apply_turbulence::Bool` - Enable/disable turbulence (default: true)
- `tlu_min::T` - Minimum Lagrangian timescale for u (s, default: 10.0)
- `tlv_min::T` - Minimum Lagrangian timescale for v (s, default: 10.0)
- `tlw_min::T` - Minimum Lagrangian timescale for w (s, default: 30.0)
- `f_coriolis::T` - Coriolis parameter (1/s, default: 1e-4)
- `ifine::Int` - Number of substeps for vertical diffusion (default: 5)
- `use_cbl::Bool` - Use Convective Boundary Layer scheme for strong convection (default: true)
- `cbl_threshold::T` - Threshold -h/L for CBL activation (default: 5.0)
- `blfullmix::Bool` - Instant mixing in BL (false = gradual mixing, default: false)
- `use_simple_convection::Bool` - Use simple convective injection in unstable PBL (default: false)
- `convection_timescale::T` - Characteristic time to mix PBL fully (s, default: 1800.0)
- `drag_coefficient::T` - Drag coefficient for u* estimation: u* = Cd × U (default: 0.05)
- `use_dynamic_L::Bool` - Estimate L from temperature gradient (default: false)

# Example
```julia
config = HannaTurbulenceConfig(
    apply_turbulence = true,
    ifine = 5,
    use_simple_convection = true,  # Enable tropical convection mode
    convection_timescale = 900.0,  # Mix PBL every 15 mins in unstable conditions
    use_dynamic_L = true           # Estimate stability from met data
)
```
"""
@kwdef struct HannaTurbulenceConfig{T<:Real}
    apply_turbulence::Bool = true
    tlu_min::T = 10.0
    tlv_min::T = 10.0
    tlw_min::T = 30.0
    f_coriolis::T = 1.0e-4
    ifine::Int = 5
    use_cbl::Bool = true
    cbl_threshold::T = 5.0
    blfullmix::Bool = false
    # Turbulence intensity multiplier for HORIZONTAL sigma values (sigu, sigv).
    # Hanna (1982) gives σ ≈ 0.5 m/s → diffusivity D = σ²τ ≈ 2.5 m²/s
    # Typical large-scale models give D ≈ 130 m²/s (52x larger)
    # Default 1.0 = original Hanna values; adjust to tune horizontal spreading
    sigma_scale::T = 1.0
    # Separate multiplier for VERTICAL sigma (sigw).
    # Keep at 1.0 to maintain proper vertical mixing even when reducing horizontal.
    sigma_scale_vertical::T = 1.0
    # Use FLEXPART-compatible formulation for vertical turbulence:
    # - Stores normalized wp = w/σ_w instead of actual velocity
    # - Drift term inside O-U step: +dt*(dsigwdz + rhograd/rhoa*sigw)
    # - Unscaled random noise: sqrt(2*dt/tlw) instead of sigw*sqrt(...)
    # This follows the FLEXPART normalised-velocity formulation (Stohl et al., 2005)
    flexpart_mode::Bool = false
    # ===== SIMPLE CONVECTIVE INJECTION MODE =====
    # Robust alternative to complex CBL bi-Gaussian scheme for tropical convection.
    # When enabled and L < 0 (unstable), particles inside the PBL have a probability
    # of being randomly redistributed to a new height within the mixing layer.
    # This mimics FLEXPART's deep convection module without the complex PDF equations.
    use_simple_convection::Bool = false
    # Characteristic timescale for convective mixing (seconds)
    # At each timestep, probability of mixing = 1 - exp(-dt/timescale)
    # 900s (15 min) = PBL becomes well-mixed in ~45 mins (3 × timescale)
    convection_timescale::T = 1800.0
    # Drag coefficient for u* estimation from surface wind speed: u* = Cd × U
    # Default 0.05 corresponds to Cd ≈ 0.0025 (u* = sqrt(Cd) × U, so 0.05² = 0.0025)
    # Can tune down (e.g., 0.02) to reduce turbulent mixing intensity
    drag_coefficient::T = 0.05
    # Enable dynamic Obukhov length estimation from temperature gradient
    # When true, L is computed from surface heat flux estimate instead of neutral (1e10)
    use_dynamic_L::Bool = false
    # Scaling factor for convective velocity scale wst (w*).
    # Increasing this increases the strength of convective updrafts in the CBL scheme.
    wst_scale::T = 1.0
    # Kolmogorov constant for the CBL Langevin scheme.
    # Controls the memory of the O-U process in the skewed PDF model.
    # Default 3.0 matches FLEXPART and standard literature.
    c0_cbl::T = 3.0
    # Type of vertical redistribution for simple convection mode.
    # :uniform (default) redistributes particles randomly throughout the PBL.
    # :upper_half redistributes particles preferentially into the top 50% of the PBL.
    convective_injection_type::Symbol = :uniform
    # Minimum PBL height (m) for convective conditions (L < 0).
    # ERA5 Richardson number often underestimates tropical convective BL depth.
    # Setting this to 500-1000m ensures adequate mixing height in unstable conditions.
    # Only applies when use_dynamic_L is true and L < 0.
    convective_h_min::T = 500.0
    # Manual trigger for convective mode.
    # When true and use_dynamic_L is true, forces L < 0 to trigger convective_h_min
    # and CBL schemes regardless of the calculated stability.
    convective_trigger::Bool = false
    # Multiplier for all Lagrangian timescales (tlu, tlv, tlw).
    # Controls the decorrelation time of the O-U process.
    # >1 = longer memory (smoother, longer-range transport), <1 = faster reversion (more local mixing).
    # Default 1.0 = original Hanna (1982) timescales.
    tl_scale::T = 1.0
end

"""
    HannaTurbulenceParameters{T<:Real}

Turbulence parameters computed by Hanna (1982) scheme.

# Fields
- `sigu::T` - Standard deviation of u velocity fluctuation (m/s)
- `sigv::T` - Standard deviation of v velocity fluctuation (m/s)
- `sigw::T` - Standard deviation of w velocity fluctuation (m/s)
- `tlu::T` - Lagrangian timescale for u (s)
- `tlv::T` - Lagrangian timescale for v (s)
- `tlw::T` - Lagrangian timescale for w (s)
- `dsigwdz::T` - Vertical gradient of sigw (1/s)
- `stability::Symbol` - Atmospheric stability regime (:neutral, :unstable, :stable)
"""
struct HannaTurbulenceParameters{T<:Real}
    sigu::T
    sigv::T
    sigw::T
    tlu::T
    tlv::T
    tlw::T
    dsigwdz::T
    stability::Symbol
end

"""
    compute_hanna_parameters(z, h, L, ust, wst, config) -> HannaTurbulenceParameters

Compute turbulence parameters using Hanna (1982) scheme.

# Arguments
- `z::T` - Height above ground (m)
- `h::T` - Planetary boundary layer height (m)
- `L::T` - Obukhov length (m), negative for unstable conditions
- `ust::T` - Friction velocity (m/s)
- `wst::T` - Convective velocity scale (m/s)
- `config::HannaTurbulenceConfig` - Configuration

# Returns
- `HannaTurbulenceParameters` - Computed turbulence parameters

# References
- Hanna (1982), Equations 7.15-7.27
- Caughey (1982), Equation 4.15
- Ryall & Maryon (1998) for vertical variance
"""
function compute_hanna_parameters(
    z::T,
    h::T,
    L::T,
    ust::T,
    wst::T,
    config::HannaTurbulenceConfig{T}
) where T<:Real

    # Normalized height in boundary layer
    zeta = z / h

    # Stability parameter
    stability_ratio = h / abs(L)

    # Determine stability regime and compute parameters
    if stability_ratio < 1.0
        # Neutral conditions
        return compute_neutral_turbulence(z, h, ust, config)
    elseif L < 0.0
        # Unstable conditions
        return compute_unstable_turbulence(z, h, L, ust, wst, zeta, config)
    else
        # Stable conditions
        return compute_stable_turbulence(z, h, L, ust, zeta, config)
    end
end

"""
    compute_neutral_turbulence(z, h, ust, config) -> HannaTurbulenceParameters

Hanna (1982) parameterization for neutral conditions (|h/L| < 1).
Equations 7.25-7.27.
"""
function compute_neutral_turbulence(
    z::T,
    h::T,
    ust::T,
    config::HannaTurbulenceConfig{T}
) where T<:Real

    # Ensure ust has minimum value
    ust = max(T(1e-4), ust)

    # Use @fastmath for heavy transcendental operations
    # Safe because inputs are bounded and NaN/Inf are not expected
    @fastmath begin
        # Correction factor (z/ust)
        corr = z / ust

        # Eq. 7.25 Hanna 1982: sigu/ust = 2.0*exp(-3*f*z/ust)
        # where f is Coriolis parameter (1e-4)
        sigu = T(1e-2) + T(2.0) * ust * exp(T(-3.0) * config.f_coriolis * corr)

        # Eq. 7.26 Hanna 1982: sigv/ust = sigw/ust = 1.3*exp(-2*f*z/ust)
        sigw = T(1.3) * ust * exp(T(-2.0) * config.f_coriolis * corr)

        # Gradient of sigw
        dsigwdz = T(-2.0) * config.f_coriolis * sigw
        sigw = sigw + T(1e-2)
        sigv = sigw

        # Eq. 7.27 Hanna 1982: TL = 0.5*z/sigw/(1+15*f*z/ust)
        # Valid for all three components
        tl = T(0.5) * z / sigw / (T(1.0) + T(1.5e-3) * corr)
    end
    tlu = max(config.tlu_min, tl) * config.tl_scale
    tlv = max(config.tlv_min, tl) * config.tl_scale
    tlw = max(config.tlw_min, tl) * config.tl_scale

    # Ensure dsigwdz is not zero
    if dsigwdz == 0.0
        dsigwdz = T(1e-10)
    end

    # Apply separate scales for horizontal and vertical turbulence
    scale_h = config.sigma_scale
    scale_v = config.sigma_scale_vertical
    return HannaTurbulenceParameters(
        sigu * scale_h, sigv * scale_h, sigw * scale_v,
        tlu, tlv, tlw, dsigwdz * scale_v, :neutral
    )
end

"""
    compute_unstable_turbulence(z, h, L, ust, wst, zeta, config) -> HannaTurbulenceParameters

Hanna (1982) parameterization for unstable conditions (L < 0).
Equations 4.15 (Caughey 1982), 7.15-7.17.
"""
function compute_unstable_turbulence(
    z::T,
    h::T,
    L::T,
    ust::T,
    wst::T,
    zeta::T,
    config::HannaTurbulenceConfig{T}
) where T<:Real

    # Clamp zeta to valid range [0, inf) - can be > 1 for above BL
    zeta = max(zeta, T(0.0))

    # Use @fastmath for heavy power and sqrt operations
    @fastmath begin
        # Eq. 4.15 Caughey 1982
        sigu = T(1e-2) + ust * (T(12.0) - T(0.5) * h / L)^T(0.33333)
        sigv = sigu

        # Ryall & Maryon 1998: Height-dependent vertical variance
        # Height-dependent formulation for improved vertical mixing
        # Ensure argument to sqrt is non-negative (can be negative for z > h)
        sigw_arg = max(
            T(1.2) * wst^2 * (T(1.0) - T(0.9) * zeta) * zeta^T(0.66666) +
            (T(1.8) - T(1.4) * zeta) * ust^2,
            T(0.0)
        )
        sigw = sqrt(sigw_arg) + T(1e-2)

        # Gradient of sigw with height
        dsigwdz = T(0.5) / sigw / h * (
            T(-1.4) * ust^2 +
            wst^2 * (
                T(0.8) * max(zeta, T(1e-3))^T(-0.33333) -
                T(1.8) * zeta^T(0.66666)
            )
        )

        # Eq. 7.17 Hanna 1982: Horizontal Lagrangian timescales
        tlu = T(0.15) * h / sigu
        tlv = tlu

        # Vertical Lagrangian timescale - varies with height
        if z < abs(L)
            tlw = T(0.1) * z / (sigw * (T(0.55) - T(0.38) * abs(z / L)))
        elseif zeta < T(0.1)
            tlw = T(0.59) * z / sigw
        else
            tlw = T(0.15) * h / sigw * (T(1.0) - exp(T(-5.0) * zeta))
        end
    end  # @fastmath

    # Apply minimum timescales
    tlu = max(config.tlu_min, tlu) * config.tl_scale
    tlv = max(config.tlv_min, tlv) * config.tl_scale
    tlw = max(config.tlw_min, tlw) * config.tl_scale

    # Ensure dsigwdz is not zero
    if dsigwdz == 0.0
        dsigwdz = T(1e-10)
    end

    # Apply separate scales for horizontal and vertical turbulence
    scale_h = config.sigma_scale
    scale_v = config.sigma_scale_vertical
    return HannaTurbulenceParameters(
        sigu * scale_h, sigv * scale_h, sigw * scale_v,
        tlu, tlv, tlw, dsigwdz * scale_v, :unstable
    )
end

"""
    compute_stable_turbulence(z, h, L, ust, zeta, config) -> HannaTurbulenceParameters

Hanna (1982) parameterization for stable conditions (L > 0).
Equations 7.19-7.24.
"""
function compute_stable_turbulence(
    z::T,
    h::T,
    L::T,
    ust::T,
    zeta::T,
    config::HannaTurbulenceConfig{T}
) where T<:Real

    # Eq. 7.20 Hanna 1982
    sigu = T(1e-2) + T(2.0) * ust * (T(1.0) - zeta)

    # Eq. 7.19 Hanna 1982
    sigv = T(1e-2) + T(1.3) * ust * (T(1.0) - zeta)
    sigw = sigv

    # Gradient of sigw
    dsigwdz = T(-1.3) * ust / h

    # Eq. 7.22 Hanna 1982
    tlu = T(0.15) * h / sigu * sqrt(zeta)

    # Eq. 7.23 Hanna 1982
    tlv = T(0.467) * tlu

    # Eq. 7.24 Hanna 1982
    tlw = T(0.1) * h / sigw * zeta^T(0.8)

    # Apply minimum timescales
    tlu = max(config.tlu_min, tlu) * config.tl_scale
    tlv = max(config.tlv_min, tlv) * config.tl_scale
    tlw = max(config.tlw_min, tlw) * config.tl_scale

    # Ensure dsigwdz is not zero
    if dsigwdz == 0.0
        dsigwdz = T(1e-10)
    end

    # Apply separate scales for horizontal and vertical turbulence
    scale_h = config.sigma_scale
    scale_v = config.sigma_scale_vertical
    return HannaTurbulenceParameters(
        sigu * scale_h, sigv * scale_h, sigw * scale_v,
        tlu, tlv, tlw, dsigwdz * scale_v, :stable
    )
end

"""
    ornstein_uhlenbeck_step(u_old, sigma, tl, dt, rnd) -> u_new

Apply one step of the Ornstein-Uhlenbeck process for turbulent velocity.

The O-U process properly accounts for temporal autocorrelation:
- du = -(u/tl)dt + sigma*sqrt(2/tl)dW

For short timesteps (dt/tl < 0.5):
    u_new = (1 - dt/tl)*u_old + sigma*sqrt(2*dt/tl)*R

For longer timesteps:
    r = exp(-dt/tl)
    u_new = r*u_old + sigma*sqrt(1-r²)*R

# Arguments
- `u_old::T` - Previous turbulent velocity (m/s)
- `sigma::T` - Standard deviation of velocity fluctuation (m/s)
- `tl::T` - Lagrangian timescale (s)
- `dt::T` - Timestep (s)
- `rnd::T` - Random number from standard normal distribution

# Returns
- `T` - New turbulent velocity (m/s)
"""
function ornstein_uhlenbeck_step(
    u_old::T,
    sigma::T,
    tl::T,
    dt::T,
    rnd::T
) where T<:Real

    dt_over_tl = dt / tl

    # Use @fastmath for sqrt and exp - safe because inputs are bounded
    @fastmath if dt_over_tl < T(0.5)
        # Short timestep formula
        return muladd(sigma * sqrt(T(2.0) * dt_over_tl), rnd,
                      (T(1.0) - dt_over_tl) * u_old)
    else
        # Long timestep formula with exponential decay
        r = exp(-dt_over_tl)
        return muladd(sigma * sqrt(T(1.0) - r^2), rnd, r * u_old)
    end
end

"""
    flexpart_vertical_step(wp_old, sigw, dsigwdz, tlw, dt, rnd, rhoa, rhograd;
                          cbl_params, h, L) -> (wp_new, delz)

FLEXPART-compatible vertical turbulence step with optional CBL scheme.

Key differences from standard O-U:
1. wp is NORMALIZED (dimensionless): wp = w/σ_w
2. Drift term is INSIDE the O-U step, not separate
3. Random noise is unscaled: sqrt(2*dt/tlw) not σ_w*sqrt(...)
4. Displacement is wp * σ_w * dt
5. **NEW**: When -h/L > 5, uses skewed Langevin terms from CBL scheme

Based on the FLEXPART normalised-velocity formulation (Stohl et al., 2005).

# Arguments
- `wp_old::T` - Previous NORMALIZED vertical velocity (dimensionless, = w/σ_w)
- `sigw::T` - Standard deviation of w (m/s)
- `dsigwdz::T` - Vertical gradient of σ_w (1/s)
- `tlw::T` - Lagrangian timescale for w (s)
- `dt::T` - Timestep (s)
- `rnd::T` - Random number from standard normal distribution
- `rhoa::T` - Air density (kg/m³)
- `rhograd::T` - Vertical gradient of air density (kg/m⁴)
- `cbl_params::Union{Nothing,CBLParameters}` - CBL parameters (if available)
- `h::T` - Boundary layer height (m) - required if cbl_params provided
- `L::T` - Obukhov length (m) - required if cbl_params provided

# Returns
- `Tuple{T, T}` - (wp_new, delz) where wp_new is normalized velocity and delz is displacement (m)
"""
function flexpart_vertical_step(
    wp_old::T,
    sigw::T,
    dsigwdz::T,
    tlw::T,
    dt::T,
    rnd::T,
    rhoa::T,
    rhograd::T;
    cbl_params=nothing,
    h::T=T(1000.0),
    L::T=T(1e10),
    C_0::T=T(3.0)
) where T<:Real

    # Check if CBL should be activated (-h/L > 5 for strong convection)
    use_cbl = !isnothing(cbl_params) && L < T(0.0) && (-h / L) > T(5.0)

    if use_cbl
        # ===== CBL LANGEVIN MODE =====
        # Use skewed drift/diffusion from bi-Gaussian PDF
        # This generates non-Gaussian vertical distributions with proper skewness
        a_th, b_th = compute_cbl_langevin_terms(
            wp_old, sigw, cbl_params, tlw, rhoa, rhograd, T(1.0), C_0
        )

        # Langevin update: wp_new = wp + a_th*dt + b_th*sqrt(dt)*rnd
        # Note: b_th is already in normalized units (includes sqrt(2/tlw))
        wp_new = wp_old + a_th * dt + b_th * sqrt(dt) * rnd

        # Displacement: wp * σ_w * dt
        delz = wp_new * sigw * dt

    else
        # ===== STANDARD GAUSSIAN O-U MODE =====
        # Use standard FLEXPART turbulence with Gaussian drift
        dt_over_tlw = dt / tlw
        rhoaux = rhograd / rhoa

        # Use @fastmath for sqrt and exp - safe because inputs are bounded
        @fastmath begin
            drift = dsigwdz + rhoaux * sigw

            if dt_over_tlw < T(0.5)
                # Short timestep formula
                # wp = (1 - dt/tlw)*wp + sqrt(2*dt/tlw)*R + dt*(dsigwdz + rhoaux*sigw)
                wp_new = muladd(T(1.0) - dt_over_tlw, wp_old,
                               muladd(sqrt(T(2.0) * dt_over_tlw), rnd, dt * drift))
            else
                # Long timestep formula with exponential decay
                # wp = exp(-dt/tlw)*wp + sqrt(1-r²)*R + tlw*(1-r)*(dsigwdz + rhoaux*sigw)
                r = exp(-dt_over_tlw)
                wp_new = muladd(r, wp_old,
                               muladd(sqrt(T(1.0) - r^2), rnd, tlw * (T(1.0) - r) * drift))
            end

            # Displacement: wp * sigw * dt
            delz = wp_new * sigw * dt
        end
    end

    return (wp_new, delz)
end

# ============================================================================
# Convective Boundary Layer (CBL) Scheme
# Based on Luhar, Hibberd & Hurley (1996)
# ============================================================================

"""
    CBLParameters{T<:Real}

Parameters for the Convective Boundary Layer scheme.

# Fields
- `w3::T` - Third moment of vertical velocity (m³/s³)
- `skew::T` - Skewness of vertical velocity distribution
- `sigma_wa::T` - Std dev of updraft velocity (m/s)
- `sigma_wb::T` - Std dev of downdraft velocity (m/s)
- `wa::T` - Mean updraft velocity (m/s)
- `wb::T` - Mean downdraft velocity (m/s)
- `a_updraft::T` - Fraction of updrafts
- `b_downdraft::T` - Fraction of downdrafts (= 1 - a_updraft)
- `transition::T` - Stability transition function (0-1)
- `dwa::T` - Vertical gradient of wa (m/s per m = 1/s)
- `dwb::T` - Vertical gradient of wb (1/s)
- `dsigma_wa::T` - Vertical gradient of sigma_wa (1/s)
- `dsigma_wb::T` - Vertical gradient of sigma_wb (1/s)
- `da_updraft::T` - Vertical gradient of a_updraft (1/m)
- `db_downdraft::T` - Vertical gradient of b_downdraft (1/m)
"""
struct CBLParameters{T<:Real}
    w3::T
    skew::T
    sigma_wa::T
    sigma_wb::T
    wa::T
    wb::T
    a_updraft::T
    b_downdraft::T
    transition::T
    # Vertical derivatives for Langevin drift term
    dwa::T
    dwb::T
    dsigma_wa::T
    dsigma_wb::T
    da_updraft::T
    db_downdraft::T
end

"""
    compute_cbl_parameters(z, h, L, ust, wst, sigmaw, dsigwdz) -> CBLParameters

Compute Convective Boundary Layer parameters for bi-Gaussian velocity PDF.

Based on Luhar-Hibberd-Britter (1996, 2000) formulation using analytic derivatives.

# Arguments
- `z::T` - Height above ground (m)
- `h::T` - Boundary layer height (m)
- `L::T` - Obukhov length (m)
- `ust::T` - Friction velocity (m/s)
- `wst::T` - Convective velocity scale (m/s)
- `sigmaw::T` - Standard deviation of vertical velocity (m/s)
- `dsigwdz::T` - Vertical gradient of sigmaw (1/s)

# Returns
- `CBLParameters` - Parameters for the CBL scheme including vertical derivatives
"""
function compute_cbl_parameters(
    z::T,
    h::T,
    L::T,
    ust::T,
    wst::T,
    sigmaw::T,
    dsigwdz::T
) where T<:Real

    # Compute base parameters
    params = compute_cbl_base_parameters(z, h, L, sigmaw, wst)
    
    # Extract base values
    w3 = params.w3
    skew = params.skew
    sigma_wa = params.sigma_wa
    sigma_wb = params.sigma_wb
    wa = params.wa
    wb = params.wb
    aluarw = params.a_updraft
    bluarw = params.b_downdraft
    transition = params.transition
    
    # Normalized height
    zeta = clamp(z / h, T(1e-4), T(0.99))
    
    # 1. Analytic derivative of w3
    # w3 = 1.2 * zeta * (1 - zeta)^1.5 * wst^3 * transition
    # dw3dz = (1.2 * wst^3 / h) * (1 - zeta)^0.5 * (1 - 2.5 * zeta) * transition
    dw3dz = (T(1.2) * wst^3 / h) * sqrt(T(1.0) - zeta) * (T(1.0) - T(2.5) * zeta) * transition
    
    # 2. Analytic derivative of skewness
    # skew = w3 / sigmaw^3
    # dskewdz = (1/sigmaw^3) * dw3dz - (3 * w3 / sigmaw^4) * dsigwdz
    dskewdz = (T(1.0) / (sigmaw^3)) * dw3dz - (T(3.0) * w3 / (sigmaw^4)) * dsigwdz
    
    # 3. Analytic derivative of fluarw (f)
    # fluarw = 0.66667 * skew^(1/3)
    costluar4 = T(0.66667)
    fluarw = costluar4 * cbrt(skew)
    # dfdz = (costluar4 / 3) * skew^(-2/3) * dskewdz
    # Use max(abs(skew), eps) to avoid singularity at skew=0
    dfdz = (costluar4 / T(3.0)) * (max(abs(skew), T(1e-6))^(T(-2.0)/T(3.0))) * dskewdz
    
    # 4. Analytic derivative of rluarw (r)
    # r = (1+f^2)^3 * skew^2 / ((3+f^2)^2 * f^2)
    # drdz = r * [ 6ff'/(1+f^2) + 2skew'/skew - 4ff'/(3+f^2) - 2f'/f ]
    f2 = fluarw^2
    rluarw = (T(1.0) + f2)^3 * skew^2 / ((T(3.0) + f2)^2 * f2)
    
    term_f1 = (T(6.0) * fluarw * dfdz) / (T(1.0) + f2)
    term_s = (T(2.0) * dskewdz) / max(abs(skew), T(1e-6))
    term_f2 = (T(4.0) * fluarw * dfdz) / (T(3.0) + f2)
    term_f3 = (T(2.0) * dfdz) / max(abs(fluarw), T(1e-6))
    
    drdz = rluarw * (term_f1 + term_s - term_f2 - term_f3)
    
    # 5. Analytic derivative of a_updraft (a)
    # a = 0.5 * (1 - sqrt(r/(4+r)))
    # dadz = - drdz / (sqrt(r) * (4+r)^1.5)
    da_updraft = - drdz / (max(sqrt(rluarw), T(1e-6)) * (T(4.0) + rluarw)^T(1.5))
    db_downdraft = -da_updraft
    
    # 6. Analytic derivative of sigma_wa and sigma_wb
    # sigma_wa = sigmaw * sqrt(b / (a*(1+f^2)))
    # dsigma_wa/sigma_wa = sigmaw'/sigmaw + 0.5 * (b'/b - a'/a - 2ff'/(1+f^2))
    term_sigw = dsigwdz / sigmaw
    term_b = db_downdraft / max(abs(bluarw), T(1e-6))
    term_a = da_updraft / max(abs(aluarw), T(1e-6))
    term_f = (T(2.0) * fluarw * dfdz) / (T(1.0) + f2)
    
    dsigma_wa = sigma_wa * (term_sigw + T(0.5) * (term_b - term_a - term_f))
    dsigma_wb = sigma_wb * (term_sigw + T(0.5) * (term_a - term_b - term_f))
    
    # 7. Analytic derivative of wa and wb
    # wa = f * sigma_wa
    dwa = dfdz * sigma_wa + fluarw * dsigma_wa
    dwb = dfdz * sigma_wb + fluarw * dsigma_wb

    return CBLParameters(
        w3, skew,
        sigma_wa, sigma_wb,
        wa, wb,
        aluarw, bluarw,
        transition,
        dwa, dwb, dsigma_wa, dsigma_wb, da_updraft, db_downdraft
    )
end

"""
    compute_cbl_base_parameters(z, h, L, sigmaw, wst) -> NamedTuple

Compute base CBL parameters without derivatives (used for finite differencing).
"""
function compute_cbl_base_parameters(
    z::T,
    h::T,
    L::T,
    sigmaw::T,
    wst::T
) where T<:Real

    zeta = clamp(z / h, T(0.0), T(1.0))  # Clamp to valid range
    eps = T(1e-6)

    # Stability transition function (Cassiani et al. 2015)
    # Smooth transition between -h/L = 5 and 15
    neg_h_over_L = -h / L
    if neg_h_over_L < T(15.0)
        transition = (sin(((neg_h_over_L + T(10.0)) / T(10.0)) * T(π))) * T(0.5) + T(0.5)
    else
        transition = T(1.0)
    end

    # Second moment (variance)
    w2 = sigmaw^2

    # Third moment profile from Lenschow et al. (2000)
    # w³(z) = 1.2*z*(1-z)^(3/2)*wst³
    # Ensure (1-zeta) is positive before fractional exponent
    zeta_safe = clamp(zeta, T(0.0), T(0.99))  # Keep away from 1.0
    w3 = (T(1.2) * zeta_safe * (T(1.0) - zeta_safe)^T(1.5) + eps) * wst^3 * transition

    # Skewness
    skew = w3 / (w2^T(1.5))
    skew2 = skew^2

    # Luhar-Hibberd formulation constants
    costluar4 = T(0.66667)
    fluarw = costluar4 * cbrt(skew)  # Cube root
    fluarw2 = fluarw^2

    # Compute r and r^(1/2)
    rluarw = (T(1.0) + fluarw2)^3 * skew2 / ((T(3.0) + fluarw2)^2 * fluarw2)
    xluarw = sqrt(rluarw)

    # Updraft and downdraft fractions
    a_updraft = T(0.5) * (T(1.0) - xluarw / sqrt(T(4.0) + rluarw))
    b_downdraft = T(1.0) - a_updraft

    # Standard deviations of updraft and downdraft
    radw2 = sqrt(w2)
    sigma_wa = radw2 * sqrt(b_downdraft / (a_updraft * (T(1.0) + fluarw2)))
    sigma_wb = radw2 * sqrt(a_updraft / (b_downdraft * (T(1.0) + fluarw2)))

    # Mean velocities
    wa = fluarw * sigma_wa
    wb = fluarw * sigma_wb

    return (
        w3=w3, skew=skew,
        sigma_wa=sigma_wa, sigma_wb=sigma_wb,
        wa=wa, wb=wb,
        a_updraft=a_updraft, b_downdraft=b_downdraft,
        transition=transition
    )
end

"""
    cbrt(x::T) where T<:Real -> T

Cube root that handles negative numbers correctly.
"""
function cbrt(x::T) where T<:Real
    return sign(x) * abs(x)^T(0.33333)
end

# ============================================================================
# CBL Langevin Scheme
# Based on Luhar-Hibberd-Hurley (1996)
# ============================================================================

"""
    compute_cbl_langevin_terms(wp, sigmaw, cbl_params, tlw, rhoa, rhograd, timedir, C_0) -> (a_th, b_th)

Compute drift (a_th) and diffusion (b_th) coefficients for CBL Langevin equation.

This implements the skewed PDF turbulence scheme using the Luhar-Hibberd-Hurley
bi-Gaussian formulation to generate non-Gaussian vertical velocity distributions
with realistic skewness and kurtosis.

# Arguments
- `wp::T` - Normalized vertical velocity (w/σ_w, dimensionless)
- `sigmaw::T` - Standard deviation of w (m/s)
- `cbl_params::CBLParameters` - CBL parameters including derivatives
- `tlw::T` - Lagrangian timescale for w (s)
- `rhoa::T` - Air density (kg/m³)
- `rhograd::T` - Vertical gradient of air density (kg/m⁴)
- `timedir::T` - Time direction (+1 for forward, -1 for backward)
- `C_0::T` - Kolmogorov constant (typically 3.0)

# Returns
- `Tuple{T, T}` - (a_th, b_th) where:
  - `a_th`: Drift coefficient (1/s) for wp update
  - `b_th`: Diffusion coefficient (1/√s) for wp update

# Physics
The Langevin update is: wp_new = wp + a_th*dt + b_th*sqrt(dt)*rand()

where a_th includes the skewed drift from bi-Gaussian PDF structure,
and b_th is the standard Kolmogorov diffusion coefficient.

# References
- Luhar et al. (1996): Comparison of closure schemes for velocity PDF
- Stohl et al. (2005): FLEXPART version 6.2
"""
function compute_cbl_langevin_terms(
    wp::T,
    sigmaw::T,
    cbl_params::CBLParameters{T},
    tlw::T,
    rhoa::T,
    rhograd::T,
    timedir::T = T(1.0),
    C_0::T = T(3.0)
) where T<:Real

    # Alpha coefficient: α = 2σ²/(C₀τ)
    alpha = T(2.0) * sigmaw^2 / (C_0 * tlw)

    # Diffusion coefficient (same as standard Gaussian)
    # b_th = sqrt(C₀ α) = sqrt(2σ²/τ)
    b_th = sqrt(C_0 * alpha)

    # Convert normalized wp to dimensional velocity for PDF calculations
    # wold = timedir * wp * σ_w
    wold = timedir * wp * sigmaw
    wold2 = wold^2

    # Extract CBL parameters
    wa = cbl_params.wa
    wb = cbl_params.wb
    sigma_wa = cbl_params.sigma_wa
    sigma_wb = cbl_params.sigma_wb
    sigma_wa2 = sigma_wa^2
    sigma_wb2 = sigma_wb^2
    aluarw = cbl_params.a_updraft
    bluarw = cbl_params.b_downdraft

    # Velocity differences for Q term
    deltawa = wold - wa
    deltawb = wold + wb  # Note sign for downdraft

    # Gaussian PDF values (dimensional velocity space)
    # pa = (1/√(2π σ_a²)) exp(-(wold - wa)²/(2σ_a²))
    pa = exp(-deltawa^2 / (T(2.0) * sigma_wa2)) / (sqrt(T(2.0) * T(π)) * sigma_wa)
    pb = exp(-deltawb^2 / (T(2.0) * sigma_wb2)) / (sqrt(T(2.0) * T(π)) * sigma_wb)

    # Total PDF (bi-Gaussian mixture)
    P_tot = aluarw * pa + bluarw * pb

    # Avoid division by zero
    if P_tot < T(1e-20)
        P_tot = T(1e-20)
    end

    # Q term: Weighted velocity difference (Luhar et al., 1996)
    # Q = timedir * (a*ρ*(w-wa)/σa² * Pa + b*ρ*(w+wb)/σb² * Pb)
    Q = timedir * (
        (aluarw * rhoa * deltawa / sigma_wa2) * pa +
        (bluarw * rhoa * deltawb / sigma_wb2) * pb
    )

    # Phi term: Vertical drift with erf components (Luhar et al., 1996)
    Phi = compute_Phi_term(
        wold, wold2, cbl_params, rhoa, rhograd, pa, pb
    )

    # Drift coefficient: a_th = (1/P_tot) * (-(C₀/2)αQ + Φ)
    # This replaces the standard Gaussian drift when CBL is active
    a_th = (T(1.0) / P_tot) * (-(C_0 / T(2.0)) * alpha * Q + Phi)

    # NORMALIZATION FIX: a_th and b_th are currently in dimensional units (m/s² and m/s^1.5).
    # Since our vertical velocity wp is normalized (dimensionless, w/σ_w), we must
    # divide both coefficients by σ_w to bring them into normalized space.
    # This ensures consistency with the Langevin update in flexpart_vertical_step.
    a_th_norm = a_th / sigmaw
    b_th_norm = b_th / sigmaw

    return (a_th_norm, b_th_norm)
end

"""
    compute_Phi_term(wold, wold2, cbl_params, rhoa, rhograd, pa, pb) -> Phi

Compute the Phi vertical drift term for CBL Langevin equation.

This is the complex drift term from the Luhar et al. (1996) formulation,
incorporating error functions and gradients of bi-Gaussian parameters.

# Arguments
- `wold::T` - Dimensional vertical velocity (m/s)
- `wold2::T` - wold squared (for efficiency)
- `cbl_params::CBLParameters` - CBL parameters with derivatives
- `rhoa::T` - Air density (kg/m³)
- `rhograd::T` - Vertical gradient of density (kg/m⁴)
- `pa::T` - Updraft Gaussian PDF value
- `pb::T` - Downdraft Gaussian PDF value

# Returns
- `T` - Phi drift term

# Formulation
Phi has 8 components:
1-2: Error function terms for mean velocities
3-6: Gaussian-weighted gradient terms for updrafts
7-8: Gaussian-weighted gradient terms for downdrafts
"""
function compute_Phi_term(
    wold::T,
    wold2::T,
    cbl_params::CBLParameters{T},
    rhoa::T,
    rhograd::T,
    pa::T,
    pb::T
) where T<:Real

    # Extract parameters
    wa = cbl_params.wa
    wb = cbl_params.wb
    sigma_wa = cbl_params.sigma_wa
    sigma_wb = cbl_params.sigma_wb
    sigma_wa2 = sigma_wa^2
    sigma_wb2 = sigma_wb^2
    aluarw = cbl_params.a_updraft
    bluarw = cbl_params.b_downdraft

    # Extract derivatives
    dwa = cbl_params.dwa
    dwb = cbl_params.dwb
    dsigma_wa = cbl_params.dsigma_wa
    dsigma_wb = cbl_params.dsigma_wb
    daluarw = cbl_params.da_updraft
    dbluarw = cbl_params.db_downdraft
    ddens = rhograd

    # Error function arguments
    # aperfa = (wold - wa) / (√2 σ_a)
    aperfa = (wold - wa) / (sqrt(T(2.0)) * sigma_wa)
    aperfb = (wold + wb) / (sqrt(T(2.0)) * sigma_wb)

    # Phi components (Luhar et al., 1996)

    # Lines 215: -0.5 * (a*ρ*dwa + ρ*wa*da + a*wa*dρ) * erf(aperfa)
    term1 = -T(0.5) * (
        aluarw * rhoa * dwa +
        rhoa * wa * daluarw +
        aluarw * wa * ddens
    ) * erf(aperfa)

    # Lines 216-218: σ_a * (...) * pa (updraft Gaussian-weighted terms)
    term2 = sigma_wa * (
        aluarw * rhoa * dsigma_wa * (wold2 / sigma_wa2 + T(1.0)) +
        sigma_wa * rhoa * daluarw +
        sigma_wa * ddens * aluarw +
        aluarw * wold * rhoa / sigma_wa2 * (sigma_wa * dwa - wa * dsigma_wa)
    ) * pa

    # Line 219: +0.5 * (b*ρ*dwb + wb*ρ*db + wb*b*dρ) * erf(aperfb)
    term3 = T(0.5) * (
        bluarw * rhoa * dwb +
        wb * rhoa * dbluarw +
        wb * bluarw * ddens
    ) * erf(aperfb)

    # Lines 220-222: σ_b * (...) * pb (downdraft Gaussian-weighted terms)
    term4 = sigma_wb * (
        bluarw * rhoa * dsigma_wb * (wold2 / sigma_wb2 + T(1.0)) +
        sigma_wb * rhoa * dbluarw +
        sigma_wb * ddens * bluarw +
        bluarw * wold * rhoa / sigma_wb2 * (-sigma_wb * dwb + wb * dsigma_wb)
    ) * pb

    return term1 + term2 + term3 + term4
end

"""
    apply_cbl_scheme(w_old, cbl_params, sigmaw, tlw, dt, rnd1, rnd2) -> (w_new, use_cbl)

DEPRECATED: Old "jump" sampler version of CBL scheme.

This function is kept for backward compatibility but is NOT used in
FLEXPART-compatible mode. Use compute_cbl_langevin_terms instead for
proper Langevin integration with skewed drift/diffusion coefficients.
"""
function apply_cbl_scheme(
    w_old::T,
    cbl_params::CBLParameters{T},
    sigmaw::T,
    tlw::T,
    dt::T,
    rnd1::T,
    rnd2::T
) where T<:Real

    # Determine if particle is in updraft or downdraft based on previous velocity
    in_updraft = w_old > T(0.0)

    # Sample new velocity from appropriate distribution
    if in_updraft || (rnd1 < cbl_params.a_updraft)
        # Sample from updraft distribution
        w_new = rnd2 * cbl_params.sigma_wa + cbl_params.wa
    else
        # Sample from downdraft distribution
        w_new = rnd2 * cbl_params.sigma_wb - cbl_params.wb
    end

    # Check if velocity is reasonable
    if isnan(w_new) || isinf(w_new) || abs(w_new) > T(10.0) * sigmaw
        # Fall back to standard Hanna scheme
        return (
            ornstein_uhlenbeck_step(w_old, sigmaw, tlw, dt, rnd2),
            false
        )
    end

    return (w_new, true)
end

# ============================================================================
# Simple Convective Injection Scheme
# Robust alternative to CBL for tropical convection
# ============================================================================

"""
    apply_simple_convection(z_m, h, L, dt, config) -> (z_new, was_mixed)

Apply simple convective injection for unstable boundary layer conditions.
...
# Arguments
- `z_m::T` - Current height above ground (m)
- `h::T` - Planetary boundary layer height (m)
- `L::T` - Obukhov length (m), negative for unstable conditions
- `dt::T` - Timestep (s)
- `config::HannaTurbulenceConfig` - Configuration containing timescale and injection type

# Returns
...
"""
function apply_simple_convection(
    z_m::T,
    h::T,
    L::T,
    dt::T,
    config::HannaTurbulenceConfig{T}
) where T<:Real

    # Only apply if atmosphere is unstable (L < 0) and we are inside the PBL
    if L < T(0.0) && z_m < h && h > T(0.0)
        # Probability of a strong convective event occurring this timestep
        # P = 1 - exp(-dt/T_mix)
        prob_mix = T(1.0) - exp(-dt / config.convection_timescale)

        if rand() < prob_mix
            # Redistribution: Determined by injection type
            if config.convective_injection_type == :upper_half
                # Preferential injection into the top 50% of the PBL
                # Simulates deep convective updrafts lofting material high
                z_new = (T(0.5) + T(0.5) * rand()) * h
            else
                # :uniform (default) - random position within the PBL
                # This ensures the PBL becomes well-mixed over time
                z_new = rand() * h
            end
            return (z_new, true)
        end
    end

    return (z_m, false)  # No change
end

"""
    estimate_obukhov_length(T_surf, T_2m, u_star, T_ref; z_ref=2.0) -> L

Estimate Monin-Obukhov length from surface temperature gradient.

The Obukhov length characterises atmospheric stability:
- L < 0: Unstable (buoyancy dominates, convection)
- L > 0: Stable (shear dominates, stratification)
- |L| → ∞: Neutral

This uses the bulk aerodynamic approximation for sensible heat flux:
    H = ρ c_p C_H U (T_surf - T_air)

Then L = -u*³ T_ref / (κ g H / (ρ c_p))

# Arguments
- `T_surf::T` - Surface (skin) temperature (K)
- `T_2m::T` - Air temperature at 2m (K)
- `u_star::T` - Friction velocity (m/s)
- `T_ref::T` - Reference temperature for buoyancy (K), typically T_2m

# Keyword Arguments
- `z_ref::T` - Reference height for temperature (m), default 2.0

# Returns
- `T` - Obukhov length L (m). Clamped to |L| ≥ 1 for numerical stability.

# Constants
- κ = 0.4 (von Kármán constant)
- g = 9.81 m/s²
- C_H ≈ 0.002 (bulk heat transfer coefficient, typical over land)
"""
function estimate_obukhov_length(
    T_surf::Real,
    T_2m::Real,
    u_star::Real,
    T_ref::Real;
    z_ref::Real = 2.0,
    hflux::Union{Nothing, Real} = nothing
)
    # Promote all arguments to a common float type for calculation
    T = promote_type(typeof(T_surf), typeof(T_2m), typeof(u_star), typeof(T_ref), typeof(z_ref))
    
    # Physical constants
    κ = T(0.4)      # von Kármán constant
    g = T(9.81)     # gravitational acceleration (m/s²)
    C_H = T(0.002)  # bulk heat transfer coefficient (dimensionless)
    rho_cp = T(1205.0) # Density * specific heat capacity (kg/m³ * J/kg/K)

    # Calculate sensible heat flux proxy or use provided flux
    H_norm = if !isnothing(hflux) && abs(hflux) > T(1e-3)
        # Use provided sensible heat flux (W/m²).
        # ERA5 convention: positive downwards (warming surface).
        # Convection requires upward flux (cooling surface), so H_up = -sshf.
        H_up = -T(hflux)
        H_up / rho_cp
    else
        # Temperature difference (positive = surface warmer = unstable)
        dT = T(T_surf) - T(T_2m)

        # Sensible heat flux proxy (normalised): H/(ρ c_p) ≈ C_H × U_surf × ΔT
        # But we don't have U_surf directly. Approximate: U_surf ≈ u_star / 0.05
        U_approx = T(u_star) / T(0.05)
        C_H * U_approx * dT  # Has units K⋅m/s
    end

    # Obukhov length: L = -u*³ T_ref / (κ g H_norm)
    # Note: H_norm already has the /(ρ c_p) built in
    denominator = κ * g * H_norm

    # Avoid division by zero - if denominator ≈ 0, atmosphere is neutral
    if abs(denominator) < T(1e-10)
        return T(1e10)  # Neutral
    end

    L = -(T(u_star)^3 * T(T_ref)) / denominator

    # Clamp to reasonable range: |L| ∈ [1, 1e10]
    # Very small |L| causes numerical issues in Hanna scheme
    if abs(L) < T(1.0)
        L = sign(L) * T(1.0)
    elseif abs(L) > T(1e10)
        L = sign(L) * T(1e10)
    end

    return L
end
