# Deposition Module - Dry and Wet Deposition
#
# Implements surface deposition mechanisms for particles

# Physical constants
const R_GAS = 287.05  # Gas constant for dry air (J/(kg·K))
const G_GRAVITY = 9.8  # Gravitational acceleration (m/s²)
const CP_AIR = 1005.0  # Specific heat of air at constant pressure (J/(kg·K))
const BOLTZMANN = 1.380649e-23  # Boltzmann constant (J/K)
const VON_KARMAN = 0.4  # von Kármán constant

# Air properties at reference conditions
const NU_AIR = 1.5e-5  # Kinematic viscosity of air (m²/s) at 15°C
const LAMBDA_AIR = 0.065e-6  # Mean free path of air molecules (m)

"""
    LandUseClass

Land use classification for surface resistance calculations.

# Classes
- Water bodies (ocean, lakes)
- Forest (deciduous, coniferous)
- Grassland
- Cropland
- Urban
- Bare soil
- Snow/ice

Note: Classification system compatible with standard land-use databases.
"""
@enum LandUseClass::Int8 begin
    WATER = 1
    DECIDUOUS_FOREST = 2
    CONIFEROUS_FOREST = 3
    MIXED_FOREST = 4
    GRASSLAND = 5
    CROPLAND = 6
    URBAN = 7
    BARE_SOIL = 8
    SNOW_ICE = 9
end

"""
    SeasonCategory

Seasonal categories for vegetation parameters.

# Categories
- WINTER: Dormant vegetation
- SPRING: Growing season
- SUMMER: Full vegetation
- AUTUMN: Senescent vegetation
"""
@enum SeasonCategory::Int8 begin
    WINTER = 1
    SPRING = 2
    SUMMER = 3
    AUTUMN = 4
end

"""
    DepositionScheme

Dry deposition parameterization schemes.

# Schemes
- Zhang2001: Zhang et al. (2001) scheme (DEFAULT)
- Emerson2020: Emerson et al. (2020) revised scheme

# References
- Zhang et al. (2001): A size-segregated particle dry deposition scheme
  https://doi.org/10.1016/S1352-2310(00)00326-5
- Emerson et al. (2020): Revisiting particle dry deposition
  https://doi.org/10.1073/pnas.2014761117
"""
@enum DepositionScheme::Int8 begin
    Zhang2001 = 1
    Emerson2020 = 2
end

"""
    DryDepositionParams{T<:Real}

Parameters for dry deposition calculations.

# Fields
- `particle_diameter::T`: Particle diameter (meters)
- `particle_density::T`: Particle density (kg/m³)
- `reference_height::T`: Reference height for deposition (m, typically 30m)
- `roughness_length::Matrix{T}`: Surface roughness length z₀ (m) for each grid cell
- `land_use::Matrix{LandUseClass}`: Land use classification for each grid cell

# Example
```julia
params = DryDepositionParams(
    particle_diameter=1e-6,  # 1 μm
    particle_density=2500.0,  # 2.5 g/cm³
    reference_height=30.0,
    roughness_length=fill(0.1, 100, 100),
    land_use=fill(GRASSLAND, 100, 100)
)
```
"""
struct DryDepositionParams{T<:Real}
    particle_diameter::T
    particle_density::T
    reference_height::T
    roughness_length::Matrix{T}
    land_use::Matrix{LandUseClass}
end

"""
    WetDepositionParams{T<:Real}

Parameters for wet deposition calculations.

# Fields
- `particle_diameter::T`: Particle diameter (meters)
- `particle_density::T`: Particle density (kg/m³)
- `washout_coefficient::T`: Below-cloud scavenging coefficient (s⁻¹/(mm/h)ᵇ)
- `rainout_coefficient::T`: In-cloud scavenging coefficient (s⁻¹)
- `precipitation_threshold::T`: Minimum precipitation for wet deposition (mm/h)

# Notes
Default coefficients from Bartnicki (2011) for aerosols.
"""
struct WetDepositionParams{T<:Real}
    particle_diameter::T
    particle_density::T
    washout_coefficient::T
    rainout_coefficient::T
    precipitation_threshold::T
end

# Constructor with defaults
function WetDepositionParams(diameter::T, density::T;
                            washout_coef::T=2.0e-5,
                            rainout_coef::T=1.0e-4,
                            precip_threshold::T=0.01) where T<:Real
    WetDepositionParams(diameter, density, washout_coef, rainout_coef, precip_threshold)
end

"""
    compute_air_density(pressure_pa::Real, temperature_k::Real)

Compute air density using ideal gas law.

# Arguments
- `pressure_pa`: Atmospheric pressure (Pa)
- `temperature_k`: Air temperature (K)

# Returns
- Air density (kg/m³)

# Formula
ρ = P / (R × T)
"""
function compute_air_density(pressure_pa::T, temperature_k::T) where T<:Real
    return pressure_pa / (R_GAS * temperature_k)
end

"""
    compute_friction_velocity(tau_x::Real, tau_y::Real, rho_air::Real)

Compute friction velocity from surface stress components.

# Arguments
- `tau_x`, `tau_y`: Surface stress components (N/m²)
- `rho_air`: Air density (kg/m³)

# Returns
- Friction velocity u* (m/s)

# Formula
u* = √(|τ| / ρ)
"""
function compute_friction_velocity(tau_x::T, tau_y::T, rho_air::T) where T<:Real
    tau_magnitude = hypot(tau_x, tau_y)
    return sqrt(tau_magnitude / rho_air)
end

"""
    compute_monin_obukhov_length(u_star::Real, temperature::Real, heat_flux::Real, rho_air::Real)

Compute Monin-Obukhov length for atmospheric stability.

# Arguments
- `u_star`: Friction velocity (m/s)
- `temperature`: Air temperature (K)
- `heat_flux`: Surface sensible heat flux (W/m²)
- `rho_air`: Air density (kg/m³)

# Returns
- Monin-Obukhov length L (m)
  - L > 0: Stable atmosphere
  - L < 0: Unstable atmosphere
  - L → ∞: Neutral atmosphere

# Formula
L = -ρ × Cₚ × T × u*³ / (κ × g × H)
"""
function compute_monin_obukhov_length(u_star::T, temperature::T, heat_flux::T, rho_air::T) where T<:Real
    # Neutral conditions: heat flux < 1 W/m²
    # At typical atmospheric conditions, |H| < 1 W/m² is effectively neutral
    if abs(heat_flux) < 1.0
        return T(Inf)  # Neutral conditions
    end

    return -rho_air * CP_AIR * temperature * u_star^3 / (VON_KARMAN * G_GRAVITY * heat_flux)
end

"""
    aerodynamic_resistance(L::Real, u_star::Real, z0::Real, z_ref::Real=30.0)

Compute aerodynamic resistance for momentum transfer.

# Arguments
- `L`: Monin-Obukhov length (m)
- `u_star`: Friction velocity (m/s)
- `z0`: Surface roughness length (m)
- `z_ref`: Reference height (m, default 30m)

# Returns
- Aerodynamic resistance Rₐ (s/m)

# Notes
Includes atmospheric stability corrections via Monin-Obukhov similarity theory.
"""
function aerodynamic_resistance(L::T, u_star::T, z0::T, z_ref::T=T(30.0)) where T<:Real
    if u_star < 1e-10
        return T(1e10)  # Very large resistance for calm conditions
    end

    # Stability parameter
    zeta_ref = z_ref / L
    zeta_0 = z0 / L

    # Stability correction functions (Paulson, 1970)
    if L < 0  # Unstable
        # Businger-Dyer formulation
        x_ref = (1 - 16 * zeta_ref)^0.25
        x_0 = (1 - 16 * zeta_0)^0.25
        psi_m_ref = 2 * log((1 + x_ref)/2) + log((1 + x_ref^2)/2) - 2 * atan(x_ref) + π/2
        psi_m_0 = 2 * log((1 + x_0)/2) + log((1 + x_0^2)/2) - 2 * atan(x_0) + π/2
    elseif L > 0  # Stable
        psi_m_ref = -5 * zeta_ref
        psi_m_0 = -5 * zeta_0
    else  # Neutral
        psi_m_ref = zero(T)
        psi_m_0 = zero(T)
    end

    # Aerodynamic resistance
    ra = (log(z_ref/z0) - psi_m_ref + psi_m_0) / (VON_KARMAN * u_star)
    return max(ra, T(1.0))  # Minimum resistance of 1 s/m
end

"""
    cunningham_slip_factor(diameter::Real, temperature::Real)

Compute Cunningham slip correction factor for small particles.

# Arguments
- `diameter`: Particle diameter (m)
- `temperature`: Air temperature (K)

# Returns
- Cunningham correction factor (dimensionless)

# Formula
Cₛ = 1 + (2λ/d) × (1.257 + 0.4 × exp(-0.55d/λ))

where λ is the mean free path of air molecules.
"""
function cunningham_slip_factor(diameter::T, temperature::T) where T<:Real
    lambda = LAMBDA_AIR * (temperature / 288.15)  # Temperature-dependent mean free path
    return 1 + 2 * lambda / diameter * (1.257 + 0.4 * exp(-0.55 * diameter / lambda))
end

"""
    brownian_diffusivity(diameter::Real, temperature::Real, rho_air::Real)

Compute Brownian diffusion coefficient for particles in air.

# Arguments
- `diameter`: Particle diameter (m)
- `temperature`: Air temperature (K)
- `rho_air`: Air density (kg/m³)

# Returns
- Diffusion coefficient D (m²/s)

# Formula
D = (k_B × T × Cₛ) / (3π × μ × d)

where Cₛ is the Cunningham slip factor.
"""
function brownian_diffusivity(diameter::T, temperature::T, rho_air::T) where T<:Real
    cslip = cunningham_slip_factor(diameter, temperature)
    mu = NU_AIR * rho_air  # Dynamic viscosity
    return BOLTZMANN * temperature * cslip / (3 * π * mu * diameter)
end

"""
    get_characteristic_radius(land_use::LandUseClass, season::SeasonCategory)

Get characteristic radius A for interception (Zhang et al. 2001).

# Arguments
- `land_use`: Land use classification
- `season`: Seasonal category

# Returns
- Characteristic radius A (mm), or NaN for water surfaces

# Notes
Values from Table 3 of Zhang et al. (2001).
"""
function get_characteristic_radius(land_use::LandUseClass, season::SeasonCategory)
    if land_use == WATER
        return NaN  # No vegetation over water
    elseif land_use in (CONIFEROUS_FOREST, MIXED_FOREST)
        return 5.0  # Evergreen needleleaf (high deposition)
    elseif land_use == DECIDUOUS_FOREST
        if season == WINTER
            return 10.0  # Bare branches
        else
            return 2.0  # Full leaves
        end
    elseif land_use in (GRASSLAND, CROPLAND)
        if season in (AUTUMN, WINTER)
            return 10.0  # Senescent/dormant
        else
            return 2.0  # Growing season
        end
    elseif land_use == URBAN
        return 10.0  # Urban roughness elements
    else  # BARE_SOIL, SNOW_ICE
        return 50.0  # Large effective radius (low interception)
    end
end

"""
    surface_resistance(land_use::LandUseClass, season::SeasonCategory,
                      particle_diameter::Real, u_star::Real, diffusivity::Real,
                      settling_velocity::Real; scheme::DepositionScheme=Zhang2001)

Compute surface resistance Rs for particle deposition.

# Arguments
- `land_use`: Land use classification
- `season`: Seasonal category
- `particle_diameter`: Particle diameter (m)
- `u_star`: Friction velocity (m/s)
- `diffusivity`: Brownian diffusivity (m²/s)
- `settling_velocity`: Gravitational settling velocity (m/s)
- `scheme`: Deposition scheme (Zhang2001 or Emerson2020, default: Zhang2001)

# Returns
- Surface resistance Rs (s/m)

# Formula
Rs = 1 / (ε₀ × u* × (EB + EIM + EIN) × R₁)

where:
- ε₀ = 3.0 (empirical constant)
- EB = Brownian diffusion efficiency
- EIM = Impaction efficiency
- EIN = Interception efficiency
- R₁ = rebound correction (1 - exp(-√St))

# Schemes
- Zhang2001 (default):
  - EB = Sc^(-0.54)
  - EIM = (St/(0.8+St))²
  - EIN = 0.5(d/A)²

- Emerson2020:
  - EB = 0.2×Sc^(-2/3)
  - EIM = 0.4×(St/(0.8+St))^1.7
  - EIN = 2.5×(d/A)^0.8
"""
function surface_resistance(land_use::LandUseClass, season::SeasonCategory,
                           particle_diameter::T, u_star::T,
                           diffusivity::T, settling_velocity::T;
                           scheme::DepositionScheme=Zhang2001) where T<:Real
    if u_star < 1e-10
        return T(1e10)  # Very large resistance for calm conditions
    end

    # Get characteristic radius A (mm -> m)
    A = get_characteristic_radius(land_use, season) * 1e-3

    # Schmidt number
    schmidt = NU_AIR / diffusivity

    if isnan(A)  # Water surface - use simplified model (no vegetation)
        # For water, use simplified model based on Schmidt number
        return 1 / (u_star * schmidt^(-2/3))
    end

    # Stokes number for impaction
    stokes = settling_velocity * u_star^2 / (G_GRAVITY * NU_AIR)

    # Collection efficiencies depend on scheme
    if scheme == Zhang2001
        # Zhang et al. (2001) - Brownian diffusion efficiency
        EB = schmidt^(-0.54)

        # Impaction efficiency
        if stokes > 0.0
            EIM = (stokes / (0.8 + stokes))^2
        else
            EIM = zero(T)
        end

        # Interception efficiency
        if A > 0.0
            EIN = 0.5 * (particle_diameter / A)^2
        else
            EIN = zero(T)
        end

    else  # Emerson2020
        # Emerson et al. (2020) - Brownian diffusion efficiency
        EB = 0.2 * schmidt^(-2/3)

        # Impaction efficiency
        if stokes > 0.0
            EIM = 0.4 * (stokes / (0.8 + stokes))^1.7
        else
            EIM = zero(T)
        end

        # Interception efficiency
        if A > 0.0
            EIN = 2.5 * (particle_diameter / A)^0.8
        else
            EIN = zero(T)
        end
    end

    # Total collection efficiency
    E_total = EB + EIM + EIN

    if E_total < 1e-10
        return T(1e10)  # Very large resistance
    end

    # Rebound correction factor R₁
    # Particles may bounce off surfaces rather than sticking
    R1 = if stokes > 0.0
        1 - exp(-sqrt(stokes))
    else
        T(0.5)  # Default for very small particles
    end

    # Empirical constant ε₀ = 3.0 (both schemes)
    epsilon_0 = T(3.0)

    # Surface resistance
    return 1 / (epsilon_0 * u_star * E_total * R1)
end

"""
    compute_dry_deposition_velocity(params::DryDepositionParams,
                                    u_star::Real, L::Real,
                                    temperature::Real, pressure::Real,
                                    season::SeasonCategory,
                                    settling_velocity::Real)

Compute dry deposition velocity using resistance analogy (Emerson et al. 2020).

# Arguments
- `params`: Dry deposition parameters (grid-dependent)
- `u_star`: Friction velocity (m/s)
- `L`: Monin-Obukhov length (m)
- `temperature`: Air temperature (K)
- `pressure`: Surface pressure (Pa)
- `season`: Current season
- `settling_velocity`: Gravitational settling velocity (m/s)

# Returns
- Matrix of deposition velocities vd (m/s) for each grid cell

# Formula
vd = vg + 1 / (Ra + Rs)

where:
- vg = gravitational settling velocity
- Ra = aerodynamic resistance
- Rs = surface resistance
"""
function compute_dry_deposition_velocity(params::DryDepositionParams{T},
                                        u_star::Matrix{T}, L::Matrix{T},
                                        temperature::Matrix{T}, pressure::Matrix{T},
                                        season::SeasonCategory,
                                        settling_velocity::T) where T<:Real
    nx, ny = size(params.roughness_length)
    @assert size(u_star) == (nx, ny)
    @assert size(L) == (nx, ny)
    @assert size(temperature) == (nx, ny)
    @assert size(pressure) == (nx, ny)

    vd = zeros(T, nx, ny)

    for j in 1:ny, i in 1:nx
        # Air density
        rho_air = compute_air_density(pressure[i,j], temperature[i,j])

        # Brownian diffusivity
        D = brownian_diffusivity(params.particle_diameter, temperature[i,j], rho_air)

        # Aerodynamic resistance
        Ra = aerodynamic_resistance(L[i,j], u_star[i,j],
                                    params.roughness_length[i,j],
                                    params.reference_height)

        # Surface resistance
        Rs = surface_resistance(params.land_use[i,j], season,
                               params.particle_diameter, u_star[i,j],
                               D, settling_velocity)

        # Total deposition velocity (resistance in series)
        vd[i,j] = settling_velocity + 1 / (Ra + Rs)
    end

    return vd
end

"""
    compute_wet_scavenging_coefficient(precipitation_rate::Real,
                                       params::WetDepositionParams,
                                       is_in_cloud::Bool=false)

Compute wet deposition scavenging coefficient (Bartnicki 2011).

# Arguments
- `precipitation_rate`: Precipitation rate (mm/h)
- `params`: Wet deposition parameters
- `is_in_cloud`: Flag for in-cloud (rainout) vs below-cloud (washout) scavenging

# Returns
- Scavenging coefficient Λ (s⁻¹)

# Formula
Below-cloud (washout): Λ = A × P^B
In-cloud (rainout): Λ = constant

where P is precipitation rate in mm/h.
"""
function compute_wet_scavenging_coefficient(precipitation_rate::T,
                                           params::WetDepositionParams{T},
                                           is_in_cloud::Bool=false) where T<:Real
    # Check threshold
    if precipitation_rate < params.precipitation_threshold
        return zero(T)
    end

    if is_in_cloud
        # In-cloud scavenging (rainout)
        # Constant coefficient for particles incorporated into cloud droplets
        return params.rainout_coefficient
    else
        # Below-cloud scavenging (washout)
        # Power law dependence on precipitation rate
        # Λ = A × P^B where B ≈ 0.79 (empirical)
        B = T(0.79)
        return params.washout_coefficient * precipitation_rate^B
    end
end

"""
    apply_dry_deposition!(particle_mass::AbstractVector,
                         deposition_velocity::Real,
                         grid_cell_area::Real,
                         dt::Real)

Apply dry deposition to particle masses.

# Arguments
- `particle_mass`: Vector of particle component masses (modified in-place)
- `deposition_velocity`: Deposition velocity at particle location (m/s)
- `grid_cell_area`: Area of grid cell (m²)
- `dt`: Time step (s)

# Notes
Deposition is proportional to concentration near the surface:
dM/dt = -vd × (M / h) where h is mixing height.

For simplicity, we use: dM/dt = -vd × M / dt
This gives: M_new = M × exp(-vd × dt / h_mix)

With h_mix = typical boundary layer height ≈ 1000 m.
"""
function apply_dry_deposition!(particle_mass::AbstractVector{T},
                              deposition_velocity::T,
                              dt::T,
                              h_mix::T=T(1000.0)) where T<:Real
    if deposition_velocity <= 0.0 || dt <= 0.0
        return zero(T)  # No deposition
    end

    # Deposition rate coefficient (1/s)
    # k = vd / h_mix
    k_dep = deposition_velocity / h_mix

    # Exponential decay: M(t+dt) = M(t) × exp(-k × dt)
    decay_factor = exp(-k_dep * dt)

    # Amount deposited (for tracking)
    deposited = zero(T)

    for i in eachindex(particle_mass)
        mass_lost = particle_mass[i] * (1 - decay_factor)
        particle_mass[i] -= mass_lost
        deposited += mass_lost
    end

    return deposited
end

"""
    apply_wet_deposition!(particle_mass::AbstractVector,
                         scavenging_coefficient::Real,
                         dt::Real)

Apply wet deposition (scavenging) to particle masses.

# Arguments
- `particle_mass`: Vector of particle component masses (modified in-place)
- `scavenging_coefficient`: Scavenging coefficient Λ (s⁻¹)
- `dt`: Time step (s)

# Formula
M(t+dt) = M(t) × exp(-Λ × dt)

# Returns
- Total mass deposited
"""
function apply_wet_deposition!(particle_mass::AbstractVector{T},
                               scavenging_coefficient::T,
                               dt::T) where T<:Real
    if scavenging_coefficient <= 0.0 || dt <= 0.0
        return zero(T)  # No deposition
    end

    # Exponential scavenging
    decay_factor = exp(-scavenging_coefficient * dt)

    deposited = zero(T)

    for i in eachindex(particle_mass)
        mass_lost = particle_mass[i] * (1 - decay_factor)
        particle_mass[i] -= mass_lost
        deposited += mass_lost
    end

    return deposited
end

# =============================================================================
# BARTNICKI WET DEPOSITION SCHEME (Bartnicki, 2003)
# =============================================================================

"""
    wet_deposition_constant(radius_μm::Real)

Compute the wet deposition constant for a given particle radius.

# Arguments
- `radius_μm`: Particle radius in micrometers

# Returns
- Deposition constant (dimensionless)

# Formula (Bartnicki 2003)
depconst = b0 + b1*r + b2*r² + b3*r³
"""
function wet_deposition_constant(radius_μm::T) where T<:Real
    b0 = T(-0.1483)
    b1 = T(0.3220133)
    b2 = T(-3.0062e-2)
    b3 = T(9.34458e-4)
    r = radius_μm
    return b0 + b1*r + b2*r*r + b3*r*r*r
end

"""
    wet_deposition_rate_bartnicki(radius_μm::Real, precip_mmh::Real, depconst::Real;
                                   use_convective::Bool=true)

Compute instantaneous wet scavenging rate using Bartnicki (2003) scheme.

# Arguments
- `radius_μm`: Particle radius in micrometers
- `precip_mmh`: Precipitation intensity in mm/hour
- `depconst`: Precomputed deposition constant from wet_deposition_constant()
- `use_convective`: Apply convective enhancement for heavy rain (default: true)

# Returns
- Scavenging rate rkw (s⁻¹)

# Size-dependent formulas
- Gas (r ≤ 0.1 μm): rkw = 1.12e-4 × q^0.79
- Small (0.05 < r ≤ 1.4 μm): rkw = 8.4e-5 × q^0.79
- Medium (1.4 < r ≤ 10 μm): rkw = depconst × (2.7e-4×q - 3.618e-6×q²)
- Large (r > 10 μm): rkw = 2.7e-4×q - 3.618e-6×q²
- Convective (q > 7 mm/h): rkw = 3.36e-4 × q^0.79
"""
function wet_deposition_rate_bartnicki(radius_μm::T, precip_mmh::T, depconst::T;
                                        use_convective::Bool=true) where T<:Real
    a0 = T(8.4e-5)
    a1 = T(2.7e-4)
    a2 = T(-3.618e-6)

    q = precip_mmh
    r = radius_μm

    rkw = zero(T)

    # Size-dependent scavenging (order matters)
    if r > 0.05 && r <= 1.4
        rkw = a0 * q^T(0.79)
    end
    if r > 1.4 && r <= 10.0
        rkw = depconst * (a1*q + a2*q*q)
    end
    if r > 10.0
        rkw = a1*q + a2*q*q
    end

    # Convective enhancement for heavy rain
    if use_convective && q > 7.0
        rkw = T(3.36e-4) * q^T(0.79)
    end

    # Gas-phase scavenging (overrides particle scavenging for very small)
    if r <= 0.1
        rkw = T(1.12e-4) * q^T(0.79)
    end

    return max(rkw, zero(T))
end

"""
    bartnicki_wet_deposition_rate(radius_μm::Real, precip_mmh::Real, tstep::Real;
                                   use_convective::Bool=true)

Compute wet deposition rate (fraction deposited per timestep) using Bartnicki scheme.

# Arguments
- `radius_μm`: Particle radius in micrometers
- `precip_mmh`: Precipitation intensity in mm/hour
- `tstep`: Timestep in seconds
- `use_convective`: Apply convective enhancement (default: true)

# Returns
- Deposition rate: fraction of mass removed per timestep [0, 1]

# Formula
deprate = 1 - exp(-tstep × rkw)
"""
function bartnicki_wet_deposition_rate(radius_μm::T, precip_mmh::T, tstep::T;
                                        use_convective::Bool=true) where T<:Real
    depconst = wet_deposition_constant(radius_μm)
    rkw = wet_deposition_rate_bartnicki(radius_μm, precip_mmh, depconst; use_convective)
    return 1 - exp(-tstep * rkw)
end

# Minimum precipitation threshold for wet deposition (mm/h)
const WETDEP_PRECMIN = 0.01

# Minimum sigma level for wet deposition (approx 550 hPa)
# Particles above this level (lower sigma) don't experience wet deposition
const WETDEP_SIGMA_MIN = 0.67

# Export public API
export LandUseClass, SeasonCategory, DepositionScheme
export WATER, DECIDUOUS_FOREST, CONIFEROUS_FOREST, MIXED_FOREST
export GRASSLAND, CROPLAND, URBAN, BARE_SOIL, SNOW_ICE
export WINTER, SPRING, SUMMER, AUTUMN
export Zhang2001, Emerson2020
export DryDepositionParams, WetDepositionParams
export compute_air_density, compute_friction_velocity, compute_monin_obukhov_length
export aerodynamic_resistance, cunningham_slip_factor, brownian_diffusivity
export get_characteristic_radius, surface_resistance
export compute_dry_deposition_velocity, compute_wet_scavenging_coefficient
export apply_dry_deposition!, apply_wet_deposition!
export wet_deposition_constant, wet_deposition_rate_bartnicki, bartnicki_wet_deposition_rate
export WETDEP_PRECMIN, WETDEP_SIGMA_MIN
