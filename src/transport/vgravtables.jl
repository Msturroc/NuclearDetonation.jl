# ==============================================================================
# GRAVITATIONAL SETTLING VELOCITY
#
# This file contains functions to calculate the gravitational settling velocity
# of particles, mirroring the logic in SNAP's `vgravtables.f90`.
#
# It supports two underlying physics models for air viscosity, which can be
# selected at runtime using dispatch:
#
# - `SutherlandViscosity`: A more modern, physically accurate model based on
#   Sutherland's formula. This is the default.
# - `FortranViscosity`: A simplified model that exactly replicates the
#   formulas used in the original Fortran SNAP code, used for validation.
#
# Key functions:
# - `vgrav_corrected`: Iteratively corrects Stokes velocity for larger Reynolds numbers.
# - `build_vgrav_tables`: Pre-computes settling velocities for different particle sizes.
# ==============================================================================

module GravitationalSettling

# Define constants locally to resolve scope issue, as requested.
const G_GRAVITY_M_S2 = 9.80665
const R_SPECIFIC_J_KG_K = 287.058

export build_vgrav_tables, vgrav_corrected, cunningham_factor, VGravTables
export AbstractViscosityModel, FortranViscosity, SutherlandViscosity

"""
    VGravTables

Type alias for the dictionary holding pre-computed gravitational settling
velocity tables. Maps a particle size index to a 2D lookup table of
`vgrav(pressure, temperature)`.
"""
const VGravTables = Dict{Int, Array{Float64, 2}}


# Constants for air viscosity and density calculations
const G_GRAVITY_CM_S2 = G_GRAVITY_M_S2 * 100.0

# ==============================================================================
# Selectable Physics Models for Air Properties
# ==============================================================================

"""
Abstract base type for viscosity models.
"""
abstract type AbstractViscosityModel end

"""
    SutherlandViscosity()

Selects the modern, physically-based Sutherland's formula for air viscosity.
This is the default model for new simulations.
"""
struct SutherlandViscosity <: AbstractViscosityModel end

"""
    FortranViscosity()

Selects the simplified air viscosity and density formulas used in the original
Fortran SNAP code. Use this model to validate against the reference implementation.
"""
struct FortranViscosity <: AbstractViscosityModel end


# ==============================================================================
# Physical Properties of Air (Dispatched by Model)
#
# Note: All internal calculations are done in CGS units (cm, g, s) to align
# with the Fortran implementation. These functions return values in CGS units.
# ==============================================================================

"""
    air_viscosity(T, ::FortranViscosity)

Calculate air viscosity using the simplified Fortran formula.

# Arguments
- `T`: Temperature (Kelvin)

# Returns
- Dynamic viscosity in Poise (g/cm·s)
"""
function air_viscosity(T::Real, ::FortranViscosity)
    # Fortran visc(t) = 1.72e-4*(393.0/(t+120.0))*(t/273.0)**1.5
    # This formula directly returns Poise (g/cm-s).
    return 1.72e-4 * (393.0 / (T + 120.0)) * (T / 273.0)^1.5
end

"""
    air_viscosity(T, ::SutherlandViscosity)

Calculate air viscosity using Sutherland's formula.

# Arguments
- `T`: Temperature (Kelvin)

# Returns
- Dynamic viscosity in Poise (g/cm·s)
"""
function air_viscosity(T::Real, ::SutherlandViscosity)
    # Sutherland's formula for dynamic viscosity in Pa·s (kg/m·s)
    μ0 = 1.716e-5  # Reference viscosity at T0 (Pa·s)
    T0 = 273.15    # Reference temperature (K)
    S = 110.4      # Sutherland's constant (K)
    μ_pas = μ0 * (T / T0)^1.5 * (T0 + S) / (T + S)
    # Convert from Pa·s (kg/m·s) to Poise (g/cm·s)
    return μ_pas * 10.0
end


"""
    air_density(P_hpa, T, ::FortranViscosity)

Calculate air density using the simplified Fortran formula.

# Arguments
- `P_hpa`: Pressure (hPa)
- `T`: Temperature (Kelvin)

# Returns
- Air density (g/cm³)
"""
function air_density(P_hpa::Real, T::Real, ::FortranViscosity)
    # Fortran roa(p,t) = 0.001*p*100.0/(r*t)
    # This formula takes hPa and returns g/cm³.
    return 0.001 * P_hpa * 100.0 / (R_SPECIFIC_J_KG_K * T)
end

"""
    air_density(P_pa, T, ::SutherlandViscosity)

Calculate air density using the ideal gas law.

# Arguments
- `P_pa`: Pressure (Pascals)
- `T`: Temperature (Kelvin)

# Returns
- Air density (g/cm³)
"""
function air_density(P_pa::Real, T::Real, ::SutherlandViscosity)
    # Standard ideal gas law returns density in kg/m³
    ρ_kg_m3 = P_pa / (R_SPECIFIC_J_KG_K * T)
    # Convert from kg/m³ to g/cm³
    return ρ_kg_m3 * 0.001
end


# ==============================================================================
# Particle Settling Velocity Calculations
# ==============================================================================

"""
    cunningham_factor(dp)

Calculate the Cunningham slip correction factor. This is essential for small
particles where the mean free path of air is comparable to the particle size.
The formula is taken directly from the Fortran `slipf` function.

# Arguments
- `dp`: Particle diameter (μm)

# Returns
- Cunningham slip correction factor (dimensionless)
"""
function cunningham_factor(dp::Real)
    # This is a direct port of the `slipf` function in `vgravtables.f90`
    # The formula is valid for dp in micrometers.
    if dp <= 0.0
        return 1.0
    end
    # Use the more precise lambda from the test suite
    lam_um = 0.0653 # Mean free path in μm
    
    # Empirical constants from the Fortran code
    A = 1.257
    B = 0.4
    C_cunn = 0.55
    
    # Corrected formula matching the test script and standard implementations
    return 1.0 + (2.0 * lam_um / dp) * (A + B * exp(-C_cunn * dp / lam_um))
end

"""
    _vgrav_stokes(dp, rp, P, T, model)

Internal function to calculate Stokes velocity using the selected physics model.

# Arguments
- `dp`: Particle diameter (μm)
- `rp`: Particle density (g/cm³)
- `P`: Ambient pressure (Pascals)
- `T`: Ambient temperature (Kelvin)
- `model`: The selected viscosity model (`FortranViscosity` or `SutherlandViscosity`)

# Returns
- Settling velocity (cm/s)
"""
function _vgrav_stokes(dp::Real, rp::Real, P::Real, T::Real, model::FortranViscosity)
    p_hpa = P / 100.0
    ρa = air_density(p_hpa, T, model) # Expects hPa, returns g/cm³
    η = air_viscosity(T, model)       # Returns g/cm·s
    C = cunningham_factor(dp)
    dp_cm = dp * 1.0e-4
    # Stokes' law in CGS units
    return dp_cm^2 * G_GRAVITY_CM_S2 * (rp - ρa) * C / (18.0 * η)
end

function _vgrav_stokes(dp::Real, rp::Real, P::Real, T::Real, model::SutherlandViscosity)
    ρa = air_density(P, T, model) # Expects Pa, returns g/cm³
    η = air_viscosity(T, model)   # Returns g/cm·s
    C = cunningham_factor(dp)
    dp_cm = dp * 1.0e-4
    # Stokes' law in CGS units
    return dp_cm^2 * G_GRAVITY_CM_S2 * (rp - ρa) * C / (18.0 * η)
end


"""
    vgrav_corrected(dp, rp, P, T, ρp, model; tol=0.001)

Calculates fall speed, corrected for larger Reynolds numbers. This is a direct
port of the `iter` subroutine in `vgravtables.f90`, using a bisection method.

# Arguments
- `dp`: Particle diameter (μm)
- `rp`: Particle density (g/cm³)
- `P`: Ambient pressure (Pascals)
- `T`: Ambient temperature (Kelvin)
- `ρp`: Particle density (kg/m³) - Note: redundant, `rp` is used.
- `model`: The viscosity model to use (`FortranViscosity()` or `SutherlandViscosity()`).
- `tol`: Relative tolerance for the bisection solver.

# Returns
- Corrected settling velocity (m/s)
"""
function vgrav_corrected(dp::Real, rp::Real, P::Real, T::Real, ρp::Real, model::AbstractViscosityModel; tol=0.001)
    # The `fit` function from Fortran, which is the target for the root-finding
    function fit(u, u0, dp, ρa, η)
        A1 = 0.15
        A2 = 0.687
        dp_cm = dp * 1.0e-4
        re = u * dp_cm * ρa / η
        return u * (1.0 + A1 * re^A2) - u0
    end

    # Initial Stokes velocity (u0) in cm/s, using the selected physics model.
    u0 = _vgrav_stokes(dp, rp, P, T, model)

    # Bisection method from Fortran `iter`
    epsilon = tol * u0
    if abs(u0) < 1e-12 # Handle zero or near-zero stokes velocity
        return 0.0
    end

    # Get air properties in CGS units using the selected model
    η = air_viscosity(T, model)
    ρa = model isa FortranViscosity ? air_density(P / 100.0, T, model) : air_density(P, T, model)

    # Bisection solver setup
    x1 = 0.0
    x2 = 2.0 * u0
    y1 = fit(x1, u0, dp, ρa, η)
    y2 = fit(x2, u0, dp, ρa, η)

    # If root is not bracketed (e.g., u0 is very small), return u0 as best guess.
    if y1 * y2 > 0
        return u0 / 100.0 # Convert cm/s to m/s
    end

    # Bisection loop
    for i in 1:20 # Max 20 iterations
        x_mid = (x1 + x2) / 2.0
        y_mid = fit(x_mid, u0, dp, ρa, η)

        if abs(x2 - x1) < epsilon
            return x_mid / 100.0 # Convert cm/s to m/s
        end

        if y_mid * y1 < 0
            x2 = x_mid
            y2 = y_mid
        else
            x1 = x_mid
            y1 = y_mid
        end
    end

    # If the loop finishes, return the best guess
    return (x1 + x2) / 2.0 / 100.0 # Convert cm/s to m/s
end


"""
    build_vgrav_tables(particle_properties_list; model=SutherlandViscosity())

Pre-computes settling velocities for a list of particle sizes. This is
analogous to the `main` program in `vgravtables.f90`.

# Arguments
- `particle_properties_list`: A list of `ParticleProperties` objects.
- `model`: The viscosity model to use. Defaults to `SutherlandViscosity()`.
           Pass `FortranViscosity()` to match the reference SNAP code.

# Returns
- A dictionary mapping particle size index to a 2D lookup table `vgrav(pressure, temperature)`.
"""
function build_vgrav_tables(particle_properties_list; model::AbstractViscosityModel=SutherlandViscosity())
    vgrav_tables = Dict{Int, Array{Float64, 2}}()

    for (i, props) in enumerate(particle_properties_list)
        dp = props.diameter_μm
        ρp_kg_m3 = props.density_gcm3 * 1000.0
        rp_g_cm3 = props.density_gcm3

        vgrav_table = zeros(Float64, NUMPRES_VG, NUMTEMP_VG)

        for ip in 1:NUMPRES_VG
            p_hpa = PBASE_VG + ip * PINCR_VG
            if p_hpa < 1.0
                p_hpa = 1.0
            end
            p_pa = p_hpa * 100.0

            for it in 1:NUMTEMP_VG
                T_k = TBASE_VG + it * TINCR_VG
                vgrav_table[ip, it] = vgrav_corrected(dp, rp_g_cm3, p_pa, T_k, ρp_kg_m3, model)
            end
        end
        vgrav_tables[i] = vgrav_table
    end

    return vgrav_tables
end

# Constants for interpolation grid, derived from Fortran vgravtables.f90
const NUMTEMP_VG = 41
const NUMPRES_VG = 25
const TINCR_VG = 200.0 / (NUMTEMP_VG - 1)
const TBASE_VG = 273.0 - 120.0 - TINCR_VG
const PINCR_VG = 1200.0 / (NUMPRES_VG - 1)
const PBASE_VG = 0.0 - PINCR_VG

"""
    interpolate_vgrav(vgrav_tables::VGravTables, size_idx::Int, P_hpa::Real, T_k::Real) -> Float64

Interpolates the pre-computed gravitational settling velocity from `vgrav_tables`
for a given particle size index, pressure, and temperature, precisely mirroring
the bilinear interpolation logic from Fortran's `forwrd.F90`.

# Arguments
- `vgrav_tables`: The pre-computed settling velocity tables.
- `size_idx`: The index of the particle size bin (Fortran's mrunning).
- `P_hpa`: Pressure in hPa.
- `T_k`: Temperature in Kelvin.

# Returns
- `Float64`: Interpolated settling velocity in m/s.
"""
function interpolate_vgrav(vgrav_tables::VGravTables, size_idx::Int, P_hpa::Real, T_k::Real)::Float64
    # Fortran vgtable is (temperature, pressure, component)
    # Julia vgrav_tables[size_idx] is (pressure, temperature)
    # So we need to transpose or adjust indexing.
    # Fortran: vgtable(it,ip,mrunning)
    # Julia: table[ip, it]
    table = vgrav_tables[size_idx]

    # Calculate indices and interpolation factors exactly as in Fortran forwrd.F90
    ip_float = (P_hpa - PBASE_VG) / PINCR_VG
    ip = floor(Int, ip_float)
    ip = max(ip, 1)
    ip = min(NUMPRES_VG - 1, ip) # Fortran uses size(vgtable,2)-1

    it_float = (T_k - TBASE_VG) / TINCR_VG
    it = floor(Int, it_float)
    it = max(it, 1)
    it = min(NUMTEMP_VG - 1, it) # Fortran uses size(vgtable,1)-1

    # Get interpolation values
    p_interp_factor = (P_hpa - (PBASE_VG + ip * PINCR_VG)) / PINCR_VG
    t_interp_factor = (T_k - (TBASE_VG + it * TINCR_VG)) / TINCR_VG

    # Clamp interpolation factors to [0, 1] to handle edge cases
    p_interp_factor = clamp(p_interp_factor, 0.0, 1.0)
    t_interp_factor = clamp(t_interp_factor, 0.0, 1.0)

    # Fortran indexing: vgtable(it,ip,mrunning)
    # Julia table[ip, it]
    # Note: Fortran uses 1-based indexing, so ip and it are already correct for Julia array access

    # Interpolate in temperature direction first
    grav_p1_t = table[ip, it] * (1.0 - t_interp_factor) + table[ip, it + 1] * t_interp_factor
    grav_p2_t = table[ip + 1, it] * (1.0 - t_interp_factor) + table[ip + 1, it + 1] * t_interp_factor

    # Then interpolate in pressure direction
    gravity = grav_p1_t * (1.0 - p_interp_factor) + grav_p2_t * p_interp_factor

    return gravity
end

end # module GravitationalSettling
