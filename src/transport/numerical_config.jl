# Numerical Configuration for SNAP Modernization
# Allows toggling between validation mode (Fortran-matching) and modern mode

using Interpolations
using OrdinaryDiffEq

export NumericalConfig, ValidationMode, ModernMode, InterpolationOrder
export ERA5NumericalConfig, ERA5ValidationMode, ERA5ModernMode
export TurbulenceModel, RandomWalk, OrnsteinUhlenbeck, HannaTurbulence
export create_numerical_config, get_ode_solver, get_interpolation_scheme, get_solve_kwargs
export LinearInterp, CubicInterp, FortranInterp

"""
    InterpolationOrder

Enumeration of supported interpolation orders for wind fields.

# Options
- `LinearInterp`: Linear interpolation via Interpolations.jl (~1-2% error)
- `CubicInterp`: Cubic spline interpolation (~0.1-0.3% error, smoother derivatives)
- `FortranInterp`: Manual floor-based trilinear interpolation (exact Fortran SNAP match)
"""
@enum InterpolationOrder begin
    LinearInterp = 1
    CubicInterp = 3
    FortranInterp = 0  # For validation - exact Fortran floor() based bilinear
end

"""
    TurbulenceModel

Enumeration of supported turbulence parameterisations for ERA5 simulations.

# Options
- `RandomWalk`: Simple uncorrelated random walk (Fortran SNAP baseline)
- `OrnsteinUhlenbeck`: Temporally correlated turbulence via O-U process
- `HannaTurbulence`: Hanna (1982) height-dependent parameterisation with O-U
"""
@enum TurbulenceModel begin
    RandomWalk = 0        # Simple random walk (baseline, matches Fortran)
    OrnsteinUhlenbeck = 1 # O-U process with temporal correlation
    HannaTurbulence = 2   # Full Hanna (1982) with stability dependence
end

"""
    NumericalConfig{T<:Real}

Configuration for numerical methods used in particle integration.

# Fields
- `interpolation_order::InterpolationOrder`: Wind field interpolation order
- `ode_solver_type::Symbol`: ODE solver type (:Euler, :Tsit5, :AutoTsit5)
- `fixed_dt::Union{T, Nothing}`: Fixed timestep (seconds), nothing for adaptive
- `reltol::T`: Relative tolerance for adaptive solvers
- `abstol::T`: Absolute tolerance for adaptive solvers
- `name::String`: Configuration name for output files

# Solver Types
- `:Euler`: Forward Euler with fixed dt (matches Fortran exactly)
- `:Tsit5`: Tsitouras 5th order Runge-Kutta with fixed dt
- `:AutoTsit5`: Tsit5 with adaptive timestepping

# Examples

```julia
# Validation mode - matches Fortran SNAP exactly
config = NumericalConfig(
    interpolation_order = LinearInterp,
    ode_solver_type = :Euler,
    fixed_dt = 300.0,
    name = "validation"
)

# Modern mode - improved accuracy
config = NumericalConfig(
    interpolation_order = CubicInterp,
    ode_solver_type = :AutoTsit5,
    fixed_dt = nothing,
    reltol = 1e-5,
    abstol = 1e-7,
    name = "modern"
)
```
"""
@kwdef struct NumericalConfig{T<:Real}
    interpolation_order::InterpolationOrder = LinearInterp
    ode_solver_type::Symbol = :Euler
    fixed_dt::Union{T, Nothing} = 300.0
    reltol::T = 1e-6
    abstol::T = 1e-8
    name::String = "default"
end

"""
    ERA5NumericalConfig{T<:Real}

Configuration for numerical methods used in ERA5 particle integration.
Extends NumericalConfig with ERA5-specific options for turbulence.

# Fields
## ODE Solver (same as NumericalConfig)
- `interpolation_order::InterpolationOrder`: Wind field interpolation order
- `ode_solver_type::Symbol`: ODE solver type (:Euler, :Tsit5, :AutoTsit5)
- `fixed_dt::Union{T, Nothing}`: Fixed timestep (seconds), nothing for adaptive
- `reltol::T`: Relative tolerance for adaptive solvers
- `abstol::T`: Absolute tolerance for adaptive solvers

## ERA5-Specific Options
- `turbulence::TurbulenceModel`: Turbulence parameterisation (default: RandomWalk)
- `store_turbulent_velocities::Bool`: Store u'/v'/w' for O-U process (default: false)
- `name::String`: Configuration name for output files

# Solver Types
- `:Euler`: Forward Euler with fixed dt (matches Fortran exactly)
- `:Tsit5`: Tsitouras 5th order Runge-Kutta with fixed dt
- `:AutoTsit5`: Tsit5 with adaptive timestepping

# Turbulence Models
- `RandomWalk`: Simple uncorrelated random walk (Fortran SNAP baseline)
- `OrnsteinUhlenbeck`: Temporally correlated turbulence via O-U process
- `HannaTurbulence`: Hanna (1982) height-dependent with O-U

# Examples

```julia
# Baseline mode - matches Fortran SNAP exactly (DEFAULT)
config = ERA5NumericalConfig(
    interpolation_order = LinearInterp,
    ode_solver_type = :Euler,
    fixed_dt = 300.0,
    turbulence = RandomWalk,
    name = "baseline"
)

# Enhanced mode - improved accuracy
config = ERA5NumericalConfig(
    interpolation_order = CubicInterp,
    ode_solver_type = :Tsit5,
    fixed_dt = 300.0,
    turbulence = OrnsteinUhlenbeck,
    store_turbulent_velocities = true,
    name = "enhanced"
)
```
"""
@kwdef struct ERA5NumericalConfig{T<:Real}
    # ODE Solver selection (mirrors NumericalConfig)
    interpolation_order::InterpolationOrder = LinearInterp
    ode_solver_type::Symbol = :Euler
    fixed_dt::Union{T, Nothing} = 300.0
    reltol::T = 1e-6
    abstol::T = 1e-8

    # ERA5-specific: Turbulence model selection
    turbulence::TurbulenceModel = OrnsteinUhlenbeck
    store_turbulent_velocities::Bool = true

    # Configuration name
    name::String = "era5_default"
end

"""
    ValidationMode(dt::T=300.0; name="validation", use_fortran_interp=true) where T<:Real

Create a NumericalConfig that exactly matches Fortran SNAP.

- FortranInterp: Manual floor-based trilinear interpolation (default for validation)
- LinearInterp: Interpolations.jl linear (available if use_fortran_interp=false)
- Forward Euler integration
- Fixed timestep (default 300s)
"""
function ValidationMode(dt::T=300.0; name="validation", use_fortran_interp::Bool=true) where T<:Real
    return NumericalConfig{T}(
        interpolation_order = use_fortran_interp ? FortranInterp : LinearInterp,
        ode_solver_type = :Euler,
        fixed_dt = dt,
        name = name
    )
end

"""
    ModernMode(; interpolation=CubicInterp, solver=:AutoTsit5,
                 reltol=1e-5, abstol=1e-7, name="modern") where T<:Real

Create a NumericalConfig with modern numerical methods.

# Keyword Arguments
- `interpolation`: Interpolation order (LinearInterp or CubicInterp)
- `solver`: ODE solver (:Tsit5 or :AutoTsit5)
- `reltol`: Relative tolerance for adaptive timestepping
- `abstol`: Absolute tolerance for adaptive timestepping
- `fixed_dt`: Fixed timestep (nothing for adaptive)
- `name`: Configuration name
"""
function ModernMode(T::Type=Float64;
                    interpolation::InterpolationOrder=CubicInterp,
                    solver::Symbol=:AutoTsit5,
                    reltol::Real=1e-5,
                    abstol::Real=1e-7,
                    fixed_dt::Union{Real, Nothing}=nothing,
                    name::String="modern")
    return NumericalConfig{T}(
        interpolation_order = interpolation,
        ode_solver_type = solver,
        fixed_dt = fixed_dt === nothing ? nothing : T(fixed_dt),
        reltol = T(reltol),
        abstol = T(abstol),
        name = name
    )
end

# ============================================================================
# ERA5-Specific Configuration Helpers
# ============================================================================

"""
    ERA5ValidationMode(dt::T=300.0; name="era5_baseline") where T<:Real

Create an ERA5NumericalConfig that exactly matches Fortran SNAP.
This is the DEFAULT configuration for ERA5 simulations.

- Linear interpolation (Fortran-compatible)
- Forward Euler integration
- Fixed timestep (default 300s)
- Simple random walk turbulence
"""
function ERA5ValidationMode(dt::T=300.0; name::String="era5_baseline") where T<:Real
    return ERA5NumericalConfig{T}(
        interpolation_order = LinearInterp,
        ode_solver_type = :Euler,
        fixed_dt = dt,
        turbulence = RandomWalk,
        store_turbulent_velocities = false,
        name = name
    )
end

"""
    ERA5ModernMode(T::Type=Float64; kwargs...)

Create an ERA5NumericalConfig with modern numerical methods.

# Keyword Arguments
- `interpolation`: Interpolation order (default: CubicInterp)
- `solver`: ODE solver (default: :Tsit5)
- `fixed_dt`: Fixed timestep (default: 300.0)
- `reltol`: Relative tolerance for adaptive timestepping
- `abstol`: Absolute tolerance for adaptive timestepping
- `turbulence`: Turbulence model (default: OrnsteinUhlenbeck)
- `name`: Configuration name

# Examples
```julia
# Tsit5 with cubic interpolation and O-U turbulence
config = ERA5ModernMode(solver=:Tsit5, turbulence=OrnsteinUhlenbeck)

# Adaptive timestepping
config = ERA5ModernMode(solver=:AutoTsit5, fixed_dt=nothing)
```
"""
function ERA5ModernMode(T::Type=Float64;
                        interpolation::InterpolationOrder=CubicInterp,
                        solver::Symbol=:Tsit5,
                        fixed_dt::Union{Real, Nothing}=300.0,
                        reltol::Real=1e-6,
                        abstol::Real=1e-8,
                        turbulence::TurbulenceModel=OrnsteinUhlenbeck,
                        name::String="era5_modern")
    return ERA5NumericalConfig{T}(
        interpolation_order = interpolation,
        ode_solver_type = solver,
        fixed_dt = fixed_dt === nothing ? nothing : T(fixed_dt),
        reltol = T(reltol),
        abstol = T(abstol),
        turbulence = turbulence,
        store_turbulent_velocities = (turbulence != RandomWalk),
        name = name
    )
end

# ============================================================================
# ODE Solver and Interpolation Dispatch
# ============================================================================

"""
    get_ode_solver(config::NumericalConfig)

Get the ODE solver algorithm object for the given configuration.

# Returns
- ODE solver algorithm from OrdinaryDiffEq.jl
"""
function get_ode_solver(config::NumericalConfig)
    if config.ode_solver_type == :Euler
        return Euler()
    elseif config.ode_solver_type == :Tsit5 || config.ode_solver_type == :AutoTsit5
        return Tsit5()
    else
        error("Unknown ODE solver type: $(config.ode_solver_type)")
    end
end

"""
    get_interpolation_scheme(config::NumericalConfig)

Get the Interpolations.jl scheme for the given configuration.

# Returns
- BSpline(Linear()) or BSpline(Cubic(Line(OnGrid())))
"""
function get_interpolation_scheme(config::NumericalConfig)
    if config.interpolation_order == LinearInterp
        return BSpline(Linear())
    elseif config.interpolation_order == CubicInterp
        # Cubic spline with natural boundary conditions
        return BSpline(Cubic(Line(OnGrid())))
    else
        error("Unknown interpolation order: $(config.interpolation_order)")
    end
end

"""
    get_solve_kwargs(config::NumericalConfig)

Get keyword arguments for ODE solver based on configuration.

# Returns
- Dict with solver-specific kwargs
"""
function get_solve_kwargs(config::NumericalConfig)
    kwargs = Dict{Symbol, Any}()

    if config.ode_solver_type == :Euler
        # Fixed timestep Euler (matches Fortran)
        kwargs[:dt] = config.fixed_dt
        kwargs[:adaptive] = false
    elseif config.ode_solver_type == :Tsit5
        # Fixed timestep Tsit5
        kwargs[:dt] = config.fixed_dt
        kwargs[:adaptive] = false
    elseif config.ode_solver_type == :AutoTsit5
        # Adaptive Tsit5
        kwargs[:adaptive] = true
        kwargs[:reltol] = config.reltol
        kwargs[:abstol] = config.abstol
        # Set maximum timestep if fixed_dt is specified
        if config.fixed_dt !== nothing
            kwargs[:dtmax] = config.fixed_dt
        end
    end

    return kwargs
end

"""
    create_numerical_config(mode::Symbol; kwargs...)

Convenience function to create NumericalConfig from a mode symbol.

# Arguments
- `mode`: :validation or :modern

# Examples
```julia
config = create_numerical_config(:validation)
config = create_numerical_config(:modern, reltol=1e-6)
```
"""
function create_numerical_config(mode::Symbol=:validation; kwargs...)
    if mode == :validation
        return ValidationMode(; kwargs...)
    elseif mode == :modern
        return ModernMode(; kwargs...)
    else
        error("Unknown mode: $mode. Use :validation or :modern")
    end
end

# ============================================================================
# ERA5NumericalConfig Dispatch Methods
# ============================================================================

"""
    get_ode_solver(config::ERA5NumericalConfig)

Get the ODE solver algorithm object for ERA5 configuration.
"""
function get_ode_solver(config::ERA5NumericalConfig)
    if config.ode_solver_type == :Euler
        return Euler()
    elseif config.ode_solver_type == :Tsit5 || config.ode_solver_type == :AutoTsit5
        return Tsit5()
    else
        error("Unknown ODE solver type: $(config.ode_solver_type)")
    end
end

"""
    get_interpolation_scheme(config::ERA5NumericalConfig)

Get the Interpolations.jl scheme for ERA5 configuration.
"""
function get_interpolation_scheme(config::ERA5NumericalConfig)
    if config.interpolation_order == LinearInterp
        return BSpline(Linear())
    elseif config.interpolation_order == CubicInterp
        return BSpline(Cubic(Line(OnGrid())))
    else
        error("Unknown interpolation order: $(config.interpolation_order)")
    end
end

"""
    get_solve_kwargs(config::ERA5NumericalConfig)

Get keyword arguments for ODE solver based on ERA5 configuration.
"""
function get_solve_kwargs(config::ERA5NumericalConfig)
    kwargs = Dict{Symbol, Any}()

    if config.ode_solver_type == :Euler
        kwargs[:dt] = config.fixed_dt
        kwargs[:adaptive] = false
    elseif config.ode_solver_type == :Tsit5
        kwargs[:dt] = config.fixed_dt
        kwargs[:adaptive] = false
    elseif config.ode_solver_type == :AutoTsit5
        kwargs[:adaptive] = true
        kwargs[:reltol] = config.reltol
        kwargs[:abstol] = config.abstol
        if config.fixed_dt !== nothing
            kwargs[:dtmax] = config.fixed_dt
        end
    end

    return kwargs
end

"""
    create_era5_numerical_config(mode::Symbol=:baseline; kwargs...)

Convenience function to create ERA5NumericalConfig from a mode symbol.

# Arguments
- `mode`: :baseline (default, Fortran parity) or :modern (enhanced numerics)

# Examples
```julia
config = create_era5_numerical_config(:baseline)
config = create_era5_numerical_config(:modern, turbulence=HannaTurbulence)
```
"""
function create_era5_numerical_config(mode::Symbol=:baseline; kwargs...)
    if mode == :baseline || mode == :validation
        return ERA5ValidationMode(; kwargs...)
    elseif mode == :modern || mode == :enhanced
        return ERA5ModernMode(; kwargs...)
    else
        error("Unknown mode: $mode. Use :baseline or :modern")
    end
end

# Export the convenience function
export create_era5_numerical_config
