# SNAP: Severe Nuclear Accident Programme
# Julia port of particleML.f90
# Original Copyright (C) 1992-2023 Norwegian Meteorological Institute
#
# Defines particle data structures for Lagrangian transport

export Particle, ParticleProperties, ExtraParticle, scale_rad!, set_rad!, get_rad, add_rad!, get_set_rad!
export is_active, inactivate!, flush_away_denormal!

# Numerical limit for particle radioactive content to avoid subnormal floats
# (smallest possible normal Float32: ~1e-38)
const NUMERIC_LIMIT_RAD = Float32(1e-35)

"""
    ParticleProperties

Defines the physical properties of a particle size class.

# Fields
- `diameter_μm::Float64`: Particle diameter in micrometers.
- `density_gcm3::Float64`: Particle density in g/cm³.
"""
struct ParticleProperties
    diameter_μm::Float64
    density_gcm3::Float64

    # Keyword constructor
    ParticleProperties(; diameter_μm::Float64, density_gcm3::Float64) = new(diameter_μm, density_gcm3)
end


"""
    Particle

A Lagrangian particle for atmospheric dispersion tracking.

# Fields
- `x::Float64` - x position in grid coordinates
- `y::Float64` - y position in grid coordinates
- `z::Float64` - sigma/eta position (vertical coordinate in model levels)
- `tbl::Float32` - sigma/eta at top of boundary layer
- `rad::Vector{Float32}` - radioactive content per component (Bq), private field accessed via methods
- `hbl::Float32` - height of boundary layer (m)
- `grv::Float32` - gravitational settling velocity (m/s), fixed or computed
- `icomp::Int16` - index to the defined component (radionuclide type)
- `u_turb::Float32` - turbulent velocity in x direction (m/s), for Ornstein-Uhlenbeck process
- `v_turb::Float32` - turbulent velocity in y direction (m/s), for Ornstein-Uhlenbeck process
- `w_turb::Float32` - turbulent velocity in z direction (m/s), for Ornstein-Uhlenbeck process

# Notes
- Radioactive content is private to enforce proper handling via accessor methods
- Particles with rad <= 0 are considered inactive
- Subnormal values are flushed to zero to prevent performance degradation
- Turbulent velocities are used by Hanna turbulence scheme for temporal autocorrelation
"""
mutable struct Particle
    x::Float64
    y::Float64
    z::Float64
    tbl::Float32
    rad::Vector{Float32}  # Radioactive content per component in Bq
    hbl::Float32
    grv::Float32
    icomp::Int16
    u_turb::Float32  # Turbulent velocity components for O-U process
    v_turb::Float32
    w_turb::Float32

    # Constructor with default values for single component
    function Particle(x=0.0, y=0.0, z=0.0, tbl=0.0f0, rad=0.0f0, hbl=0.0f0, grv=0.0f0, icomp=Int16(0))
        new(x, y, z, tbl, [Float32(rad)], hbl, grv, icomp, 0.0f0, 0.0f0, 0.0f0)
    end

    # Constructor with vector of rad values for multi-component
    function Particle(x::Real, y::Real, z::Real, tbl::Real, rad::Vector{Float32}, hbl::Real, grv::Real, icomp::Int16=Int16(0))
        new(Float64(x), Float64(y), Float64(z), Float32(tbl), rad, Float32(hbl), Float32(grv), icomp, 0.0f0, 0.0f0, 0.0f0)
    end

    # Full constructor including turbulent velocities
    function Particle(x::Real, y::Real, z::Real, tbl::Real, rad::Vector{Float32}, hbl::Real, grv::Real, icomp::Int16,
                      u_turb::Real, v_turb::Real, w_turb::Real)
        new(Float64(x), Float64(y), Float64(z), Float32(tbl), rad, Float32(hbl), Float32(grv), icomp,
            Float32(u_turb), Float32(v_turb), Float32(w_turb))
    end
end

"""
    ExtraParticle

Storage for extra particle data (velocities, map ratios, precipitation).
Used for temporary calculations during advection.

# Fields
- `u::Float32` - u-velocity component (m/s)
- `v::Float32` - v-velocity component (m/s)
- `rmx::Float64` - map ratio in x direction
- `rmy::Float64` - map ratio in y direction
- `prc::Float32` - precipitation intensity (mm/hour)
"""
mutable struct ExtraParticle
    u::Float32
    v::Float32
    rmx::Float64
    rmy::Float64
    prc::Float32

    ExtraParticle() = new(0.0f0, 0.0f0, 0.0, 0.0, 0.0f0)
end

"""
    scale_rad!(p::Particle, factor::Real) -> Float32

Scale the radioactive content of a particle by a factor (0.0 to 1.0).
Returns the amount of activity removed from the particle.

# Arguments
- `p::Particle` - The particle to modify
- `factor::Real` - Scaling factor (clamped to [0, 1])

# Returns
- `Float32` - The amount of activity removed (previous - current)

# Example
```julia
removed = scale_rad!(particle, 0.9)  # Keep 90%, remove 10%
```
"""
function scale_rad!(p::Particle, factor::Real)::Float32
    if !is_active(p)
        return 0.0f0
    end

    # Clamp factor to valid range
    factor = clamp(factor, 0.0, 1.0)

    previous = p.rad[1]
    p.rad[1] = Float32(factor * previous)
    flush_away_denormal!(p)

    return previous - p.rad[1]
end

"""
    scale_rad!(p::Particle, component::Int, factor::Real) -> Float32

Scale the radioactive content of a specific component by a factor (0.0 to 1.0).
Returns the amount of activity removed from the component.

# Arguments
- `p::Particle` - The particle to modify
- `component::Int` - Component index (1-based)
- `factor::Real` - Scaling factor (clamped to [0, 1])

# Returns
- `Float32` - The amount of activity removed (previous - current)
"""
function scale_rad!(p::Particle, component::Int, factor::Real)::Float32
    if !is_active(p)
        return 0.0f0
    end

    # Clamp factor to valid range
    factor = clamp(factor, 0.0, 1.0)

    previous = p.rad[component]
    p.rad[component] = Float32(factor * previous)
    flush_away_denormal!(p)

    return previous - p.rad[component]
end

"""
    set_rad!(p::Particle, rad::Real) -> Float32

Set the radioactive content of a particle.

# Arguments
- `p::Particle` - The particle to modify
- `rad::Real` - New radioactive content (Bq)

# Returns
- `Float32` - The new radioactive content
"""
function set_rad!(p::Particle, rad::Real)::Float32
    p.rad[1] = Float32(rad)
    return p.rad[1]
end

"""
    set_rad!(p::Particle, component::Int, rad::Real) -> Float32

Set the radioactive content of a specific component of a particle.

# Arguments
- `p::Particle` - The particle to modify
- `component::Int` - Component index (1-based)
- `rad::Real` - New radioactive content (Bq)

# Returns
- `Float32` - The new radioactive content
"""
function set_rad!(p::Particle, component::Int, rad::Real)::Float32
    @assert component >= 1 && component <= length(p.rad)
    p.rad[component] = Float32(rad)
    return p.rad[component]
end

"""
    get_rad(p::Particle) -> Float32

Get the radioactive content of a particle.

# Arguments
- `p::Particle` - The particle to query

# Returns
- `Float32` - The radioactive content (Bq)
"""
get_rad(p::Particle)::Float32 = p.rad[1]

"""
    get_rad(p::Particle, component::Int) -> Float32

Get the radioactive content of a specific component of a particle.

# Arguments
- `p::Particle` - The particle to query
- `component::Int` - Component index (1-based)

# Returns
- `Float32` - The radioactive content (Bq)
"""
function get_rad(p::Particle, component::Int)::Float32
    @assert component >= 1 && component <= length(p.rad)
    return p.rad[component]
end

"""
    add_rad!(p::Particle, rad::Real) -> Float32

Add radioactive content to a particle.

# Arguments
- `p::Particle` - The particle to modify
- `rad::Real` - Amount to add (Bq, can be negative)

# Returns
- `Float32` - The new total radioactive content
"""
function add_rad!(p::Particle, rad::Real)::Float32
    p.rad[1] += Float32(rad)
    return p.rad[1]
end

"""
    add_rad!(p::Particle, component::Int, rad::Real) -> Float32

Add radioactive content to a specific component of a particle.

# Arguments
- `p::Particle` - The particle to modify
- `component::Int` - Component index (1-based)
- `rad::Real` - Amount to add (Bq, can be negative)

# Returns
- `Float32` - The new total radioactive content
"""
function add_rad!(p::Particle, component::Int, rad::Real)::Float32
    @assert component >= 1 && component <= length(p.rad)
    p.rad[component] += Float32(rad)
    return p.rad[component]
end

"""
    get_set_rad!(p::Particle, rad::Real) -> Float32

Get the current radioactive content and set a new value atomically.

# Arguments
- `p::Particle` - The particle to modify
- `rad::Real` - New radioactive content (Bq)

# Returns
- `Float32` - The previous radioactive content
"""
function get_set_rad!(p::Particle, rad::Real)::Float32
    previous = p.rad
    p.rad = Float32(rad)
    return previous
end

"""
    inactivate!(p::Particle) -> Bool

Deactivate a particle by setting its radioactive content to negative.
Lost activity is stored as negative value.

# Arguments
- `p::Particle` - The particle to deactivate

# Returns
- `Bool` - true if particle was active and is now deactivated, false if already inactive
"""
function inactivate!(p::Particle)::Bool
    if !is_active(p)
        return false
    end
    # Inactivate by negating all radioactive content
    for i in eachindex(p.rad)
        p.rad[i] = -p.rad[i]
    end
    return true
end

"""
    is_active(p::Particle) -> Bool

Check if a particle is active (has positive radioactive content).

# Arguments
- `p::Particle` - The particle to check

# Returns
- `Bool` - true if particle is active (rad > 0), false otherwise
"""
is_active(p::Particle)::Bool = any(>(0.0f0), p.rad)  # Active if any component has positive radioactivity

"""
    flush_away_denormal!(p::Particle)

Ensure particle radioactive content has a sensible lower limit.
Values below the numeric limit are set to zero to prevent denormal float performance issues.

# Arguments
- `p::Particle` - The particle to check/modify
"""
function flush_away_denormal!(p::Particle)
    for i in eachindex(p.rad)
        if p.rad[i] < NUMERIC_LIMIT_RAD
            p.rad[i] = 0.0f0
        end
    end
end
