# Atmospheric transport — Source term modelling
#
# Handles particle release profiles including nuclear weapon mushroom clouds

"""
    ReleaseProfile

Abstract type for different temporal release profiles.

# Subtypes
- `ConstantRelease`: Constant release rate between timesteps
- `BombRelease`: Instantaneous single-timestep release (nuclear explosion)
- `LinearRelease`: Linear interpolation between specified times
- `StepRelease`: Step function release rates
"""
abstract type ReleaseProfile end

"""
    ConstantRelease <: ReleaseProfile

Constant release rate throughout the simulation period.

# Example
```julia
profile = ConstantRelease()
```
"""
struct ConstantRelease <: ReleaseProfile end

"""
    BombRelease <: ReleaseProfile

Instantaneous release at a single timestep (nuclear weapon explosion).

# Fields
- `time_hours::Float64`: Time of bomb detonation (hours since simulation start)

# Example
```julia
profile = BombRelease(time_hours=0.0)  # Detonation at t=0
```
"""
struct BombRelease <: ReleaseProfile
    time_hours::Float64
end

"""
    LinearRelease <: ReleaseProfile

Linear interpolation of release rates between specified times.

# Fields
- `times_hours::Vector{Float64}`: Release times (hours)
- `rates::Matrix{Float64}`: Release rates (Bq/s) for each component at each time [component, time]

# Example
```julia
times = [0.0, 1.0, 2.0]
rates = [1e10 1e9 1e8;   # Component 1
         2e10 2e9 2e8]   # Component 2
profile = LinearRelease(times, rates)
```
"""
struct LinearRelease <: ReleaseProfile
    times_hours::Vector{Float64}
    rates::Matrix{Float64}  # [ncomp, ntimes]
end

"""
    StepRelease <: ReleaseProfile

Step function release rates.

# Fields
- `times_hours::Vector{Float64}`: Release times (hours)
- `rates::Matrix{Float64}`: Release rates (Bq/s) [component, time]
"""
struct StepRelease <: ReleaseProfile
    times_hours::Vector{Float64}
    rates::Matrix{Float64}  # [ncomp, ntimes]
end

"""
    ReleaseGeometry

Abstract type for release geometrical configurations.

# Subtypes
- `ColumnRelease`: Vertical column (zero radius)
- `CylinderRelease`: Cylindrical volume
- `MushroomCloudRelease`: Two-cylinder mushroom cloud (stem + cap)
"""
abstract type ReleaseGeometry end

"""
    ColumnRelease <: ReleaseGeometry

Vertical column release (zero horizontal extent).

# Fields
- `hlower::Float64`: Lower height (meters above surface)
- `hupper::Float64`: Upper height (meters above surface)

# Example
```julia
geom = ColumnRelease(hlower=0.0, hupper=1000.0)
```
"""
struct ColumnRelease <: ReleaseGeometry
    hlower::Float64
    hupper::Float64
end

"""
    CylinderRelease <: ReleaseGeometry

Cylindrical release volume.

# Fields
- `hlower::Float64`: Lower height (meters)
- `hupper::Float64`: Upper height (meters)
- `radius::Float64`: Horizontal radius (meters)

# Example
```julia
geom = CylinderRelease(hlower=0.0, hupper=500.0, radius=100.0)
```
"""
struct CylinderRelease <: ReleaseGeometry
    hlower::Float64
    hupper::Float64
    radius::Float64
end

"""
    MushroomCloudRelease <: ReleaseGeometry

Two-cylinder mushroom cloud configuration for nuclear explosions.

# Fields
- `stem_height::Float64`: Height of mushroom stem (meters)
- `cap_height::Float64`: Height of mushroom cap top (meters)
- `stem_radius::Float64`: Radius of stem cylinder (meters)
- `cap_radius::Float64`: Radius of cap cylinder (meters)

# Notes
- Stem extends from surface (0) to stem_height
- Cap extends from stem_height to cap_height
- Particles distributed proportionally by volume

# Example
```julia
# 15 kt nuclear explosion
geom = MushroomCloudRelease(
    stem_height=2000.0,    # 2 km stem
    cap_height=8000.0,     # 8 km total height
    stem_radius=500.0,     # 500 m stem radius
    cap_radius=2000.0      # 2 km cap radius
)
```
"""
struct MushroomCloudRelease <: ReleaseGeometry
    stem_height::Float64
    cap_height::Float64
    stem_radius::Float64
    cap_radius::Float64

    function MushroomCloudRelease(stem_height, cap_height, stem_radius, cap_radius)
        @assert stem_height > 0 "Stem height must be positive"
        @assert cap_height > stem_height "Cap height must exceed stem height"
        @assert stem_radius > 0 "Stem radius must be positive"
        @assert cap_radius > 0 "Cap radius must be positive"
        new(stem_height, cap_height, stem_radius, cap_radius)
    end
end

"""
    ReleaseSource{T<:Real}

Complete source term specification.

# Fields
- `position::Tuple{T,T}`: Release position (x_grid, y_grid) in grid coordinates
- `geometry::ReleaseGeometry`: Spatial distribution
- `profile::ReleaseProfile`: Temporal profile
- `activity::Vector{T}`: Release activity per component (Bq/s for continuous, Bq total for bomb)
- `nparticles::Int`: Number of particles per component

# Example
```julia
source = ReleaseSource(
    position=(50.5, 75.3),
    geometry=MushroomCloudRelease(2000.0, 8000.0, 500.0, 2000.0),
    profile=BombRelease(0.0),
    activity=[1e15, 5e14],  # Two components
    nparticles=1000
)
```
"""
struct ReleaseSource{T<:Real}
    position::Tuple{T,T}  # (x_grid, y_grid)
    geometry::ReleaseGeometry
    profile::ReleaseProfile
    activity::Vector{T}  # Bq/s or total Bq per component
    nparticles::Int  # particles per component

    function ReleaseSource(position::Tuple{T,T}, geometry::ReleaseGeometry,
                          profile::ReleaseProfile, activity::Vector{T},
                          nparticles::Int) where T<:Real
        @assert nparticles > 0 "Number of particles must be positive"
        @assert all(activity .>= 0) "Activities must be non-negative"
        new{T}(position, geometry, profile, activity, nparticles)
    end
end

"""
    ReleaseSource(domain::SimulationDomain, lat, lon, geometry, profile, activity, nparticles)

Convenience constructor that converts geographic coordinates to grid coordinates.

# Arguments
- `domain`: Simulation domain (for coordinate conversion)
- `lat`: Release latitude (degrees North)
- `lon`: Release longitude (degrees East)
- `geometry`: Release geometry (ColumnRelease, CylinderRelease, etc.)
- `profile`: Temporal release profile (BombRelease, ConstantRelease, etc.)
- `activity`: Release activity per component (Bq)
- `nparticles`: Number of particles

# Example
```julia
source = ReleaseSource(domain, 37.0956, -116.1028,
    CylinderRelease(0.0, 12500.0, 2500.0),
    BombRelease(0.0), [1e15], 1000)
```
"""
function ReleaseSource(domain, lat::Real, lon::Real,
                       geometry::ReleaseGeometry, profile::ReleaseProfile,
                       activity::Vector{T}, nparticles::Int) where T<:Real
    x, y = latlon_to_grid(domain, lat, lon)
    return ReleaseSource((T(x), T(y)), geometry, profile, activity, nparticles)
end

"""
    Plume{T<:Real}

Information for a single particle plume.

# Fields
- `particle_range::UnitRange{Int}`: Range of particle indices in this plume
- `age_steps::Int`: Age of plume in timesteps
- `release_per_particle::Vector{T}`: Activity per particle per component (Bq)

# Example
```julia
plume = Plume(
    particle_range=1:100,
    age_steps=0,
    release_per_particle=[1e10, 5e9]
)
```
"""
struct Plume{T<:Real}
    particle_range::UnitRange{Int}
    age_steps::Int
    release_per_particle::Vector{T}  # Bq per particle, per component
end

"""
    compute_release_cylinders(geom::ReleaseGeometry)

Decompose release geometry into constituent cylinders.

# Returns
- `Vector{NamedTuple}`: Cylinders with (hlower, hupper, radius, volume_fraction)

# Notes
- Column and Cylinder return single cylinder
- MushroomCloud returns two cylinders (stem, cap) with volume-based fractions
"""
function compute_release_cylinders(geom::ColumnRelease)
    return [(hlower=geom.hlower, hupper=geom.hupper, radius=0.0, volume_fraction=1.0)]
end

function compute_release_cylinders(geom::CylinderRelease)
    return [(hlower=geom.hlower, hupper=geom.hupper, radius=geom.radius, volume_fraction=1.0)]
end

function compute_release_cylinders(geom::MushroomCloudRelease)
    # Stem: surface to stem_height
    stem_volume = π * geom.stem_radius^2 * geom.stem_height

    # Cap: stem_height to cap_height
    cap_height = geom.cap_height - geom.stem_height
    cap_volume = π * geom.cap_radius^2 * cap_height

    total_volume = stem_volume + cap_volume

    stem = (hlower=0.0, hupper=geom.stem_height, radius=geom.stem_radius,
            volume_fraction=stem_volume/total_volume)
    cap = (hlower=geom.stem_height, hupper=geom.cap_height, radius=geom.cap_radius,
           volume_fraction=cap_volume/total_volume)

    return [stem, cap]
end

"""
    sample_cylinder_position(rng, x_center, y_center, radius, hlower, hupper,
                            xm, ym, dx_grid, dy_grid)

Sample uniform random position within a cylinder in grid coordinates.

# Arguments
- `rng`: Random number generator
- `x_center, y_center`: Cylinder center (grid coordinates)
- `radius`: Cylinder radius (meters)
- `hlower, hupper`: Height bounds (meters)
- `xm, ym`: Map scale factors at release position
- `dx_grid, dy_grid`: Grid spacing (degrees or km)

# Returns
- `(x, y, z)`: Position in grid coordinates (x, y, height_meters)

# Notes
- For radius=0 (column), returns center position
- Otherwise uses rejection sampling for uniform distribution in elliptical grid projection
"""
function sample_cylinder_position(rng, x_center::T, y_center::T, radius::T,
                                 hlower::T, hupper::T,
                                 xm::T, ym::T, dx_grid::T, dy_grid::T) where T<:Real
    # Height: uniform in [hlower, hupper]
    z = hlower + (hupper - hlower) * rand(rng)

    # Column release: no horizontal extent
    if radius <= 0.001
        return (x_center, y_center, z)
    end

    # Cylinder: uniform distribution in ellipse (grid projection)
    # Convert radius from meters to grid units
    dx = abs(radius / (dx_grid / xm))
    dy = abs(radius / (dy_grid / ym))

    # Rejection sampling for uniform distribution in ellipse
    # (More efficient than exact ellipse sampling for moderate aspect ratios)
    while true
        # Sample in bounding box
        ξ = 2.0 * rand(rng) - 1.0  # [-1, 1]
        η = 2.0 * rand(rng) - 1.0  # [-1, 1]

        # Check if inside ellipse
        if (ξ^2 + η^2) <= 1.0
            x = x_center + ξ * dx
            y = y_center + η * dy
            return (x, y, z)
        end
    end
end

"""
    generate_release_particles(rng, source::ReleaseSource, istep::Int, nsteps_per_hour::Int,
                               xm_field, ym_field, dx_grid, dy_grid, hlevel::Vector)

Generate particle positions for a release at given timestep.

# Arguments
- `rng`: Random number generator
- `source`: ReleaseSource specification
- `istep`: Current timestep index
- `nsteps_per_hour`: Number of timesteps per hour
- `xm_field, ym_field`: Map scale factor fields (2D arrays)
- `dx_grid, dy_grid`: Grid spacing
- `hlevel`: Model level heights (meters)

# Returns
- `Vector{Tuple{Float64,Float64,Float64}}`: Particle positions [(x, y, z), ...]
- `Vector{Float64}`: Activity per particle per component
- `Bool`: Whether particles were released this timestep

# Notes
- For BombRelease: only releases at specified time
- For other profiles: releases according to rate
"""
function generate_release_particles(rng, source::ReleaseSource{T}, istep::Int,
                                   nsteps_per_hour::Int, xm_field::AbstractMatrix,
                                   ym_field::AbstractMatrix, dx_grid::T, dy_grid::T,
                                   hlevel::Vector{T}) where T<:Real
    ncomp = length(source.activity)

    # Get release rates and check if we should release this timestep
    release_this_step, rates_bq_per_sec, timestep_seconds = compute_release_rate(
        source.profile, istep, nsteps_per_hour, source.activity
    )

    if !release_this_step
        return Tuple{T,T,T}[], T[], false
    end

    # Get map scale factors at release position
    ix = round(Int, source.position[1])
    iy = round(Int, source.position[2])
    xm = xm_field[ix, iy]
    ym = ym_field[ix, iy]

    # Decompose geometry into cylinders
    cylinders = compute_release_cylinders(source.geometry)

    # Allocate particles across cylinders and components
    positions = Tuple{T,T,T}[]
    activities = T[]

    for cyl in cylinders
        # Number of particles in this cylinder (proportional to volume)
        npart_cylinder = round(Int, source.nparticles * cyl.volume_fraction)

        # Distribute particles among components
        npart_per_comp = div(npart_cylinder, ncomp)
        remainder = mod(npart_cylinder, ncomp)

        for icomp in 1:ncomp
            # Handle remainder particles
            npart_this = npart_per_comp + (icomp <= remainder ? 1 : 0)

            if npart_this > 0 && rates_bq_per_sec[icomp] > 0
                # Activity per particle (fraction of total activity for this cylinder)
                pbq = rates_bq_per_sec[icomp] * timestep_seconds * cyl.volume_fraction / npart_this

                # Generate particle positions
                for _ in 1:npart_this
                    pos = sample_cylinder_position(rng, source.position[1], source.position[2],
                                                   cyl.radius, cyl.hlower, cyl.hupper,
                                                   xm, ym, dx_grid, dy_grid)
                    push!(positions, pos)
                    push!(activities, pbq)
                end
            end
        end
    end

    return positions, activities, true
end

"""
    compute_release_rate(profile::ReleaseProfile, istep::Int, nsteps_per_hour::Int,
                        base_activity::Vector)

Compute release rate for given timestep and profile.

# Returns
- `Bool`: Whether to release this timestep
- `Vector{Float64}`: Release rates (Bq/s) per component
- `Float64`: Timestep duration (seconds)
"""
function compute_release_rate(profile::ConstantRelease, istep::Int,
                             nsteps_per_hour::Int, base_activity::Vector{T}) where T<:Real
    timestep_seconds = 3600.0 / nsteps_per_hour
    return true, base_activity, timestep_seconds
end

function compute_release_rate(profile::BombRelease, istep::Int,
                             nsteps_per_hour::Int, base_activity::Vector{T}) where T<:Real
    release_step = round(Int, profile.time_hours * nsteps_per_hour)

    # Release only at the specified timestep
    if istep == release_step
        # Complete release in one timestep
        return true, base_activity, 1.0  # timestep=1s for bomb (instantaneous)
    else
        return false, zeros(T, length(base_activity)), 1.0
    end
end

function compute_release_rate(profile::LinearRelease, istep::Int,
                             nsteps_per_hour::Int, base_activity::Vector{T}) where T<:Real
    current_time = istep / nsteps_per_hour

    # Find bracketing times
    idx = findfirst(t -> t >= current_time, profile.times_hours)

    if isnothing(idx) || idx == 1
        # Before first or after last time
        return false, zeros(T, length(base_activity)), 3600.0 / nsteps_per_hour
    end

    # Linear interpolation between times
    t1 = profile.times_hours[idx-1]
    t2 = profile.times_hours[idx]
    w2 = (current_time - t1) / (t2 - t1)
    w1 = 1.0 - w2

    rates = w1 * profile.rates[:, idx-1] + w2 * profile.rates[:, idx]
    timestep_seconds = 3600.0 / nsteps_per_hour

    return true, rates, timestep_seconds
end

function compute_release_rate(profile::StepRelease, istep::Int,
                             nsteps_per_hour::Int, base_activity::Vector{T}) where T<:Real
    current_time = istep / nsteps_per_hour

    # Find current step
    idx = findlast(t -> t <= current_time, profile.times_hours)

    if isnothing(idx)
        # Before first time
        return false, zeros(T, length(base_activity)), 3600.0 / nsteps_per_hour
    end

    rates = profile.rates[:, idx]
    timestep_seconds = 3600.0 / nsteps_per_hour

    return true, rates, timestep_seconds
end

"""
    create_mushroom_cloud_from_yield(yield_kt::Real, hob_meters::Real=0.0)

Create mushroom cloud geometry for nuclear weapon yield.

# Arguments
- `yield_kt`: Weapon yield (kilotons TNT equivalent)
- `hob_meters`: Height of burst (meters, 0=surface)

# Returns
- `MushroomCloudRelease`: Estimated mushroom cloud dimensions

# Notes
Uses scaling relationships from nuclear weapons effects literature.
Approximate formulas based on Glasstone & Dolan (1977).

# Example
```julia
# 15 kt surface burst
cloud = create_mushroom_cloud_from_yield(15.0, 0.0)
```
"""
function create_mushroom_cloud_from_yield(yield_kt::T, hob_meters::T=zero(T)) where T<:Real
    # Scaling relationships (approximate, from Glasstone & Dolan)
    # Cloud top height: H ≈ 3.5 * W^0.33 km for surface burst (W in kt)
    # Stem height: typically 60-70% of total height
    # Stem radius: typically 20-30% of cap radius
    # Cap radius: typically 40-50% of height

    cloud_top_km = 3.5 * yield_kt^0.33
    cloud_top_m = cloud_top_km * 1000.0

    # Adjust for height of burst
    if hob_meters > 0
        # Airburst: cloud extends both up and down from burst point
        cloud_top_m *= 0.8  # Reduced total height
    end

    stem_height = 0.65 * cloud_top_m
    cap_radius = 0.45 * cloud_top_m
    stem_radius = 0.25 * cap_radius

    return MushroomCloudRelease(stem_height, cloud_top_m, stem_radius, cap_radius)
end

# Export public API
export ReleaseProfile, ConstantRelease, BombRelease, LinearRelease, StepRelease
export ReleaseGeometry, ColumnRelease, CylinderRelease, MushroomCloudRelease
export ReleaseSource, Plume
export compute_release_cylinders, sample_cylinder_position
export generate_release_particles, compute_release_rate
export create_mushroom_cloud_from_yield
