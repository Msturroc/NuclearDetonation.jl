# Atmospheric transport — Main time-stepping loop
#
# Integrates advection, turbulence, settling, decay, and deposition

"""
    TimeSteppingParams{T<:Real}

Parameters controlling the time integration.

# Fields
- `dt::T`: Time step (seconds)
- `nsteps::Int`: Total number of timesteps
- `advection_substeps::Int`: Number of substeps for advection
- `enable_turbulence::Bool`: Enable random walk turbulence
- `enable_settling::Bool`: Enable gravitational settling
- `enable_decay::Bool`: Enable radioactive decay
- `enable_dry_deposition::Bool`: Enable dry deposition
- `enable_wet_deposition::Bool`: Enable wet deposition
"""
struct TimeSteppingParams{T<:Real}
    dt::T
    nsteps::Int
    advection_substeps::Int
    enable_turbulence::Bool
    enable_settling::Bool
    enable_decay::Bool
    enable_dry_deposition::Bool
    enable_wet_deposition::Bool

    function TimeSteppingParams(dt::T, nsteps::Int;
                               advection_substeps::Int=1,
                               enable_turbulence::Bool=true,
                               enable_settling::Bool=true,
                               enable_decay::Bool=true,
                               enable_dry_deposition::Bool=true,
                               enable_wet_deposition::Bool=true) where T<:Real
        @assert dt > 0
        @assert nsteps > 0
        @assert advection_substeps > 0
        new{T}(dt, nsteps, advection_substeps,
               enable_turbulence, enable_settling, enable_decay,
               enable_dry_deposition, enable_wet_deposition)
    end
end

"""
    PhysicsParams{T<:Real}

Collection of all physics parameters for the simulation.

# Fields
- `vgrav_tables::VGravTables{T}`: Settling velocity tables
- `decay_params::Vector{DecayParams{T}}`: Decay parameters per component
- `dry_deposition_params::DryDepositionParams{T}`: Dry deposition parameters
- `wet_deposition_params::WetDepositionParams{T}`: Wet deposition parameters
- `boundary_layer_height::T`: PBL top (m MSL)
"""
struct PhysicsParams{T<:Real}
    vgrav_tables::VGravTables
    decay_params::Vector{DecayParams{T}}
    dry_deposition_params::DryDepositionParams{T}
    wet_deposition_params::WetDepositionParams{T}
    boundary_layer_height::T
end

"""
    release_particles!(state::SimulationState, sources::Vector{ReleaseSource},
                      met_fields::MeteoFields, rng::Random.AbstractRNG, current_step::Int, nsteps_per_hour::Int)

Release new particles from sources at the current timestep.

# Arguments
- `state`: Simulation state to modify
- `sources`: Release sources
- `met_fields`: Current meteorological fields
- `rng`: Random number generator
- `current_step`: Current timestep number
- `nsteps_per_hour`: Number of timesteps per hour
"""
function release_particles!(state::SimulationState{T},
                           sources::Vector{ReleaseSource{T}},
                           met_fields::MeteoFields{T},
                           rng::Random.AbstractRNG,
                           current_step::Int,
                           nsteps_per_hour::Int) where T<:Real
    for source in sources
        # Check if this source releases at this timestep
        positions, activities, released = generate_release_particles(
            rng, source, current_step, nsteps_per_hour,
            met_fields.xm, met_fields.ym,
            state.domain.dx, state.domain.dy,
            state.domain.hlevel
        )

        if released
            # Add particles to ensemble
            for (pos, activity) in zip(positions, activities)
                # Convert activity vector to mass vector
                mass = zeros(T, state.ensemble.ncomponents)
                for (i, comp_activity) in enumerate(activity)
                    mass[i] = comp_activity
                end

                # Initial velocity is zero (will be set by wind)
                velocity = SVector{3,T}(0, 0, 0)

                # CRITICAL FIX: Convert height (meters) to sigma coordinate
                # pos is (x, y, height_m) from generate_release_particles
                # but particle dynamics expects (x, y, sigma)
                x, y, height_m = pos
                sigma = height_to_sigma_hybrid(x, y, height_m, met_fields, 0.0)
                position = SVector{3,T}(x, y, sigma)

                # Add particle with sigma coordinate
                add_particle!(state.ensemble, position, velocity, mass, zero(T))

                # Track released activity
                state.total_released .+= mass
            end
        end
    end
end

"""
    advect_particles!(state::SimulationState, wind_fields::WindFields,
                     params::TimeSteppingParams, rng::Random.AbstractRNG)

Advect particles using wind fields and turbulent diffusion.

# Arguments
- `state`: Simulation state
- `wind_fields`: Interpolated wind fields
- `params`: Time-stepping parameters
- `rng`: Random number generator
"""
function advect_particles!(state::SimulationState{T},
                          wind_fields::WindFields{T},
                          params::TimeSteppingParams{T},
                          rng::Random.AbstractRNG) where T<:Real
    dt_substep = params.dt / params.advection_substeps

    for pidx in 1:length(state.ensemble.particles)
        if !is_active(state.ensemble.particles[pidx])
            continue
        end

        pos = state.ensemble.positions[pidx]

        # Simple Euler integration for advection
        # In production, could use RK4 or adaptive stepping
        for substep in 1:params.advection_substeps
            x, y, z = pos[1], pos[2], pos[3]

            # Get wind at current position
            u = wind_fields.u_interp(x, y, z, state.timestep * params.dt)
            v = wind_fields.v_interp(x, y, z, state.timestep * params.dt)
            w = wind_fields.w_interp(x, y, z, state.timestep * params.dt)

            # Convert wind from m/s to grid coordinates per second
            # Use base spacings from current met window (winds.dx_m/dy_m)
            local_ix = min(floor(Int, x), state.domain.nx)
            local_iy = min(floor(Int, y), state.domain.ny)
            dx_dt = u / wind_fields.dx_m * state.domain.xm[local_ix, local_iy]
            dy_dt = v / wind_fields.dy_m * state.domain.ym[local_ix, local_iy]
            dz_dt = w  # Vertical already in m/s

            # Update position
            new_x = x + dx_dt * dt_substep
            new_y = y + dy_dt * dt_substep
            new_z = z + dz_dt * dt_substep

            # Apply periodic/reflective boundary conditions
            new_x = clamp(new_x, 1.0, T(state.domain.nx))
            new_y = clamp(new_y, 1.0, T(state.domain.ny))
            new_z = max(new_z, 1.0)  # Can't go below ground

            pos = SVector{3,T}(new_x, new_y, new_z)
        end

        state.ensemble.positions[pidx] = pos
        state.ensemble.velocities[pidx] = SVector{3,T}(u, v, w)
    end
end

"""
    apply_settling!(state::SimulationState, vgrav_tables::VGravTables, dt::Real)

Apply gravitational settling to all particles.

# Arguments
- `state`: Simulation state
- `vgrav_tables`: Settling velocity tables
- `dt`: Time step (seconds)
"""
function apply_settling!(state::SimulationState{T}, vgrav_tables::VGravTables, dt::T) where T<:Real
    for pidx in 1:length(state.ensemble.particles)
        if !is_active(state.ensemble.particles[pidx])
            continue
        end

        pos = state.ensemble.positions[pidx]
        z = pos[3]

        # Get air density at current height
        # For now, use standard atmosphere (could use met fields)
        height_m = clamp(z, state.domain.hlevel[1], state.domain.hlevel[end])
        T_air = 288.15 - 0.0065 * height_m  # Standard lapse rate
        P_air = 101325.0 * (T_air / 288.15)^5.256  # Standard atmosphere

        # Interpolate settling velocity
        # Use average of all particle size bins (simplified)
        vg = interpolate_vgrav(vgrav_tables, 1, P_air / 100.0, T_air)

        # Update vertical position
        dz = -vg * dt  # Negative because settling is downward

        new_z = clamp(pos[3] + dz, state.domain.hlevel[1], state.domain.hlevel[end])

        state.ensemble.positions[pidx] = SVector{3,T}(pos[1], pos[2], new_z)
    end
end

"""
    apply_decay!(state::SimulationState, decay_params::Vector{DecayParams}, dt::Real)

Apply radioactive decay to all particles.

# Arguments
- `state`: Simulation state
- `decay_params`: Decay parameters for each component
- `dt`: Time step (seconds)
"""
function apply_decay!(state::SimulationState{T}, decay_params::Vector{DecayParams{T}}, dt::T) where T<:Real
    for particle in state.ensemble.particles
        if !is_active(particle)
            continue
        end

        # Apply decay to each component
        for comp in 1:state.ensemble.ncomponents
            mass = get_rad(particle, comp)
            if mass > 0
                # Apply radioactive decay using pre-computed decay rate
                new_mass = apply_decay(mass, decay_params[comp])
                set_rad!(particle, comp, new_mass)
            end
        end

        # Check if particle should be inactivated (all components decayed)
        total_mass = sum(get_rad(particle, i) for i in 1:state.ensemble.ncomponents)
        if total_mass < 1e-20  # Effectively zero
            inactivate!(particle)
        end
    end
end

"""
    apply_deposition!(state::SimulationState, physics::PhysicsParams, season::SeasonCategory, dt::Real)

Apply dry and wet deposition to particles near the surface.

# Arguments
- `state`: Simulation state
- `physics`: Physics parameters
- `season`: Current season
- `dt`: Time step (seconds)
"""
function apply_deposition!(state::SimulationState{T}, physics::PhysicsParams{T},
                          season::SeasonCategory, dt::T) where T<:Real
    # For now, apply simplified deposition
    # In production, would compute deposition velocity from met fields

    for (pidx, particle) in enumerate(state.ensemble.particles)
        if !is_active(particle)
            continue
        end

        pos = state.ensemble.positions[pidx]
        z_height = pos[3]

        # Only apply deposition to particles near surface (lowest ~2 levels)
        surface_limit = state.domain.nz > 1 ? state.domain.hlevel[2] : state.domain.hlevel[end]
        if z_height > surface_limit
            continue
        end

        # Get grid cell indices
        i = clamp(floor(Int, pos[1]), 1, state.domain.nx)
        j = clamp(floor(Int, pos[2]), 1, state.domain.ny)

        # Dry deposition (simplified - use constant velocity)
        vd_dry = T(0.001)  # 1 mm/s typical dry deposition velocity

        # Apply to particle mass
        mass_before = [get_rad(particle, comp) for comp in 1:state.ensemble.ncomponents]

        # Deposition probability: P = 1 - exp(-vd * dt / h_mix)
        h_mix = T(100.0)  # Mixing height near surface
        prob_dep = 1 - exp(-vd_dry * dt / h_mix)

        for comp in 1:state.ensemble.ncomponents
            mass = get_rad(particle, comp)
            if mass > 0
                deposited = mass * prob_dep
                new_mass = mass - deposited

                set_rad!(particle, comp, new_mass)

                # Add to deposition field
                state.fields.dry_deposition[i, j, comp] += deposited
                state.total_deposited[comp] += deposited
            end
        end

        # Inactivate if all mass deposited
        total_mass = sum(get_rad(particle, i) for i in 1:state.ensemble.ncomponents)
        if total_mass < 1e-20
            inactivate!(particle)
        end
    end
end

"""
    step!(state::SimulationState, sources::Vector{ReleaseSource},
          wind_fields::WindFields, physics::PhysicsParams,
          params::TimeSteppingParams, rng::Random.AbstractRNG, season::SeasonCategory)

Advance the simulation by one timestep.

# Arguments
- `state`: Simulation state (modified in-place)
- `sources`: Release sources
- `wind_fields`: Interpolated wind fields
- `physics`: Physics parameters
- `params`: Time-stepping parameters
- `rng`: Random number generator
- `season`: Current season
"""
function step!(state::SimulationState{T},
               sources::Vector{ReleaseSource{T}},
               wind_fields::WindFields{T},
               physics::PhysicsParams{T},
               params::TimeSteppingParams{T},
               rng::Random.AbstractRNG,
               season::SeasonCategory) where T<:Real

    # 1. Release new particles
    nsteps_per_hour = round(Int, 3600.0 / params.dt)
    release_particles!(state, sources, wind_fields, rng, state.timestep, nsteps_per_hour)

    # 2. Advect particles
    advect_particles!(state, wind_fields, params, rng)

    # 3. Apply settling
    if params.enable_settling
        apply_settling!(state, physics.vgrav_tables, params.dt)
    end

    # 4. Apply decay
    if params.enable_decay
        apply_decay!(state, physics.decay_params, params.dt)
    end

    # 5. Apply deposition
    if params.enable_dry_deposition || params.enable_wet_deposition
        apply_deposition!(state, physics, season, params.dt)
    end

    # 6. Update particle ages
    for i in 1:length(state.ensemble.ages)
        state.ensemble.ages[i] += params.dt
    end

    # 7. Remove inactive particles
    remove_inactive_particles!(state.ensemble)

    # 8. Accumulate concentration
    accumulate_concentration!(state.fields, state.ensemble, state.domain, wind_fields, params.dt)

    # 9. Update time
    state.timestep += 1
    state.current_time = add_duration(state.current_time, Duration(0, 0, 0, round(Int, params.dt)))
end

# Export public API
export TimeSteppingParams, PhysicsParams
export step!
