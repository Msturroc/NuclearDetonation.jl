# SNAP: Severe Nuclear Accident Programme
# Random Walk Diffusion Module
#
# Ported from rwalk.f90
# Implements turbulent diffusion using random walk (Wiener process)
# Integrates with particle_dynamics.jl via StochasticDiffEq.jl

using StochasticDiffEq

"""
    RandomWalkParams{T<:Real}

Parameters for random walk turbulent diffusion.

# Fields
- `timestep::T`: Model timestep (seconds)
- `tmix_vertical::T`: Characteristic vertical mixing time (seconds, default: 900s = 15 min)
- `tmix_horizontal::T`: Characteristic horizontal mixing time (seconds, default: 900s = 15 min)
- `lmax::T`: Maximum l-eta in mixing layer (default: 0.28)
- `labove::T`: Standard l-eta above mixing layer (default: 0.03)
- `entrainment::T`: Entrainment zone fraction (default: 0.1 = 10% of mixing height)
- `hmax::T`: Maximum mixing height (m, default: 2500.0)
- `blfullmix::Bool`: Full mixing in boundary layer (true) or gradual (false)
- `horizontal_a_bl::T`: Horizontal diffusion coefficient in BL (default: 0.5)
- `horizontal_a_above::T`: Horizontal diffusion coefficient above BL (default: 0.25)
- `horizontal_b::T`: Power law exponent for wind speed dependence (default: 0.875)

# Notes
Parameterization from Bartnicki (2011):
- For mixing height = 2500 m and timestep = 15 min:
  - In ABL: l-eta = 0.28
  - Above ABL: l-eta = 0.03
- For arbitrary mixing height and timestep, scales proportionally

# Example
```julia
params = RandomWalkParams(timestep=600.0)  # 10 minute timestep
```
"""
@kwdef struct RandomWalkParams{T<:Real}
    timestep::T
    tmix_vertical::T = 900.0  # 15 minutes
    tmix_horizontal::T = 900.0  # 15 minutes
    lmax::T = 0.28
    labove::T = 0.03
    entrainment::T = 0.1
    hmax::T = 2500.0
    blfullmix::Bool = false
    horizontal_a_bl::T = 0.5
    horizontal_a_above::T = 0.25
    horizontal_b::T = 0.875
end

"""
    RandomWalkState{T<:Real}

Pre-computed state for random walk calculations (computed once at initialization).

# Fields
- `tfactor_v::T`: Timestep / vertical mixing time ratio
- `tsqrtfactor_v::T`: sqrt(timestep / vertical mixing time)
- `tfactor_h::T`: Timestep / horizontal mixing time ratio
- `tsqrtfactor_h::T`: sqrt(timestep / horizontal mixing time)
- `vrdbla::T`: Vertical diffusion coefficient above boundary layer
"""
struct RandomWalkState{T<:Real}
    tfactor_v::T
    tsqrtfactor_v::T
    tfactor_h::T
    tsqrtfactor_h::T
    vrdbla::T
end

"""
    initialize_random_walk(params::RandomWalkParams{T}) where T

Initialize random walk state from parameters.

# Arguments
- `params`: RandomWalkParams configuration

# Returns
- `RandomWalkState`: Pre-computed factors for efficient random walk

# Example
```julia
params = RandomWalkParams(timestep=600.0)
state = initialize_random_walk(params)
```
"""
function initialize_random_walk(params::RandomWalkParams{T}) where T
    tfactor_v = params.timestep / params.tmix_vertical
    tsqrtfactor_v = sqrt(tfactor_v)
    tfactor_h = params.timestep / params.tmix_horizontal
    tsqrtfactor_h = sqrt(tfactor_h)
    vrdbla = params.labove * tsqrtfactor_v

    return RandomWalkState(tfactor_v, tsqrtfactor_v, tfactor_h, tsqrtfactor_h, vrdbla)
end

"""
    horizontal_diffusion_length(u::Real, v::Real, z::Real, tbl::Real,
                                params::RandomWalkParams, state::RandomWalkState)

Compute horizontal diffusion length scale.

# Arguments
- `u`: Wind u-component (m/s)
- `v`: Wind v-component (m/s)
- `z`: Particle height (sigma/eta)
- `tbl`: Top of boundary layer (sigma/eta)
- `params`: RandomWalkParams
- `state`: RandomWalkState

# Returns
- Horizontal diffusion length scale (grid units)

# Formula
rl = 2 × a × (v_abs × t_mix)^b × sqrt(dt/t_mix)

where:
- a = 0.5 in boundary layer, 0.25 above
- b = 0.875 (power law exponent)
- v_abs = wind speed
- t_mix = horizontal mixing time
"""
function horizontal_diffusion_length(u::Real, v::Real, z::Real, tbl::Real,
                                     params::RandomWalkParams, state::RandomWalkState)
    # Determine if in boundary layer
    in_bl = z > tbl
    a = in_bl ? params.horizontal_a_bl : params.horizontal_a_above

    # Wind speed
    vabs = hypot(u, v)

    # Horizontal diffusion length scale
    # rl = 2*a*((vabs*tmix)^b) * sqrt(dt/tmix)
    rl = 2 * a * (vabs * params.tmix_horizontal)^params.horizontal_b * state.tsqrtfactor_h

    return rl
end

"""
    vertical_diffusion_coefficient(z::Real, tbl::Real, params::RandomWalkParams,
                                   state::RandomWalkState)

Compute vertical diffusion coefficient (standard deviation per timestep).

# Arguments
- `z`: Particle height (sigma/eta, 0=surface, 1=top)
- `tbl`: Top of boundary layer (sigma/eta)
- `params`: RandomWalkParams
- `state`: RandomWalkState

# Returns
- Vertical diffusion coefficient (sigma/eta units)

# Formula
- Above BL: σ_z = labove × sqrt(dt/tmix_v)
- In BL: σ_z = (1 - tbl) × sqrt(dt/tmix_v)

# Notes
In boundary layer, diffusion is proportional to mixing height.
Above boundary layer, uses constant small diffusion.
"""
function vertical_diffusion_coefficient(z::Real, tbl::Real, params::RandomWalkParams,
                                       state::RandomWalkState)
    if z <= tbl  # Above boundary layer
        return state.vrdbla
    else  # In boundary layer
        return (1.0 - tbl) * state.tsqrtfactor_v
    end
end

"""
    random_walk_noise!(du, u, p, t)

Stochastic diffusion function for use with StochasticDiffEq.jl.

This function computes the diagonal noise matrix (independent x, y, z noise)
for turbulent diffusion via random walk.

# Arguments
- `du`: Output noise coefficients [σ_x, σ_y, σ_z]
- `u`: Current state [x, y, z]
- `p`: Parameters tuple (winds, particle_params, rwalk_params, rwalk_state, tbl)
- `t`: Current time (seconds)

# Notes
This is the "g" function in the SDE: du = f(u,p,t)dt + g(u,p,t)dW
where dW is a Wiener process (Brownian motion).

# Example
```julia
prob = SDEProblem(particle_velocity!, random_walk_noise!, u0, tspan, p)
sol = solve(prob, EM(), dt=600.0)  # Euler-Maruyama method
```
"""
function random_walk_noise!(du, u, p, t)
    winds, particle_params, rwalk_params, rwalk_state, tbl = p

    x, y, z = u

    # Get wind components at particle position
    u_wind = winds.u_interp(x, y, z, t)
    v_wind = winds.v_interp(x, y, z, t)

    # Horizontal diffusion length
    rl = horizontal_diffusion_length(u_wind, v_wind, z, tbl, rwalk_params, rwalk_state)

    # Map ratios (convert from m/s to grid units/s)
    rmx = particle_params.map_ratio_x
    rmy = particle_params.map_ratio_y

    # Horizontal noise coefficients
    du[1] = rl * rmx
    du[2] = rl * rmy

    # Vertical diffusion coefficient
    du[3] = vertical_diffusion_coefficient(z, tbl, rwalk_params, rwalk_state)

    return nothing
end

"""
    apply_boundary_layer_reflection!(integrator)

Callback to apply reflection at boundary layer top and surface.

Particles are reflected when they:
1. Go above boundary layer top (with 10% entrainment zone)
2. Go below surface (z > 1.0 in sigma coordinates)

# Notes
- Sigma coordinate: 0 = top of atmosphere, 1 = surface
- Reflection: z_new = 2*boundary - z_old
- Entrainment zone: 10% above boundary layer allows mixing
"""
function apply_boundary_layer_reflection!(integrator)
    winds, particle_params, rwalk_params, rwalk_state, tbl = integrator.p

    x, y, z = integrator.u

    # Boundary layer entrainment thickness
    bl_entrainment_thickness = (1.0 - tbl) * (1.0 + rwalk_params.entrainment)
    top_entrainment = max(0.0, 1.0 - bl_entrainment_thickness)

    # Extended boundary layer region: particles within one entrainment thickness of the BL
    # participate in reflection. This allows particles that diffused just above the BL
    # to be reflected back, while particles far in the free atmosphere are unaffected.
    lower_reflection_bound = tbl - bl_entrainment_thickness  # = tbl + top_entrainment - 1.0

    # Only apply reflection to particles in the extended BL region
    if z >= lower_reflection_bound
        # Reflection from ABL top (with entrainment zone)
        # DISABLED: To allow particles to escape the PBL and match FLEXPART/HYSPLIT
        # if integrator.u[3] < top_entrainment
        #     integrator.u[3] = 2.0 * top_entrainment - integrator.u[3]
        # end

        # Reflection from surface
        if integrator.u[3] > 1.0
            integrator.u[3] = 2.0 - integrator.u[3]
        end

        # Enforce vertical limits
        integrator.u[3] = clamp(integrator.u[3], top_entrainment, 1.0)
    end

    return nothing
end

"""
    create_random_walk_sde_problem(initial_position, tspan, winds, particle_params,
                                   rwalk_params::RandomWalkParams; tbl::Real=0.8)

Create a Stochastic Differential Equation problem for particle transport with random walk.

# Arguments
- `initial_position`: [x0, y0, z0] initial position
- `tspan`: Time span (t_start, t_end) in seconds
- `winds`: WindFields interpolation object
- `particle_params`: ParticleParams for particle properties
- `rwalk_params`: RandomWalkParams for diffusion
- `tbl`: Top of boundary layer (sigma/eta, default: 0.8)

# Returns
- `SDEProblem`: Problem ready to solve with StochasticDiffEq.jl

# Example
```julia
using StochasticDiffEq

# Set up problem
rwalk_params = RandomWalkParams(timestep=600.0)
prob = create_random_walk_sde_problem([50.0, 50.0, 0.5], (0.0, 3600.0),
                                      winds, particle_params, rwalk_params)

# Solve with Euler-Maruyama method (explicit, simple)
sol = solve(prob, EM(), dt=600.0)

# Or use higher-order method
sol = solve(prob, SRIW1(), dt=600.0)  # Adaptive Runge-Kutta SDE solver
```
"""
function create_random_walk_sde_problem(initial_position::AbstractVector,
                                       tspan::Tuple,
                                       winds,
                                       particle_params,
                                       rwalk_params::RandomWalkParams;
                                       tbl::Real=0.8)
    # Initialize random walk state
    rwalk_state = initialize_random_walk(rwalk_params)

    # Pack parameters
    p = (winds, particle_params, rwalk_params, rwalk_state, tbl)

    # Create boundary reflection callback (discrete callback at each timestep)
    reflect_cb = DiscreteCallback(
        (u, t, integrator) -> true,  # Check every step
        apply_boundary_layer_reflection!
    )

    # For random walk, we need both drift and diffusion (random_walk_noise!)
    # Using the particle_velocity_with_rwalk! function defined in this file

    # Create SDE problem with diagonal noise
    prob = SDEProblem(particle_velocity_with_rwalk!, random_walk_noise!,
                     initial_position, tspan, p,
                     callback=reflect_cb,
                     noise_rate_prototype=zeros(3))

    return prob
end

"""
    particle_velocity_with_rwalk!(du, u, p, t)

Combined drift function for SDE (deterministic + stochastic parts).

This is a wrapper that calls the particle_velocity! function from particle_dynamics.jl.
The stochastic part (diffusion) is handled separately by random_walk_noise!.

# Notes
In SDE formulation: dX = f(X,t)dt + g(X,t)dW
- f(X,t) = drift (deterministic advection) handled by this function
- g(X,t) = diffusion (random walk) handled by random_walk_noise!
"""
function particle_velocity_with_rwalk!(du, u, p, t)
    winds, particle_params, rwalk_params, rwalk_state, tbl = p

    # Call deterministic velocity function
    # This computes du/dt = [u_wind, v_wind, w_wind + w_gravity]
    x, y, z = u

    # Interpolate wind components
    u_wind = winds.u_interp(x, y, z, t)
    v_wind = winds.v_interp(x, y, z, t)
    w_wind = winds.w_interp(x, y, z, t)

    # Gravitational settling (if enabled)
    w_grav = 0.0
    if particle_params.grav_type > 0
        # Add gravitational settling
        # TODO: Integrate with vgravtables for variable settling
        if particle_params.grav_type == 1
            # Constant settling
            w_grav = particle_params.gravity_ms * 0.001  # Approximate conversion
        end
    end

    # Apply map ratios
    du[1] = u_wind * particle_params.map_ratio_x
    du[2] = v_wind * particle_params.map_ratio_y
    du[3] = w_wind + w_grav

    return nothing
end

# Export public API
export RandomWalkParams, RandomWalkState
export initialize_random_walk
export horizontal_diffusion_length, vertical_diffusion_coefficient
export random_walk_noise!, apply_boundary_layer_reflection!
export create_random_walk_sde_problem, particle_velocity_with_rwalk!
