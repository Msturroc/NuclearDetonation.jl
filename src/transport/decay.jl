# SNAP: Severe Nuclear Accident Programme
# Radioactive Decay Module
#
# Ported from decay.f90 - handles radioactive decay of particles and deposition fields
# Supports:
#   1. Standard exponential decay (half-life based)
#   2. Nuclear weapon fallout decay (t^-1.2 power law from Glasstone/Dolan)

"""
    DecayType

Enumeration for decay types.

# Values
- `NoDecay = 0`: No radioactive decay
- `ExponentialDecay = 1`: Standard radioactive decay with half-life
- `BombDecay = 2`: Nuclear weapon fallout decay (t^-1.2 power law)

# Notes
The bomb decay follows the Glasstone/Dolan "7-10 rule" for nuclear weapon fallout:
    R(t) = R₀ * t^(-1.2)
where t is time in hours since H+1 (1 hour after detonation).

See: Glasstone & Dolan, "The Effects of Nuclear Weapons" (1977), Chapter 9.
"""
@enum DecayType begin
    NoDecay = 0
    ExponentialDecay = 1
    BombDecay = 2
end

"""
    DecayParams{T<:Real}

Parameters for radioactive decay of a component.

# Fields
- `kdecay::DecayType`: Type of decay (NoDecay, ExponentialDecay, BombDecay)
- `halftime_hours::T`: Half-life in hours (used for ExponentialDecay)
- `decayrate::T`: Pre-computed decay rate per timestep (updated by prepare_decay_rates!)

# Example
```julia
# Cs-137 with 30.17 year half-life
cs137 = DecayParams(ExponentialDecay, 30.17 * 365.25 * 24.0)

# Nuclear weapon fallout
fallout = DecayParams(BombDecay, 0.0)  # halftime not used for bomb decay
```
"""
@kwdef mutable struct DecayParams{T<:Real}
    kdecay::DecayType = NoDecay
    halftime_hours::T = 0.0
    decayrate::T = 1.0  # Multiplicative factor per timestep
end

"""
    BombDecayState{T<:Real}

State tracking for nuclear weapon fallout decay (t^-1.2 power law).

# Fields
- `total_time_s::T`: Total elapsed time since start of simulation (seconds)
- `bomb_time_s::T`: Time of nuclear detonation (seconds since start)
- `has_components::Bool`: Whether any components use bomb decay

# Notes
Decay begins at H+1 (1 hour after detonation) to satisfy C(t) = C₀ * t^(-1.2).
Before H+1, the cloud is assumed to be stabilizing and no decay occurs.
"""
@kwdef mutable struct BombDecayState{T<:Real}
    total_time_s::T = 0.0
    bomb_time_s::T = 0.0
    has_components::Bool = false
end

"""
    prepare_decay_rates!(params::Vector{DecayParams{T}}, timestep_s::Real;
                        bomb_state::Union{Nothing,BombDecayState}=nothing) where T

Compute decay rates for all components for a given timestep.

This should be called once per model timestep BEFORE applying decay to particles
or deposition fields.

# Arguments
- `params`: Vector of DecayParams for each component
- `timestep_s`: Model timestep in seconds
- `bomb_state`: Optional BombDecayState for nuclear weapon decay tracking

# Updates
- `params[i].decayrate`: Set to multiplicative decay factor for this timestep

# Decay rate formulas
For exponential decay (kdecay=1):
    decayrate = exp(-ln(2) * Δt / T₁/₂)

For bomb decay (kdecay=2):
    decayrate = ((t + Δt) / t)^(-1.2)  for t > H+1
    decayrate = 1.0                      for t ≤ H+1

# Example
```julia
params = [DecayParams(ExponentialDecay, 8.02 * 24.0)]  # I-131, 8.02 day half-life
bomb_state = BombDecayState(bomb_time_s=0.0)
timestep_s = 600.0  # 10 minute timestep

prepare_decay_rates!(params, timestep_s, bomb_state=bomb_state)
# params[1].decayrate now contains exp(-ln(2) * 600 / (8.02*24*3600))
```
"""
function prepare_decay_rates!(params::Vector{DecayParams{T}},
                              timestep_s::Real;
                              bomb_state::Union{Nothing,BombDecayState{T}}=nothing) where T

    # Check if any component uses bomb decay
    has_bomb_decay = any(p.kdecay == BombDecay for p in params)
    if !isnothing(bomb_state)
        bomb_state.has_components = has_bomb_decay
    end

    # Compute decay rates for each component
    for param in params
        if param.kdecay == ExponentialDecay
            # Standard radioactive decay: N(t) = N₀ * exp(-λt)
            # decayrate = N(t+Δt)/N(t) = exp(-λΔt) = exp(-ln(2) * Δt / T₁/₂)
            halftime_s = param.halftime_hours * 3600.0
            param.decayrate = exp(-log(2.0) * timestep_s / halftime_s)

        elseif param.kdecay == BombDecay
            # Nuclear weapon fallout decay computed below (all bomb components together)
            # Placeholder - will be set in bomb decay block
            param.decayrate = 1.0

        else  # NoDecay
            param.decayrate = 1.0
        end
    end

    # Handle bomb decay (t^-1.2 power law)
    if has_bomb_decay && !isnothing(bomb_state)
        compute_bomb_decay_rate!(params, timestep_s, bomb_state)
    end

    return nothing
end

"""
    compute_bomb_decay_rate!(params, timestep_s, bomb_state)

Compute decay rate for nuclear weapon fallout using t^-1.2 power law.

Internal function called by prepare_decay_rates! when bomb decay is active.

# Notes
- Decay starts at H+1 (1 hour after detonation)
- Formula: R(t) = R₀ * (t/t₀)^(-1.2) where t is hours since H+1
- Before H+1, decayrate = 1.0 (no decay during cloud stabilization)
"""
function compute_bomb_decay_rate!(params::Vector{DecayParams{T}},
                                  timestep_s::Real,
                                  bomb_state::BombDecayState{T}) where T

    # Time since H+1 (start decay 1 hour after detonation)
    decay_start_time = bomb_state.bomb_time_s + 3600.0

    if bomb_state.total_time_s >= decay_start_time
        # Time in hours since H+1
        t_current_hrs = (bomb_state.total_time_s - bomb_state.bomb_time_s) / 3600.0
        t_next_hrs = (bomb_state.total_time_s + timestep_s - bomb_state.bomb_time_s) / 3600.0

        # R(t) = R₀ * t^(-1.2), so R(t+Δt)/R(t) = (t+Δt)^(-1.2) / t^(-1.2)
        current_state = t_current_hrs^(-1.2)
        next_state = t_next_hrs^(-1.2)
        bomb_decay_rate = next_state / current_state
    else
        # Before H+1, no decay (cloud stabilizing)
        bomb_decay_rate = 1.0
    end

    # Apply to all bomb decay components
    for param in params
        if param.kdecay == BombDecay
            param.decayrate = bomb_decay_rate
        end
    end

    # Advance bomb state time
    bomb_state.total_time_s += timestep_s

    return nothing
end

"""
    apply_decay(activity::T, param::DecayParams{T}) where T<:Real

Apply radioactive decay to a particle's activity.

# Arguments
- `activity`: Current radioactive activity (Bq or other units)
- `param`: DecayParams with pre-computed decay rate

# Returns
- Updated activity after decay

# Example
```julia
activity = 1.0e6  # 1 MBq
param = DecayParams(ExponentialDecay, 8.02 * 24.0, 0.999)  # decayrate from prepare_decay_rates!

new_activity = apply_decay(activity, param)
# new_activity ≈ 999000 Bq (slight decay)
```

# Notes
This function should be called AFTER prepare_decay_rates! has been called for the current timestep.
"""
function apply_decay(activity::T, param::DecayParams{T}) where T<:Real
    if param.kdecay == NoDecay
        return activity
    else
        return activity * param.decayrate
    end
end

"""
    apply_decay!(field::AbstractArray{T}, param::DecayParams{T}) where T<:Real

Apply radioactive decay to a deposition or concentration field (in-place).

# Arguments
- `field`: Array of activities/concentrations to decay (modified in-place)
- `param`: DecayParams with pre-computed decay rate

# Example
```julia
deposition = rand(100, 100)  # Deposition field (Bq/m²)
param = DecayParams(ExponentialDecay, 30.17 * 365.25 * 24.0)  # Cs-137
prepare_decay_rates!([param], 3600.0)  # 1 hour timestep

apply_decay!(deposition, param)
# All values in deposition array scaled by decayrate
```
"""
function apply_decay!(field::AbstractArray{T}, param::DecayParams{T}) where T<:Real
    if param.kdecay != NoDecay
        field .*= param.decayrate
    end
    return nothing
end

# Export public API
export DecayType, NoDecay, ExponentialDecay, BombDecay
export DecayParams, BombDecayState
export prepare_decay_rates!, apply_decay, apply_decay!
