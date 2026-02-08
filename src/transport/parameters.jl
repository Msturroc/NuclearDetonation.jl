# SNAP: Severe Nuclear Accident Programme
# Julia port of snapparML.f90
# Original Copyright (C) 1992-2023 Norwegian Meteorological Institute
#
# Component parameter definitions for radionuclides

module TransportParameters

using ..TransportDimensions: mcomp

export DefinedComponent, RunningComponent, OutputComponent
export GRAV_TYPE_UNDEFINED, GRAV_TYPE_OFF, GRAV_TYPE_FIXED, GRAV_TYPE_COMPUTED
export TIME_PROFILE_CONSTANT, TIME_PROFILE_BOMB, TIME_PROFILE_LINEAR, TIME_PROFILE_STEPS, TIME_PROFILE_UNDEFINED
export def_comp, run_comp, output_component
export ncomp, nocomp, time_profile, component
export nparnum, iparnum
export push_dcomp!

# Gravity types for particle settling
const GRAV_TYPE_UNDEFINED = -1  # Undefined (error)
const GRAV_TYPE_OFF = 0         # No gravitational settling
const GRAV_TYPE_FIXED = 1       # Fixed settling velocity
const GRAV_TYPE_COMPUTED = 2    # Compute from particle properties (Stokes law)

# Time profile types for source term variation
const TIME_PROFILE_CONSTANT = 0   # Constant release
const TIME_PROFILE_BOMB = 1        # Nuclear bomb release (t^-1.2 decay)
const TIME_PROFILE_LINEAR = 2     # Linear growth to peak then linear decay
const TIME_PROFILE_STEPS = 3       # Step function (discrete releases)
const TIME_PROFILE_UNDEFINED = -1 # Undefined (error)

"""
    DefinedComponent

Definition of a radionuclide component for the simulation.

# Fields
- `name::String` - Component name (e.g., "I-131")
- `half_life_hrs::Float64` - Half-life in hours
- `decay_energy_MeV::Float64` - Average decay energy in MeV
- `yield_frac::Float64` - Fraction of total yield (0.0 to 1.0)
- `molecular_weight::Float64` - Molecular weight (g/mol)
- `particle_size_um::Float64` - Default particle size (μm)
- `density_g_cm3::Float64` - Particle density (g/cm³)
- `grav_type::Int` - Gravitational settling type (GRAV_TYPE_* constants)
- `settling_velocity_m_s::Float64` - Fixed settling velocity if grav_type == GRAV_TYPE_FIXED
- `time_profile_type::Int` - Time profile type (TIME_PROFILE_* constants)
- `deposition_velocity::Float64` - Dry deposition velocity (m/s)
- `precipitation_scavenging_coeff::Float64` - Wet scavenging coefficient
- `description::String` - Human-readable description
"""
mutable struct DefinedComponent
    name::String
    half_life_hrs::Float64
    decay_energy_MeV::Float64
    yield_frac::Float64
    molecular_weight::Float64
    particle_size_um::Float64
    density_g_cm3::Float64
    grav_type::Int
    settling_velocity_m_s::Float64
    time_profile_type::Int
    deposition_velocity::Float64
    precipitation_scavenging_coeff::Float64
    description::String

    # Constructor with defaults
    function DefinedComponent(;
        name="",
        half_life_hrs=0.0,
        decay_energy_MeV=0.0,
        yield_frac=0.0,
        molecular_weight=0.0,
        particle_size_um=0.0,
        density_g_cm3=0.0,
        grav_type=GRAV_TYPE_UNDEFINED,
        settling_velocity_m_s=0.0,
        time_profile_type=TIME_PROFILE_UNDEFINED,
        deposition_velocity=0.0,
        precipitation_scavenging_coeff=0.0,
        description="")
        new(name, half_life_hrs, decay_energy_MeV, yield_frac,
            molecular_weight, particle_size_um, density_g_cm3,
            grav_type, settling_velocity_m_s, time_profile_type,
            deposition_velocity, precipitation_scavenging_coeff,
            description)
    end
end

"""
    RunningComponent

Active component during simulation with computed properties.

# Fields
- `index::Int` - Component index (1-based)
- `lambda::Float64` - Decay constant (1/hour)
- `grav_type::Int` - Gravitational settling type
- `settling_velocity::Float64` - Computed settling velocity (m/s)
- `depo_vel::Float64` - Dry deposition velocity (m/s)
- `washout_coeff::Float64` - Wet scavenging coefficient
"""
mutable struct RunningComponent
    index::Int
    lambda::Float64  # Decay constant (1/hour)
    grav_type::Int
    settling_velocity::Float64
    depo_vel::Float64
    washout_coeff::Float64

    # Constructor
    function RunningComponent(index::Int, lambda::Float64,
                             grav_type::Int, settling_velocity::Float64,
                             depo_vel::Float64, washout_coeff::Float64)
        new(index, lambda, grav_type, settling_velocity, depo_vel, washout_coeff)
    end
end

"""
    OutputComponent

Component information for output files.

# Fields
- `name::String` - Component name
- `half_life_hrs::Float64` - Half-life in hours
- `yield_frac::Float64` - Yield fraction
- `description::String` - Description
"""
struct OutputComponent
    name::String
    half_life_hrs::Float64
    yield_frac::Float64
    description::String

    # Constructor
    function OutputComponent(name::String, half_life_hrs::Float64,
                            yield_frac::Float64, description::String)
        new(name, half_life_hrs, yield_frac, description)
    end
end

# Global component arrays
"""Defined components (user input)"""
const def_comp = Array{DefinedComponent}(undef, mcomp)

"""Running components (active during simulation)"""
const run_comp = Array{RunningComponent}(undef, mcomp)

"""Output components (for file headers)"""
const output_component = Array{OutputComponent}(undef, mcomp)

"""Number of defined components"""
ncomp = Ref(0)

"""Number of output components"""
nocomp = Ref(0)

"""Time profile functions for each component"""
const time_profile = Array{Function}(undef, mcomp)

"""Component lookup table (name → index)"""
const component = Dict{String, Int}()

"""Number of particles per component"""
nparnum = Ref(0)

"""Particle index per component"""
const iparnum = zeros(Int, mcomp)

"""
    push_dcomp!(comp::DefinedComponent)

Add a defined component to the global component array.

Increments `ncomp[]` and stores the component at that index.

# Arguments
- `comp::DefinedComponent` - Component to add
"""
function push_dcomp!(comp::DefinedComponent)
    ncomp[] += 1
    if ncomp[] > mcomp
        error("Too many components: exceeded maximum of $mcomp")
    end
    def_comp[ncomp[]] = comp
    return nothing
end

end  # module TransportParameters