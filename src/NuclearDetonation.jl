module NuclearDetonation

# Weapon effects modelling (Glasstone-based)
include("utilities.jl")
include("fallout.jl")
include("overpressure.jl")
include("thermal.jl")
include("radiation.jl")

# Re-export weapon effects functions
using .Utilities
using .Overpressure
using .Thermal
using .Radiation

# Package-relative data directory
const DATADIR = joinpath(@__DIR__, "..", "data")

# Atmospheric transport modelling
module Transport

import Random  # Stdlib for RNG support

include("transport/datetime.jl")
include("transport/dimensions.jl")
include("transport/parameters.jl")

# Physical Constants (defined once for the module)
const G_GRAVITY_M_S2 = 9.80665
const R_SPECIFIC_J_KG_K = 287.058

include("transport/particles.jl")
include("transport/milib.jl")
include("transport/met_reader.jl")
include("transport/met_formats.jl")  # Define format types FIRST for dispatch
include("transport/om2edot.jl")      # Uses format types for dispatch
include("transport/compheight.jl")
include("transport/boundary_layer.jl")
include("transport/posint.jl")
include("transport/vgravtables.jl")
using .GravitationalSettling: VGravTables
export VGravTables
using .GravitationalSettling: air_viscosity, air_density, cunningham_factor, vgrav_corrected, build_vgrav_tables, interpolate_vgrav
export air_viscosity, air_density, cunningham_factor, vgrav_corrected, build_vgrav_tables, interpolate_vgrav
using .GravitationalSettling: SutherlandViscosity, ReferenceViscosity
export SutherlandViscosity, ReferenceViscosity
include("transport/decay.jl")
include("transport/numerical_config.jl")  # Numerical method configuration
include("transport/particle_dynamics.jl")
include("transport/hybrid_coordinates.jl")
include("transport/random_walk.jl")
include("transport/turbulence_hanna.jl")  # Hanna (1982) turbulence with O-U process and CBL
include("transport/release.jl")
include("transport/deposition.jl")
include("transport/simulation.jl")
include("transport/orchestration.jl")  # High-level simulation runner
include("transport/timestepping.jl")
include("transport/output.jl")
include("transport/validation.jl")
include("transport/defaults.jl")       # Optimised parameter configurations
include("transport/data_access.jl")    # Artifact helpers for ERA5 data

end # module Transport

end # module NuclearDetonation
