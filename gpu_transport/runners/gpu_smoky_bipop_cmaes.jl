#!/usr/bin/env julia
# Smoky GPU BIPOP-CMA-ES driver — Phase D stub.
# ===============================================
# Mirrors gpu_nancy_bipop_cmaes.jl but for the Smoky 23-parameter optimisation.
# Wire-up is parked until Nancy validation is fully landed; this file exists
# so the directory structure is complete and the include graph compiles.
#
# To activate:
#   1. Confirm the upstream `smoky_cmaes_particle_size.jl` (or equivalent)
#      exists and exports SMOKY_PARAM_NAMES, SMOKY_LB/UB, MET_CACHE, etc.
#   2. Set ENV["MAX_EVALS"]="0" before include like the Nancy driver does.
#   3. Replace `rho_core_gpu` below with a Smoky version that calls
#      `generate_smoky_particles(...)` from gpu_transport/smoky_source.jl.
#   4. Reuse the BIPOP loop verbatim from gpu_nancy_bipop_cmaes.jl.

error("Phase D Smoky driver is a stub — Nancy must be validated and tuned first.")
