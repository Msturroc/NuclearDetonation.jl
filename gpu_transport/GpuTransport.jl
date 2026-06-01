# GpuTransport — GPU forward simulator for NuclearDetonation.jl Nancy/Smoky.
# ===========================================================================
# Standalone module living in /home/marc/julia_snap_explorations/.
# Does NOT modify NuclearDetonation.jl; reuses it only as a library boundary
# (Transport.MeteoFields, Transport.nancy_era5_files, ...).
#
# Public surface:
#   - load_nancy_gpu_met()                      → (GpuMet, files, start_dt)
#   - generate_nancy_particles(rng, n, params)  → ParticleHost
#   - run_gpu_simulation(particles, met, dep_geom, level_z; params, ...)
#   - run_shadow_simulation(...)                — Float32 CPU mirror
#
# Use:
#   include("/home/marc/julia_snap_explorations/gpu_transport/GpuTransport.jl")
#   using .GpuTransport

module GpuTransport

# Hanna stays — the rest of the v1 stack (kernels.jl, cpu_shadow.jl,
# met_upload.jl, sigma_coords.jl, source_release.jl, validate_gpu_vs_cpu.jl)
# was deleted along with its baked-in physics approximations. Track A/B
# rewrites land in this folder over time and get re-included here.
include("hanna.jl")

# Note: host_shadow_v2.jl is included separately by callers because it
# depends on upstream cmaes-script bindings (MET_CACHE, DOMAIN, NX, NY, NK,
# RELEASE_X, RELEASE_Y, LAYER_*, generate_bimodal_bins, etc.) that aren't
# importable from this module's lexical scope. Same for cpu_reference.jl,
# cpu_native_integrator.jl and validate_against_reference.jl.

export HannaCfgF32, hanna_f32, hanna_dev, ou_step_f32, default_hanna_cfg

end  # module
