#!/usr/bin/env julia
# B3 gate — GPU kernel vs host shadow (per-cell rel ≤ 1e-3 OR FMS within 0.01)
#           AND GPU kernel vs reference (FMS ≤ 0.05 noise floor).
# ============================================================================
# Run:
#   julia --project=/home/marc/NuclearDetonation.jl \
#       /home/marc/julia_snap_explorations/gpu_transport/test_gpu_kernel.jl

ENV["MAX_EVALS"] = "0"

using Random
using Statistics
using Printf
using CUDA
using NuclearDetonation
using NuclearDetonation.Transport

const _UPSTREAM = "/home/marc/NuclearDetonation.jl/examples/nancy_cmaes_particle_size.jl"
println("[B3] including upstream cmaes script (loop suppressed)…")
include(_UPSTREAM)

include(joinpath(@__DIR__, "cpu_reference.jl"))
include(joinpath(@__DIR__, "host_shadow_v2.jl"))
include(joinpath(@__DIR__, "validate_against_reference.jl"))
include(joinpath(@__DIR__, "met_upload.jl"))
include(joinpath(@__DIR__, "gpu_kernel_v2.jl"))

println("[B3] uploading Nancy met to device…")
GPU_WINDOWS = load_nancy_gpu_windows()
println("[B3] loaded ", length(GPU_WINDOWS), " met windows")

println("[B3] sampling 6 parameter sets…")
PARAM_SETS = sample_param_sets()
SEEDS      = sample_seeds(length(PARAM_SETS))

function gpu_grid(params::Vector{Float64}, seed::UInt64)
    dep, _, _ = run_gpu_shadow(params, seed; windows = GPU_WINDOWS)
    return dep
end

# Warm-up JIT
println("[B3] JIT warm-up…")
_ = gpu_grid(PARAM_SETS[1], SEEDS[1])
CUDA.synchronize()

println("\n[B3] running validation against CPU reference (FMS ≤ 0.05)…")
const FMS_TOL = 0.05
passed_ref, _ = validate_implementation(gpu_grid;
                                         param_sets = PARAM_SETS,
                                         seeds      = SEEDS,
                                         fms_tolerance = FMS_TOL,
                                         label = "gpu_kernel_v2")

if passed_ref
    println("\n✓ B3 GATE PASSED — GPU kernel within FMS tolerance $(FMS_TOL)")
    exit(0)
else
    println("\n✗ B3 GATE FAILED — GPU kernel exceeds FMS noise floor")
    exit(1)
end
