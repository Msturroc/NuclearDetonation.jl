#!/usr/bin/env julia
# Throwaway benchmark: GPU forward-sim (run_gpu_shadow) wall time vs particle count.
# Used to ETA the 6-test BIPOP-CMA-ES campaign (6000 evals/test, 10k particles).
# Run from repo root:  julia --project gpu_transport/bench_particle_scaling.jl

ENV["MAX_EVALS"] = "0"            # suppress upstream CMA-ES loop
using Random, Statistics, Printf, StaticArrays, CUDA
using NuclearDetonation
using NuclearDetonation.Transport

const ROOT = "/home/marc/NuclearDetonation.jl"
include(joinpath(ROOT, "examples", "nancy_cmaes_particle_size.jl"))   # LB/UB/N_DIM/WARM_START_PARAMS/grids
include(joinpath(ROOT, "gpu_transport", "host_shadow_v2.jl"))
include(joinpath(ROOT, "gpu_transport", "met_upload.jl"))
include(joinpath(ROOT, "gpu_transport", "gpu_kernel_v2.jl"))

println("[bench] uploading Nancy met to GPU…")
const GPU_WINDOWS = load_nancy_gpu_windows()
println("[bench] ", length(GPU_WINDOWS), " met windows; GPU = ", CUDA.name(CUDA.device()))

warm = copy(WARM_START_PARAMS)

# Global JIT warm-up (first launch compiles the kernel)
ENV["N_PARTICLES"] = "1000"
run_gpu_shadow(warm, UInt64(0xDEADBEEF); windows = GPU_WINDOWS)
CUDA.synchronize()

function time_eval(N::Int; reps::Int = 8)
    ENV["N_PARTICLES"] = string(N)
    # warm-up at this N
    _, _, na = run_gpu_shadow(warm, UInt64(1); windows = GPU_WINDOWS)
    CUDA.synchronize()
    ts = Float64[]
    nalive = 0
    for r in 1:reps
        CUDA.synchronize()
        t0 = time()
        _, _, na2 = run_gpu_shadow(warm, UInt64(100 + r); windows = GPU_WINDOWS)
        CUDA.synchronize()
        push!(ts, time() - t0)
        nalive = na2
    end
    return (N = N, mean_s = mean(ts), min_s = minimum(ts), max_s = maximum(ts), n_alive = nalive)
end

println("\n", "="^64)
@printf("%8s  %10s  %10s  %10s  %10s\n", "N_part", "mean s", "min s", "max s", "n_alive")
println("="^64)
results = NamedTuple[]
for N in (1000, 2000, 5000, 10000)
    r = time_eval(N)
    push!(results, r)
    @printf("%8d  %10.4f  %10.4f  %10.4f  %10d\n", r.N, r.mean_s, r.min_s, r.max_s, r.n_alive)
end
println("="^64)

# ETA projection
println("\nETA projection (per-test budget = 6000 evals; 6 tests):")
for r in results
    per_test_min = r.mean_s * 6000 / 60
    campaign_h   = r.mean_s * 6000 * 6 / 3600
    @printf("  N=%5d : %.3f s/eval  ->  %.1f min/test  ->  %.2f h for 6 tests\n",
            r.N, r.mean_s, per_test_min, campaign_h)
end
println("\n[bench] done.")
