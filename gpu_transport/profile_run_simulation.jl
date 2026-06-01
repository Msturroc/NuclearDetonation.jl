#!/usr/bin/env julia
# Profile Transport.run_simulation! — A1 of the Nancy speed-up plan.
# ====================================================================
# Loads upstream cmaes (with MAX_EVALS=0) so MET_CACHE/DOMAIN are bound,
# warms up the JIT with one call to run_reference_simulation, then runs
# a second call under Profile.@profile and dumps a flat profile.
#
# Decision gate: if >50% of samples land in DiffEq overhead + per-step
# allocation, Track A is worth its full budget. If <30%, skip Track A
# and pivot straight to Track B.
#
# Run:
#   julia --project=/home/marc/NuclearDetonation.jl \
#       gpu_transport/profile_run_simulation.jl

using Printf
using Profile

ENV["MAX_EVALS"] = "0"

println("[profile] including upstream cmaes script (loop suppressed)")
const _UPSTREAM = "/home/marc/NuclearDetonation.jl/examples/nancy_cmaes_particle_size.jl"
include(_UPSTREAM)

include(joinpath(@__DIR__, "cpu_reference.jl"))

const NANCY_PARAMS_F64 = Float64[
    127.552, 2.669, 141.861, 2.523, 0.8652,
    0.05617, 0.35074,
    4.028, 2.220, 0.2055, 4.458,
    4.397, 0.5532, 2.557, 4.105, 1.290,
    1.554, 1.174,
    48.418, 2.172,
]

const SEED = UInt64(0xBEEF_F00D)

println("\n[profile] warm-up call (JIT)...")
t_warm = @elapsed run_reference_simulation(NANCY_PARAMS_F64, SEED)
@printf "  warm-up wall=%.2fs\n" t_warm

println("\n[profile] hot call x3 for timing baseline...")
times = Float64[]
for k in 1:3
    push!(times, @elapsed run_reference_simulation(NANCY_PARAMS_F64, SEED))
end
@printf "  hot wall=%.2fs / %.2fs / %.2fs   mean=%.2fs\n" times[1] times[2] times[3] (sum(times)/3)

# Allocation count for one call.
println("\n[profile] @allocated for one call ...")
alloc_bytes = @allocated run_reference_simulation(NANCY_PARAMS_F64, SEED)
@printf "  allocated %.1f MiB / call\n" (alloc_bytes / (1024 * 1024))

println("\n[profile] running 3 hot iterations under Profile.@profile ...")
Profile.clear()
Profile.init(n = 10_000_000, delay = 0.001)
@profile begin
    for _ in 1:3
        run_reference_simulation(NANCY_PARAMS_F64, SEED)
    end
end

flat_path = joinpath(@__DIR__, "..", "gpu_profile_flat.txt")
tree_path = joinpath(@__DIR__, "..", "gpu_profile_tree.txt")

open(flat_path, "w") do io
    Profile.print(io; format = :flat, sortedby = :count, mincount = 5)
end
open(tree_path, "w") do io
    Profile.print(io; format = :tree, mincount = 20)
end

@printf "\n  wrote flat profile -> %s\n" flat_path
@printf "  wrote tree profile -> %s\n" tree_path

println("\n[profile] summary:")
@printf "  warm-up        : %.2fs\n" t_warm
@printf "  hot mean       : %.2fs\n" (sum(times)/3)
@printf "  alloc / call   : %.1f MiB\n" (alloc_bytes / (1024 * 1024))
println("\n  inspect $(basename(flat_path)) for hot frames; look for")
println("  DiffEq.* / OrdinaryDiffEq.* / *integrate* / GC frames.")
