#!/usr/bin/env julia
# Hash-match test for the Track A native integrator.
# ====================================================
# For 5 random parameter sets sampled within LB..UB, plus the optimised
# Nancy parameters, asserts that
#
#     run_native_simulation(p, seed).dep_hash ==
#     run_reference_simulation(p, seed).dep_hash
#
# Also prints a wall-time comparison so we can see the actual speedup.
#
# Run:
#   julia --project=/home/marc/NuclearDetonation.jl \
#       gpu_transport/test_native_integrator.jl

using Printf
using Random

ENV["MAX_EVALS"] = "0"

println("[test_native] including upstream cmaes script (loop suppressed)")
const _UPSTREAM = "/home/marc/NuclearDetonation.jl/examples/nancy_cmaes_particle_size.jl"
include(_UPSTREAM)

include(joinpath(@__DIR__, "cpu_reference.jl"))
include(joinpath(@__DIR__, "cpu_native_integrator.jl"))

const NANCY_PARAMS_F64 = Float64[
    127.552, 2.669, 141.861, 2.523, 0.8652,
    0.05617, 0.35074,
    4.028, 2.220, 0.2055, 4.458,
    4.397, 0.5532, 2.557, 4.105, 1.290,
    1.554, 1.174,
    48.418, 2.172,
]

const SEED = UInt64(0xBEEF_F00D)

# Build a deterministic set of 5 random params + 1 optimised set.
function random_params(rng::MersenneTwister)
    p = LB .+ rand(rng, length(LB)) .* (UB .- LB)
    return p
end

rng_param = MersenneTwister(0xC0FFEE)
param_sets = Vector{Vector{Float64}}()
push!(param_sets, NANCY_PARAMS_F64)
for _ in 1:5
    push!(param_sets, random_params(rng_param))
end

println("\n[test_native] warm-up calls (JIT)...")
_ = run_reference_simulation(NANCY_PARAMS_F64, SEED)
_ = run_native_simulation(NANCY_PARAMS_F64, SEED)
println("  warm-up done.")

println("\n[test_native] running $(length(param_sets)) param sets ...\n")
@printf "  %-3s  %-13s  %-13s  %-9s  %-9s  %s\n" "#" "ref hash[12]" "native hash[12]" "ref wall" "nat wall" "match"
println("  " * "─"^70)

n_pass = 0
total_ref_t = 0.0
total_nat_t = 0.0

for (k, p) in enumerate(param_sets)
    global n_pass, total_ref_t, total_nat_t
    t_ref = @elapsed ref = run_reference_simulation(p, SEED)
    t_nat = @elapsed nat = run_native_simulation(p, SEED)
    total_ref_t += t_ref
    total_nat_t += t_nat
    match = ref.dep_hash == nat.dep_hash
    if match; n_pass += 1; end
    mark = match ? "✓" : "✗"
    @printf "  %-3d  %-13s  %-13s  %5.2fs     %5.2fs     %s\n" k ref.dep_hash[1:12] nat.dep_hash[1:12] t_ref t_nat mark
    if !match
        @printf "       n_dep_events: ref=%d nat=%d\n" ref.n_dep_events nat.n_dep_events
        @printf "       total_dep_bq: ref=%.6e nat=%.6e\n" ref.total_dep_bq nat.total_dep_bq
    end
end

println("\n  " * "─"^70)
@printf "  PASSED: %d / %d\n" n_pass length(param_sets)
@printf "  Wall:   ref total=%.1fs   nat total=%.1fs   speedup=%.2fx\n" total_ref_t total_nat_t (total_ref_t / total_nat_t)

if n_pass == length(param_sets)
    println("\n  ✓ HASH MATCH on all $(length(param_sets)) param sets")
    exit(0)
else
    println("\n  ✗ HASH MISMATCH on $(length(param_sets) - n_pass) param sets")
    exit(1)
end
