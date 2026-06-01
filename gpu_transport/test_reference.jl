#!/usr/bin/env julia
# Reference shadow determinism test
# ==================================
# Confirms that `run_reference_simulation` produces byte-identical output
# when called twice with the same params + seed.
#
# This is the prerequisite for using the reference as ground truth — if the
# package's Transport.run_simulation! is non-deterministic even with a seeded
# global RNG, we have to fix that before any port-comparison work.
#
# Run:
#   julia --project=/home/marc/NuclearDetonation.jl gpu_transport/test_reference.jl

using Printf

# Suppress upstream optimisation loop, then include for constants/helpers.
ENV["MAX_EVALS"] = "0"

println("[test_reference] including upstream cmaes script (loop suppressed)")
const _UPSTREAM = "/home/marc/NuclearDetonation.jl/examples/nancy_cmaes_particle_size.jl"
include(_UPSTREAM)

include(joinpath(@__DIR__, "cpu_reference.jl"))

# Use the optimised Nancy params (same as nancy_optimised_config) — these are
# the values the user normally runs at, so the test exercises a representative
# trajectory.
const NANCY_PARAMS_F64 = Float64[
    127.552, 2.669, 141.861, 2.523, 0.8652,   # particle size
    0.05617, 0.35074,                          # layer fractions
    4.028, 2.220, 0.2055, 4.458,              # turbulence
    4.397, 0.5532, 2.557, 4.105, 1.290,       # physics
    1.554, 1.174,                              # deposition
    48.418, 2.172,                             # calibration
]

const SEED = UInt64(0xBEEF_F00D)

println("\n[test_reference] running reference simulation #1 ...")
t1 = @elapsed run1 = run_reference_simulation(NANCY_PARAMS_F64, SEED)
@printf "  wall=%.2fs  particles=%d  events=%d  total=%.3e Bq\n" t1 run1.n_particles run1.n_dep_events run1.total_dep_bq
@printf "  hash=%s\n" run1.dep_hash

println("\n[test_reference] running reference simulation #2 (same seed) ...")
t2 = @elapsed run2 = run_reference_simulation(NANCY_PARAMS_F64, SEED)
@printf "  wall=%.2fs  particles=%d  events=%d  total=%.3e Bq\n" t2 run2.n_particles run2.n_dep_events run2.total_dep_bq
@printf "  hash=%s\n" run2.dep_hash

println("\n" * "─"^60)
if run1.dep_hash == run2.dep_hash
    println("✓ DETERMINISM PASS — same seed → same deposition log")
else
    println("✗ DETERMINISM FAIL — same seed → different output")
    println("  Inspecting first 5 events of each run:")
    for i in 1:min(5, length(run1.deposition_log), length(run2.deposition_log))
        @printf "    [%d] r1: x=%.6f y=%.6f m=%.3e t=%.1f c=%d\n" i run1.deposition_log[i].x run1.deposition_log[i].y run1.deposition_log[i].mass run1.deposition_log[i].time run1.deposition_log[i].component
        @printf "        r2: x=%.6f y=%.6f m=%.3e t=%.1f c=%d\n"   run2.deposition_log[i].x run2.deposition_log[i].y run2.deposition_log[i].mass run2.deposition_log[i].time run2.deposition_log[i].component
    end
end

println("\n[test_reference] running with a DIFFERENT seed for sanity ...")
run3 = run_reference_simulation(NANCY_PARAMS_F64, UInt64(0xDEAD_BEEF))
@printf "  hash=%s\n" run3.dep_hash
if run3.dep_hash != run1.dep_hash
    println("  ✓ different seed → different output (as expected)")
else
    println("  ✗ different seed gave SAME output — RNG seeding is broken")
end
