#!/usr/bin/env julia
# Noise-floor probe — how much does FMS shift between two reference runs of
# the same params under different gen_seeds? Sets the achievable floor for
# Track B's host shadow / GPU kernel.
ENV["MAX_EVALS"] = "0"
using Random
using NuclearDetonation
using NuclearDetonation.Transport
include("/home/marc/NuclearDetonation.jl/examples/nancy_cmaes_particle_size.jl")
include(joinpath(@__DIR__, "cpu_reference.jl"))
include(joinpath(@__DIR__, "validate_against_reference.jl"))

PARAM_SETS = sample_param_sets()
seeds_a = sample_seeds(length(PARAM_SETS))
# Different seeds — independent RNG streams
rng = Random.MersenneTwister(0xBADC0FFEE)
seeds_b = [rand(rng, UInt64) for _ in 1:length(PARAM_SETS)]

println("\n========================================================================")
println("REFERENCE vs REFERENCE (different seeds, same params)")
println("========================================================================")

global worst = 0.0
for (idx, (params, sa, sb)) in enumerate(zip(PARAM_SETS, seeds_a, seeds_b))
    global worst
    smooth_sigma = params[20]

    ra = run_reference_simulation(params, sa)
    grid_a = bin_reference_to_grid(ra.deposition_log) .* DOSE_FACTOR
    smooth_a = gaussian_smooth(grid_a, smooth_sigma)
    fms_a = fms_per_threshold(smooth_a)

    rb = run_reference_simulation(params, sb)
    grid_b = bin_reference_to_grid(rb.deposition_log) .* DOSE_FACTOR
    smooth_b = gaussian_smooth(grid_b, smooth_sigma)
    fms_b = fms_per_threshold(smooth_b)

    println("\n[Set $idx]")
    pairs_sorted = ordered_obs_masks()
    for k in 1:length(fms_a)
        d = fms_a[k] - fms_b[k]
        worst = max(worst, abs(d))
        @printf("  %7.3f  ref_a=%.4f  ref_b=%.4f  Δ=%+.4f\n",
                pairs_sorted[k][1], fms_a[k], fms_b[k], d)
    end
end

@printf("\nWorst |Δ| across all sets/thresholds: %.4f\n", worst)
println("This is the chaotic noise floor for Track B's gate.")
