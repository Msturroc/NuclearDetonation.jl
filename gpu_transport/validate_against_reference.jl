#!/usr/bin/env julia
# Shared validation harness for Track B
# =====================================
# Compares any candidate forward-sim implementation against `cpu_reference.jl`
# via FMS over the 6 dose-rate observation thresholds. Used by both B2's
# `test_host_shadow.jl` and B3's `test_gpu_kernel.jl`.
#
# Assumes the upstream `nancy_cmaes_particle_size.jl` has been included and
# `cpu_reference.jl` is loaded. Constants accessed from outer scope:
#   LON_GRID, LAT_GRID, OBS_MASKS, DOMAIN, DOSE_FACTOR, gaussian_smooth.

using Random
using Statistics
using Printf

# Bin a deposition_log (Vector{DepositionEvent}) into the (LON_GRID, LAT_GRID)
# observation grid. Mirrors gpu_nancy_bipop_cmaes_v2.jl::rho_core_native lines 60-78.
function bin_reference_to_grid(deposition_log)
    nx_obs, ny_obs = length(LON_GRID), length(LAT_GRID)
    grid = zeros(Float64, nx_obs, ny_obs)
    @inbounds for evt in deposition_log
        lat, lon = Transport.grid_to_latlon(DOMAIN, evt.x, evt.y)
        if lon > 180.0
            lon -= 360.0
        end
        i = searchsortedlast(LON_GRID, lon)
        j = searchsortedlast(LAT_GRID, lat)
        if 1 <= i <= nx_obs && 1 <= j <= ny_obs
            grid[i, j] += evt.mass
        end
    end
    return grid
end

# Stable, sorted ordering of OBS_MASKS thresholds (Dict iteration is unordered).
function ordered_obs_masks()
    return sort(collect(pairs(OBS_MASKS)); by = first)
end

# Compute FMS at each of the 6 dose-rate thresholds for a smoothed dose grid.
function fms_per_threshold(dose_smooth_mRh::AbstractMatrix)
    pairs_sorted = ordered_obs_masks()
    n = length(pairs_sorted)
    fms = zeros(Float64, n)
    @inbounds for (k, (dose_rate, obs_mask)) in enumerate(pairs_sorted)
        if sum(obs_mask) == 0
            fms[k] = 0.0
            continue
        end
        model_mask = dose_smooth_mRh .>= dose_rate
        inter = Float64(sum(model_mask .& obs_mask))
        uni   = Float64(sum(model_mask .| obs_mask))
        fms[k] = uni > 0 ? inter / uni : 0.0
    end
    return fms
end

# Sample param sets used by both B2 and B3 (Nancy + 5 random within bounds).
function sample_param_sets()
    sets = Vector{Vector{Float64}}()

    # Set 1: Nancy optimised from gpu_nancy_cmaes_v2_best.txt
    best_file = "/home/marc/julia_snap_explorations/gpu_nancy_cmaes_v2_best.txt"
    nancy_optimal = if isfile(best_file)
        param_dict = Dict{String, Float64}()
        for line in eachline(best_file)
            startswith(line, "#") && continue
            parts = split(line, '\t')
            length(parts) >= 2 || continue
            param_dict[parts[1]] = parse(Float64, parts[2])
        end
        [param_dict[name] for name in PARAM_NAMES]
    else
        copy(WARM_START_PARAMS)
    end
    push!(sets, nancy_optimal)

    # Sets 2-6: random within LB..UB using the same seed Track A used
    rng = Random.MersenneTwister(0xC0FFEE)
    for _ in 1:5
        p = LB .+ rand(rng, length(LB)) .* DOMAIN_WIDTH
        push!(sets, p)
    end
    return sets
end

# Sample seeds — fixed list of UInt64 so runs are reproducible.
function sample_seeds(n::Int)
    rng = Random.MersenneTwister(0xCAFE)
    return [rand(rng, UInt64) for _ in 1:n]
end

"""
    validate_implementation(f::Function;
                            param_sets::Vector{Vector{Float64}},
                            seeds::Vector{UInt64},
                            fms_tolerance::Float64 = 0.005,
                            label::String = "candidate")

For each (params, seed):
  1. ref_grid    = bin_reference_to_grid(run_reference_simulation(p, s))
  2. cand_grid   = f(p, s)         — must return a 2D Matrix{<:Real} of shape (nx_obs, ny_obs)
  3. smooth both with gaussian_smooth(σ = params[20])
  4. compute FMS at 6 thresholds
  5. assert max(Δ) ≤ fms_tolerance
Returns (passed::Bool, deltas::Matrix{Float64}) where deltas[k, i] is
fms_ref[k] - fms_cand[k] for set i.
"""
function validate_implementation(f::Function;
                                 param_sets::Vector{Vector{Float64}},
                                 seeds::Vector{UInt64},
                                 fms_tolerance::Float64 = 0.005,
                                 label::String = "candidate")
    n_sets = length(param_sets)
    pairs_sorted = ordered_obs_masks()
    n_thresholds = length(pairs_sorted)
    deltas = zeros(Float64, n_thresholds, n_sets)
    all_pass = true

    println("\n" * "="^72)
    println("VALIDATION HARNESS — $(label)")
    println("="^72)
    println("Parameter sets: $(n_sets), seeds: $(length(seeds)), FMS tolerance: $(fms_tolerance)")

    for (idx, (params, seed)) in enumerate(zip(param_sets, seeds))
        smooth_sigma = params[20]
        @printf("\n[Set %d] seed=%016x  smooth=%.3f\n", idx, seed, smooth_sigma)

        # --- Reference ---
        t1 = time()
        ref = run_reference_simulation(params, seed)
        t_ref = time() - t1
        ref_grid = bin_reference_to_grid(ref.deposition_log)
        ref_grid_mRh = ref_grid .* DOSE_FACTOR
        ref_smooth = gaussian_smooth(ref_grid_mRh, smooth_sigma)
        fms_ref = fms_per_threshold(ref_smooth)

        # --- Candidate ---
        t2 = time()
        cand_grid = f(params, seed)
        t_cand = time() - t2
        cand_mRh = Float64.(cand_grid) .* DOSE_FACTOR
        cand_smooth = gaussian_smooth(cand_mRh, smooth_sigma)
        fms_cand = fms_per_threshold(cand_smooth)

        @printf("  reference: %.2fs  |  candidate: %.2fs  (%.2fx)\n",
                t_ref, t_cand, t_ref / max(t_cand, 1e-9))
        println("  threshold  ref_fms   cand_fms   Δ (ref-cand)")
        for k in 1:n_thresholds
            dose_rate = pairs_sorted[k][1]
            d = fms_ref[k] - fms_cand[k]
            deltas[k, idx] = d
            marker = abs(d) > fms_tolerance ? "  ✗" : "  ✓"
            @printf("    %7.3f  %.4f    %.4f     %+.4f%s\n",
                    dose_rate, fms_ref[k], fms_cand[k], d, marker)
        end

        worst = maximum(abs, deltas[:, idx])
        if worst > fms_tolerance
            all_pass = false
            @printf("  ✗ Set %d FAIL: worst |Δ| = %.4f > %.4f\n", idx, worst, fms_tolerance)
        else
            @printf("  ✓ Set %d PASS: worst |Δ| = %.4f\n", idx, worst)
        end
    end

    println("\n" * "="^72)
    if all_pass
        println("✓ VALIDATION PASSED — all $(n_sets) sets within FMS tolerance $(fms_tolerance)")
    else
        worst_overall = maximum(abs, deltas)
        @printf("✗ VALIDATION FAILED — worst |Δ| = %.4f > %.4f\n", worst_overall, fms_tolerance)
    end
    println("="^72)
    return all_pass, deltas
end
