#!/usr/bin/env julia
# Nancy BIPOP-CMA-ES driver — Track B v3 (GPU forward simulator)
# ===============================================================
# Sibling of gpu_nancy_bipop_cmaes_v2.jl (Track A native CPU integrator).
# Replaces the per-candidate threaded CPU evaluator with a GPU kernel launch
# (single-candidate per launch). Met windows are uploaded once at startup.
#
# Run:
#   MAX_EVALS=1000 julia --project=/home/marc/NuclearDetonation.jl \
#       /home/marc/julia_snap_explorations/gpu_nancy_bipop_cmaes_v3.jl
#
# Output:
#   - /home/marc/julia_snap_explorations/gpu_nancy_cmaes_v3_best.txt
#   - /home/marc/julia_snap_explorations/gpu_nancy_cmaes_v3_results.txt

const _USER_MAX_EVALS = parse(Int, get(ENV, "MAX_EVALS", "1000"))
ENV["MAX_EVALS"] = "0"

using Random
using Statistics
using Printf
using StaticArrays
using CUDA
using NuclearDetonation
using NuclearDetonation.Transport

const _UPSTREAM = "/home/marc/NuclearDetonation.jl/examples/nancy_cmaes_particle_size.jl"
println("[v3] including upstream cmaes script (loop suppressed)")
include(_UPSTREAM)

const MAX_EVALS_V3 = _USER_MAX_EVALS

include(joinpath(@__DIR__, "gpu_transport", "host_shadow_v2.jl"))
include(joinpath(@__DIR__, "gpu_transport", "met_upload.jl"))
include(joinpath(@__DIR__, "gpu_transport", "gpu_kernel_v2.jl"))

println("[v3] uploading Nancy met to GPU…")
const GPU_WINDOWS = load_nancy_gpu_windows()
println("[v3] loaded ", length(GPU_WINDOWS), " met windows")

# ----------------------------------------------------------------------------
# rho_core_gpu — the cmaes objective using the GPU forward simulator.
# Scoring is inlined verbatim from upstream rho_core (same FMS / shape /
# extent / TOA). The GPU kernel hands back hourly_dep snapshots which we feed
# directly to the scorer.
# ----------------------------------------------------------------------------
function rho_core_gpu(params::Vector{Float64}, gen_seed::UInt64)
    smooth_sigma = params[20]

    final_dep, hourly_dep, n_alive = run_gpu_shadow(params, gen_seed; windows = GPU_WINDOWS)
    nx_obs, ny_obs = length(LON_GRID), length(LAT_GRID)

    if sum(final_dep) <= 0
        return (loss = 1.0, fms = 0.0, shape = 0.0, extent = 0.0, toa = 0.0, combined_old = 0.0)
    end

    # 12 hourly model snapshots for TOA scoring
    model_snapshots = Vector{Matrix{Float64}}(undef, 12)
    snapshot_hours  = Float64[]
    for h in 1:12
        model_snapshots[h] = Float64.(view(hourly_dep, :, :, h))
        push!(snapshot_hours, Float64(h))
    end

    final_dose = model_snapshots[end]
    if sum(final_dose) <= 0
        return (loss = 1.0, fms = 0.0, shape = 0.0, extent = 0.0, toa = 0.0, combined_old = 0.0)
    end

    final_dose_mRh = final_dose .* DOSE_FACTOR
    dose_smooth = gaussian_smooth(final_dose_mRh, smooth_sigma)

    fms_scores = Float64[]
    shape_scores = Float64[]
    for (dose_rate, obs_mask) in OBS_MASKS
        if sum(obs_mask) == 0
            push!(fms_scores, 0.0); push!(shape_scores, 0.0); continue
        end
        model_mask = dose_smooth .>= dose_rate
        inter = Float64(sum(model_mask .& obs_mask))
        uni   = Float64(sum(model_mask .| obs_mask))
        push!(fms_scores, uni > 0 ? inter / uni : 0.0)

        obs_shape   = get(OBS_SHAPES, dose_rate, nothing)
        model_shape = sum(model_mask) > 0 ? inertia_ellipse(model_mask, LAT_GRID, LON_GRID) : nothing
        if !isnothing(obs_shape) && !isnothing(model_shape)
            ar_score = min(model_shape.ar, obs_shape.ar) / max(model_shape.ar, obs_shape.ar)
            orient_score = cos(model_shape.angle - obs_shape.angle)^2
            push!(shape_scores, 0.7 * ar_score + 0.3 * orient_score)
        else
            push!(shape_scores, 0.0)
        end
    end

    geo_mean_fms   = exp(mean(log(max(s, 0.005)) for s in fms_scores))
    geo_mean_shape = exp(mean(log(max(s, 0.005)) for s in shape_scores))

    model_max_dist_km = 0.0
    for i in 1:nx_obs, j in 1:ny_obs
        if final_dose[i, j] > 0
            dlat = LAT_GRID[j] - SOURCE_LAT
            dlon = (LON_GRID[i] - SOURCE_LON) * cosd(SOURCE_LAT)
            model_max_dist_km = max(model_max_dist_km, sqrt(dlat^2 + dlon^2) * 111.0)
        end
    end
    extent_score = clamp(model_max_dist_km / OBS_MAX_DIST_KM, 0.0, 1.0)

    model_snapshots_norm = [let s = sum(snap); s > 0 ? snap ./ s : snap; end for snap in model_snapshots]
    toa_result = Transport.compute_toa_score(model_snapshots_norm, snapshot_hours,
        NANCY_OBS.toa_contours, LAT_GRID, LON_GRID; threshold_fraction = 0.01)
    toa_score = if isnothing(toa_result) || isinf(toa_result.mean_arrival_error_hours)
        0.0
    else
        max(0.0, 1.0 - toa_result.mean_arrival_error_hours / 6.0)
    end

    combined     = 0.35 * geo_mean_fms + 0.20 * geo_mean_shape + 0.15 * extent_score + 0.30 * toa_score
    combined_old = 0.50 * geo_mean_fms + 0.20 * extent_score + 0.30 * toa_score

    return (loss = 1.0 - combined,
            fms = geo_mean_fms,
            shape = geo_mean_shape,
            extent = extent_score,
            toa = toa_score,
            combined_old = combined_old)
end

function evaluate_generation_gpu(candidates::Vector{Vector{Float64}}, gen_seed::UInt64)
    n = length(candidates)
    results = Vector{EvalResult}(undef, n)
    # Single-candidate per launch — kernel launch overhead is tiny (~10 μs)
    # and each particle batch is small enough that we can keep the GPU busy
    # serially. If this becomes the bottleneck, B4 batches multi-candidate.
    for i in 1:n
        try
            r = rho_core_gpu(candidates[i], gen_seed)
            results[i] = EvalResult(r.loss, r.fms, r.shape, r.extent, r.toa, r.combined_old)
        catch e
            @warn "GPU evaluation failed for candidate $i" exception=(e, catch_backtrace())
            results[i] = FAILED_EVAL
        end
    end
    return results
end

# ============================================================================
# JIT warm-up — compiles the GPU kernel once before timing the BIPOP loop
# ============================================================================
println("[v3] JIT warm-up…")
let warm = copy(WARM_START_PARAMS)
    _ = rho_core_gpu(warm, UInt64(0xDEADBEEF))
end
CUDA.synchronize()
println("[v3] warm-up done.")

# ============================================================================
# Main BIPOP loop — verbatim from v2, calls GPU evaluator
# ============================================================================
println("\n" * "="^70)
println("NANCY GPU BIPOP-CMA-ES   (budget: $(MAX_EVALS_V3) evals)")
println("="^70)

x0 = if isnothing(load_checkpoint_params(lowercase(string(:OU))))
    println("    no upstream checkpoint found, starting from WARM_START_PARAMS")
    copy(WARM_START_PARAMS)
else
    println("    using upstream OU checkpoint as warm start")
    load_checkpoint_params(lowercase(string(:OU)))
end

global_best_val = Inf
global_best_x = copy(x0)
global_best_diag = FAILED_EVAL
total_evals = 0
budget_large = 0; budget_small = 0
restart_count = 0
large_lambda = DEFAULT_LAMBDA

results_file    = "/home/marc/julia_snap_explorations/gpu_nancy_cmaes_v3_results.txt"
checkpoint_file = "/home/marc/julia_snap_explorations/gpu_nancy_cmaes_v3_best.txt"

t_start = time()
while total_evals < MAX_EVALS_V3
    global total_evals, global_best_val, global_best_x, global_best_diag
    global restart_count, large_lambda, budget_large, budget_small

    if restart_count == 0
        run_lambda = DEFAULT_LAMBDA
        run_type = :large
        run_sigma_frac = SIGMA_FRAC
        run_x0 = copy(x0)
    elseif budget_large <= budget_small
        large_lambda = min(large_lambda * 2, MAX_EVALS_V3 ÷ 10)
        run_lambda = large_lambda
        run_type = :large
        run_sigma_frac = SIGMA_FRAC
        run_x0 = copy(global_best_x)
    else
        run_lambda = DEFAULT_LAMBDA
        run_type = :small
        run_sigma_frac = SIGMA_FRAC * 10.0^(-2.0 * rand())
        mix = 0.3 + 0.4 * rand()
        run_x0 = mix .* global_best_x .+ (1.0 - mix) .* (LB .+ rand(N_DIM) .* DOMAIN_WIDTH)
        run_x0 .= clamp.(run_x0, LB, UB)
    end

    remaining = MAX_EVALS_V3 - total_evals
    remaining < run_lambda && break
    restart_count += 1
    run_evals = 0

    println("\n" * "-"^50)
    println("RESTART #$(restart_count) ($(run_type), λ=$(run_lambda), σ_frac=$(round(run_sigma_frac, digits=4)))")
    println("-"^50)

    es = CMAES(run_x0; lb = LB, ub = UB, popsize = run_lambda, sigma_frac = run_sigma_frac)
    es.best_ever_val = global_best_val
    es.best_ever_x = copy(global_best_x)

    while total_evals + run_lambda <= MAX_EVALS_V3
        gen_seed = rand(UInt64)
        candidates = ask(es)
        eval_results = evaluate_generation_gpu(candidates, gen_seed)
        fitvals = [r.loss for r in eval_results]
        gen_best_val, gen_best_x = tell!(es, candidates, fitvals)
        total_evals += run_lambda
        run_evals += run_lambda

        gen_best_idx = argmin(fitvals)
        gen_best_r = eval_results[gen_best_idx]

        improved = false
        if gen_best_val < global_best_val
            global_best_val = gen_best_val
            global_best_x = copy(gen_best_x)
            global_best_diag = gen_best_r
            improved = true
            open(checkpoint_file, "w") do f
                for (j, pname) in enumerate(PARAM_NAMES)
                    println(f, "$(pname)\t$(global_best_x[j])")
                end
                println(f, "# loss\t$(global_best_val)")
                println(f, "# fms\t$(global_best_diag.fms)")
                println(f, "# shape\t$(global_best_diag.shape)")
                println(f, "# extent\t$(global_best_diag.extent)")
                println(f, "# toa\t$(global_best_diag.toa)")
            end
        end

        marker = improved ? " ***" : ""
        elapsed = time() - t_start
        @printf("  Gen %3d [%5d/%d] FMS=%.2f shp=%.2f ext=%.2f toa=%.2f | new=%.1f%% | σ=%.3f [%.0fs]%s\n",
                es.generation, total_evals, MAX_EVALS_V3,
                gen_best_r.fms, gen_best_r.shape, gen_best_r.extent, gen_best_r.toa,
                (1.0 - gen_best_val) * 100, es.sigma, elapsed, marker)
        flush(stdout)

        do_restart, reason = should_restart(es)
        if do_restart
            println("  -> Restart: $(reason)")
            break
        end
    end

    if run_type == :large
        budget_large += run_evals
    else
        budget_small += run_evals
    end
    println("  Run used $(run_evals) evals.  Budget: large=$(budget_large), small=$(budget_small)")
end

t_elapsed = time() - t_start
println("\n" * "="^70)
println("GPU BIPOP-CMA-ES COMPLETE")
println("="^70)
@printf "Total evaluations: %d\n" total_evals
@printf "Restarts:          %d\n" restart_count
@printf "Wall time:         %.1f minutes\n" (t_elapsed / 60)
@printf "Best loss:         %.6f\n" global_best_val
@printf "Best score:        %.2f%%\n" ((1.0 - global_best_val) * 100)

# Defensive final write — guarantees the checkpoint file matches the in-memory
# best when the script terminates normally, even if an earlier run raced on it.
open(checkpoint_file, "w") do f
    for (j, pname) in enumerate(PARAM_NAMES)
        println(f, "$(pname)\t$(global_best_x[j])")
    end
    println(f, "# loss\t$(global_best_val)")
    println(f, "# fms\t$(global_best_diag.fms)")
    println(f, "# shape\t$(global_best_diag.shape)")
    println(f, "# extent\t$(global_best_diag.extent)")
    println(f, "# toa\t$(global_best_diag.toa)")
end

open(results_file, "w") do f
    println(f, "Nancy GPU BIPOP-CMA-ES Results (Track B v3)")
    println(f, "="^60)
    println(f, "Total evaluations: $(total_evals)")
    println(f, "Restarts: $(restart_count)")
    println(f, "Wall time: $(round(t_elapsed/60, digits=1)) minutes")
    println(f, "Best loss: $(global_best_val)")
    println(f, "Best score: $(round((1.0 - global_best_val) * 100, digits=2))%")
    println(f, "\nBest parameters:")
    for (j, pname) in enumerate(PARAM_NAMES)
        println(f, "  $(pname): $(global_best_x[j])")
    end
end
println("\nSaved $(results_file)")
println("Saved $(checkpoint_file)")
