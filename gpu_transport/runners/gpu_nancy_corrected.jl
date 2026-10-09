#!/usr/bin/env julia
# Nancy GPU BIPOP-CMA-ES — CORRECTED loss (spec §1) at 10k particles
# =====================================================================
# Based on gpu_nancy_bipop_cmaes_v3.jl, with three corrections wired in:
#   1. the shared 5-component loss incl. bearing + hard gate (CalibrationShared)
#   2. log10 search space for the scale block (params 8-19)
#   3. N_PARTICLES read from env (default 10000 via host_shadow_v2.jl §2)
#
# Run from repo root:
#   MAX_EVALS=6000 N_PARTICLES=10000 julia --project \
#       gpu_transport/runners/gpu_nancy_corrected.jl
#
# Uses ROOT-anchored include paths so it can live under runners/ but still
# resolve the gpu_transport/ sources (the legacy runners assumed a scratch dir).

include(joinpath(@__DIR__, "nancy_objective.jl"))

# ===========================================================================
# BIPOP-CMA-ES in encoded (log10) space — structure from v3, decode at eval.
# ===========================================================================
println("\n" * "="^70)
println("NANCY GPU BIPOP-CMA-ES — CORRECTED LOSS   (budget: $(MAX_EVALS_C) evals)")
println("="^70)

x0_phys = if haskey(ENV, "X0_FILE")   # e.g. a previous GPU best, to compare optimisers from one start
    let vals = Dict(strip(k) => parse(Float64, v) for (k, v) in
                    (split(ln, '\t') for ln in eachline(ENV["X0_FILE"]) if !startswith(ln, "#") && !isempty(strip(ln))))
        clamp.([vals[n] for n in PARAM_NAMES], LB, UB)
    end
elseif isnothing(load_checkpoint_params("ou"))
    copy(WARM_START_PARAMS)
else
    load_checkpoint_params("ou")
end
const LB_S = LOGSP.LB_S
const UB_S = LOGSP.UB_S
const WIDTH_S = UB_S .- LB_S

global_best_val  = Inf
global_best_xenc = LOGSP.encode(x0_phys)
global_best_diag = FAILED_CORR
total_evals = 0
budget_large = 0; budget_small = 0
restart_count = 0
large_lambda = DEFAULT_LAMBDA

const RUN_TAG = get(ENV, "RUN_TAG", "gpu_nancy_corrected")
results_file    = joinpath(ROOT, "gpu_transport", "artifacts", "$(RUN_TAG)_results.txt")
checkpoint_file = joinpath(ROOT, "gpu_transport", "artifacts", "$(RUN_TAG)_best.txt")

t_start = time()
while total_evals < MAX_EVALS_C
    global total_evals, global_best_val, global_best_xenc, global_best_diag
    global restart_count, large_lambda, budget_large, budget_small

    if restart_count == 0
        run_lambda = DEFAULT_LAMBDA; run_type = :large
        run_sigma_frac = SIGMA_FRAC; run_x0 = LOGSP.encode(x0_phys)
    elseif budget_large <= budget_small
        large_lambda = min(large_lambda * 2, MAX_EVALS_C ÷ 10)
        run_lambda = large_lambda; run_type = :large
        run_sigma_frac = SIGMA_FRAC; run_x0 = copy(global_best_xenc)
    else
        run_lambda = DEFAULT_LAMBDA; run_type = :small
        run_sigma_frac = SIGMA_FRAC * 10.0^(-2.0 * rand())
        mix = 0.3 + 0.4 * rand()
        run_x0 = mix .* global_best_xenc .+ (1.0 - mix) .* (LB_S .+ rand(N_DIM) .* WIDTH_S)
        run_x0 .= clamp.(run_x0, LB_S, UB_S)
    end

    remaining = MAX_EVALS_C - total_evals
    remaining < run_lambda && break
    restart_count += 1
    run_evals = 0

    println("\n" * "-"^50)
    println("RESTART #$(restart_count) ($(run_type), λ=$(run_lambda), σ_frac=$(round(run_sigma_frac, digits=4)))")
    println("-"^50)

    es = CMAES(run_x0; lb = LB_S, ub = UB_S, popsize = run_lambda, sigma_frac = run_sigma_frac)
    es.best_ever_val = global_best_val
    es.best_ever_x = copy(global_best_xenc)

    while total_evals + run_lambda <= MAX_EVALS_C
        gen_seed = rand(UInt64)
        candidates = ask(es)                                   # encoded space
        eval_results = evaluate_generation_corr(candidates, gen_seed)
        fitvals = [r.loss for r in eval_results]
        gen_best_val, gen_best_x = tell!(es, candidates, fitvals)
        total_evals += run_lambda
        run_evals += run_lambda

        gen_best_r = eval_results[argmin(fitvals)]
        improved = false
        if gen_best_val < global_best_val
            global_best_val = gen_best_val
            global_best_xenc = copy(gen_best_x)
            global_best_diag = gen_best_r
            improved = true
            xphys = LOGSP.decode(global_best_xenc)
            open(checkpoint_file, "w") do f
                for (j, pname) in enumerate(PARAM_NAMES)
                    println(f, "$(pname)\t$(xphys[j])")
                end
                println(f, "# loss\t$(global_best_val)")
                println(f, "# fms\t$(global_best_diag.fms)")
                println(f, "# shape\t$(global_best_diag.shape)")
                println(f, "# bearing\t$(global_best_diag.bearing)")
                println(f, "# extent\t$(global_best_diag.extent)")
                println(f, "# toa\t$(global_best_diag.toa)")
            end
        end

        elapsed = time() - t_start
        @printf("  Gen %3d [%5d/%d] FMS=%.2f shp=%.2f bear=%.2f ext=%.2f toa=%.2f | score=%.1f%% | σ=%.3f [%.0fs]%s\n",
                es.generation, total_evals, MAX_EVALS_C,
                gen_best_r.fms, gen_best_r.shape, gen_best_r.bearing,
                gen_best_r.extent, gen_best_r.toa,
                (1.0 - gen_best_val) * 100, es.sigma, elapsed, improved ? " ***" : "")
        flush(stdout)

        do_restart, reason = should_restart(es)
        if do_restart
            println("  -> Restart: $(reason)")
            break
        end
    end

    run_type == :large ? (budget_large += run_evals) : (budget_small += run_evals)
    println("  Run used $(run_evals) evals.  Budget: large=$(budget_large), small=$(budget_small)")
end

t_elapsed = time() - t_start
xphys = LOGSP.decode(global_best_xenc)
println("\n" * "="^70)
println("NANCY GPU BIPOP-CMA-ES (CORRECTED) COMPLETE")
println("="^70)
@printf "Total evaluations: %d\n" total_evals
@printf "Restarts:          %d\n" restart_count
@printf "Wall time:         %.1f minutes\n" (t_elapsed / 60)
@printf "Best loss:         %.6f  (score %.2f%%)\n" global_best_val ((1.0 - global_best_val) * 100)
@printf "  FMS=%.3f shape=%.3f bearing=%.3f extent=%.3f toa=%.3f\n" global_best_diag.fms global_best_diag.shape global_best_diag.bearing global_best_diag.extent global_best_diag.toa

open(results_file, "w") do f
    println(f, "# Nancy GPU corrected-loss BIPOP-CMA-ES")
    println(f, "# evals=$(total_evals) restarts=$(restart_count) wall_min=$(round(t_elapsed/60, digits=2))")
    println(f, "# loss=$(global_best_val) score=$((1.0-global_best_val)*100)")
    println(f, "# fms=$(global_best_diag.fms) shape=$(global_best_diag.shape) bearing=$(global_best_diag.bearing) extent=$(global_best_diag.extent) toa=$(global_best_diag.toa)")
    for (j, pname) in enumerate(PARAM_NAMES)
        println(f, "$(pname)\t$(xphys[j])")
    end
end
println("Saved $(checkpoint_file)")
println("Saved $(results_file)")
