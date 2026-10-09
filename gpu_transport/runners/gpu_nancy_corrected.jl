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

const _USER_MAX_EVALS = parse(Int, get(ENV, "MAX_EVALS", "6000"))
ENV["MAX_EVALS"] = "0"   # suppress the upstream CMA-ES loop on include

using Random, Statistics, Printf, StaticArrays, CUDA
using NuclearDetonation
using NuclearDetonation.Transport

const ROOT = "/home/marc/NuclearDetonation.jl"
include(joinpath(ROOT, "examples", "nancy_cmaes_particle_size.jl"))   # loop suppressed
const MAX_EVALS_C = _USER_MAX_EVALS

include(joinpath(ROOT, "gpu_transport", "host_shadow_v2.jl"))
include(joinpath(ROOT, "gpu_transport", "met_upload.jl"))
include(joinpath(ROOT, "gpu_transport", "gpu_kernel_v2.jl"))
include(joinpath(ROOT, "gpu_transport", "calibration_shared.jl"))
using .CalibrationShared

println("[corr] uploading Nancy met to GPU…")
const GPU_WINDOWS = load_nancy_gpu_windows()
println("[corr] ", length(GPU_WINDOWS), " met windows; N_PARTICLES=",
        get(ENV, "N_PARTICLES", "10000"), "; GPU=", CUDA.name(CUDA.device()))

# Observed centroid bearing per contour (Nancy upstream has no bearing infra).
const OBS_BEARINGS = let d = Dict{Float64,Float64}()
    for (dose_rate, obs_mask) in OBS_MASKS
        b = CalibrationShared.centroid_bearing(obs_mask, LAT_GRID, LON_GRID,
                                               SOURCE_LAT, SOURCE_LON)
        if !isnothing(b)
            d[dose_rate] = b
            @printf("   contour %7.1f mR/h : obs bearing %.1f deg\n", dose_rate, b)
        end
    end
    d
end

# log10 reparam for this dimension/bounds
const LOGSP = CalibrationShared.make_logspace(N_DIM, LB, UB)

struct CorrResult
    loss::Float64; fms::Float64; shape::Float64; bearing::Float64
    extent::Float64; toa::Float64; combined_old::Float64
end
const FAILED_CORR = CorrResult(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

# ---------------------------------------------------------------------------
# Corrected objective: same forward sim + FMS/shape/extent/TOA as v3, plus the
# bearing term, folded through the single CalibrationShared.combined_loss.
# `params` arrives in PHYSICAL space (caller decodes from log space first).
# ---------------------------------------------------------------------------
function rho_core_corrected(params::Vector{Float64}, gen_seed::UInt64)
    smooth_sigma = params[20]
    final_dep, hourly_dep, n_alive = run_gpu_shadow(params, gen_seed; windows = GPU_WINDOWS)
    nx_obs, ny_obs = length(LON_GRID), length(LAT_GRID)
    sum(final_dep) <= 0 && return FAILED_CORR

    model_snapshots = [Float64.(view(hourly_dep, :, :, h)) for h in 1:12]
    snapshot_hours  = Float64.(1:12)
    final_dose = model_snapshots[end]
    sum(final_dose) <= 0 && return FAILED_CORR

    final_dose_mRh = final_dose .* DOSE_FACTOR
    dose_smooth = gaussian_smooth(final_dose_mRh, smooth_sigma)

    fms_scores = Float64[]; shape_scores = Float64[]
    for (dose_rate, obs_mask) in OBS_MASKS
        if sum(obs_mask) == 0
            push!(fms_scores, 0.0); push!(shape_scores, 0.0); continue
        end
        model_mask = dose_smooth .>= dose_rate
        inter = Float64(sum(model_mask .& obs_mask)); uni = Float64(sum(model_mask .| obs_mask))
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
    geo_mean_fms   = CalibrationShared.geo_mean(fms_scores)
    geo_mean_shape = CalibrationShared.geo_mean(shape_scores)

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
    toa_score = (isnothing(toa_result) || isinf(toa_result.mean_arrival_error_hours)) ? 0.0 :
                max(0.0, 1.0 - toa_result.mean_arrival_error_hours / 6.0)

    bsc = CalibrationShared.bearing_score(dose_smooth, OBS_MASKS, OBS_BEARINGS,
                                          LAT_GRID, LON_GRID, SOURCE_LAT, SOURCE_LON)
    r = CalibrationShared.combined_loss(fms = geo_mean_fms, shape = geo_mean_shape,
            bearing = bsc, extent = extent_score, toa = toa_score)
    return CorrResult(r.loss, geo_mean_fms, geo_mean_shape, bsc,
                      extent_score, toa_score, r.combined_old)
end

# Candidates arrive in ENCODED (log) space → decode before the forward sim.
function evaluate_generation_corr(enc_candidates::Vector{Vector{Float64}}, gen_seed::UInt64)
    n = length(enc_candidates)
    out = Vector{CorrResult}(undef, n)
    for i in 1:n
        try
            out[i] = rho_core_corrected(LOGSP.decode(enc_candidates[i]), gen_seed)
        catch e
            @warn "GPU eval failed for candidate $i" exception=(e, catch_backtrace())
            out[i] = FAILED_CORR
        end
    end
    return out
end

# JIT warm-up
println("[corr] JIT warm-up…")
let warm = copy(WARM_START_PARAMS)
    _ = rho_core_corrected(warm, UInt64(0xDEADBEEF))
end
CUDA.synchronize()
println("[corr] warm-up done.")

# ===========================================================================
# BIPOP-CMA-ES in encoded (log10) space — structure from v3, decode at eval.
# ===========================================================================
println("\n" * "="^70)
println("NANCY GPU BIPOP-CMA-ES — CORRECTED LOSS   (budget: $(MAX_EVALS_C) evals)")
println("="^70)

x0_phys = isnothing(load_checkpoint_params("ou")) ? copy(WARM_START_PARAMS) :
          load_checkpoint_params("ou")
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

results_file    = joinpath(ROOT, "gpu_transport", "artifacts", "gpu_nancy_corrected_results.txt")
checkpoint_file = joinpath(ROOT, "gpu_transport", "artifacts", "gpu_nancy_corrected_best.txt")

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
