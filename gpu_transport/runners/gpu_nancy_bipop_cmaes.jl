#!/usr/bin/env julia
# Nancy BIPOP-CMA-ES driver — GPU forward simulator
# ===================================================
# Reuses the helpers, constants, observation grids, and scoring functions
# from the upstream `nancy_cmaes_particle_size.jl` script (no edits to that
# file). To prevent the upstream script from running its own optimiser at
# include time, we set `ENV["MAX_EVALS"]=0` first — the upstream `while`
# loop then exits immediately, leaving constants and `rho_core` available.
#
# Then we define `rho_core_gpu` (mirrors `rho_core` line-by-line, swapping
# `Transport.run_simulation!` for `run_gpu_simulation`) and run our own
# BIPOP loop using the upstream `CMAES`, `ask`, `tell!`, `should_restart`.
#
# Usage:
#   MAX_EVALS=500 julia --project=/home/marc/NuclearDetonation.jl \
#     --threads=auto /home/marc/julia_snap_explorations/gpu_nancy_bipop_cmaes.jl
#
# Output (in /home/marc/julia_snap_explorations/):
#   - gpu_nancy_cmaes_best.txt        — checkpoint with best params
#   - gpu_nancy_cmaes_results.txt     — final summary
#   - gpu_nancy_cmaes_log.txt         — per-generation timeline

const _USER_MAX_EVALS = parse(Int, get(ENV, "MAX_EVALS", "6000"))

# Force the upstream script's loop to be a no-op at include time.
ENV["MAX_EVALS"] = "0"

using Random
using Statistics
using LinearAlgebra
using Printf
using Dates
using StaticArrays
using NCDatasets
using NuclearDetonation
using NuclearDetonation.Transport

# Pull in upstream constants + scoring (this does NOT run its loop because
# MAX_EVALS=0). We `include` rather than `using` because the upstream file is
# a top-level script.
const _UPSTREAM = "/home/marc/NuclearDetonation.jl/examples/nancy_cmaes_particle_size.jl"
println("[gpu-bipop] including upstream cmaes script (with MAX_EVALS=0)")
include(_UPSTREAM)

# Restore the user's intended budget.
const MAX_EVALS_GPU = _USER_MAX_EVALS

# GPU engine
include(joinpath(@__DIR__, "gpu_transport", "GpuTransport.jl"))
using .GpuTransport

# ============================================================================
# GPU-backed rho_core
# ============================================================================
# Same parameter unpacking as upstream `rho_core`. We don't build a SimulationState
# at all — instead we generate particles through `generate_nancy_particles` and
# call `run_gpu_simulation`. Scoring (FMS / shape / extent / TOA) is reused from
# the upstream module by referencing the constants/functions it defined.

# A small cache so we don't re-upload met fields per evaluation.
const _GPU_MET_CACHE = Ref{Any}(nothing)
const _GPU_MET_LEVEL_Z = Ref{Vector{Float32}}(Float32[])

function _ensure_gpu_met()
    if _GPU_MET_CACHE[] === nothing
        println("[gpu-bipop] uploading Nancy ERA5 to VRAM (one-time)...")
        gpu_met, _, _ = load_nancy_gpu_met()
        psfc0 = mean(Array(gpu_met.psfc.data)[:, :, 1])
        _GPU_MET_LEVEL_Z[] = level_heights_from_psfc(
            Array(gpu_met.ap), Array(gpu_met.b), psfc0)
        _GPU_MET_CACHE[] = gpu_met
    end
    return _GPU_MET_CACHE[], _GPU_MET_LEVEL_Z[]
end

function rho_core_gpu(params::Vector{Float64}, gen_seed::UInt64)
    d_median_fine, sigma_g_fine, d_median_coarse, sigma_g_coarse, frac_fine = params[1:5]
    frac_lower, frac_middle = params[6:7]
    sigma_w_scale, sigma_h_scale, h_diff_scale, tl_scale = params[8:11]
    vd_scale, vgrav_scale, omega_scale, mixing_height_scale, tmix_scale = params[12:16]
    surface_height_scale, roughness_scale = params[17:18]
    activity_scale, smooth_sigma = params[19:20]
    frac_upper = clamp(1.0 - frac_lower - frac_middle, 0.05, 1.0)
    total_activity = activity_scale * 1.0e15

    rng = MersenneTwister(gen_seed)
    gpu_met, level_z = _ensure_gpu_met()

    # Float32-friendly NamedTuple consumed by GpuTransport
    f32_params = (
        d_fine = Float32(d_median_fine), sg_fine = Float32(sigma_g_fine),
        d_coarse = Float32(d_median_coarse), sg_coarse = Float32(sigma_g_coarse),
        frac_fine = Float32(frac_fine),
        frac_lower = Float32(frac_lower), frac_middle = Float32(frac_middle),
        sigma_h_scale = Float32(sigma_h_scale),
        sigma_w_scale = Float32(sigma_w_scale),
        tl_scale = Float32(tl_scale),
        vd_scale = Float32(vd_scale),
        vgrav_scale = Float32(vgrav_scale),
        omega_scale = Float32(omega_scale),
        surface_height_scale = Float32(surface_height_scale),
        activity_Bq = Float32(total_activity),
    )

    n_particles = 1000
    particles = generate_nancy_particles(rng, n_particles, f32_params)

    # Deposition geom matches upstream LON_GRID / LAT_GRID exactly so the
    # scoring helpers can read the result without resampling.
    dep_geom = (
        lon0 = Float32(first(LON_GRID)),
        lat0 = Float32(first(LAT_GRID)),
        dlon = Float32(LON_GRID[2] - LON_GRID[1]),
        dlat = Float32(LAT_GRID[2] - LAT_GRID[1]),
        nx   = length(LON_GRID),
        ny   = length(LAT_GRID),
    )

    sim = run_gpu_simulation(
        particles, gpu_met, dep_geom, level_z;
        params = f32_params,
        n_steps = 144, dt = 300.0f0,
        steps_per_hour = 12, n_hours = 12,
        rng_seed = gen_seed,
    )

    # Convert per-hour cumulative deposition to dose rate snapshots so that
    # the upstream scoring helpers (FMS, shape, extent, TOA) work unchanged.
    n_hours = 12
    model_snapshots = Vector{Matrix{Float64}}(undef, n_hours)
    for h in 1:n_hours
        model_snapshots[h] = Float64.(sim.cumulative[:, :, h])
    end

    final_dose = model_snapshots[end]
    total = sum(final_dose)
    if total <= 0
        return (loss = 1.0, fms = 0.0, shape = 0.0, extent = 0.0, toa = 0.0, combined_old = 0.0)
    end

    final_dose_mRh = final_dose .* DOSE_FACTOR
    dose_smooth = gaussian_smooth(final_dose_mRh, smooth_sigma)

    # FMS + shape per dose threshold — reuses upstream OBS_MASKS / OBS_SHAPES.
    fms_scores = Float64[]
    shape_scores = Float64[]
    for (dose_rate, obs_mask) in OBS_MASKS
        obs_area = sum(obs_mask)
        if obs_area == 0
            push!(fms_scores, 0.0); push!(shape_scores, 0.0); continue
        end
        model_mask = dose_smooth .>= dose_rate
        inter = Float64(sum(model_mask .& obs_mask))
        uni   = Float64(sum(model_mask .| obs_mask))
        push!(fms_scores, uni > 0 ? inter / uni : 0.0)

        obs_shape = get(OBS_SHAPES, dose_rate, nothing)
        model_shape = sum(model_mask) > 0 ? inertia_ellipse(model_mask, LAT_GRID, LON_GRID) : nothing
        if !isnothing(obs_shape) && !isnothing(model_shape)
            ar_score = min(model_shape.ar, obs_shape.ar) / max(model_shape.ar, obs_shape.ar)
            angle_diff = model_shape.angle - obs_shape.angle
            orient_score = cos(angle_diff)^2
            push!(shape_scores, 0.7 * ar_score + 0.3 * orient_score)
        else
            push!(shape_scores, 0.0)
        end
    end

    geo_mean_fms   = exp(mean(log(max(s, 0.005)) for s in fms_scores))
    geo_mean_shape = exp(mean(log(max(s, 0.005)) for s in shape_scores))

    # Plume extent
    nx_obs, ny_obs = length(LON_GRID), length(LAT_GRID)
    model_max_dist_km = 0.0
    for i in 1:nx_obs, j in 1:ny_obs
        if final_dose[i, j] > 0
            dlat = LAT_GRID[j] - SOURCE_LAT
            dlon = (LON_GRID[i] - SOURCE_LON) * cosd(SOURCE_LAT)
            model_max_dist_km = max(model_max_dist_km, sqrt(dlat^2 + dlon^2) * 111.0)
        end
    end
    extent_score = clamp(model_max_dist_km / OBS_MAX_DIST_KM, 0.0, 1.0)

    # TOA
    model_snapshots_norm = [let s = sum(snap)
            s > 0 ? snap ./ s : snap end for snap in model_snapshots]
    snapshot_hours = Float64[h for h in 1:n_hours]
    toa_result = Transport.compute_toa_score(model_snapshots_norm, snapshot_hours,
        NANCY_OBS.toa_contours, LAT_GRID, LON_GRID; threshold_fraction = 0.01)
    toa_score = if isnothing(toa_result) || isinf(toa_result.mean_arrival_error_hours)
        0.0
    else
        max(0.0, 1.0 - toa_result.mean_arrival_error_hours / 6.0)
    end

    combined = 0.35 * geo_mean_fms + 0.20 * geo_mean_shape + 0.15 * extent_score + 0.30 * toa_score
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
    # Single-stream serial loop — for small λ the per-eval cost is dominated
    # by the GPU kernel and parallelising via CPU threads doesn't help. Phase
    # C batched mode (one kernel for all candidates) would land here once the
    # multi-candidate kernel is online.
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
# Main BIPOP loop — copy of the upstream version, calls through GPU evaluator
# ============================================================================
println("\n" * "="^70)
println("NANCY GPU BIPOP-CMA-ES   (budget: $(MAX_EVALS_GPU) evals)")
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

results_file = "/home/marc/julia_snap_explorations/gpu_nancy_cmaes_results.txt"
checkpoint_file = "/home/marc/julia_snap_explorations/gpu_nancy_cmaes_best.txt"

t_start = time()
while total_evals < MAX_EVALS_GPU
    global total_evals, global_best_val, global_best_x, global_best_diag
    global restart_count, large_lambda, budget_large, budget_small

    if restart_count == 0
        run_lambda = DEFAULT_LAMBDA
        run_type = :large
        run_sigma_frac = SIGMA_FRAC
        run_x0 = copy(x0)
    elseif budget_large <= budget_small
        large_lambda = min(large_lambda * 2, MAX_EVALS_GPU ÷ 10)
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

    remaining = MAX_EVALS_GPU - total_evals
    remaining < run_lambda && break
    restart_count += 1
    run_evals = 0

    println("\n" * "-"^50)
    println("RESTART #$(restart_count) ($(run_type), λ=$(run_lambda), σ_frac=$(round(run_sigma_frac, digits=4)))")
    println("-"^50)

    es = CMAES(run_x0; lb=LB, ub=UB, popsize=run_lambda, sigma_frac=run_sigma_frac)
    es.best_ever_val = global_best_val
    es.best_ever_x = copy(global_best_x)

    while total_evals + run_lambda <= MAX_EVALS_GPU
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
                es.generation, total_evals, MAX_EVALS_GPU,
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

open(results_file, "w") do f
    println(f, "Nancy GPU BIPOP-CMA-ES Results")
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
