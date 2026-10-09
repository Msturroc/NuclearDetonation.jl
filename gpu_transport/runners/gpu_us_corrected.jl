#!/usr/bin/env julia
# US-test GPU BIPOP-CMA-ES — CORRECTED uniform loss (spec §1) at 10k particles.
# One runner, driven by the EXACT per-test config in cmaes_calibration.jl
# (release time, met anchor CACHE_START/END, SIM_HOURS, source, yield, obs).
# We include that reference for setup (MAX_EVALS=0 suppresses its CPU loop),
# reuse its log10 reparam (encode/decode/LB_S/UB_S) and obs, then run the GPU
# forward sim + the shared uniform combined_loss (no per-test branch).
#
# Run from repo root:
#   MAX_EVALS=6000 N_PARTICLES=10000 julia --project \
#       gpu_transport/runners/gpu_us_corrected.jl {trinity|harry|smallboy|doppler} OU

length(ARGS) >= 1 || error("usage: gpu_us_corrected.jl {trinity|harry|smallboy|doppler} [OU|RW]")
const _USER_MAX_EVALS = parse(Int, get(ENV, "MAX_EVALS", "6000"))
ENV["MAX_EVALS"] = "0"                 # suppress cmaes_calibration's CPU loop
ENV["N_PARTICLES"] = get(ENV, "N_PARTICLES", "10000")

using Random, Statistics, Printf, StaticArrays, CUDA
using NuclearDetonation
using NuclearDetonation.Transport

const ROOT = "/home/marc/NuclearDetonation.jl"
# Sets up TEST_CONFIG, OBS, OBS_MASKS/SHAPES, grids, DOMAIN, RELEASE_X/Y, MET_CACHE,
# CACHE_START/END, SIM_HOURS, SOURCE_LAT/LON, DOSE_FACTOR, gaussian_smooth,
# inertia_ellipse, encode_params/decode_params/LB_S/UB_S, CMAES, ask/tell!, etc.
include(joinpath(ROOT, "examples", "calibration_us_tests", "cmaes_calibration.jl"))
const MAX_EVALS_C = _USER_MAX_EVALS

include(joinpath(ROOT, "gpu_transport", "host_shadow_v2.jl"))
include(joinpath(ROOT, "gpu_transport", "met_upload.jl"))
include(joinpath(ROOT, "gpu_transport", "gpu_kernel_v2.jl"))
include(joinpath(ROOT, "gpu_transport", "calibration_shared.jl"))
using .CalibrationShared

println("\n[us-gpu] $(TEST_CONFIG.label): uploading met to GPU…")
const GPU_WINDOWS = load_nancy_gpu_windows()
println("[us-gpu] ", length(GPU_WINDOWS), " met windows; SIM_HOURS=", SIM_HOURS,
        "; N_PARTICLES=", get(ENV,"N_PARTICLES","10000"), "; GPU=", CUDA.name(CUDA.device()))

# Observed bearings via the shared module (single-sourced bearing math).
const OBS_BEARINGS_C = let d = Dict{Float64,Float64}()
    for (dr, m) in OBS_MASKS
        b = CalibrationShared.centroid_bearing(m, LAT_GRID, LON_GRID, SOURCE_LAT, SOURCE_LON)
        isnothing(b) || (d[dr] = b)
    end; d
end

struct CorrResult
    loss::Float64; fms::Float64; shape::Float64; bearing::Float64
    extent::Float64; toa::Float64; combined_old::Float64
end
const FAILED_CORR = CorrResult(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

function rho_core_corrected(params::Vector{Float64}, gen_seed::UInt64)
    smooth_sigma = params[20]
    final_dep, hourly_dep, n_alive = run_gpu_shadow(params, gen_seed;
                                                    n_hours = SIM_HOURS, windows = GPU_WINDOWS)
    nx_obs, ny_obs = length(LON_GRID), length(LAT_GRID)
    sum(final_dep) <= 0 && return FAILED_CORR

    model_snapshots = [Float64.(view(hourly_dep, :, :, h)) for h in 1:SIM_HOURS]
    final_dose = model_snapshots[end]
    sum(final_dose) <= 0 && return FAILED_CORR

    dose_smooth = gaussian_smooth(final_dose .* DOSE_FACTOR, smooth_sigma)

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

    snaps_norm = [let s = sum(snap); s > 0 ? snap ./ s : snap; end for snap in model_snapshots]
    toa_result = Transport.compute_toa_score(snaps_norm, Float64.(1:SIM_HOURS),
        OBS.toa_contours, LAT_GRID, LON_GRID; threshold_fraction = 0.01)
    toa_score = (isnothing(toa_result) || isinf(toa_result.mean_arrival_error_hours)) ? 0.0 :
                max(0.0, 1.0 - toa_result.mean_arrival_error_hours / 6.0)

    bsc = CalibrationShared.bearing_score(dose_smooth, OBS_MASKS, OBS_BEARINGS_C,
                                          LAT_GRID, LON_GRID, SOURCE_LAT, SOURCE_LON)
    r = CalibrationShared.combined_loss(fms = geo_mean_fms, shape = geo_mean_shape,
            bearing = bsc, extent = extent_score, toa = toa_score)
    return CorrResult(r.loss, geo_mean_fms, geo_mean_shape, bsc, extent_score, toa_score, r.combined_old)
end

# encode/decode come from cmaes_calibration (its log10 reparam, params 8-19).
evaldecode(s) = decode_params(s)
function evaluate_generation_corr(enc_candidates::Vector{Vector{Float64}}, gen_seed::UInt64)
    out = Vector{CorrResult}(undef, length(enc_candidates))
    for i in eachindex(enc_candidates)
        try
            out[i] = rho_core_corrected(evaldecode(enc_candidates[i]), gen_seed)
        catch e
            @warn "GPU eval failed for candidate $i" exception=(e, catch_backtrace())
            out[i] = FAILED_CORR
        end
    end
    return out
end

println("[us-gpu] JIT warm-up…")
let warm = copy(x0); _ = rho_core_corrected(warm, UInt64(0xDEADBEEF)); end
CUDA.synchronize()
println("[us-gpu] warm-up done.")

println("\n" * "="^70)
println("$(uppercase(TEST_CONFIG.label)) GPU BIPOP-CMA-ES — CORRECTED LOSS  (budget: $(MAX_EVALS_C))")
println("="^70)

gbest_val  = Inf
gbest_xenc = encode_params(copy(x0))
gbest_diag = FAILED_CORR
tot_evals = 0; bud_large = 0; bud_small = 0; rcount = 0
glambda = DEFAULT_LAMBDA
WIDTH_S = UB_S .- LB_S

ckpt = joinpath(ROOT, "gpu_transport", "artifacts", "gpu_$(TEST_NAME)_corrected_best.txt")
resf = joinpath(ROOT, "gpu_transport", "artifacts", "gpu_$(TEST_NAME)_corrected_results.txt")
t0 = time()

while tot_evals < MAX_EVALS_C
    global tot_evals, gbest_val, gbest_xenc, gbest_diag, rcount, glambda, bud_large, bud_small
    if rcount == 0
        run_lambda = DEFAULT_LAMBDA; run_sf = SIGMA_FRAC; run_x0 = encode_params(copy(x0)); rt = :large
    elseif bud_large <= bud_small
        glambda = min(glambda * 2, MAX_EVALS_C ÷ 10); run_lambda = glambda; run_sf = SIGMA_FRAC
        run_x0 = copy(gbest_xenc); rt = :large
    else
        run_lambda = DEFAULT_LAMBDA; run_sf = SIGMA_FRAC * 10.0^(-2.0 * rand()); rt = :small
        mix = 0.3 + 0.4 * rand()
        run_x0 = mix .* gbest_xenc .+ (1.0 - mix) .* (LB_S .+ rand(N_DIM) .* WIDTH_S)
        run_x0 .= clamp.(run_x0, LB_S, UB_S)
    end
    (MAX_EVALS_C - tot_evals) < run_lambda && break
    rcount += 1; run_evals = 0
    println("\n--- RESTART #$(rcount) ($(rt), λ=$(run_lambda), σ_frac=$(round(run_sf,digits=4))) ---")

    es = CMAES(run_x0; lb = LB_S, ub = UB_S, popsize = run_lambda, sigma_frac = run_sf)
    es.best_ever_val = gbest_val; es.best_ever_x = copy(gbest_xenc)

    while tot_evals + run_lambda <= MAX_EVALS_C
        gen_seed = rand(UInt64)
        cands = ask(es)
        res = evaluate_generation_corr(cands, gen_seed)
        fit = [r.loss for r in res]
        gbv, gbx = tell!(es, cands, fit)
        tot_evals += run_lambda; run_evals += run_lambda
        gr = res[argmin(fit)]
        improved = false
        if gbv < gbest_val
            gbest_val = gbv; gbest_xenc = copy(gbx); gbest_diag = gr; improved = true
            xp = decode_params(gbest_xenc)
            open(ckpt, "w") do f
                for (j, pn) in enumerate(PARAM_NAMES); println(f, "$(pn)\t$(xp[j])"); end
                println(f, "# loss\t$(gbest_val)"); println(f, "# fms\t$(gbest_diag.fms)")
                println(f, "# shape\t$(gbest_diag.shape)"); println(f, "# bearing\t$(gbest_diag.bearing)")
                println(f, "# extent\t$(gbest_diag.extent)"); println(f, "# toa\t$(gbest_diag.toa)")
            end
        end
        @printf("  Gen %3d [%5d/%d] FMS=%.2f shp=%.2f bear=%.2f ext=%.2f toa=%.2f | score=%.1f%% | σ=%.3f [%.0fs]%s\n",
                es.generation, tot_evals, MAX_EVALS_C, gr.fms, gr.shape, gr.bearing, gr.extent, gr.toa,
                (1.0 - gbv) * 100, es.sigma, time() - t0, improved ? " ***" : "")
        flush(stdout)
        dr, reason = should_restart(es); if dr; println("  -> Restart: $(reason)"); break; end
    end
    rt == :large ? (bud_large += run_evals) : (bud_small += run_evals)
end

xp = decode_params(gbest_xenc)
println("\n" * "="^70)
println("$(uppercase(TEST_CONFIG.label)) GPU BIPOP-CMA-ES (CORRECTED) COMPLETE")
@printf "evals=%d  wall=%.1f min  loss=%.5f  score=%.2f%%\n" tot_evals ((time()-t0)/60) gbest_val ((1-gbest_val)*100)
@printf "  FMS=%.3f shape=%.3f bearing=%.3f extent=%.3f toa=%.3f\n" gbest_diag.fms gbest_diag.shape gbest_diag.bearing gbest_diag.extent gbest_diag.toa
open(resf, "w") do f
    println(f, "# $(TEST_CONFIG.label) GPU corrected-loss BIPOP-CMA-ES; evals=$(tot_evals) score=$((1-gbest_val)*100)")
    println(f, "# fms=$(gbest_diag.fms) shape=$(gbest_diag.shape) bearing=$(gbest_diag.bearing) extent=$(gbest_diag.extent) toa=$(gbest_diag.toa)")
    for (j, pn) in enumerate(PARAM_NAMES); println(f, "$(pn)\t$(xp[j])"); end
end
println("Saved $(ckpt)\nSaved $(resf)")
