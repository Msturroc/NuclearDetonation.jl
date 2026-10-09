# Nancy GPU calibration objective, shared by the runners
# ======================================================
# Loads the Nancy setup (examples/nancy_cmaes_particle_size.jl, its own CMA-ES loop
# suppressed), the GPU forward model and the corrected 5-component loss, and
# defines `rho_core_corrected(params, seed)` (params in physical units) and
# `evaluate_generation_corr`. Moved out of gpu_nancy_corrected.jl unchanged so the
# BIPOP and champion5 runners score candidates identically.

const _USER_MAX_EVALS = parse(Int, get(ENV, "MAX_EVALS", "6000"))
ENV["MAX_EVALS"] = "0"   # suppress the upstream CMA-ES loop on include

using Random, Statistics, Printf, StaticArrays, CUDA
using NuclearDetonation
using NuclearDetonation.Transport

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
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
