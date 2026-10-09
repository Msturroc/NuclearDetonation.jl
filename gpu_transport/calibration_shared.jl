# ============================================================================
# calibration_shared.jl — single source of truth for the calibration loss
# ============================================================================
# Every calibration runner (CPU and GPU, all 6 US/NTS tests) imports this
# module so the scoring formula, the bearing gate, and the log10 search-space
# transform exist in EXACTLY ONE place and can never drift apart again.
#
# Provenance (see gpu_transport/GPU_RECALIBRATION_SPEC.md §1):
#   - `centroid_bearing` and `bearing_score` are ported verbatim from the CPU
#     reference examples/smoky_example/smoky_cmaes_particle_size.jl
#     (centroid_bearing lines 365-387, bearing scoring lines 691-716).
#   - `combined_loss` (weights + hard bearing gate) and the log10 reparam are
#     the CORRECTED loss from the spec — these REPLACE the older per-test
#     combine formulas that omitted the bearing term and the gate.
#
# A runner computes its five raw component scores from its own forward sim and
# observations (those depend on test-specific grids/contours), then calls
# `combined_loss(...)` to fold them into the single ranked loss. The CMA-ES
# search runs in the encoded (log10) space; runners `decode` before every
# forward-sim call and before writing any checkpoint so stored vectors stay
# physical.
# ============================================================================

module CalibrationShared

using Statistics: mean

export centroid_bearing, bearing_score, geo_mean,
       combined_loss,
       log_mask, encode_params, decode_params, make_logspace

# ----------------------------------------------------------------------------
# log10 reparameterisation
# ----------------------------------------------------------------------------
# Params 8-19 are the multiplicative scale factors (turbulence, physics,
# deposition, activity). They span orders of magnitude, so CMA-ES searches
# them in log10 space. All other params (sizes, fractions, heights, smoothing)
# stay linear. `8:19` is identical for the 20-param (Nancy) and 23-param (US)
# layouts because the scale block sits in the same slots in both.

"""
    log_mask(n_dim) -> BitVector

`true` for parameter slots searched in log10 space (the scale block, params
8-19), `false` otherwise. Safe for any `n_dim >= 7`.
"""
function log_mask(n_dim::Integer)
    m = falses(n_dim)
    for j in 8:min(19, n_dim)
        m[j] = true
    end
    return m
end

"""
    encode_params(x, mask) -> Vector{Float64}

Map physical params `x` into search (encoded) space: log10 where `mask` is set.
"""
encode_params(x::AbstractVector, mask::AbstractVector{Bool}) =
    Float64[mask[j] ? log10(x[j]) : Float64(x[j]) for j in eachindex(x)]

"""
    decode_params(s, mask) -> Vector{Float64}

Inverse of [`encode_params`](@ref): map encoded params `s` back to physical
space (`10^s` where `mask` is set). `decode∘encode` is the identity.
"""
decode_params(s::AbstractVector, mask::AbstractVector{Bool}) =
    Float64[mask[j] ? 10.0^s[j] : Float64(s[j]) for j in eachindex(s)]

"""
    make_logspace(n_dim, lb, ub) -> NamedTuple

Convenience bundle for a runner of dimension `n_dim` with physical bounds
`lb`/`ub`. Returns `(mask, encode, decode, LB_S, UB_S)` where `encode`/`decode`
are 1-arg closures over the mask and `LB_S`/`UB_S` are the encoded bounds to
hand to the CMA-ES constructor.
"""
function make_logspace(n_dim::Integer, lb::AbstractVector, ub::AbstractVector)
    mask = log_mask(n_dim)
    enc(x) = encode_params(x, mask)
    dec(s) = decode_params(s, mask)
    return (mask = mask, encode = enc, decode = dec, LB_S = enc(lb), UB_S = enc(ub))
end

# ----------------------------------------------------------------------------
# Bearing geometry  (verbatim from smoky_cmaes_particle_size.jl)
# ----------------------------------------------------------------------------

"""
    centroid_bearing(mask, lat_grid, lon_grid, source_lat, source_lon; min_cells=10)

Bearing (degrees, 0=N, 90=E) from `source` to the centroid of binary `mask`.
Returns `nothing` if fewer than `min_cells` cells are set.
"""
function centroid_bearing(mask::AbstractMatrix, lat_grid, lon_grid,
                          source_lat::Float64, source_lon::Float64;
                          min_cells::Int=10)
    ref_lat = 0.5 * (first(lat_grid) + last(lat_grid))
    sum_x = 0.0
    sum_y = 0.0
    n = 0
    for i in eachindex(lon_grid)
        for j in eachindex(lat_grid)
            if mask isa AbstractMatrix{Bool} ? mask[i, j] : mask[i, j] > 0
                sum_x += (lon_grid[i] - source_lon) * cosd(ref_lat)
                sum_y += lat_grid[j] - source_lat
                n += 1
            end
        end
    end
    n < min_cells && return nothing
    cx = sum_x / n
    cy = sum_y / n
    bearing = atand(cx, cy)  # atan2(east, north) -> degrees from north
    bearing < 0 && (bearing += 360.0)
    return bearing
end

"""
    bearing_score(dose_smooth, obs_masks, obs_bearings, lat_grid, lon_grid,
                  source_lat, source_lon; min_cells=10) -> Float64

Dose-rate-weighted cos⁴(Δbearing) of model-vs-observed plume bearing, summed
over contour levels. Each contour `dose_rate` is weighted by `dose_rate` so the
close-in, high-dose contours dominate the score. A contour the model fails to
produce contributes 0 with full weight (penalising missing plume reach).
Sharpness of cos⁴: 10°→0.94, 20°→0.77, 30°→0.56, 45°→0.25.

`obs_masks` is an iterable of `(dose_rate, obs_mask)` pairs (matching the
runner's `OBS_MASKS`); `obs_bearings` is a `Dict{dose_rate => bearing_deg}`.
"""
function bearing_score(dose_smooth::AbstractMatrix, obs_masks, obs_bearings,
                       lat_grid, lon_grid, source_lat::Float64, source_lon::Float64;
                       min_cells::Int=10)
    bearing_sum = 0.0
    bearing_weight_sum = 0.0
    for (dose_rate, _obs_mask) in obs_masks
        obs_bearing = get(obs_bearings, dose_rate, nothing)
        isnothing(obs_bearing) && continue
        model_mask = dose_smooth .>= dose_rate
        model_bearing = if sum(model_mask) > 0
            centroid_bearing(model_mask, lat_grid, lon_grid, source_lat, source_lon;
                             min_cells=min_cells)
        else
            nothing
        end
        w = dose_rate  # weight by dose rate: 1000 mR/h gets 1000x the weight of 1 mR/h
        if !isnothing(model_bearing)
            diff_deg = abs(model_bearing - obs_bearing)
            diff_deg > 180.0 && (diff_deg = 360.0 - diff_deg)
            bearing_sum += w * cosd(diff_deg)^4
        end
        # else: no model contour at this level -> zero score with full weight
        bearing_weight_sum += w
    end
    return bearing_weight_sum > 0 ? bearing_sum / bearing_weight_sum : 0.0
end

# ----------------------------------------------------------------------------
# Score reduction + combined loss
# ----------------------------------------------------------------------------

"""
    geo_mean(scores; floor=0.005) -> Float64

Geometric mean of per-contour scores, with a small floor so a single zero
contour cannot collapse the whole product to 0. Returns 0 for an empty list.
"""
geo_mean(scores; floor::Real=0.005) =
    isempty(scores) ? 0.0 : exp(mean(log(max(s, floor)) for s in scores))

"""
    combined_loss(; fms, shape, bearing, extent, toa) -> NamedTuple

Fold the five component scores into the single ranked loss used by CMA-ES.
Returns `(loss, fms, shape, bearing, extent, toa, combined_old)`.

ONE identical formula for every test (Trinity, Harry, SmallBoy, Nancy, Smoky,
Doppler) — there is deliberately no per-test branch, so the loss cannot drift
between tests:

    combined = 0.25·FMS + 0.15·shape + 0.20·bearing + 0.10·extent + 0.30·TOA

The bearing term is always included and always carries 20% of the score.

Hard bearing gate: if `bearing < 0.5` the model plume points >~45° off the
observed direction, so the loss is pinned to **2.0** regardless of the other
components — this stops geometric-cheating fits (e.g. SmallBoy pointing E
instead of NE) from ever ranking best. `2.0` (not `Inf`) keeps a finite
gradient back toward fixing the bearing.

`combined_old` is the immediately-prior 4-term loss (no bearing, no gate),
reported for apples-to-apples comparison with pre-correction fits; it never
feeds the gate or the ranked `loss`.
"""
function combined_loss(; fms::Real, shape::Real, bearing::Real,
                       extent::Real, toa::Real)
    combined = 0.25 * fms + 0.15 * shape + 0.20 * bearing + 0.10 * extent + 0.30 * toa
    combined_old = 0.35 * fms + 0.20 * shape + 0.15 * extent + 0.30 * toa

    if bearing < 0.5
        return (loss = 2.0, fms = fms, shape = shape, bearing = bearing,
                extent = extent, toa = toa, combined_old = combined_old)
    end
    return (loss = 1.0 - combined, fms = fms, shape = shape, bearing = bearing,
            extent = extent, toa = toa, combined_old = combined_old)
end

end # module CalibrationShared
