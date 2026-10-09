#!/usr/bin/env julia
# 2x2 panel of the corrected-loss Nancy GPU calibration result.
#   [1,1] calibrated GPU dose rate (H+12, log10 mR/h) + observed dose contours
#   [1,2] model dose-rate contours vs observed contours
#   [2,1] score component breakdown
#   [2,2] time-of-arrival simulation (model arrival field + observed TOA fronts)
#
# Run from repo root:  julia --project gpu_transport/plot_nancy_corrected_2x2.jl

ENV["MAX_EVALS"] = "0"
ENV["N_PARTICLES"] = get(ENV, "N_PARTICLES", "10000")
using Random, Statistics, Printf, StaticArrays, CUDA, CairoMakie
using NuclearDetonation
using NuclearDetonation.Transport

const ROOT = "/home/marc/NuclearDetonation.jl"
include(joinpath(ROOT, "examples", "nancy_cmaes_particle_size.jl"))
include(joinpath(ROOT, "gpu_transport", "host_shadow_v2.jl"))
include(joinpath(ROOT, "gpu_transport", "met_upload.jl"))
include(joinpath(ROOT, "gpu_transport", "gpu_kernel_v2.jl"))
include(joinpath(ROOT, "gpu_transport", "calibration_shared.jl"))
using .CalibrationShared

const GPU_WINDOWS = load_nancy_gpu_windows()
const OBS_BEARINGS = let d = Dict{Float64,Float64}()
    for (dr, m) in OBS_MASKS
        b = CalibrationShared.centroid_bearing(m, LAT_GRID, LON_GRID, SOURCE_LAT, SOURCE_LON)
        isnothing(b) || (d[dr] = b)
    end; d
end

# --- load calibrated vector (physical units) ---
const BEST = joinpath(ROOT, "gpu_transport", "artifacts", "gpu_nancy_corrected_best.txt")
params = Float64[]
for ln in eachline(BEST)
    startswith(ln, "#") && continue
    isempty(strip(ln)) && continue
    push!(params, parse(Float64, split(ln, '\t')[2]))
end
@printf("[plot] loaded %d-param calibrated vector\n", length(params))

# --- one forward sim with the calibrated vector ---
final_dep, hourly_dep, n_alive = run_gpu_shadow(params, UInt64(0xCA11B0A7); windows = GPU_WINDOWS)
CUDA.synchronize()
smooth_sigma = params[20]
model_snapshots = [Float64.(view(hourly_dep, :, :, h)) for h in 1:12]
final_dose = model_snapshots[end]
dose_mRh = final_dose .* DOSE_FACTOR
dose_smooth = gaussian_smooth(dose_mRh, smooth_sigma)
nx, ny = length(LON_GRID), length(LAT_GRID)

# --- recompute the 5 components for THIS field (self-consistent with the plot) ---
fms_scores = Float64[]; shape_scores = Float64[]
for (dose_rate, obs_mask) in OBS_MASKS
    if sum(obs_mask) == 0; push!(fms_scores, 0.0); push!(shape_scores, 0.0); continue; end
    mm = dose_smooth .>= dose_rate
    push!(fms_scores, sum(mm .| obs_mask) > 0 ? sum(mm .& obs_mask) / sum(mm .| obs_mask) : 0.0)
    os = get(OBS_SHAPES, dose_rate, nothing)
    ms = sum(mm) > 0 ? inertia_ellipse(mm, LAT_GRID, LON_GRID) : nothing
    if !isnothing(os) && !isnothing(ms)
        push!(shape_scores, 0.7 * (min(ms.ar, os.ar)/max(ms.ar, os.ar)) + 0.3 * cos(ms.angle - os.angle)^2)
    else; push!(shape_scores, 0.0); end
end
geo_fms = CalibrationShared.geo_mean(fms_scores); geo_shape = CalibrationShared.geo_mean(shape_scores)
dists = Float64[]
for i in 1:nx, j in 1:ny
    final_dose[i,j] > 0 || continue
    dl = LAT_GRID[j]-SOURCE_LAT; dn = (LON_GRID[i]-SOURCE_LON)*cosd(SOURCE_LAT)
    push!(dists, sqrt(dl^2+dn^2)*111.0)
end
maxd = isempty(dists) ? 0.0 : maximum(dists)
extent_sc = clamp(maxd / OBS_MAX_DIST_KM, 0.0, 1.0)
snaps_norm = [let s=sum(v); s>0 ? v./s : v; end for v in model_snapshots]
toa_res = Transport.compute_toa_score(snaps_norm, Float64.(1:12), NANCY_OBS.toa_contours,
                                      LAT_GRID, LON_GRID; threshold_fraction=0.01)
toa_sc = (isnothing(toa_res) || isinf(toa_res.mean_arrival_error_hours)) ? 0.0 :
         max(0.0, 1.0 - toa_res.mean_arrival_error_hours/6.0)
bear_sc = CalibrationShared.bearing_score(dose_smooth, OBS_MASKS, OBS_BEARINGS,
                                          LAT_GRID, LON_GRID, SOURCE_LAT, SOURCE_LON)
comps = (fms=geo_fms, shape=geo_shape, bearing=bear_sc, extent=extent_sc, toa=toa_sc)
score = CalibrationShared.combined_loss(; comps...)
score_pct = round((1 - score.loss)*100, digits=1)
@printf("[plot] FMS=%.3f shape=%.3f bearing=%.3f extent=%.3f toa=%.3f  -> %.1f%%\n",
        comps.fms, comps.shape, comps.bearing, comps.extent, comps.toa, score_pct)

# --- model time-of-arrival field ---
# Per plume cell (final dose >= floor), arrival = first hour the cumulative
# deposition reaches 10% of that cell's final value. Fills the plume footprint
# with "when did fallout arrive here", comparable to the observed TOA fronts.
const TOA_FLOOR_mRh = 0.5
arrival = fill(NaN, nx, ny)
for i in 1:nx, j in 1:ny
    dose_mRh[i,j] >= TOA_FLOOR_mRh || continue
    target = 0.10 * final_dose[i,j]
    for h in 1:12
        if model_snapshots[h][i,j] >= target; arrival[i,j] = Float64(h); break; end
    end
end

# --- plotting helpers (style from plot_cpu_vs_gpu.jl) ---
const NTS_LAT, NTS_LON = SOURCE_LAT, SOURCE_LON
const OBS_LEVELS = sort(collect(keys(OBS_MASKS)))
const CCOL = [:blue, :cyan, :green, :yellow, :orange, :red]
obs_lats = Float64[]; obs_lons = Float64[]
for c in NANCY_OBS.dose_rate_contours, p in c.polygons, pt in p
    push!(obs_lats, pt[1]); push!(obs_lons, pt[2])
end
minds = findall(>(0.0), dose_smooth)
mlons = [LON_GRID[i[1]] for i in minds]; mlats = [LAT_GRID[i[2]] for i in minds]
lon_lo = min(minimum(obs_lons), minimum(mlons)) - 0.2
lon_hi = max(maximum(obs_lons), maximum(mlons)) + 0.2
lat_lo = min(minimum(obs_lats), minimum(mlats)) - 0.2
lat_hi = max(maximum(obs_lats), maximum(mlats)) + 0.2
ax_lims = (lon_lo, lon_hi, lat_lo, lat_hi)

logdose = [v < 0.5 ? NaN : log10(v) for v in dose_smooth]
const MODEL_LEVELS = filter(>(0), OBS_LEVELS)   # drop the 0 mR/h level (traces speckle)
fin = filter(isfinite, logdose); llo = isempty(fin) ? -1.0 : minimum(fin); lhi = isempty(fin) ? 3.0 : maximum(fin)

draw_obs_contours!(ax) = for (lvl, col) in zip(OBS_LEVELS, CCOL), c in NANCY_OBS.dose_rate_contours
    c.dose_rate_mR_hr == lvl || continue
    for poly in c.polygons
        lines!(ax, [p[2] for p in poly], [p[1] for p in poly], color=col, linewidth=1.8)
    end
end

fig = Figure(size = (1400, 1250), fontsize = 14)
Label(fig[0, 1:2], "Nancy 24 kT — GPU calibrated (corrected loss, 10k particles) — score $(score_pct)%",
      fontsize = 19, font = :bold)

# [1,1] dose rate
ax1 = Axis(fig[1,1], title="Calibrated dose rate (H+12)", xlabel="Longitude (°)", ylabel="Latitude (°)",
           limits=ax_lims, aspect=DataAspect())
hm1 = heatmap!(ax1, collect(LON_GRID), collect(LAT_GRID), logdose, colormap=:viridis,
               colorrange=(llo,lhi), nan_color=:transparent)
draw_obs_contours!(ax1)
scatter!(ax1, [NTS_LON],[NTS_LAT], marker=:star5, markersize=20, color=:white, strokecolor=:black, strokewidth=1)
Colorbar(fig[1,1, Right()], hm1, label="log₁₀(mR/h)")

# [1,2] model vs observed contours
ax2 = Axis(fig[1,2], title="Model (– – dashed) vs observed (solid) contours", xlabel="Longitude (°)",
           ylabel="Latitude (°)", limits=ax_lims, aspect=DataAspect())
draw_obs_contours!(ax2)
contour_field = gaussian_smooth(dose_mRh, 3.0)   # heavier smooth → coherent shape lines
contour!(ax2, collect(LON_GRID), collect(LAT_GRID), contour_field, levels=MODEL_LEVELS,
         color=:black, linestyle=:dash, linewidth=1.6)
scatter!(ax2, [NTS_LON],[NTS_LAT], marker=:star5, markersize=20, color=:white, strokecolor=:black, strokewidth=1)

# [2,1] score components
ax3 = Axis(fig[2,1], title="Score components → $(score_pct)%", ylabel="score (0–1)",
           xticks=(1:5, ["FMS","shape","bearing","extent","TOA"]), limits=(nothing, (0,1.05)))
vals = [comps.fms, comps.shape, comps.bearing, comps.extent, comps.toa]
barplot!(ax3, 1:5, vals, color=[:steelblue,:teal,:crimson,:seagreen,:goldenrod])
for (k,v) in enumerate(vals); text!(ax3, k, v+0.02, text=string(round(v,digits=2)), align=(:center,:bottom)); end
hlines!(ax3, [0.5], color=:red, linestyle=:dot)  # bearing gate threshold
text!(ax3, 0.6, 0.52, text="bearing gate (0.5)", color=:red, fontsize=10, align=(:left,:bottom))

# [2,2] TOA simulation
ax4 = Axis(fig[2,2], title="Time of arrival (model field + observed TOA fronts)", xlabel="Longitude (°)",
           ylabel="Latitude (°)", limits=ax_lims, aspect=DataAspect())
hm4 = heatmap!(ax4, collect(LON_GRID), collect(LAT_GRID), arrival, colormap=:plasma,
               colorrange=(1,12), nan_color=:transparent)
for tc in NANCY_OBS.toa_contours, ln in tc.lines
    lines!(ax4, [p[2] for p in ln], [p[1] for p in ln],
           color=fill(tc.hour, length(ln)), colormap=:plasma, colorrange=(1,12), linewidth=3.0)
end
scatter!(ax4, [NTS_LON],[NTS_LAT], marker=:star5, markersize=20, color=:white, strokecolor=:black, strokewidth=1)
Colorbar(fig[2,2, Right()], hm4, label="arrival (h post-detonation)")

# shared contour legend
Legend(fig[3, 1:2], [LineElement(color=c, linewidth=3) for c in CCOL],
       ["$(round(Int,l)) mR/h" for l in OBS_LEVELS], "Observed dose-rate contours",
       orientation=:horizontal, tellwidth=false, tellheight=true)

const OUT = joinpath(ROOT, "gpu_transport", "artifacts", "gpu_nancy_corrected_2x2.png")
save(OUT, fig, px_per_unit=2)
println("[plot] saved $(OUT)")
