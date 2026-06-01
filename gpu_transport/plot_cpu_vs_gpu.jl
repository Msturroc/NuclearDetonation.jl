#!/usr/bin/env julia
# CPU reference vs GPU shadow side-by-side dose-rate plot
# ========================================================
# Reads the latest best params from gpu_nancy_cmaes_v3_best.txt, runs both
# the Float64 CPU reference and the Float32 GPU kernel on those params, bins
# both to (LON_GRID, LAT_GRID), smooths and converts to mR/h, then produces
# a 3-panel CairoMakie comparison:
#   left   — CPU reference dose grid
#   middle — GPU shadow dose grid
#   right  — pixel-wise difference (GPU - CPU) clamped to ±50%
# All three panels overlay the digitised Nancy observed dose-rate contours.
#
# Run:
#   julia --project=/home/marc/NuclearDetonation.jl \
#       gpu_transport/plot_cpu_vs_gpu.jl
#
# Output:
#   /home/marc/julia_snap_explorations/gpu_nancy_cpu_vs_gpu.png

ENV["MAX_EVALS"] = "0"

using Random
using Statistics
using Printf
using CUDA
using CairoMakie
using NuclearDetonation
using NuclearDetonation.Transport

const _UPSTREAM = "/home/marc/NuclearDetonation.jl/examples/nancy_cmaes_particle_size.jl"
println("[plot] including upstream cmaes script (loop suppressed)…")
include(_UPSTREAM)

include(joinpath(@__DIR__, "cpu_reference.jl"))
include(joinpath(@__DIR__, "host_shadow_v2.jl"))
include(joinpath(@__DIR__, "validate_against_reference.jl"))
include(joinpath(@__DIR__, "met_upload.jl"))
include(joinpath(@__DIR__, "gpu_kernel_v2.jl"))

# ----------------------------------------------------------------------------
# Read best params from gpu_nancy_cmaes_v3_best.txt
# ----------------------------------------------------------------------------
const BEST_FILE = "/home/marc/julia_snap_explorations/gpu_nancy_cmaes_v3_best.txt"
println("[plot] loading params from $(BEST_FILE)…")
param_dict = Dict{String,Float64}()
diag_dict  = Dict{String,Float64}()
for line in eachline(BEST_FILE)
    isempty(strip(line)) && continue
    if startswith(line, "#")
        parts = split(strip(replace(line, "#" => "")), '\t')
        length(parts) >= 2 || continue
        diag_dict[parts[1]] = parse(Float64, parts[2])
        continue
    end
    parts = split(line, '\t')
    length(parts) >= 2 || continue
    param_dict[parts[1]] = parse(Float64, parts[2])
end
params = [param_dict[name] for name in PARAM_NAMES]
println("[plot] best loss=", get(diag_dict, "loss", NaN),
        " (score=", round((1 - get(diag_dict, "loss", 1.0)) * 100, digits = 2), "%)")

# Same seed for both runs — particle initial positions match
const PLOT_SEED = UInt64(0xCAFE_F00D_BEEF)

# ----------------------------------------------------------------------------
# Run CPU reference (Float64, ~13 s)
# ----------------------------------------------------------------------------
println("[plot] running CPU reference (Float64)…")
t1 = time()
ref = run_reference_simulation(params, PLOT_SEED)
println("[plot]   ", round(time() - t1, digits = 2), " s, ", ref.n_dep_events,
        " events, total Bq = ", round(ref.total_dep_bq, sigdigits = 4))

cpu_grid_bq = bin_reference_to_grid(ref.deposition_log)

# ----------------------------------------------------------------------------
# Run GPU shadow (Float32, ~30 ms)
# ----------------------------------------------------------------------------
println("[plot] uploading met to GPU and running GPU shadow…")
windows = load_nancy_gpu_windows()
# JIT warm-up
_ = run_gpu_shadow(params, PLOT_SEED; windows = windows)
CUDA.synchronize()
t2 = time()
gpu_grid_f32, _, n_alive = run_gpu_shadow(params, PLOT_SEED; windows = windows)
CUDA.synchronize()
println("[plot]   ", round(time() - t2, digits = 3), " s, ",
        round(sum(gpu_grid_f32), sigdigits = 4), " Bq, ", n_alive, " alive")

gpu_grid_bq = Float64.(gpu_grid_f32)

# ----------------------------------------------------------------------------
# Smooth + convert to mR/h
# ----------------------------------------------------------------------------
smooth_sigma = params[20]
cpu_dose_mRh = gaussian_smooth(cpu_grid_bq .* DOSE_FACTOR, smooth_sigma)
gpu_dose_mRh = gaussian_smooth(gpu_grid_bq .* DOSE_FACTOR, smooth_sigma)

# Per-threshold FMS for both
function fms_dict(dose)
    out = Dict{Float64,Float64}()
    for (lvl, mask) in OBS_MASKS
        m = dose .>= lvl
        inter = sum(m .& mask)
        uni   = sum(m .| mask)
        out[lvl] = uni > 0 ? inter / uni : 0.0
    end
    return out
end
fms_cpu = fms_dict(cpu_dose_mRh)
fms_gpu = fms_dict(gpu_dose_mRh)

println("\n[plot] FMS comparison:")
@printf("  %7s   %s   %s    Δ\n", "thresh", "  cpu", "  gpu")
for k in sort(collect(keys(fms_cpu)))
    @printf("  %7.2f   %.4f    %.4f   %+.4f\n", k, fms_cpu[k], fms_gpu[k], fms_gpu[k] - fms_cpu[k])
end

# ----------------------------------------------------------------------------
# Build the figure
# ----------------------------------------------------------------------------
println("\n[plot] building figure…")
const NTS_LAT = 37.0956
const NTS_LON = -116.1028
const OBS_LEVELS = sort(collect(keys(OBS_MASKS)))
const CONTOUR_COLORS = [:blue, :cyan, :green, :yellow, :orange, :red]

# Set the axis limits to comfortably contain both the obs contours and the model deposition.
function dose_extent(dose)
    inds = findall(>(0.0), dose)
    isempty(inds) && return nothing
    lons = [LON_GRID[i[1]] for i in inds]
    lats = [LAT_GRID[i[2]] for i in inds]
    return extrema(lons), extrema(lats)
end

cpu_ext = dose_extent(cpu_dose_mRh)
gpu_ext = dose_extent(gpu_dose_mRh)
obs_lats = Float64[]
obs_lons = Float64[]
for c in NANCY_OBS.dose_rate_contours, p in c.polygons, pt in p
    push!(obs_lats, pt[1])
    push!(obs_lons, pt[2])
end

lon_lo = min(minimum(obs_lons), cpu_ext[1][1], gpu_ext[1][1]) - 0.2
lon_hi = max(maximum(obs_lons), cpu_ext[1][2], gpu_ext[1][2]) + 0.2
lat_lo = min(minimum(obs_lats), cpu_ext[2][1], gpu_ext[2][1]) - 0.2
lat_hi = max(maximum(obs_lats), cpu_ext[2][2], gpu_ext[2][2]) + 0.2
ax_lims = (lon_lo, lon_hi, lat_lo, lat_hi)

# Use log10 colour scale for dose grids (values span 4-5 decades)
function safe_log10_transform(grid; floor_val = 0.05)
    out = similar(grid, Float64)
    @inbounds for i in eachindex(grid)
        v = grid[i]
        out[i] = v < floor_val ? NaN : log10(v)
    end
    return out
end

cpu_log = safe_log10_transform(cpu_dose_mRh)
gpu_log = safe_log10_transform(gpu_dose_mRh)
all_finite = vcat(filter(isfinite, cpu_log), filter(isfinite, gpu_log))
log_lo = isempty(all_finite) ? -1.0 : minimum(all_finite)
log_hi = isempty(all_finite) ? 3.0 : maximum(all_finite)

fig = Figure(size = (1600, 720), fontsize = 13)

score_pct = round((1 - get(diag_dict, "loss", 1.0)) * 100, digits = 2)
title_str = "Nancy 24 kT — CPU reference vs GPU shadow on best params (score $(score_pct)%)"
Label(fig[0, 1:3], title_str, fontsize = 17, halign = :center)

function draw_dose!(ax, log_dose, label, fms_d)
    # heatmap with NaN → transparent
    hm = heatmap!(ax, collect(LON_GRID), collect(LAT_GRID), log_dose,
                  colormap = :viridis, colorrange = (log_lo, log_hi), nan_color = :transparent)
    # observed contours
    for (level, col) in zip(OBS_LEVELS, CONTOUR_COLORS)
        for c in NANCY_OBS.dose_rate_contours
            c.dose_rate_mR_hr != level && continue
            for poly in c.polygons
                lats = [p[1] for p in poly]
                lons = [p[2] for p in poly]
                lines!(ax, lons, lats, color = col, linewidth = 1.6)
            end
        end
    end
    scatter!(ax, [NTS_LON], [NTS_LAT], marker = :star5, markersize = 18, color = :white, strokecolor = :black, strokewidth = 1)
    # FMS overlay
    fms_str = join(["$(round(Int, k)) mR: $(round(fms_d[k], digits = 2))" for k in OBS_LEVELS], "\n")
    text!(ax, lon_lo + 0.05, lat_hi - 0.05, text = label * "\n" * fms_str,
          align = (:left, :top), color = :white, fontsize = 11)
    return hm
end

ax_cpu = Axis(fig[1, 1], title = "CPU reference (Float64)",
              xlabel = "Longitude (°)", ylabel = "Latitude (°)",
              limits = ax_lims, aspect = DataAspect())
hm_cpu = draw_dose!(ax_cpu, cpu_log, "", fms_cpu)

ax_gpu = Axis(fig[1, 2], title = "GPU shadow (Float32)",
              xlabel = "Longitude (°)", ylabel = "",
              limits = ax_lims, aspect = DataAspect())
hm_gpu = draw_dose!(ax_gpu, gpu_log, "", fms_gpu)

# Difference panel — relative diff (gpu - cpu) / max(cpu, floor)
floor_d = 0.5
rel_diff = similar(cpu_dose_mRh, Float64)
@inbounds for i in eachindex(rel_diff)
    a = cpu_dose_mRh[i]
    b = gpu_dose_mRh[i]
    if max(a, b) < floor_d
        rel_diff[i] = NaN
    else
        rel_diff[i] = (b - a) / max(a, floor_d)
    end
end

ax_dif = Axis(fig[1, 3], title = "GPU − CPU (relative to max(CPU, $(floor_d)))",
              xlabel = "Longitude (°)", ylabel = "",
              limits = ax_lims, aspect = DataAspect())
hm_dif = heatmap!(ax_dif, collect(LON_GRID), collect(LAT_GRID), rel_diff,
                  colormap = :balance, colorrange = (-1.0, 1.0), nan_color = :transparent)
for (level, col) in zip(OBS_LEVELS, CONTOUR_COLORS)
    for c in NANCY_OBS.dose_rate_contours
        c.dose_rate_mR_hr != level && continue
        for poly in c.polygons
            lats = [p[1] for p in poly]
            lons = [p[2] for p in poly]
            lines!(ax_dif, lons, lats, color = col, linewidth = 1.6)
        end
    end
end
scatter!(ax_dif, [NTS_LON], [NTS_LAT], marker = :star5, markersize = 18, color = :white, strokecolor = :black, strokewidth = 1)

Colorbar(fig[1, 4], hm_cpu, label = "log₁₀(dose, mR/h)", height = Relative(0.85))
Colorbar(fig[1, 5], hm_dif, label = "(GPU − CPU) / max(CPU, $(floor_d))", height = Relative(0.85))

# Legend for contours
legend_elements = [LineElement(color = c, linewidth = 3) for c in CONTOUR_COLORS]
legend_labels   = ["$(l) mR/h" for l in OBS_LEVELS]
Legend(fig[2, 1:5], legend_elements, legend_labels, "Observed dose-rate contours",
       orientation = :horizontal, tellwidth = false, tellheight = true)

const OUTFILE = "/home/marc/julia_snap_explorations/gpu_nancy_cpu_vs_gpu.png"
save(OUTFILE, fig, px_per_unit = 2)
println("\n[plot] saved $(OUTFILE)")
