# Nancy Contour Comparison
# ========================
# Loads digitised historical fallout observations and compares against
# model output. Computes FMS (Figure of Merit in Space) scores.
#
# Requires PlotlyJS for visualisation:
#   ] add PlotlyJS

using NuclearDetonation
using NuclearDetonation.Transport

# --- Load digitised Nancy observations (bundled with package) ---
nancy_obs = load_nancy_observations()

println("Nancy test metadata:")
println("  Detonation: $(nancy_obs.detonation_lat)°N, $(nancy_obs.detonation_lon)°E")
println("  Yield: $(nancy_obs.yield_kt) kT")
println("  HOB: $(nancy_obs.hob_m) m")
println("  Time: $(nancy_obs.detonation_utc)")
println("  Dose rate contours: $(length(nancy_obs.dose_rate_contours))")
println("  TOA contours: $(length(nancy_obs.toa_contours))")

# --- Suggest a comparison grid ---
lat_grid, lon_grid = suggest_grid(nancy_obs, resolution_km=2.0)
println("\nComparison grid: $(length(lat_grid)) × $(length(lon_grid)) " *
        "($(round(lat_grid[1], digits=2))–$(round(lat_grid[end], digits=2))°N, " *
        "$(round(lon_grid[1], digits=2))–$(round(lon_grid[end], digits=2))°E)")

# --- Rasterise observation contours ---
obs_masks = rasterise_all_contours(nancy_obs.dose_rate_contours, lat_grid, lon_grid)
println("Rasterised $(length(obs_masks)) dose rate thresholds")

# --- Example: compute FMS against synthetic/model dose rate grid ---
# Replace this with actual model output from nancy_bomb_release.jl
println("\nTo compute FMS scores, load your model dose rate grid:")
println("  fms_results = compute_multi_threshold_fms(model_dose_rate, obs_masks, lat_grid, lon_grid)")
println("  score = compute_combined_score(fms_results)")
println("  print_validation_summary(score)")

# --- Plotting with PlotlyJS (optional) ---
# using PlotlyJS
#
# traces = GenericTrace[]
# for contour in nancy_obs.dose_rate_contours
#     for polygon in contour.polygons
#         lats = [p[1] for p in polygon]
#         lons = [p[2] for p in polygon]
#         push!(traces, scatter(x=lons, y=lats, mode="lines",
#               name="$(contour.dose_rate_mR_hr) mR/hr",
#               line=attr(width=2)))
#     end
# end
# plot(traces, Layout(title="Nancy Dose Rate Contours",
#                     xaxis_title="Longitude", yaxis_title="Latitude",
#                     scaleanchor="x", scaleratio=1))
