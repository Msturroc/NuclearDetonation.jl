# Plot observed vs simulated dose rate AND time-of-arrival contours for a
# single forward simulation at the warm-start (best) parameters. Included
# from cmaes_calibration.jl when VIZ_ONLY=1.
#
# Layout: symmetric 2x2 of panel+legend subgrids.
#   (1,1) observed dose  + legend     (1,2) model dose  + legend
#   (2,1) observed TOA   + legend     (2,2) model TOA   + legend

using CairoMakie

const OUT_PNG = joinpath(@__DIR__, "$(TEST_NAME)_cmaes_$(lowercase(string(TURB_SCHEME)))_fit.png")

dose_smooth = LAST_DOSE_SMOOTH[]
isnothing(dose_smooth) && error("VIZ_ONLY: rho_core did not produce a dose field (early return).")

model_snapshots = LAST_MODEL_SNAPSHOTS[]
snapshot_hours  = LAST_SNAPSHOT_HOURS[]
isnothing(model_snapshots) && error("VIZ_ONLY: missing model snapshots (TOA panel).")

const _GZ_LAT = TEST_CONFIG.source_lat
const _GZ_LON = TEST_CONFIG.source_lon

# LAT_GRID/LON_GRID is already padded to a 1.3:1 panel aspect at the top
# of cmaes_calibration.jl, so the model dose covers the entire panel.
ax_lon_min, ax_lon_max = first(LON_GRID), last(LON_GRID)
ax_lat_min, ax_lat_max = first(LAT_GRID), last(LAT_GRID)

# Dose contour palette: pick observation-set levels, assign colours.
contour_levels = sort!(unique(Float64[c.dose_rate_mR_hr for c in OBS.dose_rate_contours]))
const _PALETTE = [:blue, :cyan, :green, :gold, :orange, :red, :darkred,
                  :purple, :magenta, :brown]
contour_colors = [_PALETTE[clamp(i, 1, length(_PALETTE))]
                  for i in eachindex(contour_levels)]

fig = Figure(size = (1400, 1200), fontsize = 14)

Label(fig[0, 1:2],
    "$(TEST_CONFIG.label) $(Int(round(TEST_CONFIG.yield_kt))) kT: Observed vs Model",
    fontsize = 18, font = :bold)

gl_dose_obs = fig[1, 1] = GridLayout()
gl_dose_mod = fig[1, 2] = GridLayout()
gl_toa_obs  = fig[2, 1] = GridLayout()
gl_toa_mod  = fig[2, 2] = GridLayout()

# --- (1,1) observed dose ----------------------------------------------
ax_dose_obs = Axis(gl_dose_obs[1, 1];
    title = "Observed Dose Rate at H+12",
    xlabel = "Longitude (°)", ylabel = "Latitude (°)",
    limits = (ax_lon_min, ax_lon_max, ax_lat_min, ax_lat_max),
    aspect = DataAspect())
for (level, col) in zip(contour_levels, contour_colors)
    for cont in OBS.dose_rate_contours
        cont.dose_rate_mR_hr != level && continue
        for poly in cont.polygons
            length(poly) < 2 && continue
            lats = [pt[1] for pt in poly]
            lons = [pt[2] for pt in poly]
            lines!(ax_dose_obs, lons, lats; color = col, linewidth = 2.5)
        end
    end
end
scatter!(ax_dose_obs, [_GZ_LON], [_GZ_LAT];
    marker = :star5, markersize = 20, color = :black)

# --- (1,2) model dose -------------------------------------------------
ax_dose_mod = Axis(gl_dose_mod[1, 1];
    title = "Model Dose Rate at H+12",
    xlabel = "Longitude (°)", ylabel = "Latitude (°)",
    limits = (ax_lon_min, ax_lon_max, ax_lat_min, ax_lat_max),
    aspect = DataAspect())
for (level, col) in zip(contour_levels, contour_colors)
    maximum(dose_smooth) >= level || continue
    contour!(ax_dose_mod, collect(LON_GRID), collect(LAT_GRID), dose_smooth;
        levels = [level], color = col, linewidth = 2.5)
end
scatter!(ax_dose_mod, [_GZ_LON], [_GZ_LAT];
    marker = :star5, markersize = 20, color = :black)

dose_entries = [LineElement(color = c, linewidth = 3) for c in contour_colors]
dose_labels  = ["$(l) mR/h" for l in contour_levels]
Legend(gl_dose_obs[2, 1], dose_entries, dose_labels, "Dose Rate (H+12)";
    orientation = :horizontal, tellwidth = false, tellheight = true, nbanks = 1)
Legend(gl_dose_mod[2, 1], dose_entries, dose_labels, "Dose Rate (H+12)";
    orientation = :horizontal, tellwidth = false, tellheight = true, nbanks = 1)

# --- TOA preparation --------------------------------------------------
obs_toa = OBS.toa_contours
obs_hours = sort!(unique(Float64[tc.hour for tc in obs_toa]))
isempty(obs_hours) && (obs_hours = collect(1.0:1.0:maximum(snapshot_hours)))
toa_cmap = cgrad(:viridis, length(obs_hours), categorical = true)

nx_grid, ny_grid = length(LON_GRID), length(LAT_GRID)
toa_extra_sigma = 5.0
model_snapshots_smooth = [gaussian_smooth(snap, toa_extra_sigma)
                          for snap in model_snapshots]
max_dose_toa = maximum(model_snapshots_smooth[end])
toa_threshold = max_dose_toa * 0.001
model_toa = fill(NaN, nx_grid, ny_grid)
for (idx, snap) in enumerate(model_snapshots_smooth)
    hr = snapshot_hours[idx]
    for j in 1:ny_grid, i in 1:nx_grid
        if isnan(model_toa[i, j]) && snap[i, j] > toa_threshold
            model_toa[i, j] = hr
        end
    end
end

# --- (2,1) observed TOA -----------------------------------------------
ax_toa_obs = Axis(gl_toa_obs[1, 1];
    title = "Observed Time of Arrival",
    xlabel = "Longitude (°)", ylabel = "Latitude (°)",
    limits = (ax_lon_min, ax_lon_max, ax_lat_min, ax_lat_max),
    aspect = DataAspect())
for tc in obs_toa
    ci = clamp(searchsortedlast(obs_hours, tc.hour), 1, length(obs_hours))
    col = toa_cmap[ci]
    for line in tc.lines
        length(line) < 2 && continue
        lats = [pt[1] for pt in line]
        lons = [pt[2] for pt in line]
        lines!(ax_toa_obs, lons, lats; color = col, linewidth = 2.5)
    end
end
scatter!(ax_toa_obs, [_GZ_LON], [_GZ_LAT];
    marker = :star5, markersize = 20, color = :black)

# --- (2,2) model TOA --------------------------------------------------
ax_toa_mod = Axis(gl_toa_mod[1, 1];
    title = "Model Time of Arrival",
    xlabel = "Longitude (°)", ylabel = "Latitude (°)",
    limits = (ax_lon_min, ax_lon_max, ax_lat_min, ax_lat_max),
    aspect = DataAspect())
if any(isfinite, model_toa)
    for (hi, h) in enumerate(obs_hours)
        contour!(ax_toa_mod, collect(LON_GRID), collect(LAT_GRID), model_toa;
            levels = [h], color = toa_cmap[hi], linewidth = 2.5)
    end
end
scatter!(ax_toa_mod, [_GZ_LON], [_GZ_LAT];
    marker = :star5, markersize = 20, color = :black)

toa_entries = [LineElement(color = toa_cmap[i], linewidth = 3)
               for i in eachindex(obs_hours)]
toa_labels  = ["H+$(Int(round(h)))" for h in obs_hours]
Legend(gl_toa_obs[2, 1], toa_entries, toa_labels, "Time of Arrival";
    orientation = :horizontal, tellwidth = false, tellheight = true, nbanks = 1)
Legend(gl_toa_mod[2, 1], toa_entries, toa_labels, "Time of Arrival";
    orientation = :horizontal, tellwidth = false, tellheight = true, nbanks = 1)

linkaxes!(ax_dose_obs, ax_dose_mod, ax_toa_obs, ax_toa_mod)
colsize!(fig.layout, 1, Relative(0.5))
colsize!(fig.layout, 2, Relative(0.5))

# Right-column y axis is redundant — drop ticklabels + label so the two
# columns sit flush.
hideydecorations!(ax_dose_mod, ticks=false, grid=false, ticklabels=true, label=true)
hideydecorations!(ax_toa_mod,  ticks=false, grid=false, ticklabels=true, label=true)

colgap!(fig.layout, 4)
rowgap!(fig.layout, 12)
for gl in (gl_dose_obs, gl_dose_mod, gl_toa_obs, gl_toa_mod)
    rowgap!(gl, 4)
end

save(OUT_PNG, fig; px_per_unit = 2)
println("Saved fit plot: $(basename(OUT_PNG))")
