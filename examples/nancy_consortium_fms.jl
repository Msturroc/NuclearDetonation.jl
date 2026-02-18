#!/usr/bin/env julia
# Nancy Consortium FMS Comparison
# ================================
# Computes FMS scores for all WP3 consortium members and our model,
# produces individual side-by-side plots (observations | model) for each.
#
# Usage:
#   julia --project=. examples/nancy_consortium_fms.jl

using NuclearDetonation
using NuclearDetonation.Transport
using CairoMakie
using NCDatasets
using Shapefile
using Printf
using Statistics: mean

println("="^70)
println("NANCY CONSORTIUM FMS COMPARISON")
println("="^70)

# ============================================================================
# 1. Load observations and create common grid
# ============================================================================

println("\n1. Loading Nancy observations...")
const NANCY_CONTOUR_DIR = joinpath(
    "/home/marc/NuclearDetonations.jl/WP3_Model_Comparison/Task 3.1",
    "Nancy_exposurerate_contours_digitised", "geoJSON")
const NANCY_OBS = Transport.load_nancy_observations(NANCY_CONTOUR_DIR)
const LAT_GRID, LON_GRID = Transport.suggest_grid(NANCY_OBS; resolution_km=2.0, buffer_fraction=0.1)
const OBS_MASKS = Transport.rasterise_all_contours(NANCY_OBS.dose_rate_contours,
    collect(LAT_GRID), collect(LON_GRID))
const NX_OBS, NY_OBS = length(LON_GRID), length(LAT_GRID)
println("   Grid: $(NX_OBS) x $(NY_OBS) cells")

const OBS_LEVELS = sort(collect(keys(OBS_MASKS)))
println("   Contour levels: $(OBS_LEVELS) mR/h")

const NTS_LAT = 37.0956
const NTS_LON = -116.1028

# Axis limits from observation bounds + buffer
const OBS_BOUNDS = Transport.contour_bounds(NANCY_OBS.dose_rate_contours)
const LAT_BUF = 0.15 * (OBS_BOUNDS[2] - OBS_BOUNDS[1])
const LON_BUF = 0.15 * (OBS_BOUNDS[4] - OBS_BOUNDS[3])
const AX_LIMS = (OBS_BOUNDS[3] - LON_BUF, OBS_BOUNDS[4] + LON_BUF,
                 OBS_BOUNDS[1] - LAT_BUF, OBS_BOUNDS[2] + LAT_BUF)

const CONTOUR_COLORS = [:blue, :cyan, :green, :yellow, :orange, :red]

const BASE = "/home/marc/NuclearDetonations.jl/WP3_Model_Comparison/Task 3.1"
const OUTPUTS = joinpath(BASE, "Model_Outputs_Phase1")
const OUTDIR = joinpath(@__DIR__, "nancy_fms_plots")
mkpath(OUTDIR)

# ============================================================================
# 2. Helper functions
# ============================================================================

function gaussian_smooth(field::Matrix{T}, sigma::Real; truncate::Real=4.0) where T
    radius = ceil(Int, sigma * truncate)
    kernel_1d = [exp(-0.5 * (x / sigma)^2) for x in -radius:radius]
    kernel_1d ./= sum(kernel_1d)
    nx, ny = size(field)
    temp = zeros(T, nx, ny)
    smoothed = zeros(T, nx, ny)
    for j in 1:ny, i in 1:nx
        val, weight = zero(T), zero(T)
        for k in -radius:radius
            ii = i + k
            if 1 <= ii <= nx
                w = kernel_1d[k + radius + 1]
                val += field[ii, j] * w; weight += w
            end
        end
        temp[i, j] = weight > 0 ? val / weight : zero(T)
    end
    for i in 1:nx, j in 1:ny
        val, weight = zero(T), zero(T)
        for k in -radius:radius
            jj = j + k
            if 1 <= jj <= ny
                w = kernel_1d[k + radius + 1]
                val += temp[i, jj] * w; weight += w
            end
        end
        smoothed[i, j] = weight > 0 ? val / weight : zero(T)
    end
    return smoothed
end

"""Nearest-neighbor interpolation from source grid to observation grid."""
function interpolate_to_obs_grid(src_lon, src_lat, src_data)
    result = zeros(NX_OBS, NY_OBS)
    for j in 1:NY_OBS
        jj = argmin(abs.(src_lat .- LAT_GRID[j]))
        for i in 1:NX_OBS
            ii = argmin(abs.(src_lon .- LON_GRID[i]))
            if 1 <= ii <= size(src_data, 1) && 1 <= jj <= size(src_data, 2)
                v = src_data[ii, jj]
                result[i, j] = ismissing(v) ? 0.0 : Float64(v)
            end
        end
    end
    return result
end

"""Mercator (EPSG:3857) to lon/lat."""
function merc_to_lonlat(x, y)
    lon = x / 6378137.0 * (180.0 / π)
    lat = atan(exp(y / 6378137.0)) * (360.0 / π) - 90.0
    return lon, lat
end

"""Rasterise shapefile polygons onto the observation grid using centroid + fill."""
function rasterise_shapefile_polygons(shp, values; convert_merc=false)
    result = zeros(NX_OBS, NY_OBS)
    lon_arr = collect(LON_GRID)
    lat_arr = collect(LAT_GRID)

    for (idx, row) in enumerate(shp)
        val = values[idx]
        val <= 0 && continue
        geom = Shapefile.shape(row)
        geom === nothing && continue

        # Extract points from polygon
        local pts_list
        if geom isa Shapefile.Polygon || geom isa Shapefile.PolygonZ || geom isa Shapefile.PolygonM
            pts_list = geom.points
        else
            continue
        end

        # Get lon/lat coordinates
        coords = if convert_merc
            [merc_to_lonlat(p.x, p.y) for p in pts_list]
        else
            [(p.x, p.y) for p in pts_list]
        end

        isempty(coords) && continue

        # Bounding box
        lons_poly = [c[1] for c in coords]
        lats_poly = [c[2] for c in coords]
        lon_min, lon_max = extrema(lons_poly)
        lat_min, lat_max = extrema(lats_poly)

        # Find grid cells in bounding box
        i_min = searchsortedfirst(lon_arr, lon_min) - 1
        i_max = searchsortedlast(lon_arr, lon_max) + 1
        j_min = searchsortedfirst(lat_arr, lat_min) - 1
        j_max = searchsortedlast(lat_arr, lat_max) + 1
        i_min = clamp(i_min, 1, NX_OBS)
        i_max = clamp(i_max, 1, NX_OBS)
        j_min = clamp(j_min, 1, NY_OBS)
        j_max = clamp(j_max, 1, NY_OBS)

        # Convert to (lat, lon) tuples for point_in_polygon
        poly_latlon = [(c[2], c[1]) for c in coords]

        for j in j_min:j_max
            for i in i_min:i_max
                if Transport.point_in_polygon(lat_arr[j], lon_arr[i], poly_latlon)
                    result[i, j] = max(result[i, j], val)
                end
            end
        end
    end
    return result
end

"""Compute per-level FMS for a model dose rate field on the observation grid."""
function compute_fms_table(model_dose_mRh::Matrix{Float64})
    fms_scores = Dict{Float64, Float64}()
    for level in OBS_LEVELS
        obs_mask = OBS_MASKS[level]
        model_mask = model_dose_mRh .>= level
        inter = Float64(sum(model_mask .& obs_mask))
        uni = Float64(sum(model_mask .| obs_mask))
        fms_scores[level] = uni > 0 ? inter / uni : 0.0
    end
    return fms_scores
end

"""Create a side-by-side figure: observations (left) vs model (right)."""
function make_comparison_plot(model_name, model_dose, lon_grid, lat_grid;
                              filename="", subtitle="")
    fig = Figure(size=(1400, 800), fontsize=14)

    # Left: observations
    ax_obs = Axis(fig[1, 1],
        title = "Nancy 24 kT — Observed Dose Rate at H+12",
        xlabel = "Longitude (°)", ylabel = "Latitude (°)",
        limits = AX_LIMS, aspect = DataAspect())

    for (level, col) in zip(OBS_LEVELS, CONTOUR_COLORS)
        for contour in NANCY_OBS.dose_rate_contours
            contour.dose_rate_mR_hr != level && continue
            for polygon in contour.polygons
                lats = [p[1] for p in polygon]
                lons = [p[2] for p in polygon]
                lines!(ax_obs, lons, lats, color=col, linewidth=2.5)
            end
        end
    end
    scatter!(ax_obs, [NTS_LON], [NTS_LAT], marker=:star5, markersize=20, color=:black)

    # Right: model
    ax_mod = Axis(fig[1, 2],
        title = "$(model_name) — Model Dose Rate at H+12\n$(subtitle)",
        xlabel = "Longitude (°)", ylabel = "Latitude (°)",
        limits = AX_LIMS, aspect = DataAspect())

    for (level, col) in zip(OBS_LEVELS, CONTOUR_COLORS)
        contour!(ax_mod, collect(lon_grid), collect(lat_grid), model_dose,
            levels=[level], color=col, linewidth=2.5)
    end
    scatter!(ax_mod, [NTS_LON], [NTS_LAT], marker=:star5, markersize=20, color=:black)

    # Legend
    legend_elements = [LineElement(color=c, linewidth=3) for c in CONTOUR_COLORS]
    legend_labels = ["$(l) mR/h" for l in OBS_LEVELS]
    Legend(fig[2, :], legend_elements, legend_labels, "Dose Rate (H+12)",
        orientation=:horizontal, tellwidth=false, tellheight=true)

    if !isempty(filename)
        save(filename, fig, px_per_unit=2)
    end
    return fig
end

# Decay correction: 48h → 12h using t^(-1.2) law
const DECAY_48_TO_12 = (12.0 / 48.0)^(-1.2)   # = 4^1.2 ≈ 5.278
const MSV_TO_MRH = 100.0                        # 1 mSv/h ≈ 100 mR/h

# ============================================================================
# 3. Load and process each model
# ============================================================================

# Store results: model_name => (fms_dict, dose_field)
all_results = Dict{String, Dict{Float64, Float64}}()

# ── BfS (Germany) ──
println("\n2. BfS (Germany) — JRODOS, 48h NetCDF...")
try
    bfs_file = joinpath(OUTPUTS, "BfS_results", "Nancy", "doserate_48h.nc")
    NCDataset(bfs_file) do ds
        lon = Float64.(ds["lon"][:]); lat = Float64.(ds["lat"][:])
        data_raw = Float64.(ds["doserate"][:,:])  # mSv/h at 48h, dims: lon × lat
        data_mRh = data_raw .* DECAY_48_TO_12 .* MSV_TO_MRH
        println("   Raw: max=$(round(maximum(data_raw), digits=2)) mSv/h at 48h")
        println("   Converted: max=$(round(maximum(data_mRh), digits=1)) mR/h at H+12")

        dose_on_grid = interpolate_to_obs_grid(lon, lat, data_mRh)
        fms = compute_fms_table(dose_on_grid)
        all_results["BfS"] = fms

        make_comparison_plot("BfS (Germany)", dose_on_grid, LON_GRID, LAT_GRID,
            filename=joinpath(OUTDIR, "nancy_fms_BfS.png"),
            subtitle="JRODOS | 48h→12h decay-corrected")
        println("   Saved plot")
    end
catch e
    println("   ERROR: $e")
end

# ── UKMO (UK) ──
println("\n3. UKMO (UK) — NAME model, 48h NetCDF...")
try
    ukmo_file = joinpath(OUTPUTS, "UKMO", "Nancy_netCDF", "Fields_grid8_C1_T1_195303261310.nc")
    NCDataset(ukmo_file) do ds
        lon = Float64.(ds["longitude"][:]); lat = Float64.(ds["latitude"][:])
        data_raw = Float64.(ds["effectiveexternal"][:,:])  # mSv/h at 48h
        data_mRh = data_raw .* DECAY_48_TO_12 .* MSV_TO_MRH
        println("   Raw: max=$(round(maximum(data_raw), digits=2)) mSv/h at 48h")
        println("   Converted: max=$(round(maximum(data_mRh), digits=1)) mR/h at H+12")

        dose_on_grid = interpolate_to_obs_grid(lon, lat, data_mRh)
        fms = compute_fms_table(dose_on_grid)
        all_results["UKMO"] = fms

        make_comparison_plot("UKMO (UK)", dose_on_grid, LON_GRID, LAT_GRID,
            filename=joinpath(OUTDIR, "nancy_fms_UKMO.png"),
            subtitle="NAME | 48h→12h decay-corrected")
        println("   Saved plot")
    end
catch e
    println("   ERROR: $e")
end

# ── KIT (Germany) ──
println("\n4. KIT (Germany) — JRODOS, 12h shapefiles...")
try
    kit_shp_path = joinpath(BASE, "KIT-results-April-2025", "Nancy", "Nancy-hybrid-dose-rate-12h.shp")
    shp = Shapefile.Table(kit_shp_path)
    rows = collect(shp)
    vals_mSv = Float64[row.Value for row in rows]  # mSv/h at 12h
    vals_mRh = vals_mSv .* MSV_TO_MRH
    println("   $(length(rows)) features, max=$(round(maximum(vals_mSv), digits=3)) mSv/h = $(round(maximum(vals_mRh), digits=1)) mR/h")

    dose_on_grid = rasterise_shapefile_polygons(rows, vals_mRh; convert_merc=true)
    fms = compute_fms_table(dose_on_grid)
    all_results["KIT"] = fms

    make_comparison_plot("KIT (Germany)", dose_on_grid, LON_GRID, LAT_GRID,
        filename=joinpath(OUTDIR, "nancy_fms_KIT.png"),
        subtitle="JRODOS | 12h shapefile, Mercator→WGS84")
    println("   Saved plot")
catch e
    println("   ERROR: $e")
end

# ── DSA/MET Norway ──
println("\n5. DSA/MET Norway — SNAP model, 12h shapefiles...")
try
    dsa_shp_path = joinpath(OUTPUTS, "DSA_MET", "NANCY", "Longrange  nancy_20250318_gdrd.shp")
    shp = Shapefile.Table(dsa_shp_path)
    rows = collect(shp)
    # Column "mRprhr@12h" — already in mR/h at H+12
    vals_mRh = Float64[]
    col_sym = Symbol("mRprhr@12h")
    for row in rows
        push!(vals_mRh, Float64(getproperty(row, col_sym)))
    end
    println("   $(length(rows)) features, max=$(round(maximum(vals_mRh), digits=1)) mR/h")

    dose_on_grid = rasterise_shapefile_polygons(rows, vals_mRh; convert_merc=false)
    fms = compute_fms_table(dose_on_grid)
    all_results["DSA"] = fms

    make_comparison_plot("DSA/MET Norway", dose_on_grid, LON_GRID, LAT_GRID,
        filename=joinpath(OUTDIR, "nancy_fms_DSA.png"),
        subtitle="SNAP | 12h shapefile, mR/h native")
    println("   Saved plot")
catch e
    println("   ERROR: $e")
end

# ── DEMA (Denmark) ──
println("\n6. DEMA (Denmark) — 48h shapefiles from zip...")
try
    dema_zip = joinpath(OUTPUTS, "DEMA_Results", "Nancy_Results", "Nancy_24kT", "Nancy_Results.zip")
    dema_dir = joinpath(OUTPUTS, "DEMA_Results", "Nancy_Results", "Nancy_24kT", "extracted")
    if !isdir(dema_dir)
        mkpath(dema_dir)
        run(`unzip -o $dema_zip -d $dema_dir`)
    end
    dema_shp_path = joinpath(dema_dir, "Longrange  Nancy_24kT_ERA5_gdrd.shp")
    shp = Shapefile.Table(dema_shp_path)
    rows = collect(shp)

    # The Value column contains dose rate
    first_row = rows[1]
    col_names = propertynames(first_row)
    println("   Columns: $col_names")
    println("   Using column: Value")
    vals_raw = Float64[Float64(row.Value) for row in rows]

    # DEMA overview says units are Sv/hr at 48h. Convert: Sv→mSv(×1000), 48h→12h, mSv→mR
    # But max was 9.69e-07 mSv/hr — that's tiny. Check the actual max.
    raw_max = maximum(vals_raw)
    println("   $(length(rows)) features, raw max=$raw_max")

    # Determine conversion: if max < 1, likely Sv/h at 48h
    local vals_mRh
    if raw_max < 0.01  # Likely Sv/h
        vals_mRh = vals_raw .* 1000.0 .* DECAY_48_TO_12 .* MSV_TO_MRH  # Sv→mSv→mR, 48h→12h
        println("   Interpreted as Sv/h at 48h → max $(round(maximum(vals_mRh), digits=1)) mR/h at H+12")
    elseif raw_max < 10.0  # Likely mSv/h
        vals_mRh = vals_raw .* DECAY_48_TO_12 .* MSV_TO_MRH
        println("   Interpreted as mSv/h at 48h → max $(round(maximum(vals_mRh), digits=1)) mR/h at H+12")
    else
        vals_mRh = vals_raw  # Assume already mR/h
        println("   Interpreted as mR/h → max $(round(maximum(vals_mRh), digits=1)) mR/h")
    end

    dose_on_grid = rasterise_shapefile_polygons(rows, vals_mRh; convert_merc=false)
    fms = compute_fms_table(dose_on_grid)
    all_results["DEMA"] = fms

    make_comparison_plot("DEMA (Denmark)", dose_on_grid, LON_GRID, LAT_GRID,
        filename=joinpath(OUTDIR, "nancy_fms_DEMA.png"),
        subtitle="ARGOS/RIMPUFF")
    println("   Saved plot")
catch e
    println("   DEMA ERROR: $e")
end

# ── Our Model (OU) ──
println("\n7. Our Model — CMA-ES OU best (latest from nancy_bomb_release.jl)...")
try
    our_file = joinpath(@__DIR__, "nancy_dosegrid_ou.nc")
    params = nancy_optimised_config()
    smooth_sigma = params.physics_scales.smooth_sigma

    NCDataset(our_file) do ds
        lon = Float64.(ds["lon"][:]); lat = Float64.(ds["lat"][:])
        data_mRh = Float64.(ds["doserate"][:,:])  # already mR/h at H+12
        data_smooth = gaussian_smooth(data_mRh, smooth_sigma)
        println("   max=$(round(maximum(data_mRh), digits=1)) mR/h (raw), $(round(maximum(data_smooth), digits=1)) (smoothed, σ=$(round(smooth_sigma, digits=2)))")

        dose_on_grid = interpolate_to_obs_grid(lon, lat, data_smooth)
        fms = compute_fms_table(dose_on_grid)
        all_results["Ours"] = fms

        make_comparison_plot("Our Model (OU)", dose_on_grid, LON_GRID, LAT_GRID,
            filename=joinpath(OUTDIR, "nancy_fms_OurModel.png"))
        println("   Saved plot")
    end
catch e
    println("   ERROR: $e")
end

# ============================================================================
# 4. Print FMS comparison table
# ============================================================================

println("\n" * "="^70)
println("FMS COMPARISON TABLE — Nancy 24 kT (H+12)")
println("="^70)

let
    col_order = ["Ours", "UKMO", "BfS", "KIT", "DSA", "DEMA"]
    active_cols = filter(c -> haskey(all_results, c), col_order)

    # Header
    hdr = @sprintf("%-10s", "mR/h")
    for name in active_cols
        hdr *= @sprintf(" %10s", name)
    end
    println(hdr)
    println("-"^(10 + 11 * length(active_cols)))

    # Rows
    for level in OBS_LEVELS
        row = @sprintf("%-10s", "$(level)")
        for name in active_cols
            fms_val = get(all_results[name], level, 0.0)
            row *= @sprintf(" %9.1f%%", fms_val * 100)
        end
        println(row)
    end

    # Mean row
    println("-"^(10 + 11 * length(active_cols)))
    row = @sprintf("%-10s", "Mean")
    for name in active_cols
        scores = [get(all_results[name], level, 0.0) for level in OBS_LEVELS]
        row *= @sprintf(" %9.1f%%", mean(scores) * 100)
    end
    println(row)

    # Geometric mean row
    row = @sprintf("%-10s", "Geo Mean")
    for name in active_cols
        scores = [max(get(all_results[name], level, 0.0), 0.005) for level in OBS_LEVELS]
        gmean = exp(sum(log.(scores)) / length(scores))
        row *= @sprintf(" %9.1f%%", gmean * 100)
    end
    println(row)
end

# ============================================================================
# 5. LaTeX table
# ============================================================================

let
    col_order = ["Ours", "UKMO", "BfS", "KIT", "DSA", "DEMA"]
    active_cols = filter(c -> haskey(all_results, c), col_order)
    # Nice display names
    display_names = Dict("Ours"=>"Our Model", "UKMO"=>"UKMO", "BfS"=>"BfS",
                         "KIT"=>"KIT", "DSA"=>"DSA/MET", "DEMA"=>"DEMA")
    ncols = length(active_cols)

    tex_file = joinpath(OUTDIR, "nancy_fms_table.tex")
    open(tex_file, "w") do f
        println(f, "\\begin{table}[htbp]")
        println(f, "\\centering")
        println(f, "\\caption{Figure of Merit in Space (FMS) for the Nancy 24\\,kT test at H+12, comparing consortium model outputs against digitised observation contours.}")
        println(f, "\\label{tab:nancy_fms}")
        println(f, "\\begin{tabular}{l" * repeat("r", ncols) * "}")
        println(f, "\\toprule")
        hdr = "Dose Rate (mR/h)"
        for name in active_cols
            hdr *= " & $(get(display_names, name, name))"
        end
        println(f, hdr * " \\\\")
        println(f, "\\midrule")

        for level in OBS_LEVELS
            row = level == round(level) ? @sprintf("%.0f", level) : @sprintf("%.1f", level)
            for name in active_cols
                fms_val = get(all_results[name], level, 0.0) * 100
                if fms_val < 0.05
                    row *= " & ---"
                else
                    row *= @sprintf(" & %.1f\\%%", fms_val)
                end
            end
            println(f, row * " \\\\")
        end

        println(f, "\\midrule")

        # Mean
        row = "Mean"
        for name in active_cols
            scores = [get(all_results[name], level, 0.0) for level in OBS_LEVELS]
            row *= @sprintf(" & \\textbf{%.1f\\%%}", mean(scores) * 100)
        end
        println(f, row * " \\\\")

        println(f, "\\bottomrule")
        println(f, "\\end{tabular}")
        println(f, "\\end{table}")
    end
    println("\nLaTeX table saved to: $(tex_file)")
end

println("\n" * "="^70)
println("Plots saved to: $(OUTDIR)/")
println("="^70)
