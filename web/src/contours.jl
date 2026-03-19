# GeoJSON contour generation from dose rate grids

using Contour
using JSON3

const CONTOUR_LEVELS = [0.004, 0.01, 0.04, 0.1, 0.4, 1.0]
const CONTOUR_COLORS = ["#3366FF", "#00CCCC", "#33AA33", "#CCCC00", "#FF8800", "#CC0000"]
const CONTOUR_LABELS = ["0.004 mSv/h", "0.01 mSv/h", "0.04 mSv/h", "0.1 mSv/h", "0.4 mSv/h", "1.0 mSv/h"]

# NPP deposition contours (Chernobyl zoning thresholds, kBq/m²)
const NPP_LEVELS = [1.0, 10.0, 37.0, 185.0, 555.0, 1480.0]
const NPP_COLORS = ["#3366FF", "#00CCCC", "#33AA33", "#CCCC00", "#FF8800", "#CC0000"]
const NPP_LABELS = ["1 kBq/m²", "10 kBq/m²", "37 kBq/m²", "185 kBq/m²", "555 kBq/m²", "1480 kBq/m²"]

"""Chaikin corner-cutting: smooth a polyline by iteratively cutting corners."""
function _chaikin_smooth(coords::Vector{Vector{Float64}}, iterations::Int=3)
    pts = coords
    for _ in 1:iterations
        length(pts) < 3 && return pts
        new_pts = Vector{Float64}[]
        push!(new_pts, pts[1])  # keep first point
        for i in 1:length(pts)-1
            q = 0.75 .* pts[i] .+ 0.25 .* pts[i+1]
            r = 0.25 .* pts[i] .+ 0.75 .* pts[i+1]
            push!(new_pts, q, r)
        end
        push!(new_pts, pts[end])  # keep last point
        pts = new_pts
    end
    return pts
end

"""Bilinear upsample a 2D grid by factor n."""
function _upsample_grid(grid::Matrix{Float64}, n::Int)
    nx, ny = size(grid)
    out_nx = (nx - 1) * n + 1
    out_ny = (ny - 1) * n + 1
    out = zeros(out_nx, out_ny)
    for j in 1:out_ny, i in 1:out_nx
        fx = 1.0 + (i - 1) * (nx - 1) / (out_nx - 1)
        fy = 1.0 + (j - 1) * (ny - 1) / (out_ny - 1)
        x0 = clamp(floor(Int, fx), 1, nx - 1)
        y0 = clamp(floor(Int, fy), 1, ny - 1)
        dx = fx - x0; dy = fy - y0
        out[i, j] = (1-dx)*(1-dy)*grid[x0,y0] + dx*(1-dy)*grid[x0+1,y0] +
                     (1-dx)*dy*grid[x0,y0+1] + dx*dy*grid[x0+1,y0+1]
    end
    return out
end

"""
    dose_to_geojson(result::SimulationResult) -> String

Convert a dose rate grid into a GeoJSON FeatureCollection with contour lines
at standard fallout survey levels.
"""
function dose_to_geojson(result::SimulationResult)
    lon_vec = collect(result.lon_grid)
    lat_vec = collect(result.lat_grid)

    # Pick levels/labels based on units
    if result.units == "kBq/m²"
        levels = NPP_LEVELS
        colors = NPP_COLORS
        labels = NPP_LABELS
    else
        levels = CONTOUR_LEVELS
        colors = CONTOUR_COLORS
        labels = CONTOUR_LABELS
    end

    # Upsample grid 4× for smoother contours
    up_factor = 4
    up_grid = _upsample_grid(result.dose_grid, up_factor)
    up_lon = range(first(lon_vec), last(lon_vec), length=size(up_grid, 1))
    up_lat = range(first(lat_vec), last(lat_vec), length=size(up_grid, 2))

    features = Dict{String,Any}[]

    cl = Contour.contours(collect(up_lon), collect(up_lat), up_grid, levels)

    for (level_idx, level_obj) in enumerate(Contour.levels(cl))
        level_val = Contour.level(level_obj)
        ci = findfirst(l -> abs(l - level_val) / max(l, 1e-10) < 0.01, levels)
        isnothing(ci) && continue

        for line in Contour.lines(level_obj)
            xs, ys = Contour.coordinates(line)
            coords = [[x, y] for (x, y) in zip(xs, ys)]
            length(coords) < 2 && continue

            # Smooth the contour line with Chaikin corner-cutting
            coords = _chaikin_smooth(coords, 2)
            coords = [[round(c[1], digits=5), round(c[2], digits=5)] for c in coords]

            feature = Dict{String,Any}(
                "type" => "Feature",
                "geometry" => Dict{String,Any}(
                    "type" => "LineString",
                    "coordinates" => coords,
                ),
                "properties" => Dict{String,Any}(
                    "level" => level_val,
                    "label" => labels[ci],
                    "color" => colors[ci],
                ),
            )
            push!(features, feature)
        end
    end

    geojson = Dict{String,Any}(
        "type" => "FeatureCollection",
        "features" => features,
    )
    return JSON3.write(geojson)
end

"""
    export_deposition_csv(result::SimulationResult, domain) -> String

Export deposition events as a CSV string with lat/lon coordinates.
"""
function export_deposition_csv(result::SimulationResult, domain)
    io = IOBuffer()
    println(io, "latitude,longitude,deposition_Bq,time_s")
    for evt in result.deposition_log
        lat, lon = Transport.grid_to_latlon(domain, evt.x, evt.y)
        lon > 180.0 && (lon -= 360.0)
        println(io, "$(round(lat, digits=5)),$(round(lon, digits=5)),$(evt.mass),$(evt.time)")
    end
    return String(take!(io))
end
