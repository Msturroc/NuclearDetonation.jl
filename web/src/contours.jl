# GeoJSON contour generation from dose rate grids

using Contour
using JSON3

const CONTOUR_LEVELS = [0.4, 1.0, 4.0, 10.0, 40.0, 100.0]
const CONTOUR_COLORS = ["#3366FF", "#00CCCC", "#33AA33", "#CCCC00", "#FF8800", "#CC0000"]
const CONTOUR_LABELS = ["0.4 mR/h", "1.0 mR/h", "4.0 mR/h", "10.0 mR/h", "40.0 mR/h", "100.0 mR/h"]

# NPP deposition contours (Chernobyl zoning thresholds, kBq/m²)
const NPP_LEVELS = [1.0, 10.0, 37.0, 185.0, 555.0, 1480.0]
const NPP_COLORS = ["#3366FF", "#00CCCC", "#33AA33", "#CCCC00", "#FF8800", "#CC0000"]
const NPP_LABELS = ["1 kBq/m²", "10 kBq/m²", "37 kBq/m²", "185 kBq/m²", "555 kBq/m²", "1480 kBq/m²"]

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

    features = Dict{String,Any}[]

    # Contour.jl expects contours(x, y, z, levels) where z is (nx, ny)
    # Our dose_grid is already (nx_lon, ny_lat) which matches (x, y)
    cl = Contour.contours(lon_vec, lat_vec, result.dose_grid, levels)

    for (level_idx, level_obj) in enumerate(Contour.levels(cl))
        level_val = Contour.level(level_obj)
        # Find matching index for colour/label
        ci = findfirst(l -> abs(l - level_val) / max(l, 1e-10) < 0.01, levels)
        isnothing(ci) && continue

        for line in Contour.lines(level_obj)
            xs, ys = Contour.coordinates(line)
            # GeoJSON coordinates are [lon, lat]
            coords = [[round(x, digits=5), round(y, digits=5)] for (x, y) in zip(xs, ys)]
            length(coords) < 2 && continue

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
