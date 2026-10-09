# Offline map background, drawn under the map tiles so the map still works
# without internet: Natural Earth 1:50m country polygons (public domain,
# naturalearthdata.com), shipped in data/naturalearth/ with properties stripped
# and coordinates rounded to 0.001°.

const BASEMAP = Ref{Any}(nothing)

_ring(coords) = Point2f[merc(c[1], c[2]) for c in coords]

"""
    load_basemap() -> Vector{Polygon}

Country outlines as Makie polygons in Web Mercator map coordinates. Parsed once and cached.
"""
function load_basemap()
    isnothing(BASEMAP[]) || return BASEMAP[]
    path = _resolve_bundled_path(joinpath("naturalearth", "ne_50m_admin_0_countries.geojson"))
    gj = JSON3.read(read(path, String))
    polys = map(Iterators.flatten(
        f.geometry.type == "Polygon" ? (f.geometry.coordinates,) : f.geometry.coordinates
        for f in gj.features)) do rings
        Makie.GeometryBasics.Polygon(_ring(first(rings)), Vector{Point2f}[_ring(r) for r in rings[2:end]])
    end
    BASEMAP[] = polys
    return polys
end
