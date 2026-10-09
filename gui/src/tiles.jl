# Map tiles from OpenStreetMap's standard tile servers. Their usage policy
# (operations.osmfoundation.org/policies/tiles) requires requests to identify the
# application: generic library User-Agents get an "Access blocked" tile. So wrap
# the provider and send our own User-Agent with each tile request.

const TILE_USER_AGENT = "NuclearDetonation.jl-GUI/2.0 (+https://github.com/Msturroc/NuclearDetonation.jl)"
const TILE_ATTRIBUTION = "Map © OpenStreetMap contributors"

struct OSMTiles <: TileProviders.AbstractProvider end

TileProviders.options(::OSMTiles) = nothing
TileProviders.min_zoom(::OSMTiles) = 0
TileProviders.max_zoom(::OSMTiles) = 19
TileProviders.geturl(::OSMTiles, x::Integer, y::Integer, z::Integer) =
    "https://tile.openstreetmap.org/$z/$x/$y.png"

# Tyler gives up on a tile after 3 s and doesn't retry it, which leaves holes on
# slow connections
Tyler.get_downloader(::OSMTiles) = Tyler.ByteDownloader(15)

# Downloads.jl adds its own User-Agent header unless one is passed with the request
function Tyler.download_tile_data(dl::Tyler.ByteDownloader, ::OSMTiles, url)
    Downloads.download(url, dl.io; downloader = dl.downloader, timeout = dl.timeout,
                       headers = ["User-Agent" => TILE_USER_AGENT])
    resize!(dl.bytes, dl.io.ptr - 1)
    copyto!(dl.bytes, 1, dl.io.data, 1, dl.io.ptr - 1)
    seekstart(dl.io)
    return dl.bytes
end
