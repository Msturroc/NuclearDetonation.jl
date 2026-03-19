# Animation generation from simulation snapshots
# Renders concentration heatmap frames for client-side playback and GIF/MP4 export

using Base64

# --- Storage for snapshot data ---

struct AnimationState
    concentrations::Vector{Array{Float32,4}}  # (nx, ny, nz, ncomp) per timestep
    times_s::Vector{Float64}
    lat_min::Float64
    lat_max::Float64
    lon_min::Float64
    lon_max::Float64
    nx::Int
    ny::Int
    nz::Int
    pressure_levels::Vector{Float64}  # ascending: TOA → surface (matching nz dim)
    release_mode::String              # "bomb" or "npp"
    units::String                     # e.g. "mSv/h" or "Bq"
end

const ANIMATION_STATE = Ref{Any}(nothing)

# Ireland must always be visible in the animation viewport (EPA Ireland requirement)
const IRELAND_VIEWPORT = (lat_min=50.0, lat_max=56.0, lon_min=-12.0, lon_max=-4.5)

# NPP locations for marker rendering on exported animations
const NPP_PLANTS = [
    (name="Hinkley Point C", lat=51.2086, lon=-3.1304),
    (name="Wylfa",           lat=53.4167, lon=-4.4822),
    (name="Paluel",          lat=49.8584, lon=0.6354),
    (name="Flamanville",     lat=49.5381, lon=-1.8802),
    (name="Sizewell B",      lat=52.2145, lon=1.6206),
    (name="Heysham",         lat=54.0285, lon=-2.9161),
]

"""
    store_animation_data!(snapshots, domain, pressure_levels; release_mode, units)

Extract concentration fields from simulation snapshots and store for animation.
"""
function store_animation_data!(snapshots, domain, pressure_levels_ascending;
                               release_mode::String="bomb", units::String="mSv/h")
    isempty(snapshots) && return

    concentrations = [Float32.(snap.concentration) for snap in snapshots]
    times_s = [Float64(snap.time) for snap in snapshots]

    # Convert domain lon from 0-360 to -180..180 for display
    lon_min = domain.lon_min > 180 ? domain.lon_min - 360 : domain.lon_min
    lon_max = domain.lon_max > 180 ? domain.lon_max - 360 : domain.lon_max

    nx, ny, nz = size(concentrations[1], 1), size(concentrations[1], 2), size(concentrations[1], 3)

    # If no pressure levels provided, generate generic indices
    plevs = if isempty(pressure_levels_ascending)
        Float64.(1:nz)
    else
        Float64.(pressure_levels_ascending)
    end

    ANIMATION_STATE[] = AnimationState(
        concentrations, times_s,
        domain.lat_min, domain.lat_max, lon_min, lon_max,
        nx, ny, nz, plevs, release_mode, units,
    )
end

# --- Colormap (blue → cyan → green → yellow → red) ---

function conc_to_rgba(value::Float32, log_min::Float64, log_range::Float64)
    if value <= 0 || isnan(value)
        return 0x00, 0x00, 0x00, 0x00
    end
    t = clamp((log10(Float64(value)) - log_min) / log_range, 0.0, 1.0)

    r, g, b = if t < 0.25
        (0.0, t / 0.25, 1.0)
    elseif t < 0.5
        (0.0, 1.0, 1.0 - (t - 0.25) / 0.25)
    elseif t < 0.75
        ((t - 0.5) / 0.25, 1.0, 0.0)
    else
        (1.0, 1.0 - (t - 0.75) / 0.25, 0.0)
    end

    alpha = clamp(0.15 + t * 0.7, 0.0, 0.85)
    return round(UInt8, r * 255), round(UInt8, g * 255), round(UInt8, b * 255), round(UInt8, alpha * 255)
end

# --- Frame generation ---

"""Compute geographic viewport bounds from plume bounding box (grid indices),
with padding and Ireland viewport union when the domain is near Europe."""
function _compute_viewport_bounds(anim, i_min, i_max, j_min, j_max; pad_frac=0.15)
    nx, ny = anim.nx, anim.ny
    # Convert plume bbox from grid indices to geographic coords
    plume_lon_min = anim.lon_min + (i_min - 1) / nx * (anim.lon_max - anim.lon_min)
    plume_lon_max = anim.lon_min + i_max / nx * (anim.lon_max - anim.lon_min)
    plume_lat_min = anim.lat_min + (j_min - 1) / ny * (anim.lat_max - anim.lat_min)
    plume_lat_max = anim.lat_min + j_max / ny * (anim.lat_max - anim.lat_min)
    # Pad
    lon_pad = (plume_lon_max - plume_lon_min) * pad_frac
    lat_pad = (plume_lat_max - plume_lat_min) * pad_frac
    v_lon_min = plume_lon_min - lon_pad
    v_lon_max = plume_lon_max + lon_pad
    v_lat_min = plume_lat_min - lat_pad
    v_lat_max = plume_lat_max + lat_pad
    # Union with Ireland viewport if domain is in/near Europe (within 10° of Ireland)
    if anim.lon_max > IRELAND_VIEWPORT.lon_min - 10 && anim.lon_min < IRELAND_VIEWPORT.lon_max + 10 &&
       anim.lat_max > IRELAND_VIEWPORT.lat_min - 10 && anim.lat_min < IRELAND_VIEWPORT.lat_max + 10
        v_lon_min = min(v_lon_min, IRELAND_VIEWPORT.lon_min)
        v_lon_max = max(v_lon_max, IRELAND_VIEWPORT.lon_max)
        v_lat_min = min(v_lat_min, IRELAND_VIEWPORT.lat_min)
        v_lat_max = max(v_lat_max, IRELAND_VIEWPORT.lat_max)
    end
    return v_lat_min, v_lat_max, v_lon_min, v_lon_max
end

"""Extract a 2D slice for a given level, or column-integrated (level=0)."""
function _get_slice(conc::Array{Float32,4}, level::Int)
    if level == 0
        # Column-integrated: sum across all height levels
        return dropdims(sum(conc[:, :, :, 1], dims=3), dims=3)
    else
        return conc[:, :, level, 1]
    end
end

function get_animation_frames(level::Int)
    anim = ANIMATION_STATE[]
    isnothing(anim) && error("No animation data available. Run a simulation first.")
    level == 0 || 1 <= level <= anim.nz || error("Level $level out of range")

    nx, ny = anim.nx, anim.ny

    # Find global max and tight bounding box around non-zero concentration
    all_max = 0.0f0
    i_min, i_max = nx, 1
    j_min, j_max = ny, 1
    for conc in anim.concentrations
        slice = _get_slice(conc, level)
        m = maximum(slice)
        all_max = max(all_max, m)
        for j in 1:ny, i in 1:nx
            if slice[i, j] > 0
                i_min = min(i_min, i); i_max = max(i_max, i)
                j_min = min(j_min, j); j_max = max(j_max, j)
            end
        end
    end
    all_max <= 0 && return Dict(
        "n_frames" => 0,
        "message" => "No concentration data",
    )

    # Compute geographic viewport (includes Ireland guarantee + padding)
    crop_lat_min, crop_lat_max, crop_lon_min, crop_lon_max =
        _compute_viewport_bounds(anim, i_min, i_max, j_min, j_max; pad_frac=0.15)

    # Output dimensions match the viewport aspect ratio (longest side = 512px for browser)
    lon_span = crop_lon_max - crop_lon_min
    lat_span = crop_lat_max - crop_lat_min
    if lon_span >= lat_span
        crop_nx = 512
        crop_ny = max(256, round(Int, 512 * lat_span / lon_span))
    else
        crop_ny = 512
        crop_nx = max(256, round(Int, 512 * lon_span / lat_span))
    end

    log_max = log10(Float64(all_max))
    log_min = log_max - 5.0
    log_range = 5.0

    level_label = level == 0 ? "Column Total" :
        (anim.pressure_levels[level] > anim.nz ?
            "$(round(Int, anim.pressure_levels[level])) hPa" : "Level $level")

    encoded_frames = String[]
    for conc in anim.concentrations
        slice = _get_slice(conc, level)
        bytes = Vector{UInt8}(undef, crop_nx * crop_ny * 4)
        idx = 0
        for py in 1:crop_ny  # top row = lat_max
            lat = crop_lat_max - (py - 0.5) / crop_ny * lat_span
            for px in 1:crop_nx
                lon = crop_lon_min + (px - 0.5) / crop_nx * lon_span
                # Convert geo coords to fractional grid indices (1-based, cell centers at integer values)
                fx = (lon - anim.lon_min) / (anim.lon_max - anim.lon_min) * nx + 0.5
                fy = (lat - anim.lat_min) / (anim.lat_max - anim.lat_min) * ny + 0.5
                # Outside grid domain → transparent
                if fx < 0.5 || fx > nx + 0.5 || fy < 0.5 || fy > ny + 0.5
                    r, g, b, a = 0x00, 0x00, 0x00, 0x00
                else
                    fx = clamp(fx, 1.0, Float64(nx))
                    fy = clamp(fy, 1.0, Float64(ny))
                    conc_val = Float32(_bilinear_sample(slice, fx, fy))
                    r, g, b, a = conc_to_rgba(conc_val, log_min, log_range)
                end
                bytes[idx + 1] = r
                bytes[idx + 2] = g
                bytes[idx + 3] = b
                bytes[idx + 4] = a
                idx += 4
            end
        end
        push!(encoded_frames, base64encode(bytes))
    end

    return Dict(
        "n_frames" => length(encoded_frames),
        "width" => crop_nx,
        "height" => crop_ny,
        "times_hours" => [t / 3600.0 for t in anim.times_s],
        "max_value" => Float64(all_max),
        "units" => anim.units,
        "level_label" => level_label,
        "bounds" => Dict(
            "lat_min" => crop_lat_min,
            "lat_max" => crop_lat_max,
            "lon_min" => crop_lon_min,
            "lon_max" => crop_lon_max,
        ),
        "frames" => encoded_frames,
    )
end

"""
    get_available_levels() -> Dict

Return available height levels for animation.
"""
function get_available_levels()
    anim = ANIMATION_STATE[]
    isnothing(anim) && error("No animation data available")

    # Pre-scan: find which levels have any non-zero concentration
    has_data = falses(anim.nz)
    for conc in anim.concentrations
        for k in 1:anim.nz
            has_data[k] && continue
            if maximum(conc[:, :, k, 1]) > 0
                has_data[k] = true
            end
        end
    end

    levels = Dict{String,Any}[]
    # Column Total as first (default) option
    push!(levels, Dict("index" => 0, "hpa" => 0.0, "label" => "Column Total (all heights)"))

    has_real_pressures = maximum(anim.pressure_levels) > anim.nz
    for k in 1:anim.nz
        has_data[k] || continue  # skip empty levels
        hpa = anim.pressure_levels[k]
        label = if !has_real_pressures
            k == anim.nz ? "Level $k (lowest)" : k == 1 ? "Level $k (highest)" : "Level $k"
        elseif hpa >= 950
            alt_m = round(Int, 44330 * (1.0 - (hpa / 1013.25)^0.19))
            "Surface (~$(alt_m)m)"
        else
            alt_km = round(44.33 * (1.0 - (hpa / 1013.25)^0.19), digits=1)
            "$(round(Int, hpa)) hPa (~$(alt_km)km)"
        end
        push!(levels, Dict("index" => k, "hpa" => hpa, "label" => label))
    end
    return Dict(
        "n_levels" => length(levels),
        "n_frames" => length(anim.concentrations),
        "levels" => levels,
    )
end

# --- Map tile download for animation background ---

using HTTP

"""Fetch an OSM tile as raw RGB pixels via ffmpeg PNG decode."""
function _fetch_tile_rgb(z::Int, x::Int, y::Int)::Union{Vector{UInt8}, Nothing}
    url = "https://tile.openstreetmap.org/$z/$x/$y.png"
    try
        resp = HTTP.get(url; headers=["User-Agent" => "NuclearDetonation.jl/1.0"],
                        connect_timeout=5, readtimeout=10)
        pngpath = tempname() * ".png"
        rawpath = tempname() * ".rgb"
        write(pngpath, resp.body)
        run(pipeline(`ffmpeg -y -loglevel error -i $pngpath -f rawvideo -pix_fmt rgb24 $rawpath`))
        rgb = read(rawpath)
        rm(pngpath, force=true)
        rm(rawpath, force=true)
        return rgb
    catch
        return nothing
    end
end

"""Convert lat/lon to OSM tile coordinates at zoom level z."""
function _latlon_to_tile(lat::Float64, lon::Float64, z::Int)
    n = 2^z
    x = floor(Int, (lon + 180.0) / 360.0 * n)
    lat_rad = deg2rad(lat)
    y = floor(Int, (1.0 - log(tan(lat_rad) + 1.0 / cos(lat_rad)) / π) / 2.0 * n)
    return clamp(x, 0, n - 1), clamp(y, 0, n - 1)
end

"""Build a background map image from OSM tiles for the given bounds.
Returns (rgb_pixels::Vector{UInt8}, img_width, img_height, crop_x, crop_y, crop_w, crop_h)."""
function _build_map_background(lat_min, lat_max, lon_min, lon_max, target_width)
    # Choose zoom so tiles match output pixel density
    lon_span = lon_max - lon_min
    px_per_degree = target_width / lon_span
    z = clamp(round(Int, log2(px_per_degree * 360 / 256)), 3, 10)

    x_min, y_min = _latlon_to_tile(lat_max, lon_min, z)  # NW corner
    x_max, y_max = _latlon_to_tile(lat_min, lon_max, z)  # SE corner

    n_tiles_x = x_max - x_min + 1
    n_tiles_y = y_max - y_min + 1
    tile_sz = 256

    # Fetch tiles
    full_w = n_tiles_x * tile_sz
    full_h = n_tiles_y * tile_sz
    bg = fill(UInt8(220), full_w * full_h * 3)  # light grey fallback

    for ty in y_min:y_max, tx in x_min:x_max
        rgb = _fetch_tile_rgb(z, tx, ty)
        isnothing(rgb) && continue
        length(rgb) == tile_sz * tile_sz * 3 || continue
        ox = (tx - x_min) * tile_sz
        oy = (ty - y_min) * tile_sz
        for py in 0:(tile_sz-1), px in 0:(tile_sz-1)
            src = (py * tile_sz + px) * 3
            dst = ((oy + py) * full_w + (ox + px)) * 3
            bg[dst+1] = rgb[src+1]
            bg[dst+2] = rgb[src+2]
            bg[dst+3] = rgb[src+3]
        end
    end

    # Calculate pixel bounds for our lat/lon region within the tile mosaic
    n = 2^z
    px_left = ((lon_min + 180.0) / 360.0 * n - x_min) * tile_sz
    px_right = ((lon_max + 180.0) / 360.0 * n - x_min) * tile_sz
    lat_min_rad = deg2rad(lat_min)
    lat_max_rad = deg2rad(lat_max)
    px_top = ((1.0 - log(tan(lat_max_rad) + 1.0/cos(lat_max_rad)) / π) / 2.0 * n - y_min) * tile_sz
    px_bot = ((1.0 - log(tan(lat_min_rad) + 1.0/cos(lat_min_rad)) / π) / 2.0 * n - y_min) * tile_sz

    crop_x = max(0, round(Int, px_left))
    crop_y = max(0, round(Int, px_top))
    crop_w = min(full_w - crop_x, round(Int, px_right - px_left))
    crop_h = min(full_h - crop_y, round(Int, px_bot - px_top))

    return bg, full_w, full_h, crop_x, crop_y, crop_w, crop_h
end

# --- Colorbar rendering ---

const COLORBAR_WIDTH = 30  # pixels (base, scaled up with resolution)
const LABEL_MARGIN = 120   # space for labels (base, scaled up with resolution)

"""Render a colorbar strip as RGB pixels (colorbar_w × height × 3)."""
function _render_colorbar_rgb(height::Int, log_min::Float64, log_range::Float64; cb_width::Int=COLORBAR_WIDTH)
    w = cb_width
    pixels = Vector{UInt8}(undef, w * height * 3)
    for row in 1:height
        # Map row to colormap: top=high, bottom=low
        t_val = 1.0 - (row - 1) / max(height - 1, 1)
        r, g, b, _ = conc_to_rgba(Float32(10.0^(log_min + t_val * log_range)), log_min, log_range)
        for col in 1:w
            idx = ((row - 1) * w + (col - 1)) * 3
            pixels[idx+1] = r
            pixels[idx+2] = g
            pixels[idx+3] = b
        end
    end
    return pixels, w
end

# --- Frame compositing ---

"""Bilinear sample from a 2D grid at fractional coordinates (1-based)."""
function _bilinear_sample(grid, fx::Float64, fy::Float64)
    nx, ny = size(grid)
    x0 = clamp(floor(Int, fx), 1, nx - 1)
    y0 = clamp(floor(Int, fy), 1, ny - 1)
    x1 = min(x0 + 1, nx)
    y1 = min(y0 + 1, ny)
    dx = clamp(fx - x0, 0.0, 1.0)
    dy = clamp(fy - y0, 0.0, 1.0)
    return (1 - dx) * (1 - dy) * grid[x0, y0] +
           dx       * (1 - dy) * grid[x1, y0] +
           (1 - dx) * dy       * grid[x0, y1] +
           dx       * dy       * grid[x1, y1]
end

"""Bilinear sample RGB from a flat pixel buffer at fractional pixel coords (0-based)."""
function _bilinear_sample_rgb(bg::Vector{UInt8}, w::Int, h::Int, fx::Float64, fy::Float64)
    x0 = clamp(floor(Int, fx), 0, w - 2)
    y0 = clamp(floor(Int, fy), 0, h - 2)
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)
    dx = clamp(fx - x0, 0.0, 1.0)
    dy = clamp(fy - y0, 0.0, 1.0)
    w00 = (1 - dx) * (1 - dy)
    w10 = dx * (1 - dy)
    w01 = (1 - dx) * dy
    w11 = dx * dy
    r = g = b = 0.0
    for (ix, iy, wt) in ((x0,y0,w00),(x1,y0,w10),(x0,y1,w01),(x1,y1,w11))
        idx = (iy * w + ix) * 3
        r += wt * Float64(bg[idx+1])
        g += wt * Float64(bg[idx+2])
        b += wt * Float64(bg[idx+3])
    end
    return round(UInt8, r), round(UInt8, g), round(UInt8, b)
end

"""Apply separable Gaussian blur to smooth grid-scale artifacts."""
function _gaussian_smooth(field::Matrix, sigma::Float64)
    sigma <= 0 && return field
    nx, ny = size(field)
    r = ceil(Int, 3 * sigma)
    k1d = [exp(-i^2 / (2 * sigma^2)) for i in -r:r]
    k1d ./= sum(k1d)
    tmp = zeros(Float64, nx, ny)
    out = zeros(Float64, nx, ny)
    for y in 1:ny, x in 1:nx
        for di in -r:r
            tmp[x, y] += k1d[di + r + 1] * field[clamp(x + di, 1, nx), y]
        end
    end
    for y in 1:ny, x in 1:nx
        for dj in -r:r
            out[x, y] += k1d[dj + r + 1] * tmp[x, clamp(y + dj, 1, ny)]
        end
    end
    return Float32.(out)
end

"""Generate animation frames as raw RGB data, composited over map background with colorbar.
Returns (rawpath, total_width, total_height, n_frames, log_min, log_max_val, ..., npp_positions)."""
function _render_frames_to_file(level::Int; target_px::Int=1200)
    anim = ANIMATION_STATE[]
    isnothing(anim) && error("No animation data available")
    level == 0 || 1 <= level <= anim.nz || error("Level $level out of range")

    nx, ny = anim.nx, anim.ny

    # Find global max and tight bounding box around non-zero concentration
    all_max = 0.0f0
    i_min, i_max = nx, 1
    j_min, j_max = ny, 1
    for conc in anim.concentrations
        slice = _get_slice(conc, level)
        all_max = max(all_max, maximum(slice))
        for j in 1:ny, i in 1:nx
            if slice[i, j] > 0
                i_min = min(i_min, i); i_max = max(i_max, i)
                j_min = min(j_min, j); j_max = max(j_max, j)
            end
        end
    end
    all_max <= 0 && error("No concentration data")

    # Compute geographic viewport (includes Ireland guarantee + padding)
    view_lat_min, view_lat_max, view_lon_min, view_lon_max =
        _compute_viewport_bounds(anim, i_min, i_max, j_min, j_max; pad_frac=0.15)

    crop_lon_min, crop_lon_max = view_lon_min, view_lon_max
    crop_lat_min, crop_lat_max = view_lat_min, view_lat_max

    # Drive map dimensions from target_px and viewport aspect ratio
    target_px = max(target_px, 800)
    lon_span = crop_lon_max - crop_lon_min
    lat_span = crop_lat_max - crop_lat_min
    if lon_span >= lat_span
        map_w = target_px
        map_h = max(1, round(Int, map_w * lat_span / lon_span))
    else
        map_h = target_px
        map_w = max(1, round(Int, map_h * lon_span / lat_span))
    end

    log_max = log10(Float64(all_max))
    log_min = log_max - 5.0
    log_range = 5.0

    # Fetch map background tiles for the cropped region
    bg_rgb, bg_full_w, bg_full_h, cx, cy, cw, ch =
        _build_map_background(crop_lat_min, crop_lat_max, crop_lon_min, crop_lon_max, map_w)

    # Scale factors from map crop to our output size
    sx = Float64(map_w) / max(cw, 1)
    sy = Float64(map_h) / max(ch, 1)

    # Pre-render the base map at output resolution using bilinear interpolation
    base_map = Vector{UInt8}(undef, map_w * map_h * 3)
    for row in 1:map_h, col in 1:map_w
        src_fx = Float64(cx) + (col - 1) / sx
        src_fy = Float64(cy) + (row - 1) / sy
        r, g, b = _bilinear_sample_rgb(bg_rgb, bg_full_w, bg_full_h, src_fx, src_fy)
        dst_idx = ((row - 1) * map_w + (col - 1)) * 3
        base_map[dst_idx+1] = r
        base_map[dst_idx+2] = g
        base_map[dst_idx+3] = b
    end

    # Compute NPP marker pixel positions for ffmpeg text overlay (only in NPP mode)
    npp_positions = Tuple{String, Int, Int}[]
    if anim.release_mode == "npp"
        for plant in NPP_PLANTS
            px = round(Int, (plant.lon - crop_lon_min) / (crop_lon_max - crop_lon_min) * map_w)
            py = round(Int, (1.0 - (plant.lat - crop_lat_min) / (crop_lat_max - crop_lat_min)) * map_h)
            if 1 <= px <= map_w && 1 <= py <= map_h
                push!(npp_positions, (plant.name, px, py))
            end
        end
    end

    # Render colorbar — scale width with map height
    cb_w_scaled = max(COLORBAR_WIDTH, round(Int, map_h * 0.03))
    label_margin_scaled = max(LABEL_MARGIN, round(Int, map_h * 0.12))
    cb_pixels, cb_w = _render_colorbar_rgb(map_h, log_min, log_range; cb_width=cb_w_scaled)

    # Total frame: map + gap + colorbar + label margin
    gap = max(4, round(Int, map_h * 0.005))
    total_w = map_w + gap + cb_w + label_margin_scaled
    total_h = map_h
    # Ensure even dimensions
    total_w += total_w % 2
    total_h += total_h % 2

    rawpath = tempname() * ".rgb"
    open(rawpath, "w") do f
        for (fidx, conc) in enumerate(anim.concentrations)
            slice = _gaussian_smooth(_get_slice(conc, level), 1.0)

            for row in 1:total_h
                for col in 1:total_w
                    if col <= map_w && row <= map_h
                        # Map region: convert pixel to geo coords, then to grid coords
                        lon = crop_lon_min + (col - 0.5) / map_w * lon_span
                        lat = crop_lat_max - (row - 0.5) / map_h * lat_span  # flip Y
                        fx = (lon - anim.lon_min) / (anim.lon_max - anim.lon_min) * nx + 0.5
                        fy = (lat - anim.lat_min) / (anim.lat_max - anim.lat_min) * ny + 0.5
                        conc_val = if fx < 0.5 || fx > nx + 0.5 || fy < 0.5 || fy > ny + 0.5
                            0.0f0  # outside grid domain
                        else
                            Float32(_bilinear_sample(slice, clamp(fx, 1.0, Float64(nx)), clamp(fy, 1.0, Float64(ny))))
                        end
                        r, g, b, a = conc_to_rgba(conc_val, log_min, log_range)
                        af = Float64(a) / 255.0
                        base_idx = ((row - 1) * map_w + (col - 1)) * 3
                        br = base_map[base_idx+1]
                        bg_c = base_map[base_idx+2]
                        bb = base_map[base_idx+3]
                        write(f, round(UInt8, r * af + br * (1 - af)))
                        write(f, round(UInt8, g * af + bg_c * (1 - af)))
                        write(f, round(UInt8, b * af + bb * (1 - af)))
                    elseif col > map_w + gap && col <= map_w + gap + cb_w && row <= map_h
                        # Colorbar region
                        cb_col = col - map_w - gap
                        cb_idx = ((row - 1) * cb_w + (cb_col - 1)) * 3
                        write(f, cb_pixels[cb_idx+1], cb_pixels[cb_idx+2], cb_pixels[cb_idx+3])
                    else
                        # White background (gap, label area)
                        write(f, 0xff, 0xff, 0xff)
                    end
                end
            end
        end
    end

    return rawpath, total_w, total_h, length(anim.concentrations), log_min, all_max, map_w, gap, cb_w, npp_positions
end

# --- GIF/MP4 generation ---

"""
    generate_gif(level::Int; fps::Int=2, target_px::Int=1200) -> Vector{UInt8}

Generate a GIF with map background, colorbar, and timestamp labels.
"""
function generate_gif(level::Int; fps::Int=2, target_px::Int=1200)
    anim = ANIMATION_STATE[]
    rawpath, w, h, n_frames, log_min, max_val, map_w, gap, cb_w, npp_positions = _render_frames_to_file(level; target_px)
    outpath = tempname() * ".gif"

    # Scale font sizes proportionally to output resolution
    fs_big = max(16, round(Int, h * 0.03))     # timestamp
    fs_med = max(12, round(Int, h * 0.022))    # level label
    fs_sm  = max(11, round(Int, h * 0.018))    # colorbar labels
    bw_big = max(2, round(Int, fs_big * 0.15))
    bw_med = max(1, round(Int, fs_med * 0.12))
    pad = max(8, round(Int, h * 0.01))

    level_str = level == 0 ? "Column Total" : begin
        hpa = anim.pressure_levels[level]
        hpa > anim.nz ? "$(round(Int, hpa)) hPa" : "Level $(round(Int, hpa))"
    end

    log_range = 5.0
    hi_label = _sci_label(10.0^(log_min + log_range))
    lo_label = _sci_label(10.0^log_min)
    unit_label = anim.units

    cb_x = map_w + gap + cb_w + 4

    # NPP marker overlays (☢ symbols at plant locations, only in NPP mode)
    npp_vf = ""
    if !isempty(npp_positions)
        fs_npp = max(16, round(Int, h * 0.025))
        bw_npp = max(2, round(Int, fs_npp * 0.15))
        for (_, px, py) in npp_positions
            npp_vf *= "drawtext=text='☢':fontsize=$fs_npp:fontcolor=gold@0.9:" *
                      "borderw=$bw_npp:bordercolor=black:x=$(px - fs_npp÷2):y=$(py - fs_npp÷2),"
        end
    end

    vf = "drawtext=text='H+%{expr_int_format\\:(n+1)\\:d\\:2}h':fontsize=$fs_big:fontcolor=white:" *
         "borderw=$bw_big:bordercolor=black:x=$pad:y=$pad," *
         "drawtext=text='$level_str':fontsize=$fs_med:fontcolor=white:" *
         "borderw=$bw_med:bordercolor=black:x=$pad:y=$(pad + fs_big + 4)," *
         "drawtext=text='$hi_label':fontsize=$fs_sm:fontcolor=black:x=$cb_x:y=$pad," *
         "drawtext=text='$unit_label':fontsize=$fs_sm:fontcolor=black:x=$cb_x:y=$(div(h, 2) - div(fs_sm, 2))," *
         "drawtext=text='$lo_label':fontsize=$fs_sm:fontcolor=black:x=$cb_x:y=$(h - pad - fs_sm)," *
         npp_vf *
         "split[s0][s1];[s0]palettegen=max_colors=256:stats_mode=full[p];[s1][p]paletteuse=dither=sierra2_4a"

    try
        run(pipeline(`ffmpeg -y -loglevel error
            -f rawvideo -pixel_format rgb24 -video_size $(w)x$(h)
            -framerate $fps -i $rawpath
            -vf $vf
            $outpath`))
        return read(outpath)
    finally
        rm(rawpath, force=true)
        rm(outpath, force=true)
    end
end

"""
    generate_mp4(level::Int; fps::Int=4, target_px::Int=1200) -> Vector{UInt8}

Generate an MP4 with map background, colorbar, and timestamp labels.
"""
function generate_mp4(level::Int; fps::Int=4, target_px::Int=1200)
    anim = ANIMATION_STATE[]
    rawpath, w, h, n_frames, log_min, max_val, map_w, gap, cb_w, npp_positions = _render_frames_to_file(level; target_px)
    outpath = tempname() * ".mp4"

    fs_big = max(16, round(Int, h * 0.03))
    fs_med = max(12, round(Int, h * 0.022))
    fs_sm  = max(11, round(Int, h * 0.018))
    bw_big = max(2, round(Int, fs_big * 0.15))
    bw_med = max(1, round(Int, fs_med * 0.12))
    pad = max(8, round(Int, h * 0.01))

    level_str = level == 0 ? "Column Total" : begin
        hpa = anim.pressure_levels[level]
        hpa > anim.nz ? "$(round(Int, hpa)) hPa" : "Level $(round(Int, hpa))"
    end

    log_range = 5.0
    hi_label = _sci_label(10.0^(log_min + log_range))
    lo_label = _sci_label(10.0^log_min)
    unit_label = anim.units

    cb_x = map_w + gap + cb_w + 4

    # NPP marker overlays (☢ symbols at plant locations, only in NPP mode)
    npp_vf = ""
    if !isempty(npp_positions)
        fs_npp = max(16, round(Int, h * 0.025))
        bw_npp = max(2, round(Int, fs_npp * 0.15))
        for (_, px, py) in npp_positions
            npp_vf *= ",drawtext=text='☢':fontsize=$fs_npp:fontcolor=gold@0.9:" *
                       "borderw=$bw_npp:bordercolor=black:x=$(px - fs_npp÷2):y=$(py - fs_npp÷2)"
        end
    end

    vf = "drawtext=text='H+%{expr_int_format\\:(n+1)\\:d\\:2}h':fontsize=$fs_big:fontcolor=white:" *
         "borderw=$bw_big:bordercolor=black:x=$pad:y=$pad," *
         "drawtext=text='$level_str':fontsize=$fs_med:fontcolor=white:" *
         "borderw=$bw_med:bordercolor=black:x=$pad:y=$(pad + fs_big + 4)," *
         "drawtext=text='$hi_label':fontsize=$fs_sm:fontcolor=black:x=$cb_x:y=$pad," *
         "drawtext=text='$unit_label':fontsize=$fs_sm:fontcolor=black:x=$cb_x:y=$(div(h, 2) - div(fs_sm, 2))," *
         "drawtext=text='$lo_label':fontsize=$fs_sm:fontcolor=black:x=$cb_x:y=$(h - pad - fs_sm)" *
         npp_vf

    try
        run(pipeline(`ffmpeg -y -loglevel error
            -f rawvideo -pixel_format rgb24 -video_size $(w)x$(h)
            -framerate $fps -i $rawpath
            -vf $vf
            -c:v libx264 -pix_fmt yuv420p -crf 20
            $outpath`))
        return read(outpath)
    finally
        rm(rawpath, force=true)
        rm(outpath, force=true)
    end
end

"""Format a value in scientific notation for labels."""
function _sci_label(val::Float64)
    if val >= 1.0
        return string(round(val, sigdigits=2))
    else
        e = floor(Int, log10(val))
        m = val / 10.0^e
        return "$(round(m, digits=1))e$(e)"
    end
end
