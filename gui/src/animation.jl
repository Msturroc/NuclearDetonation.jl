# Plume animation from simulation snapshots: per-level concentration frames
# for playback on the map and GIF/MP4 export

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
    heights_m::Vector{Float64}        # surface → TOA in metres MSL (matching nz dim)
    release_mode::String              # "bomb" or "npp"
    units::String                     # e.g. "mSv/h" or "Bq"
end

const ANIMATION_STATE = Ref{Any}(nothing)

# Ireland must always be visible in the animation viewport (EPA Ireland requirement)
const IRELAND_VIEWPORT = (lat_min=50.0, lat_max=56.0, lon_min=-12.0, lon_max=-4.5)

# NPP sites shown on the map (site key matches the prediction model filenames)
const NPP_PLANTS = [
    (name="Hinkley Point C", site="hinkley",     lat=51.2086, lon=-3.1304),
    (name="Wylfa",           site="wylfa",       lat=53.4167, lon=-4.4822),
    (name="Paluel",          site="paluel",      lat=49.8584, lon=0.6354),
    (name="Flamanville",     site="flamanville", lat=49.5381, lon=-1.8802),
    (name="Sizewell B",      site="sizewell",    lat=52.2145, lon=1.6206),
    (name="Heysham",         site="heysham",     lat=54.0285, lon=-2.9161),
]

"""
    store_animation_data!(snapshots, domain, pressure_levels_ascending;
                          heights_m_ascending=Float64[],
                          release_mode, units)

Extract concentration fields from simulation snapshots and store for animation.

`heights_m_ascending`: heights in metres MSL, in surface→TOA order (matching the
nz dimension of the concentration array — `domain.hlevel` after `update_domain_vertical!`).
When non-empty, takes precedence over `pressure_levels_ascending` for level labelling.
"""
function store_animation_data!(snapshots, domain, pressure_levels_ascending;
                               heights_m_ascending::AbstractVector=Float64[],
                               release_mode::String="bomb", units::String="mSv/h",
                               release_offset_s::Real=0.0)
    # Frames start at the release; times are relative to it
    snapshots = [s for s in snapshots if s.time >= release_offset_s - 1e-6]
    isempty(snapshots) && return

    concentrations = [Float32.(snap.concentration) for snap in snapshots]
    times_s = [Float64(snap.time) - release_offset_s for snap in snapshots]

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

    hmeters = Float64.(heights_m_ascending)

    ANIMATION_STATE[] = AnimationState(
        concentrations, times_s,
        domain.lat_min, domain.lat_max, lon_min, lon_max,
        nx, ny, nz, plevs, hmeters, release_mode, units,
    )
end

# --- Level label formatting ---

"""Format a height (metres MSL) as a short label."""
function _height_label(h_m::Real)
    h_m <= 0 && return "Surface"
    h_m < 1000 && return "$(round(Int, h_m)) m"
    # Consistent 1-decimal km for everything from 1 km upward, so adjacent levels
    # don't mix "10.0 km" with "10 km" formatting.
    return "$(round(h_m / 1000, digits=1)) km"
end

"""Build a level label for index k (1=surface, nz=TOA per accumulate_concentration)."""
function _level_label(anim::AnimationState, k::Int)
    k == 0 && return "Column Total"
    1 <= k <= anim.nz || return "Level $k"
    if !isempty(anim.heights_m) && length(anim.heights_m) >= k
        h = anim.heights_m[k]
        return "$(_height_label(h))" *
               (k == 1 ? " (surface)" : k == anim.nz ? " (top)" : "")
    elseif maximum(anim.pressure_levels) > anim.nz
        hpa = anim.pressure_levels[k]
        if hpa >= 950
            alt_m = round(Int, 44330 * (1.0 - (hpa / 1013.25)^0.19))
            return "Surface (~$(alt_m)m)"
        else
            alt_km = round(44.33 * (1.0 - (hpa / 1013.25)^0.19), digits=1)
            return "$(round(Int, hpa)) hPa (~$(alt_km)km)"
        end
    else
        return k == anim.nz ? "Level $k (top)" :
               k == 1 ? "Level $k (surface)" : "Level $k"
    end
end

# --- Colormap (blue → cyan → green → yellow → red, fading in with value) ---

function _plume_rgba(t::Float64)
    r, g, b = if t < 0.25
        (0.0, t / 0.25, 1.0)
    elseif t < 0.5
        (0.0, 1.0, 1.0 - (t - 0.25) / 0.25)
    elseif t < 0.75
        ((t - 0.5) / 0.25, 1.0, 0.0)
    else
        (1.0, 1.0 - (t - 0.75) / 0.25, 0.0)
    end
    return RGBAf(r, g, b, clamp(0.15 + t * 0.7, 0.0, 0.85))
end

const PLUME_COLORMAP = [_plume_rgba(t) for t in range(0, 1, length=256)]

# Frames span five decades below the peak, as log10 values
const PLUME_DECADES = 5.0

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

"""
    get_available_levels() -> Vector{Tuple{String,Int}}

Animation levels as `(label, index)` pairs: column total (index 0) first, then
every height level that ever holds particles, highest first.
"""
function get_available_levels()
    anim = ANIMATION_STATE[]
    isnothing(anim) && return Tuple{String,Int}[]

    has_data = falses(anim.nz)
    for conc in anim.concentrations, k in 1:anim.nz
        has_data[k] && continue
        maximum(view(conc, :, :, k, 1)) > 0 && (has_data[k] = true)
    end

    levels = [("Column total (all heights)", 0)]
    for k in anim.nz:-1:1
        has_data[k] && push!(levels, (_level_label(anim, k), k))
    end
    return levels
end

"""
    animation_frames(level) -> NamedTuple or nothing

Smoothed log10 concentration frames for one level (0 = column total), ready for
`image!`. Cells more than `PLUME_DECADES` below the peak sit under the colour
range and render transparent. Also returns the plume viewport (padded, and
widened to include Ireland for European domains) used for exports.
"""
function animation_frames(level::Int)
    anim = ANIMATION_STATE[]
    isnothing(anim) && return nothing
    level == 0 || 1 <= level <= anim.nz || error("Level $level out of range")

    nx, ny = anim.nx, anim.ny
    all_max = 0.0f0
    i_min, i_max = nx, 1
    j_min, j_max = ny, 1
    slices = Matrix{Float32}[]
    for conc in anim.concentrations
        slice = _gaussian_smooth(_get_slice(conc, level), 1.0)
        push!(slices, slice)
        all_max = max(all_max, maximum(slice))
        for j in 1:ny, i in 1:nx
            if slice[i, j] > 0
                i_min = min(i_min, i); i_max = max(i_max, i)
                j_min = min(j_min, j); j_max = max(j_max, j)
            end
        end
    end
    all_max <= 0 && return nothing

    log_max = log10(Float64(all_max))
    log_min = log_max - PLUME_DECADES
    below = Float32(log_min - 1)
    frames = [map(v -> v > 0 ? Float32(log10(v)) : below, s) for s in slices]

    vlat_min, vlat_max, vlon_min, vlon_max =
        _compute_viewport_bounds(anim, i_min, i_max, j_min, j_max; pad_frac=0.15)

    return (frames = frames,
            times_h = anim.times_s ./ 3600.0,
            lon_min = anim.lon_min, lon_max = anim.lon_max,
            lat_min = anim.lat_min, lat_max = anim.lat_max,
            log_min = log_min, log_max = log_max,
            max_value = Float64(all_max),
            units = anim.units,
            label = _level_label(anim, level),
            viewport = (lon_min = vlon_min, lon_max = vlon_max,
                        lat_min = vlat_min, lat_max = vlat_max))
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
