# Observed fallout / tracer data for overlaying on simulation results

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
    load_observations(dataset) -> NamedTuple or nothing

Observed contours for the active dataset, as `(kind, title, ...)`:

- `kind = :grid` — a gridded field to contour (`lons`, `lats`, `grid`), used for ETEX
  time-integrated concentrations
- `kind = :polygons` — digitised contour polygons (`polygons` as vectors of lon/lat
  points, one value each), used for the Nancy dose-rate survey

Both carry `levels`, `colors` and `labels`. Returns `nothing` for datasets without
observations.
"""
function load_observations(dataset::String)
    dataset == "etex"  && return _load_etex_observations()
    dataset == "nancy" && return _load_nancy_observations()
    return nothing
end

function _load_etex_observations()
    meas_file = _resolve_bundled_path(joinpath("etex", "meas-t1.txt"))
    # Parse station data: compute TIC per station
    stations = Dict{Int, @NamedTuple{lat::Float64, lon::Float64, tic::Float64}}()
    for line in readlines(meas_file)[3:end]
        parts = split(strip(line))
        length(parts) >= 9 || continue
        lat = parse(Float64, parts[6])
        lon = parse(Float64, parts[7])
        conc = parse(Float64, parts[8])
        stn = parse(Int, parts[9])
        dur_min = parse(Int, parts[5])
        dur_hours = dur_min / 100  # format HHMM
        conc >= 0.0 || continue
        if haskey(stations, stn)
            s = stations[stn]
            stations[stn] = (lat=lat, lon=lon, tic=s.tic + conc * dur_hours)
        else
            stations[stn] = (lat=lat, lon=lon, tic=conc * dur_hours)
        end
    end

    # Grid TIC onto a coarse lat/lon grid
    grid_res = 1.5  # degrees
    lon_range = range(-10.0, 30.0, step=grid_res)
    lat_range = range(40.0, 62.0, step=grid_res)
    nx, ny = length(lon_range), length(lat_range)
    tic_grid = zeros(nx, ny)
    counts = zeros(Int, nx, ny)
    for (_, s) in stations
        s.tic > 0 || continue
        i = round(Int, (s.lon - first(lon_range)) / grid_res) + 1
        j = round(Int, (s.lat - first(lat_range)) / grid_res) + 1
        if 1 <= i <= nx && 1 <= j <= ny
            tic_grid[i, j] += s.tic
            counts[i, j] += 1
        end
    end
    for k in eachindex(tic_grid)
        counts[k] > 0 && (tic_grid[k] /= counts[k])
    end

    # Smooth with simple 3×3 averaging for nicer contours
    smoothed = copy(tic_grid)
    for j in 2:ny-1, i in 2:nx-1
        smoothed[i,j] = (tic_grid[i-1,j-1] + tic_grid[i,j-1] + tic_grid[i+1,j-1] +
                          tic_grid[i-1,j]   + tic_grid[i,j]   + tic_grid[i+1,j] +
                          tic_grid[i-1,j+1] + tic_grid[i,j+1] + tic_grid[i+1,j+1]) / 9.0
    end

    # Upsample for smoother contours
    up_grid = _upsample_grid(smoothed, 4)
    up_lon = range(first(lon_range), last(lon_range), length=size(up_grid, 1))
    up_lat = range(first(lat_range), last(lat_range), length=size(up_grid, 2))

    return (kind = :grid, title = "Observed TIC",
            lons = collect(up_lon), lats = collect(up_lat), grid = up_grid,
            levels = [100.0, 500.0, 2000.0, 5000.0, 10000.0],
            colors = ["#3288bd", "#66c2a5", "#fee08b", "#f46d43", "#d53e4f"],
            labels = ["100 ng·h/m³", "500 ng·h/m³", "2000 ng·h/m³", "5000 ng·h/m³", "10000 ng·h/m³"])
end

function _load_nancy_observations()
    obs = Transport.load_nancy_observations()
    palette = Dict(0.4 => "#3288bd", 1.0 => "#66c2a5", 4.0 => "#abdda4",
                   10.0 => "#fee08b", 40.0 => "#f46d43", 100.0 => "#d53e4f")
    levels = sort(unique(c.dose_rate_mR_hr for c in obs.dose_rate_contours))
    polygons = Tuple{Float64, Vector{Point2f}}[]
    for c in obs.dose_rate_contours, poly in c.polygons
        pts = [Point2f(pt[2], pt[1]) for pt in poly]  # (lat, lon) → (lon, lat)
        push!(pts, first(pts))
        push!(polygons, (c.dose_rate_mR_hr, pts))
    end
    return (kind = :polygons, title = "Observed dose rate",
            polygons = polygons, levels = levels,
            colors = [get(palette, l, "#999999") for l in levels],
            labels = ["$(l) mR/h" for l in levels])
end
