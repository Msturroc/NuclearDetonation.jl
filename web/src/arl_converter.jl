# ARL → ERA5-compatible NetCDF converter
# Reads ARL binary files and writes temporary NetCDF files that the existing
# ERA5Format reader can ingest directly.

using NCDatasets
import Dates

# --- ARL directory scanning ---

struct ARLFileEntry
    path::String
    date::Dates.Date
end

"""
    scan_arl_directory(path::String) -> Vector{ARLFileEntry}

Find and sort .ARL files. Accepts a single `.ARL` file path or a directory.
- Single file: returns just that one entry
- Directory: scans for all `.ARL` files
"""
function scan_arl_directory(path::String)
    if isfile(path) && uppercase(path)[end-3:end] == ".ARL"
        # Single file mode
        arl = ARLReader.read_arl(path)
        dt = ARLReader.get_date(arl)
        return [ARLFileEntry(path, dt)]
    elseif isdir(path)
        entries = ARLFileEntry[]
        for fname in readdir(path)
            length(fname) > 4 || continue
            uppercase(fname)[end-3:end] == ".ARL" || continue
            fpath = joinpath(path, fname)
            isfile(fpath) || continue
            arl = ARLReader.read_arl(fpath)
            dt = ARLReader.get_date(arl)
            push!(entries, ARLFileEntry(fpath, dt))
        end
        sort!(entries, by = e -> e.date)
        return entries
    else
        error("Path is not an .ARL file or directory: $path\n" *
              "Provide either a single .ARL file (e.g. /path/to/ERA5_20150605.ARL) " *
              "or a directory containing .ARL files.")
    end
end

"""
    get_arl_bounds(path::String) -> NamedTuple

Get grid bounds, date range, and file info from ARL data.
Accepts a single .ARL file or a directory.

Optimized: only reads the first file header for grid info. Gets date range
from sorted filenames (ERA5_YYYYMMDD.ARL pattern) when possible, falling
back to reading first+last file headers.
"""
function get_arl_bounds(path::String)
    # Resolve path to directory + file list
    if isfile(path) && uppercase(path)[end-3:end] == ".ARL"
        dir = dirname(path)
        arl_files = [basename(path)]
    elseif isdir(path)
        dir = path
        arl_files = [f for f in readdir(path)
                     if length(f) > 4 && uppercase(f)[end-3:end] == ".ARL" && isfile(joinpath(path, f))]
    else
        error("Path is not an .ARL file or directory: $path\n" *
              "Provide either a single .ARL file or a directory containing .ARL files.")
    end

    if isempty(arl_files)
        present = readdir(dir)
        error("No .ARL files found in $dir\nFiles present: $(join(first(present, 10), ", "))" *
              (length(present) > 10 ? " ... ($(length(present)) total)" : ""))
    end

    sort!(arl_files)
    n_files = length(arl_files)

    # Read ONLY the first file header for grid/level info
    first_path = joinpath(dir, arl_files[1])
    arl = ARLReader.read_arl(first_path)
    plevs = ARLReader.get_pressure_levels(arl)

    # Get 3D variable names at first pressure level
    vars_3d = String[]
    for k in keys(arl.levels)
        if arl.levels[k]["level"] == plevs[1]
            vars_3d = [v[1] for v in arl.levels[k]["vars"]]
            break
        end
    end

    # Try to get date range from filenames (e.g. ERA5_20150605.ARL)
    date_min, date_max = _parse_date_range_from_filenames(arl_files)

    if isnothing(date_min)
        # Fallback: read first file header (already have it) + last file header
        date_min = ARLReader.get_date(arl)
        if n_files > 1
            last_arl = ARLReader.read_arl(joinpath(dir, arl_files[end]))
            date_max = ARLReader.get_date(last_arl)
        else
            date_max = date_min
        end
    end

    return (
        lat_min = minimum(arl.grid.lats),
        lat_max = maximum(arl.grid.lats),
        lon_min = minimum(arl.grid.lons),
        lon_max = maximum(arl.grid.lons),
        date_min = string(date_min),
        date_max = string(date_max),
        n_files = n_files,
        resolution = arl.resolution,
        pressure_levels = plevs,
        variables_3d = vars_3d,
        hours_per_file = arl.n_timesteps,
    )
end

"""
Parse date range from sorted ARL filenames matching patterns like:
  ERA5_YYYYMMDD.ARL, GDAS_YYYYMMDD.ARL, etc.
Returns (date_min, date_max) or (nothing, nothing) if parsing fails.
"""
function _parse_date_range_from_filenames(filenames::Vector{String})
    date_re = r"(\d{4})(\d{2})(\d{2})\.ARL$"i
    dates = Dates.Date[]
    for fname in filenames
        m = match(date_re, fname)
        isnothing(m) && return (nothing, nothing)
        try
            push!(dates, Dates.Date(parse(Int, m[1]), parse(Int, m[2]), parse(Int, m[3])))
        catch
            return (nothing, nothing)
        end
    end
    isempty(dates) && return (nothing, nothing)
    sort!(dates)
    return (dates[1], dates[end])
end

"""
    _find_arl_files_for_range(path, start_dt, duration_hours) -> Vector{ARLFileEntry}

Given a single .ARL file or directory, find all ARL files covering the simulation
time range. When given a single file, auto-discovers sibling files from the same
directory that fall within start_dt to start_dt + duration_hours + 24h buffer.
"""
function _find_arl_files_for_range(path::String, start_dt::Dates.DateTime, duration_hours::Int)
    if isfile(path) && uppercase(path)[end-3:end] == ".ARL"
        dir = dirname(path)
        start_date = Dates.Date(start_dt)
        end_date = Dates.Date(start_dt + Dates.Hour(duration_hours + 24))  # buffer

        # Scan sibling .ARL files and keep only those in the date range
        entries = ARLFileEntry[]
        for fname in readdir(dir)
            length(fname) > 4 || continue
            uppercase(fname)[end-3:end] == ".ARL" || continue
            fpath = joinpath(dir, fname)
            isfile(fpath) || continue

            # Try fast filename date parsing first
            fdate = _parse_date_from_filename(fname)
            if !isnothing(fdate)
                if fdate >= start_date && fdate <= end_date
                    push!(entries, ARLFileEntry(fpath, fdate))
                end
            else
                # Fallback: read header (slow, but only for non-standard names)
                arl = ARLReader.read_arl(fpath)
                fdate = ARLReader.get_date(arl)
                if fdate >= start_date && fdate <= end_date
                    push!(entries, ARLFileEntry(fpath, fdate))
                end
            end
        end
        sort!(entries, by = e -> e.date)
        return entries
    else
        # Directory: use existing scan (loads all files)
        return scan_arl_directory(path)
    end
end

"""Parse date from an ARL filename like ERA5_20150605.ARL. Returns Date or nothing."""
function _parse_date_from_filename(fname::String)
    m = match(r"(\d{4})(\d{2})(\d{2})\.ARL$"i, fname)
    isnothing(m) && return nothing
    try
        return Dates.Date(parse(Int, m[1]), parse(Int, m[2]), parse(Int, m[3]))
    catch
        return nothing
    end
end

# --- ARL to NetCDF conversion ---

"""
    convert_arl_region(dir_path, lat_center, lon_center, start_datetime, duration_hours; ...)

Convert a regional subset of ARL data to a single ERA5-compatible NetCDF file
containing exactly the hours needed for the simulation.

Returns (nc_files::Vector{String}, metadata::NamedTuple).
The returned nc_files contains a single file with all needed timesteps.
"""
function convert_arl_region(dir_path::String, lat_center::Float64, lon_center::Float64,
                            start_datetime::Dates.DateTime, duration_hours::Int;
                            radius_lat::Float64=5.0, radius_lon::Float64=10.0,
                            progress_callback=nothing)
    update!(pct, msg) = isnothing(progress_callback) || progress_callback(pct, msg)

    # When given a single file, auto-discover sibling files needed for the
    # simulation duration from the same directory.
    entries = _find_arl_files_for_range(dir_path, start_datetime, duration_hours)
    isempty(entries) && error("No .ARL files found for $dir_path")

    # Build list of (ARL_file_path, day_of_month, hour) for each needed hour
    end_datetime = start_datetime + Dates.Hour(duration_hours + 1)  # +1 for interpolation buffer
    needed_hours = Tuple{String, Int, Int, Dates.DateTime}[]  # (path, day, hour, datetime)

    for entry in entries
        arl = ARLReader.read_arl(entry.path)
        n_hours = min(arl.n_timesteps, 24)
        for h in 0:(n_hours - 1)
            dt = Dates.DateTime(entry.date) + Dates.Hour(h)
            if dt >= start_datetime && dt <= end_datetime
                push!(needed_hours, (entry.path, Dates.day(entry.date), h, dt))
            end
        end
    end

    isempty(needed_hours) && error("No ARL data covering time range $(start_datetime) to $(end_datetime). " *
        "Found $(length(entries)) file(s): $(join([basename(e.path) for e in entries], ", "))")

    update!(5, "Reading ARL grid structure...")

    # Read first relevant file for grid/level info
    arl_ref = ARLReader.read_arl(needed_hours[1][1])
    plevs = ARLReader.get_pressure_levels(arl_ref)  # descending: 1000, 975, ..., 50
    nk = length(plevs)

    # Determine wind variable names
    u_var = ARLReader.has_variable(arl_ref, plevs[1], "UWND") ? "UWND" :
            ARLReader.has_variable(arl_ref, plevs[1], "UGRD") ? "UGRD" :
            error("No wind variable (UWND/UGRD) found in ARL data")
    v_var = ARLReader.has_variable(arl_ref, plevs[1], "VWND") ? "VWND" :
            ARLReader.has_variable(arl_ref, plevs[1], "VGRD") ? "VGRD" :
            error("No wind variable (VWND/VGRD) found in ARL data")

    # Determine subset indices (ARL grid: lats south→north, lons -180→+180)
    lat_lo = lat_center - radius_lat
    lat_hi = lat_center + radius_lat
    lon_lo = lon_center - radius_lon
    lon_hi = lon_center + radius_lon

    all_lats = arl_ref.grid.lats
    all_lons = arl_ref.grid.lons

    j_start = max(1, searchsortedfirst(all_lats, lat_lo))
    j_end = min(length(all_lats), searchsortedlast(all_lats, lat_hi))
    i_start = max(1, searchsortedfirst(all_lons, lon_lo))
    i_end = min(length(all_lons), searchsortedlast(all_lons, lon_hi))

    j_start > j_end && error("No latitude points in range [$lat_lo, $lat_hi]")
    i_start > i_end && error("No longitude points in range [$lon_lo, $lon_hi]")

    sub_lats = all_lats[j_start:j_end]
    sub_lons = all_lons[i_start:i_end]
    nx_sub = length(sub_lons)
    ny_sub = length(sub_lats)

    # Pressure levels for NetCDF: ascending from TOA to surface (ERA5 convention)
    plevs_ascending = reverse(plevs)  # 50, ..., 975, 1000
    n_total_hours = length(needed_hours)

    update!(10, "Converting $(n_total_hours) hours ($(nx_sub)x$(ny_sub) grid, $(nk) levels)...")

    # Create single NetCDF file with all needed timesteps
    tmpdir = mktempdir(prefix="arl_nc_")
    date_str = Dates.format(Dates.Date(start_datetime), "yyyymmdd")
    nc_path = joinpath(tmpdir, "era5_$(date_str)_snap.nc")

    # Cache: avoid re-opening the same ARL file for every hour
    arl_cache = Dict{String, Any}()

    NCDataset(nc_path, "c") do ds
        # Dimensions
        defDim(ds, "longitude", nx_sub)
        defDim(ds, "latitude", ny_sub)
        defDim(ds, "hybrid", nk)
        defDim(ds, "time", n_total_hours)

        # Coordinate variables
        v_lon = defVar(ds, "longitude", Float64, ("longitude",))
        v_lat = defVar(ds, "latitude", Float64, ("latitude",))
        v_hyb = defVar(ds, "hybrid", Int32, ("hybrid",))
        v_time = defVar(ds, "time", Float64, ("time",),
            attrib = Dict("units" => "hours since 1900-01-01 00:00:00.0",
                          "calendar" => "standard"))

        v_lon[:] = sub_lons
        v_lat[:] = sub_lats
        v_hyb[:] = collect(Int32, 1:nk)

        # Time values
        ref_date = Dates.DateTime(1900, 1, 1)
        for (tidx, (_, _, _, dt)) in enumerate(needed_hours)
            v_time[tidx] = Float64(Dates.value(dt - ref_date)) / 3600000.0
        end

        # Hybrid coefficients (pure pressure levels: ap = pressure, b = 0)
        v_ap = defVar(ds, "ap", Float64, ("hybrid",))
        v_b = defVar(ds, "b", Float64, ("hybrid",))
        v_p0 = defVar(ds, "p0", Float64, ())
        v_ap[:] = plevs_ascending .* 100.0  # hPa → Pa
        v_b[:] = zeros(nk)
        v_p0[] = 100000.0

        # 3D met variables
        v_u = defVar(ds, "x_wind_ml", Float32, ("longitude", "latitude", "hybrid", "time"),
            attrib = Dict("units" => "m/s"))
        v_v = defVar(ds, "y_wind_ml", Float32, ("longitude", "latitude", "hybrid", "time"),
            attrib = Dict("units" => "m/s"))
        v_t = defVar(ds, "air_temperature_ml", Float32, ("longitude", "latitude", "hybrid", "time"),
            attrib = Dict("units" => "K"))
        v_sp = defVar(ds, "surface_air_pressure", Float32, ("longitude", "latitude", "time"),
            attrib = Dict("units" => "Pa"))

        # Fill each timestep
        for (tidx, (arl_path, day, hour, _)) in enumerate(needed_hours)
            pct = 10 + round(Int, 80 * (tidx - 1) / n_total_hours)
            if tidx % 6 == 1
                update!(pct, "Reading hour $(tidx)/$(n_total_hours)...")
            end

            # Get or create ARL reader (cache per file)
            arl = get!(arl_cache, arl_path) do
                ARLReader.read_arl(arl_path)
            end

            # Surface pressure
            if ARLReader.has_variable(arl, 0, "PRSS")
                ps_full = ARLReader.load_field(arl, day, hour, 0, "PRSS")
                v_sp[:, :, tidx] = Float32.(ps_full[i_start:i_end, j_start:j_end]) .* 100.0f0
            else
                v_sp[:, :, tidx] = fill(Float32(101325.0), nx_sub, ny_sub)
            end

            # 3D fields at each pressure level (ascending: TOA first)
            for (k_nc, plev) in enumerate(plevs_ascending)
                u_data = ARLReader.load_field(arl, day, hour, plev, u_var)
                v_data = ARLReader.load_field(arl, day, hour, plev, v_var)
                t_data = ARLReader.load_field(arl, day, hour, plev, "TEMP")

                v_u[:, :, k_nc, tidx] = Float32.(u_data[i_start:i_end, j_start:j_end])
                v_v[:, :, k_nc, tidx] = Float32.(v_data[i_start:i_end, j_start:j_end])
                v_t[:, :, k_nc, tidx] = Float32.(t_data[i_start:i_end, j_start:j_end])
            end
        end
    end

    update!(95, "ARL conversion complete")

    nc_files = [nc_path]
    metadata = (
        nx = nx_sub, ny = ny_sub, nk = nk,
        lat_range = sub_lats, lon_range = sub_lons,
        pressure_levels = plevs,
        n_files = 1,
        tmpdir = tmpdir,
    )

    return nc_files, metadata
end
