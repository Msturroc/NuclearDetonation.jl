# ARL binary format reader for NOAA HYSPLIT meteorological data
# Adapted from /media/marc/EPA_work/.../ARLreader.jl
# Supports ERA5, GFS, GDAS ARL files

module ARLReader

using Dates

export GridInfo, RecordInfo, ARLFile, read_arl, load_field, get_n_timesteps

struct GridInfo
    lats::Vector{Float64}
    lons::Vector{Float64}
end

struct RecordInfo
    y::Int; m::Int; d::Int; h::Int; fc::Int
    lvl::Float64; grid::String; name::String
    exp::Int; prec::Float64; initval::Float64
end

struct ARLFile
    fname::String
    indexinfo::RecordInfo
    headerinfo::Dict{String,Any}
    levels::Dict{Int,Dict{String,Any}}
    grid::GridInfo
    resolution::Float64
    recs_per_timestep::Int
    n_timesteps::Int
end

# --- Internal helpers ---

function split_format(fmtstring, s)
    s_str = String(s)
    result = []
    pos = 1
    for fmt_orig in split(fmtstring, ",")
        fmt = strip(fmt_orig)
        isempty(fmt) && continue
        code = fmt[1]
        len = tryparse(Int, fmt[2:end])
        isnothing(len) && continue
        pos + len - 1 > length(s_str) && break
        value = s_str[pos:pos+len-1]
        pos += len
        if code == 's'
            push!(result, strip(value))
        elseif code == 'i'
            v = strip(value)
            push!(result, (isempty(v) || !all(c -> isdigit(c) || c == '-', v)) ? 0 : parse(Int, v))
        elseif code == 'f'
            v = strip(value)
            push!(result, (isempty(v) || !all(c -> isdigit(c) || c in ('-', '.', 'E', '+'), v)) ? 0.0 : parse(Float64, v))
        end
    end
    result
end

function make_record_info(l)
    while length(l) < 11; push!(l, 0); end
    RecordInfo(l[1], l[2], l[3], l[4], l[5], Float64(l[6]),
              string(l[7]), string(l[8]), l[9], Float64(l[10]), Float64(l[11]))
end

function index_length(headerinfo)
    rec_len = (headerinfo["Nx"] * headerinfo["Ny"]) + 50
    idx_len = rec_len
    if haskey(headerinfo, "headerlength") && headerinfo["headerlength"] > (headerinfo["Nx"] * headerinfo["Ny"])
        extra = headerinfo["headerlength"] - 108
        idx_len += ceil(Int, extra / rec_len) * rec_len
    end
    Int64(idx_len)
end

function unpack_data(binarray, nx, ny, initval, exp_val, prec)
    scale = Float32(2.0^(7.0 - exp_val))
    data = zeros(Float32, nx, ny)
    idx = 0
    vold = Float32(initval)
    for j in 1:ny
        row_first = nothing
        for i in 1:nx
            v = (Float32(Int(binarray[idx+1]) - 127) / scale) + vold
            data[i, j] = v
            isnothing(row_first) && (row_first = v)
            idx += 1
            vold = v
        end
        !isnothing(row_first) && (vold = row_first)
    end
    data[abs.(data) .< prec] .= 0.0f0
    data
end

# --- Public API ---

"""
    read_arl(fname::String) -> ARLFile

Parse ARL file header and return metadata including grid, levels, and timing.
"""
function read_arl(fname::String)
    open(fname, "r") do f
        content = read(f, 5000)

        # Index record (first 50 bytes)
        idx_info = make_record_info(split_format("i2,i2,i2,i2,i2,i2,s2,s4,i4,f14,f14", content[1:50]))

        # Header record (bytes 51-158)
        hdr = split_format("s4,i3,i2," * repeat("f7,", 12) * "i3,i3,i3,i2,i4", content[51:158])
        while length(hdr) < 20; push!(hdr, 0); end

        headerinfo = Dict{String,Any}(
            "source" => hdr[1], "fcth" => hdr[2], "minDatatime" => hdr[3],
            "griddef" => hdr[4:min(15, length(hdr))],
            "Nx" => length(hdr) > 15 ? hdr[16] : 360,
            "Ny" => length(hdr) > 16 ? hdr[17] : 181,
            "Nz" => length(hdr) > 17 ? hdr[18] : 1,
            "Coordzflag" => length(hdr) > 18 ? hdr[19] : 0,
            "headerlength" => length(hdr) > 19 ? hdr[20] : 0,
        )

        # Determine resolution
        nx, ny = headerinfo["Nx"], headerinfo["Ny"]
        resolution = if nx == 720 && ny == 361; 0.5
        elseif nx == 360 && ny == 181; 1.0
        elseif nx == 1440 && ny == 721; 0.25
        else; 360.0 / nx
        end

        # Parse level/variable structure
        levels = Dict{Int,Dict{String,Any}}()
        cur = 159
        for ih in 0:(headerinfo["Nz"]-1)
            height_lvl = tryparse(Float64, strip(String(content[cur:cur+5])))
            isnothing(height_lvl) && (height_lvl = 0.0)
            nvars = tryparse(Int, strip(String(content[cur+6:cur+7])))
            isnothing(nvars) && (nvars = 0)
            cur += 8
            levels[ih] = Dict{String,Any}("level" => height_lvl, "vars" => Tuple{String,String}[])
            for _ in 1:nvars
                varname = strip(String(content[cur:cur+3]))
                checksum = strip(String(content[cur+4:cur+6]))
                cur += 8
                push!(levels[ih]["vars"], (varname, checksum))
            end
        end

        # Build lat/lon grid from header definition
        # griddef: [center_lat, center_lon, dx, dy, -, -, -,
        #           sync_x, sync_y, sync_lat, sync_lon, -]
        gd = headerinfo["griddef"]
        dx_deg = length(gd) >= 3 ? Float64(gd[3]) : 0.0
        dy_deg = length(gd) >= 4 ? Float64(gd[4]) : 0.0
        sync_x = length(gd) >= 8 ? Float64(gd[8]) : 1.0
        sync_y = length(gd) >= 9 ? Float64(gd[9]) : 1.0
        sync_lat = length(gd) >= 10 ? Float64(gd[10]) : -90.0
        sync_lon = length(gd) >= 11 ? Float64(gd[11]) : 0.0

        # If dx/dy are zero or invalid, fall back to global grid
        if dx_deg <= 0.0 || dy_deg <= 0.0
            dx_deg = 360.0 / nx
            dy_deg = 180.0 / (ny - 1)
            sync_lat = -90.0
            sync_lon = -180.0
            sync_x = 1.0
            sync_y = 1.0
        end

        lats = [sync_lat + (j - sync_y) * dy_deg for j in 1:ny]
        lons_raw = [sync_lon + (i - sync_x) * dx_deg for i in 1:nx]
        lons = [l > 180.0 ? l - 360.0 : l for l in lons_raw]
        grid = GridInfo(lats, lons)

        # Records per timestep
        vars_total = sum(length(v["vars"]) for v in values(levels))
        recs_per_ts = vars_total + 1  # +1 for index record

        # Compute number of timesteps from file size
        fsize = filesize(fname)
        rec_len = (nx * ny) + 50
        idx_len = index_length(headerinfo)
        data_bytes = fsize - idx_len
        n_ts = max(1, Int(floor(data_bytes / (recs_per_ts * rec_len))))

        return ARLFile(fname, idx_info, headerinfo, levels, grid,
                       resolution, recs_per_ts, n_ts)
    end
end

"""
    load_field(arl::ARLFile, day::Int, hour::Int, level::Number, variable::String) -> Matrix{Float32}

Read a 2D field from the ARL file for the given day, hour, level, and variable name.
"""
function load_field(arl::ARLFile, day::Int, hour::Int, level::Number, variable::String)
    level_f = Float64(level)
    nx, ny = arl.headerinfo["Nx"], arl.headerinfo["Ny"]
    rec_len = (nx * ny) + 50
    idx_len = index_length(arl.headerinfo)

    # Find level index and variable position
    total_vars_before = 0
    var_position = -1
    for k in sort(collect(keys(arl.levels)))
        ldata = arl.levels[k]
        if ldata["level"] == level_f
            vars = [v[1] for v in ldata["vars"]]
            vi = findfirst(isequal(variable), vars)
            isnothing(vi) && error("Variable '$variable' not found at level $level. Available: $vars")
            var_position = vi - 1
            break
        else
            total_vars_before += length(ldata["vars"])
        end
    end
    var_position < 0 && error("Level $level not found in ARL file")

    # Time offset
    time_index = (day - arl.indexinfo.d) * 24 + hour - arl.indexinfo.h
    time_index < 0 && (time_index = 0)

    # Byte position
    offset = idx_len + time_index * arl.recs_per_timestep * rec_len +
             (total_vars_before + var_position) * rec_len

    # Read and unpack
    open(arl.fname, "r") do f
        seek(f, offset)
        hdr_bytes = read(f, 50)
        recinfo = make_record_info(split_format("i2,i2,i2,i2,i2,i2,s2,s4,i4,f14,f14", hdr_bytes))
        data_bytes = read(f, rec_len - 50)
        unpack_data(convert(Vector{UInt8}, data_bytes), nx, ny,
                    recinfo.initval, recinfo.exp, recinfo.prec)
    end
end

"""
    has_variable(arl::ARLFile, level::Number, variable::String) -> Bool

Check if a variable exists at the given level.
"""
function has_variable(arl::ARLFile, level::Number, variable::String)
    level_f = Float64(level)
    for k in keys(arl.levels)
        if arl.levels[k]["level"] == level_f
            vars = [v[1] for v in arl.levels[k]["vars"]]
            return variable in vars
        end
    end
    false
end

"""
    get_pressure_levels(arl::ARLFile) -> Vector{Float64}

Return sorted pressure levels (descending: surface first), excluding surface level 0.
"""
function get_pressure_levels(arl::ARLFile)
    plevs = Float64[]
    for k in keys(arl.levels)
        lvl = arl.levels[k]["level"]
        lvl > 0 && push!(plevs, lvl)
    end
    sort!(plevs, rev=true)  # 1000, 975, ... , 50
end

"""
    get_date(arl::ARLFile) -> Date

Return the date of the first record in the ARL file.
"""
function get_date(arl::ARLFile)
    y = arl.indexinfo.y
    # Handle 2-digit year
    year = y < 100 ? (y > 50 ? 1900 + y : 2000 + y) : y
    Date(year, arl.indexinfo.m, arl.indexinfo.d)
end

end # module ARLReader
