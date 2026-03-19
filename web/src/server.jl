# HTTP server for the web GUI

using HTTP
using JSON3
using Dates
using Base64

# --- Application state ---

mutable struct AppStatus
    running::Bool
    progress_pct::Int
    progress_msg::String
    error_msg::String
    geojson::String
    max_dose::Float64
    n_events::Int
    csv_data::String
    units::String
end

const APP = Ref(AppStatus(false, 0, "", "", "", 0.0, 0, "", "mSv/h"))

const WEB_DIR = dirname(@__DIR__)  # web/

function mime_type(path::String)
    endswith(path, ".html") && return "text/html"
    endswith(path, ".css")  && return "text/css"
    endswith(path, ".js")   && return "application/javascript"
    endswith(path, ".json") && return "application/json"
    endswith(path, ".png")  && return "image/png"
    endswith(path, ".svg")  && return "image/svg+xml"
    endswith(path, ".csv")  && return "text/csv"
    endswith(path, ".gif")  && return "image/gif"
    endswith(path, ".mp4")  && return "video/mp4"
    return "application/octet-stream"
end

function cors_headers()
    return ["Access-Control-Allow-Origin" => "*",
            "Access-Control-Allow-Methods" => "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers" => "Content-Type"]
end

# --- Route handler ---

function handle_request(req::HTTP.Request)
    method = req.method
    path = HTTP.URI(req.target).path

    # CORS preflight
    method == "OPTIONS" && return HTTP.Response(204, cors_headers())

    try
        # Static files
        if path == "/" || path == ""
            return serve_file("public/index.html")
        elseif startswith(path, "/public/")
            return serve_file(path[2:end])  # strip leading /
        end

        # API endpoints
        if path == "/api/status" && method == "GET"
            return api_status()
        elseif path == "/api/simulate" && method == "POST"
            return api_simulate(req)
        elseif path == "/api/results.csv" && method == "GET"
            return api_results_csv()
        elseif path == "/api/era5-bounds" && method == "GET"
            return api_era5_bounds()
        elseif path == "/api/load-dataset" && method == "POST"
            return api_load_dataset(req)
        elseif path == "/api/active-dataset" && method == "GET"
            return api_active_dataset()
        elseif path == "/api/upload-arl" && method == "POST"
            return api_upload_arl(req)
        elseif path == "/api/load-arl" && method == "POST"
            return api_load_arl(req)
        elseif path == "/api/arl-bounds" && method == "GET"
            return api_arl_bounds()
        elseif path == "/api/animation-levels" && method == "GET"
            return api_animation_levels()
        elseif path == "/api/animation-frames" && method == "POST"
            return api_animation_frames(req)
        elseif path == "/api/animation.gif" && method == "POST"
            return api_animation_export(req, "gif")
        elseif path == "/api/animation.mp4" && method == "POST"
            return api_animation_export(req, "mp4")
        elseif path == "/api/stitch-frames" && method == "POST"
            return api_stitch_frames(req)
        end

        return HTTP.Response(404, cors_headers(), "Not found: $path")
    catch e
        @error "Request error" path exception=(e, catch_backtrace())
        return HTTP.Response(500, cors_headers(), "Internal error: $(sprint(showerror, e))")
    end
end

function serve_file(relpath::String)
    fullpath = joinpath(WEB_DIR, relpath)
    if isfile(fullpath)
        content = read(fullpath)
        headers = vcat(cors_headers(), ["Content-Type" => mime_type(relpath)])
        return HTTP.Response(200, headers, content)
    else
        return HTTP.Response(404, cors_headers(), "File not found: $relpath")
    end
end

function api_status()
    s = APP[]
    body = JSON3.write(Dict(
        "running"      => s.running,
        "progress_pct" => s.progress_pct,
        "progress_msg" => s.progress_msg,
        "error_msg"    => s.error_msg,
        "geojson"      => s.geojson,
        "max_dose"     => s.max_dose,
        "n_events"     => s.n_events,
        "complete"     => !s.running && !isempty(s.geojson),
        "units"        => s.units,
    ))
    headers = vcat(cors_headers(), ["Content-Type" => "application/json"])
    return HTTP.Response(200, headers, body)
end

function api_simulate(req::HTTP.Request)
    s = APP[]
    if s.running
        body = JSON3.write(Dict("error" => "Simulation already running"))
        headers = vcat(cors_headers(), ["Content-Type" => "application/json"])
        return HTTP.Response(409, headers, body)
    end

    # Parse parameters
    params = JSON3.read(String(req.body))
    lat            = Float64(get(params, :lat, 37.0956))
    lon            = Float64(get(params, :lon, -116.1028))
    yield_kt       = Float64(get(params, :yield_kt, 24.0))
    start_date     = String(get(params, :start_date, "1953-03-24"))
    start_hour     = Int(get(params, :start_hour, 13))
    duration_hours = Int(get(params, :duration_hours, 12))
    n_particles    = Int(get(params, :n_particles, 5000))
    release_mode   = String(get(params, :release_mode, "bomb"))
    activity_tbq   = Float64(get(params, :activity_tbq, 1.0))
    stack_height_m = Float64(get(params, :stack_height_m, 100.0))
    isotope        = String(get(params, :isotope, "Cs-137"))
    weather_source = String(get(params, :weather_source, "era5"))
    arl_dir        = String(get(params, :arl_dir, ""))

    # Reset state
    APP[] = AppStatus(true, 0, "Starting...", "", "", 0.0, 0, "", "mSv/h")

    # Run simulation asynchronously
    Threads.@spawn begin
        try
            result = run_simulation_with_source(;
                weather_source, arl_dir,
                lat, lon, yield_kt, start_date, start_hour,
                duration_hours, n_particles,
                release_mode, activity_tbq, stack_height_m, isotope,
                progress_callback = (pct, msg) -> begin
                    APP[].progress_pct = pct
                    APP[].progress_msg = msg
                end,
            )

            geojson = dose_to_geojson(result)
            csv = export_deposition_csv(result, result.domain)

            APP[].geojson = geojson
            APP[].csv_data = csv
            APP[].max_dose = result.max_dose
            APP[].n_events = length(result.deposition_log)
            APP[].units = result.units
            APP[].progress_pct = 100
            APP[].progress_msg = "Complete"
            APP[].running = false
        catch e
            bt = catch_backtrace()
            # Full error with stacktrace
            err_full = sprint() do io
                showerror(io, e, bt)
            end
            APP[].error_msg = sprint(showerror, e)
            APP[].running = false
            # Write to log file for debugging
            logpath = joinpath(WEB_DIR, "error.log")
            open(logpath, "a") do f
                println(f, "\n", "="^60)
                println(f, Dates.now(), " — Simulation error:")
                println(f, err_full)
            end
            @error "Simulation failed — see web/error.log" exception=(e, bt)
        end
    end

    body = JSON3.write(Dict("status" => "started"))
    headers = vcat(cors_headers(), ["Content-Type" => "application/json"])
    return HTTP.Response(202, headers, body)
end

function api_results_csv()
    s = APP[]
    if isempty(s.csv_data)
        return HTTP.Response(404, cors_headers(), "No results available")
    end
    headers = vcat(cors_headers(), [
        "Content-Type" => "text/csv",
        "Content-Disposition" => "attachment; filename=\"deposition_results.csv\"",
    ])
    return HTTP.Response(200, headers, s.csv_data)
end

function api_era5_bounds()
    era5 = ERA5_STATE[]
    if isnothing(era5)
        return HTTP.Response(503, cors_headers(), "ERA5 data not loaded")
    end
    # Convert lon from 0-360 to -180-180 for display
    lons = era5.lon_range
    display_lons = [l > 180.0 ? l - 360.0 : l for l in lons]
    body = JSON3.write(Dict(
        "lat_min" => minimum(era5.lat_range),
        "lat_max" => maximum(era5.lat_range),
        "lon_min" => minimum(display_lons),
        "lon_max" => maximum(display_lons),
    ))
    headers = vcat(cors_headers(), ["Content-Type" => "application/json"])
    return HTTP.Response(200, headers, body)
end

# --- Dataset switching ---

function api_load_dataset(req::HTTP.Request)
    s = APP[]
    if s.running
        body = JSON3.write(Dict("error" => "Cannot switch dataset while simulation is running"))
        headers = vcat(cors_headers(), ["Content-Type" => "application/json"])
        return HTTP.Response(409, headers, body)
    end

    params = JSON3.read(String(req.body))
    dataset = String(get(params, :dataset, "nancy"))

    try
        preload_era5!(dataset=dataset,
            progress_callback = (pct, msg) -> @info "[$pct%] $msg")

        era5 = ERA5_STATE[]
        lons = era5.lon_range
        display_lons = [l > 180.0 ? l - 360.0 : l for l in lons]
        body = JSON3.write(Dict(
            "dataset" => dataset,
            "lat_min" => minimum(era5.lat_range),
            "lat_max" => maximum(era5.lat_range),
            "lon_min" => minimum(display_lons),
            "lon_max" => maximum(display_lons),
        ))
        headers = vcat(cors_headers(), ["Content-Type" => "application/json"])
        return HTTP.Response(200, headers, body)
    catch e
        body = JSON3.write(Dict("error" => sprint(showerror, e)))
        headers = vcat(cors_headers(), ["Content-Type" => "application/json"])
        return HTTP.Response(500, headers, body)
    end
end

function api_active_dataset()
    body = JSON3.write(Dict("dataset" => ACTIVE_DATASET[]))
    headers = vcat(cors_headers(), ["Content-Type" => "application/json"])
    return HTTP.Response(200, headers, body)
end

# --- ARL endpoints ---

# Mutable state for ARL metadata (set by load-arl, used by UI)
const ARL_METADATA = Ref{Any}(nothing)

function api_upload_arl(req::HTTP.Request)
    try
        # Extract boundary from Content-Type header
        content_type = HTTP.header(req, "Content-Type", "")
        if !contains(content_type, "multipart/form-data")
            return HTTP.Response(400, cors_headers(), "Expected multipart/form-data")
        end
        bm = match(r"boundary=(.+)", content_type)
        isnothing(bm) && return HTTP.Response(400, cors_headers(), "No boundary in Content-Type")
        boundary = String(bm.captures[1])

        upload_dir = mktempdir()
        n_files = 0
        raw = req.body
        delim = Vector{UInt8}("--" * boundary)

        # Split body on boundary markers and extract file parts
        parts = _split_multipart(raw, delim)
        for part_bytes in parts
            part_str = String(copy(part_bytes))
            # Find header/body separator (double CRLF)
            sep_idx = findfirst("\r\n\r\n", part_str)
            isnothing(sep_idx) && continue
            headers_str = part_str[1:sep_idx.start-1]
            body_start = sep_idx.stop + 1
            # Extract filename from Content-Disposition
            fm = match(r"filename=\"([^\"]+)\"", headers_str)
            isnothing(fm) && continue
            fname = basename(fm.captures[1])
            isempty(fname) && continue
            # Body is after \r\n\r\n, strip trailing \r\n before next boundary
            file_data = part_bytes[body_start:end]
            if length(file_data) >= 2 && file_data[end-1:end] == UInt8[0x0d, 0x0a]
                file_data = file_data[1:end-2]
            end
            write(joinpath(upload_dir, fname), file_data)
            n_files += 1
        end

        if n_files == 0
            rm(upload_dir, force=true, recursive=true)
            return HTTP.Response(400, cors_headers(), "No files uploaded")
        end
        body = JSON3.write(Dict("upload_dir" => upload_dir, "n_files" => n_files))
        hdrs = vcat(cors_headers(), ["Content-Type" => "application/json"])
        return HTTP.Response(200, hdrs, body)
    catch e
        body = JSON3.write(Dict("error" => sprint(showerror, e)))
        hdrs = vcat(cors_headers(), ["Content-Type" => "application/json"])
        return HTTP.Response(500, hdrs, body)
    end
end

"""Split multipart body on boundary delimiter, returning content between boundaries."""
function _split_multipart(data::Vector{UInt8}, delim::Vector{UInt8})
    parts = Vector{UInt8}[]
    dlen = length(delim)
    n = length(data)
    # Find all positions of delimiter
    positions = Int[]
    for i in 1:(n - dlen + 1)
        if data[i:i+dlen-1] == delim
            push!(positions, i)
        end
    end
    # Content is between consecutive delimiters
    for k in 1:(length(positions) - 1)
        # Skip delimiter + potential \r\n after it
        start = positions[k] + dlen
        if start <= n && data[start] == 0x0d; start += 1; end
        if start <= n && data[start] == 0x0a; start += 1; end
        stop = positions[k+1] - 1
        start <= stop || continue
        push!(parts, data[start:stop])
    end
    return parts
end

function api_load_arl(req::HTTP.Request)
    params = JSON3.read(String(req.body))
    dir_path = String(get(params, :path, ""))
    isempty(dir_path) && return HTTP.Response(400, cors_headers(), "Missing 'path' parameter")

    try
        bounds = load_arl_metadata!(dir_path)
        ARL_METADATA[] = merge(bounds, (dir_path = dir_path,))
        body = JSON3.write(Dict(
            "lat_min" => bounds.lat_min,
            "lat_max" => bounds.lat_max,
            "lon_min" => bounds.lon_min,
            "lon_max" => bounds.lon_max,
            "date_min" => bounds.date_min,
            "date_max" => bounds.date_max,
            "n_files" => bounds.n_files,
            "resolution" => bounds.resolution,
            "pressure_levels" => bounds.pressure_levels,
            "hours_per_file" => bounds.hours_per_file,
        ))
        headers = vcat(cors_headers(), ["Content-Type" => "application/json"])
        return HTTP.Response(200, headers, body)
    catch e
        body = JSON3.write(Dict("error" => sprint(showerror, e)))
        headers = vcat(cors_headers(), ["Content-Type" => "application/json"])
        return HTTP.Response(500, headers, body)
    end
end

function api_arl_bounds()
    meta = ARL_METADATA[]
    if isnothing(meta)
        return HTTP.Response(404, cors_headers(), "No ARL data loaded")
    end
    body = JSON3.write(Dict(
        "lat_min" => meta.lat_min,
        "lat_max" => meta.lat_max,
        "lon_min" => meta.lon_min,
        "lon_max" => meta.lon_max,
        "date_min" => meta.date_min,
        "date_max" => meta.date_max,
        "dir_path" => meta.dir_path,
    ))
    headers = vcat(cors_headers(), ["Content-Type" => "application/json"])
    return HTTP.Response(200, headers, body)
end

# --- Animation endpoints ---

function api_animation_levels()
    try
        data = get_available_levels()
        body = JSON3.write(data)
        headers = vcat(cors_headers(), ["Content-Type" => "application/json"])
        return HTTP.Response(200, headers, body)
    catch e
        body = JSON3.write(Dict("error" => sprint(showerror, e)))
        headers = vcat(cors_headers(), ["Content-Type" => "application/json"])
        return HTTP.Response(isnothing(ANIMATION_STATE[]) ? 404 : 500, headers, body)
    end
end

function api_animation_frames(req::HTTP.Request)
    try
        params = JSON3.read(String(req.body))
        level = Int(get(params, :level, 1))
        data = get_animation_frames(level)
        body = JSON3.write(data)
        headers = vcat(cors_headers(), ["Content-Type" => "application/json"])
        return HTTP.Response(200, headers, body)
    catch e
        body = JSON3.write(Dict("error" => sprint(showerror, e)))
        headers = vcat(cors_headers(), ["Content-Type" => "application/json"])
        return HTTP.Response(500, headers, body)
    end
end

function api_animation_export(req::HTTP.Request, format::String)
    try
        params = JSON3.read(String(req.body))
        level = Int(get(params, :level, 1))
        fps = Int(get(params, :fps, 2))
        target_px = Int(get(params, :target_px, 1200))

        bytes = if format == "gif"
            generate_gif(level; fps, target_px)
        else
            generate_mp4(level; fps, target_px)
        end

        content_type = format == "gif" ? "image/gif" : "video/mp4"
        headers = vcat(cors_headers(), [
            "Content-Type" => content_type,
            "Content-Disposition" => "attachment; filename=\"dispersion_animation.$format\"",
        ])
        return HTTP.Response(200, headers, bytes)
    catch e
        body = JSON3.write(Dict("error" => sprint(showerror, e)))
        headers = vcat(cors_headers(), ["Content-Type" => "application/json"])
        return HTTP.Response(500, headers, body)
    end
end

function api_stitch_frames(req::HTTP.Request)
    try
        params = JSON3.read(String(req.body))
        frames_b64 = collect(params[:frames])
        fps = Int(get(params, :fps, 2))
        format = String(get(params, :format, "gif"))

        isempty(frames_b64) && error("No frames provided")

        # Write PNG frames to a temp directory
        tmpdir = mktempdir()
        for (i, b64) in enumerate(frames_b64)
            png_data = base64decode(String(b64))
            write(joinpath(tmpdir, "frame_$(lpad(i, 4, '0')).png"), png_data)
        end

        outpath = tempname() * "." * format

        if format == "gif"
            run(pipeline(`ffmpeg -y -loglevel error
                -framerate $fps -i $(joinpath(tmpdir, "frame_%04d.png"))
                -vf "split[s0][s1];[s0]palettegen=max_colors=256:stats_mode=full[p];[s1][p]paletteuse=dither=sierra2_4a"
                $outpath`))
        else
            run(pipeline(`ffmpeg -y -loglevel error
                -framerate $fps -i $(joinpath(tmpdir, "frame_%04d.png"))
                -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2"
                -c:v libx264 -pix_fmt yuv420p -crf 20
                $outpath`))
        end

        bytes = read(outpath)
        rm(tmpdir, recursive=true, force=true)
        rm(outpath, force=true)

        content_type = format == "gif" ? "image/gif" : "video/mp4"
        headers = vcat(cors_headers(), [
            "Content-Type" => content_type,
            "Content-Disposition" => "attachment; filename=\"dispersion_animation.$format\"",
        ])
        return HTTP.Response(200, headers, bytes)
    catch e
        body = JSON3.write(Dict("error" => sprint(showerror, e)))
        headers = vcat(cors_headers(), ["Content-Type" => "application/json"])
        return HTTP.Response(500, headers, body)
    end
end

# --- Server start ---

function start_server(; port::Int=9000, open_browser::Bool=true)
    url = "http://localhost:$port"
    println("Starting server at $url")

    if open_browser
        @async begin
            sleep(1.0)
            if Sys.iswindows()
                run(`cmd /c start $url`, wait=false)
            elseif Sys.isapple()
                run(`open $url`, wait=false)
            else
                try run(`xdg-open $url`, wait=false) catch; end
            end
        end
    end

    server = HTTP.serve(handle_request, "0.0.0.0", port)
    return server
end
