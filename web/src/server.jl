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
    db_run_id::Union{Int, Nothing}  # current run's database ID
end

const APP = Ref(AppStatus(false, 0, "", "", "", 0.0, 0, "", "mSv/h", nothing))

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
        # Static files — serve from React build (public_react/) if it exists,
        # otherwise fall back to vanilla public/
        react_dir = joinpath(WEB_DIR, "public_react")
        use_react = isdir(react_dir)
        static_dir = use_react ? "public_react" : "public"

        if path == "/" || path == ""
            return serve_file("$(static_dir)/index.html")
        elseif startswith(path, "/public/")
            return serve_file(path[2:end])  # strip leading /
        elseif use_react && startswith(path, "/assets/")
            return serve_file("public_react" * path)
        elseif use_react
            # SPA fallback: try the file, then serve index.html
            trypath = "public_react" * path
            fullpath = joinpath(WEB_DIR, trypath)
            if isfile(fullpath)
                return serve_file(trypath)
            end
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
        elseif path == "/api/predict" && method == "POST"
            return api_predict(req)
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
        elseif path == "/api/observations" && method == "GET"
            return api_observations()
        elseif path == "/api/runs" && method == "GET"
            return api_list_runs(req)
        elseif startswith(path, "/api/runs/") && endswith(path, "/load") && method == "POST"
            id_str = replace(replace(path, "/api/runs/" => ""), "/load" => "")
            return api_load_run(parse(Int, id_str))
        elseif startswith(path, "/api/runs/") && method == "GET"
            id_str = replace(path, "/api/runs/" => "")
            return api_get_run(parse(Int, id_str))
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
    release_duration_hours = Float64(get(params, :release_duration_hours, 1.0))

    # Build parameter dict for database operations
    run_params = Dict(
        "dataset" => get(params, :dataset, nothing),
        "release_mode" => release_mode,
        "weather_source" => weather_source,
        "lat" => lat, "lon" => lon,
        "start_date" => start_date,
        "start_hour" => start_hour,
        "duration_hours" => duration_hours,
        "n_particles" => n_particles,
        "yield_kt" => release_mode == "bomb" ? yield_kt : nothing,
        "activity_tbq" => release_mode == "npp" ? activity_tbq : nothing,
        "stack_height_m" => release_mode == "npp" ? stack_height_m : nothing,
        "isotope" => release_mode == "npp" ? isotope : nothing,
        "release_duration_hours" => release_mode == "npp" ? release_duration_hours : nothing,
        "arl_dir" => weather_source == "arl" ? arl_dir : nothing,
    )

    # Check for cached result with identical parameters
    # Skip cache when client requests it (e.g. to generate animation data)
    skip_cache = Bool(get(params, :skip_cache, false))
    cached = if skip_cache
        nothing
    else
        try
            db_find_cached_run(run_params)
        catch e
            @warn "Cache lookup failed" exception=e
            nothing
        end
    end

    if cached !== nothing
        # Return cached results immediately
        APP[] = AppStatus(false, 100, "Loaded from cache (run #$(cached["id"]))",
                          "", cached["geojson_result"],
                          cached["peak_dose"], cached["n_events"],
                          cached["csv_result"], cached["dose_units"], nothing)
        @info "Cache hit: returning results from run #$(cached["id"])"
        body = JSON3.write(Dict("status" => "cached", "run_id" => cached["id"]))
        headers = vcat(cors_headers(), ["Content-Type" => "application/json"])
        return HTTP.Response(200, headers, body)
    end

    # No cache hit — log new run and compute
    db_id = try
        db_insert_run(run_params)
    catch e
        @warn "Failed to log run to database" exception=e
        nothing
    end

    # Reset state
    APP[] = AppStatus(true, 0, "Starting...", "", "", 0.0, 0, "", "mSv/h", db_id)
    t_start = time()

    # Run simulation asynchronously
    Threads.@spawn begin
        try
            result = run_simulation_with_source(;
                weather_source, arl_dir,
                lat, lon, yield_kt, start_date, start_hour,
                duration_hours, n_particles,
                release_mode, activity_tbq, stack_height_m, isotope, release_duration_hours,
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

            # Store results in database for future cache hits
            if db_id !== nothing
                try
                    db_complete_run(db_id;
                        peak_dose = result.max_dose,
                        dose_units = result.units,
                        n_events = length(result.deposition_log),
                        elapsed_seconds = time() - t_start,
                        geojson = geojson,
                        csv = csv)
                catch e
                    @warn "Failed to update run in database" exception=e
                end
            end
        catch e
            bt = catch_backtrace()
            # Full error with stacktrace
            err_full = sprint() do io
                showerror(io, e, bt)
            end
            APP[].error_msg = sprint(showerror, e)
            APP[].running = false

            # Update database record with failure
            if db_id !== nothing
                try db_fail_run(db_id, sprint(showerror, e)) catch end
            end

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

# --- Prediction endpoint ---

function api_predict(req::HTTP.Request)
    try
        params = JSON3.read(String(req.body))
        site = lowercase(String(get(params, :site, "")))
        date_str = String(get(params, :date, ""))
        hour = Int(get(params, :hour, 0))
        release_duration = Float64(get(params, :release_duration, 48.0))
        release_height = Float64(get(params, :release_height, 100.0))

        isempty(site) && return HTTP.Response(400, cors_headers(), "Missing 'site' parameter")
        isempty(date_str) && return HTTP.Response(400, cors_headers(), "Missing 'date' parameter")

        meta = ARL_METADATA[]
        isnothing(meta) && return HTTP.Response(400, cors_headers(), "No ARL data loaded")

        date = Date(date_str)
        result = Prediction.predict_from_arl(
            site, meta.dir_path, date, hour;
            release_duration=release_duration,
            release_height=release_height
        )

        body = JSON3.write(Dict(
            "impact" => result.impact,
            "probability" => result.probability,
            "site" => result.site,
        ))
        headers = vcat(cors_headers(), ["Content-Type" => "application/json"])
        return HTTP.Response(200, headers, body)
    catch e
        @error "Prediction error" exception=(e, catch_backtrace())
        body = JSON3.write(Dict("error" => sprint(showerror, e)))
        headers = vcat(cors_headers(), ["Content-Type" => "application/json"])
        return HTTP.Response(500, headers, body)
    end
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

# --- Observation data overlays ---

function api_observations()
    dataset = ACTIVE_DATASET[]
    try
        if dataset == "etex"
            data = _load_etex_observations()
        elseif dataset == "nancy"
            data = _load_nancy_observations_geojson()
        else
            return HTTP.Response(404, cors_headers(), "No observations for dataset: $dataset")
        end
        body = JSON3.write(data)
        headers = vcat(cors_headers(), ["Content-Type" => "application/json"])
        return HTTP.Response(200, headers, body)
    catch e
        body = JSON3.write(Dict("error" => sprint(showerror, e)))
        headers = vcat(cors_headers(), ["Content-Type" => "application/json"])
        return HTTP.Response(500, headers, body)
    end
end

function _load_etex_observations()
    meas_file = joinpath(pkgdir(NuclearDetonation), "data", "etex", "meas-t1.txt")
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

    # Grid TIC onto a coarse lat/lon grid and generate contours
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

    # Contour levels spanning observed TIC range
    levels = [100.0, 500.0, 2000.0, 5000.0, 10000.0]
    colors = ["#3288bd", "#66c2a5", "#fee08b", "#f46d43", "#d53e4f"]
    labels = ["100 ng·h/m³", "500 ng·h/m³", "2000 ng·h/m³", "5000 ng·h/m³", "10000 ng·h/m³"]

    cl = Contour.contours(collect(up_lon), collect(up_lat), up_grid, levels)
    features = Dict{String,Any}[]
    for level_obj in Contour.levels(cl)
        level_val = Contour.level(level_obj)
        ci = findfirst(l -> abs(l - level_val) / max(l, 1e-10) < 0.01, levels)
        isnothing(ci) && continue
        for line in Contour.lines(level_obj)
            xs, ys = Contour.coordinates(line)
            coords = [[round(x, digits=4), round(y, digits=4)] for (x, y) in zip(xs, ys)]
            length(coords) < 2 && continue
            coords = _chaikin_smooth(coords, 2)
            push!(features, Dict{String,Any}(
                "type" => "Feature",
                "geometry" => Dict{String,Any}("type" => "LineString", "coordinates" => coords),
                "properties" => Dict{String,Any}(
                    "level" => level_val, "label" => labels[ci], "color" => colors[ci]),
            ))
        end
    end

    geojson = Dict{String,Any}("type" => "FeatureCollection", "features" => features)
    return Dict("type" => "etex", "geojson" => geojson)
end

function _load_nancy_observations_geojson()
    obs = Transport.load_nancy_observations()
    # Convert dose rate contours to GeoJSON features
    features = []
    # Dose rate colors (matching typical fallout contour palette)
    colors = Dict(0.4 => "#3288bd", 1.0 => "#66c2a5", 4.0 => "#abdda4",
                  10.0 => "#fee08b", 40.0 => "#f46d43", 100.0 => "#d53e4f")
    for c in obs.dose_rate_contours
        for poly_coords in c.polygons
            coords = [[pt[2], pt[1]] for pt in poly_coords]  # (lat,lon) → [lon,lat] for GeoJSON
            feature = Dict(
                "type" => "Feature",
                "geometry" => Dict("type" => "Polygon", "coordinates" => [coords]),
                "properties" => Dict("dose_rate" => c.dose_rate_mR_hr,
                    "label" => "$(c.dose_rate_mR_hr) mR/h",
                    "color" => get(colors, c.dose_rate_mR_hr, "#999")),
            )
            push!(features, feature)
        end
    end
    geojson = Dict("type" => "FeatureCollection", "features" => features)
    return Dict("type" => "nancy", "geojson" => geojson,
                "detonation_lat" => obs.detonation_lat,
                "detonation_lon" => obs.detonation_lon)
end

# --- Simulation history (PostgreSQL) ---

function api_list_runs(req::HTTP.Request)
    uri = HTTP.URI(req.target)
    query = HTTP.queryparams(uri)
    limit = parse(Int, get(query, "limit", "50"))
    offset = parse(Int, get(query, "offset", "0"))
    limit = clamp(limit, 1, 200)

    try
        runs = db_list_runs(; limit, offset)
        total = db_run_count()

        # Convert columntable to array of dicts
        n = length(runs.id)
        rows = [Dict(
            "id" => runs.id[i],
            "created_at" => string(runs.created_at[i]),
            "dataset" => runs.dataset[i],
            "release_mode" => runs.release_mode[i],
            "weather_source" => runs.weather_source[i],
            "latitude" => runs.latitude[i],
            "longitude" => runs.longitude[i],
            "start_date" => runs.start_date[i],
            "start_hour" => runs.start_hour[i],
            "duration_hours" => runs.duration_hours[i],
            "n_particles" => runs.n_particles[i],
            "yield_kt" => runs.yield_kt[i],
            "activity_tbq" => runs.activity_tbq[i],
            "isotope" => runs.isotope[i],
            "status" => runs.status[i],
            "peak_dose" => runs.peak_dose[i],
            "dose_units" => runs.dose_units[i],
            "n_events" => runs.n_events[i],
            "elapsed_seconds" => runs.elapsed_seconds[i],
        ) for i in 1:n]

        body = JSON3.write(Dict("runs" => rows, "total" => total,
                                "limit" => limit, "offset" => offset))
        headers = vcat(cors_headers(), ["Content-Type" => "application/json"])
        return HTTP.Response(200, headers, body)
    catch e
        body = JSON3.write(Dict("error" => sprint(showerror, e)))
        headers = vcat(cors_headers(), ["Content-Type" => "application/json"])
        return HTTP.Response(500, headers, body)
    end
end

function api_get_run(id::Int)
    try
        run = db_get_run(id)
        if run === nothing
            body = JSON3.write(Dict("error" => "Run not found"))
            headers = vcat(cors_headers(), ["Content-Type" => "application/json"])
            return HTTP.Response(404, headers, body)
        end
        body = JSON3.write(run)
        headers = vcat(cors_headers(), ["Content-Type" => "application/json"])
        return HTTP.Response(200, headers, body)
    catch e
        body = JSON3.write(Dict("error" => sprint(showerror, e)))
        headers = vcat(cors_headers(), ["Content-Type" => "application/json"])
        return HTTP.Response(500, headers, body)
    end
end

function api_load_run(id::Int)
    try
        run = db_get_run(id)
        if run === nothing
            body = JSON3.write(Dict("error" => "Run not found"))
            headers = vcat(cors_headers(), ["Content-Type" => "application/json"])
            return HTTP.Response(404, headers, body)
        end
        if run["status"] != "completed" || run["geojson_result"] === missing || run["geojson_result"] === nothing
            body = JSON3.write(Dict("error" => "Run has no stored results"))
            headers = vcat(cors_headers(), ["Content-Type" => "application/json"])
            return HTTP.Response(400, headers, body)
        end

        # Load results into APP[] as if the simulation just completed
        APP[] = AppStatus(false, 100, "Loaded from history (run #$id)",
                          "", run["geojson_result"],
                          run["peak_dose"], run["n_events"],
                          something(run["csv_result"], ""),
                          something(run["dose_units"], "mSv/h"), nothing)

        body = JSON3.write(Dict(
            "status" => "loaded",
            "run_id" => id,
            "peak_dose" => run["peak_dose"],
            "dose_units" => run["dose_units"],
            "n_events" => run["n_events"],
            "release_mode" => run["release_mode"],
            "latitude" => run["latitude"],
            "longitude" => run["longitude"],
            "yield_kt" => run["yield_kt"],
            "start_date" => run["start_date"],
            "start_hour" => run["start_hour"],
            "duration_hours" => run["duration_hours"],
            "n_particles" => run["n_particles"],
        ))
        headers = vcat(cors_headers(), ["Content-Type" => "application/json"])
        return HTTP.Response(200, headers, body)
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
