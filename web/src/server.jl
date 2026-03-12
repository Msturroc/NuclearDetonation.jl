# HTTP server for the web GUI

using HTTP
using JSON3

# --- Application state ---

mutable struct AppStatus
    running::Bool
    progress_pct::Int
    progress_msg::String
    error_msg::String
    geojson::String
    max_dose_mRh::Float64
    n_events::Int
    csv_data::String
    units::String
end

const APP = Ref(AppStatus(false, 0, "", "", "", 0.0, 0, "", "mR/h"))

const WEB_DIR = dirname(@__DIR__)  # web/

function mime_type(path::String)
    endswith(path, ".html") && return "text/html"
    endswith(path, ".css")  && return "text/css"
    endswith(path, ".js")   && return "application/javascript"
    endswith(path, ".json") && return "application/json"
    endswith(path, ".png")  && return "image/png"
    endswith(path, ".svg")  && return "image/svg+xml"
    endswith(path, ".csv")  && return "text/csv"
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
        "max_dose_mRh" => s.max_dose_mRh,
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

    # Reset state
    APP[] = AppStatus(true, 0, "Starting...", "", "", 0.0, 0, "", "mR/h")

    # Run simulation asynchronously
    Threads.@spawn begin
        try
            result = run_dispersion_simulation(;
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
            APP[].max_dose_mRh = result.max_dose_mRh
            APP[].n_events = length(result.deposition_log)
            APP[].units = result.units
            APP[].progress_pct = 100
            APP[].progress_msg = "Complete"
            APP[].running = false
        catch e
            APP[].error_msg = sprint(showerror, e)
            APP[].running = false
            @error "Simulation failed" exception=(e, catch_backtrace())
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
