# GLMakie desktop window: scenario controls on the left, map on the right

const PANEL_WIDTH = 410
const OCEAN_COLOR  = RGBf(0.85, 0.91, 0.96)
const LAND_COLOR   = RGBf(0.97, 0.96, 0.93)
const BORDER_COLOR = RGBf(0.62, 0.62, 0.62)
const DOMAIN_COLOR = RGBf(0.20, 0.40, 0.85)
const ERROR_COLOR  = RGBf(0.75, 0.10, 0.10)
const OK_COLOR     = RGBf(0.10, 0.50, 0.20)
const MUTED_COLOR  = RGBf(0.35, 0.35, 0.35)

# Built-in ERA5 datasets: default release point, time window and source
const DATASET_PRESETS = Dict(
    "nancy" => (label = "Nancy (NTS) — Nevada, Mar 1953",
                lat = 37.0956, lon = -116.1028, hour = 13, duration = 12,
                date_min = Date(1953, 3, 24), date_max = Date(1953, 3, 27),
                mode = "bomb", source_terms = "Cs-137 1.0",
                stack_height = 100.0, release_duration = 1.0),
    "etex"  => (label = "ETEX (Europe) — Monterfil, Oct 1994",
                lat = 48.058, lon = -2.008, hour = 16, duration = 48,
                date_min = Date(1994, 10, 23), date_max = Date(1994, 10, 27),
                mode = "npp", source_terms = "Generic 1.0",
                stack_height = 10.0, release_duration = 12.0),
)
const DATASET_ORDER = ["nancy", "etex"]

# Dose-rate contours (mSv/h at H+duration)
const DOSE_LEVELS = [0.004, 0.01, 0.04, 0.1, 0.4, 1.0]
const DOSE_COLORS = ["#3366FF", "#00CCCC", "#33AA33", "#CCCC00", "#FF8800", "#CC0000"]

# Deposition contours (kBq/m²). Lower levels are for visualising small releases;
# the upper four are the IAEA/Chernobyl Cs-137 zoning thresholds.
const DEP_LEVELS = [0.001, 0.01, 0.1, 1.0, 10.0, 37.0, 185.0, 555.0, 1480.0]
const DEP_COLORS = ["#C8DCFF", "#8FB7E8", "#5A8DD0", "#3366FF", "#00CCCC",
                    "#33AA33", "#CCCC00", "#FF8800", "#CC0000"]
const DEP_ZONES = Dict(37.0 => "monitoring", 185.0 => "resettlement",
                       555.0 => "relocation", 1480.0 => "exclusion")

# Display units: (label, factor from the base unit)
const DOSE_UNITS = [("mSv/h", 1.0), ("μSv/h", 1000.0), ("Sv/h", 0.001), ("mR/h", 100.0)]
const DEP_UNITS  = [("kBq/m²", 1.0), ("Ci/km²", 0.02703)]

# Menu options whose selection is the whole (label, factor) pair
_unit_options(units) = [(u, (u, f)) for (u, f) in units]

# --- Formatting helpers ---

function _fmt(v::Real)
    a = abs(v)
    a == 0 && return "0"
    a >= 100 && return string(round(Int, v))
    a >= 1 && return @sprintf("%.1f", v)
    a >= 0.01 && return @sprintf("%.3f", v)
    return @sprintf("%.2e", v)
end

_trim(x) = (r = round(x, digits=2); isinteger(r) ? string(Int(r)) : string(r))
_lon_label(v) = string(_trim(abs(v)), "°", v < 0 ? "W" : v > 0 ? "E" : "")
_lat_label(v) = string(_trim(abs(v)), "°", v < 0 ? "S" : v > 0 ? "N" : "")

function _log_ticks(lo, hi)
    ks = ceil(Int, lo):floor(Int, hi)
    return (Float64.(collect(ks)), [rich("10", superscript(string(k))) for k in ks])
end

_text(tb) = strip(something(tb.displayed_string[], ""))
_set!(tb, s) = Makie.set!(tb, string(s))

"""
    parse_source_terms(text) -> (isotopes, activities_tbq, halflives_hours)

Comma- or semicolon-separated entries of `isotope activity_TBq [half-life_h]`,
e.g. `"Cs-137 1.0, I-131 5.0, X-1 2.0 12.5"`. Isotopes in `ISOTOPE_HALFLIVES` take
their preset half-life unless one is given; anything else needs one (0 = no decay).
"""
function parse_source_terms(text::AbstractString)
    isotopes, activities, halflives = String[], Float64[], Float64[]
    for entry in split(text, r"[,;]")
        parts = split(strip(entry))
        isempty(parts) && continue
        length(parts) in (2, 3) ||
            error("Source term \"$(strip(entry))\" should be: isotope activity_TBq [half-life_h]")
        name = _canonical_isotope(parts[1])
        act = tryparse(Float64, parts[2])
        (act === nothing || act <= 0) && error("Activity for $name must be a positive number of TBq")
        hl = if length(parts) == 3
            h = tryparse(Float64, parts[3])
            (h === nothing || h < 0) && error("Half-life for $name must be a number of hours ≥ 0")
            h
        else
            haskey(ISOTOPE_HALFLIVES, name) ||
                error("No preset half-life for $name — add one in hours, e.g. \"$name $(parts[2]) 24\"")
            ISOTOPE_HALFLIVES[name]
        end
        push!(isotopes, name); push!(activities, act); push!(halflives, hl)
    end
    isempty(isotopes) && error("Enter at least one source term, e.g. \"Cs-137 1.0\"")
    return isotopes, activities, halflives
end

function _canonical_isotope(name)
    for k in keys(ISOTOPE_HALFLIVES)
        lowercase(k) == lowercase(name) && return k
    end
    return String(name)
end

function default_output_dir()
    docs = joinpath(homedir(), "Documents")
    return joinpath(isdir(docs) ? docs : homedir(), "NuclearDetonation")
end

# --- Application state ---

mutable struct ProgressReport
    lock::ReentrantLock
    pct::Int
    msg::String
end

mutable struct GUI
    fig::Figure
    ax::Axis
    w::NamedTuple                     # widgets
    busy::Observable{Bool}
    progress::Observable{Float64}
    status::Observable{String}
    status_color::Observable{RGBf}
    prediction::Observable{String}
    prediction_color::Observable{RGBf}
    results_text::Observable{String}
    time_text::Observable{String}
    cursor_text::Observable{String}
    release_pt::Observable{Point2f}
    domain_pts::Observable{Vector{Point2f}}
    dataset::String
    arl_meta::Any
    result::Any                       # SimulationResult of the last run
    run_info::Any                     # parameters of the last run
    contour_plots::Vector{Any}
    observations::Any
    obs_plots::Vector{Any}
    legend::Any
    anim::Any                         # NamedTuple from animation_frames
    anim_key::Any                     # (result, level) the frames were built for
    plume_field::Observable{Matrix{Float32}}
    plume_plot::Any
    frame::Observable{Int}
    playing::Bool
    colorbar::Colorbar
    colorbar_box::Box
end

function status!(g::GUI, msg::AbstractString; level::Symbol = :info)
    g.status[] = msg
    g.status_color[] = level === :error ? ERROR_COLOR : level === :ok ? OK_COLOR : MUTED_COLOR
end

# --- Map layers (shared by the window and the exported figures) ---

function map_axis(pos; kwargs...)
    ax = Axis(pos; backgroundcolor = OCEAN_COLOR,
              xgridcolor = (:black, 0.07), ygridcolor = (:black, 0.07),
              xtickformat = vs -> _lon_label.(vs), ytickformat = vs -> _lat_label.(vs),
              kwargs...)
    poly!(ax, load_basemap(); color = LAND_COLOR, strokecolor = BORDER_COLOR,
          strokewidth = 0.6, inspectable = false)
    return ax
end

function draw_npp!(ax)
    pts = [Point2f(p.lon, p.lat) for p in NPP_PLANTS]
    s = scatter!(ax, pts; color = :gold, strokecolor = :black, strokewidth = 1.5, markersize = 12)
    t = text!(ax, pts; text = [p.name for p in NPP_PLANTS], offset = (8, 0),
              align = (:left, :center), fontsize = 11, color = :black)
    translate!(s, 0, 0, 40); translate!(t, 0, 0, 40)
    return s
end

function draw_release!(ax, pt)
    s = scatter!(ax, pt; marker = :star5, markersize = 22, color = :red,
                 strokecolor = :black, strokewidth = 1)
    translate!(s, 0, 0, 50)
    return s
end

function draw_contours(ax, res, visible)
    levels, colors = res.units == "kBq/m²" ? (DEP_LEVELS, DEP_COLORS) : (DOSE_LEVELS, DOSE_COLORS)
    xs, ys = collect(res.lon_grid), collect(res.lat_grid)
    plots = Any[]
    for (lv, c) in zip(levels, colors)
        lv < res.max_dose || continue  # never reached
        p = contour!(ax, xs, ys, res.dose_grid; levels = [lv], color = c,
                     linewidth = 2, visible)
        translate!(p, 0, 0, 20)
        push!(plots, p)
    end
    return plots
end

function draw_observations(ax, obs, visible)
    plots = Any[]
    if obs.kind === :grid
        for (lv, c) in zip(obs.levels, obs.colors)
            push!(plots, contour!(ax, obs.lons, obs.lats, obs.grid; levels = [lv], color = c,
                                  linewidth = 2, linestyle = :dash, visible))
        end
    else
        for (val, pts) in obs.polygons
            c = obs.colors[findfirst(==(val), obs.levels)]
            push!(plots, lines!(ax, pts; color = c, linewidth = 2, linestyle = :dash, visible))
        end
    end
    foreach(p -> translate!(p, 0, 0, 25), plots)
    return plots
end

function plume_image!(ax, a, field, visible = true)
    p = image!(ax, a.lon_min .. a.lon_max, a.lat_min .. a.lat_max, field;
               colormap = PLUME_COLORMAP, colorrange = (a.log_min, a.log_max),
               lowclip = RGBAf(0, 0, 0, 0), interpolate = true, visible)
    translate!(p, 0, 0, 10)
    return p
end

"""Legend contents for the visible contour layers, or `nothing`."""
function legend_groups(g::GUI)
    groups, labels, titles = Vector{Any}[], Vector{Any}[], String[]
    res = g.result
    if res !== nothing && g.w.contours.active[]
        unit, factor = something(g.w.units.selection[], (res.units, 1.0))
        dep = res.units == "kBq/m²"
        levels, colors = dep ? (DEP_LEVELS, DEP_COLORS) : (DOSE_LEVELS, DOSE_COLORS)
        push!(groups, [LineElement(color = c, linewidth = 3) for c in reverse(colors)])
        push!(labels, [string(_fmt(lv * factor), " ", unit,
                              haskey(DEP_ZONES, lv) && dep ? " ($(DEP_ZONES[lv]))" : "")
                       for lv in reverse(levels)])
        push!(titles, (dep ? "Deposition" : "Dose rate") * " at H+$(g.run_info.duration) h")
    end
    obs = g.observations
    if obs !== nothing && g.w.obs.active[]
        push!(groups, [LineElement(color = c, linewidth = 2, linestyle = :dash) for c in reverse(obs.colors)])
        push!(labels, reverse(obs.labels))
        push!(titles, obs.title)
    end
    return isempty(groups) ? nothing : (groups, labels, titles)
end

function add_legend!(pos, contents)
    groups, labels, titles = contents
    return Legend(pos, groups, labels, titles;
                  tellwidth = false, tellheight = false, halign = :left, valign = :bottom,
                  margin = (12, 12, 12, 12), backgroundcolor = (:white, 0.92),
                  framecolor = (:black, 0.3), labelsize = 11, titlesize = 12,
                  patchsize = (18, 10), rowgap = 0, padding = (8, 8, 6, 6))
end

function rebuild_legend!(g::GUI)
    isnothing(g.legend) || delete!(g.legend)
    contents = legend_groups(g)
    g.legend = contents === nothing ? nothing : add_legend!(g.fig[1, 2], contents)
end

function set_view!(ax, lon_min, lon_max, lat_min, lat_max; pad = 0.05)
    dx, dy = (lon_max - lon_min) * pad, (lat_max - lat_min) * pad
    ax.autolimitaspect[] = 1 / cosd(clamp((lat_min + lat_max) / 2, -80, 80))
    ax.limits[] = (lon_min - dx, lon_max + dx, lat_min - dy, lat_max + dy)
    reset_limits!(ax)
end

function weather_bounds(g::GUI)
    if g.w.weather.selection[] == "arl"
        m = g.arl_meta
        m === nothing && return nothing
        return (lon_min = m.lon_min, lon_max = m.lon_max, lat_min = m.lat_min, lat_max = m.lat_max)
    end
    era5 = ERA5_STATE[]
    era5 === nothing && return nothing
    lons = [l > 180 ? l - 360 : l for l in era5.lon_range]
    return (lon_min = minimum(lons), lon_max = maximum(lons),
            lat_min = minimum(era5.lat_range), lat_max = maximum(era5.lat_range))
end

function refresh_domain!(g::GUI; fit = true)
    b = weather_bounds(g)
    if b === nothing
        g.domain_pts[] = [Point2f(NaN, NaN)]
        return
    end
    g.domain_pts[] = Point2f[(b.lon_min, b.lat_min), (b.lon_max, b.lat_min),
                             (b.lon_max, b.lat_max), (b.lon_min, b.lat_max),
                             (b.lon_min, b.lat_min)]
    fit && set_view!(g.ax, b.lon_min, b.lon_max, b.lat_min, b.lat_max)
end

# --- Background work ---

"""
    background!(work, g; done, failed)

Run `work(report)` on a worker thread so the window stays responsive, where
`report(pct, msg)` updates the progress bar. `done(result)` then runs on the GUI
task. Errors land in the status line and in `error.log` in the output folder.
"""
function background!(work, g::GUI; done = _ -> nothing, failed = "Failed")
    prog = ProgressReport(ReentrantLock(), 0, "")
    report = (pct, msg) -> lock(prog.lock) do
        prog.pct = pct; prog.msg = msg
    end
    g.busy[] = true
    task = Threads.@spawn work(report)
    @async try
        while !istaskdone(task)
            _show_progress!(g, prog)
            sleep(0.15)
        end
        _show_progress!(g, prog)
        done(fetch(task))
    catch e
        _report_error!(g, failed, e)
    finally
        g.busy[] = false
    end
    return task
end

function _show_progress!(g, prog)
    pct, msg = lock(() -> (prog.pct, prog.msg), prog.lock)
    isempty(msg) && return
    g.progress[] = pct
    status!(g, msg)
end

function _report_error!(g::GUI, what, e)
    inner = e isa TaskFailedException ? e.task.exception : e
    msg = sprint(showerror, inner)
    status!(g, "$what: $(first(msg, 400))"; level = :error)
    try
        dir = mkpath(_text(g.w.outdir))
        open(joinpath(dir, "error.log"), "a") do io
            println(io, "\n", "="^60, "\n", Dates.now(), " — ", what)
            e isa TaskFailedException ? showerror(io, e) : showerror(io, e, catch_backtrace())
            println(io)
        end
    catch
    end
    @error what exception = e
end

# --- Actions ---

function set_release!(g::GUI, lat, lon)
    _set!(g.w.lat, round(lat, digits = 4))
    _set!(g.w.lon, round(lon, digits = 4))
    g.release_pt[] = Point2f(lon, lat)
end

function apply_preset!(g::GUI, key)
    p = DATASET_PRESETS[key]
    set_release!(g, p.lat, p.lon)
    _set!(g.w.date, p.date_min)
    _set!(g.w.hour, p.hour)
    _set!(g.w.duration, p.duration)
    _set!(g.w.yield, 24.0)
    _set!(g.w.sources, p.source_terms)
    _set!(g.w.stack, p.stack_height)
    _set!(g.w.release_hours, p.release_duration)
    g.w.release.i_selected[] = p.mode == "bomb" ? 1 : 2
end

function load_dataset!(g::GUI, key::String)
    g.busy[] && return status!(g, "Busy — wait for the current task to finish"; level = :error)
    label = DATASET_PRESETS[key].label
    clear_results!(g)
    background!(g; failed = "Loading $label failed",
                done = _ -> begin
                    g.dataset = key
                    g.observations = nothing
                    foreach(p -> delete!(g.ax, p), g.obs_plots); empty!(g.obs_plots)
                    g.w.obs.active[] = false
                    g.w.weather.i_selected[] = 1
                    apply_preset!(g, key)
                    refresh_domain!(g)
                    g.progress[] = 100
                    status!(g, "$label ready — click inside the dashed box to place the release";
                            level = :ok)
                end) do report
        preload_era5!(dataset = key, progress_callback = report)
    end
end

function load_arl!(g::GUI)
    g.busy[] && return status!(g, "Busy — wait for the current task to finish"; level = :error)
    path = String(_text(g.w.arl_path))
    isempty(path) && return status!(g, "Enter an ARL file or folder path first"; level = :error)
    background!(g; failed = "Could not read ARL data",
                done = b -> begin
                    g.arl_meta = merge(b, (dir_path = path,))
                    g.w.weather.i_selected[] = 2
                    _set!(g.w.date, b.date_min)
                    _set!(g.w.hour, 0)
                    set_release!(g, (b.lat_min + b.lat_max) / 2, (b.lon_min + b.lon_max) / 2)
                    refresh_domain!(g)
                    status!(g, "Loaded $(b.n_files) ARL file(s) ($(_trim(b.resolution))° grid, " *
                               "$(length(b.pressure_levels)) levels), $(b.date_min) to $(b.date_max). " *
                               "Click inside the dashed box to place the release."; level = :ok)
                end) do report
        load_arl_metadata!(path; progress_callback = report)
    end
end

function on_map_click!(g::GUI, lon, lat)
    lims = g.ax.finallimits[]
    w, h = lims.widths
    for plant in NPP_PLANTS
        if hypot((plant.lon - lon) / w, (plant.lat - lat) / h) < 0.012
            return select_plant!(g, plant)
        end
    end
    b = weather_bounds(g)
    if b !== nothing && !(b.lon_min <= lon <= b.lon_max && b.lat_min <= lat <= b.lat_max)
        return status!(g, "That point is outside the weather data — click inside the dashed box";
                       level = :error)
    end
    set_release!(g, lat, lon)
end

function select_plant!(g::GUI, plant)
    set_release!(g, plant.lat, plant.lon)
    g.w.release.i_selected[] = 2
    if g.w.weather.selection[] != "arl" || g.arl_meta === nothing
        g.prediction_color[] = MUTED_COLOR
        g.prediction[] = "$(plant.name) selected. Load ARL weather files to run the Ireland impact prediction."
        return
    end
    date = tryparse(Date, _text(g.w.date))
    hour = tryparse(Int, _text(g.w.hour))
    rel = tryparse(Float64, _text(g.w.release_hours))
    stack = tryparse(Float64, _text(g.w.stack))
    if any(isnothing, (date, hour, rel, stack))
        return status!(g, "Check the start date, hour, release duration and stack height"; level = :error)
    end
    g.prediction_color[] = MUTED_COLOR
    g.prediction[] = "Running impact prediction for $(plant.name)…"
    dir = g.arl_meta.dir_path
    background!(g; failed = "Impact prediction failed",
                done = r -> begin
                    g.prediction_color[] = r.impact ? ERROR_COLOR : OK_COLOR
                    g.prediction[] = "XGBoost, $(plant.name): Ireland " *
                                     (r.impact ? "WILL" : "will NOT") * " be impacted " *
                                     "(probability $(round(100 * r.probability, digits = 1))%)"
                end) do _
        Prediction.predict_from_arl(plant.site, dir, date, hour;
                                    release_duration = rel, release_height = stack)
    end
end

"""Validate the form and turn it into keyword arguments for `run_simulation_with_source`."""
function collect_params(g::GUI)
    w = g.w
    function num(tb, name)
        v = tryparse(Float64, _text(tb))
        v === nothing && error("$name must be a number")
        return v
    end
    function int(tb, name)
        v = tryparse(Int, _text(tb))
        v === nothing && error("$name must be a whole number")
        return v
    end

    lat = num(w.lat, "Latitude")
    lon = num(w.lon, "Longitude")
    date = tryparse(Date, _text(w.date))
    date === nothing && error("Start date must look like 1953-03-24")
    hour = int(w.hour, "Hour")
    0 <= hour <= 23 || error("Hour must be between 0 and 23")
    duration = int(w.duration, "Duration")
    1 <= duration <= 168 || error("Duration must be between 1 and 168 hours")
    particles = int(w.particles, "Particles")
    100 <= particles <= 50_000 || error("Particles must be between 100 and 50,000")
    mode = w.release.selection[]
    weather = w.weather.selection[]

    if weather == "arl"
        g.arl_meta === nothing && error("Load ARL weather files first")
        dmin, dmax = Date(string(g.arl_meta.date_min)), Date(string(g.arl_meta.date_max))
    else
        p = DATASET_PRESETS[g.dataset]
        dmin, dmax = p.date_min, p.date_max
    end
    dmin <= date <= dmax || error("Start date must be between $dmin and $dmax for this weather data")
    b = weather_bounds(g)
    if b !== nothing && !(b.lon_min <= lon <= b.lon_max && b.lat_min <= lat <= b.lat_max)
        error("The release point is outside the weather data domain")
    end

    kwargs = (; weather_source = weather,
                arl_dir = weather == "arl" ? g.arl_meta.dir_path : "",
                lat, lon, start_date = string(date), start_hour = hour,
                duration_hours = duration, n_particles = particles, release_mode = mode)
    summary = ""
    if mode == "bomb"
        yield_kt = num(w.yield, "Yield")
        0.1 <= yield_kt <= 1000 || error("Yield must be between 0.1 and 1000 kt")
        kwargs = merge(kwargs, (; yield_kt))
        summary = "$(_trim(yield_kt)) kt"
    else
        isotopes, activities_tbq, halflives_hours = parse_source_terms(_text(w.sources))
        stack_height_m = num(w.stack, "Stack height")
        release_duration_hours = num(w.release_hours, "Release duration")
        release_duration_hours > 0 || error("Release duration must be positive")
        stack_height_m >= 0 || error("Stack height can't be negative")
        kwargs = merge(kwargs, (; isotopes, activities_tbq, halflives_hours,
                                  stack_height_m, release_duration_hours))
        summary = join(("$i $(_trim(a)) TBq" for (i, a) in zip(isotopes, activities_tbq)), ", ")
    end
    source = weather == "arl" ? "ARL" : DATASET_PRESETS[g.dataset].label
    return (; kwargs, mode, duration, summary, source,
              name = weather == "arl" ? "arl" : g.dataset,
              start = Dates.DateTime(date) + Dates.Hour(hour))
end

function run_clicked!(g::GUI)
    g.busy[] && return
    params = try
        collect_params(g)
    catch e
        e isa ErrorException || rethrow()
        return status!(g, e.msg; level = :error)
    end
    clear_results!(g)
    g.progress[] = 0
    status!(g, "Starting…")
    t0 = time()
    background!(g; failed = "Simulation failed",
                done = res -> show_result!(g, res, params, time() - t0)) do report
        run_simulation_with_source(; params.kwargs..., progress_callback = report)
    end
end

function clear_results!(g::GUI)
    stop_playback!(g)
    g.result = nothing
    foreach(p -> delete!(g.ax, p), g.contour_plots); empty!(g.contour_plots)
    isnothing(g.plume_plot) || delete!(g.ax, g.plume_plot)
    g.plume_plot = nothing
    g.anim = nothing
    g.anim_key = nothing
    g.colorbar.blockscene.visible[] = false
    g.colorbar_box.visible[] = false
    g.results_text[] = ""
    g.time_text[] = ""
    g.w.level.options[] = [("Run a simulation first", -1)]
    g.w.level.i_selected[] = 1
    rebuild_legend!(g)
end

function show_result!(g::GUI, res, params, elapsed)
    g.result = res
    g.run_info = params
    g.contour_plots = draw_contours(g.ax, res, g.w.contours.active)
    g.w.units.options[] = _unit_options(res.units == "kBq/m²" ? DEP_UNITS : DOSE_UNITS)
    g.w.units.i_selected[] = 1
    update_results_text!(g)
    rebuild_legend!(g)

    levels = get_available_levels()
    g.w.level.options[] = isempty(levels) ? [("No airborne particles", -1)] : levels
    g.w.level.i_selected[] = 1
    isempty(levels) || load_animation!(g, last(first(levels)))

    g.ax.title[] = "$(params.source) · $(params.summary)"
    g.progress[] = 100
    status!(g, "Finished in $(round(Int, elapsed)) s — $(params.summary), $(params.source)"; level = :ok)
end

function update_results_text!(g::GUI)
    res = g.result
    res === nothing && return
    unit, factor = something(g.w.units.selection[], (res.units, 1.0))
    peak = res.units == "kBq/m²" ? "Peak deposition" : "Peak dose rate"
    g.results_text[] = "$peak: $(_fmt(res.max_dose * factor)) $unit · " *
                       "$(length(res.deposition_log)) deposition events"
end

function toggle_observations!(g::GUI, show::Bool)
    show || return rebuild_legend!(g)
    if g.observations === nothing
        obs = g.w.weather.selection[] == "arl" ? nothing : try
            load_observations(g.dataset)
        catch e
            _report_error!(g, "Observations unavailable", e)
            nothing
        end
        if obs === nothing
            g.w.obs.active[] = false
            return status!(g, "No observations for this weather data"; level = :error)
        end
        g.observations = obs
        g.obs_plots = draw_observations(g.ax, obs, g.w.obs.active)
    end
    rebuild_legend!(g)
end

# --- Animation ---

function load_animation!(g::GUI, level::Int)
    level < 0 && return
    g.anim_key === (g.result, level) && return
    g.anim_key = (g.result, level)
    stop_playback!(g)
    isnothing(g.plume_plot) || delete!(g.ax, g.plume_plot)
    g.plume_plot = nothing
    a = animation_frames(level)
    g.anim = a
    if a === nothing
        g.colorbar.blockscene.visible[] = false
        g.colorbar_box.visible[] = false
        g.time_text[] = "No airborne particles at this level"
        return
    end
    n = length(a.frames)
    i = clamp(g.frame[], 1, n)
    g.plume_field = Observable(a.frames[i])
    g.plume_plot = plume_image!(g.ax, a, g.plume_field, g.w.plume.active)
    g.colorbar.limits[] = (a.log_min, a.log_max)
    g.colorbar.ticks[] = _log_ticks(a.log_min, a.log_max)
    g.colorbar.label[] = "Airborne activity"
    g.colorbar.blockscene.visible[] = g.w.plume.active[]
    g.colorbar_box.visible[] = g.w.plume.active[]
    g.w.slider.range[] = 1:n
    Makie.set_close_to!(g.w.slider, i)
    show_frame!(g, i)
end

function show_frame!(g::GUI, i::Int)
    a = g.anim
    a === nothing && return
    i = clamp(i, 1, length(a.frames))
    g.plume_field[] = a.frames[i]
    g.time_text[] = _frame_time(g, a, i)
end

function _frame_time(g, a, i)
    t = a.times_h[i]
    stamp = Dates.format(g.run_info.start + Dates.Minute(round(Int, 60t)), "yyyy-mm-dd HH:MM")
    return "H+$(_trim(t)) h · $stamp UTC"
end

function _fps(g::GUI)
    v = tryparse(Int, _text(g.w.fps))
    return v === nothing ? 2 : clamp(v, 1, 30)
end

function start_playback!(g::GUI)
    (g.anim === nothing || g.playing) && return
    g.playing = true
    g.w.plume.active[] = true
    g.w.play.label[] = "Pause"
    @async while g.playing && g.anim !== nothing
        Makie.set_close_to!(g.w.slider, g.frame[] % length(g.anim.frames) + 1)
        sleep(1 / _fps(g))
    end
end

function stop_playback!(g::GUI)
    g.playing = false
    g.w.play.label[] = "Play"
end

function step_frame!(g::GUI, delta)
    g.anim === nothing && return
    stop_playback!(g)
    g.w.plume.active[] = true
    n = length(g.anim.frames)
    Makie.set_close_to!(g.w.slider, mod1(g.frame[] + delta, n))
end

# --- Exports ---

function output_path(g::GUI, what, ext)
    dir = mkpath(String(_text(g.w.outdir)))
    name = g.run_info === nothing ? "nucdet" : "$(g.run_info.name)_$(g.run_info.mode)"
    return joinpath(dir, "$(name)_$(Dates.format(Dates.now(), "yyyymmdd-HHMMSS"))_$what.$ext")
end

"""Standalone map figure of the current results, used for PNG and animation export."""
function export_figure(g::GUI; limits, frame = nothing, size = (1400, 1000))
    fig = Figure(; size, fontsize = 15)
    info = g.run_info
    title = info === nothing ? "NuclearDetonation.jl" :
        "$(info.source) · $(info.summary) · release $(Dates.format(info.start, "yyyy-mm-dd HH:MM")) UTC"
    subtitle = Observable("")
    ax = map_axis(fig[1, 1]; title, subtitle)
    res = g.result
    res === nothing || draw_contours(ax, res, g.w.contours.active[])
    g.observations === nothing || draw_observations(ax, g.observations, g.w.obs.active[])
    field = nothing
    if frame !== nothing && g.anim !== nothing
        a = g.anim
        field = Observable(a.frames[frame])
        plume_image!(ax, a, field)
        Colorbar(fig[1, 2]; colormap = PLUME_COLORMAP, limits = (a.log_min, a.log_max),
                 ticks = _log_ticks(a.log_min, a.log_max), label = "Airborne activity · $(a.label)",
                 height = Relative(0.6))
    end
    draw_npp!(ax)
    draw_release!(ax, g.release_pt[])
    contents = legend_groups(g)
    contents === nothing || add_legend!(fig[1, 1], contents)
    ax.autolimitaspect = 1 / cosd(clamp((limits[3] + limits[4]) / 2, -80, 80))
    limits!(ax, limits...)
    return fig, field, subtitle
end

function export_csv!(g::GUI)
    g.result === nothing && return status!(g, "Run a simulation first"; level = :error)
    path = output_path(g, "deposition", "csv")
    write(path, export_deposition_csv(g.result))
    _saved!(g, path)
end

_saved!(g, path) = status!(g, "Saved $(basename(path)) to the output folder"; level = :ok)

function save_png!(g::GUI)
    lims = g.ax.finallimits[]
    (x0, y0), (w, h) = lims.origin, lims.widths
    frame = g.w.plume.active[] && g.anim !== nothing ? g.frame[] : nothing
    fig, _, subtitle = export_figure(g; limits = (x0, x0 + w, y0, y0 + h), frame)
    frame === nothing || (subtitle[] = _frame_time(g, g.anim, frame))
    path = output_path(g, "map", "png")
    save(path, fig; px_per_unit = 1.5)
    _saved!(g, path)
end

function export_animation!(g::GUI, ext)
    g.anim === nothing && return status!(g, "Run a simulation first"; level = :error)
    stop_playback!(g)
    a, fps = g.anim, _fps(g)
    v = a.viewport
    path = output_path(g, "animation", ext)
    status!(g, "Rendering $(length(a.frames)) frames to $ext…")
    @async try
        sleep(0.05)  # let the status line draw before rendering blocks the window
        fig, field, subtitle = export_figure(g; limits = (v.lon_min, v.lon_max, v.lat_min, v.lat_max),
                                             frame = 1, size = (1200, 900))
        record(fig, path, eachindex(a.frames); framerate = fps) do i
            field[] = a.frames[i]
            subtitle[] = _frame_time(g, a, i)
        end
        _saved!(g, path)
    catch e
        _report_error!(g, "Animation export failed", e)
    end
end

function open_output_folder(g::GUI)
    dir = mkpath(String(_text(g.w.outdir)))
    cmd = Sys.iswindows() ? `explorer $dir` : Sys.isapple() ? `open $dir` : `xdg-open $dir`
    try
        run(cmd; wait = false)
    catch e
        status!(g, "Could not open $dir: $(sprint(showerror, e))"; level = :error)
    end
end

# --- Window layout ---

"""
    build_gui(; size) -> GUI

Lay out the window and wire up its callbacks. Doesn't load any weather data;
`launch` does that once the window is showing.
"""
function build_gui(; size = (1500, 960))
    fig = Figure(; size, fontsize = 12, figure_padding = 10)
    panel = GridLayout(fig[1:2, 1]; valign = :top, tellheight = false, width = PANEL_WIDTH)
    colsize!(fig.layout, 1, Fixed(PANEL_WIDTH))

    r = 0
    row!() = (r += 1)
    label(pos, text; kw...) = Label(pos, text; halign = :left, tellwidth = false, kw...)
    textbox(pos, text = ""; kw...) = Textbox(pos; stored_string = string(text), width = Relative(1),
                                             tellwidth = false, textpadding = (6, 6, 3, 3), kw...)
    menu(pos, options; kw...) = Menu(pos; options, width = Relative(1), tellwidth = false,
                                     textpadding = (8, 8, 3, 3), kw...)
    button(pos, text) = Button(pos; label = text, width = Relative(1), tellwidth = false,
                               padding = (6, 6, 4, 4))
    header(text) = Label(panel[row!(), 1:4], text; font = :bold, fontsize = 13,
                         halign = :left, tellwidth = false, padding = (0, 0, 0, 6))

    Label(panel[row!(), 1:4], "Fallout Dispersion"; fontsize = 20, font = :bold,
          halign = :left, tellwidth = false)
    Label(panel[row!(), 1:4], "NuclearDetonation.jl"; color = MUTED_COLOR,
          halign = :left, tellwidth = false)

    header("Scenario")
    label(panel[row!(), 1], "Dataset")
    dataset = menu(panel[r, 2:4], [(DATASET_PRESETS[k].label, k) for k in DATASET_ORDER])
    label(panel[row!(), 1], "Release")
    release = menu(panel[r, 2:4], [("Bomb release", "bomb"), ("Point release (NPP)", "npp")])
    label(panel[row!(), 1], "Weather")
    weather = menu(panel[r, 2:4], [("Built-in ERA5", "era5"), ("Local ARL files", "arl")])
    label(panel[row!(), 1], "ARL path")
    arl_path = textbox(panel[r, 2:3]; placeholder = "folder or file.ARL")
    arl_load = button(panel[r, 4], "Load ARL")

    label(panel[row!(), 1], "Latitude"); lat = textbox(panel[r, 2])
    label(panel[r, 3], "Longitude");     lon = textbox(panel[r, 4])
    label(panel[row!(), 1], "Start date"); date = textbox(panel[r, 2])
    label(panel[r, 3], "Hour (UTC)");      hour = textbox(panel[r, 4])
    label(panel[row!(), 1], "Duration (h)"); duration = textbox(panel[r, 2])
    label(panel[r, 3], "Particles");         particles = textbox(panel[r, 4], 2500)

    header("Source")
    label(panel[row!(), 1], "Yield (kt)"); yield = textbox(panel[r, 2], 24.0)
    label(panel[r, 3:4], "bomb release only"; color = MUTED_COLOR)
    label(panel[row!(), 1], "Isotopes"); sources = textbox(panel[r, 2:4], "Cs-137 1.0")
    label(panel[row!(), 1], "Release (h)"); release_hours = textbox(panel[r, 2], 1.0)
    label(panel[r, 3], "Stack (m)");        stack = textbox(panel[r, 4], 100.0)
    label(panel[row!(), 1:4],
          "Point release: \"isotope TBq [half-life h]\", comma separated. Presets: " *
          join(sort(collect(keys(ISOTOPE_HALFLIVES))), ", ") * ".";
          color = MUTED_COLOR, fontsize = 11, word_wrap = true, width = PANEL_WIDTH)

    run = Button(panel[row!(), 1:4]; label = "Run Simulation", width = Relative(1),
                 tellwidth = false, font = :bold, fontsize = 14, padding = (8, 8, 8, 8),
                 buttoncolor = RGBf(0.80, 0.88, 0.98))
    progress = Observable(0.0)
    Box(panel[row!(), 1:4]; color = (:black, 0.08), strokevisible = false, height = 6)
    Box(panel[r, 1:4]; color = DOMAIN_COLOR, strokevisible = false, height = 6, halign = :left,
        width = lift(p -> Relative(clamp(p, 0, 100) / 100), progress))
    status = Observable("Starting…")
    status_color = Observable(MUTED_COLOR)
    Label(panel[row!(), 1:4], status; color = status_color, halign = :left, valign = :top,
          word_wrap = true, width = PANEL_WIDTH, justification = :left)
    prediction = Observable("")
    prediction_color = Observable(MUTED_COLOR)
    Label(panel[row!(), 1:4], prediction; color = prediction_color, font = :bold,
          halign = :left, word_wrap = true, width = PANEL_WIDTH)

    header("Results")
    results_text = Observable("")
    Label(panel[row!(), 1:4], results_text; halign = :left, tellwidth = false)
    label(panel[row!(), 1], "Units"); units = menu(panel[r, 2], _unit_options(DOSE_UNITS))
    contours = Toggle(panel[r, 3]; active = true, halign = :right)
    label(panel[r, 4], "Contours")
    plume = Toggle(panel[row!(), 1]; active = false, halign = :right)
    label(panel[r, 2], "Plume")
    obs = Toggle(panel[r, 3]; active = false, halign = :right)
    label(panel[r, 4], "Observations")
    csv = button(panel[row!(), 1:2], "Export CSV")
    png = button(panel[r, 3:4], "Save map PNG")

    header("Animation")
    label(panel[row!(), 1], "Level"); level = menu(panel[r, 2:4], [("Run a simulation first", -1)])
    play = button(panel[row!(), 1], "Play")
    back = button(panel[r, 2], "◀")
    fwd = button(panel[r, 3], "▶")
    label(panel[r, 4], "FPS"; halign = :center)
    slider = Slider(panel[row!(), 1:3]; range = 1:1, startvalue = 1)
    fps = textbox(panel[r, 4], 2)
    time_text = Observable("")
    Label(panel[row!(), 1:4], time_text; halign = :left, tellwidth = false)
    gif = button(panel[row!(), 1:2], "Export GIF")
    mp4 = button(panel[r, 3:4], "Export MP4")

    header("Output")
    outdir = textbox(panel[row!(), 1:3], default_output_dir())
    open_dir = button(panel[r, 4], "Open folder")

    rowgap!(panel, 5)
    colgap!(panel, 6)
    colsize!(panel, 1, Fixed(78))
    colsize!(panel, 3, Fixed(72))

    # Map
    ax = map_axis(fig[1, 2]; title = "Click the map to place the release")
    deactivate_interaction!(ax, :rectanglezoom)
    domain_pts = Observable([Point2f(NaN, NaN)])
    dl = lines!(ax, domain_pts; color = DOMAIN_COLOR, linestyle = :dash, linewidth = 1.5)
    translate!(dl, 0, 0, 30)
    draw_npp!(ax)
    release_pt = Observable(Point2f(NaN, NaN))
    draw_release!(ax, release_pt)

    cb_layout = GridLayout(fig[1, 2]; tellwidth = false, tellheight = false,
                           halign = :right, valign = :top, alignmode = Outside(14))
    cb_box = Box(cb_layout[1, 1]; color = (:white, 0.92), strokecolor = (:black, 0.3),
                 cornerradius = 4, visible = false)
    cb = Colorbar(cb_layout[1, 1]; colormap = PLUME_COLORMAP, limits = (0.0, 1.0),
                  height = 260, width = 14, alignmode = Outside(10),
                  labelsize = 11, ticklabelsize = 11)
    cb.blockscene.visible[] = false

    cursor_text = Observable("")
    Label(fig[2, 2], cursor_text; halign = :left, color = MUTED_COLOR, tellwidth = false)
    Label(fig[2, 2], "Left-click: place release or pick an NPP site · Scroll: zoom · " *
                     "Right-drag: pan · Ctrl+click: reset view";
          halign = :right, color = MUTED_COLOR, tellwidth = false)

    w = (; dataset, release, weather, arl_path, arl_load, lat, lon, date, hour, duration,
           particles, yield, sources, release_hours, stack, run, units, contours, plume, obs,
           csv, png, level, play, back, fwd, slider, fps, gif, mp4, outdir, open_dir)

    g = GUI(fig, ax, w, Observable(false), progress, status, status_color,
            prediction, prediction_color, results_text, time_text, cursor_text,
            release_pt, domain_pts, "nancy", nothing, nothing, nothing, Any[], nothing,
            Any[], nothing, nothing, nothing, Observable(zeros(Float32, 1, 1)), nothing,
            Observable(1), false, cb, cb_box)

    # --- Callbacks ---
    on(_ -> run_clicked!(g), run.clicks)
    on(g.busy) do busy
        run.label[] = busy ? "Working…" : "Run Simulation"
    end
    on(dataset.selection) do key
        key === nothing || key == g.dataset || load_dataset!(g, key)
    end
    on(weather.selection) do src
        src === nothing && return
        g.observations = nothing
        foreach(p -> delete!(g.ax, p), g.obs_plots); empty!(g.obs_plots)
        obs.active[] = false
        if src == "arl" && g.arl_meta === nothing
            status!(g, "Enter an ARL file or folder path and press Load ARL")
        end
        refresh_domain!(g)
    end
    on(_ -> load_arl!(g), arl_load.clicks)
    on(arl_path.stored_string) do _
        load_arl!(g)
    end
    for tb in (lat, lon)
        on(tb.stored_string) do _
            la, lo = tryparse(Float64, _text(g.w.lat)), tryparse(Float64, _text(g.w.lon))
            (la === nothing || lo === nothing) || (release_pt[] = Point2f(lo, la))
        end
    end
    register_interaction!(ax, :place_release) do event, axis
        event isa Makie.MouseEvent || return Makie.Consume(false)
        event.type === Makie.MouseEventTypes.leftclick || return Makie.Consume(false)
        ispressed(axis.scene, Keyboard.left_control) && return Makie.Consume(false)
        on_map_click!(g, event.data[1], event.data[2])
        return Makie.Consume(false)
    end
    on(events(fig).mouseposition) do _
        if Makie.is_mouseinside(ax.scene)
            x, y = mouseposition(ax.scene)
            cursor_text[] = "$(_lat_label(round(y, digits = 3)))  $(_lon_label(round(x, digits = 3)))"
        end
        return Makie.Consume(false)
    end

    on(units.selection) do _
        update_results_text!(g)
        rebuild_legend!(g)
    end
    on(_ -> rebuild_legend!(g), contours.active)
    on(show -> toggle_observations!(g, show), obs.active)
    on(plume.active) do show
        show || stop_playback!(g)
        visible = show && g.anim !== nothing
        cb.blockscene.visible[] = visible
        cb_box.visible[] = visible
    end
    on(_ -> export_csv!(g), csv.clicks)
    on(_ -> save_png!(g), png.clicks)

    on(level.selection) do lv
        lv === nothing || g.result === nothing || load_animation!(g, lv)
    end
    on(slider.value) do i
        g.frame[] = i
        show_frame!(g, i)
    end
    on(_ -> g.playing ? stop_playback!(g) : start_playback!(g), play.clicks)
    on(_ -> step_frame!(g, -1), back.clicks)
    on(_ -> step_frame!(g, 1), fwd.clicks)
    on(_ -> export_animation!(g, "gif"), gif.clicks)
    on(_ -> export_animation!(g, "mp4"), mp4.clicks)
    on(_ -> open_output_folder(g), open_dir.clicks)

    return g
end

"""
    launch(; dataset = "nancy", wait = true) -> GUI

Open the GUI window and load the built-in ERA5 dataset in the background.
With `wait = true`, blocks until the window is closed.
"""
function launch(; dataset::String = "nancy", wait::Bool = true)
    GLMakie.activate!(; title = "NuclearDetonation.jl — Fallout Dispersion", focus_on_show = true)
    g = build_gui()
    screen = display(g.fig)
    g.dataset = ""  # force a load of the requested dataset
    g.w.dataset.i_selected[] = findfirst(==(dataset), DATASET_ORDER)
    load_dataset!(g, dataset)
    wait && Base.wait(screen)
    return g
end
