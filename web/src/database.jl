# PostgreSQL persistence layer for simulation runs
#
# Stores simulation parameters, results and metadata so that
# past runs survive server restarts and can be queried/compared.
# Completed runs are cached: re-running identical parameters
# returns stored results instantly without recomputing.
#
# Set NUCDET_DISABLE_DB=1 to skip all DB work (used by the standalone
# Windows installer build, which ships without a Postgres server).

using Dates

const DB_DISABLED = get(ENV, "NUCDET_DISABLE_DB", "0") == "1"

if DB_DISABLED
    # No-op stubs so server.jl call sites work unchanged.
    db_init() = nothing
    db_insert_run(_params) = nothing
    db_complete_run(_id; kwargs...) = nothing
    db_fail_run(_id, _msg) = nothing
    db_find_cached_run(_params) = nothing
    db_get_run(_id) = nothing
    db_run_count() = 0
    db_list_runs(; limit::Int=0, offset::Int=0) = (;
        id = Int[],
        created_at = String[],
        dataset = Union{String,Missing}[],
        release_mode = String[],
        weather_source = String[],
        latitude = Float64[],
        longitude = Float64[],
        start_date = String[],
        start_hour = Int[],
        duration_hours = Int[],
        n_particles = Int[],
        yield_kt = Union{Float64,Missing}[],
        activity_tbq = Union{Float64,Missing}[],
        isotope = Union{String,Missing}[],
        status = String[],
        peak_dose = Union{Float64,Missing}[],
        dose_units = Union{String,Missing}[],
        n_events = Union{Int,Missing}[],
        elapsed_seconds = Union{Float64,Missing}[],
    )
else

using LibPQ

const DB_CONNECTION_STRING = get(ENV, "NUCDET_DB_URL",
    "host=localhost dbname=nucleardetonation user=marc password=nuclear_dev")

# Hold a single connection (reconnect on failure)
const _db_conn = Ref{Union{LibPQ.Connection, Nothing}}(nothing)

function db_connect()
    if _db_conn[] === nothing || !LibPQ.isopen(_db_conn[])
        _db_conn[] = LibPQ.Connection(DB_CONNECTION_STRING)
    end
    return _db_conn[]
end

function db_init()
    conn = db_connect()

    execute(conn, """
        CREATE TABLE IF NOT EXISTS simulation_runs (
            id              SERIAL PRIMARY KEY,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            dataset         TEXT,
            release_mode    TEXT NOT NULL,
            weather_source  TEXT NOT NULL,
            latitude        DOUBLE PRECISION NOT NULL,
            longitude       DOUBLE PRECISION NOT NULL,
            start_date      TEXT NOT NULL,
            start_hour      INTEGER NOT NULL,
            duration_hours  INTEGER NOT NULL,
            n_particles     INTEGER NOT NULL,
            yield_kt        DOUBLE PRECISION,
            activity_tbq    DOUBLE PRECISION,
            stack_height_m  DOUBLE PRECISION,
            isotope         TEXT,
            release_duration_hours DOUBLE PRECISION,
            arl_dir         TEXT,
            status          TEXT NOT NULL DEFAULT 'running',
            peak_dose       DOUBLE PRECISION,
            dose_units      TEXT,
            n_events        INTEGER,
            elapsed_seconds DOUBLE PRECISION,
            error_message   TEXT,
            geojson_result  TEXT,
            csv_result      TEXT
        );
    """)

    # Add result columns if upgrading from older schema
    for col in ["geojson_result TEXT", "csv_result TEXT"]
        name = split(col)[1]
        try
            execute(conn, "ALTER TABLE simulation_runs ADD COLUMN IF NOT EXISTS $col;")
        catch; end
    end

    execute(conn, """
        CREATE INDEX IF NOT EXISTS idx_runs_created
            ON simulation_runs (created_at DESC);
    """)

    execute(conn, """
        CREATE INDEX IF NOT EXISTS idx_runs_status
            ON simulation_runs (status);
    """)

    println("  PostgreSQL: database ready (simulation_runs table)")
    return conn
end

_null(x) = x === nothing ? missing : x

function db_insert_run(params::Dict)
    conn = db_connect()
    result = execute(conn, """
        INSERT INTO simulation_runs
            (dataset, release_mode, weather_source,
             latitude, longitude, start_date, start_hour,
             duration_hours, n_particles,
             yield_kt, activity_tbq, stack_height_m, isotope,
             release_duration_hours, arl_dir, status)
        VALUES (\$1, \$2, \$3, \$4, \$5, \$6, \$7, \$8, \$9,
                \$10, \$11, \$12, \$13, \$14, \$15, 'running')
        RETURNING id
    """, [
        _null(get(params, "dataset", nothing)),
        params["release_mode"],
        params["weather_source"],
        params["lat"],
        params["lon"],
        params["start_date"],
        params["start_hour"],
        params["duration_hours"],
        params["n_particles"],
        _null(get(params, "yield_kt", nothing)),
        _null(get(params, "activity_tbq", nothing)),
        _null(get(params, "stack_height_m", nothing)),
        _null(get(params, "isotope", nothing)),
        _null(get(params, "release_duration_hours", nothing)),
        _null(get(params, "arl_dir", nothing)),
    ])
    row = LibPQ.columntable(result)
    return row.id[1]
end

function db_complete_run(id::Integer; peak_dose::Float64, dose_units::String,
                         n_events::Int, elapsed_seconds::Float64,
                         geojson::String="", csv::String="")
    conn = db_connect()
    execute(conn, """
        UPDATE simulation_runs
        SET status = 'completed',
            peak_dose = \$1,
            dose_units = \$2,
            n_events = \$3,
            elapsed_seconds = \$4,
            geojson_result = \$5,
            csv_result = \$6
        WHERE id = \$7
    """, [peak_dose, dose_units, n_events, elapsed_seconds, geojson, csv, id])
end

function db_fail_run(id::Integer, error_msg::String)
    conn = db_connect()
    execute(conn, """
        UPDATE simulation_runs
        SET status = 'failed', error_message = \$1
        WHERE id = \$2
    """, [error_msg, id])
end

"""
    db_find_cached_run(params) -> Union{Dict, Nothing}

Look for a completed run with identical simulation parameters.
Returns the full row (including stored geojson/csv) or nothing.
"""
function db_find_cached_run(params::Dict)
    conn = db_connect()

    # Build WHERE clause matching all simulation parameters
    # For nullable fields, use IS NOT DISTINCT FROM to handle NULLs
    result = execute(conn, """
        SELECT id, peak_dose, dose_units, n_events, elapsed_seconds,
               geojson_result, csv_result, created_at
        FROM simulation_runs
        WHERE status = 'completed'
          AND geojson_result IS NOT NULL
          AND geojson_result != ''
          AND release_mode = \$1
          AND weather_source = \$2
          AND latitude = \$3
          AND longitude = \$4
          AND start_date = \$5
          AND start_hour = \$6
          AND duration_hours = \$7
          AND n_particles = \$8
          AND yield_kt IS NOT DISTINCT FROM \$9
          AND activity_tbq IS NOT DISTINCT FROM \$10
          AND stack_height_m IS NOT DISTINCT FROM \$11
          AND isotope IS NOT DISTINCT FROM \$12
          AND release_duration_hours IS NOT DISTINCT FROM \$13
        ORDER BY created_at DESC
        LIMIT 1
    """, [
        params["release_mode"],
        params["weather_source"],
        params["lat"],
        params["lon"],
        params["start_date"],
        params["start_hour"],
        params["duration_hours"],
        params["n_particles"],
        _null(get(params, "yield_kt", nothing)),
        _null(get(params, "activity_tbq", nothing)),
        _null(get(params, "stack_height_m", nothing)),
        _null(get(params, "isotope", nothing)),
        _null(get(params, "release_duration_hours", nothing)),
    ])
    tbl = LibPQ.columntable(result)
    isempty(tbl.id) && return nothing
    return Dict(
        "id" => tbl.id[1],
        "peak_dose" => tbl.peak_dose[1],
        "dose_units" => tbl.dose_units[1],
        "n_events" => tbl.n_events[1],
        "elapsed_seconds" => tbl.elapsed_seconds[1],
        "geojson_result" => tbl.geojson_result[1],
        "csv_result" => tbl.csv_result[1],
        "created_at" => tbl.created_at[1],
    )
end

function db_list_runs(; limit::Int=50, offset::Int=0)
    conn = db_connect()
    result = execute(conn, """
        SELECT id, created_at, dataset, release_mode, weather_source,
               latitude, longitude, start_date, start_hour,
               duration_hours, n_particles,
               yield_kt, activity_tbq, isotope,
               status, peak_dose, dose_units, n_events, elapsed_seconds,
               error_message
        FROM simulation_runs
        ORDER BY created_at DESC
        LIMIT \$1 OFFSET \$2
    """, [limit, offset])
    return LibPQ.columntable(result)
end

function db_get_run(id::Integer)
    conn = db_connect()
    result = execute(conn, """
        SELECT * FROM simulation_runs WHERE id = \$1
    """, [id])
    tbl = LibPQ.columntable(result)
    isempty(tbl.id) && return nothing
    return Dict(String(k) => v[1] for (k, v) in pairs(tbl))
end

function db_run_count()
    conn = db_connect()
    result = execute(conn, "SELECT COUNT(*) AS n FROM simulation_runs")
    return LibPQ.columntable(result).n[1]
end

end  # if DB_DISABLED / else
