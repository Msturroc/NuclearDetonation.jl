module NuclearDetonationGUI

# All `using` statements and `include`s happen at module scope so PackageCompiler
# bakes them into the sysimage. Don't move them into julia_main() — runtime
# `include()` from a compiled exe can't find Base64 et al.

using NuclearDetonation
using NuclearDetonation.Transport
using NCDatasets
using StaticArrays
using Random
using Dates
using HTTP
using JSON3
using Contour
using Base64
using XGBoost

const _SRC = @__DIR__

include(joinpath(_SRC, "arl_reader.jl"))
include(joinpath(_SRC, "arl_converter.jl"))
include(joinpath(_SRC, "simulation.jl"))
include(joinpath(_SRC, "contours.jl"))
include(joinpath(_SRC, "animation.jl"))
include(joinpath(_SRC, "prediction.jl"))
include(joinpath(_SRC, "database.jl"))
include(joinpath(_SRC, "server.jl"))

# Locate the bundled web/ root. Compiled layout: exe in <app>/bin/, assets in
# <app>/web/. Source layout: this file is in <repo>/web/src/.
function _web_dir()
    candidate = dirname(_SRC)
    isdir(joinpath(candidate, "public_react")) && return candidate
    return joinpath(dirname(Sys.BINDIR), "web")
end

function julia_main()::Cint
    try
        # PackageCompiler's C wrapper doesn't always set DEPOT_PATH correctly
        # for JLL artifacts. The .bat launcher sets JULIA_DEPOT_PATH, but also
        # push the app depot in case the exe is run directly.
        app_depot = joinpath(dirname(Sys.BINDIR), "share", "julia")
        if isdir(app_depot) && app_depot ∉ DEPOT_PATH
            pushfirst!(DEPOT_PATH, app_depot)
        end

        # prediction.jl references Main.ARLReader.* (legacy of the dev path
        # where everything lives at Main scope). In the compiled bundle this
        # module is the parent, so alias it into Main for back-compat.
        @eval Main const ARLReader = $(@__MODULE__).ARLReader

        web_dir = _web_dir()
        ENV["NUCLEAR_DETONATION_WEB_DIR"] = web_dir

        println("="^60)
        println("  NuclearDetonation.jl — Fallout Dispersion GUI")
        println("="^60)

        if get(ENV, "NUCDET_DISABLE_DB", "0") == "1"
            println("Simulation history disabled (NUCDET_DISABLE_DB=1)")
        else
            println("Connecting to PostgreSQL...")
            try
                db_init()
            catch e
                @warn "PostgreSQL unavailable — simulation history will not be recorded" exception=e
            end
        end

        println("Loading impact prediction models...")
        Prediction.load_prediction_models!(joinpath(web_dir, "models"))

        println("Pre-loading ERA5 meteorological data...")
        preload_era5!(progress_callback = (pct, msg) -> println("  [$pct%] $msg"))

        println("\nReady — starting web server...")
        start_server(port=9000, open_browser=true)
    catch e
        @error "Fatal error" exception=(e, catch_backtrace())
        println("\nPress Enter to exit...")
        readline()
        return 1
    end
    return 0
end

end # module
