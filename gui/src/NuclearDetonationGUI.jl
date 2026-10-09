module NuclearDetonationGUI

# All `using` statements and `include`s happen at module scope so PackageCompiler
# bakes them into the sysimage. Don't move them into julia_main() — runtime
# `include()` from a compiled exe can't find stdlibs et al.

using NuclearDetonation
using NuclearDetonation.Transport
using NCDatasets
using StaticArrays
using Random
using Dates
using Printf
using JSON3
using GLMakie
using GLMakie: Makie
using Tyler
using TileProviders
using Downloads
using XGBoost

export launch

const _SRC = @__DIR__

include(joinpath(_SRC, "arl_reader.jl"))
include(joinpath(_SRC, "arl_converter.jl"))
include(joinpath(_SRC, "simulation.jl"))
include(joinpath(_SRC, "animation.jl"))
include(joinpath(_SRC, "prediction.jl"))
include(joinpath(_SRC, "basemap.jl"))
include(joinpath(_SRC, "observations.jl"))
include(joinpath(_SRC, "tiles.jl"))
include(joinpath(_SRC, "window.jl"))

# Locate the bundled gui/ root (prediction models). Compiled layout: exe in
# <app>/bin/, assets in <app>/gui/. Source layout: this file is in <repo>/gui/src/.
function _gui_dir()
    candidate = dirname(_SRC)
    isdir(joinpath(candidate, "models")) && return candidate
    return joinpath(dirname(Sys.BINDIR), "gui")
end

function load_models!()
    println("Loading impact prediction models...")
    Prediction.load_prediction_models!(joinpath(_gui_dir(), "models"))
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

        println("="^60)
        println("  NuclearDetonation.jl — Fallout Dispersion GUI")
        println("="^60)
        load_models!()
        launch()
    catch e
        @error "Fatal error" exception=(e, catch_backtrace())
        println("\nPress Enter to exit...")
        readline()
        return 1
    end
    return 0
end

end # module
