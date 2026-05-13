#!/usr/bin/env julia
# NuclearDetonation.jl Web GUI
# ============================
# Local web application for running fallout dispersion simulations.
#
# Usage:
#   cd web && julia --threads=2 --project app.jl
#   julia --threads=2 --project=web web/app.jl
#
# Opens browser at http://localhost:9000
#
# The --threads=2 flag keeps the server responsive while
# a simulation runs. It works without threads but the UI
# will freeze until the simulation completes.

using Pkg
Pkg.instantiate()

println("="^60)
println("  NuclearDetonation.jl — Fallout Dispersion GUI")
println("="^60)
if Threads.nthreads() < 2
    println("\n  Tip: start with --threads=2 for a responsive UI")
    println("  julia --threads=2 --project app.jl\n")
end

println("Loading packages...")
include(joinpath(@__DIR__, "src", "arl_reader.jl"))
include(joinpath(@__DIR__, "src", "arl_converter.jl"))
include(joinpath(@__DIR__, "src", "simulation.jl"))
include(joinpath(@__DIR__, "src", "contours.jl"))
include(joinpath(@__DIR__, "src", "animation.jl"))
include(joinpath(@__DIR__, "src", "prediction.jl"))
include(joinpath(@__DIR__, "src", "database.jl"))
include(joinpath(@__DIR__, "src", "server.jl"))

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
Prediction.load_prediction_models!(joinpath(@__DIR__, "models"))

println("Pre-loading ERA5 meteorological data...")
preload_era5!(progress_callback = (pct, msg) -> println("  [$pct%] $msg"))

println("\nReady — starting web server...")
start_server(port=9000, open_browser=true)
