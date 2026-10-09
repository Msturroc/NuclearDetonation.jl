#!/usr/bin/env julia
# NuclearDetonation.jl GUI
# ========================
# Desktop application for running fallout dispersion simulations.
#
# Usage:
#   julia --threads=2 --project=gui gui/app.jl
#
# The --threads=2 flag runs simulations on a worker thread so the window
# stays responsive. It works without threads but the window will freeze
# until the simulation completes.

using Pkg
Pkg.instantiate()

println("="^60)
println("  NuclearDetonation.jl — Fallout Dispersion GUI")
println("="^60)
if Threads.nthreads() < 2
    println("\n  Tip: start with --threads=2 for a responsive window")
    println("  julia --threads=2 --project=gui gui/app.jl\n")
end

println("Loading packages...")
using NuclearDetonationGUI

NuclearDetonationGUI.load_models!()
NuclearDetonationGUI.launch()
