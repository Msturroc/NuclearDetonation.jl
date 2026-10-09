#!/usr/bin/env julia
# Plot a US-test GPU calibrated fit using the committed viz_plot_fit.jl unchanged.
# Includes cmaes_calibration.jl for the exact per-test setup (OBS, grids, refs),
# runs ONE GPU forward sim at the calibrated best vector, populates the LAST_*
# refs, then includes the plotter.
#   julia --project gpu_transport/runners/viz_us_gpu.jl {trinity|harry|smallboy|doppler} OU
length(ARGS) >= 1 || error("usage: viz_us_gpu.jl {trinity|harry|smallboy|doppler} [OU|RW]")
ENV["MAX_EVALS"] = "0"
ENV["N_PARTICLES"] = get(ENV, "N_PARTICLES", "10000")
using Random, Statistics, Printf, StaticArrays, CUDA
using NuclearDetonation
using NuclearDetonation.Transport
const ROOT = "/home/marc/NuclearDetonation.jl"
include(joinpath(ROOT, "examples", "calibration_us_tests", "cmaes_calibration.jl"))
include(joinpath(ROOT, "gpu_transport", "host_shadow_v2.jl"))
include(joinpath(ROOT, "gpu_transport", "met_upload.jl"))
include(joinpath(ROOT, "gpu_transport", "gpu_kernel_v2.jl"))
const GPU_WINDOWS = load_nancy_gpu_windows()

const BEST = joinpath(ROOT, "gpu_transport", "artifacts", "gpu_$(TEST_NAME)_corrected_best.txt")
bestp = Float64[]
for ln in eachline(BEST)
    (startswith(ln, "#") || isempty(strip(ln))) && continue
    push!(bestp, parse(Float64, split(ln, '\t')[2]))
end

final_dep, hourly_dep, n_alive = run_gpu_shadow(bestp, UInt64(0xCA11B0A7);
                                                n_hours = SIM_HOURS, windows = GPU_WINDOWS)
CUDA.synchronize()
ms = [Float64.(view(hourly_dep, :, :, h)) for h in 1:SIM_HOURS]
LAST_DOSE_SMOOTH[]     = gaussian_smooth(ms[end] .* DOSE_FACTOR, bestp[20])
LAST_MODEL_SNAPSHOTS[] = ms
LAST_SNAPSHOT_HOURS[]  = Float64.(1:SIM_HOURS)

include(joinpath(ROOT, "examples", "calibration_us_tests", "viz_plot_fit.jl"))
