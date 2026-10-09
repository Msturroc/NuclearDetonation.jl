#!/usr/bin/env julia
# Thin driver: plot the Nancy GPU calibrated fit using the committed
# examples/calibration_us_tests/viz_plot_fit.jl UNCHANGED.
#
# viz_plot_fit.jl only needs: TEST_NAME, TURB_SCHEME, TEST_CONFIG(.source_lat,
# .source_lon, .label, .yield_kt), OBS(.dose_rate_contours,.toa_contours),
# LON_GRID, LAT_GRID, gaussian_smooth, and the LAST_DOSE_SMOOTH /
# LAST_MODEL_SNAPSHOTS / LAST_SNAPSHOT_HOURS refs. We build all of those from a
# single GPU forward sim at the calibrated vector, then include the plotter.
#
# Run from repo root:  julia --project gpu_transport/runners/viz_nancy_gpu.jl

ENV["MAX_EVALS"] = "0"
ENV["N_PARTICLES"] = get(ENV, "N_PARTICLES", "10000")
using Random, Statistics, Printf, StaticArrays, CUDA
using NuclearDetonation
using NuclearDetonation.Transport

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(ROOT, "examples", "nancy_cmaes_particle_size.jl"))
include(joinpath(ROOT, "gpu_transport", "host_shadow_v2.jl"))
include(joinpath(ROOT, "gpu_transport", "met_upload.jl"))
include(joinpath(ROOT, "gpu_transport", "gpu_kernel_v2.jl"))

const GPU_WINDOWS = load_nancy_gpu_windows()

# Calibrated vector (physical units)
const BEST = get(ENV, "BEST_FILE", joinpath(ROOT, "gpu_transport", "artifacts", "gpu_nancy_corrected_best.txt"))
params = Float64[]
for ln in eachline(BEST)
    (startswith(ln, "#") || isempty(strip(ln))) && continue
    push!(params, parse(Float64, split(ln, '\t')[2]))
end

# One GPU forward sim → dose field + hourly snapshots
final_dep, hourly_dep, n_alive = run_gpu_shadow(params, UInt64(0xCA11B0A7); windows = GPU_WINDOWS)
CUDA.synchronize()
model_snapshots = [Float64.(view(hourly_dep, :, :, h)) for h in 1:12]
final_dose = model_snapshots[end]
dose_smooth_field = gaussian_smooth(final_dose .* DOSE_FACTOR, params[20])

# Globals expected by viz_plot_fit.jl
const TEST_NAME    = get(ENV, "VIZ_NAME", "nancy_gpu")
const TURB_SCHEME  = :OU
const TEST_CONFIG  = (source_lat = SOURCE_LAT, source_lon = SOURCE_LON,
                      label = "Nancy (GPU, corrected loss)", yield_kt = 24.0)
const OBS          = NANCY_OBS
const LAST_DOSE_SMOOTH     = Ref{Union{Nothing,Matrix{Float64}}}(dose_smooth_field)
const LAST_MODEL_SNAPSHOTS = Ref{Union{Nothing,Vector{Matrix{Float64}}}}(model_snapshots)
const LAST_SNAPSHOT_HOURS  = Ref{Union{Nothing,Vector{Float64}}}(Float64.(1:12))

include(joinpath(ROOT, "examples", "calibration_us_tests", "viz_plot_fit.jl"))
