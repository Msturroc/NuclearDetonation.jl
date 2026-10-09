#!/usr/bin/env julia
# Plot the Smoky GPU calibrated fit using the committed viz_plot_fit.jl unchanged.
ENV["MAX_EVALS"] = "0"
ENV["N_PARTICLES"] = get(ENV, "N_PARTICLES", "10000")
using Random, Statistics, Printf, StaticArrays, CUDA
using NuclearDetonation
using NuclearDetonation.Transport
const ROOT = "/home/marc/NuclearDetonation.jl"
include(joinpath(ROOT, "examples", "smoky_example", "smoky_cmaes_particle_size.jl"))
include(joinpath(ROOT, "gpu_transport", "host_shadow_v2.jl"))
include(joinpath(ROOT, "gpu_transport", "met_upload.jl"))
include(joinpath(ROOT, "gpu_transport", "gpu_kernel_v2.jl"))
const GPU_WINDOWS = load_nancy_gpu_windows()

const BEST = joinpath(ROOT, "gpu_transport", "artifacts", "gpu_smoky_corrected_best.txt")
params = Float64[]
for ln in eachline(BEST)
    (startswith(ln, "#") || isempty(strip(ln))) && continue
    push!(params, parse(Float64, split(ln, '\t')[2]))
end

final_dep, hourly_dep, n_alive = run_gpu_shadow(params, UInt64(0xCA11B0A7); windows = GPU_WINDOWS)
CUDA.synchronize()
model_snapshots = [Float64.(view(hourly_dep, :, :, h)) for h in 1:12]
final_dose = model_snapshots[end]
dose_smooth_field = gaussian_smooth(final_dose .* DOSE_FACTOR, params[20])

const TEST_NAME    = "smoky_gpu"
const TURB_SCHEME  = :OU
const TEST_CONFIG  = (source_lat = SOURCE_LAT, source_lon = SOURCE_LON,
                      label = "Smoky (GPU, corrected loss)", yield_kt = 44.0)
const OBS          = SMOKY_OBS
const LAST_DOSE_SMOOTH     = Ref{Union{Nothing,Matrix{Float64}}}(dose_smooth_field)
const LAST_MODEL_SNAPSHOTS = Ref{Union{Nothing,Vector{Matrix{Float64}}}}(model_snapshots)
const LAST_SNAPSHOT_HOURS  = Ref{Union{Nothing,Vector{Float64}}}(Float64.(1:12))

include(joinpath(ROOT, "examples", "calibration_us_tests", "viz_plot_fit.jl"))
