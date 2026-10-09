#!/usr/bin/env julia
# §8.2/§8.3 gate for Smoky (23-param): compare CPU rho_core vs GPU forward sim
# at the SAME seed vector. If FMS/extent/toa agree (within MC noise) the §3
# geometry is correct; a big FMS gap means the 23-param release is wrong.
ENV["MAX_EVALS"] = "0"
ENV["N_PARTICLES"] = "10000"
using Random, Statistics, Printf, StaticArrays, CUDA
using NuclearDetonation
using NuclearDetonation.Transport
const ROOT = "/home/marc/NuclearDetonation.jl"
include(joinpath(ROOT, "examples", "smoky_example", "smoky_cmaes_particle_size.jl"))  # CPU rho_core + globals
include(joinpath(ROOT, "gpu_transport", "host_shadow_v2.jl"))
include(joinpath(ROOT, "gpu_transport", "met_upload.jl"))
include(joinpath(ROOT, "gpu_transport", "gpu_kernel_v2.jl"))
const GPU_WINDOWS = load_nancy_gpu_windows()

x0 = isnothing(load_checkpoint_params("ou")) ? copy(WARM_START_PARAMS) : load_checkpoint_params("ou")
println("seed heights params[21..23] = ", round.(x0[21:23], digits=1))
println("seed frac_lower,middle params[6,7] = ", round.(x0[6:7], digits=3))

seed = UInt64(0xBEEF)

# ---- CPU reference (smoky upstream rho_core; 1000 particles, Float64) ----
cpu = rho_core(x0, :OU, seed)
@printf("CPU : fms=%.3f shape=%.3f extent=%.3f toa=%.3f  (loss=%.3f)\n",
        cpu.fms, cpu.shape, cpu.extent, cpu.toa, cpu.loss)

# ---- GPU forward sim at same vector, same scoring (10k particles, Float32) ----
final_dep, hourly_dep, n_alive = run_gpu_shadow(x0, seed; windows = GPU_WINDOWS)
CUDA.synchronize()
model_snapshots = [Float64.(view(hourly_dep, :, :, h)) for h in 1:12]
final_dose = model_snapshots[end]
dose_smooth = gaussian_smooth(final_dose .* DOSE_FACTOR, x0[20])
fms_scores = Float64[]
for (dr, m) in OBS_MASKS
    sum(m) == 0 && (push!(fms_scores, 0.0); continue)
    mm = dose_smooth .>= dr
    push!(fms_scores, sum(mm .| m) > 0 ? sum(mm .& m)/sum(mm .| m) : 0.0)
end
gpu_fms = exp(mean(log(max(s,0.005)) for s in fms_scores))
@printf("GPU : fms=%.3f  n_alive=%d  sum(dose_mRh)=%.1f  max(dose_mRh)=%.3f\n",
        gpu_fms, n_alive, sum(final_dose .* DOSE_FACTOR), maximum(final_dose .* DOSE_FACTOR))
println("per-contour FMS (GPU): ", round.(fms_scores, digits=3))
println("obs contour levels: ", round.(sort(collect(keys(OBS_MASKS))), digits=2))
