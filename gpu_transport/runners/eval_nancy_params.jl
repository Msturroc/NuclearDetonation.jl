#!/usr/bin/env julia
# Re-score Nancy parameter files on the GPU at a fixed set of seeds, so fits from
# different runs (and different optimisers' noisy "best" values) compare fairly.
#
#   julia --project gpu_transport/runners/eval_nancy_params.jl <best.txt>...
#
# Env: N_SEEDS (default 8), PARITY=1 to also check the CPU shadow against the GPU
# kernel for the first file, BRIDGE_MET_FILES / NANCY_START_TIME_IDX as in
# host_shadow_v2.jl.

include(joinpath(@__DIR__, "nancy_objective.jl"))

function read_params(path)
    vals = Dict{String,Float64}()
    for ln in eachline(path)
        (startswith(ln, "#") || isempty(strip(ln))) && continue
        k, v = split(ln, '\t')
        vals[strip(k)] = parse(Float64, v)
    end
    return [vals[n] for n in PARAM_NAMES]
end

const EVAL_SEEDS = [UInt64(0x9E3779B97F4A7C15) * UInt64(k) for k in 1:parse(Int, get(ENV, "N_SEEDS", "8"))]

println("\nwindows: ", length(GPU_WINDOWS), "  bridge=", _bridge_met_files(),
        "  start_time_idx=", _start_time_idx())
for path in ARGS
    p = read_params(path)
    rs = [rho_core_corrected(p, s) for s in EVAL_SEEDS]
    sc = [100 * (1 - r.loss) for r in rs]
    @printf("%-60s score %.2f ± %.2f %%  (min %.2f)  fms %.3f shape %.3f bear %.3f ext %.3f toa %.3f\n",
            basename(path), mean(sc), std(sc), minimum(sc),
            mean(r.fms for r in rs), mean(r.shape for r in rs), mean(r.bearing for r in rs),
            mean(r.extent for r in rs), mean(r.toa for r in rs))
end

if get(ENV, "PARITY", "0") == "1" && !isempty(ARGS)
    p = read_params(ARGS[1])
    seed = EVAL_SEEDS[1]
    dep_gpu, _, _ = run_gpu_shadow(p, seed; windows = GPU_WINDOWS)
    dep_cpu, _, _ = run_host_shadow(p, seed)
    rel = maximum(abs.(dep_gpu .- dep_cpu)) / max(maximum(abs.(dep_cpu)), eps(Float32))
    @printf("CPU shadow vs GPU kernel: max |Δ| / max = %.3e  (sum cpu %.4e, gpu %.4e)\n",
            rel, sum(dep_cpu), sum(dep_gpu))
end
