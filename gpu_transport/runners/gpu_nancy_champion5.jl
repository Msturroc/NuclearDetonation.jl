#!/usr/bin/env julia
# Nancy GPU calibration with champion5 (best_global_optimiser)
# ============================================================
# Same forward model and corrected loss as gpu_nancy_corrected.jl
# (nancy_objective.jl), optimised with champion5: surrogate-assisted CMA-ES with
# IPOP restarts, which beat lq-CMA-ES at d2-d20 and BIPOP-CMA-ES 3-4x on the
# official COCO suite at budgets around 1e4*d, and is built for expensive
# objectives.
#
# Differences from the BIPOP runner:
#   - One fixed seed for every evaluation (common random numbers), so the
#     objective is deterministic and the surrogate models a fixed landscape.
#     The BIPOP runner draws a new seed per generation, so its recorded best is a
#     max over noise; compare runs with eval_nancy_params.jl instead.
#   - h_diff_scale and tmix_scale are held fixed: the forward model unpacks them
#     but never uses them, so they would only add two dead dimensions.
#
#   MAX_EVALS=6000 N_PARTICLES=10000 RUN_TAG=gpu_nancy_champion5 \
#       julia --project gpu_transport/runners/gpu_nancy_champion5.jl
#
# Env: X0_FILE (start point, default the previous GPU best), OPT_SEED (optimiser
# RNG seed), OBJ_SEED (the fixed forward-model seed), OPTIMISER_DIR,
# VARIANT (champion5 | c5-mbh | c5-alloc | c5-mbh-alloc, default champion5),
# STOP_EVALSCALED=0 to count the stop windows in generations rather than
# stretching them by the surrogate's savings (champion5 was tuned at ~1e4*d
# evaluations; at a few hundred per dimension the stretched stagnation window can
# outlast the whole budget, so a stalled run never restarts).

include(joinpath(@__DIR__, "nancy_objective.jl"))

const OPT_DIR = get(ENV, "OPTIMISER_DIR", "/home/marc/best_global_optimiser")
include(joinpath(OPT_DIR, "src", "driver.jl"))
const VARIANTS = Dict{String,Any}()
register!(name, v) = (VARIANTS[name] = v)
include(joinpath(OPT_DIR, "bench", "variants_lever5.jl"))   # champion5 and its restart levers

const RUN_TAG  = get(ENV, "RUN_TAG", "gpu_nancy_champion5")
const VARIANT  = get(ENV, "VARIANT", "champion5")
const OPT_SEED = parse(Int, get(ENV, "OPT_SEED", "1"))
const OBJ_SEED = parse(UInt64, get(ENV, "OBJ_SEED", "0x5eedca11"))
# Each evaluation averages the loss over this many fixed seeds. With one seed the
# objective is deterministic but the optimiser can fit that seed's particle noise
# (held-out scores drop 3-5 points); averaging trades evaluations for robustness.
const N_OBJ_SEEDS = parse(Int, get(ENV, "OBJ_SEEDS", "1"))
const OBJ_SEED_LIST = [OBJ_SEED + UInt64(k - 1) * 0x9e3779b97f4a7c15 for k in 1:N_OBJ_SEEDS]
const X0_FILE  = get(ENV, "X0_FILE", joinpath(ROOT, "gpu_transport", "artifacts", "gpu_nancy_corrected_best.txt"))

function read_params(path)
    vals = Dict{String,Float64}()
    for ln in eachline(path)
        (startswith(ln, "#") || isempty(strip(ln))) && continue
        k, v = split(ln, '\t')
        vals[strip(k)] = parse(Float64, v)
    end
    return clamp.([vals[n] for n in PARAM_NAMES], LB, UB)
end

const INERT  = [findfirst(==(n), PARAM_NAMES) for n in ("h_diff_scale", "tmix_scale")]
const ACTIVE = setdiff(1:N_DIM, INERT)
const X0_ENC = LOGSP.encode(isfile(X0_FILE) ? read_params(X0_FILE) : copy(WARM_START_PARAMS))
const LB_A, UB_A = LOGSP.LB_S[ACTIVE], LOGSP.UB_S[ACTIVE]

full_params(xa) = (x = copy(X0_ENC); x[ACTIVE] .= xa; LOGSP.decode(x))

mutable struct Tracker
    evals::Int
    best::Float64
    best_params::Vector{Float64}
    best_r::CorrResult
    trace::Vector{Tuple{Int,Float64}}
end
const TR = Tracker(0, Inf, full_params(X0_ENC[ACTIVE]), FAILED_CORR, Tuple{Int,Float64}[])

const BEST_FILE    = joinpath(ROOT, "gpu_transport", "artifacts", "$(RUN_TAG)_best.txt")
const RESULTS_FILE = joinpath(ROOT, "gpu_transport", "artifacts", "$(RUN_TAG)_results.txt")
const TRACE_FILE   = joinpath(ROOT, "gpu_transport", "artifacts", "$(RUN_TAG)_trace.csv")

function write_params(io, params, r, loss)
    for (j, pname) in enumerate(PARAM_NAMES)
        println(io, "$(pname)\t$(params[j])")
    end
    println(io, "# loss\t$(loss)")
    for k in (:fms, :shape, :bearing, :extent, :toa)
        println(io, "# $(k)\t$(getfield(r, k))")
    end
end

function objective(xa)
    params = full_params(xa)
    r = try
        rs = [rho_core_corrected(params, s) for s in OBJ_SEED_LIST]
        length(rs) == 1 ? rs[1] :
            CorrResult((mean(getfield(x, k) for x in rs) for k in fieldnames(CorrResult))...)
    catch e
        @warn "GPU eval failed" exception = (e, catch_backtrace())
        FAILED_CORR
    end
    TR.evals += 1
    if r.loss < TR.best
        TR.best, TR.best_params, TR.best_r = r.loss, params, r
        push!(TR.trace, (TR.evals, r.loss))
        open(io -> write_params(io, params, r, r.loss), BEST_FILE, "w")
        @printf("  eval %5d  score %.2f%%  FMS=%.3f shp=%.3f bear=%.3f ext=%.2f toa=%.3f\n",
                TR.evals, 100 * (1 - r.loss), r.fms, r.shape, r.bearing, r.extent, r.toa)
        flush(stdout)
    end
    return r.loss
end

println("\n" * "="^70)
println("NANCY GPU champion5 — CORRECTED LOSS  (budget $(MAX_EVALS_C) evals, d=$(length(ACTIVE)))")
println("  start: $(basename(X0_FILE))  opt seed $(OPT_SEED)  objective seeds $(N_OBJ_SEEDS) from $(repr(OBJ_SEED))")
println("  windows: $(length(GPU_WINDOWS))  bridge=$(_bridge_met_files())  start_time_idx=$(_start_time_idx())")
println("="^70)

t0 = time()
function optimiser_options()
    o = VARIANTS[VARIANT](length(ACTIVE))
    get(ENV, "STOP_EVALSCALED", "1") == "0" || return o
    kw = Dict{Symbol,Any}(k => getfield(o, k) for k in fieldnames(CMAOpts))
    kw[:stop_evalscaled] = false
    return CMAOpts(; kw...)
end
const OPTS = optimiser_options()
println("  optimiser: $(VARIANT)  stop_evalscaled=$(OPTS.stop_evalscaled)")

res = optimise(Problem("nancy", objective, LB_A, UB_A, 0.0), MAX_EVALS_C, OPTS;
               rng = MersenneTwister(OPT_SEED), x0 = (X0_ENC[ACTIVE] .- LB_A) ./ (UB_A .- LB_A))
elapsed = time() - t0

@printf("\nDone: %d evals, %d restarts, %.1f min. Best score %.2f%%\n",
        TR.evals, res.restarts, elapsed / 60, 100 * (1 - TR.best))
open(RESULTS_FILE, "w") do io
    println(io, "# Nancy GPU $(VARIANT) (corrected loss) stop_evalscaled=$(OPTS.stop_evalscaled)")
    println(io, "# evals=$(TR.evals) restarts=$(res.restarts) wall_min=$(round(elapsed / 60, digits = 2))")
    println(io, "# opt_seed=$(OPT_SEED) obj_seed=$(repr(OBJ_SEED)) obj_seeds=$(N_OBJ_SEEDS) x0=$(basename(X0_FILE)) n_particles=$(get(ENV, "N_PARTICLES", "10000"))")
    println(io, "# windows=$(length(GPU_WINDOWS)) bridge=$(_bridge_met_files()) start_time_idx=$(_start_time_idx())")
    println(io, "# score=$(100 * (1 - TR.best))")
    write_params(io, TR.best_params, TR.best_r, TR.best)
end
open(TRACE_FILE, "w") do io
    println(io, "eval,best_loss")
    foreach(((e, l),) -> println(io, "$e,$l"), TR.trace)
end
println("Saved $(BEST_FILE)\nSaved $(RESULTS_FILE)")
