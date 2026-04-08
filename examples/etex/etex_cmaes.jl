#!/usr/bin/env julia
#= ============================================================================
   ETEX-1 — CMA-ES Parameter Optimisation

   Optimises transport/turbulence parameters against 168 PMCH sampling
   stations across Europe.  ETEX is an inert gas tracer (no settling,
   no particle size) so only the 9 transport parameters are searched.

   Scoring: Gridded FMS (Figure of Merit in Space) — model deposition and
   observed TIC are binned onto a 2° grid and compared via multi-threshold
   spatial overlap (same metric as PREDICT WP3.3).

   Usage:
     julia --threads=auto --project=../.. etex_cmaes.jl
     MAX_EVALS=500 julia --threads=12 --project=../.. etex_cmaes.jl
   ============================================================================ =#

using NuclearDetonation
using NuclearDetonation.Transport
using NCDatasets
using StaticArrays
using Random
using Dates: DateTime, Hour, format, now
using Printf
using Statistics
using LinearAlgebra
using Base.Threads

# ============================================================================
# PATHS AND CONSTANTS
# ============================================================================

const DATA_DIR = joinpath(pkgdir(NuclearDetonation), "data", "etex")
const OUTPUT_DIR = joinpath(@__DIR__, "cmaes_results")
mkpath(OUTPUT_DIR)

const RELEASE_LAT = 48.058
const RELEASE_LON = -2.008
const RELEASE_HEIGHT_M = 15.0    # chimney ~8m, column 5-15m
const RELEASE_MASS = 340e3       # 340 kg PMCH in grams
const RELEASE_DURATION_H = 12    # 12-hour constant release
const SIM_HOURS = 48
const N_PARTICLES = 1500         # per CMA-ES evaluation (reduced from 3000 for speed)
const MAX_EVALS = parse(Int, get(ENV, "MAX_EVALS", "400"))

println("="^70)
println("ETEX-1 — CMA-ES Parameter Optimisation")
println("  Max evals: $MAX_EVALS, Particles: $N_PARTICLES, Threads: $(nthreads())")
println("="^70)

# ============================================================================
# LOAD OBSERVATIONS
# ============================================================================

println("\n1. Loading ETEX observations...")

struct ETEXObs
    station_id::Int
    lat::Float64
    lon::Float64
    time_hours::Float64    # hours after release start (23 Oct 15:00 UTC)
    duration_hours::Float64
    concentration::Float64 # ng/m³
end

function load_observations()
    obs = ETEXObs[]
    # Reference time: 23 Oct 1994, 15:00 UTC (first sampling window)
    ref_dt = DateTime(1994, 10, 23, 15, 0)

    for line in readlines(joinpath(DATA_DIR, "meas-t1.txt"))[3:end]
        parts = split(strip(line))
        length(parts) >= 9 || continue
        yr = parse(Int, parts[1])
        mn = parse(Int, parts[2])
        dy = parse(Int, parts[3])
        shr = parse(Int, parts[4])
        dur_min = parse(Int, parts[5])
        lat = parse(Float64, parts[6])
        lon = parse(Float64, parts[7])
        conc = parse(Float64, parts[8])
        stn = parse(Int, parts[9])

        # Convert start hour to DateTime
        hr = shr ÷ 100
        dt = DateTime(yr, mn, dy, hr, 0)
        t_hours = (dt - ref_dt).value / (3600 * 1000)
        dur_hours = dur_min / 100  # format is HHMM, e.g. 0300 = 3h

        # Only keep valid measurements (conc >= 0)
        conc >= 0.0 || continue
        t_hours >= 0.0 || continue

        push!(obs, ETEXObs(stn, lat, lon, t_hours, dur_hours, conc))
    end
    return obs
end

const OBSERVATIONS = load_observations()
const STATION_IDS = sort(unique(o.station_id for o in OBSERVATIONS))
println("  $(length(OBSERVATIONS)) valid measurements at $(length(STATION_IDS)) stations")

# Time-integrated concentration per station (ng/m³ × h)
const STATION_TIC = let
    tic = Dict{Int, Float64}()
    for o in OBSERVATIONS
        tic[o.station_id] = get(tic, o.station_id, 0.0) + o.concentration * o.duration_hours
    end
    tic
end

# Station coordinates lookup
const STATION_COORDS = let
    coords = Dict{Int, Tuple{Float64,Float64}}()
    for o in OBSERVATIONS
        coords[o.station_id] = (o.lat, o.lon)
    end
    coords
end

println("  Stations with nonzero TIC: $(count(v -> v > 0, values(STATION_TIC)))")

# ============================================================================
# LOAD MET DATA (cached once)
# ============================================================================

println("\n2. Loading ERA5 met data...")
const ERA5_FILES = let
    era5_files = Transport.etex_era5_files()
    println("  $(length(era5_files)) ERA5 files")
    era5_files
end

const MET_FORMAT = Transport.detect_met_format(ERA5_FILES[1])
const NX, NY, NK = NCDataset(ERA5_FILES[1]) do ds
    Transport.get_met_dimensions(MET_FORMAT, ds)
end
const LAT_RANGE, LON_RANGE = NCDataset(ERA5_FILES[1]) do ds
    Float64.(ds["latitude"][:]), Float64.(ds["longitude"][:])
end

const START_FILE = 5
const END_FILE = min(START_FILE + SIM_HOURS ÷ 3 + 2, length(ERA5_FILES))
const N_CACHED_FILES = END_FILE - START_FILE + 1
const MET_CACHE = let
    cache = Dict{Tuple{Int,Int}, Any}()
    for fi in START_FILE:END_FILE
        NCDataset(ERA5_FILES[fi]) do ds
            times = Transport.get_time_variable(MET_FORMAT, ds)
            for ti in 1:length(times)
                mf = Transport.MeteoFields(NX, NY, NK, T=Float32)
                t2 = ti < length(times) ? ti + 1 : ti
                Transport.read_initial_met_fields!(MET_FORMAT, mf, ds, ti, t2)
                cache[(fi, ti)] = mf
            end
        end
    end
    println("  Cached $(length(cache)) met timesteps (files $START_FILE-$END_FILE, $N_CACHED_FILES files)")
    cache
end

# ============================================================================
# PARAMETER SPACE (9 transport dimensions)
# ============================================================================

# No particle size params — ETEX is an inert gas tracer.
# Deposition velocity is fixed (used only as plume diagnostic, not physics).
const PARAM_NAMES = [
    "sigma_w_scale", "sigma_h_scale", "h_diff_scale", "tl_scale",
    "omega_scale", "mixing_height_scale", "tmix_scale",
    "roughness_scale",
]

const LB = Float64[0.1, 0.1, 0.005, 0.01,  0.1, 0.1, 0.1,  0.1]
const UB = Float64[10.0, 10.0, 50.0, 100.0,  5.0, 20.0, 8.0,  8.0]
const N_DIM = length(LB)

# No deposition — ETEX is an inert gas tracer, scoring uses surface air concentration

# Baseline: unit scales
const BASELINE = clamp.(ones(Float64, N_DIM), LB, UB)

# ============================================================================
# SIMULATION + SCORING
# ============================================================================

function run_etex_sim(params::Vector{Float64}, seed::UInt64)
    sigma_w, sigma_h, h_diff, tl = params[1:4]
    omega, mix_h, tmix = params[5:7]
    rough = params[8]

    # Domain: start 4h before release to let BL develop
    start_dt = DateTime(1994, 10, 23, 12, 0)
    domain = Transport.SimulationDomain(
        lon_min=minimum(LON_RANGE), lon_max=maximum(LON_RANGE),
        lat_min=minimum(LAT_RANGE), lat_max=maximum(LAT_RANGE),
        z_min=0.0, z_max=35000.0, nx=NX, ny=NY, nz=NK,
        start_time=start_dt, end_time=start_dt + Hour(SIM_HOURS))

    rel_x, rel_y = Transport.latlon_to_grid(domain, RELEASE_LAT, RELEASE_LON)
    geometry = ColumnRelease(5.0, 15.0)
    source = ReleaseSource((rel_x, rel_y), geometry,
                           ConstantRelease(), [RELEASE_MASS], N_PARTICLES)

    decay = [Transport.DecayParams(kdecay=Transport.NoDecay)]
    state = Transport.initialize_simulation(domain, [source], ["PMCH"], decay;
                                            log_depositions=false)

    rng = Random.MersenneTwister(seed)
    init_met = MET_CACHE[(START_FILE, 1)]
    particle_prop = ParticleProperties(diameter_μm=1.0, density_gcm3=1.2)

    p_radii = Float64[]; p_dens = Float64[]; p_idx = Int[]
    pos_s, act_s, rel_s = Transport.generate_release_particles(
        rng, source, 0, 1,
        ones(Float64, NX, NY), ones(Float64, NY, NY),
        domain.dx, domain.dy, domain.hlevel)

    if rel_s && !isempty(pos_s)
        for (pos, activity) in zip(pos_s, act_s)
            sz = Transport.height_to_sigma_hybrid(rel_x, rel_y, pos[3], init_met, 0.0)
            Transport.add_particle!(state.ensemble,
                SVector{3,Float64}(pos[1], pos[2], sz),
                SVector{3,Float64}(0.0, 0.0, 0.0),
                [activity], 0.0, icomp=1)
            push!(p_radii, 0.5e-6)
            push!(p_dens, 1200.0)
            push!(p_idx, 1)
        end
    end

    psc = ParticleSizeConfig(size_bins=[particle_prop],
        particle_radii=p_radii, particle_densities=p_dens,
        particle_size_indices=p_idx)

    hanna = HannaTurbulenceConfig{Float64}(
        sigma_scale=sigma_h, sigma_scale_vertical=sigma_w,
        tl_scale=tl, use_cbl=true)

    dep_cfg = Transport.DepositionConfig{Float64}(
        apply_dry_deposition=false, apply_wet_deposition=false,
        mixing_height=1000.0 * mix_h,
        surface_roughness=0.1 * rough)

    num_cfg = ERA5NumericalConfig{Float64}(
        interpolation_order=Transport.LinearInterp,
        ode_solver_type=:Euler, fixed_dt=300.0,
        turbulence=Transport.OrnsteinUhlenbeck)

    out_cfg = OutputConfig(trace_frequency=TRACE_DISABLED,
                           verbosity=VERBOSITY_QUIET, trace_enabled=false)

    # Save snapshots every 3 hours (matching observation windows)
    save_times = [Float64(h) * 3600.0 for h in 3:3:SIM_HOURS]
    sim_cfg = Transport.SimulationConfig{Float64}(
        saveat=save_times,
        verbose=false, max_duration=Float64(SIM_HOURS) * 3600.0,
        save_snapshots=true, dt_particle=300.0,
        use_reference_stepping=true,
        max_files=N_CACHED_FILES,
        omega_scale=omega,
        output_config=out_cfg)

    snapshots = Transport.run_simulation!(state, ERA5_FILES,
        particle_size_config=psc, deposition_config=dep_cfg,
        hanna_config=hanna, decay_params=decay, config=sim_cfg,
        numerical_config=num_cfg,
        advection_enabled=true, settling_enabled=false,
        dry_deposition_enabled=false, wet_deposition_enabled=false,
        release_height_m=RELEASE_HEIGHT_M,
        met_data_cache=MET_CACHE,
        met_format_override=MET_FORMAT,
        met_dimensions=(NX, NY, NK),
        cache_init_file_idx=START_FILE,
        cache_init_time_idx=1,
        sigma_already_initialized=true)

    return snapshots, domain
end

# Pre-compute observation TIC grid (constant across evaluations)
const FMS_GRID_RES = 2.0  # degrees — coarse enough to aggregate particles
const FMS_LON = collect(range(-15.0, 35.0, step=FMS_GRID_RES))
const FMS_LAT = collect(range(35.0, 70.0, step=FMS_GRID_RES))
const FMS_NX = length(FMS_LON)
const FMS_NY = length(FMS_LAT)

# Grid observed TIC (time-integrated concentration) onto FMS grid
# Only include observations within the simulation window (SIM_HOURS)
const OBS_TIC_GRID = let
    g = zeros(FMS_NX, FMS_NY)
    counts = zeros(Int, FMS_NX, FMS_NY)
    n_filtered = 0
    for o in OBSERVATIONS
        o.concentration > 0 || continue
        # Only use observations within the simulation time window
        o.time_hours + o.duration_hours <= Float64(SIM_HOURS) || (n_filtered += 1; continue)
        i = round(Int, (o.lon - FMS_LON[1]) / FMS_GRID_RES) + 1
        j = round(Int, (o.lat - FMS_LAT[1]) / FMS_GRID_RES) + 1
        if 1 <= i <= FMS_NX && 1 <= j <= FMS_NY
            g[i, j] += o.concentration * o.duration_hours  # TIC in ng/m³·h
            counts[i, j] += 1
        end
    end
    # Average per cell (avoid double-counting stations in same cell)
    for i in eachindex(g)
        counts[i] > 0 && (g[i] /= counts[i])
    end
    n_nonzero = count(>(0), g)
    println("  Obs TIC grid: $(n_nonzero) nonzero cells out of $(FMS_NX)×$(FMS_NY) ($n_filtered obs filtered out beyond $(SIM_HOURS)h)")
    g
end

function compute_fms(model::Matrix, reference::Matrix)
    fractions = [0.01, 0.05, 0.10, 0.20, 0.50]
    ref_vals = filter(>(0), vec(reference))
    mod_vals = filter(>(0), vec(model))
    (isempty(ref_vals) || isempty(mod_vals)) && return 0.0

    fms_scores = Float64[]
    for frac in fractions
        n_ref = max(1, round(Int, length(ref_vals) * frac))
        n_mod = max(1, round(Int, length(mod_vals) * frac))
        ref_thresh = partialsort(ref_vals, n_ref, rev=true)
        mod_thresh = partialsort(mod_vals, n_mod, rev=true)
        ref_mask = reference .>= ref_thresh
        mod_mask = model .>= mod_thresh
        inter = sum(ref_mask .& mod_mask)
        uni = sum(ref_mask .| mod_mask)
        push!(fms_scores, uni > 0 ? Float64(inter) / Float64(uni) : 0.0)
    end
    return exp(mean(log.(max.(fms_scores, 1e-4))))
end

function score_against_observations(snapshots, domain)
    # Score using ground-level (k=1) air concentration from snapshots,
    # matching the ETEX surface sampling stations (~2m intake height).
    # Grid the model surface concentration TIC onto the same 2° grid
    # as observations and compute FMS (spatial overlap metric).

    isempty(snapshots) && return 0.0

    sim_offset = 3.0 * 3600.0  # release starts 3h into simulation

    # Accumulate time-integrated surface concentration on FMS grid
    model_grid = zeros(FMS_NX, FMS_NY)

    for snap in snapshots
        t_since_release = snap.time - sim_offset
        t_since_release >= 0.0 || continue

        # Extract surface concentration (k=1, component 1)
        surf_conc = snap.concentration[:, :, 1, 1]

        # Map each met grid cell to the FMS grid and accumulate
        for iy in 1:size(surf_conc, 2), ix in 1:size(surf_conc, 1)
            c = surf_conc[ix, iy]
            c > 0.0 || continue
            lat, lon = Transport.grid_to_latlon(domain, Float64(ix), Float64(iy))
            lon > 180.0 && (lon -= 360.0)
            fi = round(Int, (lon - FMS_LON[1]) / FMS_GRID_RES) + 1
            fj = round(Int, (lat - FMS_LAT[1]) / FMS_GRID_RES) + 1
            if 1 <= fi <= FMS_NX && 1 <= fj <= FMS_NY
                model_grid[fi, fj] += c
            end
        end
    end

    return compute_fms(model_grid, OBS_TIC_GRID)
end

function objective(params::Vector{Float64}, seed::UInt64)
    try
        snapshots, domain = run_etex_sim(params, seed)
        score = score_against_observations(snapshots, domain)
        GC.gc(false)
        return (loss=1.0 - score, score=score)
    catch e
        @warn "Objective failed" exception=(e, catch_backtrace())
        return (loss=1.0, score=0.0)
    end
end

# ============================================================================
# CMA-ES (same implementation as predict_wp33_cmaes_individual.jl)
# ============================================================================

mutable struct CMAES
    N::Int; lambda::Int; mu::Int
    weights::Vector{Float64}; mueff::Float64
    cc::Float64; cs::Float64; c1::Float64; cmu::Float64; damps::Float64; chiN::Float64
    xmean::Vector{Float64}; sigma::Float64
    pc::Vector{Float64}; ps::Vector{Float64}
    C::Matrix{Float64}; B::Matrix{Float64}; D::Vector{Float64}
    eigeneval::Int; lb::Vector{Float64}; ub::Vector{Float64}
    counteval::Int; generation::Int
    best_ever_val::Float64; best_ever_x::Vector{Float64}; stagnation_counter::Int
end

function CMAES(xstart::Vector{Float64}; lb, ub, popsize::Int=0, sigma_frac::Float64=0.3)
    N = length(xstart)
    lambda = popsize > 0 ? popsize : 4 + floor(Int, 3 * log(N))
    mu = lambda ÷ 2
    raw_w = [log(lambda/2 + 0.5) - log(i) for i in 1:mu]
    weights = raw_w ./ sum(raw_w)
    mueff = sum(weights)^2 / sum(weights.^2)
    cc = (4 + mueff/N) / (N + 4 + 2*mueff/N)
    cs = (mueff + 2) / (N + mueff + 5)
    c1 = 2 / ((N + 1.3)^2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((N + 2)^2 + mueff))
    damps = 2 * mueff / lambda + 0.3 + cs
    chiN = sqrt(N) * (1 - 1/(4*N) + 1/(21*N^2))
    ranges = ub .- lb
    initial_stds = sigma_frac .* ranges
    C = diagm(initial_stds.^2)
    CMAES(N, lambda, mu, weights, mueff, cc, cs, c1, cmu, damps, chiN,
          copy(xstart), 1.0, zeros(N), zeros(N), C,
          Matrix{Float64}(I, N, N), copy(initial_stds),
          0, lb, ub, 0, 0, Inf, copy(xstart), 0)
end

function update_eigensystem!(es::CMAES)
    es.C .= (es.C .+ es.C') ./ 2
    F = eigen(Symmetric(es.C))
    es.D .= sqrt.(max.(F.values, 1e-20))
    es.B .= F.vectors
    es.eigeneval = es.counteval
end

function ask(es::CMAES)
    if es.counteval - es.eigeneval > es.lambda / (es.c1 + es.cmu) / es.N / 10
        update_eigensystem!(es)
    end
    [clamp.(es.xmean .+ es.sigma .* (es.B * (es.D .* randn(es.N))), es.lb, es.ub)
     for _ in 1:es.lambda]
end

function tell!(es::CMAES, candidates, fitvals)
    es.counteval += es.lambda; es.generation += 1
    idx = sortperm(fitvals)
    xold = copy(es.xmean)
    es.xmean .= sum(es.weights[i] .* candidates[idx[i]] for i in 1:es.mu)
    y = (es.xmean .- xold) ./ es.sigma
    z = es.B' * y; z ./= (es.D .+ 1e-20)
    Cinvsqrt_y = es.B * z
    csn = sqrt(es.cs * (2 - es.cs) * es.mueff)
    es.ps .= (1 - es.cs) .* es.ps .+ csn .* Cinvsqrt_y
    pslen = norm(es.ps)
    threshold = (1.4 + 2/(es.N + 1)) * es.chiN * sqrt(1 - (1 - es.cs)^(2 * es.counteval / es.lambda))
    hsig = pslen < threshold ? 1.0 : 0.0
    ccn = sqrt(es.cc * (2 - es.cc) * es.mueff)
    es.pc .= (1 - es.cc) .* es.pc .+ hsig * ccn .* y
    c1a = es.c1 * (1 - (1 - hsig^2) * es.cc * (2 - es.cc))
    rank_mu = zeros(es.N, es.N)
    for i in 1:es.mu
        yi = (candidates[idx[i]] .- xold) ./ es.sigma
        rank_mu .+= es.weights[i] .* (yi * yi')
    end
    es.C .= (1 - c1a - es.cmu * sum(es.weights)) .* es.C
    es.C .+= es.c1 .* (es.pc * es.pc') .+ es.cmu .* rank_mu
    es.sigma *= exp(min(1.0, (es.cs / es.damps) * (pslen / es.chiN - 1) / 2))
    if fitvals[idx[1]] < es.best_ever_val
        es.best_ever_val = fitvals[idx[1]]
        es.best_ever_x = copy(candidates[idx[1]])
        es.stagnation_counter = 0
    else
        es.stagnation_counter += 1
    end
    return fitvals[idx[1]], candidates[idx[1]]
end

# ============================================================================
# MAIN OPTIMISATION LOOP
# ============================================================================

println("\n3. Starting CMA-ES optimisation...")
println("   Parameters: $(N_DIM) ($(join(PARAM_NAMES, ", ")))")
println("   Population: $(4 + floor(Int, 3*log(N_DIM)))")

es = CMAES(BASELINE, lb=LB, ub=UB)
total_evals = 0
gen_seed = UInt64(42)
t_start = time()

while total_evals < MAX_EVALS
    candidates = ask(es)
    global gen_seed += UInt64(1)

    # Parallel evaluation (safe: met cache covers all files, no NCDataset I/O)
    results = Vector{Any}(undef, length(candidates))
    Threads.@threads for i in eachindex(candidates)
        try
            results[i] = objective(candidates[i], gen_seed)
        catch e
            @warn "Eval $i failed" exception=e
            results[i] = (loss=1.0, score=0.0)
        end
    end

    fitvals = [r.loss for r in results]
    scores = [r.score for r in results]
    best_loss, best_x = tell!(es, candidates, fitvals)
    global total_evals += length(candidates)

    best_score = 1.0 - es.best_ever_val
    gen_best = maximum(scores)
    elapsed = time() - t_start

    @printf("  Gen %3d | evals %4d | gen_best %.3f | overall_best %.3f | sigma %.2e | %.0fs\n",
            es.generation, total_evals, gen_best, best_score, es.sigma, elapsed)
    flush(stdout)

    # Save checkpoint
    if es.generation % 5 == 0
        open(joinpath(OUTPUT_DIR, "etex_cmaes_best.txt"), "w") do io
            println(io, "# ETEX CMA-ES best parameters (gen $(es.generation), score $(round(best_score, digits=4)))")
            println(io, "# $(join(PARAM_NAMES, ", "))")
            println(io, join(round.(es.best_ever_x, digits=6), ", "))
        end
    end
end

# ============================================================================
# FINAL RESULTS
# ============================================================================

best_score = 1.0 - es.best_ever_val
best_x = es.best_ever_x

println("\n" * "="^70)
println("OPTIMISATION COMPLETE")
println("="^70)
@printf("Best score: %.4f\n", best_score)
println("\nBest parameters:")
for (name, val, lo, hi) in zip(PARAM_NAMES, best_x, LB, UB)
    pct = (val - lo) / (hi - lo) * 100
    @printf("  %-24s = %8.4f  (%.0f%% of [%.3f, %.3f])\n", name, val, pct, lo, hi)
end

# Save final results
open(joinpath(OUTPUT_DIR, "etex_cmaes_final.txt"), "w") do io
    println(io, "# ETEX-1 CMA-ES final results")
    println(io, "# Score: $(round(best_score, digits=4))")
    println(io, "# Evals: $total_evals, Generations: $(es.generation)")
    println(io, "# Parameters: $(join(PARAM_NAMES, ", "))")
    println(io, join(round.(best_x, digits=6), ", "))
    println(io, "\n# Comparison with Nancy/Smoky mean:")
    nancy_smoky = [4.42, 2.60, 0.72, 3.26, 2.49, 4.26, 1.02, 2.38]
    for (name, etex_val, ns_val) in zip(PARAM_NAMES, best_x, nancy_smoky)
        @printf(io, "#   %-24s  ETEX=%.3f  Nancy/Smoky=%.3f  ratio=%.2f\n",
                name, etex_val, ns_val, etex_val / ns_val)
    end
end

println("\nResults saved to: $OUTPUT_DIR")
println("Done!")
