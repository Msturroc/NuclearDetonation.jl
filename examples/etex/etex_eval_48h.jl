#!/usr/bin/env julia
#= Quick single evaluation of ETEX best params at 48h vs 72h =#

using NuclearDetonation
using NuclearDetonation.Transport
using NCDatasets
using StaticArrays
using Random
using Dates: DateTime, Hour
using Printf
using Statistics

const DATA_DIR = joinpath(pkgdir(NuclearDetonation), "data", "etex")

const RELEASE_LAT = 48.058
const RELEASE_LON = -2.008
const RELEASE_MASS = 340e3
const N_PARTICLES = 2000
const FIXED_VD = 0.003

# Best params from expanded-bounds CMA-ES run (FMS=0.572 at 72h)
const BEST_PARAMS = [0.202063, 7.608075, 36.90909, 42.368524, 0.815504, 8.0, 6.021502, 4.431433]

# Load observations
struct ETEXObs; station_id::Int; lat::Float64; lon::Float64; time_hours::Float64; duration_hours::Float64; concentration::Float64; end

function load_observations()
    obs = ETEXObs[]
    ref_dt = DateTime(1994, 10, 23, 15, 0)
    for line in readlines(joinpath(DATA_DIR, "meas-t1.txt"))[3:end]
        parts = split(strip(line))
        length(parts) >= 9 || continue
        yr,mn,dy = parse(Int,parts[1]), parse(Int,parts[2]), parse(Int,parts[3])
        hr = parse(Int, parts[4]) ÷ 100
        dur_hours = parse(Int, parts[5]) / 100
        lat,lon = parse(Float64,parts[6]), parse(Float64,parts[7])
        conc = parse(Float64, parts[8])
        stn = parse(Int, parts[9])
        dt = DateTime(yr,mn,dy,hr,0)
        t_hours = (dt - ref_dt).value / (3600*1000)
        conc >= 0 && t_hours >= 0 && push!(obs, ETEXObs(stn, lat, lon, t_hours, dur_hours, conc))
    end
    return obs
end

const OBS = load_observations()
println("Loaded $(length(OBS)) observations")

# FMS grid
const FMS_GRID_RES = 2.0
const FMS_LON = collect(range(-15.0, 35.0, step=FMS_GRID_RES))
const FMS_LAT = collect(range(35.0, 70.0, step=FMS_GRID_RES))
const FMS_NX, FMS_NY = length(FMS_LON), length(FMS_LAT)

function obs_tic_grid(max_hours::Float64)
    g = zeros(FMS_NX, FMS_NY); counts = zeros(Int, FMS_NX, FMS_NY)
    for o in OBS
        o.concentration > 0 || continue
        o.time_hours + o.duration_hours <= max_hours || continue
        i = round(Int, (o.lon - FMS_LON[1]) / FMS_GRID_RES) + 1
        j = round(Int, (o.lat - FMS_LAT[1]) / FMS_GRID_RES) + 1
        if 1 <= i <= FMS_NX && 1 <= j <= FMS_NY
            g[i,j] += o.concentration * o.duration_hours
            counts[i,j] += 1
        end
    end
    for i in eachindex(g); counts[i] > 0 && (g[i] /= counts[i]); end
    return g
end

function compute_fms(model, reference)
    fracs = [0.01, 0.05, 0.10, 0.20, 0.50]
    rv = filter(>(0), vec(reference)); mv = filter(>(0), vec(model))
    (isempty(rv) || isempty(mv)) && return 0.0
    scores = Float64[]
    for f in fracs
        rt = partialsort(rv, max(1,round(Int,length(rv)*f)), rev=true)
        mt = partialsort(mv, max(1,round(Int,length(mv)*f)), rev=true)
        inter = sum((reference .>= rt) .& (model .>= mt))
        uni = sum((reference .>= rt) .| (model .>= mt))
        push!(scores, uni > 0 ? Float64(inter)/Float64(uni) : 0.0)
    end
    return exp(mean(log.(max.(scores, 1e-4))))
end

# Load met data
println("Loading ERA5 data...")
const ERA5_FILES = Transport.etex_era5_files()
const MET_FORMAT = Transport.detect_met_format(ERA5_FILES[1])
const NX, NY, NK = NCDataset(ERA5_FILES[1]) do ds; Transport.get_met_dimensions(MET_FORMAT, ds); end
const LAT_RANGE, LON_RANGE = NCDataset(ERA5_FILES[1]) do ds
    Float64.(ds["latitude"][:]), Float64.(ds["longitude"][:])
end

const START_FILE = 5
const END_FILE = min(START_FILE + 72 ÷ 3 + 2, length(ERA5_FILES))
println("Caching met files $START_FILE-$END_FILE...")
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
    cache
end
println("Cached $(length(MET_CACHE)) timesteps")

function run_sim(params, sim_hours)
    sigma_w, sigma_h, h_diff, tl, omega, mix_h, tmix, rough = params
    n_cached = min(START_FILE + sim_hours ÷ 3 + 2, length(ERA5_FILES)) - START_FILE + 1

    start_dt = DateTime(1994, 10, 23, 12, 0)
    domain = Transport.SimulationDomain(
        lon_min=minimum(LON_RANGE), lon_max=maximum(LON_RANGE),
        lat_min=minimum(LAT_RANGE), lat_max=maximum(LAT_RANGE),
        z_min=0.0, z_max=35000.0, nx=NX, ny=NY, nz=NK,
        start_time=start_dt, end_time=start_dt + Hour(sim_hours))

    rel_x, rel_y = Transport.latlon_to_grid(domain, RELEASE_LAT, RELEASE_LON)
    source = ReleaseSource((rel_x, rel_y), ColumnRelease(5.0, 15.0),
                           ConstantRelease(), [RELEASE_MASS], N_PARTICLES)
    decay = [Transport.DecayParams(kdecay=Transport.NoDecay)]
    state = Transport.initialize_simulation(domain, [source], ["PMCH"], decay; log_depositions=true)

    rng = Random.MersenneTwister(42)
    init_met = MET_CACHE[(START_FILE, 1)]
    pp = ParticleProperties(diameter_μm=1.0, density_gcm3=1.2)
    p_radii = Float64[]; p_dens = Float64[]; p_idx = Int[]
    pos_s, act_s, rel_s = Transport.generate_release_particles(rng, source, 0, 1,
        ones(Float64, NX, NY), ones(Float64, NY, NY), domain.dx, domain.dy, domain.hlevel)
    if rel_s && !isempty(pos_s)
        for (pos, activity) in zip(pos_s, act_s)
            sz = Transport.height_to_sigma_hybrid(rel_x, rel_y, pos[3], init_met, 0.0)
            Transport.add_particle!(state.ensemble, SVector{3,Float64}(pos[1],pos[2],sz),
                SVector{3,Float64}(0.0,0.0,0.0), [activity], 0.0, icomp=1)
            push!(p_radii, 0.5e-6); push!(p_dens, 1200.0); push!(p_idx, 1)
        end
    end

    psc = ParticleSizeConfig(size_bins=[pp], particle_radii=p_radii,
        particle_densities=p_dens, particle_size_indices=p_idx)
    hanna = HannaTurbulenceConfig{Float64}(sigma_scale=sigma_h, sigma_scale_vertical=sigma_w,
        tl_scale=tl, use_cbl=true)
    dep_cfg = Transport.DepositionConfig{Float64}(apply_dry_deposition=true, apply_wet_deposition=false,
        use_simple_deposition=true, simple_deposition_velocity=FIXED_VD, simple_surface_height=500.0,
        mixing_height=1000.0*mix_h, surface_roughness=0.1*rough)
    num_cfg = ERA5NumericalConfig{Float64}(interpolation_order=Transport.LinearInterp,
        ode_solver_type=:Euler, fixed_dt=300.0, turbulence=Transport.OrnsteinUhlenbeck)
    out_cfg = OutputConfig(trace_frequency=TRACE_DISABLED, verbosity=VERBOSITY_QUIET, trace_enabled=false)
    sim_cfg = Transport.SimulationConfig{Float64}(
        saveat=[Float64(h)*3600.0 for h in 3:3:sim_hours],
        verbose=false, max_duration=Float64(sim_hours)*3600.0,
        save_snapshots=true, dt_particle=300.0, use_reference_stepping=true,
        max_files=n_cached, omega_scale=omega, output_config=out_cfg)

    Transport.run_simulation!(state, ERA5_FILES, particle_size_config=psc, deposition_config=dep_cfg,
        hanna_config=hanna, decay_params=decay, config=sim_cfg, numerical_config=num_cfg,
        advection_enabled=true, settling_enabled=false, dry_deposition_enabled=true,
        wet_deposition_enabled=false, release_height_m=15.0, met_data_cache=MET_CACHE,
        met_format_override=MET_FORMAT, met_dimensions=(NX,NY,NK),
        cache_init_file_idx=START_FILE, cache_init_time_idx=1, sigma_already_initialized=true)
    return state, domain
end

function score_sim(state, domain, max_hours)
    isempty(state.deposition_log) && return 0.0
    sim_offset = 3.0 * 3600.0
    model_grid = zeros(FMS_NX, FMS_NY)
    for evt in state.deposition_log
        t_since_release = evt.time - sim_offset
        t_since_release >= 0.0 || continue
        t_since_release <= max_hours * 3600.0 || continue
        lat, lon = Transport.grid_to_latlon(domain, evt.x, evt.y)
        lon > 180.0 && (lon -= 360.0)
        i = round(Int, (lon - FMS_LON[1]) / FMS_GRID_RES) + 1
        j = round(Int, (lat - FMS_LAT[1]) / FMS_GRID_RES) + 1
        if 1 <= i <= FMS_NX && 1 <= j <= FMS_NY
            model_grid[i,j] += evt.mass
        end
    end
    return compute_fms(model_grid, obs_tic_grid(max_hours))
end

# Run evaluations
println("\n--- Running 72h simulation ---")
state72, dom72 = run_sim(BEST_PARAMS, 72)
fms72_full = score_sim(state72, dom72, 72.0)
fms72_at48 = score_sim(state72, dom72, 48.0)
@printf("72h sim, scored at 72h: FMS = %.4f\n", fms72_full)
@printf("72h sim, scored at 48h: FMS = %.4f\n", fms72_at48)

println("\n--- Running 48h simulation ---")
state48, dom48 = run_sim(BEST_PARAMS, 48)
fms48 = score_sim(state48, dom48, 48.0)
@printf("48h sim, scored at 48h: FMS = %.4f\n", fms48)

println("\nDeposition events: 72h=$(length(state72.deposition_log)), 48h=$(length(state48.deposition_log))")
println("Done!")
