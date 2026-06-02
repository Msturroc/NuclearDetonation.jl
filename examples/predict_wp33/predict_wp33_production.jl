#!/usr/bin/env julia
#= ============================================================================
   PREDICT WP3.3 — Production Run (16 Scenarios)

   Runs all 8 weather scenarios × 2 detonation heights (low/high burst) for
   10 kT detonations at Bergen (Norway) and Emden (Germany).

   Physics parameters are the element-wise mean of Nancy OU and Smoky OU
   CMA-ES optimised parameter vectors — calibrated against real fallout data,
   not tuned to other models.

   Outputs per scenario:
     - PNG dose rate contour plot (mR/h at H+12)
     - CF-compliant NetCDF (dose_rate_mR_hr + deposition_Bq_m2)

   Environment variables:
     TEST_SCENARIO=N   — run only scenario N (for testing)
     TEST_PARTICLES=N   — override particle count (for testing)

   Usage:
     julia --threads=12 --project=../.. predict_wp33_production.jl

   Full production:
     nohup julia --threads=12 --project=../.. predict_wp33_production.jl \
       > production_results/production_log.txt 2>&1 &
   ============================================================================ =#

using NuclearDetonation
using NuclearDetonation.Transport
using CairoMakie
using NCDatasets
using StaticArrays
using Random
using Dates
using Printf
using Statistics
using JSON3
using Interpolations

println("="^70)
println("PREDICT WP3.3 — Production Run")
println("="^70)

# ============================================================================
# PATHS AND CONSTANTS
# ============================================================================

# ERA5 source. If WP33_ERA5_LOCAL points at a directory of scenario subdirs (the
# external-drive workflow), use it; otherwise fall back to the Zenodo artifact
# `predict_wp33_era5_data` (DOI 10.5281/zenodo.20515925), downloaded on demand.
const ERA5_LOCAL_BASE = get(ENV, "WP33_ERA5_LOCAL",
    "/run/media/marc/e34b80f3-0992-4981-a17f-e396750ea8b4/era5_predict")

const DETONATION_HOUR = 12
const SIM_HOURS = 48
const YIELD_KT = parse(Float64, get(ENV, "YIELD_KT", "10.0"))

const OUTPUT_DIR = joinpath(@__DIR__,
    YIELD_KT == 10.0 ? "production_results" :
                       "production_results_$(Int(YIELD_KT))kT")
mkpath(OUTPUT_DIR)
println("  Yield: $(YIELD_KT) kT → outputs to $(OUTPUT_DIR)")
const TOTAL_ACTIVITY_BQ = 2.38e12  # Cs-137 from PREDICT spec

const N_PARTICLES = parse(Int, get(ENV, "TEST_PARTICLES", "10000"))
println("  Particles per scenario: $N_PARTICLES")

# ============================================================================
# SCENARIO DEFINITIONS
# ============================================================================

const ALL_SCENARIOS = [
    (id=1, loc="Bergen", lat=60.389, lon=5.328, date=Date(2024,1,6),
     era5="scenario_1_bergen_20240106"),
    (id=2, loc="Bergen", lat=60.389, lon=5.328, date=Date(2024,1,22),
     era5="scenario_2_bergen_20240122"),
    (id=3, loc="Bergen", lat=60.389, lon=5.328, date=Date(2024,5,25),
     era5="scenario_3_bergen_20240525"),
    (id=4, loc="Bergen", lat=60.389, lon=5.328, date=Date(2024,8,3),
     era5="scenario_4_bergen_20240803"),
    (id=5, loc="Emden",  lat=53.365, lon=7.206, date=Date(2023,1,16),
     era5="scenario_5_emden_20230116"),
    (id=6, loc="Emden",  lat=53.365, lon=7.206, date=Date(2023,7,6),
     era5="scenario_6_emden_20230706"),
    (id=7, loc="Emden",  lat=53.365, lon=7.206, date=Date(2023,8,30),
     era5="scenario_7_emden_20230830"),
    (id=8, loc="Emden",  lat=53.365, lon=7.206, date=Date(2023,9,26),
     era5="scenario_8_emden_20230926"),
]

# CLI: --only N [low|high|both] mirrors TEST_SCENARIO/TEST_HEIGHT env vars.
function parse_only_args()
    only_id = nothing
    only_height = nothing
    args = ARGS
    i = 1
    while i <= length(args)
        if args[i] == "--only" && i + 1 <= length(args)
            only_id = parse(Int, args[i+1])
            if i + 2 <= length(args) && args[i+2] in ("low", "high", "both")
                only_height = args[i+2]
                i += 3
            else
                i += 2
            end
        else
            i += 1
        end
    end
    return only_id, only_height
end

const CLI_ONLY_ID, CLI_ONLY_HEIGHT = parse_only_args()

# Filter to single scenario if TEST_SCENARIO env var or --only N CLI arg is set
const SCENARIOS = if !isnothing(CLI_ONLY_ID)
    filter(s -> s.id == CLI_ONLY_ID, ALL_SCENARIOS)
elseif haskey(ENV, "TEST_SCENARIO")
    sc_id = parse(Int, ENV["TEST_SCENARIO"])
    filter(s -> s.id == sc_id, ALL_SCENARIOS)
else
    ALL_SCENARIOS
end

# CLI burst override propagates into the TEST_HEIGHT-style filter below.
const CLI_HEIGHT_OVERRIDE = CLI_ONLY_HEIGHT

println("  Running $(length(SCENARIOS)) scenario(s) × 2 heights = $(length(SCENARIOS)*2) simulations")

# ============================================================================
# CALIBRATED PARAMETERS: Trinity surface-burst (used for both burst modes)
# ============================================================================
# Single BIPOP-CMA-ES calibration against the Trinity 21 kT 1945-07-16
# surface-burst fallout observations is used as the baseline for BOTH
# `:low` (HOB = 0 m) and `:high` (HOB = 710 m) Emden/Bergen scenarios.
#
# We also digitised the DASA-1251 Vol I Fig.~202 Doppler off-site plate
# and ran a separate air-burst calibration against it, considering
# Doppler as a more direct historical analogue to the WP1.3 10 kT /
# 710 m air-burst regime. The Doppler digitised dataset has only three
# usable dose-rate contours (0.0019, 0.0952, 0.19 mR/h) and the off-site
# plate is partly Shasta co-contaminated (DASA-1251 Vol I p. 316), so
# the resulting calibration target was too sparse and the FMS plateaued
# at 0.23 — insufficient to justify using it as the air-burst
# parameterisation. Trinity is therefore used for both burst modes and
# the burst-mode difference enters via the HOB geometry only, not via
# physics parameters.
#
# Trinity 6k score: 81.1% combined, FMS = 0.535, shape = 0.900,
# bearing = 0.983, extent = 0.960, TOA = 1.000.

const TRINITY_BEST = joinpath(@__DIR__, "..", "calibration_us_tests",
                              "trinity_cmaes_ou_best.txt")

const PARAM_KEYS = (
    :d_fine, :sg_fine, :d_coarse, :sg_coarse, :frac_fine,
    :sigma_w, :sigma_h, :h_diff, :tl,
    :vd, :vgrav,
    :omega, :mix_h, :tmix, :sfc_h, :rough,
)

# CMA-ES checkpoint field → production NamedTuple field mapping
const CMAES_TO_PROD = Dict(
    "d_median_fine"       => :d_fine,
    "sigma_g_fine"        => :sg_fine,
    "d_median_coarse"     => :d_coarse,
    "sigma_g_coarse"      => :sg_coarse,
    "frac_fine"           => :frac_fine,
    "sigma_w_scale"       => :sigma_w,
    "sigma_h_scale"       => :sigma_h,
    "h_diff_scale"        => :h_diff,
    "tl_scale"            => :tl,
    "vd_scale"            => :vd,
    "vgrav_scale"         => :vgrav,
    "omega_scale"         => :omega,
    "mixing_height_scale" => :mix_h,
    "tmix_scale"          => :tmix,
    "surface_height_scale" => :sfc_h,
    "roughness_scale"     => :rough,
)

"""
    load_calibration(path) -> (params::NamedTuple, activity_scale::Float64)

Read a CMA-ES `*_best.txt` checkpoint and return the production-shaped
parameter NamedTuple plus the activity_scale (×1e15 Bq, used for the
FP_ACTIVITY_BQ per-kT term).
"""
function load_calibration(path::AbstractString)
    isfile(path) || error("Calibration checkpoint missing: $path")
    vals = Dict{Symbol, Float64}()
    activity_scale = NaN
    for line in eachline(path)
        startswith(line, "#") && continue
        parts = split(line, "\t", limit=2)
        length(parts) == 2 || continue
        key = strip(parts[1]); val = parse(Float64, strip(parts[2]))
        if haskey(CMAES_TO_PROD, key)
            vals[CMAES_TO_PROD[key]] = val
        elseif key == "activity_scale"
            activity_scale = val
        end
    end
    for k in PARAM_KEYS
        haskey(vals, k) || error("Checkpoint $path is missing field for $k")
    end
    isnan(activity_scale) && error("Checkpoint $path is missing activity_scale")
    return (NamedTuple{PARAM_KEYS}(Tuple(vals[k] for k in PARAM_KEYS)), activity_scale)
end

const PARAMS_SURFACE, ACTIVITY_SCALE_SURFACE = load_calibration(TRINITY_BEST)

# Air-burst now uses the Trinity baseline (see comment block above).
# Aliases retained so downstream call sites that branch on `height_mode`
# keep working unchanged — both modes resolve to the Trinity calibration.
const PARAMS_AIRBURST          = PARAMS_SURFACE
const ACTIVITY_SCALE_AIRBURST  = ACTIVITY_SCALE_SURFACE
println("  Air-burst parameterisation: Trinity (unified with :low; Doppler digitised obs was insufficient — see report)")

# Per-kT fission-product activity, derived from the Trinity calibration's
# activity_scale (430.84e15 Bq at 21 kT → 2.052e16 Bq/kT). Both burst
# modes scale from the same per-kT figure.
const TRINITY_YIELD_KT = 21.0
const FP_ACTIVITY_BQ_SURFACE  = ACTIVITY_SCALE_SURFACE * 1e15 / TRINITY_YIELD_KT * YIELD_KT
const FP_ACTIVITY_BQ_AIRBURST = FP_ACTIVITY_BQ_SURFACE

const SMOOTH_SIGMA = 1.0    # Gaussian smoothing width (cells)
const GRID_RES = 0.05       # output grid resolution (degrees)

# WP3.3 MET delivery time grid:
#   - dose rate:   H+1..H+12 hourly, then H+15..H+48 3-hourly (24 values)
#   - deposition:  H+12, H+24, H+36, H+48 (4 values)
#   - precip 12h:  H+12, H+24, H+36, H+48 (4 windows, same labels)
const DOSE_HOURS = vcat(collect(1:12), collect(15:3:48))   # 24 values
const DEP_HOURS  = [12, 24, 36, 48]                         # 4 values
const PRECIP_HOURS = DEP_HOURS                              # 4 values
const SAVEAT_S = sort!(unique!(Float64.(vcat(DOSE_HOURS, DEP_HOURS)) .* 3600.0))
const DOSE_REF_HOUR = 12    # legacy single H+12 plot still uses this

# Load 50m Natural Earth basemap layers (land/ocean fill + country borders + coastline)
const ND_DATA = joinpath(pkgdir(NuclearDetonation), "data")

function _load_polygons(path)
    polys = Vector{Tuple{Vector{Float64},Vector{Float64}}}()
    gj = JSON3.read(read(path, String))
    for feat in gj.features
        geom = feat.geometry
        if geom.type == "Polygon"
            for ring in geom.coordinates
                push!(polys, (Float64[c[1] for c in ring], Float64[c[2] for c in ring]))
            end
        elseif geom.type == "MultiPolygon"
            for poly in geom.coordinates, ring in poly
                push!(polys, (Float64[c[1] for c in ring], Float64[c[2] for c in ring]))
            end
        end
    end
    polys
end

function _load_linestrings(path)
    segs = Vector{Tuple{Vector{Float64},Vector{Float64}}}()
    gj = JSON3.read(read(path, String))
    for feat in gj.features
        geom = feat.geometry
        if geom.type == "LineString"
            push!(segs, (Float64[c[1] for c in geom.coordinates], Float64[c[2] for c in geom.coordinates]))
        elseif geom.type == "MultiLineString"
            for line in geom.coordinates
                push!(segs, (Float64[c[1] for c in line], Float64[c[2] for c in line]))
            end
        end
    end
    segs
end

const LAND_POLYS    = _load_polygons(joinpath(ND_DATA, "ne_50m_land.geojson"))
const COUNTRY_POLYS = _load_polygons(joinpath(ND_DATA, "ne_50m_admin_0_countries.geojson"))
const COASTLINES    = _load_linestrings(joinpath(ND_DATA, "ne_50m_coastline.geojson"))
println("  Loaded basemap: $(length(LAND_POLYS)) land polys, $(length(COUNTRY_POLYS)) country polys, $(length(COASTLINES)) coast segments")

const SEA_COLOUR   = RGBf(0.86, 0.92, 0.98)
const LAND_COLOUR  = RGBf(0.97, 0.95, 0.86)
const COAST_COLOUR = RGBf(0.20, 0.20, 0.20)
const ADMIN_COLOUR = RGBf(0.55, 0.55, 0.55)

# Dose conversion constants
# The simulation tracks 2.38e12 Bq Cs-137 as tracer.  To compute total
# mixed-FP dose rate we scale by the ratio of total FP activity to Cs-137.
# Both burst modes now use the Trinity-derived per-kT FP figure
# (FP_SCALE_AIRBURST is aliased to FP_SCALE_SURFACE).
const FP_SCALE_SURFACE  = FP_ACTIVITY_BQ_SURFACE  / TOTAL_ACTIVITY_BQ
const FP_SCALE_AIRBURST = FP_SCALE_SURFACE
const K_DOSE = 1.9e-6       # mSv/h per Bq/m² at H+1 (Glasstone & Dolan)
const DECAY_12H = 12.0^(-1.2)  # bomb decay factor to H+12
const MSV_TO_MR = 100.0     # 1 mSv/h ≈ 100 mR/h

# ============================================================================
# PARTICLE SIZE HELPERS
# ============================================================================

function snap_settling_velocity(d_um::Float64)
    snap_d = [2.2, 4.4, 8.6, 14.6, 22.8, 36.1, 56.5, 92.3, 173.2]
    snap_v = [0.2, 0.7, 2.5, 6.9, 15.9, 35.6, 71.2, 137.0, 277.3]
    log_d = log.(snap_d); log_v = log.(snap_v)
    ld = log(d_um)
    if ld <= log_d[1]
        slope = (log_v[2] - log_v[1]) / (log_d[2] - log_d[1])
        return exp(log_v[1] + slope * (ld - log_d[1]))
    elseif ld >= log_d[end]
        slope = (log_v[end] - log_v[end-1]) / (log_d[end] - log_d[end-1])
        return exp(log_v[end] + slope * (ld - log_d[end]))
    end
    i = searchsortedlast(log_d, ld)
    i = clamp(i, 1, length(log_d) - 1)
    frac = (ld - log_d[i]) / (log_d[i+1] - log_d[i])
    return exp(log_v[i] + frac * (log_v[i+1] - log_v[i]))
end

function make_bimodal_bins(d_fine, sg_fine, d_coarse, sg_coarse; n_bins=15)
    lo = max(min(log(d_fine) - 3log(sg_fine), log(d_coarse) - 3log(sg_coarse)), log(1.0))
    hi = min(max(log(d_fine) + 3log(sg_fine), log(d_coarse) + 3log(sg_coarse)), log(500.0))
    [(d=d, v=snap_settling_velocity(d)) for d in exp.(range(lo, hi, length=n_bins))]
end

function bimodal_weights(d_fine, sg_fine, d_coarse, sg_coarse, frac_fine, bins)
    w = [frac_fine * exp(-0.5*((log(b.d)-log(d_fine))/log(sg_fine))^2)/log(sg_fine) +
         (1-frac_fine) * exp(-0.5*((log(b.d)-log(d_coarse))/log(sg_coarse))^2)/log(sg_coarse)
         for b in bins]
    w ./= sum(w)
end

function gaussian_smooth(field::Matrix{T}, sigma::Real; truncate::Real=4.0) where T
    radius = ceil(Int, sigma * truncate)
    kernel_1d = [exp(-0.5 * (x / sigma)^2) for x in -radius:radius]
    kernel_1d ./= sum(kernel_1d)
    nx, ny = size(field)
    temp = zeros(T, nx, ny)
    smoothed = zeros(T, nx, ny)
    for j in 1:ny, i in 1:nx
        val, weight = zero(T), zero(T)
        for k in -radius:radius
            ii = i + k
            if 1 <= ii <= nx
                w = kernel_1d[k + radius + 1]
                val += field[ii, j] * w; weight += w
            end
        end
        temp[i, j] = weight > 0 ? val / weight : zero(T)
    end
    for i in 1:nx, j in 1:ny
        val, weight = zero(T), zero(T)
        for k in -radius:radius
            jj = j + k
            if 1 <= jj <= ny
                w = kernel_1d[k + radius + 1]
                val += temp[i, jj] * w; weight += w
            end
        end
        smoothed[i, j] = weight > 0 ? val / weight : zero(T)
    end
    return smoothed
end

# ============================================================================
# MET DATA LOADING
# ============================================================================

struct ScenarioCache
    id::Int
    loc::String
    date::Date
    lat::Float64
    lon::Float64
    era5_files::Vector{String}
    met_format::Any
    nx::Int; ny::Int; nk::Int
    lat_range::Vector{Float64}
    lon_range::Vector{Float64}
    met_cache::Dict{Tuple{Int,Int}, Any}
    start_file::Int
end

function load_scenario_cache(sc)
    # Prefer the local external-drive copy if present; else pull the scenario
    # from the Zenodo artifact (downloads the combined 2.9 GB record on first use).
    local_scen = joinpath(ERA5_LOCAL_BASE, sc.era5)
    era5_files = if isdir(local_scen)
        sort(filter(f -> endswith(f, "_snap.nc"), readdir(local_scen, join=true)))
    else
        Transport.predict_wp33_era5_files(sc.era5)
    end
    met_format = Transport.detect_met_format(era5_files[1])
    nx, ny, nk = NCDataset(era5_files[1]) do ds
        Transport.get_met_dimensions(met_format, ds)
    end
    lat_range, lon_range = NCDataset(era5_files[1]) do ds
        Float64.(ds["latitude"][:]), Float64.(ds["longitude"][:])
    end

    # Noon detonation: file 5 (=12h) through file 5+16=21 (=48h window)
    start_file = DETONATION_HOUR ÷ 3 + 1   # file 5
    end_file = min(start_file + SIM_HOURS ÷ 3 + 1, length(era5_files))
    met_cache = Dict{Tuple{Int,Int}, Any}()
    for fi in start_file:end_file
        NCDataset(era5_files[fi]) do ds
            times = Transport.get_time_variable(met_format, ds)
            for ti in 1:length(times)
                mf = Transport.MeteoFields(nx, ny, nk, T=Float32)
                t2 = ti < length(times) ? ti + 1 : ti
                Transport.read_initial_met_fields!(met_format, mf, ds, ti, t2)
                met_cache[(fi, ti)] = mf
            end
        end
    end
    println("    Cached $(length(met_cache)) met timesteps (files $(start_file)-$(end_file))")

    ScenarioCache(sc.id, sc.loc, sc.date, sc.lat, sc.lon, era5_files, met_format,
                  nx, ny, nk, lat_range, lon_range, met_cache, start_file)
end

# ============================================================================
# CORE SIMULATION
# ============================================================================

function run_scenario(cache::ScenarioCache, height_mode::Symbol)
    # Both `:low` and `:high` now resolve to the Trinity surface-burst
    # calibration (PARAMS_AIRBURST is aliased to PARAMS_SURFACE — see
    # the constants block). The Doppler digitised observations were
    # considered as an air-burst calibration target but the three
    # available dose-rate contours plus Shasta co-contamination made
    # the resulting fit insufficient. Burst-mode differs only in HOB.
    p = height_mode == :high ? PARAMS_AIRBURST : PARAMS_SURFACE
    nx, ny, nk = cache.nx, cache.ny, cache.nk
    det_time = Dates.Dates.DateTime(cache.date) + Dates.Hour(DETONATION_HOUR)

    domain = Transport.SimulationDomain(
        lon_min=minimum(cache.lon_range), lon_max=maximum(cache.lon_range),
        lat_min=minimum(cache.lat_range), lat_max=maximum(cache.lat_range),
        z_min=0.0, z_max=35000.0, nx=nx, ny=ny, nz=nk,
        start_time=det_time, end_time=det_time + Dates.Hour(SIM_HOURS))

    rel_x, rel_y = Transport.latlon_to_grid(domain, cache.lat, cache.lon)

    # PREDICT WP1.3 (D9.31) defines the two scenarios by HOB and
    # fireball–ground interaction, not cloud shape:
    #   low  = Regime 6 surface burst (HOB = 0 m, both yields)
    #   high = Regime 2 low air burst
    #          (HOB = 710 m at 10 kT, 1500 m at 100 kT — WP1.3 spec)
    # Cloud geometry uses KDFOC3 scaling from YIELD_KT and HOB.
    high_hob = YIELD_KT == 10.0 ? 710.0 :
               YIELD_KT == 100.0 ? 1500.0 :
               error("WP3.3 spec only defines high-HOB for 10 kT and 100 kT, got $(YIELD_KT) kT")
    hob = height_mode == :low ? 0.0 : high_hob
    cloud = Transport.create_mushroom_cloud_from_yield(Float64(YIELD_KT), hob)
    geom = cloud
    release_height = cloud.cap_height
    label = height_mode == :low ?
        "Surface burst (HOB = 0 m)" :
        "Low air burst (HOB = $(Int(high_hob)) m)"

    source = ReleaseSource((rel_x, rel_y), geom, BombRelease(0.0),
                           [TOTAL_ACTIVITY_BQ], N_PARTICLES)

    decay = [Transport.DecayParams(kdecay=Transport.NoDecay)]
    state = Transport.initialize_simulation(domain, [source], ["Cs137"], decay;
                                            log_depositions=true)

    rng = Random.MersenneTwister(42)
    init_met = cache.met_cache[(cache.start_file, 1)]

    bins = make_bimodal_bins(p.d_fine, p.sg_fine, p.d_coarse, p.sg_coarse)
    weights = bimodal_weights(p.d_fine, p.sg_fine, p.d_coarse, p.sg_coarse, p.frac_fine, bins)
    cw = cumsum(weights)

    snap_bins = [ParticleProperties(diameter_μm=b.d, density_gcm3=2.5) for b in bins]
    p_radii = Float64[]; p_dens = Float64[]; p_idx = Int[]
    fixed_grav = [b.v * p.vgrav for b in bins]

    pos_s, act_s, rel_s = Transport.generate_release_particles(
        rng, source, 0, 1,
        ones(Float64, nx, ny), ones(Float64, ny, ny),
        domain.dx, domain.dy, domain.hlevel)

    if rel_s && !isempty(pos_s)
        for (pos, activity) in zip(pos_s, act_s)
            sz = Transport.height_to_sigma_hybrid(rel_x, rel_y, pos[3], init_met, 0.0)
            Transport.add_particle!(state.ensemble,
                SVector{3,Float64}(pos[1], pos[2], sz),
                SVector{3,Float64}(0.0, 0.0, 0.0),
                [activity], 0.0, icomp=1)
            idx = clamp(searchsortedfirst(cw, rand(rng)), 1, length(bins))
            push!(p_radii, bins[idx].d * 0.5e-6)
            push!(p_dens, 2500.0)
            push!(p_idx, idx)
            state.ensemble.particles[end].grv = Float32(bins[idx].v * 0.01 * p.vgrav)
        end
    end

    psc = ParticleSizeConfig(size_bins=snap_bins, particle_radii=p_radii,
        particle_densities=p_dens, particle_size_indices=p_idx,
        fixed_gravity_cm_s=fixed_grav)

    hanna = HannaTurbulenceConfig{Float64}(
        sigma_scale=p.sigma_h, sigma_scale_vertical=p.sigma_w,
        tl_scale=p.tl, use_cbl=true)

    dep_cfg = Transport.DepositionConfig{Float64}(
        apply_dry_deposition=true, apply_wet_deposition=true,
        use_simple_deposition=true,
        simple_deposition_velocity=0.002 * p.vd,
        simple_surface_height=30.0 * p.sfc_h,
        mixing_height=1000.0 * p.mix_h,
        surface_roughness=0.1 * p.rough)

    num_cfg = ERA5NumericalConfig{Float64}(
        interpolation_order=Transport.LinearInterp,
        ode_solver_type=:Euler, fixed_dt=300.0,
        turbulence=Transport.OrnsteinUhlenbeck)

    out_cfg = OutputConfig(trace_frequency=TRACE_DISABLED,
                           verbosity=VERBOSITY_QUIET, trace_enabled=false)

    sim_cfg = Transport.SimulationConfig{Float64}(
        saveat=SAVEAT_S,
        verbose=false, max_duration=(Float64(SIM_HOURS) + 0.5) * 3600.0,
        save_snapshots=true, dt_particle=300.0,
        use_reference_stepping=true,
        max_files=length(cache.era5_files),
        omega_scale=p.omega,
        output_config=out_cfg)

    snapshots = Transport.run_simulation!(state, cache.era5_files,
        particle_size_config=psc, deposition_config=dep_cfg,
        hanna_config=hanna, decay_params=decay, config=sim_cfg,
        numerical_config=num_cfg,
        advection_enabled=true, settling_enabled=true,
        dry_deposition_enabled=true, wet_deposition_enabled=true,
        release_height_m=release_height,
        met_data_cache=cache.met_cache,
        met_format_override=cache.met_format,
        met_dimensions=(nx, ny, nk),
        cache_init_file_idx=cache.start_file,
        cache_init_time_idx=1,
        sigma_already_initialized=true)

    return state, domain, cloud, label, snapshots
end

# ============================================================================
# DOSE FIELD BUILDER
# ============================================================================

function build_dose_fields(state, domain, cache, height_mode::Symbol)
    # Output grid at GRID_RES° resolution covering ERA5 domain
    lon_grid = range(minimum(cache.lon_range), maximum(cache.lon_range), step=GRID_RES)
    lat_grid = range(minimum(cache.lat_range), maximum(cache.lat_range), step=GRID_RES)
    nx_out, ny_out = length(lon_grid), length(lat_grid)

    # Bin deposition events onto grid (only up to H+12 for dose rate snapshot)
    max_time = Float64(DOSE_REF_HOUR) * 3600.0
    fine_dep = zeros(nx_out, ny_out)
    n_used = 0
    for evt in state.deposition_log
        evt.time > max_time && continue
        lat, lon = Transport.grid_to_latlon(domain, evt.x, evt.y)
        lon > 180.0 && (lon -= 360.0)
        i = searchsortedlast(lon_grid, lon)
        j = searchsortedlast(lat_grid, lat)
        if 1 <= i <= nx_out && 1 <= j <= ny_out
            fine_dep[i, j] += evt.mass
            n_used += 1
        end
    end
    println("    Deposition events used (H+$(DOSE_REF_HOUR)): $n_used / $(length(state.deposition_log))")

    # Convert Bq/cell → Bq/m²
    dlat = length(lat_grid) > 1 ? abs(lat_grid[2] - lat_grid[1]) : GRID_RES
    dlon = length(lon_grid) > 1 ? abs(lon_grid[2] - lon_grid[1]) : GRID_RES
    ref_lat = 0.5 * (first(lat_grid) + last(lat_grid))
    dy_m = dlat * 111_000.0
    dx_m = dlon * 111_000.0 * cosd(ref_lat)
    cell_area_m2 = dx_m * dy_m

    dep_bqm2 = fine_dep ./ cell_area_m2

    # Convert to mR/h at H+12: scale Cs-137 → total mixed FP, then apply dose conversion.
    # FP_SCALE differs between surface and air-burst regimes (Trinity vs Doppler calibration).
    fp_scale = height_mode == :high ? FP_SCALE_AIRBURST : FP_SCALE_SURFACE
    dose_mRh = dep_bqm2 .* (fp_scale * K_DOSE * DECAY_12H * MSV_TO_MR)
    dose_smooth = gaussian_smooth(dose_mRh, SMOOTH_SIGMA)

    return (lon_grid=collect(lon_grid), lat_grid=collect(lat_grid),
            dose_smooth=dose_smooth, dose_raw=dose_mRh, dep_bqm2=dep_bqm2)
end

# ============================================================================
# PLOTTING
# ============================================================================

function plot_dose_contours(filename, lon_grid, lat_grid, dose_smooth,
                            det_lat, det_lon, title_str)
    contour_levels = [1.0, 4.0, 10.0, 40.0, 100.0, 400.0, 1000.0]
    contour_colors = [:blue, :dodgerblue, :green, :yellow, :orange, :red, :darkred]

    # Dynamic axis limits: bounding box of dose > 0.5 mR/h with at least a 3°
    # window so country context (Norway / Germany / North Sea) is visible.
    mask = dose_smooth .> 0.5
    if any(mask)
        lon_idx = findall(any(mask, dims=2)[:])
        lat_idx = findall(any(mask, dims=1)[:])
        lon_min = lon_grid[first(lon_idx)]
        lon_max = lon_grid[last(lon_idx)]
        lat_min = lat_grid[first(lat_idx)]
        lat_max = lat_grid[last(lat_idx)]
    else
        lon_min = lon_max = det_lon
        lat_min = lat_max = det_lat
    end
    lon_min = min(lon_min, det_lon); lon_max = max(lon_max, det_lon)
    lat_min = min(lat_min, det_lat); lat_max = max(lat_max, det_lat)
    # Enforce a minimum 3° window in each dimension
    if (lon_max - lon_min) < 3.0
        c = 0.5 * (lon_min + lon_max); lon_min = c - 1.5; lon_max = c + 1.5
    end
    if (lat_max - lat_min) < 3.0
        c = 0.5 * (lat_min + lat_max); lat_min = c - 1.5; lat_max = c + 1.5
    end
    # 30% buffer on each side
    dlon = lon_max - lon_min; dlat = lat_max - lat_min
    lon_min -= 0.3 * dlon; lon_max += 0.3 * dlon
    lat_min -= 0.3 * dlat; lat_max += 0.3 * dlat

    # Correct aspect ratio for latitude (1° lon shrinks by cos(lat))
    mid_lat = 0.5 * (lat_min + lat_max)
    aspect_ratio = cosd(mid_lat)

    fig = Figure(size=(900, 800), fontsize=22)
    ax = Axis(fig[1, 1],
        title=title_str, titlesize=24,
        xlabel="Longitude", ylabel="Latitude",
        limits=(lon_min, lon_max, lat_min, lat_max),
        aspect=AxisAspect(aspect_ratio))

    # Basemap: sea background, land fill, country borders, coastlines on top.
    poly!(ax, [lon_min, lon_max, lon_max, lon_min],
              [lat_min, lat_min, lat_max, lat_max];
              color=SEA_COLOUR, strokewidth=0)
    for (clons, clats) in LAND_POLYS
        any(lon_min .<= clons .<= lon_max) && any(lat_min .<= clats .<= lat_max) || continue
        poly!(ax, clons, clats; color=LAND_COLOUR, strokewidth=0)
    end
    for (clons, clats) in COUNTRY_POLYS
        any(lon_min .<= clons .<= lon_max) && any(lat_min .<= clats .<= lat_max) || continue
        lines!(ax, clons, clats; color=ADMIN_COLOUR, linewidth=0.6)
    end
    for (clons, clats) in COASTLINES
        any(lon_min .<= clons .<= lon_max) && any(lat_min .<= clats .<= lat_max) || continue
        lines!(ax, clons, clats; color=COAST_COLOUR, linewidth=1.0)
    end

    for (level, col) in zip(contour_levels, contour_colors)
        if maximum(dose_smooth) >= level
            contour!(ax, lon_grid, lat_grid, dose_smooth,
                levels=[level], color=col, linewidth=2.0)
        end
    end

    scatter!(ax, [det_lon], [det_lat]; marker=:star5, markersize=22,
        color=:black, strokecolor=:white, strokewidth=1.5)

    legend_elements = [LineElement(color=c, linewidth=3) for c in contour_colors]
    legend_labels = ["$(Int(l)) mR/h" for l in contour_levels]
    Legend(fig[2, 1], legend_elements, legend_labels, "Dose Rate (H+12)",
        orientation=:horizontal, tellwidth=false, tellheight=true, nbanks=1,
        labelsize=18, titlesize=18)

    save(filename, fig, px_per_unit=2)
    println("    Saved: $(basename(filename))")
    return nothing
end

# ============================================================================
# NETCDF EXPORT
# ============================================================================

function save_netcdf(filename, lon_grid, lat_grid, dose_smooth, dep_bqm2, dose_raw,
                     det_time, scenario_label, height_mode::Symbol)
    NCDataset(filename, "c") do ds
        defDim(ds, "longitude", length(lon_grid))
        defDim(ds, "latitude", length(lat_grid))

        lon_var = defVar(ds, "longitude", Float64, ("longitude",))
        lon_var[:] = lon_grid
        lon_var.attrib["units"] = "degrees_east"
        lon_var.attrib["long_name"] = "Longitude"
        lon_var.attrib["standard_name"] = "longitude"

        lat_var = defVar(ds, "latitude", Float64, ("latitude",))
        lat_var[:] = lat_grid
        lat_var.attrib["units"] = "degrees_north"
        lat_var.attrib["long_name"] = "Latitude"
        lat_var.attrib["standard_name"] = "latitude"

        dose_var = defVar(ds, "dose_rate_mR_hr", Float32, ("longitude", "latitude"),
                          fillvalue=Float32(-9999))
        dose_var[:, :] = Float32.(dose_smooth)
        dose_var.attrib["units"] = "mR/hr"
        dose_var.attrib["long_name"] = "Smoothed dose rate at H+12 (total mixed FP)"
        dose_var.attrib["reference_time"] = Dates.format(det_time + Dates.Hour(12),
                                                          "yyyy-mm-dd HH:MM:SS") * " UTC"
        dose_var.attrib["valid_min"] = Float32(0.0)

        dose_raw_var = defVar(ds, "dose_rate_raw_mR_hr", Float32, ("longitude", "latitude"),
                              fillvalue=Float32(-9999))
        dose_raw_var[:, :] = Float32.(dose_raw)
        dose_raw_var.attrib["units"] = "mR/hr"
        dose_raw_var.attrib["long_name"] = "Raw dose rate at H+12 (total mixed FP)"
        dose_raw_var.attrib["valid_min"] = Float32(0.0)

        dep_var = defVar(ds, "deposition_Bq_m2", Float32, ("longitude", "latitude"),
                         fillvalue=Float32(-9999))
        dep_var[:, :] = Float32.(dep_bqm2)
        dep_var.attrib["units"] = "Bq/m2"
        dep_var.attrib["long_name"] = "Cs-137 surface deposition"
        dep_var.attrib["valid_min"] = Float32(0.0)

        # CRS
        crs_var = defVar(ds, "crs", Int32, ())
        crs_var.attrib["grid_mapping_name"] = "latitude_longitude"
        crs_var.attrib["long_name"] = "WGS84"
        crs_var.attrib["semi_major_axis"] = 6378137.0
        crs_var.attrib["inverse_flattening"] = 298.257223563

        dose_var.attrib["grid_mapping"] = "crs"
        dose_raw_var.attrib["grid_mapping"] = "crs"
        dep_var.attrib["grid_mapping"] = "crs"

        # Global attributes
        ds.attrib["title"] = "PREDICT WP3.3 - $scenario_label"
        ds.attrib["institution"] = "NuclearDetonation.jl"
        ds.attrib["source"] = "NuclearDetonation.jl Lagrangian particle dispersion"
        ds.attrib["Conventions"] = "CF-1.8"
        ds.attrib["yield_kt"] = YIELD_KT
        ds.attrib["activity_Cs137_Bq"] = TOTAL_ACTIVITY_BQ
        ds.attrib["activity_FP_Bq"] = height_mode == :high ? FP_ACTIVITY_BQ_AIRBURST : FP_ACTIVITY_BQ_SURFACE
        ds.attrib["n_particles"] = N_PARTICLES
        ds.attrib["detonation_time"] = Dates.format(det_time, "yyyy-mm-dd HH:MM:SS") * " UTC"
        ds.attrib["history"] = Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS") *
                               " - Created by predict_wp33_production.jl"
    end
    println("    Saved: $(basename(filename))")
end

# ============================================================================
# MET DELIVERY HELPERS — regrid, time-series, TOA, precip, CF NetCDF
# ============================================================================

"""
    sim_lonlat_axes(domain) -> (lon_axis, lat_axis)

Per-cell (lon, lat) for the simulation grid (nx_sim × ny_sim).  Built from
the simulation domain using `grid_to_latlon`, so it matches whatever
longitude convention (0–360 vs −180–180) the domain itself uses.
"""
function sim_lonlat_axes(domain)
    lon_axis = [Transport.grid_to_latlon(domain, Float64(i), 1.0)[2] for i in 1:domain.nx]
    lat_axis = [Transport.grid_to_latlon(domain, 1.0, Float64(j))[1] for j in 1:domain.ny]
    return lon_axis, lat_axis
end

"""
    regrid_field(field, lon_in, lat_in, lon_out, lat_out)

Bilinear interpolation of a (nx_in, ny_in) field onto (nx_out, ny_out)
using `Interpolations.LinearInterpolation`. Cells outside the input
extent are filled with `NaN`. Handles either ascending or descending
input axes (ERA5 latitude is typically N→S).
"""
function regrid_field(field::AbstractMatrix{<:Real},
                       lon_in::AbstractVector{<:Real}, lat_in::AbstractVector{<:Real},
                       lon_out::AbstractVector{<:Real}, lat_out::AbstractVector{<:Real})
    lon_v = Float64.(collect(lon_in))
    lat_v = Float64.(collect(lat_in))
    z = Float64.(field)

    if length(lon_v) > 1 && lon_v[2] < lon_v[1]
        lon_v = reverse(lon_v); z = z[end:-1:1, :]
    end
    if length(lat_v) > 1 && lat_v[2] < lat_v[1]
        lat_v = reverse(lat_v); z = z[:, end:-1:1]
    end

    itp = Interpolations.linear_interpolation((lon_v, lat_v), z,
        extrapolation_bc=NaN)

    out = Array{Float64}(undef, length(lon_out), length(lat_out))
    @inbounds for j in eachindex(lat_out), i in eachindex(lon_out)
        out[i, j] = itp(Float64(lon_out[i]), Float64(lat_out[j]))
    end
    return out
end

"""
    snapshot_field(snapshot, kind) -> Matrix

Return a 2-D (nx_sim, ny_sim) view of the requested snapshot quantity
summed over species (here: a single Cs-137 species).
  - `:dry`        — accumulated dry deposition  [Bq/m²]
  - `:wet`        — accumulated wet deposition  [Bq/m²]
  - `:total`      — total deposition (dry+wet)  [Bq/m²]
  - `:surf_conc`  — near-surface (k=1) air concentration [Bq/m³]
"""
function snapshot_field(snap, kind::Symbol)
    if kind === :dry
        return Float64.(dropdims(sum(snap.dry_deposition; dims=3); dims=3))
    elseif kind === :wet
        return Float64.(dropdims(sum(snap.wet_deposition; dims=3); dims=3))
    elseif kind === :total
        return Float64.(dropdims(sum(snap.total_deposition; dims=3); dims=3))
    elseif kind === :surf_conc
        c = @view snap.concentration[:, :, 1, :]
        return Float64.(dropdims(sum(c; dims=3); dims=3))
    else
        error("Unknown snapshot field kind: $kind")
    end
end

"""
    build_field_timeseries(snapshots, hours; kind, sim_lon, sim_lat, lon_out, lat_out)

Build a `(nx_out, ny_out, length(hours))` array by picking the snapshot
whose `time` matches `hours[k] * 3600` (within 1 s) and regridding it
onto the output grid.
"""
function build_field_timeseries(snapshots, hours::AbstractVector{<:Integer};
                                 kind::Symbol,
                                 sim_lon, sim_lat,
                                 lon_out, lat_out)
    out = Array{Float64}(undef, length(lon_out), length(lat_out), length(hours))
    # Cumulative fields (deposition) can fall back to the last preceding
    # snapshot when the simulation terminates early (all particle masses
    # underflow to zero); for concentration that's never sensible.
    cumulative = kind in (:dry, :wet, :total)
    for (k, hr) in enumerate(hours)
        target_t = Float64(hr) * 3600.0
        idx = findfirst(s -> abs(Float64(s.time) - target_t) < 1.0, snapshots)
        if isnothing(idx) && cumulative
            # Fall back to most recent snapshot whose time < target_t
            best = 0
            for (i, s) in enumerate(snapshots)
                if Float64(s.time) <= target_t + 1.0
                    best = i
                end
            end
            if best > 0
                idx = best
                @info "Snapshot H+$hr missing; using H+$(Int(round(snapshots[best].time/3600))) ($kind)"
            end
        end
        if isnothing(idx)
            @warn "No snapshot at H+$hr; filling with NaN"
            fill!(view(out, :, :, k), NaN)
            continue
        end
        f_sim = snapshot_field(snapshots[idx], kind)
        out[:, :, k] = regrid_field(f_sim, sim_lon, sim_lat, lon_out, lat_out)
    end
    return out
end

"""
    compute_toa(snapshots, total_dep_series, surf_conc_series, hours;
                dep_threshold=100.0, conc_int_threshold=1000.0)

Time-of-arrival (hours since detonation) at each output grid cell.
Triggers:
  (a) ∫ near-surface air concentration > 1000 Bq·s/m³
  (c) total ground deposition          > 100 Bq/m²

`hours` are the dose hours (cumulative across the simulation). The
integral over conc is built via trapezoidal integration in time from
hour 0 to `hours[k]`. Cells that never trigger return `NaN`.
"""
function compute_toa(total_dep_series::Array{Float64,3},
                      surf_conc_series::Array{Float64,3},
                      hours::AbstractVector{<:Integer};
                      dep_threshold::Real=100.0,
                      conc_int_threshold::Real=1000.0)
    nx, ny, nt = size(total_dep_series)
    @assert size(surf_conc_series) == size(total_dep_series)
    toa = fill(NaN, nx, ny)

    # Trapezoidal integral of surface concentration in time (Bq·s/m³).
    conc_int = Array{Float64}(undef, nx, ny, nt)
    t_prev = 0.0
    c_prev = zeros(Float64, nx, ny)
    running = zeros(Float64, nx, ny)
    for k in 1:nt
        t_now = Float64(hours[k]) * 3600.0
        dt = t_now - t_prev
        c_now = view(surf_conc_series, :, :, k)
        running .+= 0.5 .* dt .* (c_prev .+ c_now)
        conc_int[:, :, k] .= running
        t_prev = t_now
        c_prev = Array(c_now)
    end

    for k in 1:nt
        h = Float64(hours[k])
        for j in 1:ny, i in 1:nx
            isnan(toa[i, j]) || continue
            d = total_dep_series[i, j, k]
            c = conc_int[i, j, k]
            if (isfinite(d) && d > dep_threshold) ||
               (isfinite(c) && c > conc_int_threshold)
                toa[i, j] = h
            end
        end
    end
    return toa
end

"""
    aggregate_precip_12h(precip_path, det_time, precip_hours, lon_out, lat_out)

Read hourly precipitation from `era5_precip_hourly.nc` and return a
`(nx_out, ny_out, length(precip_hours))` array of 12-h accumulated mm.
Window `k` is `(precip_hours[k]-12, precip_hours[k]]` after detonation.
"""
function aggregate_precip_12h(precip_path::AbstractString, det_time::Dates.DateTime,
                               precip_hours::AbstractVector{<:Integer},
                               lon_out, lat_out)
    NCDataset(precip_path) do ds
        lon_in = Float64.(ds["longitude"][:])
        lat_in = Float64.(ds["latitude"][:])
        time_units = ds["time"].attrib["units"]   # e.g. "hours since 2024-01-06 00:00:00"
        m = match(r"hours since (\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2})", time_units)
        @assert m !== nothing "Unexpected precip time units: $time_units"
        epoch = Dates.DateTime(m.captures[1] * "T" * m.captures[2])
        det_offset_hrs = (det_time - epoch) / Dates.Hour(1)
        # ds["time"] auto-converts to DateTime; use .var to get raw numeric values
        times = Float64.(ds["time"].var[:])   # hours since epoch
        pr = Float64.(ds["precipitation_rate"][:, :, :])  # NCDatasets reverses dims to Fortran order

        # Determine variable axis order: dims for precipitation_rate
        dim_names = NCDatasets.dimnames(ds["precipitation_rate"])
        # Reshape to (lon, lat, time) if needed
        if dim_names == ("time", "latitude", "longitude")
            pr = permutedims(pr, (3, 2, 1))
        elseif dim_names == ("longitude", "latitude", "time")
            # already correct
        elseif dim_names == ("latitude", "longitude", "time")
            pr = permutedims(pr, (2, 1, 3))
        else
            @warn "Unexpected precip dims order: $dim_names — assuming (time, lat, lon)"
            pr = permutedims(pr, (3, 2, 1))
        end

        out = Array{Float64}(undef, length(lon_out), length(lat_out), length(precip_hours))
        for (k, hr) in enumerate(precip_hours)
            t_end_h = det_offset_hrs + hr        # absolute file-hour at window end
            t_start_h = t_end_h - 12              # window start (exclusive)
            # accumulate hourly rates whose timestamp falls in (t_start_h, t_end_h]
            mask = (times .> t_start_h - 1e-6) .& (times .<= t_end_h + 1e-6)
            # rates are mm/h; assume 1-h dt
            accum = dropdims(sum(@view(pr[:, :, mask]); dims=3); dims=3)
            out[:, :, k] = regrid_field(accum, lon_in, lat_in, lon_out, lat_out)
        end
        return out
    end
end

"""
    way_wigner(hr) = hr^(-1.2)
"""
way_wigner_decay(hr::Real) = hr <= 0 ? 1.0 : Float64(hr)^(-1.2)

"""
    save_met_netcdf(filename, lon_grid, lat_grid, dose_hours, dep_hours, precip_hours,
                    dose_rate_t, dry_dep_t, wet_dep_t, total_dep_t, precip_t, toa,
                    det_time, scenario_label, height_mode, fp_activity_bq)

CF-1.8 NetCDF writer matching the WP3.3 MET delivery spec.
"""
function save_met_netcdf(filename::AbstractString,
                         lon_grid, lat_grid,
                         dose_hours::AbstractVector{<:Integer},
                         dep_hours::AbstractVector{<:Integer},
                         precip_hours::AbstractVector{<:Integer},
                         dose_rate_t::Array{Float64,3},
                         dry_dep_t::Array{Float64,3},
                         wet_dep_t::Array{Float64,3},
                         total_dep_t::Array{Float64,3},
                         precip_t::Array{Float64,3},
                         toa::Matrix{Float64},
                         det_time::Dates.DateTime,
                         scenario_label::AbstractString,
                         height_mode::Symbol,
                         fp_activity_bq::Real)
    det_iso = Dates.format(det_time, "yyyy-mm-ddTHH:MM:SS")
    NCDataset(filename, "c") do ds
        defDim(ds, "lon", length(lon_grid))
        defDim(ds, "lat", length(lat_grid))
        defDim(ds, "time_dose", length(dose_hours))
        defDim(ds, "time_dep", length(dep_hours))
        defDim(ds, "time_precip", length(precip_hours))
        defDim(ds, "nv", 2)

        lon_var = defVar(ds, "lon", Float64, ("lon",))
        lon_var[:] = lon_grid
        lon_var.attrib["units"] = "degrees_east"
        lon_var.attrib["standard_name"] = "longitude"
        lon_var.attrib["long_name"] = "Longitude"
        lon_var.attrib["axis"] = "X"

        lat_var = defVar(ds, "lat", Float64, ("lat",))
        lat_var[:] = lat_grid
        lat_var.attrib["units"] = "degrees_north"
        lat_var.attrib["standard_name"] = "latitude"
        lat_var.attrib["long_name"] = "Latitude"
        lat_var.attrib["axis"] = "Y"

        time_units = "seconds since " * det_iso
        td = defVar(ds, "time_dose", Float64, ("time_dose",))
        td[:] = Float64.(dose_hours) .* 3600.0
        td.attrib["units"] = time_units
        td.attrib["standard_name"] = "time"
        td.attrib["calendar"] = "gregorian"
        td.attrib["long_name"] = "Time after detonation (dose rate snapshots)"
        td.attrib["axis"] = "T"

        tdp = defVar(ds, "time_dep", Float64, ("time_dep",))
        tdp[:] = Float64.(dep_hours) .* 3600.0
        tdp.attrib["units"] = time_units
        tdp.attrib["standard_name"] = "time"
        tdp.attrib["calendar"] = "gregorian"
        tdp.attrib["long_name"] = "Time after detonation (cumulative deposition snapshots)"
        tdp.attrib["axis"] = "T"

        tp = defVar(ds, "time_precip", Float64, ("time_precip",))
        tp[:] = Float64.(precip_hours) .* 3600.0
        tp.attrib["units"] = time_units
        tp.attrib["standard_name"] = "time"
        tp.attrib["calendar"] = "gregorian"
        tp.attrib["long_name"] = "Time after detonation (precipitation window end)"
        tp.attrib["axis"] = "T"
        tp.attrib["bounds"] = "time_precip_bounds"

        tpb = defVar(ds, "time_precip_bounds", Float64, ("nv", "time_precip"))
        for k in eachindex(precip_hours)
            tpb[1, k] = (Float64(precip_hours[k]) - 12.0) * 3600.0
            tpb[2, k] =  Float64(precip_hours[k])         * 3600.0
        end

        # CRS
        crs_var = defVar(ds, "crs", Int32, ())
        crs_var.attrib["grid_mapping_name"] = "latitude_longitude"
        crs_var.attrib["long_name"] = "WGS84"
        crs_var.attrib["semi_major_axis"] = 6378137.0
        crs_var.attrib["inverse_flattening"] = 298.257223563

        # NCDatasets reverses dim declarations on disk to match CF C-order:
        # declaring `("lon","lat","time")` here means the file dims (CF order)
        # become `("time","lat","lon")`. The Julia array we assign therefore
        # keeps shape (lon, lat, time) — no permute needed.
        function _add_field(name, dims_jl, data, units, long_name; extra=Dict{String,Any}())
            v = defVar(ds, name, Float32, dims_jl, fillvalue=Float32(NaN))
            arr = Float32.(data)
            if length(dims_jl) == 2
                v[:, :] = arr
            else
                v[:, :, :] = arr
            end
            v.attrib["units"] = units
            v.attrib["long_name"] = long_name
            v.attrib["coordinates"] = "lat lon"
            v.attrib["grid_mapping"] = "crs"
            for (k, val) in extra
                v.attrib[k] = val
            end
            return v
        end

        # Time of arrival (2-D, lon×lat — file order will be (lat, lon))
        toa_arr = Float32.(toa)
        v_toa = defVar(ds, "time_of_arrival", Float32, ("lon", "lat"), fillvalue=Float32(NaN))
        v_toa[:, :] = toa_arr
        v_toa.attrib["units"] = "h"
        v_toa.attrib["long_name"] = "Time of arrival"
        v_toa.attrib["comment"] = "Earliest hour after detonation when integrated near-surface air concentration > 1000 Bq s/m^3 OR total ground contamination > 100 Bq/m^2"
        v_toa.attrib["coordinates"] = "lat lon"
        v_toa.attrib["grid_mapping"] = "crs"

        _add_field("dose_rate", ("lon", "lat", "time_dose"), dose_rate_t,
                   "mR h-1",
                   "Gamma dose rate at 1 m above ground (Way–Wigner decay applied)";
                   extra=Dict("comment" => "Total mixed fission product dose rate (Cs-137 tracer scaled to total FP × Way–Wigner t^-1.2)"))

        _add_field("dry_deposition", ("lon", "lat", "time_dep"), dry_dep_t,
                   "Bq m-2", "Accumulated dry deposition (Cs-137)";
                   extra=Dict("cell_methods" => "time_dep: sum"))
        _add_field("wet_deposition", ("lon", "lat", "time_dep"), wet_dep_t,
                   "Bq m-2", "Accumulated wet deposition (Cs-137)";
                   extra=Dict("cell_methods" => "time_dep: sum"))
        _add_field("total_deposition", ("lon", "lat", "time_dep"), total_dep_t,
                   "Bq m-2", "Accumulated total (dry+wet) deposition (Cs-137)";
                   extra=Dict("cell_methods" => "time_dep: sum"))
        _add_field("accumulated_precip_12h", ("lon", "lat", "time_precip"), precip_t,
                   "mm", "12-hour accumulated precipitation (ERA5)";
                   extra=Dict("cell_methods" => "time_precip: sum"))

        # Global attributes
        ds.attrib["Conventions"]      = "CF-1.8"
        ds.attrib["title"]            = "PREDICT WP3.3 — $scenario_label"
        ds.attrib["institution"]      = "Dublin City University"
        ds.attrib["source"]           = "NuclearDetonation.jl Lagrangian particle dispersion"
        ds.attrib["group"]            = "EPA_Ireland"
        ds.attrib["yield_kt"]         = YIELD_KT
        ds.attrib["activity_Cs137_Bq_H1"] = TOTAL_ACTIVITY_BQ
        ds.attrib["activity_FP_Bq_H1"]    = Float64(fp_activity_bq)
        ds.attrib["n_particles"]      = N_PARTICLES
        ds.attrib["detonation_time"]  = det_iso * " UTC"
        ds.attrib["burst_mode"]       = String(height_mode)
        ds.attrib["surface_burst_calibration"] = "Trinity (1945, 21 kT)"
        ds.attrib["air_burst_calibration"]     = "Plumbbob Doppler (1957, 11 kT)"
        ds.attrib["history"]          = Dates.format(Dates.now(Dates.UTC), "yyyy-mm-ddTHH:MM:SS") *
                                        "Z — Created by predict_wp33_production.jl"
        ds.attrib["spec_reference"]   = "WP3.3 Calculated fields to be delivered to MET"
    end
    println("    Saved (MET): $(basename(filename))")
end

# ============================================================================
# MAIN LOOP
# ============================================================================

results_table = []

for sc in SCENARIOS
    println("\n" * "="^70)
    println("SCENARIO S$(sc.id): $(sc.loc) - $(sc.date)")
    println("="^70)

    println("  Loading ERA5 met data...")
    t_load = @elapsed cache = load_scenario_cache(sc)
    @printf("  Met loaded in %.1f s\n", t_load)

    height_modes = let hf = isnothing(CLI_HEIGHT_OVERRIDE) ?
                            get(ENV, "TEST_HEIGHT", "both") : CLI_HEIGHT_OVERRIDE
        hf == "low"  ? [:low]  :
        hf == "high" ? [:high] :
                       [:low, :high]
    end
    for height_mode in height_modes
        height_label = height_mode == :low ? "low" : "high"
        println("\n  --- $(uppercase(height_label)) burst ---")

        println("  Running $(SIM_HOURS)h simulation...")
        t_sim = @elapsed begin
            state, domain, cloud, burst_label, snapshots = run_scenario(cache, height_mode)
        end
        println("  Snapshots captured: $(length(snapshots))")
        @printf("  Simulation done in %.1f s\n", t_sim)
        println("  Cloud: top=$(round(cloud.cap_height, digits=0))m, " *
                "stem=$(round(cloud.stem_height, digits=0))m")
        println("  Deposition events: $(length(state.deposition_log))")
        println("  Particles remaining: $(length(state.ensemble.particles))")

        println("  Building dose fields...")
        fields = build_dose_fields(state, domain, cache, height_mode)
        max_dose = maximum(fields.dose_smooth)
        max_dep = maximum(fields.dep_bqm2)
        @printf("  Max dose rate: %.1f mR/h (smooth), %.1f mR/h (raw)\n",
                max_dose, maximum(fields.dose_raw))
        @printf("  Max Cs-137 deposition: %.1f Bq/m2\n", max_dep)

        # Deposited fraction
        total_deposited = sum(evt.mass for evt in state.deposition_log; init=0.0)
        dep_frac = total_deposited / TOTAL_ACTIVITY_BQ * 100.0
        @printf("  Deposited fraction: %.1f%%\n", dep_frac)

        # Plume bearing (direction of max deposition from detonation site)
        if max_dose > 0
            max_idx = argmax(fields.dose_smooth)
            max_lon = fields.lon_grid[max_idx[1]]
            max_lat = fields.lat_grid[max_idx[2]]
            bearing = atand(max_lon - sc.lon, max_lat - sc.lat)
            bearing < 0 && (bearing += 360)
        else
            bearing = NaN
        end

        # Save outputs
        base = "predict_wp33_S$(sc.id)_$(height_label)"
        det_time = Dates.Dates.DateTime(sc.date) + Dates.Hour(DETONATION_HOUR)
        title_str = "S$(sc.id) $(sc.loc) $(sc.date) - $(burst_label), H+12"
        scenario_label = "S$(sc.id) $(sc.loc) $(sc.date) $(burst_label)"

        plot_dose_contours(joinpath(OUTPUT_DIR, base * ".png"),
            fields.lon_grid, fields.lat_grid, fields.dose_smooth,
            sc.lat, sc.lon, title_str)

        # ---- MET delivery (CF-1.8 time-series NetCDF) ----
        println("  Building MET delivery time series...")
        sim_lon, sim_lat = sim_lonlat_axes(domain)
        lon_out = fields.lon_grid
        lat_out = fields.lat_grid

        total_dep_dose_t = build_field_timeseries(snapshots, DOSE_HOURS;
            kind=:total, sim_lon=sim_lon, sim_lat=sim_lat,
            lon_out=lon_out, lat_out=lat_out)
        surf_conc_dose_t = build_field_timeseries(snapshots, DOSE_HOURS;
            kind=:surf_conc, sim_lon=sim_lon, sim_lat=sim_lat,
            lon_out=lon_out, lat_out=lat_out)

        fp_scale_use = height_mode == :high ? FP_SCALE_AIRBURST : FP_SCALE_SURFACE
        fp_act_use   = height_mode == :high ? FP_ACTIVITY_BQ_AIRBURST : FP_ACTIVITY_BQ_SURFACE
        dose_rate_t = similar(total_dep_dose_t)
        for k in eachindex(DOSE_HOURS)
            decay = way_wigner_decay(DOSE_HOURS[k])
            @views dose_rate_t[:, :, k] .= total_dep_dose_t[:, :, k] .*
                (fp_scale_use * K_DOSE * decay * MSV_TO_MR)
        end

        dry_dep_t   = build_field_timeseries(snapshots, DEP_HOURS;
            kind=:dry, sim_lon=sim_lon, sim_lat=sim_lat,
            lon_out=lon_out, lat_out=lat_out)
        wet_dep_t   = build_field_timeseries(snapshots, DEP_HOURS;
            kind=:wet, sim_lon=sim_lon, sim_lat=sim_lat,
            lon_out=lon_out, lat_out=lat_out)
        total_dep_t = build_field_timeseries(snapshots, DEP_HOURS;
            kind=:total, sim_lon=sim_lon, sim_lat=sim_lat,
            lon_out=lon_out, lat_out=lat_out)

        precip_path = joinpath(ERA5_BASE, sc.era5, "era5_precip_hourly.nc")
        precip_t = if isfile(precip_path)
            aggregate_precip_12h(precip_path, det_time, PRECIP_HOURS, lon_out, lat_out)
        else
            @warn "Missing $precip_path — precip field will be NaN"
            fill(NaN, length(lon_out), length(lat_out), length(PRECIP_HOURS))
        end

        toa = compute_toa(total_dep_dose_t, surf_conc_dose_t, DOSE_HOURS)
        n_toa = count(!isnan, toa)
        println("    TOA non-NaN cells: $n_toa / $(length(toa))")

        save_met_netcdf(joinpath(OUTPUT_DIR, base * ".nc"),
            lon_out, lat_out, DOSE_HOURS, DEP_HOURS, PRECIP_HOURS,
            dose_rate_t, dry_dep_t, wet_dep_t, total_dep_t, precip_t, toa,
            det_time, scenario_label, height_mode, fp_act_use)

        push!(results_table, (
            scenario = "S$(sc.id)", location = sc.loc, date = string(sc.date),
            height = height_label, max_dose_mRh = max_dose,
            max_dep_Bqm2 = max_dep, dep_frac_pct = dep_frac,
            bearing_deg = bearing, sim_time_s = t_sim))

        state = nothing
        GC.gc(false)
    end

    cache = nothing
    GC.gc()
end

# ============================================================================
# SUMMARY TABLE
# ============================================================================

println("\n\n" * "="^70)
println("SUMMARY - All Scenarios")
println("="^70)
@printf("%-6s %-8s %-12s %-6s %10s %12s %8s %8s %7s\n",
    "ID", "Loc", "Date", "Burst", "MaxDose", "MaxDep", "Dep%", "Bear", "Time")
@printf("%-6s %-8s %-12s %-6s %10s %12s %8s %8s %7s\n",
    "", "", "", "", "mR/h", "Bq/m2", "", "deg", "s")
println("-"^70)
for r in results_table
    @printf("%-6s %-8s %-12s %-6s %10.1f %12.1f %7.1f%% %7.0f  %6.0fs\n",
        r.scenario, r.location, r.date, r.height,
        r.max_dose_mRh, r.max_dep_Bqm2, r.dep_frac_pct,
        r.bearing_deg, r.sim_time_s)
end
println("="^70)
println("\nAll outputs saved to: $OUTPUT_DIR")
println("Done!")
