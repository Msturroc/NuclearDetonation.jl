#!/usr/bin/env julia
# US-test calibration BIPOP-CMA-ES — parameterised over Trinity, Harry, Small Boy.
# Logic copied from smoky_cmaes_particle_size.jl. Per-test specifics live in a
# TEST_CONFIG block driven by ARGS[1]: yield, lat/lon, surface elevation, date,
# observation loader, ERA5 directory, and warm-start cloud heights (scaled from
# Smoky reference by yield^0.215 per Glasstone & Dolan).
#
# 23 parameters. Combined score (CMA-ES ranking):
#   default tests: 25% FMS + 15% shape + 20% bearing + 10% extent + 30% TOA
#   doppler:       45% FMS + 10% shape + 20% bearing +  5% extent + 20% TOA
# Plus a hard gate: any solution with bearing_score < 0.5 (≳45° plume mis-aim)
# is rejected with loss=2.0 so geometric-cheating spreads cannot win.
#
# Usage:
#   julia --threads=12 --project=../.. cmaes_calibration.jl {trinity|harry|smallboy} [OU|RW]
#
# Environment variables:
#   MAX_EVALS=6000       Default budget (matches Nancy/Smoky calibration depth)
#   WARM_START=1         Start from Smoky-best yield-scaled (default: 1)

using LinearAlgebra
using Random
using Statistics
using Printf
using Base.Threads
using NCDatasets
using StaticArrays
using Dates
using NuclearDetonation
using NuclearDetonation.Transport

# ============================================================================
# PARSE ARGUMENTS + PER-TEST CONFIG
# ============================================================================

length(ARGS) >= 1 || error("Usage: julia cmaes_calibration.jl {trinity|harry|smallboy|doppler} [OU|RW]")
const TEST_NAME = lowercase(ARGS[1])

const TURB_SCHEME = if length(ARGS) >= 2 && uppercase(ARGS[2]) == "OU"
    :OU
else
    :RW
end
const TURB_NAME = TURB_SCHEME == :OU ? "Ornstein-Uhlenbeck" : "RandomWalk"

# Test-specific config. Surface elevations: White Sands ~1216 m, NTS ~1409 m.
const TEST_CONFIG = if TEST_NAME == "trinity"
    (
        label = "Trinity",
        yield_kt = 21.0,
        source_lat = 33.6773,
        source_lon = -106.4754,
        surface_elev_m = 1216.0,
        start_dt = Dates.DateTime(1945, 7, 16, 11, 29),
        load_obs = Transport.load_trinity_observations,
        era5_dir = "/run/media/marc/e34b80f3-0992-4981-a17f-e396750ea8b4/era5_calibration/scenario_1_trinity_19450716",
    )
elseif TEST_NAME == "harry"
    (
        label = "Harry",
        yield_kt = 32.0,
        source_lat = 37.0980,
        source_lon = -116.0228,
        surface_elev_m = 1409.0,
        start_dt = Dates.DateTime(1953, 5, 19, 12, 5),
        load_obs = Transport.load_harry_observations,
        era5_dir = "/run/media/marc/e34b80f3-0992-4981-a17f-e396750ea8b4/era5_calibration/scenario_2_harry_19530519",
    )
elseif TEST_NAME == "smallboy"
    (
        label = "Small Boy",
        yield_kt = 1.65,
        # GZ corrected 2026-05-27: was 37.2250 / -115.9570 / 1409 m (Yucca Flat
        # coords); SmallBoy was on Frenchman Flat (Area 5) ~47 km SOUTH of that.
        source_lat = 36.8070,
        source_lon = -115.9350,
        surface_elev_m = 954.0,                          # Frenchman Lake bed (~3,130 ft)
        start_dt = Dates.DateTime(1962, 7, 14, 18, 30),  # 11:30 PDT (Quinn 1984 NVO-285, Wikipedia Operation Sunbeam); was 17:30
        load_obs = Transport.load_smallboy_observations,
        era5_dir = "/run/media/marc/e34b80f3-0992-4981-a17f-e396750ea8b4/era5_calibration/scenario_3_smallboy_19620714",
    )
elseif TEST_NAME == "doppler"
    # Plumbbob Doppler — DASA-1251 Vol I, p.316:
    #   11 kT, balloon at 1,500 ft (457 m) HOB, NTS Yucca Flat Area Ta.
    #   GZ 37°05'12" N / 116°01'25" W, site elev 4,230 ft (1,289 m).
    #   0530 PDT = 12:30 UTC on 23 Aug 1957.
    # Closest historical fission air-burst analogue to the WP1.3 10 kT /
    # 710 m air-burst regime; provides calibration for `:high` production.
    (
        label = "Doppler",
        yield_kt = 11.0,
        source_lat = 37.0867,
        source_lon = -116.0237,
        surface_elev_m = 1289.0,
        start_dt = Dates.DateTime(1957, 8, 23, 12, 30),
        load_obs = Transport.load_doppler_observations,
        era5_dir = "/run/media/marc/e34b80f3-0992-4981-a17f-e396750ea8b4/era5_calibration/scenario_4_doppler_19570823",
    )
else
    error("Unknown test '$TEST_NAME'. Expected: trinity, harry, smallboy, doppler.")
end

# Glasstone-Dolan: cloud heights scale as W^0.215. Smoky reference at 44 kT.
const YIELD_SCALE = (TEST_CONFIG.yield_kt / 44.0)^0.215

println("="^70)
println("$(uppercase(TEST_CONFIG.label)) calibration BIPOP-CMA-ES — $(TURB_NAME) ($(nthreads()) threads)")
println("Yield: $(TEST_CONFIG.yield_kt) kT   Yield-scale factor: $(round(YIELD_SCALE, digits=3))")
println("Release: ($(TEST_CONFIG.source_lat), $(TEST_CONFIG.source_lon))   Surface elev $(TEST_CONFIG.surface_elev_m) m")
println("Start: $(TEST_CONFIG.start_dt)")
println("ERA5 dir: $(TEST_CONFIG.era5_dir)")
println("="^70)

# ============================================================================
# PARAMETER BOUNDS — v5: bearing penalty, tighter particle/physics bounds
# ============================================================================

const PARAM_NAMES = [
    "d_median_fine", "sigma_g_fine", "d_median_coarse", "sigma_g_coarse", "frac_fine",
    "frac_lower", "frac_middle",
    "sigma_w_scale", "sigma_h_scale", "h_diff_scale", "tl_scale",
    "vd_scale", "vgrav_scale", "omega_scale", "mixing_height_scale", "tmix_scale",
    "surface_height_scale", "roughness_scale",
    "activity_scale", "smooth_sigma",
    "stem_top_m", "cap_mid_m", "cloud_top_m"
]

const LB = Float64[
    20.0, 1.1, 80.0, 1.1, 0.05,          # particle size: d_fine≥20 μm (settles from 5km in <12h), d_coarse≥80 μm
    0.01, 0.01,                            # layer fractions (normalised in rho_core)
    0.01, 0.1, 0.05, 0.1,                # turbulence: sigma_w, sigma_h, h_diff, tl_scale
    0.1, 0.1, 0.1, 0.1, 0.1,             # physics: vd, vgrav, omega, mixing_height, tmix
    0.1, 0.1,                              # deposition: surface_height, roughness
    10.0, 0.5,                             # calibration: activity ×[10,200]e15 Bq, smooth σ [0.5,5] cells
    # Layer heights (m AGL). Wider lower bound to accommodate Small Boy (1.65 kT, cloud top ~5 km).
    400.0, 1500.0, 3000.0                  # stem_top, cap_mid, cloud_top
]

const UB = Float64[
    100.0, 3.5, 200.0, 5.0, 0.70,        # particle size: d_fine≤100, d_coarse≤200, sigma_g_coarse≤5, frac_fine≤0.70
    0.50, 0.50,                            # layer fractions: cap at 0.50 each (ensure ≥15% upper cap)
    10.0, 8.0, 2.0, 10.0,                # turbulence: sigma_w≤10, sigma_h, h_diff, tl_scale≤10
    20.0, 10.0, 5.0, 5.0, 5.0,            # physics: vd≤20, vgrav≤10, omega≤5, mixing_height≤5, tmix≤5
    10.0, 5.0,                             # deposition: surface_height≤10, roughness
    500.0, 5.0,                            # calibration: activity ×[10,500]e15 Bq, smooth σ [0.5,5] cells
    8000.0, 9000.0, 12000.0               # layer heights (m AGL): stem_top≤8km, cap_mid, cloud_top≤12km
]

if TEST_NAME == "doppler"
    # Doppler-specific bounds — REVISED from particle-trajectory diagnostic
    # (May 2026, doppler_trajectory_diagnostic.jl). The earlier 14-17 km
    # "jet band" hypothesis was wrong: empirically 61/200 particles from the
    # Gen-56 vector (cloud at 8-10 km, d_fine=76 μm, vd=13) reach the obs
    # NE-peak bbox at median altitude 6.3 km, median time 4.9 h. FMS was
    # 0.21 only because (i) too much mass blob-deposits at GZ (vd=13 is
    # extreme) and (ii) jet-borne particles overshoot to lon -107 instead
    # of depositing within obs.
    # Revised box: cloud at 4-12 km AGL, d_fine ∈ [15, 45] μm (settles
    # ~24 cm/s, falls ~6 km in 7 h — i.e. reaches obs and deposits there
    # rather than overshooting), frac_fine ∈ [0.60, 0.95], frac_lower kept
    # very low so the GZ blob does not return, vd moderate (≤5), vgrav up
    # to 3 (allow modest gravity boost to keep deposition within obs).
    LB[1]  = 10.0    # d_median_fine ≥ 10 μm
    UB[1]  = 100.0   # d_median_fine ≤ 100 μm (encompasses Gen-56 at 76 μm)
    LB[3]  = 80.0    # d_median_coarse: keep wide
    UB[3]  = 250.0
    LB[5]  = 0.30    # frac_fine ≥ 0.30 (encompasses Gen-56 at 0.50)
    UB[5]  = 0.95
    LB[6]  = 0.01    # frac_lower ≥ 0.01
    UB[6]  = 0.20    # frac_lower ≤ 0.20 (cap GZ blob contribution)
    LB[12] = 0.5     # vd_scale ≥ 0.5
    UB[12] = 20.0    # vd_scale ≤ 20 (encompasses Gen-56 at 13)
    LB[13] = 0.5     # vgrav_scale ≥ 0.5
    UB[13] = 3.0     # vgrav_scale ≤ 3 (allow modest gravity boost)
    LB[21] = 3000.0  # stem_top  ≥ 3 km AGL
    UB[21] = 9000.0  # stem_top  ≤ 9 km
    LB[22] = 5000.0  # cap_mid   ≥ 5 km
    UB[22] = 11000.0 # cap_mid   ≤ 11 km
    LB[23] = 7000.0  # cloud_top ≥ 7 km (covers Gen-56's actual basin)
    UB[23] = 13000.0 # cloud_top ≤ 13 km
    println("   Doppler (revised, post-trajectory): cloud at 4-12 km,")
    println("            d_fine ∈ [$(LB[1]), $(UB[1])] μm, vd ∈ [$(LB[12]), $(UB[12])],")
    println("            vgrav ∈ [$(LB[13]), $(UB[13])], frac_fine ∈ [$(LB[5]), $(UB[5])],")
    println("            frac_lower ∈ [$(LB[6]), $(UB[6])]")
elseif TEST_NAME == "smallboy"
    # SmallBoy-specific bound relaxation. v3 (overnight, 2026-05-27):
    # final2 run (58.7%, NE-going plume) had 8 parameters pinned or near
    # bounds. Extend each in the direction CMA-ES wanted to push, for
    # the 6000-eval overnight run.
    LB[2]  = 1.05    # sigma_g_fine ≥ 1.05
    UB[3]  = 300.0   # d_median_coarse ≤ 300 (was 200; final2 hit UB)
    UB[4]  = 8.0     # sigma_g_coarse ≤ 8 (was 5; final2 at 4.47)
    UB[8]  = 20.0    # sigma_w_scale ≤ 20 (was 10; final2 hit UB)
    UB[9]  = 15.0    # sigma_h_scale ≤ 15
    UB[10] = 4.0     # h_diff_scale ≤ 4 (was 2; final2 at 1.86)
    # vgrav_scale floor lowered 0.01 → 1e-4 (4000-eval run pinned it at 0.01,
    # wanting less settling to extend the far-field 0.1 mR/h tail). Meaningful
    # now that param 13 is searched in log10 space (LOG_MASK) — the two extra
    # decades get uniform sampling resolution instead of being crushed into
    # ~0.1% of a linear domain. Physically this suspends the coarse mode further
    # than 246 μm particles really do; the optimiser may prefer it, but watch
    # that it is buying far-field reach honestly (via fines aloft) not by faking
    # coarse-particle levitation.
    LB[13] = 1e-4    # vgrav_scale ≥ 1e-4 (was 0.01; 4000-eval hit 0.01)
    UB[14] = 10.0    # omega_scale ≤ 10 (was 5; final2 hit UB)
    UB[16] = 10.0    # tmix_scale ≤ 10 (was 5; final2 at 3.79)
    UB[18] = 10.0    # roughness_scale ≤ 10 (was 5; final2 at 4.24)
    LB[19] = 1.0     # activity_scale ≥ 1.0
    UB[19] = 1000.0  # activity_scale ≤ 1000 (was 500; lets the dilute far tail
                     # cross 0.1 mR/h. Honest lever for extent, log10-searched.)
    UB[20] = 10.0    # smooth_sigma ≤ 10
    LB[21] = 100.0   # stem_top_m ≥ 100 m
    LB[22] = 500.0   # cap_mid ≥ 500 m (was 1000; final2 hit LB)
    LB[23] = 2000.0  # cloud_top ≥ 2 km
    println("   SmallBoy (bound-relaxed v3): extended UB for d_coarse, sigma_g_coarse,")
    println("            sigma_w, h_diff, omega, tmix, roughness; LB lowered for vgrav, cap_mid")
end

const N_DIM = length(LB)
const DOMAIN_WIDTH = UB .- LB

# ============================================================================
# SEARCH-SPACE REPARAMETERISATION (log10 for multiplicative-scale parameters)
# ============================================================================
# Parameters 8–19 are all strictly-positive multiplicative scales (turbulence,
# physics, deposition, activity) spanning 2–4 orders of magnitude. In LINEAR
# search space CMA-ES wastes nearly all its resolution on the high decade — e.g.
# vgrav_scale ∈ [0.01, 10] gives the entire [0.01, 1] sub-range only ~10% of the
# domain, so σ collapses before small settling values can be explored, and the
# optimiser pins the lower bound (as the SmallBoy 4000-eval run did at 0.01).
#
# We run CMA-ES in log10 space for these params: bounds, x0, sampling, and the
# internal best-ever vector are all ENCODED, while rho_core, checkpoints, warm
# starts, LHS, and NC_EXPORT stay in PHYSICAL units. The transform is a pure
# reparameterisation — it changes how the search explores, never what the model
# computes — so Trinity/Harry/Doppler are mathematically unaffected (Nancy/Smoky
# are separate scripts). decode∘encode == identity to float precision.
const LOG_MASK = let m = falses(N_DIM)
    for j in 8:19; m[j] = true; end   # sigma_w..activity_scale: the scale block
    m
end
encode_params(x::AbstractVector) = Float64[LOG_MASK[j] ? log10(x[j]) : x[j] for j in 1:N_DIM]
decode_params(s::AbstractVector) = Float64[LOG_MASK[j] ? 10.0^s[j] : s[j] for j in 1:N_DIM]
const LB_S = encode_params(LB)
const UB_S = encode_params(UB)

# Warm start: Smoky-best vector. Cloud heights scaled by yield^0.215 (Glasstone-Dolan).
# Smoky reference AGL: stem_top 1822 m, cap_mid 5541 m, cloud_top 9259 m.
const WARM_START_PARAMS = clamp.([
    127.552, 2.669, 141.861, 2.523, 0.8652,   # particle size (Nancy-OU best — yield-insensitive)
    0.15, 0.35,                                 # layer fractions (DASA-1251: 15% stem, 35% lower cap, 50% upper cap)
    4.028, 2.220, 0.2055, 4.458,               # turbulence
    4.397, 0.5532, 2.557, 4.105, 1.290,        # physics
    1.554, 1.174,                               # deposition
    48.418, 2.172,                              # calibration: activity=48.418e15, σ=2.172 cells
    1822.0 * YIELD_SCALE, 5541.0 * YIELD_SCALE, 9259.0 * YIELD_SCALE,   # cloud heights scaled to test yield
], LB, UB)

# ============================================================================
# PARTICLE SIZE DISTRIBUTION FUNCTIONS
# ============================================================================

function snap_settling_velocity(d_um::Float64)
    snap_d = [2.2, 4.4, 8.6, 14.6, 22.8, 36.1, 56.5, 92.3, 173.2]
    snap_v = [0.2, 0.7, 2.5, 6.9, 15.9, 35.6, 71.2, 137.0, 277.3]
    log_d = log.(snap_d)
    log_v = log.(snap_v)
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

function generate_bimodal_bins(d_fine::Float64, sg_fine::Float64,
                               d_coarse::Float64, sg_coarse::Float64;
                               n_bins::Int=15)
    log_d_min = min(log(d_fine) - 3*log(sg_fine), log(d_coarse) - 3*log(sg_coarse))
    log_d_max = max(log(d_fine) + 3*log(sg_fine), log(d_coarse) + 3*log(sg_coarse))
    log_d_min = max(log_d_min, log(1.0))
    log_d_max = min(log_d_max, log(500.0))
    log_d_centres = range(log_d_min, log_d_max, length=n_bins)
    d_centres = exp.(log_d_centres)
    [(d=d, v=snap_settling_velocity(d)) for d in d_centres]
end

function compute_bimodal_weights(d_fine::Float64, sg_fine::Float64,
                                  d_coarse::Float64, sg_coarse::Float64,
                                  frac_fine::Float64, bins)
    log_d_fine = log(d_fine)
    log_sg_fine = log(sg_fine)
    log_d_coarse = log(d_coarse)
    log_sg_coarse = log(sg_coarse)
    weights = Float64[]
    for bin in bins
        ld = log(bin.d)
        w_fine = exp(-0.5 * ((ld - log_d_fine) / log_sg_fine)^2) / log_sg_fine
        w_coarse = exp(-0.5 * ((ld - log_d_coarse) / log_sg_coarse)^2) / log_sg_coarse
        push!(weights, frac_fine * w_fine + (1.0 - frac_fine) * w_coarse)
    end
    weights ./= sum(weights)
    weights
end

# ============================================================================
# PRE-LOAD ALL DATA
# ============================================================================

println("\n1. Loading ERA5 met data...")
const ERA5_DIR = TEST_CONFIG.era5_dir
isdir(ERA5_DIR) || error("ERA5 directory not found: $ERA5_DIR — run download + merge first.")
const ERA5_FILES = sort(filter(f -> endswith(f, "_snap.nc"),
                               [joinpath(ERA5_DIR, f) for f in readdir(ERA5_DIR)]))
isempty(ERA5_FILES) && error("No _snap.nc files in $ERA5_DIR — run merge step first.")
println("   Using $(length(ERA5_FILES)) snap files from $ERA5_DIR")

const MET_FORMAT = Transport.detect_met_format(ERA5_FILES[1])
const NX, NY, NK = NCDataset(ERA5_FILES[1]) do ds
    Transport.get_met_dimensions(MET_FORMAT, ds)
end

const MET_CACHE = Dict{Tuple{Int,Int}, Transport.MeteoFields}()
# Doppler uses a trimmed download (7 windows covering 12 UTC Aug 23 → 09 UTC
# Aug 24), so file indices start at 1 instead of 5.
# SmallBoy detonates at 18:30 UTC, so it starts from the 18-20 UTC ERA5 block
# (file 7) — anchoring the cache at noon (file 5) introduced a 6.5 h wind-field
# offset. All other US tests sit within an hour of noon so file 5 still applies.
const CACHE_START_FILE     = TEST_NAME == "doppler"  ? 1 :
                             TEST_NAME == "smallboy" ? 7 : 5
const CACHE_START_TIME_IDX = 1   # all tests: first hour in the start file
const CACHE_END_FILE       = TEST_NAME == "doppler"  ? 7 :
                             TEST_NAME == "smallboy" ? 13 : 11
const SIM_HOURS            = TEST_NAME == "smallboy" ? 18 : 12   # SmallBoy needs H+15 TOA coverage
println("   Pre-loading met data (files $(CACHE_START_FILE)-$(CACHE_END_FILE))...")
for file_idx in CACHE_START_FILE:CACHE_END_FILE
    NCDataset(ERA5_FILES[file_idx]) do ds
        times = Transport.get_time_variable(MET_FORMAT, ds)
        for t_idx in 1:length(times)
            mf = Transport.MeteoFields(NX, NY, NK, T=Float32)
            if t_idx < length(times)
                Transport.read_initial_met_fields!(MET_FORMAT, mf, ds, t_idx, t_idx + 1)
            else
                Transport.read_initial_met_fields!(MET_FORMAT, mf, ds, t_idx, t_idx)
            end
            MET_CACHE[(file_idx, t_idx)] = mf
        end
    end
end
println("   Loaded $(length(MET_CACHE)) timesteps")

println("\n2. Setting up domain...")
const LAT_RANGE, LON_RANGE = NCDataset(ERA5_FILES[1]) do ds
    Float64.(ds["latitude"][:]), Float64.(ds["longitude"][:])
end

const START_DT = TEST_CONFIG.start_dt
const DOMAIN = Transport.SimulationDomain(
    lon_min = minimum(LON_RANGE), lon_max = maximum(LON_RANGE),
    lat_min = minimum(LAT_RANGE), lat_max = maximum(LAT_RANGE),
    z_min = 0.0, z_max = 35000.0, nx = NX, ny = NY, nz = NK,
    start_time = START_DT, end_time = START_DT + Dates.Hour(SIM_HOURS)
)

const RELEASE_X, RELEASE_Y = Transport.latlon_to_grid(DOMAIN, TEST_CONFIG.source_lat, TEST_CONFIG.source_lon)
println("   Release: grid ($(round(RELEASE_X, digits=1)), $(round(RELEASE_Y, digits=1)))")

println("\n3. Loading $(TEST_CONFIG.label) observations...")
const OBS = TEST_CONFIG.load_obs()
let _lat0, _lon0
    _lat0, _lon0 = Transport.suggest_grid(OBS; resolution_km=2.0, buffer_fraction=0.5)
    # Pad the grid out to a 1.3:1 panel aspect so model contours don't clip
    # at the obs-buffer edge when the panel is rendered as ~square.
    target_aspect = 1.3
    gz_lat = TEST_CONFIG.source_lat
    cur_aspect = (last(_lon0) - first(_lon0)) /
                 ((last(_lat0) - first(_lat0)) * cosd(gz_lat))
    if cur_aspect > target_aspect
        # extend latitude range
        needed = (last(_lon0) - first(_lon0)) / (target_aspect * cosd(gz_lat))
        extra = (needed - (last(_lat0) - first(_lat0))) / 2
        step = _lat0[2] - _lat0[1]
        n_extra = max(round(Int, extra / step), 0)
        lat_pre  = first(_lat0) .- step .* (n_extra:-1:1)
        lat_post = last(_lat0)  .+ step .* (1:n_extra)
        global const LAT_GRID = Float64.(vcat(lat_pre, _lat0, lat_post))
        global const LON_GRID = Float64.(_lon0)
    elseif cur_aspect < target_aspect
        needed = (last(_lat0) - first(_lat0)) * cosd(gz_lat) * target_aspect
        extra = (needed - (last(_lon0) - first(_lon0))) / 2
        step = _lon0[2] - _lon0[1]
        n_extra = max(round(Int, extra / step), 0)
        lon_pre  = first(_lon0) .- step .* (n_extra:-1:1)
        lon_post = last(_lon0)  .+ step .* (1:n_extra)
        global const LAT_GRID = Float64.(_lat0)
        global const LON_GRID = Float64.(vcat(lon_pre, _lon0, lon_post))
    else
        global const LAT_GRID = Float64.(_lat0)
        global const LON_GRID = Float64.(_lon0)
    end
end
const OBS_MASKS = Transport.rasterise_all_contours(OBS.dose_rate_contours, LAT_GRID, LON_GRID)
println("   Fine obs grid: $(length(LAT_GRID))x$(length(LON_GRID)) (2km, padded to 1.3:1 panel aspect)")

# Dose rate conversion: Bq/m² → mR/h at H+12
const CELL_AREA_M2 = let
    dlat = length(LAT_GRID) > 1 ? abs(LAT_GRID[2] - LAT_GRID[1]) : 0.018
    dlon = length(LON_GRID) > 1 ? abs(LON_GRID[2] - LON_GRID[1]) : 0.023
    ref_lat = 0.5 * (first(LAT_GRID) + last(LAT_GRID))
    (dlat * 111_000.0) * (dlon * 111_000.0 * cosd(ref_lat))
end
const DOSE_FACTOR = 1.9e-6 * 12.0^(-1.2) * 100.0 / CELL_AREA_M2  # K_DOSE * decay_12h * mSv→mR / area

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

const SOURCE_LAT = TEST_CONFIG.source_lat
const SOURCE_LON = TEST_CONFIG.source_lon
const OBS_MAX_DIST_KM = let
    max_dist = 0.0
    for (_, obs_mask) in OBS_MASKS
        for i in 1:length(LON_GRID)
            for j in 1:length(LAT_GRID)
                if obs_mask[i, j]
                    dlat = LAT_GRID[j] - SOURCE_LAT
                    dlon = (LON_GRID[i] - SOURCE_LON) * cosd(SOURCE_LAT)
                    dist_km = sqrt(dlat^2 + dlon^2) * 111.0
                    max_dist = max(max_dist, dist_km)
                end
            end
        end
    end
    max_dist
end
println("   Observed max plume extent: $(round(OBS_MAX_DIST_KM, digits=0)) km")

# Pre-compute observed inertia ellipse properties per contour level.
# Used for shape scoring: penalises models that produce blobs instead of
# the elongated plume shape seen in the Smoky observations.

"""
    inertia_ellipse(mask, lat_grid, lon_grid; min_cells=10)

Compute the inertia ellipse (aspect ratio + orientation) of a binary mask.
Coordinates converted to approximate km to avoid lat/lon distortion.
Returns (ar=aspect_ratio, angle=angle_rad) or nothing if too few cells.
"""
function inertia_ellipse(mask::AbstractMatrix, lat_grid, lon_grid; min_cells::Int=10)
    ref_lat = 0.5 * (first(lat_grid) + last(lat_grid))
    km_per_lon = cosd(ref_lat) * 111.0
    km_per_lat = 111.0

    xs = Float64[]
    ys = Float64[]
    for i in eachindex(lon_grid)
        for j in eachindex(lat_grid)
            if mask isa AbstractMatrix{Bool} ? mask[i, j] : mask[i, j] > 0
                push!(xs, (lon_grid[i] - lon_grid[1]) * km_per_lon)
                push!(ys, (lat_grid[j] - lat_grid[1]) * km_per_lat)
            end
        end
    end

    n = length(xs)
    if n < min_cells
        return nothing
    end

    cx = mean(xs)
    cy = mean(ys)

    # 2x2 inertia tensor
    Ixx = mean((xs .- cx).^2)
    Iyy = mean((ys .- cy).^2)
    Ixy = mean((xs .- cx) .* (ys .- cy))

    # Eigenvalues of [[Ixx, Ixy], [Ixy, Iyy]]
    T = Ixx + Iyy
    D = Ixx * Iyy - Ixy^2
    disc = max(T^2 / 4 - D, 0.0)
    lambda1 = T / 2 + sqrt(disc)
    lambda2 = T / 2 - sqrt(disc)
    lambda1 = max(lambda1, 1e-10)
    lambda2 = max(lambda2, 1e-10)

    ar = sqrt(lambda1 / lambda2)
    angle = 0.5 * atan(2 * Ixy, Ixx - Iyy)

    return (ar=ar, angle=angle)
end

# Pre-compute observed shapes (constant for the entire run)
const OBS_SHAPES = let
    shapes = Dict{Float64, @NamedTuple{ar::Float64, angle::Float64}}()
    for (dose_rate, obs_mask) in OBS_MASKS
        props = inertia_ellipse(obs_mask, LAT_GRID, LON_GRID)
        if !isnothing(props)
            shapes[dose_rate] = props
            println("   Contour $(dose_rate) mR/h: AR=$(round(props.ar, digits=1)), angle=$(round(rad2deg(props.angle), digits=0))°")
        end
    end
    shapes
end
println("   Computed observed shapes for $(length(OBS_SHAPES)) contour levels")

# Pre-compute observed centroid bearings per contour level.
# Used for bearing scoring: penalises models whose plume centroid points
# in the wrong direction (e.g. ENE instead of ESE).

"""
    centroid_bearing(mask, lat_grid, lon_grid, source_lat, source_lon; min_cells=10)

Compute the bearing (degrees, 0=N, 90=E) from `source` to the centroid of
binary `mask`. Returns nothing if too few cells.
"""
function centroid_bearing(mask::AbstractMatrix, lat_grid, lon_grid,
                          source_lat::Float64, source_lon::Float64;
                          min_cells::Int=10)
    ref_lat = 0.5 * (first(lat_grid) + last(lat_grid))
    sum_x = 0.0
    sum_y = 0.0
    n = 0
    for i in eachindex(lon_grid)
        for j in eachindex(lat_grid)
            if mask isa AbstractMatrix{Bool} ? mask[i, j] : mask[i, j] > 0
                sum_x += (lon_grid[i] - source_lon) * cosd(ref_lat)
                sum_y += lat_grid[j] - source_lat
                n += 1
            end
        end
    end
    n < min_cells && return nothing
    cx = sum_x / n
    cy = sum_y / n
    bearing = atand(cx, cy)  # atan2(east, north) → degrees from north
    bearing < 0 && (bearing += 360.0)
    return bearing
end

const OBS_BEARINGS = let
    bearings = Dict{Float64, Float64}()
    for (dose_rate, obs_mask) in OBS_MASKS
        b = centroid_bearing(obs_mask, LAT_GRID, LON_GRID, SOURCE_LAT, SOURCE_LON)
        if !isnothing(b)
            bearings[dose_rate] = b
            println("   Contour $(dose_rate) mR/h: bearing=$(round(b, digits=1))°")
        end
    end
    bearings
end
println("   Computed observed bearings for $(length(OBS_BEARINGS)) contour levels")

# Layer geometry now tuneable via CMA-ES parameters (stem_top_m, cap_mid_m, cloud_top_m)
# Warm start from DASA-1251 (AGL): stem top 1822 m, cap mid 5541 m, cloud top 9259 m
println("\n4. Layer geometry: tuneable (warm start from DASA-1251 Smoky cloud obs, AGL)")

println("\n5. All data pre-loaded")

# ============================================================================
# DISTANCE FUNCTION — with per-generation CRN
# ============================================================================

"""
    rho_core(params, turb_scheme, gen_seed)

Run Transport simulation and return loss = 1 - combined_score.
`gen_seed` provides CRN: all candidates in a generation share the same
random particle-to-bin assignments, reducing ranking noise for CMA-ES.
Fresh seeds across generations prevent convergence to noise artefacts.
"""
const LAST_DOSE_SMOOTH    = Ref{Union{Nothing, Matrix{Float64}}}(nothing)
const LAST_DOSE_RAW       = Ref{Union{Nothing, Matrix{Float64}}}(nothing)
const LAST_MODEL_SNAPSHOTS = Ref{Union{Nothing, Vector{Matrix{Float64}}}}(nothing)
const LAST_SNAPSHOT_HOURS  = Ref{Union{Nothing, Vector{Float64}}}(nothing)

function rho_core(params::Vector{Float64}, turb_scheme::Symbol, gen_seed::UInt64)
    # Unpack 23 parameters (v6: added stem_top_m, cap_mid_m, cloud_top_m)
    d_median_fine     = params[1]
    sigma_g_fine      = params[2]
    d_median_coarse   = params[3]
    sigma_g_coarse    = params[4]
    frac_fine         = params[5]
    frac_lower_raw    = params[6]
    frac_middle_raw   = params[7]
    sigma_w_scale     = params[8]
    sigma_h_scale     = params[9]
    h_diff_scale      = params[10]
    tl_scale          = params[11]
    vd_scale          = params[12]
    vgrav_scale       = params[13]
    omega_scale       = params[14]
    mixing_height_scale = params[15]
    tmix_scale        = params[16]
    surface_height_scale = params[17]
    roughness_scale   = params[18]
    activity_scale    = params[19]
    smooth_sigma      = params[20]
    stem_top_m        = params[21]
    cap_mid_m         = params[22]
    cloud_top_m       = params[23]

    # Normalise layer fractions so they always sum to 1
    frac_upper_raw = max(1.0 - frac_lower_raw - frac_middle_raw, 0.05)
    frac_total = frac_lower_raw + frac_middle_raw + frac_upper_raw
    frac_lower  = frac_lower_raw / frac_total
    frac_middle = frac_middle_raw / frac_total
    frac_upper  = frac_upper_raw / frac_total

    # Enforce layer height ordering: stem_top < cap_mid < cloud_top
    layer_heights = sort([stem_top_m, cap_mid_m, cloud_top_m])
    stem_top_m  = layer_heights[1]
    cap_mid_m   = layer_heights[2]
    cloud_top_m = layer_heights[3]

    # CRN: deterministic per generation, identical across candidates
    rng = Random.MersenneTwister(gen_seed)

    # Generate bimodal particle size distribution
    size_bins = generate_bimodal_bins(d_median_fine, sigma_g_fine,
                                      d_median_coarse, sigma_g_coarse; n_bins=15)
    bin_weights = compute_bimodal_weights(d_median_fine, sigma_g_fine,
                                          d_median_coarse, sigma_g_coarse,
                                          frac_fine, size_bins)

    # 3-layer release: particle counts proportional to mass fractions
    n_particles = parse(Int, get(ENV, "N_PARTICLES", "2500"))
    total_activity = activity_scale * 1.0e15
    n_lower  = max(round(Int, n_particles * frac_lower), 1)
    n_middle = max(round(Int, n_particles * frac_middle), 1)
    n_upper  = max(n_particles - n_lower - n_middle, 1)

    # Build layer geometry from tuneable heights (radii scale with height)
    layer_lower  = Transport.CylinderRelease(0.0, stem_top_m, 0.2 * stem_top_m)
    layer_middle = Transport.CylinderRelease(stem_top_m, cap_mid_m, 0.25 * (cap_mid_m - stem_top_m))
    layer_upper  = Transport.CylinderRelease(cap_mid_m, cloud_top_m, 0.25 * (cloud_top_m - cap_mid_m))

    sources = [
        Transport.ReleaseSource((RELEASE_X, RELEASE_Y), layer_lower,
                           Transport.BombRelease(0.0), [total_activity * frac_lower], n_lower),
        Transport.ReleaseSource((RELEASE_X, RELEASE_Y), layer_middle,
                           Transport.BombRelease(0.0), [total_activity * frac_middle], n_middle),
        Transport.ReleaseSource((RELEASE_X, RELEASE_Y), layer_upper,
                           Transport.BombRelease(0.0), [total_activity * frac_upper], n_upper),
    ]
    decay_params = [Transport.DecayParams(kdecay=Transport.NoDecay, halftime_hours=0.0)]

    state = Transport.initialize_simulation(DOMAIN, sources, ["MixedFP"], decay_params;
                                        log_depositions=true)

    init_met = MET_CACHE[(CACHE_START_FILE, CACHE_START_TIME_IDX)]

    # Generate particles from all 3 layers
    positions_m = Tuple{Float64,Float64,Float64}[]
    activities = Float64[]
    for src in sources
        pos_s, act_s, released_s = Transport.generate_release_particles(
            rng, src, 0, 1,
            ones(Float64, NX, NY), ones(Float64, NX, NY),
            DOMAIN.dx, DOMAIN.dy, DOMAIN.hlevel
        )
        if released_s && !isempty(pos_s)
            append!(positions_m, pos_s)
            append!(activities, act_s)
        end
    end

    if isempty(positions_m)
        GC.gc(false)
        return (loss = 1.0, fms = 0.0, shape = 0.0, bearing = 0.0, extent = 0.0, toa = 0.0, combined_old = 0.0)
    end

    n_part = length(positions_m)
    n_classes = length(size_bins)

    snap_size_bins = [Transport.ParticleProperties(diameter_μm=b.d, density_gcm3=2.5) for b in size_bins]
    particle_radii = Float64[]
    particle_densities = Float64[]
    particle_size_indices = Int[]
    fixed_gravity = [b.v * vgrav_scale for b in size_bins]

    cum_weights = cumsum(bin_weights)
    base_density = 2500.0

    # Assign size bins (CRN: rng consumption order deterministic)
    assigned_bins = Vector{Int}(undef, n_part)
    for i in 1:n_part
        r = rand(rng)
        idx = searchsortedfirst(cum_weights, r)
        assigned_bins[i] = clamp(idx, 1, n_classes)
    end

    # Add particles to ensemble with size properties (no vertical IC transforms — layers are fixed)
    for i in 1:n_part
        pos = positions_m[i]
        activity = activities[i]
        sigma_z = Transport.height_to_sigma_hybrid(RELEASE_X, RELEASE_Y, pos[3], init_met, 0.0)
        Transport.add_particle!(state.ensemble,
                           SVector{3,Float64}(pos[1], pos[2], sigma_z),
                           SVector{3,Float64}(0.0, 0.0, 0.0),
                           [activity], 0.0, icomp=1)

        idx = assigned_bins[i]
        push!(particle_radii, size_bins[idx].d * 0.5e-6)
        push!(particle_densities, base_density)
        push!(particle_size_indices, idx)
        state.ensemble.particles[i].grv = Float32(size_bins[idx].v * 0.01 * vgrav_scale)
    end

    particle_size_config = Transport.ParticleSizeConfig(
        size_bins=snap_size_bins, particle_radii=particle_radii,
        particle_densities=particle_densities,
        particle_size_indices=particle_size_indices,
        fixed_gravity_cm_s=fixed_gravity
    )

    hanna_config = Transport.HannaTurbulenceConfig{Float64}(
        sigma_scale = sigma_h_scale,
        sigma_scale_vertical = sigma_w_scale,
        tl_scale = tl_scale,
        use_cbl = true
    )

    deposition_config = Transport.DepositionConfig{Float64}(
        apply_dry_deposition = true,
        apply_wet_deposition = false,
        use_simple_deposition = true,
        simple_deposition_velocity = 0.002 * vd_scale,
        simple_surface_height = 30.0 * surface_height_scale,
        mixing_height = 1000.0 * mixing_height_scale,
        surface_roughness = 0.1 * roughness_scale
    )

    snapshot_times = [Float64(h) * 3600.0 for h in 1:SIM_HOURS]

    numerical_config = Transport.ERA5NumericalConfig{Float64}(
        interpolation_order = Transport.LinearInterp,
        ode_solver_type = :Euler,
        fixed_dt = 300.0,
        turbulence = turb_scheme == :OU ? Transport.OrnsteinUhlenbeck : Transport.RandomWalk
    )

    sim_config = Transport.SimulationConfig{Float64}(
        saveat = snapshot_times,
        verbose = false,
        max_duration = Float64(SIM_HOURS) * 3600.0,
        save_snapshots = true,
        dt_particle = 300.0,
        use_reference_stepping = true,
        max_files = CACHE_END_FILE - CACHE_START_FILE + 1,
        omega_scale = omega_scale
    )

    Transport.run_simulation!(state, ERA5_FILES,
        particle_size_config=particle_size_config,
        deposition_config=deposition_config,
        hanna_config=hanna_config,
        decay_params=decay_params,
        config=sim_config,
        numerical_config=numerical_config,
        advection_enabled=true,
        settling_enabled=true,
        dry_deposition_enabled=true,
        wet_deposition_enabled=false,
        release_height_m=cloud_top_m,
        met_data_cache=MET_CACHE,
        met_format_override=MET_FORMAT,
        met_dimensions=(NX, NY, NK),
        cache_init_file_idx=CACHE_START_FILE,
        cache_init_time_idx=CACHE_START_TIME_IDX,
        sigma_already_initialized=true
    )

    # Build hourly fine-grid deposition for TOA + final FMS
    nx_obs, ny_obs = length(LON_GRID), length(LAT_GRID)
    sorted_events = sort(state.deposition_log, by=e->e.time)

    model_snapshots = Vector{Matrix{Float64}}()
    snapshot_hours = Float64[]

    for hour in 1:SIM_HOURS
        hour_end_time = Float64(hour) * 3600.0
        hourly_dep = zeros(nx_obs, ny_obs)
        for evt in sorted_events
            if evt.time <= hour_end_time
                lat, lon = Transport.grid_to_latlon(DOMAIN, evt.x, evt.y)
                if lon > 180.0
                    lon = lon - 360.0
                end
                i_fine = searchsortedlast(LON_GRID, lon)
                j_fine = searchsortedlast(LAT_GRID, lat)
                if 1 <= i_fine <= nx_obs && 1 <= j_fine <= ny_obs
                    hourly_dep[i_fine, j_fine] += evt.mass
                end
            end
        end
        push!(model_snapshots, hourly_dep)
        push!(snapshot_hours, Float64(hour))
    end

    final_dose = model_snapshots[end]
    total = sum(final_dose)
    if total <= 0
        GC.gc(false)
        return (loss = 1.0, fms = 0.0, shape = 0.0, bearing = 0.0, extent = 0.0, toa = 0.0, combined_old = 0.0)
    end

    # Convert to dose rate (mR/h at H+12) and smooth for FMS
    final_dose_mRh = final_dose .* DOSE_FACTOR
    dose_smooth = gaussian_smooth(final_dose_mRh, smooth_sigma)

    # Stash latest dose fields for VIZ_ONLY mode to read after a single call.
    LAST_DOSE_SMOOTH[] = copy(dose_smooth)
    LAST_DOSE_RAW[]    = copy(final_dose_mRh)

    # FMS scoring using ACTUAL dose rate thresholds + shape scoring
    fms_scores = Float64[]
    shape_scores = Float64[]
    for (dose_rate, obs_mask) in OBS_MASKS
        obs_area = sum(obs_mask)
        if obs_area == 0
            push!(fms_scores, 0.0)
            push!(shape_scores, 0.0)
            continue
        end
        model_mask = dose_smooth .>= dose_rate
        inter = Float64(sum(model_mask .& obs_mask))
        uni = Float64(sum(model_mask .| obs_mask))
        fms = uni > 0 ? inter / uni : 0.0
        push!(fms_scores, fms)

        # Shape score: compare inertia ellipse of model vs observed mask
        obs_shape = get(OBS_SHAPES, dose_rate, nothing)
        model_shape = if sum(model_mask) > 0
            inertia_ellipse(model_mask, LAT_GRID, LON_GRID)
        else
            nothing
        end
        if !isnothing(obs_shape) && !isnothing(model_shape)
            ar_score = min(model_shape.ar, obs_shape.ar) / max(model_shape.ar, obs_shape.ar)
            angle_diff = model_shape.angle - obs_shape.angle
            orient_score = cos(angle_diff)^2
            push!(shape_scores, 0.7 * ar_score + 0.3 * orient_score)
        else
            push!(shape_scores, 0.0)
        end
    end

    geo_mean_fms = exp(mean(log(max(s, 0.005)) for s in fms_scores))
    geo_mean_shape = exp(mean(log(max(s, 0.005)) for s in shape_scores))

    # Bearing score: compare model centroid bearing to observed bearing per contour.
    # Dose-rate-weighted so close-in contours (50-1000 mR/h at ~100° ESE) dominate
    # over far-field (1 mR/h at ~80° ENE). Uses cos⁴(Δ) for sharp penalty:
    #   10° error → 0.94,  20° → 0.77,  30° → 0.56,  45° → 0.25
    bearing_sum = 0.0
    bearing_weight_sum = 0.0
    for (dose_rate, obs_mask) in OBS_MASKS
        obs_bearing = get(OBS_BEARINGS, dose_rate, nothing)
        isnothing(obs_bearing) && continue
        model_mask = dose_smooth .>= dose_rate
        model_bearing = if sum(model_mask) > 0
            centroid_bearing(model_mask, LAT_GRID, LON_GRID, SOURCE_LAT, SOURCE_LON)
        else
            nothing
        end
        w = dose_rate  # weight by dose rate: 1000 mR/h gets 1000× weight of 1 mR/h
        if !isnothing(model_bearing)
            diff_deg = abs(model_bearing - obs_bearing)
            diff_deg > 180.0 && (diff_deg = 360.0 - diff_deg)
            bearing_sum += w * cosd(diff_deg)^4
        else
            # No model contour at this level → zero score with full weight
        end
        bearing_weight_sum += w
    end
    bearing_score = bearing_weight_sum > 0 ? bearing_sum / bearing_weight_sum : 0.0

    # Plume extent
    model_max_dist_km = 0.0
    for i in 1:nx_obs
        for j in 1:ny_obs
            if final_dose[i, j] > 0
                dlat = LAT_GRID[j] - SOURCE_LAT
                dlon = (LON_GRID[i] - SOURCE_LON) * cosd(SOURCE_LAT)
                dist_km = sqrt(dlat^2 + dlon^2) * 111.0
                model_max_dist_km = max(model_max_dist_km, dist_km)
            end
        end
    end
    extent_score = clamp(model_max_dist_km / OBS_MAX_DIST_KM, 0.0, 1.0)

    # TOA scoring — smooth each snapshot so sparse particle deposits produce a
    # continuous field for cell-level arrival detection. Uses the optimised
    # smooth_sigma (same as FMS). TOA is currently non-discriminating (≈1.0 for
    # all reasonable parameter sets) but ensures correct plume direction is rewarded.
    model_snapshots_smooth = Vector{Matrix{Float64}}()
    for snap in model_snapshots
        snap_smooth = gaussian_smooth(snap, smooth_sigma)
        snap_total = sum(snap_smooth)
        push!(model_snapshots_smooth, snap_total > 0 ? snap_smooth ./ snap_total : snap_smooth)
    end

    LAST_MODEL_SNAPSHOTS[] = deepcopy(model_snapshots_smooth)
    LAST_SNAPSHOT_HOURS[]  = copy(snapshot_hours)

    toa_result = Transport.compute_toa_score(model_snapshots_smooth, snapshot_hours,
                                         OBS.toa_contours, LAT_GRID, LON_GRID;
                                         threshold_fraction=0.01)
    toa_score = if isnothing(toa_result) || isinf(toa_result.mean_arrival_error_hours)
        0.0
    else
        max(0.0, 1.0 - toa_result.mean_arrival_error_hours / 6.0)
    end

    # Combined (used for CMA-ES ranking): now includes a 20% bearing term so the
    # optimiser cannot trade plume direction for TOA/extent. See header for weights.
    # Doppler override: shape/bearing/extent metrics are degenerate at the
    # observation's very-low dose levels (any tiny model lobe scores ≈1.0),
    # so the optimiser games them.  Boost FMS, kill shape/extent.
    combined = if TEST_NAME == "doppler"
        0.45 * geo_mean_fms + 0.10 * geo_mean_shape + 0.20 * bearing_score +
            0.05 * extent_score + 0.20 * toa_score
    else
        0.25 * geo_mean_fms + 0.15 * geo_mean_shape + 0.20 * bearing_score +
            0.10 * extent_score + 0.30 * toa_score
    end
    # Old combined (for apples-to-apples comparison with APMC v7/v9): 50% FMS + 20% extent + 30% TOA
    combined_old = 0.50 * geo_mean_fms + 0.20 * extent_score + 0.30 * toa_score

    # Hard gate: reject geometric-cheating solutions whose plume points the wrong
    # way (bearing error ≳45°). loss=2.0 (not Inf) keeps a finite gradient toward
    # improving bearing so CMA-ES can climb back out.
    if bearing_score < 0.5
        return (loss = 2.0,
                fms = geo_mean_fms,
                shape = geo_mean_shape,
                bearing = bearing_score,
                extent = extent_score,
                toa = toa_score,
                combined_old = combined_old)
    end

    GC.gc(false)
    return (loss = 1.0 - combined,
            fms = geo_mean_fms,
            shape = geo_mean_shape,
            bearing = bearing_score,
            extent = extent_score,
            toa = toa_score,
            combined_old = combined_old)
end

# ============================================================================
# CMA-ES IMPLEMENTATION
# ============================================================================

mutable struct CMAES
    N::Int              # dimension
    lambda::Int         # population size
    mu::Int             # number of parents
    weights::Vector{Float64}
    mueff::Float64
    cc::Float64
    cs::Float64
    c1::Float64
    cmu::Float64
    damps::Float64
    chiN::Float64
    xmean::Vector{Float64}
    sigma::Float64
    pc::Vector{Float64}
    ps::Vector{Float64}
    C::Matrix{Float64}
    B::Matrix{Float64}
    D::Vector{Float64}
    eigeneval::Int
    lb::Vector{Float64}
    ub::Vector{Float64}
    counteval::Int
    generation::Int
    best_ever_val::Float64
    best_ever_x::Vector{Float64}
    stagnation_counter::Int
end

"""
    CMAES(xstart, lb, ub; popsize=0, sigma_frac=0.3)

Initialise CMA-ES with automatic sigma and covariance scaling from bounds.
`sigma_frac` controls what fraction of each dimension's range is covered
by ±1σ initially (0.3 = initial search covers ~60% of each dimension).
The covariance diagonal is set so that all dimensions are explored equally
despite wildly different scales (e.g. d_median 5-250 μm vs sigma_g 1.2-3.0).
"""
function CMAES(xstart::Vector{Float64};
               lb::Vector{Float64}, ub::Vector{Float64},
               popsize::Int=0, sigma_frac::Float64=0.3)
    N = length(xstart)
    lambda = popsize > 0 ? popsize : 4 + floor(Int, 3 * log(N))
    mu = lambda ÷ 2
    raw_weights = [log(lambda/2 + 0.5) - log(i) for i in 1:mu]
    weights = raw_weights ./ sum(raw_weights)
    mueff = sum(weights)^2 / sum(weights.^2)

    cc = (4 + mueff/N) / (N + 4 + 2*mueff/N)
    cs = (mueff + 2) / (N + mueff + 5)
    c1 = 2 / ((N + 1.3)^2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((N + 2)^2 + mueff))
    damps = 2 * mueff / lambda + 0.3 + cs
    chiN = sqrt(N) * (1 - 1/(4*N) + 1/(21*N^2))

    xmean = copy(xstart)
    pc = zeros(N)
    ps = zeros(N)

    # Auto-scale: set sigma=1 and encode per-dimension scales in C.
    # C_ii = (sigma_frac * range_i)^2 so that sigma * sqrt(C_ii) = sigma_frac * range_i.
    # This means ±3σ covers ±3*sigma_frac ≈ 90% of each dimension from the start.
    ranges = ub .- lb
    initial_stds = sigma_frac .* ranges
    C = diagm(initial_stds.^2)
    B = Matrix{Float64}(I, N, N)
    D_vec = copy(initial_stds)
    sigma = 1.0  # all scaling is in C

    CMAES(N, lambda, mu, weights, mueff, cc, cs, c1, cmu, damps, chiN,
          xmean, sigma, pc, ps, C, B, D_vec, 0, lb, ub, 0, 0,
          Inf, copy(xstart), 0)
end

function update_eigensystem!(es::CMAES)
    es.C .= (es.C .+ es.C') ./ 2
    F = eigen(Symmetric(es.C))
    es.D .= sqrt.(max.(F.values, 1e-20))
    es.B .= F.vectors
    es.eigeneval = es.counteval
end

function ask(es::CMAES)::Vector{Vector{Float64}}
    if es.counteval - es.eigeneval > es.lambda / (es.c1 + es.cmu) / es.N / 10
        update_eigensystem!(es)
    end
    candidates = Vector{Vector{Float64}}(undef, es.lambda)
    for k in 1:es.lambda
        z = randn(es.N)
        y = es.B * (es.D .* z)
        x = es.xmean .+ es.sigma .* y
        x .= clamp.(x, es.lb, es.ub)
        candidates[k] = x
    end
    return candidates
end

function tell!(es::CMAES, candidates::Vector{Vector{Float64}}, fitvals::Vector{Float64})
    es.counteval += es.lambda
    es.generation += 1

    idx = sortperm(fitvals)
    xold = copy(es.xmean)

    # Update mean
    es.xmean .= sum(es.weights[i] .* candidates[idx[i]] for i in 1:es.mu)

    # Update evolution paths
    y = (es.xmean .- xold) ./ es.sigma
    z = es.B' * y
    z ./= (es.D .+ 1e-20)
    Cinvsqrt_y = es.B * z

    csn = sqrt(es.cs * (2 - es.cs) * es.mueff)
    es.ps .= (1 - es.cs) .* es.ps .+ csn .* Cinvsqrt_y
    pslen = norm(es.ps)

    threshold = (1.4 + 2/(es.N + 1)) * es.chiN * sqrt(1 - (1 - es.cs)^(2 * es.counteval / es.lambda))
    hsig = pslen < threshold ? 1.0 : 0.0

    ccn = sqrt(es.cc * (2 - es.cc) * es.mueff)
    es.pc .= (1 - es.cc) .* es.pc .+ hsig * ccn .* y

    # Update covariance matrix
    c1a = es.c1 * (1 - (1 - hsig^2) * es.cc * (2 - es.cc))
    rank_mu = zeros(es.N, es.N)
    for i in 1:es.mu
        yi = (candidates[idx[i]] .- xold) ./ es.sigma
        rank_mu .+= es.weights[i] .* (yi * yi')
    end
    es.C .= (1 - c1a - es.cmu * sum(es.weights)) .* es.C
    es.C .+= es.c1 .* (es.pc * es.pc')
    es.C .+= es.cmu .* rank_mu

    # Update step size
    es.sigma *= exp(min(1.0, (es.cs / es.damps) * (pslen / es.chiN - 1) / 2))

    # Track best-ever
    if fitvals[idx[1]] < es.best_ever_val
        es.best_ever_val = fitvals[idx[1]]
        es.best_ever_x = copy(candidates[idx[1]])
        es.stagnation_counter = 0
    else
        es.stagnation_counter += 1
    end

    return fitvals[idx[1]], candidates[idx[1]]
end

function should_restart(es::CMAES)::Tuple{Bool, String}
    if any(isnan.(es.C)) || any(isinf.(es.C))
        return true, "numerical error in C"
    end
    if isnan(es.sigma) || isinf(es.sigma)
        return true, "numerical error in sigma"
    end

    # Step size too small
    if es.sigma * maximum(es.D) < 1e-12
        return true, "sigma * max(D) < 1e-12"
    end

    # Condition number too large
    cond_C = maximum(es.D)^2 / (minimum(es.D)^2 + 1e-20)
    if cond_C > 1e14
        return true, @sprintf("condition number %.1e > 1e14", cond_C)
    end

    # Stagnation: no improvement for 100 + 30*N/lambda generations
    stag_limit = 100 + ceil(Int, 30 * es.N / es.lambda)
    if es.stagnation_counter > stag_limit
        return true, @sprintf("no improvement for %d generations", es.stagnation_counter)
    end

    # All principal axes too small relative to domain
    max_std = es.sigma * maximum(es.D)
    min_domain = minimum(es.ub .- es.lb)
    if max_std < 1e-8 * min_domain
        return true, "search radius negligible vs domain"
    end

    return false, ""
end

# ============================================================================
# PARALLEL EVALUATION WITH PER-GENERATION CRN
# ============================================================================

struct EvalResult
    loss::Float64
    fms::Float64
    shape::Float64
    bearing::Float64
    extent::Float64
    toa::Float64
    combined_old::Float64
end

const FAILED_EVAL = EvalResult(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

function evaluate_generation(candidates::Vector{Vector{Float64}},
                              turb_scheme::Symbol, gen_seed::UInt64)
    n = length(candidates)
    results = Vector{EvalResult}(undef, n)
    Threads.@threads for i in 1:n
        try
            # candidates are in ENCODED (log) search space; decode to physical
            # units before the forward model. rho_core never sees log space.
            r = rho_core(decode_params(candidates[i]), turb_scheme, gen_seed)
            results[i] = EvalResult(r.loss, r.fms, r.shape, r.bearing, r.extent, r.toa, r.combined_old)
        catch e
            @warn "Evaluation failed for candidate $i" exception=e
            results[i] = FAILED_EVAL
        end
    end
    return results
end

# ============================================================================
# BIPOP-CMA-ES MAIN LOOP
# ============================================================================

const MAX_EVALS = parse(Int, get(ENV, "MAX_EVALS", "6000"))
const USE_WARM_START = parse(Int, get(ENV, "WARM_START", "1")) == 1

const DEFAULT_LAMBDA = 4 + floor(Int, 3 * log(N_DIM))  # 12 for d=18
const SIGMA_FRAC = 0.3  # ±1σ covers 30% of each dimension's range

# Starting point: prefer per-test checkpoint, fall back to yield-scaled Smoky warm start
function load_checkpoint_params(turb_suffix::String)
    ckpt = joinpath(@__DIR__, "$(TEST_NAME)_cmaes_$(turb_suffix)_best.txt")
    isfile(ckpt) || return nothing
    params = Dict{String, Float64}()
    for line in eachline(ckpt)
        startswith(line, "#") && continue
        parts = split(line, "\t", limit=2)
        length(parts) == 2 || continue
        params[strip(parts[1])] = parse(Float64, strip(parts[2]))
    end
    x = Float64[]
    for pname in PARAM_NAMES
        haskey(params, pname) || return nothing
        push!(x, params[pname])
    end
    return clamp.(x, LB, UB)
end

x0 = if USE_WARM_START
    turb_suffix = lowercase(string(TURB_SCHEME))
    ckpt_params = load_checkpoint_params(turb_suffix)
    if ckpt_params !== nothing
        println("\n   Warm start from checkpoint: $(TEST_NAME)_cmaes_$(turb_suffix)_best.txt")
        ckpt_params
    else
        println("\n   Warm start from Smoky-best vector (cloud heights × yield^0.215 = $(round(YIELD_SCALE, digits=3)))")
        copy(WARM_START_PARAMS)
    end
else
    println("\n   Starting from centre of domain")
    (LB .+ UB) ./ 2.0
end

if TEST_NAME == "doppler"
    # NOTE: Earlier code pushed warm-start heights into the 9-14 km jet
    # band on the (now disproven) hypothesis that only that band could
    # carry mass to the obs core. The May 2026 trajectory diagnostic
    # showed 61/200 particles from the Gen-56 vector (cloud at 8-10 km)
    # reach the obs NE-peak bbox at median altitude 6.3 km. Bumper
    # removed; warm start now honoured as-saved in best.txt.
    x0 .= clamp.(x0, LB, UB)
    println("   Doppler: warm start honoured " *
            "(stem=$(round(x0[21])), cap_mid=$(round(x0[22])), cloud_top=$(round(x0[23]))) m")
end

println("\n" * "="^70)
println("BIPOP-CMA-ES Configuration:")
println("  Turbulence: $(TURB_NAME)")
println("  Dimension: $(N_DIM)")
println("  Default lambda: $(DEFAULT_LAMBDA)")
println("  Auto-sigma: C diagonal scaled from bounds (sigma_frac=$(SIGMA_FRAC))")
println("  Budget: $(MAX_EVALS) evaluations")
println("  Threads: $(nthreads())")
println("  Noise handling: per-generation CRN")
println("  Warm start: $(USE_WARM_START)")
println("="^70)

function _viz_sweep(x0_local, turb, n_seeds)
    best_seed = UInt64(0)
    best_diag = nothing
    best_loss = Inf
    sweep_rng = Random.MersenneTwister(0xC0DE)
    for k in 1:n_seeds
        seed = rand(sweep_rng, UInt64)
        diag = rho_core(x0_local, turb, seed)
        if diag.loss < best_loss
            best_loss = diag.loss
            best_diag = diag
            best_seed = seed
        end
        println("  seed $(lpad(k, 3)): loss=$(round(diag.loss, digits=4))  fms=$(round(diag.fms, digits=4))  " *
                "shape=$(round(diag.shape, digits=4))  ext=$(round(diag.extent, digits=4))")
    end
    return best_seed, best_diag
end

if get(ENV, "VIZ_ONLY", "0") == "1"
    n_seeds = parse(Int, get(ENV, "VIZ_SEEDS", "20"))
    println("\nVIZ_ONLY: sweeping $n_seeds seeds at x0 to find the best fit.")
    best_seed, best_diag = _viz_sweep(x0, TURB_SCHEME, n_seeds)
    # Rerun the best seed so LAST_DOSE_SMOOTH holds its dose field.
    println("\nBest seed: $(best_seed)")
    rho_core(x0, TURB_SCHEME, best_seed)
    println("  loss=$(round(best_diag.loss, digits=4))  fms=$(round(best_diag.fms, digits=4))  " *
            "shape=$(round(best_diag.shape, digits=4))  bearing=$(round(best_diag.bearing, digits=4))  " *
            "extent=$(round(best_diag.extent, digits=4))  toa=$(round(best_diag.toa, digits=4))")
    sweep_csv = get(ENV, "SMOOTH_SWEEP", "")
    if !isempty(sweep_csv)
        sweep_vals = [parse(Float64, strip(s)) for s in split(sweep_csv, ",")]
        out_base = joinpath(@__DIR__, "$(TEST_NAME)_cmaes_$(lowercase(string(TURB_SCHEME)))_fit.png")
        println("\nSMOOTH_SWEEP: rendering for sigma values $(sweep_vals)")
        for sv in sweep_vals
            x_sv = copy(x0); x_sv[20] = sv
            diag = rho_core(x_sv, TURB_SCHEME, best_seed)
            println("  σ=$(round(sv, digits=2)) cells  loss=$(round(diag.loss, digits=4))  " *
                    "fms=$(round(diag.fms, digits=4))  shape=$(round(diag.shape, digits=4))  " *
                    "brg=$(round(diag.bearing, digits=4))  ext=$(round(diag.extent, digits=4))  " *
                    "toa=$(round(diag.toa, digits=4))")
            include(joinpath(@__DIR__, "viz_plot_fit.jl"))
            tag = replace(string(round(sv, digits=2)), "." => "p")
            renamed = replace(out_base, "_fit.png" => "_fit_smooth$(tag).png")
            mv(out_base, renamed; force=true)
            println("    saved: $(basename(renamed))")
        end
        exit()
    end
    include(joinpath(@__DIR__, "viz_plot_fit.jl"))
    exit()
end

# ============================================================================
# LHS_SWEEP — coarse Latin Hypercube scout under the gated loss, run BEFORE
# BIPOP-CMA-ES to map the feasible (correct-bearing) basin globally and pick a
# good warm start. Mirrors the threaded generation loop: rho_core is serial, so
# we parallelise across samples. Shared CRN seed gives paired (low-noise)
# ranking. Writes a sorted table + the best vector as $(TEST)_lhs_best.txt.
# ============================================================================
if get(ENV, "LHS_SWEEP", "0") == "1"
    lhs_n = parse(Int, get(ENV, "LHS_N", "256"))
    crn   = UInt64(parse(Int, get(ENV, "LHS_CRN", "371629")))
    rng_lhs = Random.MersenneTwister(0xC0FFEE)

    # Latin hypercube design in [LB, UB] (raw; rho_core sorts heights and
    # normalises layer fractions internally, so unconstrained samples are fine).
    strata = [Random.randperm(rng_lhs, lhs_n) for _ in 1:N_DIM]
    samples = Vector{Vector{Float64}}()
    labels  = String[]
    for i in 1:lhs_n
        x = Vector{Float64}(undef, N_DIM)
        for j in 1:N_DIM
            u = (strata[j][i] - 1 + rand(rng_lhs)) / lhs_n
            x[j] = LB[j] + u * (UB[j] - LB[j])
        end
        push!(samples, x); push!(labels, "lhs$(lpad(i, 4, '0'))")
    end

    # Append physics-prior candidates ($(TEST)_warmstart_cand*.txt) and the
    # current warm-start x0 as guaranteed-evaluated points — LHS is sparse in 23D.
    function _read_param_file(path)
        isfile(path) || return nothing
        d = Dict{String,Float64}()
        for line in eachline(path)
            startswith(line, "#") && continue
            p = split(line, "\t", limit=2)
            length(p) == 2 || continue
            d[strip(p[1])] = parse(Float64, strip(p[2]))
        end
        v = Float64[]
        for pn in PARAM_NAMES
            haskey(d, pn) || return nothing
            push!(v, d[pn])
        end
        return clamp.(v, LB, UB)
    end
    for f in sort(filter(p -> occursin("warmstart_cand", p), readdir(@__DIR__)))
        v = _read_param_file(joinpath(@__DIR__, f))
        v === nothing && continue
        push!(samples, v); push!(labels, f)
    end
    push!(samples, clamp.(copy(x0), LB, UB)); push!(labels, "warm_start_x0")

    n_total = length(samples)
    println("\nLHS_SWEEP: evaluating $(n_total) points " *
            "($(lhs_n) LHS + $(n_total - lhs_n) priors), " *
            "N_PARTICLES=$(get(ENV, "N_PARTICLES", "2500")), CRN=$(crn), threads=$(nthreads())")

    losses   = Vector{Float64}(undef, n_total)
    fmss     = Vector{Float64}(undef, n_total)
    shapes   = Vector{Float64}(undef, n_total)
    bearings = Vector{Float64}(undef, n_total)
    extents  = Vector{Float64}(undef, n_total)
    toas     = Vector{Float64}(undef, n_total)
    done = Threads.Atomic{Int}(0)
    Threads.@threads for i in 1:n_total
        dret = rho_core(samples[i], TURB_SCHEME, crn)
        losses[i]   = dret.loss;   fmss[i]    = dret.fms;    shapes[i] = dret.shape
        bearings[i] = dret.bearing; extents[i] = dret.extent; toas[i]  = dret.toa
        c = Threads.atomic_add!(done, 1) + 1
        c % 16 == 0 && println("  evaluated $(c)/$(n_total)")
    end

    order = sortperm(losses)
    out = joinpath(@__DIR__, "$(TEST_NAME)_lhs_results.txt")
    open(out, "w") do io
        println(io, "# LHS sweep sorted by loss (lower better). gate: bearing<0.5 -> loss=2.0")
        println(io, "# rank\tlabel\tloss\tfms\tshape\tbearing\textent\ttoa")
        for (rank, idx) in enumerate(order)
            @printf(io, "%d\t%s\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n",
                    rank, labels[idx], losses[idx], fmss[idx], shapes[idx],
                    bearings[idx], extents[idx], toas[idx])
        end
    end
    println("\nTop 20 LHS points:")
    for rank in 1:min(20, n_total)
        idx = order[rank]
        @printf("  %2d  %-22s loss=%.4f fms=%.4f shape=%.4f brg=%.4f ext=%.4f toa=%.4f\n",
                rank, labels[idx], losses[idx], fmss[idx], shapes[idx],
                bearings[idx], extents[idx], toas[idx])
    end

    best_idx = order[1]
    best_x = samples[best_idx]
    wf = joinpath(@__DIR__, "$(TEST_NAME)_lhs_best.txt")
    open(wf, "w") do io
        for (j, pn) in enumerate(PARAM_NAMES)
            @printf(io, "%s\t%.10g\n", pn, best_x[j])
        end
        @printf(io, "# source\t%s\n", labels[best_idx])
        @printf(io, "# loss\t%.6f\n", losses[best_idx])
        @printf(io, "# fms\t%.6f\n", fmss[best_idx])
        @printf(io, "# bearing\t%.6f\n", bearings[best_idx])
        @printf(io, "# extent\t%.6f\n", extents[best_idx])
        @printf(io, "# toa\t%.6f\n", toas[best_idx])
    end
    println("\nBest LHS point ($(labels[best_idx])) -> $(basename(wf))")
    println("Copy it to $(TEST_NAME)_cmaes_$(lowercase(string(TURB_SCHEME)))_best.txt " *
            "to warm-start BIPOP-CMA-ES.")
    exit()
end

# ============================================================================
# NC_EXPORT — write the H+12 dose-rate field (mR/h) for the current x0 to a
# CF-1.8 NetCDF on the WGS84 LON_GRID×LAT_GRID, for the RIVM intercomparison
# bundle. Reuses rho_core's LAST_DOSE_SMOOTH/RAW Refs — same forward sim as the
# calibration, no separate machinery. Env vars:
#   NC_EXPORT=1            enable
#   NC_OUT=<path>          output .nc path (default $(TEST)_dose.nc in @__DIR__)
#   NC_LABEL=<str>         scenario_label global attr (default TEST_CONFIG.label)
#   NC_IMPROVEMENT=<str>   "pre" or "post" tag stored as an attribute
#   VIZ_SEEDS=<n>          seeds to sweep for a representative field (default 5)
# x0 is the warm-start vector (best.txt if present), so set the checkpoint to the
# vector you want exported before running. For "pre", point the checkpoint at a
# baseline vector (see bundle_rivm_netcdf.jl).
# ============================================================================
if get(ENV, "NC_EXPORT", "0") == "1"
    n_seeds = parse(Int, get(ENV, "VIZ_SEEDS", "5"))
    println("\nNC_EXPORT: sweeping $n_seeds seeds at x0 for a representative field.")
    best_seed, best_diag = _viz_sweep(x0, TURB_SCHEME, n_seeds)
    println("Best seed: $(best_seed)  loss=$(round(best_diag.loss, digits=4)) " *
            "fms=$(round(best_diag.fms, digits=4)) brg=$(round(best_diag.bearing, digits=4))")
    # Rerun the best seed so LAST_DOSE_SMOOTH / LAST_DOSE_RAW hold its field.
    rho_core(x0, TURB_SCHEME, best_seed)
    dose_smooth = LAST_DOSE_SMOOTH[]
    dose_raw    = LAST_DOSE_RAW[]
    dose_smooth === nothing && error("NC_EXPORT: no dose field stashed — sim produced no deposition.")

    nc_out = get(ENV, "NC_OUT", joinpath(@__DIR__, "$(TEST_NAME)_dose.nc"))
    nc_label = get(ENV, "NC_LABEL", TEST_CONFIG.label)
    improvement = get(ENV, "NC_IMPROVEMENT", "")
    det_time = TEST_CONFIG.start_dt

    NCDataset(nc_out, "c") do ds
        defDim(ds, "longitude", length(LON_GRID))
        defDim(ds, "latitude", length(LAT_GRID))

        lon_var = defVar(ds, "longitude", Float64, ("longitude",))
        lon_var[:] = collect(LON_GRID)
        lon_var.attrib["units"] = "degrees_east"
        lon_var.attrib["long_name"] = "Longitude"
        lon_var.attrib["standard_name"] = "longitude"

        lat_var = defVar(ds, "latitude", Float64, ("latitude",))
        lat_var[:] = collect(LAT_GRID)
        lat_var.attrib["units"] = "degrees_north"
        lat_var.attrib["long_name"] = "Latitude"
        lat_var.attrib["standard_name"] = "latitude"

        dose_var = defVar(ds, "dose_rate_mR_hr", Float32, ("longitude", "latitude"),
                          fillvalue=Float32(-9999))
        dose_var[:, :] = Float32.(dose_smooth)
        dose_var.attrib["units"] = "mR/hr"
        dose_var.attrib["long_name"] = "Smoothed dose rate at H+12 (total mixed FP)"
        dose_var.attrib["reference_time"] =
            Dates.format(det_time + Dates.Hour(12), "yyyy-mm-dd HH:MM:SS") * " UTC"
        dose_var.attrib["valid_min"] = Float32(0.0)
        dose_var.attrib["grid_mapping"] = "crs"

        raw_var = defVar(ds, "dose_rate_raw_mR_hr", Float32, ("longitude", "latitude"),
                         fillvalue=Float32(-9999))
        raw_var[:, :] = Float32.(dose_raw)
        raw_var.attrib["units"] = "mR/hr"
        raw_var.attrib["long_name"] = "Raw (unsmoothed) dose rate at H+12 (total mixed FP)"
        raw_var.attrib["valid_min"] = Float32(0.0)
        raw_var.attrib["grid_mapping"] = "crs"

        crs_var = defVar(ds, "crs", Int32, ())
        crs_var.attrib["grid_mapping_name"] = "latitude_longitude"
        crs_var.attrib["long_name"] = "WGS84"
        crs_var.attrib["semi_major_axis"] = 6378137.0
        crs_var.attrib["inverse_flattening"] = 298.257223563

        ds.attrib["title"] = "PREDICT US-test intercomparison — $nc_label"
        ds.attrib["institution"] = "NuclearDetonation.jl"
        ds.attrib["source"] = "NuclearDetonation.jl Lagrangian particle dispersion"
        ds.attrib["Conventions"] = "CF-1.8"
        ds.attrib["test"] = TEST_CONFIG.label
        ds.attrib["yield_kt"] = TEST_CONFIG.yield_kt
        ds.attrib["source_lat"] = TEST_CONFIG.source_lat
        ds.attrib["source_lon"] = TEST_CONFIG.source_lon
        ds.attrib["detonation_time"] = Dates.format(det_time, "yyyy-mm-dd HH:MM:SS") * " UTC"
        ds.attrib["turbulence_scheme"] = TURB_NAME
        ds.attrib["n_particles"] = parse(Int, get(ENV, "N_PARTICLES", "2500"))
        ds.attrib["dose_decay"] = "Way-Wigner t^-1.2 to H+12"
        ds.attrib["smooth_sigma_cells"] = x0[20]
        isempty(improvement) || (ds.attrib["improvement_stage"] = improvement)
        ds.attrib["param_vector"] = join([@sprintf("%s=%.6g", PARAM_NAMES[k], x0[k])
                                          for k in 1:N_DIM], "; ")
        ds.attrib["fms_score"] = best_diag.fms
        ds.attrib["bearing_score"] = best_diag.bearing
        ds.attrib["combined_loss"] = best_diag.loss
    end
    println("NC_EXPORT: saved $(nc_out)  (improvement=$(isempty(improvement) ? "n/a" : improvement))")
    exit()
end

# BIPOP state
global_best_val = Inf
global_best_x = copy(x0)
global_best_diag = FAILED_EVAL
total_evals = 0
budget_large = 0   # evals spent on large-population restarts
budget_small = 0   # evals spent on small-population restarts
restart_count = 0
large_lambda = DEFAULT_LAMBDA  # doubles each large restart

results_file = joinpath(@__DIR__, "$(TEST_NAME)_cmaes_$(lowercase(string(TURB_SCHEME)))_results.txt")
checkpoint_file = joinpath(@__DIR__, "$(TEST_NAME)_cmaes_$(lowercase(string(TURB_SCHEME)))_best.txt")

t_start = time()

while total_evals < MAX_EVALS
    global total_evals, global_best_val, global_best_x, global_best_diag
    global restart_count, large_lambda, budget_large, budget_small

    # Decide restart type and sigma_frac
    # NOTE: global_best_x and x0 are in PHYSICAL units. run_x0 is built in
    # physical units here, then encoded once at the CMAES constructor below so
    # the optimiser explores log space for the scale params (see LOG_MASK).
    if restart_count == 0
        # First run: default lambda, standard sigma, counts toward large budget
        run_lambda = DEFAULT_LAMBDA
        run_type = :large
        run_sigma_frac = SIGMA_FRAC
        run_x0 = copy(x0)
    elseif budget_large <= budget_small
        # Large restart: double lambda, full exploration from global best
        large_lambda = min(large_lambda * 2, MAX_EVALS ÷ 10)
        run_lambda = large_lambda
        run_type = :large
        run_sigma_frac = SIGMA_FRAC
        run_x0 = copy(global_best_x)
    else
        # Small restart: default lambda, tighter random sigma, perturbed start.
        # The random component is drawn in ENCODED space so the perturbation is
        # log-uniform for scale params, then decoded back to physical units.
        run_lambda = DEFAULT_LAMBDA
        run_type = :small
        # Random sigma_frac in [0.01, 0.3] (log-uniform)
        run_sigma_frac = SIGMA_FRAC * 10.0^(-2.0 * rand())
        # Random starting point biased toward global best (mix in encoded space)
        mix = 0.3 + 0.4 * rand()
        rand_s = LB_S .+ rand(N_DIM) .* (UB_S .- LB_S)
        run_x0_s = mix .* encode_params(global_best_x) .+ (1.0 - mix) .* rand_s
        run_x0_s .= clamp.(run_x0_s, LB_S, UB_S)
        run_x0 = decode_params(run_x0_s)
    end

    remaining = MAX_EVALS - total_evals
    if remaining < run_lambda
        break
    end

    restart_count += 1
    run_evals = 0

    println("\n" * "-"^50)
    println("RESTART #$(restart_count) ($(run_type), lambda=$(run_lambda), sigma_frac=$(round(run_sigma_frac, digits=4)))")
    println("-"^50)

    # Construct CMA-ES in ENCODED (log) search space: encode the start point and
    # use the encoded bounds. All ask()/tell! internals operate in this space;
    # candidates are decoded inside evaluate_generation and before storage.
    es = CMAES(encode_params(run_x0); lb=LB_S, ub=UB_S, popsize=run_lambda, sigma_frac=run_sigma_frac)
    es.best_ever_val = global_best_val
    es.best_ever_x = encode_params(global_best_x)

    while total_evals + run_lambda <= MAX_EVALS
        # Draw fresh generation seed for CRN
        gen_seed = rand(UInt64)

        candidates = ask(es)
        eval_results = evaluate_generation(candidates, TURB_SCHEME, gen_seed)
        fitvals = [r.loss for r in eval_results]

        gen_best_val, gen_best_x = tell!(es, candidates, fitvals)
        total_evals += run_lambda
        run_evals += run_lambda

        # Find the best result this generation for diagnostics
        gen_best_idx = argmin(fitvals)
        gen_best_r = eval_results[gen_best_idx]

        # Update global best. gen_best_x is in ENCODED space (tell! returns a
        # candidate verbatim); decode to physical units before storing so the
        # checkpoint, warm-start reuse, and global_best_x stay physical.
        improved = false
        if gen_best_val < global_best_val
            global_best_val = gen_best_val
            global_best_x = decode_params(gen_best_x)
            global_best_diag = gen_best_r
            improved = true

            # Save checkpoint
            open(checkpoint_file, "w") do f
                for (j, pname) in enumerate(PARAM_NAMES)
                    println(f, "$(pname)\t$(global_best_x[j])")
                end
                println(f, "# loss\t$(global_best_val)")
                println(f, "# score_new\t$(1.0 - global_best_val)")
                println(f, "# score_old\t$(round(global_best_diag.combined_old * 100, digits=2))%")
                println(f, "# fms\t$(global_best_diag.fms)")
                println(f, "# shape\t$(global_best_diag.shape)")
                println(f, "# bearing\t$(global_best_diag.bearing)")
                println(f, "# extent\t$(global_best_diag.extent)")
                println(f, "# toa\t$(global_best_diag.toa)")
            end
        end

        marker = improved ? " ***" : ""
        elapsed = time() - t_start
        # Print both old-style score (comparable to APMC) and component breakdown
        @printf("  Gen %3d [%5d/%d] FMS=%.2f shp=%.2f brg=%.2f ext=%.2f toa=%.2f | old=%.1f%% new=%.1f%% | σ=%.3f [%.0fs]%s\n",
                es.generation, total_evals, MAX_EVALS,
                gen_best_r.fms, gen_best_r.shape, gen_best_r.bearing, gen_best_r.extent, gen_best_r.toa,
                gen_best_r.combined_old * 100,
                (1.0 - gen_best_val) * 100,
                es.sigma, elapsed, marker)
        flush(stdout)

        # Check restart conditions
        do_restart, reason = should_restart(es)
        if do_restart
            println("  -> Restart triggered: $(reason)")
            break
        end
    end

    # Update BIPOP budget tracking
    if run_type == :large
        budget_large += run_evals
    else
        budget_small += run_evals
    end

    println("  Run used $(run_evals) evals ($(run_type)). Budget: large=$(budget_large), small=$(budget_small)")
end

t_elapsed = time() - t_start

# ============================================================================
# RESULTS
# ============================================================================

println("\n" * "="^70)
println("BIPOP-CMA-ES COMPLETE — $(TURB_NAME)")
println("="^70)
println("Total evaluations: $(total_evals)")
println("Restarts: $(restart_count)")
println("Wall time: $(round(t_elapsed/60, digits=1)) minutes")
println("Best loss: $(round(global_best_val, digits=6))")
println("\nScore breakdown (best particle):")
println("  FMS (geo mean):   $(round(global_best_diag.fms, digits=4))")
println("  Shape:            $(round(global_best_diag.shape, digits=4))")
println("  Bearing:          $(round(global_best_diag.bearing, digits=4))")
println("  Extent:           $(round(global_best_diag.extent, digits=4))")
println("  TOA:              $(round(global_best_diag.toa, digits=4))")
println("  Old combined (50%FMS+20%ext+30%TOA): $(round(global_best_diag.combined_old * 100, digits=2))%  <- compare to APMC")
println("  New combined (30%FMS+10%shp+20%brg+10%ext+30%TOA): $(round((1.0 - global_best_val) * 100, digits=2))%")

println("\nBest parameters:")
for (j, pname) in enumerate(PARAM_NAMES)
    println("  $(rpad(pname, 24)) $(round(global_best_x[j], digits=4))")
end

# Interpret particle size
println("\nParticle size distribution:")
println("  Fine mode:   $(round(global_best_x[1], digits=1)) μm (sigma_g=$(round(global_best_x[2], digits=2)))")
println("  Coarse mode: $(round(global_best_x[3], digits=1)) μm (sigma_g=$(round(global_best_x[4], digits=2)))")
println("  Fine fraction: $(round(global_best_x[5]*100, digits=1))%")
frac_l = global_best_x[6]
frac_m = global_best_x[7]
frac_u_raw = max(1.0 - frac_l - frac_m, 0.05)
ft = frac_l + frac_m + frac_u_raw
st = round(global_best_x[21], digits=0)
cm = round(global_best_x[22], digits=0)
ct = round(global_best_x[23], digits=0)
println("\nLayer geometry & mass fractions:")
println("  Lower  (0–$(st) m):        $(round(frac_l/ft*100, digits=1))%")
println("  Middle ($(st)–$(cm) m):     $(round(frac_m/ft*100, digits=1))%")
println("  Upper  ($(cm)–$(ct) m):     $(round(frac_u_raw/ft*100, digits=1))%")
println("\nCalibration:")
println("  Total activity: $(round(global_best_x[19], digits=1))×10¹⁵ Bq = $(round(global_best_x[19]*1e15, sigdigits=3)) Bq")
println("  Smooth sigma:   $(round(global_best_x[20], digits=2)) cells")

# Save full results
open(results_file, "w") do f
    println(f, "$(TEST_CONFIG.label) BIPOP-CMA-ES Results — $(TURB_NAME)")
    println(f, "="^60)
    println(f, "Total evaluations: $(total_evals)")
    println(f, "Restarts: $(restart_count)")
    println(f, "Wall time: $(round(t_elapsed/60, digits=1)) minutes")
    println(f, "Best loss: $(global_best_val)")
    println(f, "Best score: $(round((1.0 - global_best_val) * 100, digits=2))%")
    println(f, "\nBest parameters:")
    for (j, pname) in enumerate(PARAM_NAMES)
        println(f, "  $(pname): $(global_best_x[j])")
    end
    println(f, "\nParticle size distribution:")
    println(f, "  Fine: $(round(global_best_x[1], digits=1)) μm (σ_g=$(round(global_best_x[2], digits=2)))")
    println(f, "  Coarse: $(round(global_best_x[3], digits=1)) μm (σ_g=$(round(global_best_x[4], digits=2)))")
    println(f, "  Fine fraction: $(round(global_best_x[5]*100, digits=1))%")
    println(f, "\nBounds:")
    for (j, pname) in enumerate(PARAM_NAMES)
        println(f, "  $(pname): [$(LB[j]), $(UB[j])]")
    end
end

println("\nResults saved to $(basename(results_file))")
println("Checkpoint saved to $(basename(checkpoint_file))")
