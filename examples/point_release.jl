#!/usr/bin/env julia
# Point Release Example
# =====================
# Simulates a constant point-source release using ERA5 reanalysis data.
# Demonstrates a non-weapon use case (e.g. industrial stack in Nevada).
#
# This example shows how to:
#   - Configure a constant release (vs bomb release)
#   - Use OrnsteinUhlenbeck turbulence with Tsit5 solver
#   - Set up a single particle size bin
#
# Requirements:
#   - ERA5 data artifact (~96 MB, downloaded automatically on first run)
#
# Usage:
#   julia --project=.. point_release.jl

using NuclearDetonation
using NuclearDetonation.Transport
using NCDatasets
using StaticArrays
using Random
using Dates

println("="^70)
println("POINT RELEASE — Ornstein-Uhlenbeck + Tsit5")
println("="^70)

# ============================================================================
# 1. Load ERA5 met data
# ============================================================================

println("\n1. Loading ERA5 met data...")
era5_files = nancy_era5_files()
println("   Found $(length(era5_files)) ERA5 files")

met_format = Transport.detect_met_format(era5_files[1])
nx_met, ny_met, nk_met = NCDataset(era5_files[1]) do ds
    Transport.get_met_dimensions(met_format, ds)
end

# Pre-cache met fields for files 5–11 (covers 12 h from 13:00 UTC)
met_cache = Dict{Tuple{Int,Int}, Transport.MeteoFields}()
for file_idx in 5:min(11, length(era5_files))
    NCDataset(era5_files[file_idx]) do ds
        times = Transport.get_time_variable(met_format, ds)
        for t_idx in 1:length(times)
            mf = Transport.MeteoFields(nx_met, ny_met, nk_met, T=Float32)
            t2 = t_idx < length(times) ? t_idx + 1 : t_idx
            Transport.read_initial_met_fields!(met_format, mf, ds, t_idx, t2)
            met_cache[(file_idx, t_idx)] = mf
        end
    end
end
println("   Loaded $(length(met_cache)) timesteps")

# ============================================================================
# 2. Set up domain and source
# ============================================================================

println("\n2. Setting up domain...")
lat_range, lon_range = NCDataset(era5_files[1]) do ds
    Float64.(ds["latitude"][:]), Float64.(ds["longitude"][:])
end

start_dt = Dates.DateTime(1953, 3, 24, 13, 0)
domain = Transport.SimulationDomain(
    lon_min = minimum(lon_range), lon_max = maximum(lon_range),
    lat_min = minimum(lat_range), lat_max = maximum(lat_range),
    z_min = 0.0, z_max = 35000.0, nx = nx_met, ny = ny_met, nz = nk_met,
    start_time = start_dt, end_time = start_dt + Dates.Hour(12),
)

# Release from a 100 m stack at the Nevada Test Site
release_lat, release_lon = 37.0956, -116.1028
release_x, release_y = Transport.latlon_to_grid(domain, release_lat, release_lon)
println("   Release: ($(round(release_lat, digits=2))°N, $(round(release_lon, digits=2))°E)")
println("   Grid: ($(round(release_x, digits=1)), $(round(release_y, digits=1)))")

# Column release: 90–110 m height (simulating a 100 m stack)
geometry = ColumnRelease(90.0, 110.0)
n_particles = 1000
total_activity = 1e12  # 1 TBq

source = ReleaseSource(
    (release_x, release_y), geometry,
    ConstantRelease(), [total_activity], n_particles,
)

# ============================================================================
# 3. Initialise simulation state
# ============================================================================

println("\n3. Initialising particles...")
# Cs-137: 30.17 year half-life
decay_params = [Transport.DecayParams(
    kdecay = Transport.ExponentialDecay,
    halftime_hours = 30.17 * 365.25 * 24.0,
)]
state = Transport.initialize_simulation(domain, [source], ["Cs137"], decay_params;
                                         log_depositions=true)

rng = Random.MersenneTwister(42)
init_met = met_cache[(5, 1)]

# Generate particles from the source
pos_s, act_s, released_s = Transport.generate_release_particles(
    rng, source, 0, 1,
    ones(Float64, nx_met, ny_met), ones(Float64, ny_met, ny_met),
    domain.dx, domain.dy, domain.hlevel,
)

if released_s && !isempty(pos_s)
    for (pos, activity) in zip(pos_s, act_s)
        sigma_z = Transport.height_to_sigma_hybrid(
            release_x, release_y, pos[3], init_met, 0.0)
        Transport.add_particle!(state.ensemble,
            SVector{3,Float64}(pos[1], pos[2], sigma_z),
            SVector{3,Float64}(0.0, 0.0, 0.0),
            [activity], 0.0, icomp=1)
    end
end
println("   $(length(state.ensemble.particles)) particles generated")

# ============================================================================
# 4. Configure physics
# ============================================================================

# Single particle size bin: 5 μm aerosol
particle = ParticleProperties(diameter_μm=5.0, density_gcm3=2.0)
psc = ParticleSizeConfig(size_bins=[particle])

hanna = HannaTurbulenceConfig{Float64}(use_cbl=true)

dep = Transport.DepositionConfig{Float64}(
    apply_dry_deposition=true, apply_wet_deposition=false,
    use_simple_deposition=true, simple_deposition_velocity=0.002)

# Tsit5 solver with O-U turbulence
num_cfg = ERA5NumericalConfig{Float64}(
    interpolation_order=Transport.LinearInterp,
    ode_solver_type=:Tsit5,
    fixed_dt=300.0,
    turbulence=Transport.OrnsteinUhlenbeck,
    store_turbulent_velocities=true,
    name="point_release_tsit5",
)

sim_cfg = Transport.SimulationConfig{Float64}(
    saveat=[12.0 * 3600.0], verbose=true, max_duration=12.0 * 3600.0,
    save_snapshots=true, dt_particle=300.0, use_reference_stepping=true,
    max_files=7)

# ============================================================================
# 5. Run simulation
# ============================================================================

println("\n4. Running 12-hour point release simulation (Tsit5 solver)...")
Transport.run_simulation!(state, era5_files,
    particle_size_config=psc, deposition_config=dep,
    hanna_config=hanna, decay_params=decay_params, config=sim_cfg,
    numerical_config=num_cfg, advection_enabled=true, settling_enabled=false,
    dry_deposition_enabled=true, wet_deposition_enabled=false,
    release_height_m=110.0, met_data_cache=met_cache,
    met_format_override=met_format, met_dimensions=(nx_met, ny_met, nk_met),
    cache_init_file_idx=5, cache_init_time_idx=1,
    sigma_already_initialized=true)

n_active = count(Transport.is_active(p) for p in state.ensemble.particles)
n_deposited = length(state.deposition_log)
println("\n   Simulation complete")
println("   Active particles: $n_active / $(length(state.ensemble.particles))")
println("   Deposition events: $n_deposited")
