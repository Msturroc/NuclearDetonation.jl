#!/usr/bin/env julia
# ETEX Release 1 — Simulation with NuclearDetonation.jl
# =====================================================
# Simulates the ETEX-1 tracer release from Monterfil, Brittany
# using ERA5 model-level data downloaded from NCAR RDA.

using NuclearDetonation
using NuclearDetonation.Transport
using NCDatasets
using StaticArrays
using Random
using Dates

println("="^70)
println("ETEX Release 1 — Point Release Simulation")
println("="^70)

# ============================================================================
# 1. Load ERA5 data
# ============================================================================

println("\n1. Loading ERA5 data...")
era5_dir = joinpath(@__DIR__, "ERA5_data")
era5_files = sort(filter(f -> endswith(f, "_snap.nc"), readdir(era5_dir, join=true)))
println("   Found $(length(era5_files)) ERA5 files")
println("   First: $(basename(era5_files[1]))")
println("   Last:  $(basename(era5_files[end]))")

met_format = Transport.detect_met_format(era5_files[1])
nx_met, ny_met, nk_met = NCDataset(era5_files[1]) do ds
    Transport.get_met_dimensions(met_format, ds)
end
println("   Grid: $(nx_met) x $(ny_met) x $(nk_met)")

lat_range, lon_range = NCDataset(era5_files[1]) do ds
    Float64.(ds["latitude"][:]), Float64.(ds["longitude"][:])
end
println("   Lat: $(round(minimum(lat_range), digits=1)) to $(round(maximum(lat_range), digits=1))")
println("   Lon: $(round(minimum(lon_range), digits=1)) to $(round(maximum(lon_range), digits=1))")

# Pre-cache met fields for the release period
# ETEX release: 23 Oct 16:00 - 24 Oct 03:50 UTC
# Files are 3-hour chunks, release starts in file index 6 (15-17Z)
println("\n   Caching met fields...")
met_cache = Dict{Tuple{Int,Int}, Transport.MeteoFields}()
# Cache files covering 23 Oct 12Z through 26 Oct (files 5-24 or so)
for file_idx in 5:min(24, length(era5_files))
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
println("   Cached $(length(met_cache)) timesteps")

# ============================================================================
# 2. Set up domain and source
# ============================================================================

println("\n2. Setting up domain...")

# ETEX release: 23 Oct 1994, 16:00 UTC, 12-hour release duration
# Simulate 72 hours of transport
start_dt = Dates.DateTime(1994, 10, 23, 12, 0)  # start 4h before release
duration_hours = 72

domain = Transport.SimulationDomain(
    lon_min = minimum(lon_range), lon_max = maximum(lon_range),
    lat_min = minimum(lat_range), lat_max = maximum(lat_range),
    z_min = 0.0, z_max = 35000.0,
    nx = nx_met, ny = ny_met, nz = nk_met,
    start_time = start_dt,
    end_time = start_dt + Dates.Hour(duration_hours),
)

# Release from Monterfil, Brittany: 48.058°N, 2.008°W
# 340 kg PMCH over ~12 hours from ~8m chimney
release_lat, release_lon = 48.058, -2.008
release_x, release_y = Transport.latlon_to_grid(domain, release_lat, release_lon)
println("   Release: Monterfil ($(release_lat)°N, $(release_lon)°E)")
println("   Grid: ($(round(release_x, digits=1)), $(round(release_y, digits=1)))")

# Column release: 5-15 m height (chimney at ~8m)
n_particles = 5000
total_activity = 340e3  # 340 kg in grams (as proxy for activity)

geometry = ColumnRelease(5.0, 15.0)
source = ReleaseSource(
    (release_x, release_y), geometry,
    ConstantRelease(), [total_activity], n_particles,
)

# ============================================================================
# 3. Initialise simulation
# ============================================================================

println("\n3. Initialising particles...")
# No decay for PMCH tracer
decay_params = [Transport.DecayParams(kdecay=Transport.NoDecay, halftime_hours=0.0)]
state = Transport.initialize_simulation(domain, [source], ["PMCH"], decay_params;
                                         log_depositions=true)

rng = Random.MersenneTwister(42)
init_met = met_cache[(5, 1)]

particle_prop = ParticleProperties(diameter_μm=1.0, density_gcm3=1.2)  # gas tracer
npp_radii = Float64[]
npp_densities = Float64[]
npp_size_indices = Int[]

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
        push!(npp_radii, 1.0 * 0.5e-6)
        push!(npp_densities, 1200.0)
        push!(npp_size_indices, 1)
    end
end
println("   $(length(state.ensemble.particles)) particles generated")

# ============================================================================
# 4. Configure physics
# ============================================================================

println("\n4. Configuring physics...")
psc = ParticleSizeConfig(size_bins=[particle_prop],
    particle_radii=npp_radii, particle_densities=npp_densities,
    particle_size_indices=npp_size_indices)

hanna = HannaTurbulenceConfig{Float64}(use_cbl=true)

dep = Transport.DepositionConfig{Float64}(
    apply_dry_deposition=true, apply_wet_deposition=false,
    use_simple_deposition=true, simple_deposition_velocity=0.001)

num_cfg = ERA5NumericalConfig{Float64}(
    interpolation_order=Transport.LinearInterp,
    ode_solver_type=:Euler, fixed_dt=300.0,
    turbulence=Transport.OrnsteinUhlenbeck)

out_cfg = OutputConfig(
    trace_frequency=TRACE_DISABLED,
    verbosity=VERBOSITY_QUIET,
    trace_enabled=false)

sim_cfg = Transport.SimulationConfig{Float64}(
    saveat=[Float64(h) * 3600.0 for h in 1:duration_hours],
    verbose=true, max_duration=Float64(duration_hours) * 3600.0,
    save_snapshots=true, dt_particle=300.0, use_reference_stepping=true,
    max_files=length(era5_files),
    output_config=out_cfg)

# ============================================================================
# 5. Run simulation
# ============================================================================

println("\n5. Running $(duration_hours)-hour ETEX simulation...")
Transport.run_simulation!(state, era5_files,
    particle_size_config=psc, deposition_config=dep,
    hanna_config=hanna, decay_params=decay_params, config=sim_cfg,
    numerical_config=num_cfg, advection_enabled=true, settling_enabled=false,
    dry_deposition_enabled=true, wet_deposition_enabled=false,
    release_height_m=15.0, met_data_cache=met_cache,
    met_format_override=met_format,
    met_dimensions=(nx_met, ny_met, nk_met),
    cache_init_file_idx=5, cache_init_time_idx=1,
    sigma_already_initialized=true)

n_active = count(Transport.is_active(p) for p in state.ensemble.particles)
n_deposited = length(state.deposition_log)
println("\n   Simulation complete!")
println("   Active particles: $n_active / $(length(state.ensemble.particles))")
println("   Deposition events: $n_deposited")
