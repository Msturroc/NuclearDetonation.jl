#!/usr/bin/env julia
# CPU reference shadow — calls Transport.run_simulation! directly.
# =================================================================
# This file produces the GROUND TRUTH `state.deposition_log` for a Nancy
# forward simulation, by calling the package's own `Transport.run_simulation!`
# with controlled seeding and returning the raw deposition events. Any future
# CPU/Float32/GPU shadow we write must reproduce these events exactly.
#
# The package uses the GLOBAL Mersenne Twister for `randn()` calls inside
# `integrate_timestep!` (turbulence). We seed it explicitly before each call
# and run serially. Don't @threads this — global rng + threads = nondeterminism.
#
# Usage: load this file once after the upstream cmaes script is included
# (which provides MET_CACHE, DOMAIN, etc.). Then call
# `run_reference_simulation(params, gen_seed)` to get a NamedTuple with the
# deposition log + hash.

using Random
using SHA
using StaticArrays
using NuclearDetonation
using NuclearDetonation.Transport

# This file assumes the upstream `nancy_cmaes_particle_size.jl` has already
# been `include`d at module scope (with ENV["MAX_EVALS"]="0" to suppress its
# own loop) so that the following names are bound:
#   MET_CACHE, DOMAIN, NX, NY, NK, MET_FORMAT, ERA5_FILES,
#   CACHE_START_FILE, CACHE_END_FILE, RELEASE_X, RELEASE_Y,
#   LAYER_LOWER, LAYER_MIDDLE, LAYER_UPPER, RELEASE_HEIGHT_M,
#   generate_bimodal_bins, compute_bimodal_weights

"""
    run_reference_simulation(params::Vector{Float64}, gen_seed::UInt64)

Run ONE Nancy forward simulation through the upstream
`Transport.run_simulation!` with deterministic seeding.

Returns a NamedTuple with:
- `deposition_log`  — Vector{DepositionEvent} as accumulated by the sim
- `n_particles`     — number of particles at the end
- `n_dep_events`    — number of deposition events logged
- `total_dep_bq`    — sum of all deposited mass (Bq)
- `dep_hash`        — SHA-256 hex of the (sorted, rounded) deposition log,
                      used for byte-equal comparisons across runs
- `final_positions` — Vector of SVector{3,Float64} with each surviving
                      particle's (x_domain, y_domain, σ) state

The function takes a 20-dim `params` vector in the same layout as
`rho_core`'s argument and a `gen_seed::UInt64` that gets fed BOTH to a local
MersenneTwister (for particle bin assignment) AND to `Random.seed!()` (for
the global rng used by turbulence inside `run_simulation!`).
"""
function run_reference_simulation(params::Vector{Float64}, gen_seed::UInt64)
    # --- Unpack params (verbatim from rho_core) ---
    d_median_fine     = params[1]
    sigma_g_fine      = params[2]
    d_median_coarse   = params[3]
    sigma_g_coarse    = params[4]
    frac_fine         = params[5]
    frac_lower        = params[6]
    frac_middle       = params[7]
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

    frac_upper = clamp(1.0 - frac_lower - frac_middle, 0.05, 1.0)

    # Local rng for size-bin assignments (same as rho_core's CRN trick).
    rng = Random.MersenneTwister(gen_seed)

    size_bins = generate_bimodal_bins(d_median_fine, sigma_g_fine,
                                      d_median_coarse, sigma_g_coarse;
                                      n_bins=15)
    bin_weights = compute_bimodal_weights(d_median_fine, sigma_g_fine,
                                          d_median_coarse, sigma_g_coarse,
                                          frac_fine, size_bins)

    n_particles = 1000
    total_activity = activity_scale * 1.0e15
    n_lower  = round(Int, n_particles * frac_lower)
    n_middle = round(Int, n_particles * frac_middle)
    n_upper  = n_particles - n_lower - n_middle

    sources = [
        Transport.ReleaseSource((RELEASE_X, RELEASE_Y), LAYER_LOWER,
                                Transport.BombRelease(0.0),
                                [total_activity * frac_lower],  max(n_lower, 1)),
        Transport.ReleaseSource((RELEASE_X, RELEASE_Y), LAYER_MIDDLE,
                                Transport.BombRelease(0.0),
                                [total_activity * frac_middle], max(n_middle, 1)),
        Transport.ReleaseSource((RELEASE_X, RELEASE_Y), LAYER_UPPER,
                                Transport.BombRelease(0.0),
                                [total_activity * frac_upper],  max(n_upper, 1)),
    ]
    decay_params = [Transport.DecayParams(kdecay=Transport.NoDecay, halftime_hours=0.0)]

    state = Transport.initialize_simulation(DOMAIN, sources, ["MixedFP"], decay_params;
                                            log_depositions=true)

    init_met = MET_CACHE[(CACHE_START_FILE, 1)]

    # Generate release particles using the CRN local rng — exactly as rho_core
    positions_m = Tuple{Float64,Float64,Float64}[]
    activities  = Float64[]
    for src in sources
        pos_s, act_s, released_s = Transport.generate_release_particles(
            rng, src, 0, 1,
            ones(Float64, NX, NY), ones(Float64, NY, NY),
            DOMAIN.dx, DOMAIN.dy, DOMAIN.hlevel)
        if released_s && !isempty(pos_s)
            append!(positions_m, pos_s)
            append!(activities,  act_s)
        end
    end

    if isempty(positions_m)
        return (deposition_log = Transport.DepositionEvent{Float64}[],
                n_particles = 0, n_dep_events = 0,
                total_dep_bq = 0.0, dep_hash = "",
                final_positions = SVector{3,Float64}[])
    end

    n_part = length(positions_m)
    n_classes = length(size_bins)
    snap_size_bins = [Transport.ParticleProperties(diameter_μm=b.d, density_gcm3=2.5)
                      for b in size_bins]
    particle_radii         = Float64[]
    particle_densities     = Float64[]
    particle_size_indices  = Int[]
    fixed_gravity = [b.v * vgrav_scale for b in size_bins]

    cum_weights = cumsum(bin_weights)
    base_density = 2500.0

    assigned_bins = Vector{Int}(undef, n_part)
    for i in 1:n_part
        r = rand(rng)
        idx = searchsortedfirst(cum_weights, r)
        assigned_bins[i] = clamp(idx, 1, n_classes)
    end

    for i in 1:n_part
        pos = positions_m[i]
        activity = activities[i]
        sigma_z = Transport.height_to_sigma_hybrid(RELEASE_X, RELEASE_Y, pos[3], init_met, 0.0)
        Transport.add_particle!(state.ensemble,
                                SVector{3,Float64}(pos[1], pos[2], sigma_z),
                                SVector{3,Float64}(0.0, 0.0, 0.0),
                                [activity], 0.0, icomp=1)

        idx = assigned_bins[i]
        push!(particle_radii,        size_bins[idx].d * 0.5e-6)
        push!(particle_densities,    base_density)
        push!(particle_size_indices, idx)
        state.ensemble.particles[i].grv = Float32(size_bins[idx].v * 0.01 * vgrav_scale)
    end

    particle_size_config = Transport.ParticleSizeConfig(
        size_bins = snap_size_bins, particle_radii = particle_radii,
        particle_densities = particle_densities,
        particle_size_indices = particle_size_indices,
        fixed_gravity_cm_s = fixed_gravity)

    hanna_config = Transport.HannaTurbulenceConfig{Float64}(
        sigma_scale = sigma_h_scale,
        sigma_scale_vertical = sigma_w_scale,
        tl_scale = tl_scale,
        use_cbl = true)

    deposition_config = Transport.DepositionConfig{Float64}(
        apply_dry_deposition = true,
        apply_wet_deposition = false,
        use_simple_deposition = true,
        simple_deposition_velocity = 0.002 * vd_scale,
        simple_surface_height = 30.0 * surface_height_scale,
        mixing_height = 1000.0 * mixing_height_scale,
        surface_roughness = 0.1 * roughness_scale)

    snapshot_times = [Float64(h) * 3600.0 for h in 1:12]
    numerical_config = Transport.ERA5NumericalConfig{Float64}(
        interpolation_order = Transport.LinearInterp,
        ode_solver_type = :Euler,
        fixed_dt = 300.0,
        turbulence = Transport.OrnsteinUhlenbeck)

    sim_config = Transport.SimulationConfig{Float64}(
        saveat = snapshot_times, verbose = false,
        max_duration = 12.0 * 3600.0, save_snapshots = true,
        dt_particle = 300.0, use_reference_stepping = true,
        max_files = CACHE_END_FILE - CACHE_START_FILE + 1,
        omega_scale = omega_scale)

    # *** THE MAGIC — seed the global rng so turbulence is deterministic. ***
    Random.seed!(gen_seed)

    Transport.run_simulation!(state, ERA5_FILES,
        particle_size_config = particle_size_config,
        deposition_config    = deposition_config,
        hanna_config         = hanna_config,
        decay_params         = decay_params,
        config               = sim_config,
        numerical_config     = numerical_config,
        advection_enabled = true, settling_enabled = true,
        dry_deposition_enabled = true, wet_deposition_enabled = false,
        release_height_m = RELEASE_HEIGHT_M,
        met_data_cache = MET_CACHE,
        met_format_override = MET_FORMAT,
        met_dimensions = (NX, NY, NK),
        cache_init_file_idx = CACHE_START_FILE,
        cache_init_time_idx = 1,
        sigma_already_initialized = true)

    dep_log = state.deposition_log
    total_bq = sum(e.mass for e in dep_log; init = 0.0)

    # Hash the deposition log: round each field to 12 decimal places to avoid
    # spurious diffs from inconsequential rounding, sort to a canonical order,
    # then SHA-256 the resulting byte string.
    rounded = sort([(round(e.x, digits=12), round(e.y, digits=12),
                     round(e.mass, digits=12), round(e.time, digits=12),
                     e.component) for e in dep_log])
    io = IOBuffer()
    for tup in rounded
        write(io, tup[1]); write(io, tup[2]); write(io, tup[3])
        write(io, tup[4]); write(io, Int32(tup[5]))
    end
    dep_hash = bytes2hex(SHA.sha256(take!(io)))

    final_positions = [state.ensemble.positions[i] for i in 1:length(state.ensemble.particles)]

    return (deposition_log = dep_log,
            n_particles = length(state.ensemble.particles),
            n_dep_events = length(dep_log),
            total_dep_bq = total_bq,
            dep_hash = dep_hash,
            final_positions = final_positions)
end
