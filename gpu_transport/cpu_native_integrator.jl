#!/usr/bin/env julia
# Allocation-light CPU integrator (Track A of the Nancy speed-up plan)
# =====================================================================
# Goal: drop single-thread per-eval wall time from 4.5 s (the package's
# Transport.run_simulation!) toward ~3 s by stripping bookkeeping that
# never affects state.deposition_log:
#
#   - trace file open/write (TRACE_DISABLED via OutputConfig)
#   - accumulate_concentration!        (writes state.fields.atm_conc)
#   - snapshot saving                  (writes Vector{SimulationSnapshot})
#   - h_interp alignment debug checks  (winds0/winds blocks)
#   - istep=0 trace writes             (lines 1877-1901 in run_simulation!)
#
# The inner kernel is the unmodified Transport.integrate_timestep!, so the
# floating-point arithmetic and randn() draw order are byte-for-byte the
# same as the reference. The deposition_log SHA-256 must match.
#
# The script assumes that the upstream `nancy_cmaes_particle_size.jl` has
# already been included in the calling scope (with MAX_EVALS=0 to suppress
# its loop) so that MET_CACHE / DOMAIN / NX / NY / NK / etc. are bound.

using Random
using SHA
using StaticArrays
using NuclearDetonation
using NuclearDetonation.Transport

# ---------------------------------------------------------------------------
# run_native_simulation!  — fork of Transport.run_simulation! with bookkeeping
# stripped out. Mutates `state` in place. Always uses the cache fast path
# (the optimisation pipeline always hands us a populated met_data_cache).
# ---------------------------------------------------------------------------
function run_native_simulation!(state::Transport.SimulationState{T},
                                era5_files::Vector{String};
                                particle_size_config::Transport.ParticleSizeConfig,
                                deposition_config::Transport.DepositionConfig{T},
                                hanna_config::Union{Nothing,Transport.HannaTurbulenceConfig{T}},
                                decay_params::Vector{Transport.DecayParams{T}},
                                config::Transport.SimulationConfig{T},
                                advection_enabled::Bool,
                                settling_enabled::Bool,
                                dry_deposition_enabled::Bool,
                                wet_deposition_enabled::Bool,
                                numerical_config::Union{Nothing,Transport.ERA5NumericalConfig,Transport.NumericalConfig},
                                met_data_cache::Dict,
                                met_format::Transport.MetFormat,
                                met_dimensions::Tuple{Int,Int,Int},
                                cache_init_file_idx::Int,
                                cache_init_time_idx::Int) where T<:Real

    nx_met, ny_met, nk_met = met_dimensions
    met_fields = Transport.MeteoFields(nx_met, ny_met, nk_met, T=Float32)

    # OutputConfig that tells integrate_timestep! to skip every trace branch.
    silent_oc = Transport.OutputConfig(
        trace_frequency = Transport.TRACE_DISABLED,
        verbosity       = Transport.VERBOSITY_QUIET,
        trace_enabled   = false,
        progress_interval_hours            = 0.0,
        settling_diagnostic_interval_hours = 0.0,
    )
    silent_config = Transport.SimulationConfig{T}(
        dt_output    = config.dt_output,
        saveat       = config.saveat,
        dt_met       = config.dt_met,
        save_snapshots = false,
        verbose      = false,
        max_files    = config.max_files,
        max_duration = config.max_duration,
        reltol       = config.reltol,
        abstol       = config.abstol,
        dt_particle  = config.dt_particle,
        use_trilinear_gridding = config.use_trilinear_gridding,
        omega_scale  = config.omega_scale,
        use_reference_stepping = config.use_reference_stepping,
        output_config = silent_oc,
    )

    # ------------------------------------------------------------------
    # Init met fields from cache (matches run_simulation! cache path).
    # ------------------------------------------------------------------
    init_file_idx = cache_init_file_idx
    init_time_idx1 = cache_init_time_idx

    cached_mf = met_data_cache[(init_file_idx, init_time_idx1)]
    Transport.copy_met_fields!(met_fields, cached_mf)
    Transport.update_domain_vertical!(state.domain, met_fields)
    state.domain.xm .= met_fields.xm
    state.domain.ym .= met_fields.ym

    init_time_diff = T(3600.0)  # cached path always 1 h
    n_files = silent_config.max_files > 0 ?
              min(silent_config.max_files, length(era5_files)) :
              length(era5_files)

    # winds0 — used for the istep=0 physics step. We skip the alignment
    # debug check that the package wraps around it.
    winds0 = Transport.create_wind_interpolants(
        met_fields, 0.0, init_time_diff,
        config = numerical_config,
        negate_v = false,
        negate_w = false,
        lon_min = state.domain.lon_min,
        lon_max = state.domain.lon_max,
        lat_min = state.domain.lat_min,
        lat_max = state.domain.lat_max)

    # We assume sigma_already_initialized = true (cpu_reference.jl always does)
    # so the istep=0 trace-write loop in run_simulation! is pure trace I/O —
    # skip it entirely.

    # PARITY: still perform the istep=0 physics step.
    _ = Transport.integrate_timestep!(state, winds0, T(silent_config.dt_particle),
        particle_size_config, deposition_config, decay_params, silent_config;
        hanna_config        = hanna_config,
        advection_enabled   = advection_enabled,
        settling_enabled    = settling_enabled,
        dry_enabled         = dry_deposition_enabled,
        wet_enabled         = wet_deposition_enabled,
        current_time_global = T(0.0),
        local_time_offset   = T(0.0),
        numerical_config    = numerical_config,
        trace_filename      = "",
        trace_time_override = T(0.0),
        output_config       = silent_oc)

    current_time = 0.0
    file_range_start = init_file_idx
    file_range_end = min(init_file_idx + n_files - 1, length(era5_files))

    for file_idx in file_range_start:file_range_end
        # Count cached time windows for this file
        n_time_windows_file = 0
        for k in keys(met_data_cache)
            if k[1] == file_idx
                n_time_windows_file = max(n_time_windows_file, k[2])
            end
        end
        n_time_windows_file = max(0, n_time_windows_file - 1)

        for window_idx in 1:n_time_windows_file
            if !(file_idx == init_file_idx && window_idx == cache_init_time_idx)
                cached_mf2 = met_data_cache[(file_idx, window_idx)]
                Transport.copy_met_fields!(met_fields, cached_mf2)
                state.domain.xm .= met_fields.xm
                state.domain.ym .= met_fields.ym
            end
            time_diff = 3600.0

            n_substeps = max(1, Int(ceil(time_diff / silent_config.dt_particle)))
            dt_sub = time_diff / n_substeps

            # Build wind interpolants ONCE per met window (same as upstream).
            winds = Transport.create_wind_interpolants(
                met_fields, 0.0, time_diff,
                config = numerical_config,
                negate_v = false,
                negate_w = false,
                lon_min = state.domain.lon_min,
                lon_max = state.domain.lon_max,
                lat_min = state.domain.lat_min,
                lat_max = state.domain.lat_max)
            # NOTE: alignment debug check skipped on purpose.

            local_time = 0.0
            all_done = false

            for sub_idx in 1:n_substeps
                if silent_config.max_duration > 0 && current_time >= silent_config.max_duration
                    all_done = true
                    break
                end

                Transport.prepare_decay_rates!(decay_params, dt_sub; bomb_state = nothing)

                n_active = 0
                @inbounds for p in state.ensemble.particles
                    if Transport.is_active(p)
                        n_active += 1
                    end
                end
                if n_active == 0
                    all_done = true
                    break
                end

                Transport.integrate_timestep!(state, winds, T(dt_sub),
                    particle_size_config, deposition_config, decay_params, silent_config;
                    hanna_config        = hanna_config,
                    advection_enabled   = advection_enabled,
                    settling_enabled    = settling_enabled,
                    dry_enabled         = dry_deposition_enabled,
                    wet_enabled         = wet_deposition_enabled,
                    current_time_global = T(current_time),
                    local_time_offset   = T(local_time),
                    numerical_config    = numerical_config,
                    trace_filename      = "",
                    output_config       = silent_oc)

                current_time += dt_sub
                local_time   += dt_sub
                state.timestep += 1

                # Skipped: accumulate_concentration!, snapshot save, verbose printf
            end

            if all_done
                break
            end
        end
    end

    return nothing
end

# ---------------------------------------------------------------------------
# run_native_simulation — top-level wrapper, signature mirrors
# run_reference_simulation in cpu_reference.jl.
# ---------------------------------------------------------------------------
function run_native_simulation(params::Vector{Float64}, gen_seed::UInt64;
                               seed_global_rng::Bool = true)
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

    rng = Random.MersenneTwister(gen_seed)

    size_bins = generate_bimodal_bins(d_median_fine, sigma_g_fine,
                                      d_median_coarse, sigma_g_coarse;
                                      n_bins = 15)
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
    decay_params = [Transport.DecayParams(kdecay = Transport.NoDecay, halftime_hours = 0.0)]

    state = Transport.initialize_simulation(DOMAIN, sources, ["MixedFP"], decay_params;
                                            log_depositions = true)

    init_met = MET_CACHE[(CACHE_START_FILE, 1)]

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
    snap_size_bins = [Transport.ParticleProperties(diameter_μm = b.d, density_gcm3 = 2.5)
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
                                [activity], 0.0, icomp = 1)

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
        max_duration = 12.0 * 3600.0, save_snapshots = false,
        dt_particle = 300.0, use_reference_stepping = true,
        max_files = CACHE_END_FILE - CACHE_START_FILE + 1,
        omega_scale = omega_scale)

    # *** seed the global rng — required for bit-identity tests, optional
    # for the BIPOP driver where upstream rho_core also does not reseed. ***
    if seed_global_rng
        Random.seed!(gen_seed)
    end

    run_native_simulation!(state, ERA5_FILES;
        particle_size_config = particle_size_config,
        deposition_config    = deposition_config,
        hanna_config         = hanna_config,
        decay_params         = decay_params,
        config               = sim_config,
        advection_enabled    = true,
        settling_enabled     = true,
        dry_deposition_enabled = true,
        wet_deposition_enabled = false,
        numerical_config     = numerical_config,
        met_data_cache       = MET_CACHE,
        met_format           = MET_FORMAT,
        met_dimensions       = (NX, NY, NK),
        cache_init_file_idx  = CACHE_START_FILE,
        cache_init_time_idx  = 1)

    dep_log = state.deposition_log
    total_bq = sum(e.mass for e in dep_log; init = 0.0)

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
