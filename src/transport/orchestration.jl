# Orchestration - High-Level Simulation Runner
#
# Provides top-level functions to run complete simulations with minimal boilerplate.
# Integrates all transport components: particle dynamics, deposition, decay, concentration fields.

using NCDatasets
using Dates: value
using Printf
using OrdinaryDiffEq
using OrdinaryDiffEq: Euler  # Import Euler method for reference alignment
using StaticArrays
using NuclearDetonation.Transport:
    update_domain_vertical!,
    interpolate_vgrav,
    LandUseClass,
    GRASSLAND,
    DryDepositionParams,
    WetDepositionParams,
    compute_dry_deposition_velocity,
    compute_wet_scavenging_coefficient,
    apply_dry_deposition!,
    apply_wet_deposition!,
    wet_deposition_constant,
    wet_deposition_rate_bartnicki,
    bartnicki_wet_deposition_rate,
    WETDEP_PRECMIN,
    WETDEP_SIGMA_MIN,
    NumericalConfig,
    ERA5NumericalConfig,
    get_ode_solver,
    get_solve_kwargs,
    ERA5Format

"""
    TraceFrequency

Enumeration for trace output frequency options.
- `TRACE_EVERY_TIMESTEP`: Output at every integration timestep (default, legacy behavior)
- `TRACE_HOURLY`: Output only hourly snapshots (matches FLEXPART output)
- `TRACE_DISABLED`: No trace output (fastest, for production runs)
"""
@enum TraceFrequency begin
    TRACE_EVERY_TIMESTEP
    TRACE_HOURLY
    TRACE_DISABLED
end

"""
    Verbosity

Enumeration for output verbosity levels.
- `VERBOSITY_QUIET`: Minimal output (errors only)
- `VERBOSITY_NORMAL`: Standard progress messages
- `VERBOSITY_DEBUG`: Full diagnostic output including settling, wind alignment checks
"""
@enum Verbosity begin
    VERBOSITY_QUIET
    VERBOSITY_NORMAL
    VERBOSITY_DEBUG
end

"""
    OutputConfig

Configuration for simulation output and logging.

# Fields
- `trace_frequency::TraceFrequency`: How often to write particle traces (default: TRACE_EVERY_TIMESTEP)
- `verbosity::Verbosity`: Output verbosity level (default: VERBOSITY_NORMAL)
- `trace_enabled::Bool`: Whether to write trace file at all (default: true)
- `progress_interval_hours::Float64`: Hours between progress messages (default: 1.0)
- `settling_diagnostic_interval_hours::Float64`: Hours between settling diagnostics (default: 1.0, 0 = disabled)

# Examples
```julia
# Minimal output for production runs
output_config = OutputConfig(
    trace_frequency = TRACE_HOURLY,
    verbosity = VERBOSITY_QUIET,
    settling_diagnostic_interval_hours = 0.0
)

# Full diagnostic output for debugging
output_config = OutputConfig(
    trace_frequency = TRACE_EVERY_TIMESTEP,
    verbosity = VERBOSITY_DEBUG
)
```
"""
@kwdef struct OutputConfig
    trace_frequency::TraceFrequency = TRACE_EVERY_TIMESTEP
    verbosity::Verbosity = VERBOSITY_NORMAL
    trace_enabled::Bool = true
    progress_interval_hours::Float64 = 1.0
    settling_diagnostic_interval_hours::Float64 = 1.0  # 0 = disabled
end

# Helper functions for verbosity checks
is_quiet(config::OutputConfig) = config.verbosity == VERBOSITY_QUIET
is_verbose(config::OutputConfig) = config.verbosity >= VERBOSITY_NORMAL
is_debug(config::OutputConfig) = config.verbosity == VERBOSITY_DEBUG

# Helper to check if trace should be written at given time
function should_write_trace(output_config::OutputConfig, time_s::Real, dt::Real)
    !output_config.trace_enabled && return false
    output_config.trace_frequency == TRACE_DISABLED && return false
    output_config.trace_frequency == TRACE_EVERY_TIMESTEP && return true
    # TRACE_HOURLY: write exactly once per hour when we cross an hourly boundary
    # This matches FLEXPART output timing (T=1hr, 2hr, etc.)
    if output_config.trace_frequency == TRACE_HOURLY
        hour_s = 3600.0
        # Check if we just crossed an hour boundary: current hour > previous hour
        # This triggers exactly once when time advances past each hour mark
        current_hour = floor(Int, time_s / hour_s)
        previous_hour = floor(Int, (time_s - dt) / hour_s)
        return current_hour > previous_hour
    end
    return false
end

"""
    SimulationConfig{T<:Real}

Configuration options for simulation execution.

# Fields
- `dt_output::T`: Output/snapshot interval (seconds)
- `dt_met::T`: Meteorological data time resolution (seconds)
- `save_snapshots::Bool`: Whether to save concentration/deposition snapshots
- `verbose::Bool`: Print progress messages (DEPRECATED: use OutputConfig.verbosity)
- `max_files::Int`: Maximum number of met files to process (0 = all)
- `max_duration::T`: Maximum simulation duration (seconds, 0 = unlimited)
- `reltol::T`: Relative tolerance for ODE solver
- `abstol::T`: Absolute tolerance for ODE solver
- `output_config::OutputConfig`: Output and logging configuration
"""
@kwdef struct SimulationConfig{T<:Real}
    dt_output::T = 3600.0  # 1 hour
    saveat::Union{Nothing,Vector{T}} = nothing
    dt_met::T = 3600.0     # 1 hour (ERA5 hourly data - CRITICAL for tropical convection!)
    save_snapshots::Bool = true
    verbose::Bool = true   # DEPRECATED: use output_config.verbosity
    max_files::Int = 0     # 0 = process all files
    max_duration::T = 0.0  # 0 = unlimited (run all met data)
    reltol::T = 1e-4
    abstol::T = 1e-6
    dt_particle::T = 300.0  # Default particle timestep (s), matches reference TIME.STEP
    use_trilinear_gridding::Bool = false  # false = nearest-neighbor (reference implementation), true = trilinear (smoother)
    # Validation option: use two-stage Euler stepping
    # NOTE: Testing showed little difference vs ODE.jl for advection parity (Issue #1)
    # Scaling factor for vertical advection (omega field)
    omega_scale::T = 1.0
    use_reference_stepping::Bool = false
    # New output configuration (defaults to legacy behavior for backward compatibility)
    output_config::OutputConfig = OutputConfig()
end

# Helper to get effective verbosity from SimulationConfig
# Respects both legacy `verbose` flag and new OutputConfig
function get_verbosity(config::SimulationConfig)
    # If verbose=false was explicitly set, respect it (backward compatibility)
    if !config.verbose
        return VERBOSITY_QUIET
    end
    return config.output_config.verbosity
end

function should_print(config::SimulationConfig)
    get_verbosity(config) >= VERBOSITY_NORMAL
end

function should_print_debug(config::SimulationConfig)
    get_verbosity(config) == VERBOSITY_DEBUG
end

"""
    DepositionConfig{T<:Real}

Configuration for deposition physics.

# Fields
- `mixing_height::T`: Mixing layer height (m)
- `surface_roughness::T`: Surface roughness length z₀ (m)
- `friction_velocity::T`: Friction velocity u* (m/s)
- `monin_obukhov_length::T`: Monin-Obukhov length L (m)
- `season::SeasonCategory`: Season for vegetation parameters
- `apply_dry_deposition::Bool`: Enable dry deposition
- `apply_wet_deposition::Bool`: Enable wet deposition
- `roughness_length_map::Union{Nothing,Matrix{T}}`: Optional per-cell roughness override
- `friction_velocity_map::Union{Nothing,Matrix{T}}`: Optional per-cell friction velocity (m/s)
- `monin_obukhov_length_map::Union{Nothing,Matrix{T}}`: Optional per-cell Monin-Obukhov length (m)
- `land_use_map::Union{Nothing,Matrix{LandUseClass}}`: Optional per-cell land-use classes
"""
@kwdef struct DepositionConfig{T<:Real}
    mixing_height::T = 1000.0
    surface_roughness::T = 0.1
    friction_velocity::T = 0.3
    monin_obukhov_length::T = 1e10  # Neutral
    season::SeasonCategory = SUMMER
    apply_dry_deposition::Bool = true
    apply_wet_deposition::Bool = false
    use_simple_deposition::Bool = false  # Use simplified approach (DRY.DEPOSITION.NEW scheme)
    simple_deposition_velocity::T = 0.002  # NEW scheme uses 0.002 m/s for particles
    simple_surface_height::T = 30.0  # Reference uses 30m surface layer height in drydep2
    wet_deposition_precip_threshold::T = 0.01  # Min precip for wet depo (mm/hr). Set to 0.0 for FLEXPART-like continuous scavenging
    roughness_length_map::Union{Nothing,Matrix{T}} = nothing
    friction_velocity_map::Union{Nothing,Matrix{T}} = nothing
    monin_obukhov_length_map::Union{Nothing,Matrix{T}} = nothing
    land_use_map::Union{Nothing,Matrix{LandUseClass}} = nothing
end

"""
    TurbulentDiffusionConfig{T<:Real}

Configuration for turbulent diffusion (random walk).

# Fields
- `apply_diffusion::Bool`: Enable turbulent diffusion
- `tmix_v::T`: Vertical mixing time (seconds, default: 900s = 15 min)
- `tmix_h::T`: Horizontal mixing time (seconds, default: 900s = 15 min)
- `lmax::T`: Maximum l-eta in mixing layer (default: 0.28)
- `labove::T`: l-eta above mixing layer (default: 0.03)
- `entrainment::T`: Entrainment zone fraction (default: 0.1 = 10%)
- `hmax::T`: Maximum mixing height (m, default: 2500.0)
- `horizontal_a_bl::T`: Horizontal diffusion coefficient in BL (default: 0.5)
- `horizontal_a_above::T`: Horizontal diffusion coefficient above BL (default: 0.25)
- `horizontal_b::T`: Power law exponent for wind speed dependence (default: 0.875)
- `blfullmix::Bool`: Full mixing in boundary layer (default: false)

# Reference
Bartnicki (2011) parameterisation for turbulent diffusion
"""
@kwdef struct TurbulentDiffusionConfig{T<:Real}
    apply_diffusion::Bool = true
    tmix_v::T = 900.0           # 15 minutes vertical mixing time
    tmix_h::T = 900.0           # 15 minutes horizontal mixing time
    lmax::T = 0.28              # Maximum l-eta in mixing layer
    labove::T = 0.03            # l-eta above mixing layer
    entrainment::T = 0.1        # 10% entrainment zone
    hmax::T = 2500.0            # Maximum mixing height (m)
    horizontal_a_bl::T = 0.5    # Horizontal coeff in BL
    horizontal_a_above::T = 0.25 # Horizontal coeff above BL
    horizontal_b::T = 0.875     # Power law exponent
    blfullmix::Bool = false     # Gradual mixing (not instant)
end

"""
    ParticleSizeConfig

Configuration for particle size distribution and settling.

# Fields
- `size_bins::Vector{ParticleProperties}`: Particle size bins for vgrav tables
- `vgrav_tables::Union{Nothing,VGravTables}`: Pre-computed settling velocity tables
- `particle_radii::Vector{Float64}`: Individual particle radii (m)
- `particle_densities::Vector{Float64}`: Individual particle densities (kg/m³)
- `particle_size_indices::Vector{Int}`: Size bin index for each particle
"""
@kwdef mutable struct ParticleSizeConfig
    size_bins::Vector{ParticleProperties} = ParticleProperties[]
    vgrav_tables::Union{Nothing,VGravTables} = nothing
    particle_radii::Vector{Float64} = Float64[]
    particle_densities::Vector{Float64} = Float64[]
    particle_size_indices::Vector{Int} = Int[]
    fixed_gravity_cm_s::Union{Nothing,Vector{Float64}} = nothing
end

"""
    SimulationSnapshot{T<:Real}

Snapshot of simulation state at a particular time.

# Fields
- `time::T`: Simulation time (seconds)
- `particle_positions::Vector{SVector{3,T}}`: Active particle positions
- `concentration::Array{T,4}`: Atmospheric concentration field
- `deposition::Array{T,3}`: Cumulative deposition field
- `n_active::Int`: Number of active particles
"""
struct SimulationSnapshot{T<:Real}
    time::T
    particle_positions::Vector{SVector{3,T}}
    concentration::Array{T,4}
    dry_deposition::Array{T,3}
    wet_deposition::Array{T,3}
    total_deposition::Array{T,3}
    n_active::Int
end

"""
    integrate_timestep!(state::SimulationState, winds::WindFields,
                       dt::Real, particle_size_config::ParticleSizeConfig,
                       deposition_config::DepositionConfig, decay_params,
                       config::SimulationConfig)

Integrate one simulation timestep: advection, deposition, decay.

# Arguments
- `state`: Current simulation state (modified in-place)
- `winds`: Wind field interpolants for this time window
- `dt`: Time step duration (seconds)
- `particle_size_config`: Particle size/settling configuration
- `deposition_config`: Deposition physics configuration
- `decay_params`: Radioactive decay parameters
- `config`: Simulation configuration

# Returns
- `n_deposited::Int`: Number of particles deposited this timestep

# Performance
- Uses @views to eliminate array slice copies
"""
@views function integrate_timestep!(state::SimulationState{T},
                            winds::WindFields,
                            dt::T,
                            particle_size_config::ParticleSizeConfig,
                            deposition_config::DepositionConfig{T},
                            decay_params::Vector{DecayParams{T}},
                            config::SimulationConfig{T};
                            diffusion_config::TurbulentDiffusionConfig{T}=TurbulentDiffusionConfig{T}(),
                            hanna_config::Union{Nothing,HannaTurbulenceConfig{T}}=nothing,
                            advection_enabled::Bool=true,
                            settling_enabled::Bool=true,
                            dry_enabled::Bool=false,
                            wet_enabled::Bool=true,
                            current_time_global::T=T(0.0),
                            local_time_offset::T=T(0.0),
                            numerical_config::Union{NumericalConfig, ERA5NumericalConfig, Nothing}=nothing,
                            trace_filename::String="julia_particles_trace.csv",
                            is_era5::Bool=false,
                            trace_time_override::Union{Nothing, T}=nothing,
                            output_config::OutputConfig=OutputConfig()) where T<:Real

    n_deposited = 0

    # Grid scaling (assuming domain and met grids are aligned - adjust if needed)
    domain = state.domain
    nx_met, ny_met = winds.nx, winds.ny
    grid_scale_x = (nx_met - 1) / (domain.nx - 1)
    grid_scale_y = (ny_met - 1) / (domain.ny - 1)
    z_max_m = maximum(domain.hlevel)

    # CRITICAL: Detect if met data has reversed latitude (ERA5: N→S, GFS: S→N)
    lat_reversed = (winds.y_grid[end] < winds.y_grid[1])

    # Compute map ratios for horizontal advection (convert m/s to grid/s)
    # CRITICAL FIX (Issue #39): For ERA5 lat/lon grids, use equatorial grid spacing
    # using the mapfield approach:
    #   hlon = R_earth * dlon_rad (grid spacing at equator)
    #   rmx = xm / hlon where xm = 1/cos(lat) at particle position
    # The xm factor is applied later in particle_velocity for ERA5.
    #
    # Previous bug: domain.dx included cos(lat_mid) from mid-latitude conversion,
    # but then 1/cos(lat) was also applied at particle position, double-counting
    # the latitude correction and causing ~19% longitude drift.
    if is_era5
        # ERA5: Use equatorial grid spacing (mapfield approach)
        # Per-cell spacing in degrees = total span / (nx-1) intervals
        dlon_deg = (domain.lon_max - domain.lon_min) / (domain.nx - 1)
        dlat_deg = (domain.lat_max - domain.lat_min) / (domain.ny - 1)
        R_earth = T(6.371e6)  # metres
        # Grid spacing at equator (metres per grid cell)
        dx_equator = R_earth * dlon_deg * T(π) / T(180.0)
        dy_equator = R_earth * dlat_deg * T(π) / T(180.0)
        map_ratio_x = T(1.0) / dx_equator
        map_ratio_y = T(1.0) / dy_equator
    else
        # GFS: Use domain.dx/dy (proven in oct29 reference implementation)
        map_ratio_x = T(1.0) / domain.dx
        map_ratio_y = T(1.0) / domain.dy
    end

    σ_grid_min = 0.0
    σ_grid_max = 1.0

    dry_active = dry_enabled && deposition_config.apply_dry_deposition
    wet_active = wet_enabled && deposition_config.apply_wet_deposition

    cell_area = domain.cell_area
    nx_domain, ny_domain = domain.nx, domain.ny
    size_bins = particle_size_config.size_bins
    n_size_bins = length(size_bins)
    use_advanced_dry = dry_active && !deposition_config.use_simple_deposition && n_size_bins > 0
    use_wet_deposition = wet_active && n_size_bins > 0

    roughness_field = if isnothing(deposition_config.roughness_length_map)
        fill(T(deposition_config.surface_roughness), nx_domain, ny_domain)
    else
        Array{T}(deposition_config.roughness_length_map)
    end

    land_use_field = if isnothing(deposition_config.land_use_map)
        fill(GRASSLAND, nx_domain, ny_domain)
    else
        copy(deposition_config.land_use_map)
    end

    friction_field = if isnothing(deposition_config.friction_velocity_map)
        fill(T(deposition_config.friction_velocity), nx_domain, ny_domain)
    else
        Array{T}(deposition_config.friction_velocity_map)
    end

    monin_field = if isnothing(deposition_config.monin_obukhov_length_map)
        fill(T(deposition_config.monin_obukhov_length), nx_domain, ny_domain)
    else
        Array{T}(deposition_config.monin_obukhov_length_map)
    end

    # Use local_time_offset for field evaluation (correct time within met window)
    t_eval = local_time_offset
    sigma_surface = clamp(T(0.995), σ_grid_min, σ_grid_max)

    temperature_field = fill(T(0), nx_domain, ny_domain)
    pressure_field = fill(T(0), nx_domain, ny_domain)
    precip_field = fill(T(0), nx_domain, ny_domain)

    for j in 1:ny_domain
        # Use same lat reversal logic as particle transforms
        if lat_reversed
            y_met = ny_met - ((j - 1.0) * grid_scale_y)  # ERA5
        else
            y_met = (j - 1.0) * grid_scale_y + 1.0  # GFS
        end
        for i in 1:nx_domain
            x_met = (i - 1.0) * grid_scale_x + 1.0
            temperature_field[i, j] = winds.t_interp(x_met, y_met, sigma_surface, t_eval)
            pressure_field[i, j] = winds.ps_interp(x_met, y_met, t_eval) * T(100.0)
            precip_field[i, j] = winds.precip_interp(x_met, y_met, t_eval)
        end
    end

    dry_velocity_fields = use_advanced_dry ? Vector{Matrix{T}}(undef, n_size_bins) : nothing
    wet_lambda_fields = use_wet_deposition ? Vector{Matrix{T}}(undef, n_size_bins) : nothing

    if use_advanced_dry || use_wet_deposition
        for (idx, props) in enumerate(size_bins)
            diameter_m = T(props.diameter_μm) * T(1e-6)
            density_kg_m3 = T(props.density_gcm3) * T(1000.0)
            settling_velocity = 0.0
            if !isnothing(particle_size_config.fixed_gravity_cm_s) &&
               idx <= length(particle_size_config.fixed_gravity_cm_s)
                settling_velocity = T(particle_size_config.fixed_gravity_cm_s[idx] * 0.01)
            end

            if use_advanced_dry
                dry_params = DryDepositionParams(
                    diameter_m,
                    density_kg_m3,
                    T(deposition_config.simple_surface_height),
                    roughness_field,
                    land_use_field
                )
                dry_velocity_fields[idx] = compute_dry_deposition_velocity(
                    dry_params,
                    friction_field,
                    monin_field,
                    temperature_field,
                    pressure_field,
                    deposition_config.season,
                    T(settling_velocity)
                )
            end

            if use_wet_deposition
                # Use Bartnicki (2003) scheme
                radius_μm = T(props.diameter_μm) / T(2.0)  # diameter → radius
                depconst = wet_deposition_constant(radius_μm)
                lambda_field = similar(precip_field)
                precip_threshold = T(deposition_config.wet_deposition_precip_threshold)
                for j in 1:ny_domain, i in 1:nx_domain
                    precip_mmh = precip_field[i, j]
                    if precip_mmh > precip_threshold
                        # Bartnicki scavenging rate (s⁻¹)
                        lambda_field[i, j] = wet_deposition_rate_bartnicki(
                            radius_μm, T(precip_mmh), depconst; use_convective=true
                        )
                    else
                        lambda_field[i, j] = zero(T)
                    end
                end
                wet_lambda_fields[idx] = lambda_field
            end
        end
    end

    # Integrate each active particle
    for (i, particle) in enumerate(state.ensemble.particles)
        if !is_active(particle)
            continue
        end

        # Get current position in domain coordinates
        pos_domain = state.ensemble.positions[i]

        # Convert to meteorology grid coordinates
        x_era5 = (pos_domain[1] - 1.0) * grid_scale_x + 1.0
        # Use lat_reversed (computed at function start) for coordinate transform
        if lat_reversed
            y_era5 = ny_met - (pos_domain[2] - 1.0) * grid_scale_y  # Reverse for ERA5
        else
            y_era5 = (pos_domain[2] - 1.0) * grid_scale_y + 1.0  # No reversal for GFS
        end
        # Stored position uses sigma for vertical coordinate (see simulation.jl).
        stored_sigma = clamp(Float64(pos_domain[3]), σ_grid_min, σ_grid_max)
        prev_sigma = isfinite(particle.z) ? clamp(Float64(particle.z), σ_grid_min, σ_grid_max) : nothing
        profile_start = hybrid_profile(winds, x_era5, y_era5, local_time_offset)

        # DEBUG: Disabled for performance
        # if i == 1
        #     println("DEBUG: Height Profile for Particle 1")
        # end

        # Convert stored sigma to physical height for the current meteorological slice
        z_height = height_from_sigma(profile_start, stored_sigma; fallback_height=nothing)
        if !isfinite(z_height)
            z_height = stored_sigma * z_max_m
        end

        # DEBUG: Disabled for performance (orchestration verbose logging)

        prev_height = if prev_sigma === nothing
            z_height
        else
            height_from_sigma(profile_start, prev_sigma; fallback_height=z_height)
        end

        fallback_sigma = prev_sigma === nothing ? stored_sigma : prev_sigma
        z_sigma = sigma_from_height(profile_start, z_height; fallback_sigma=fallback_sigma)
        if !isfinite(z_sigma)
            # CRITICAL: Don't fall back to σ_grid_max (surface)! Particles at high altitude
            # should NOT be forced to the surface. Compute sigma from height directly.
            @warn "sigma_from_height returned non-finite value" idx=i z_height=z_height prev_sigma=prev_sigma
            # Use a simple linear approximation as last resort
            z_sigma = clamp(1.0 - (z_height / z_max_m), σ_grid_min, σ_grid_max)
        end
        z_sigma = clamp(z_sigma, σ_grid_min, σ_grid_max)
        z_height_resolved = height_from_sigma(profile_start, z_sigma; fallback_height=z_height)
        prev_height = isfinite(z_height_resolved) ? z_height_resolved : z_height
        prev_sigma = z_sigma

        pos_era5 = [x_era5, y_era5, z_sigma]

        # Update boundary layer diagnostics from meteorology (matches reference bldp)
        blk_time = local_time_offset
        tbl_sigma = winds.tbl_interp(pos_era5[1], pos_era5[2], blk_time)
        if !isfinite(tbl_sigma)
            tbl_sigma = 1.0 - (diffusion_config.hmax / z_max_m)
        end
        tbl_sigma = clamp(tbl_sigma, 0.0, 1.0)
        particle.tbl = Float32(tbl_sigma)

        hbl_meter = winds.hbl_interp(pos_era5[1], pos_era5[2], blk_time)
        if !isfinite(hbl_meter)
            hbl_meter = diffusion_config.hmax
        end
        particle.hbl = Float32(max(hbl_meter, 0.0))

        # =============================================================================
        # STEP 1: DRY DEPOSITION (at CURRENT position, before movement)
        # Match reference order: drydep → advection → random walk
        # =============================================================================
        surface_height_sigma = 0.996  # Surface/constant flux layer at ~30m height

        # DEBUG: Disabled for performance (SURFACE_LAYER logging)

            if dry_active && z_sigma > surface_height_sigma
                ii = round(Int, clamp(pos_domain[1], 1, domain.nx))
                jj = round(Int, clamp(pos_domain[2], 1, domain.ny))

                # Use simplified deposition approach (DRY.DEPOSITION.NEW scheme) when configured
                if deposition_config.use_simple_deposition
                    # Use simplified approach: vd = simple_deposition_velocity + settling_velocity
                    vg = Float64(particle.grv)
                size_idx_particle = particle_size_config.particle_size_indices[i]
                if abs(vg) < eps() && !isnothing(particle_size_config.fixed_gravity_cm_s)
                    if 1 <= size_idx_particle <= length(particle_size_config.fixed_gravity_cm_s)
                        vg = particle_size_config.fixed_gravity_cm_s[size_idx_particle] * 0.01  # cm/s → m/s
                    end
                elseif abs(vg) < eps() && !isnothing(particle_size_config.vgrav_tables)
                    # Get actual pressure and temperature at particle position (near surface)
                    # Use 3D pressure field for consistency (though near-surface P ≈ surface P)
                    P_hpa = winds.p_interp(x_era5, y_era5, z_sigma, local_time_offset)
                    T_k = winds.t_interp(x_era5, y_era5, z_sigma, local_time_offset)
                    vg = interpolate_vgrav(particle_size_config.vgrav_tables,
                                          size_idx_particle,
                                          P_hpa, T_k)
                end

                # DRY.DEPOSITION.NEW scheme: vd_particles = 0.002 m/s + settling_velocity
                vd_simple = deposition_config.simple_deposition_velocity + vg
                h_surface = deposition_config.simple_surface_height  # Reference uses 30m in drydep2

                # Deposition rate coefficient
                k_dep = vd_simple / h_surface

                # Exponential decay: M(t+dt) = M(t) * exp(-k * dt)
                decay_factor = exp(-k_dep * dt)

                # Apply deposition exponentially (M_new = M_old * exp(-k*dt))
                for comp in 1:state.ensemble.ncomponents
                    mass = get_rad(particle, comp)
                    if mass > 0
                        new_mass = mass * decay_factor
                        deposited = mass - new_mass
                        set_rad!(particle, comp, Float32(new_mass))

                        # Add to deposition fields (convert Bq to Bq/m²)
                        cell_area_val = cell_area[ii, jj]
                        increment = deposited / cell_area_val
                        state.fields.dry_deposition[ii, jj, comp] += increment
                        state.fields.total_deposition[ii, jj, comp] += increment
                        state.total_deposited[comp] += deposited

                        # Log deposition event with actual particle position
                        if state.log_depositions
                            push!(state.deposition_log, DepositionEvent(
                                Float64(pos_domain[1]), Float64(pos_domain[2]), Float64(deposited),
                                Float64(current_time_global), comp))
                        end
                    end
                end

                # Complete deposition for large particles at ground level
                # Reference: if (part%z == vlevel(1)) deprate = 1.0
                # This applies to particles with radius >= 10 um when at the surface
                if z_sigma >= 0.999  # At ground level (vlevel(1) in reference)
                    size_idx_particle = particle_size_config.particle_size_indices[i]
                    if size_idx_particle > 0 && size_idx_particle <= length(particle_size_config.size_bins)
                        diameter_μm = particle_size_config.size_bins[size_idx_particle].diameter_μm
                        if diameter_μm >= 20.0  # radius >= 10 μm
                            # Complete deposition (100%)
                            for comp in 1:state.ensemble.ncomponents
                                mass = get_rad(particle, comp)
                                if mass > 0
                                    increment = mass / cell_area[ii, jj]
                                    state.fields.dry_deposition[ii, jj, comp] += increment
                                    state.fields.total_deposition[ii, jj, comp] += increment
                                    state.total_deposited[comp] += mass
                                    # Log deposition event
                                    if state.log_depositions
                                        push!(state.deposition_log, DepositionEvent(
                                            Float64(pos_domain[1]), Float64(pos_domain[2]), Float64(mass),
                                            Float64(current_time_global), comp))
                                    end
                                end
                                set_rad!(particle, comp, -1.0f0)
                            end
                            n_deposited += 1
                            continue
                        end
                    end
                end

                # Check if particle should be removed (mass below threshold)
                if sum(particle.rad) < 1e-10
                    for comp in 1:state.ensemble.ncomponents
                        set_rad!(particle, comp, -1.0f0)
                    end
                    n_deposited += 1
                    continue  # Skip to next particle
                end
            elseif use_advanced_dry
                size_idx_particle = particle_size_config.particle_size_indices[i]
                if dry_velocity_fields !== nothing &&
                   1 <= size_idx_particle <= length(dry_velocity_fields)
                    vd = dry_velocity_fields[size_idx_particle][ii, jj]
                    if vd > eps(T)
                        previous_mass = copy(particle.rad)
                        mass_type = eltype(particle.rad)
                        deposited_mass = apply_dry_deposition!(
                            particle.rad,
                            mass_type(vd),
                            mass_type(dt),
                            mass_type(deposition_config.mixing_height)
                        )
                        if deposited_mass > 0
                            area_local = cell_area[ii, jj]
                            for comp in 1:state.ensemble.ncomponents
                                mass_loss = Float64(previous_mass[comp] - particle.rad[comp])
                                if mass_loss > 0
                                    increment = mass_loss / area_local
                                    state.fields.dry_deposition[ii, jj, comp] += increment
                                    state.fields.total_deposition[ii, jj, comp] += increment
                                    state.total_deposited[comp] += mass_loss
                                    # Log deposition event
                                    if state.log_depositions
                                        push!(state.deposition_log, DepositionEvent(
                                            Float64(pos_domain[1]), Float64(pos_domain[2]), Float64(mass_loss),
                                            Float64(current_time_global), comp))
                                    end
                                end
                            end
                        end
                        if sum(particle.rad) < 1e-10
                            for comp in 1:state.ensemble.ncomponents
                                set_rad!(particle, comp, -1.0f0)
                            end
                            n_deposited += 1
                            continue
                        end
                    end
                end
            end
        end

        # BARTNICKI WET DEPOSITION - Bartnicki (2003) scavenging scheme
        # Conditions: particle above surface layer AND below ~550 hPa (sigma > 0.67)
        # Reference: if (kwetdep == 1 .AND. prc > precmin .AND. part%z > 0.67) then
        if use_wet_deposition && z_sigma > surface_height_sigma && z_sigma > WETDEP_SIGMA_MIN
            ii = round(Int, clamp(pos_domain[1], 1, domain.nx))
            jj = round(Int, clamp(pos_domain[2], 1, domain.ny))
            area_local = cell_area[ii, jj]
            size_idx_particle = particle_size_config.particle_size_indices[i]
            if wet_lambda_fields !== nothing &&
               1 <= size_idx_particle <= length(wet_lambda_fields)
                lambda = wet_lambda_fields[size_idx_particle][ii, jj]
                if lambda > eps(T)
                    previous_mass = copy(particle.rad)
                    mass_type = eltype(particle.rad)
                    deposited_mass = apply_wet_deposition!(
                        particle.rad,
                        mass_type(lambda),
                        mass_type(dt)
                    )
                    if deposited_mass > 0
                        for comp in 1:state.ensemble.ncomponents
                            mass_loss = Float64(previous_mass[comp] - particle.rad[comp])
                            if mass_loss > 0
                                increment = mass_loss / area_local
                                state.fields.wet_deposition[ii, jj, comp] += increment
                                state.fields.total_deposition[ii, jj, comp] += increment
                                state.total_deposited[comp] += mass_loss
                                # Log deposition event
                                if state.log_depositions
                                    push!(state.deposition_log, DepositionEvent(
                                        Float64(pos_domain[1]), Float64(pos_domain[2]), Float64(mass_loss),
                                        Float64(current_time_global), comp))
                                end
                            end
                        end
                    end
                    if sum(particle.rad) < 1e-10
                        for comp in 1:state.ensemble.ncomponents
                            set_rad!(particle, comp, -1.0f0)
                        end
                        n_deposited += 1
                        continue
                    end
                end
            end
        end

        # Create particle velocity function with settling
        function particle_velocity_settling!(du, u, p, t)
            x, y, z = u

            # Interpolate wind components
            u_wind = p.advection_enabled ? p.winds.u_interp(x, y, z, t) : 0.0
            v_wind = p.advection_enabled ? p.winds.v_interp(x, y, z, t) : 0.0
            w_wind = p.advection_enabled ? p.winds.w_interp(x, y, z, t) * p.config.omega_scale : 0.0

            # DIAGNOSTIC: Check for NaNs
            if !isfinite(u_wind) || !isfinite(v_wind) || !isfinite(w_wind)
                @warn "Non-finite wind value detected" x=x y=y z=z t=t u=u_wind v=v_wind w=w_wind
            end

            # DEBUG: Disabled for performance (W_WIND and GRID PARAMETER diagnostics)

            # NOTE: Do NOT zero out meteorological w_wind in surface layer!
            # Reference only zeros gravitational settling (wg), not meteorological wind.
            # It only zeros wg when dry deposition is active.

            profile_local = hybrid_profile(p.winds, x, y, t)

            # Add gravitational settling if configured AND enabled
            vg_sigma = 0.0
            if p.settling_enabled && !isempty(p.size_indices)
                size_idx_local = p.size_indices[p.particle_idx]
                vg_ms = 0.0
                if !isnothing(p.fixed_gravity_cm_s)
                    if 1 <= size_idx_local <= length(p.fixed_gravity_cm_s)
                        vg_ms = p.fixed_gravity_cm_s[size_idx_local] * 0.01  # cm/s → m/s
                    end
                elseif !isnothing(p.vgrav_tables)
                    # CRITICAL FIX: Get actual pressure at particle altitude, not surface pressure!
                    # At 20 km: P ≈ 50 hPa (not 1000 hPa!) - this makes a HUGE difference!
                    # Pressure at particle altitude (hPa) from 3D pressure field
                    P_hpa = p.winds.p_interp(x, y, z, t)
                    # Temperature at particle altitude (K)
                    T_k = p.winds.t_interp(x, y, z, t)
                    # Interpolate settling velocity using actual altitude conditions
                    vg_ms = interpolate_vgrav(p.vgrav_tables, size_idx_local, P_hpa, T_k)

                    # DEBUG: Disabled for performance (SETTLING logging)
                end

                # CRITICAL: Match reference behaviour — disable settling velocity in the surface
                # layer ONLY if dry deposition is enabled.
                # Reference: if (def_comp(part%icomp)%kdrydep == 1 .and. part%z > surface_height_sigma) then wg = 0.0
                # When dry deposition is disabled, allow settling all the way to ground to prevent
                # turbulence from bouncing particles back up.
                if p.dry_enabled && z > 0.996
                    vg_sigma = 0.0  # Dry deposition will handle removal in surface layer
                else
                    # Convert settling velocity (m/s) to sigma tendency using local layer thickness
                    # Oct29 proven method: Direct height interpolation (works for GFS, simpler than hypsometric)
                    z_grid = p.winds.z_grid
                    # Ensure within bounds (avoid extrapolation artifacts)
                    z_clamped = clamp(z, z_grid[1] + eps(T), z_grid[end] - eps(T))
                    idx = searchsortedlast(z_grid, z_clamped)
                    idx = clamp(idx, 1, length(z_grid) - 1)
                    sigma_upper = z_grid[idx]
                    sigma_lower = z_grid[idx + 1]

                    # Retrieve local layer heights (m) using pre-computed height field
                    h_upper = height_from_sigma(profile_local, sigma_upper; fallback_height=p.previous_height)
                    h_lower = height_from_sigma(profile_local, sigma_lower; fallback_height=h_upper)

                    dsigma = sigma_lower - sigma_upper
                    dz = h_upper - h_lower

                    if !isfinite(h_upper) || !isfinite(h_lower) || abs(dz) < eps(T)
                        vg_sigma = vg_ms / p.z_max
                    else
                        vg_sigma = vg_ms * dsigma / dz
                    end

                    if p.particle_idx == 1 && abs(t - 300.0) < 1.0
                        height_m = height_from_sigma(profile_local, z; fallback_height=p.previous_height)
                        # println("\n" * "="^70)
                        # println("SETTLING DIAGNOSTIC - Particle 1 at t=$(round(t))s")
                        # println("="^70)
                        # println("Position: x=$x, y=$y, z=$z (sigma)")
                        # println("Height: $(round(height_m, digits=2)) m")
                        # println("  sigma_upper=$sigma_upper, sigma_lower=$sigma_lower, dsigma=$(round(dsigma,digits=6))")
                        # println("  h_upper=$(round(h_upper,digits=2))m, h_lower=$(round(h_lower,digits=2))m, dz=$(round(dz,digits=3))m")
                        # println("  vg_ms=$(round(vg_ms,digits=6)) m/s → vg_sigma=$(round(vg_sigma,digits=12)) σ/s")
                        # println("  Meteorological w=$(round(w_wind, digits=12)) σ/s (scaled by $(p.config.omega_scale)), total=$(round(w_wind + vg_sigma, digits=12)) σ/s")
                        # println("="^70 * "\n")
                    end
                end
            end

            # TRACE: DISABLED - moved to after ensemble.positions update to fix 300km coordinate bug
            # Writing trace here captures intermediate ODE state, not final position!
            # See line ~1010 for new trace writing location.
            if false  # Disabled: p.particle_idx % 200 == 1 || p.particle_idx == 1
                w_total = w_wind + vg_sigma
                in_surface = z > 0.996

                # Compute diagnostic variables to match reference trace
                # Get layer thickness (dz), temperature (theta), and vertical grid info
                T_k_trace = p.winds.t_interp(x, y, z, t)
                z_grid = p.winds.z_grid
                z_clamped = clamp(z, z_grid[1] + eps(T), z_grid[end] - eps(T))
                idx_trace = searchsortedlast(z_grid, z_clamped)
                idx_trace = clamp(idx_trace, 1, length(z_grid) - 1)
                sigma_upper_trace = z_grid[idx_trace]
                sigma_lower_trace = z_grid[idx_trace + 1]
                h_upper_trace = height_from_sigma(profile_local, sigma_upper_trace; fallback_height=p.previous_height)
                h_lower_trace = height_from_sigma(profile_local, sigma_lower_trace; fallback_height=h_upper_trace)
                dz_trace = h_upper_trace - h_lower_trace
                deta_trace = sigma_lower_trace - sigma_upper_trace

                global_time = p.current_time_global + t
                # Only write trace if within simulation duration (respect max_duration)
                if p.config.max_duration == 0.0 || global_time <= p.config.max_duration
                    open("julia_particles_trace.csv", "a") do io
                        # NOTE: This code block is now DISABLED - trace moved to after ensemble.positions update
                        # to fix 300km coordinate bug (see line ~1010).
                        # Keeping this for reference but it's inside "if false" block above.
                        # Convert domain coords to lat/lon
                        lon = p.domain.lon_min + (x_domain - 1) * (p.domain.lon_max - p.domain.lon_min) / (p.domain.nx - 1)
                        lat = p.domain.lat_min + (y_domain - 1) * (p.domain.lat_max - p.domain.lat_min) / (p.domain.ny - 1)
                        # Compute altitude from sigma using hybrid coordinate profile
                        altitude_m = height_from_sigma(profile_local, z; fallback_height=p.previous_height)
                        println(io, "$(p.particle_idx),$(global_time),$(lat),$(lon),$(z),$(altitude_m),$(w_wind),$(vg_sigma),$(w_total),$(in_surface),$(dz_trace),$(T_k_trace),$(deta_trace),$(idx_trace),$(idx_trace+1)")
                    end
                end
            end

            # CRITICAL: Apply latitude-dependent map scale factor xm for longitude advection
            # mapfield: xm(i,j) = 1/cos(lat)
            # posint: rmx = xm/dxgrid
            # advection: x = x + u*dt*rmx
            #
            # For geographic grids, 1° of longitude shrinks toward poles:
            # physical_distance_x = grid_distance_x * cos(lat)
            # So to convert m/s to grid/s: dx/dt = u * xm / hx where xm = 1/cos(lat)
            #
            # NOTE: This xm factor is ONLY needed for ERA5 lat/lon grids.
            # GFS already has correct map ratios pre-computed - applying xm breaks GFS!

            if p.is_era5
                # ERA5: Apply latitude-dependent xm factor
                # Convert y-position to latitude (y is in met grid coordinates, typically 1 to ny)
                ny_met = length(p.winds.y_grid)
                lat_frac = (y - 1.0) / max(ny_met - 1, 1)  # 0 to 1
                lat_deg = p.domain.lat_min + lat_frac * (p.domain.lat_max - p.domain.lat_min)
                lat_rad = lat_deg * π / 180.0
                clat = cos(lat_rad)
                clat = max(clat, 0.01745)  # ~cos(89°), avoid division by zero near poles
                xm_factor = 1.0 / clat
                du[1] = u_wind * p.map_ratio_x * xm_factor
            else
                # GFS: Use oct29 proven method - direct map ratios without xm correction
                du[1] = u_wind * p.map_ratio_x
            end
            du[2] = v_wind * p.map_ratio_y
            # Sigma tendency: w is dσ/dt (positive increases σ toward surface)
            du[3] = w_wind + vg_sigma

            # DEBUG: Disabled for performance (VELOCITY SCALING diagnostic)
        end

        # Setup parameters
        settling_params = (
            winds = winds,
            vgrav_tables = particle_size_config.vgrav_tables,
            size_indices = particle_size_config.particle_size_indices,
            fixed_gravity_cm_s = particle_size_config.fixed_gravity_cm_s,
            particle_idx = i,
            z_max = z_max_m,
            map_ratio_x = map_ratio_x,
            map_ratio_y = map_ratio_y,
            previous_height = prev_height,
            previous_sigma = prev_sigma,
            advection_enabled = advection_enabled,
            settling_enabled = settling_enabled,
            dry_enabled = dry_enabled,
            current_time_global = current_time_global,
            domain = domain,
            config = config,
            numerical_config = numerical_config, # Added for interpolation access
            is_era5 = is_era5  # Pass format flag for conditional xm scaling
        )

        # Integrate trajectory
        # Option 1: Two-stage Euler stepping (validation)
        #           NOTE: Although the reference default is simple Euler, we use Heun for better accuracy.
        #           Testing showed Heun gives slightly better agreement than simple Euler.
        # Option 2: OrdinaryDiffEq.jl solver (default/production)
        if config.use_reference_stepping
            # Two-stage (Heun/trapezoidal) stepping
            # Session 7 finding: Pure Euler gives marginal lon improvement
            # but worse lat/alt. Heun is slightly better overall, so keep it.
            # The ~0.04 deg lat / 0.08 deg lon error is primarily due to chaotic trajectory divergence
            # from small initial differences, not the time-stepping scheme.
            t1_local = local_time_offset
            du1 = zeros(T, 3)
            particle_velocity_settling!(du1, pos_era5, settling_params, t1_local)
            pred = SVector{3,T}(pos_era5[1] + du1[1]*dt,
                                 pos_era5[2] + du1[2]*dt,
                                 pos_era5[3] + du1[3]*dt)
            du2 = zeros(T, 3)
            particle_velocity_settling!(du2, pred, settling_params, t1_local + dt)
            pos_final = SVector{3,T}(pos_era5[1] + (du1[1] + du2[1]) * (dt*T(0.5)),
                                     pos_era5[2] + (du1[2] + du2[2]) * (dt*T(0.5)),
                                     pos_era5[3] + (du1[3] + du2[3]) * (dt*T(0.5)))

            # Convert back to domain coordinates
            x_domain_final = (pos_final[1] - 1.0) / grid_scale_x + 1.0
            if lat_reversed
                y_domain_final = (ny_met - pos_final[2]) / grid_scale_y + 1.0
            else
                y_domain_final = (pos_final[2] - 1.0) / grid_scale_y + 1.0
            end
            t_eval = t1_local + dt
            profile_eval = hybrid_profile(winds, pos_final[1], pos_final[2], T(t_eval))
        else
            # OrdinaryDiffEq path
            tspan = (local_time_offset, local_time_offset + dt)
            prob = ODEProblem(particle_velocity_settling!, pos_era5, tspan, settling_params)

            if !isnothing(numerical_config)
                solver = get_ode_solver(numerical_config)
                solve_kwargs = get_solve_kwargs(numerical_config)
                sol = solve(prob, solver; solve_kwargs...)
            else
                sol = solve(prob, Euler(), dt=config.dt_particle, adaptive=false)
            end

            if sol.retcode != OrdinaryDiffEq.ReturnCode.Success
                @warn "ODE solver failed" retcode=sol.retcode particle=i t=sol.t[end] u=sol.u[end]
                # Mark particle as inactive
                for comp in 1:state.ensemble.ncomponents
                    set_rad!(particle, comp, -1.0f0)  # Negative = inactive
                end
                n_deposited += 1
                continue
            end

            # Update position (convert back to domain coordinates)
            pos_final = sol.u[end]
            x_domain_final = (pos_final[1] - 1.0) / grid_scale_x + 1.0
            if lat_reversed
                y_domain_final = (ny_met - pos_final[2]) / grid_scale_y + 1.0
            else
                y_domain_final = (pos_final[2] - 1.0) / grid_scale_y + 1.0
            end
            t_eval = sol.t[end]
            profile_eval = hybrid_profile(winds, pos_final[1], pos_final[2], t_eval)
        end
        # Track final sigma coordinate (will be updated by turbulent diffusion)
        z_sigma_deposition = clamp(pos_final[3], σ_grid_min, σ_grid_max)
        z_height_final = height_from_sigma(profile_eval, z_sigma_deposition; fallback_height=prev_height)

        # If particle reaches the surface (numeric undershoot), deposit only if dry deposition is enabled.
        # Otherwise, keep particle active and let clamping/reflection handle it.
        if dry_enabled && z_height_final < 0.0
            ii = round(Int, clamp(x_domain_final, 1, domain.nx))
            jj = round(Int, clamp(y_domain_final, 1, domain.ny))
            area_local = cell_area[ii, jj]

            for comp in 1:state.ensemble.ncomponents
                mass = get_rad(particle, comp)
                if mass > 0
                    increment = mass / area_local
                    state.fields.dry_deposition[ii, jj, comp] += increment
                    state.fields.total_deposition[ii, jj, comp] += increment
                    state.total_deposited[comp] += mass
                    # Log deposition event
                    if state.log_depositions
                        push!(state.deposition_log, DepositionEvent(
                            Float64(x_domain_final), Float64(y_domain_final), Float64(mass),
                            Float64(t), comp))
                    end
                end
                set_rad!(particle, comp, -1.0f0)
            end

            n_deposited += 1
            continue
        end

        prev_height = isfinite(z_height_final) ? z_height_final : z_height
        prev_sigma = z_sigma_deposition

        # Apply turbulent diffusion
        # Choice between simple random walk or Hanna (1982) with O-U process and CBL
        if !isnothing(hanna_config) && hanna_config.apply_turbulence
            # ===== HANNA (1982) TURBULENCE SCHEME =====
            # Height-dependent, stability-aware, with Ornstein-Uhlenbeck process
            # and CBL scheme for strong convection

            # Convert z_sigma to height (m)
            z_sigma = pos_final[3]
            z_m = height_from_sigma(profile_eval, z_sigma; fallback_height=prev_height)

            # Get meteorological parameters
            # NOTE: These are static values. Dynamic interpolation was tested but made things worse:
            # ===== DYNAMIC BL PARAMETERS (IMPROVED) =====
            # Use dynamic met data for h and ust instead of static defaults.
            # This is critical for height-coordinate turbulence to work correctly.
            
            # Calculate met grid coordinates for interpolation from pos_final
            x_met_curr = T(1.0) + (pos_final[1] - T(1.0)) * grid_scale_x
            y_met_curr = if lat_reversed
                T(ny_met) - (pos_final[2] - T(1.0)) * grid_scale_y
            else
                T(1.0) + (pos_final[2] - T(1.0)) * grid_scale_y
            end

            # Dynamic Mixing Height (h)
            # Use hbl_interp (calculated via Richardson number in met_reader)
            h_dynamic = winds.hbl_interp(x_met_curr, y_met_curr, local_time_offset)
            h = h_dynamic > 0 ? h_dynamic : deposition_config.mixing_height
            h = max(h, T(50.0)) # Safety floor

            # Dynamic Friction Velocity (ust)
            # Estimate from surface/lowest-level wind speed using drag coefficient approximation
            # Evaluate u,v at sigma=1.0 (surface/lowest model level)
            u_surf = winds.u_interp(x_met_curr, y_met_curr, T(1.0), local_time_offset)
            v_surf = winds.v_interp(x_met_curr, y_met_curr, T(1.0), local_time_offset)
            u_mag = sqrt(u_surf^2 + v_surf^2)

            # Estimate ust using configurable drag coefficient
            # Default 0.05 corresponds to Cd ≈ 0.0025; can tune down to reduce mixing
            drag_coeff = hanna_config.drag_coefficient
            ust_dynamic = max(drag_coeff * u_mag, T(0.01))
            ust = ust_dynamic

            # ===== DYNAMIC OBUKHOV LENGTH (L) ESTIMATION =====
            # When enabled, estimate stability from temperature gradient
            # Otherwise use static neutral value from deposition_config
            L = if hanna_config.use_dynamic_L
                # Determine if manual trigger should be active (Daytime: 06:00 to 18:00)
                # Use current_time_global (seconds since simulation start)
                # Since start is 00:00, (time/3600) % 24 gives hour of day.
                hr_day = (current_time_global / 3600.0) % 24
                is_daytime = hr_day >= 6.0 && hr_day <= 18.0

                if hanna_config.convective_trigger && is_daytime
                    # Manual trigger: force unstable conditions during day
                    T(-100.0)
                else
                    # Get surface and lowest-level temperatures for stability estimate
                    # T(1.0) is lowest model level (~10m in ERA5 L137)
                    T_surf = winds.t_interp(x_met_curr, y_met_curr, T(1.0), local_time_offset)
                    # Use a very near-surface level for gradient
                    T_lowest = winds.t_interp(x_met_curr, y_met_curr, T(0.999), local_time_offset)
                    
                    # Force everything to Float64 for stable dispatch
                    let ts = Float64(T_surf), t2 = Float64(T_lowest), us = Float64(ust)
                        T(estimate_obukhov_length(ts, t2, us, t2))
                    end
                end
            else
                T(deposition_config.monin_obukhov_length)
            end

            # For convective conditions, apply minimum PBL height
            # ERA5 Richardson number systematically underestimates tropical CBL depth
            # (e.g., gives ~200m when reality is 1000-2000m during daytime convection)
            if L < T(0.0)
                h = max(h, hanna_config.convective_h_min)
            end

            # Compute convective velocity scale w* from friction velocity
            # For unstable conditions: w* = u* * (-h / (k*L))^(1/3)
            # Standard theoretical relationship (Hanna 1982, Stohl et al. 2005)
            # Use k = 0.4 (von Karman constant)
            wst_base = L < 0.0 ? ust * cbrt(max(-h / (T(0.4) * L), T(0.0))) : T(0.0)
            wst = wst_base * hanna_config.wst_scale

            # Compute Hanna turbulence parameters
            hanna_params = compute_hanna_parameters(z_m, h, L, ust, wst, hanna_config)

            # Generate random numbers
            # Check if particle is actually AT ground (sigma = 1.0)
            # NOTE: The old threshold of 0.996 was killing vertical mixing for any particle
            # below ~80m! FLEXPART doesn't do this - it uses reflections at ground/BL top.
            # Only disable vertical turbulence for particles essentially at the surface.
            at_ground = pos_final[3] >= 0.9999  # Within ~2m of surface

            # Initialize z_sigma_current for the sub-timestepping loop
            z_sigma_current = pos_final[3]

            # Always apply horizontal turbulence - particles spread horizontally even at ground level
            rnd_u = randn()
            rnd_v = randn()
            particle.u_turb = Float32(ornstein_uhlenbeck_step(
                T(particle.u_turb), hanna_params.sigu, hanna_params.tlu, dt, rnd_u))
            particle.v_turb = Float32(ornstein_uhlenbeck_step(
                T(particle.v_turb), hanna_params.sigv, hanna_params.tlv, dt, rnd_v))

            if at_ground
                # Particle in surface layer - only zero vertical turbulence, keep horizontal
                particle.w_turb = 0.0f0
            else
                # Particle in atmosphere - apply full vertical turbulence
                # (horizontal already applied above)

                # ===== SIMPLE CONVECTIVE INJECTION =====
                # If enabled and atmosphere is unstable (effective L < 0), randomly redistribute
                # the particle within the PBL. This mimics FLEXPART's deep convection
                # and protects material from rapid surface deposition in the tropics.
                z_m_current = z_m
                convection_applied = false
                if hanna_config.use_simple_convection
                    # Use effective L (calculated above) to trigger convection
                    z_m_new, convection_applied = apply_simple_convection(
                        z_m_current, h, L, dt, hanna_config)

                    if convection_applied
                        z_m_current = z_m_new
                        # Update sigma immediately so the change persists
                        z_sigma_current = sigma_from_height(profile_eval, z_m_current;
                            fallback_sigma=z_sigma_current)
                        # Zero out w_turb so the O-U process doesn't fight the convective jump
                        particle.w_turb = 0.0f0
                    end
                end

                # ===== VERTICAL SUB-TIMESTEPPING (ifine) =====
                # Use multiple substeps for vertical diffusion to improve accuracy
                # near BL top and ground where σ_w varies rapidly with height.
                # Each substep recomputes Hanna params at the current height.
                ifine = hanna_config.ifine
                dt_sub = dt / ifine
                # z_m_current already set above (possibly modified by convection)

                # Pre-compute air density for FLEXPART mode
                x_met_sub = T(1.0) + (pos_domain[1] - T(1.0)) * grid_scale_x
                y_met_sub = if lat_reversed
                    T(ny_met) - (pos_domain[2] - T(1.0)) * grid_scale_y
                else
                    T(1.0) + (pos_domain[2] - T(1.0)) * grid_scale_y
                end
                T_k = winds.t_interp(x_met_sub, y_met_sub, z_sigma_current, local_time_offset)
                P_Pa = winds.ps_interp(x_met_sub, y_met_sub, local_time_offset) * T(100.0) * z_sigma_current
                R_air = T(287.0)
                g = T(9.81)
                rhoa = P_Pa / (R_air * T_k)
                rhograd = -rhoa * g / (R_air * T_k)

                for i_sub in 1:ifine
                    # Recompute Hanna params at current height for each substep
                    hanna_params_sub = if i_sub == 1
                        hanna_params  # Use already-computed params for first substep
                    else
                        compute_hanna_parameters(z_m_current, h, L, ust, wst, hanna_config)
                    end

                    # Generate random numbers for this substep
                    rnd_w_sub = randn()

                    sigw = hanna_params_sub.sigw
                    dsigwdz = hanna_params_sub.dsigwdz
                    tlw = hanna_params_sub.tlw

                    if hanna_config.flexpart_mode
                        # ===== FLEXPART-COMPATIBLE MODE =====
                        # Uses normalized wp = w/σ_w, drift inside O-U step
                        # Matches FLEXPART turbulence module
                        #
                        # Note: particle.w_turb stores NORMALIZED wp in this mode
                        wp_old = T(particle.w_turb)

                        # Compute CBL parameters if strong convection conditions met
                        # CBL activates when -h/L > 5 (checked inside flexpart_vertical_step)
                        cbl_params_sub = if hanna_config.use_cbl && L < T(0.0) && (-h / L) > T(5.0)
                            compute_cbl_parameters(z_m_current, h, L, ust, wst, sigw, dsigwdz)
                        else
                            nothing
                        end

                        wp_new, delz = flexpart_vertical_step(
                            wp_old, sigw, dsigwdz, tlw, dt_sub, rnd_w_sub, rhoa, rhograd;
                            cbl_params=cbl_params_sub, h=h, L=L, C_0=hanna_config.c0_cbl)

                        # ===== FLEXPART DISPLACEMENT LIMIT (line 150) =====
                        # Prevents single-step jumps larger than BL height
                        # Use rem() not mod() - Julia's mod doesn't preserve sign
                        if abs(delz) > h
                            delz = rem(delz, h)
                        end

                        # ===== FLEXPART REFLECTION WITH VELOCITY FLIP (lines 165-174) =====
                        # Work in height coordinates like FLEXPART, not sigma
                        z_m_new = z_m_current + delz
                        icbt = 1  # Reflection flag: 1 = no reflection, -1 = reflected

                        if z_m_new < T(0.0)
                            # Ground reflection
                            z_m_new = -z_m_new
                            icbt = -1
                        elseif z_m_new > h
                            # BL top reflection
                            z_m_new = T(2.0) * h - z_m_new
                            icbt = -1
                        end

                        # Flip velocity if reflected
                        if icbt == -1
                            wp_new = -wp_new
                        end

                        # Clamp to valid range (0 to domain height)
                        # Note: z_max_m is domain height, h is BL height
                        # Particles can be above BL, so use z_max_m for upper bound
                        z_m_new = clamp(z_m_new, T(0.0), z_max_m)

                        particle.w_turb = Float32(wp_new)

                        # Update z_m_current for next substep (avoids recomputing from sigma)
                        z_m_current = z_m_new

                        # ===== CONVERT HEIGHT TO SIGMA =====
                        # Use proper hybrid profile interpolation, NOT linear approximation
                        z_sigma_new = sigma_from_height(profile_eval, z_m_new;
                            fallback_sigma=z_sigma_current)
                        z_sigma_current = z_sigma_new

                    else
                        # ===== ORIGINAL MODE (separate drift correction) =====
                        # Determine if CBL scheme should be used
                        use_cbl = hanna_config.use_cbl && L < 0.0 && (-h / L) > hanna_config.cbl_threshold

                        if use_cbl
                            # Compute CBL parameters
                            cbl_params = compute_cbl_parameters(z_m_current, h, L, ust, wst, sigw, dsigwdz)

                            # Apply CBL scheme for vertical velocity
                            rnd_choice = rand()
                            w_new, cbl_success = apply_cbl_scheme(
                                T(particle.w_turb), cbl_params, sigw, tlw, dt_sub, rnd_choice, rnd_w_sub)

                            if cbl_success
                                particle.w_turb = Float32(w_new)
                            else
                                # Fall back to O-U process
                                particle.w_turb = Float32(ornstein_uhlenbeck_step(
                                    T(particle.w_turb), sigw, tlw, dt_sub, rnd_w_sub))
                            end
                        else
                            # Standard O-U process for vertical component
                            particle.w_turb = Float32(ornstein_uhlenbeck_step(
                                T(particle.w_turb), sigw, tlw, dt_sub, rnd_w_sub))
                        end

                        # Drift correction (separate from O-U in this mode)
                        w_turb = Float64(particle.w_turb)

                        # 1. Standard gradient correction (Thomson 1987)
                        w_drift_gradient = sigw * dsigwdz

                        # 2. Skewness correction
                        w_drift_skewness = if abs(sigw) > T(0.01)
                            (w_turb^2 / sigw) * dsigwdz
                        else
                            T(0.0)
                        end

                        # 3. Density correction
                        w_drift_density = if abs(rhoa) > T(0.01)
                            (sigw^2 / rhoa) * rhograd
                        else
                            T(0.0)
                        end

                        # Total drift and displacement
                        w_drift = w_drift_gradient + w_drift_skewness + w_drift_density
                        w_total = w_turb + w_drift

                        # ===== HEIGHT-COORDINATE FIX =====
                        # Work in height (metres), not sigma - like FLEXPART
                        # The linear approximation dz_sigma = -w*dt/z_max_m is wrong
                        # because sigma is NOT linearly proportional to height
                        delz_m = w_total * dt_sub  # Displacement in metres
                        z_m_new = z_m_current + delz_m
                        z_m_new = clamp(z_m_new, T(0.0), z_max_m)

                        # Convert back to sigma using proper hybrid profile interpolation
                        z_sigma_new = sigma_from_height(profile_eval, z_m_new;
                            fallback_sigma=z_sigma_current)
                        z_sigma_current = z_sigma_new
                        z_m_current = z_m_new
                    end

                    # Clamp to valid range within substep
                    z_sigma_current = clamp(z_sigma_current, σ_grid_min, σ_grid_max)

                    # Update height and density for next substep
                    z_m_current = height_from_sigma(profile_eval, z_sigma_current; fallback_height=prev_height)
                    z_m_current = clamp(z_m_current, T(0.0), z_max_m)

                    # Update rhoa/rhograd for next substep (FLEXPART does this)
                    if i_sub < ifine
                        T_k = winds.t_interp(x_met_sub, y_met_sub, z_sigma_current, local_time_offset)
                        P_Pa = winds.ps_interp(x_met_sub, y_met_sub, local_time_offset) * T(100.0) * z_sigma_current
                        rhoa = P_Pa / (R_air * T_k)
                        rhograd = -rhoa * g / (R_air * T_k)
                    end
                end
            end  # Close else block for at_ground check

            # Apply turbulent displacements
            # NOTE: Wind alignment (FLEXPART-style) was tested but gave worse FMS.
            # Using simple x/y perturbations (random walk) instead.
            # The O-U process already gives temporally correlated velocities.
            x_domain_final += particle.u_turb * dt * map_ratio_x
            y_domain_final += particle.v_turb * dt * map_ratio_y

            # Vertical displacement already handled in sub-timestepping loop above
            # BUGFIX: Use z_sigma_current for both cases - don't force to 1.0!
            z_sigma_final = z_sigma_current

            # Note: Do NOT snap particles to z=1.0 if they're in the surface layer (>= 0.996).
            # That would cause particles starting at z_sigma=0.998 to immediately deposit.
            # Instead, just zero turbulent velocity to prevent upward turbulent motion.
            if z_sigma_final >= 0.996
                particle.w_turb = 0.0f0
            end

            # Reflect at boundaries
            # DISABLED hard barrier at PBL top to allow escape/entrainment
            # if z_sigma_final > tbl
            #     bl_entrainment_thickness = (1.0 - tbl) * (1.0 + 0.1)  # 10% entrainment
            #     top_entrainment = max(0.0, 1.0 - bl_entrainment_thickness)
            #
            #     if z_sigma_final < top_entrainment
            #         z_sigma_final = 2.0 * top_entrainment - z_sigma_final
            #         particle.w_turb = -particle.w_turb
            #     end
            #
            #     z_sigma_final = clamp(z_sigma_final, top_entrainment, 1.0)
            # else
            #     # Above BL
            #     z_sigma_final = clamp(z_sigma_final, 0.0, tbl)
            # end
            z_sigma_final = clamp(z_sigma_final, 0.0, 1.0)

            # Convert back to physical height
            z_sigma_final = clamp(z_sigma_final, σ_grid_min, σ_grid_max)
            z_height_final = height_from_sigma(profile_eval, z_sigma_final; fallback_height=prev_height)

            # PARITY FIX: Do NOT deposit particles after turbulence!
            # Reference order is: drydep → advection → random walk, with NO deposition after walk.
            # Particles that turbulence pushes to the surface survive until the NEXT timestep's
            # drydep call. The previous code here was depositing particles immediately after
            # turbulence, causing 12x higher particle loss than the reference.
            # Now we just clamp the position and let the next timestep's drydep handle it.

            z_height_final = clamp(z_height_final, 0.0, z_max_m)
            prev_sigma = z_sigma_final
            prev_height = z_height_final
            z_sigma_deposition = z_sigma_final  # Update for deposition check

        elseif diffusion_config.apply_diffusion
            # ===== SIMPLE TURBULENT DIFFUSION (random walk) =====
            # Original random walk algorithm for backward compatibility

            # Get wind components at particle position for horizontal diffusion
            # Use end of timestep for wind evaluation (after advection)
            t_diffusion = local_time_offset + dt
            u_wind = winds.u_interp(pos_final[1], pos_final[2], pos_final[3], t_diffusion)
            v_wind = winds.v_interp(pos_final[1], pos_final[2], pos_final[3], t_diffusion)
            vabs = hypot(u_wind, v_wind)

            # Compute top of boundary layer in sigma coordinates (tbl)
            # z_sigma ranges from 0 (top) to 1 (surface)
            # If mixing height is 1500m and domain top is z_max_m, then:
            tbl_from_particle = Float64(particle.tbl)
            tbl_computed = 1.0 - (diffusion_config.hmax / z_max_m)
            tbl = particle.tbl > 0f0 ?
                clamp(Float64(particle.tbl), 0.0, 1.0) :
                clamp(tbl_computed, 0.0, 1.0)

            # Pre-compute time factors
            ratio_v = dt / diffusion_config.tmix_v
            ratio_h = dt / diffusion_config.tmix_h
            if ratio_v <= 0 || ratio_h <= 0
                throw(DomainError(ratio_v,
                    "Invalid turbulent diffusion timestep: dt=$(dt) s, tmix_v=$(diffusion_config.tmix_v) s, tmix_h=$(diffusion_config.tmix_h) s, ratio_h=$(ratio_h)"))
            end
            tsqrtfactor_v = sqrt(ratio_v)
            tsqrtfactor_h = sqrt(ratio_h)

            # Horizontal diffusion (wind-speed dependent)
            # rl = 2*a*((vabs*tmix_h)^b) * tsqrtfactor_h
            in_bl = pos_final[3] > tbl  # Remember: z_sigma > tbl means IN boundary layer
            a = in_bl ? diffusion_config.horizontal_a_bl : diffusion_config.horizontal_a_above

            rl = 2.0 * a * (vabs * diffusion_config.tmix_h)^diffusion_config.horizontal_b * tsqrtfactor_h

            # Random perturbations [-0.5, 0.5]
            rnd_x = rand() - 0.5
            rnd_y = rand() - 0.5
            rnd_z = rand() - 0.5

            # Apply horizontal diffusion in grid coordinates
            x_domain_final += rl * rnd_x * map_ratio_x
            y_domain_final += rl * rnd_y * map_ratio_y

            # Vertical diffusion (dimensionless in sigma coordinates)
            # ERA5 FIX: Use clamped sigma for large particle settling (Issue #38)
            # For particles with large settling velocities, pos_final[3] can be >> 1.0.
            # The reflection formula 2.0 - z would give negative sigma, incorrectly
            # pushing particles to top of BL. GFS doesn't need this fix.
            # NOTE: Only apply this fix when settling is enabled; for pure advection+turbulence,
            # use raw pos_final[3] to avoid systematic downward drift.
            z_sigma_final = (is_era5 && settling_enabled) ? z_sigma_deposition : pos_final[3]

            # DIAGNOSTIC: Log turbulence parameters for first particle at each hour
            hour = current_time_global / 3600.0
            if i == 1 && abs(hour - round(hour)) < 0.01 && get(ENV, "TRANSPORT_TURB_DIAG", "") == "1"
                @info "TURB_DIAG" hour=round(Int, hour) particle=i tbl=round(tbl, digits=4) hbl_m=round(particle.hbl, digits=1) z_sigma=round(z_sigma_final, digits=4) tsqrtfactor_v=round(tsqrtfactor_v, digits=4) in_bl=(z_sigma_final > tbl)
            end

            if z_sigma_final <= tbl  # Above boundary layer
                # vrdbla = labove * tsqrtfactor_v (above BL diffusion)
                rv = diffusion_config.labove * tsqrtfactor_v
                z_sigma_final += rv * rnd_z
                # DIAGNOSTIC
                if i == 1 && abs(hour - round(hour)) < 0.01 && get(ENV, "TRANSPORT_TURB_DIAG", "") == "1"
                    @info "TURB_RV_ABOVE" rv=round(rv, digits=6) rnd_z=round(rnd_z, digits=3) z_after=round(z_sigma_final, digits=4)
                end
            else  # In boundary layer
                # Check if full mixing should be used
                bl_entrainment_thickness = (1.0 - tbl) * (1.0 + diffusion_config.entrainment)
                top_entrainment = max(0.0, 1.0 - bl_entrainment_thickness)

                # CRITICAL: Use full mixing when timestep is large
                # If blfullmix=true OR tsqrtfactor_v > 1.0, redistribute particle randomly in BL
                if diffusion_config.blfullmix || tsqrtfactor_v > 1.0
                    # Full mixing mode - randomly distribute in boundary layer
                    # z = 1.0 - bl_entrainment_thickness * (rnd(3) + 0.5)
                    # This gives z in range [top_entrainment, 1.0]
                    z_sigma_final = 1.0 - bl_entrainment_thickness * (rnd_z + 0.5)
                else
                    # Incremental mixing mode with small displacements and reflections
                    # rv = (1-tbl)*tsqrtfactor_v (in-BL incremental mixing)
                    rv = (1.0 - tbl) * tsqrtfactor_v
                    z_before = z_sigma_final
                    z_sigma_final += rv * rnd_z
                    # DIAGNOSTIC
                    if i == 1 && abs(hour - round(hour)) < 0.01 && get(ENV, "TRANSPORT_TURB_DIAG", "") == "1"
                        @info "TURB_RV_INBL" rv=round(rv, digits=6) rnd_z=round(rnd_z, digits=3) z_before=round(z_before, digits=4) z_after=round(z_sigma_final, digits=4) top_ent=round(top_entrainment, digits=4)
                    end

                    # Reflection from ABL top (with entrainment zone)
                    # DISABLED: To allow particles to escape the PBL and match FLEXPART/HYSPLIT
                    # if z_sigma_final < top_entrainment
                    #     # CRITICAL FIX: Reflect from tbl, not top_entrainment!
                    #     # Reference: part%z = 2.0*part%tbl - part%z
                    #     z_sigma_final = 2.0 * tbl - z_sigma_final
                    # end

                    # Bottom reflection (reference implementation)
                    # ERA5 FIX: Enable bottom reflection for ERA5 (matches reference behaviour)
                    # GFS: Keep clamping to avoid upward drift issues
                    if is_era5 && z_sigma_final > 1.0
                        z_sigma_final = 2.0 - z_sigma_final  # Standard reflection
                    end

                    # Enforce vertical limits
                    z_sigma_final = min(z_sigma_final, 1.0)
                    z_sigma_final = max(z_sigma_final, top_entrainment)
                end
            end

            # Convert back to domain vertical coordinates
            z_sigma_final = clamp(z_sigma_final, σ_grid_min, σ_grid_max)
            z_height_final = height_from_sigma(profile_eval, z_sigma_final; fallback_height=prev_height)

            # PARITY FIX: Do NOT deposit particles after turbulence!
            # (Same fix as Hanna turbulence path above - see comment there for details)
            z_height_final = clamp(z_height_final, 0.0, z_max_m)
            prev_sigma = z_sigma_final
            prev_height = z_height_final
            z_sigma_deposition = z_sigma_final  # Update for deposition check
        end

        # Check bounds
        if !(1.0 <= x_domain_final <= domain.nx && 1.0 <= y_domain_final <= domain.ny)
            # Particle left domain
            for comp in 1:state.ensemble.ncomponents
                set_rad!(particle, comp, -1.0f0)
            end
            n_deposited += 1
            continue
        end

        # Update position (CRITICAL: store sigma not height!)
        # CRITICAL FIX: Do NOT update particle.z after transport!
        # The reference code does not change particle z coordinate after deposition.
        # It only reduces radioactivity (rad_). The position is determined solely by
        # advection and diffusion. Updating particle.z here was causing particles to
        # artificially remain in the surface layer, leading to repeated deposition.
        # particle.z = Float64(z_sigma_deposition)  # REMOVED - causes 66% vs 38% discrepancy!
        state.ensemble.positions[i] = SVector{3,T}(x_domain_final, y_domain_final, T(z_sigma_deposition))

        # TRACE: Write particle trace AFTER position update (FIX for 300km coordinate bug)
        # Trace ALL particles for full statistics
        if (config.max_duration == 0.0 || current_time_global + dt <= config.max_duration)

            # Convert domain coords to lat/lon
            lon = domain.lon_min + (x_domain_final - 1) * (domain.lon_max - domain.lon_min) / (domain.nx - 1)
            lat = domain.lat_min + (y_domain_final - 1) * (domain.lat_max - domain.lat_min) / (domain.ny - 1)

            # Get altitude from sigma using MET grid coordinates (not domain coords)
            # Use local_time_offset within the current met window to match winds.t_range
            # (current_time_global may span multiple windows; we clamp below regardless.)
            t_eval = T(local_time_offset)
            # Map domain → met coordinates consistently with earlier mapping
            x_met_trace = 1.0 + (x_domain_final - 1.0) * grid_scale_x
            y_met_trace = if lat_reversed
                (ny_met - (y_domain_final - 1.0) * grid_scale_y)
            else
                1.0 + (y_domain_final - 1.0) * grid_scale_y
            end
            # Use the same 4D height interpolant as integration, evaluated directly
            # at the particle's (x,y,sigma,t). This mirrors the bilinear + vertical + temporal
            # interpolation and avoids extra error from re-tabulating a 1D profile.
            xq = clamp(x_met_trace, 1.0f0, Float32(winds.nx))
            yq = clamp(y_met_trace, 1.0f0, Float32(winds.ny))
            σq = clamp(Float32(z_sigma_deposition), Float32(winds.z_grid[1]), Float32(winds.z_grid[end]))
            tq = clamp(Float32(t_eval), Float32(winds.t_range[1]), Float32(winds.t_range[2]))
            altitude_m = Float64(winds.h_interp(xq, yq, σq, tq))

            # DEBUG: Disabled for performance (TRACE WRITE logging)

            # Compute diagnostic variables for trace (matching reference output)
            w_wind_trace = advection_enabled ? winds.w_interp(x_met_trace, y_met_trace, z_sigma_deposition, T(local_time_offset + dt)) : 0.0
            vg_sigma_trace = 0.0  # Will be computed if settling enabled
            if settling_enabled && !isempty(particle_size_config.particle_size_indices)
                size_idx_local = particle_size_config.particle_size_indices[i]
                vg_ms = 0.0
                if !isnothing(particle_size_config.fixed_gravity_cm_s) && 1 <= size_idx_local <= length(particle_size_config.fixed_gravity_cm_s)
                    vg_ms = particle_size_config.fixed_gravity_cm_s[size_idx_local] * 0.01  # cm/s → m/s
                end
                # Convert to sigma tendency
                if z_sigma_deposition <= 0.996  # Not in surface layer
                    z_grid = winds.z_grid
                    z_clamped = clamp(z_sigma_deposition, z_grid[1] + eps(T), z_grid[end] - eps(T))
                    idx = searchsortedlast(z_grid, z_clamped)
                    idx = clamp(idx, 1, length(z_grid) - 1)
                    sigma_upper = z_grid[idx]
                    sigma_lower = z_grid[idx + 1]
                    # Use h_interp directly for height calculation (consistent with altitude_m above)
                    h_upper = Float64(winds.h_interp(xq, yq, Float32(sigma_upper), tq))
                    h_lower = Float64(winds.h_interp(xq, yq, Float32(sigma_lower), tq))
                    dz = h_upper - h_lower
                    dsigma = sigma_lower - sigma_upper
                    if abs(dz) > eps(T)
                        vg_sigma_trace = vg_ms * dsigma / dz
                    end
                end
            end
            w_total_trace = w_wind_trace + vg_sigma_trace
            in_surface_trace = z_sigma_deposition > 0.996

            # Optional diagnostic: write detailed w interpolation breakdown for particle 1
            # Enable by setting environment variable TRANSPORT_W_DIAG=1
            if i == 1 && get(ENV, "TRANSPORT_W_DIAG", "0") == "1"
                # Write hourly diagnostics under outputs/ to avoid mixing with step diagnostics
                base_dir = dirname(trace_filename)
                diag_dir = joinpath(base_dir, "outputs")
                if !isdir(diag_dir)
                    mkpath(diag_dir)
                end
                diag_path = joinpath(diag_dir, "w_diag_hour_particle1.csv")

                # Compute reference-style contributions from raw w fields
                # Indices and fractional offsets
                ii = clamp(floor(Int, x_domain_final), 1, winds.nx)
                jj = clamp(floor(Int, y_domain_final), 1, winds.ny)
                dxf = clamp(T(x_domain_final - ii), zero(T), one(T))
                dyf = clamp(T(y_domain_final - jj), zero(T), one(T))
                c1 = (one(T) - dyf) * (one(T) - dxf)
                c2 = (one(T) - dyf) * dxf
                c3 = dyf * (one(T) - dxf)
                c4 = dyf * dxf

                # Time weights
                t1w, t2w = winds.t_range
                tloc = clamp(T(local_time_offset + dt), t1w, t2w)
                denom = max(t2w - t1w, eps(T))
                rt1 = (t2w - tloc) / denom
                rt2 = (tloc - t1w) / denom

                # Vertical bracket
                zgrid = winds.z_grid
                if z_sigma_deposition <= zgrid[1]
                    k2 = 1; k1 = 2
                elseif z_sigma_deposition >= zgrid[end]
                    k1 = winds.nk; k2 = k1 - 1
                else
                    idxv = searchsortedlast(zgrid, z_sigma_deposition)
                    k2 = clamp(idxv, 1, winds.nk - 1)
                    k1 = k2 + 1
                end
                v1 = zgrid[k1]; v2 = zgrid[k2]
                denomv = max(v1 - v2, eps(T))
                dz1 = (z_sigma_deposition - v2) / denomv
                dz2 = one(T) - dz1

                # Horizontal bilinear for w1_raw/w2_raw at k1/k2
                w1_k1 = c1 * winds.w1_raw[ii, jj, k1] + c2 * winds.w1_raw[min(ii+1, winds.nx), jj, k1] +
                         c3 * winds.w1_raw[ii, min(jj+1, winds.ny), k1] + c4 * winds.w1_raw[min(ii+1, winds.nx), min(jj+1, winds.ny), k1]
                w1_k2 = c1 * winds.w1_raw[ii, jj, k2] + c2 * winds.w1_raw[min(ii+1, winds.nx), jj, k2] +
                         c3 * winds.w1_raw[ii, min(jj+1, winds.ny), k2] + c4 * winds.w1_raw[min(ii+1, winds.nx), min(jj+1, winds.ny), k2]
                w2_k1 = c1 * winds.w2_raw[ii, jj, k1] + c2 * winds.w2_raw[min(ii+1, winds.nx), jj, k1] +
                         c3 * winds.w2_raw[ii, min(jj+1, winds.ny), k1] + c4 * winds.w2_raw[min(ii+1, winds.nx), min(jj+1, winds.ny), k1]
                w2_k2 = c1 * winds.w2_raw[ii, jj, k2] + c2 * winds.w2_raw[min(ii+1, winds.nx), jj, k2] +
                         c3 * winds.w2_raw[ii, min(jj+1, winds.ny), k2] + c4 * winds.w2_raw[min(ii+1, winds.nx), min(jj+1, winds.ny), k2]
                wk1 = rt1 * w1_k1 + rt2 * w2_k1
                wk2 = rt1 * w1_k2 + rt2 * w2_k2
                w_blend = wk1 * dz1 + wk2 * dz2
                w_interp_val = winds.w_interp(x_domain_final, y_domain_final, z_sigma_deposition, tloc)

                # Write header once
                if !isfile(diag_path)
                    open(diag_path, "w") do io
                        println(io, "time_s,x,y,z_sigma,i,j,k1,k2,v1,v2,rt1,rt2,dz1,dz2,w1_k1,w1_k2,w2_k1,w2_k2,wk1,wk2,w_blend,w_interp")
                    end
                end
                open(diag_path, "a") do io
                    println(io, join([current_time_global + dt,
                                      x_domain_final, y_domain_final, z_sigma_deposition,
                                      ii, jj, k1, k2, v1, v2, rt1, rt2, dz1, dz2,
                                      w1_k1, w1_k2, w2_k1, w2_k2, wk1, wk2, w_blend, w_interp_val], ","))
                end
            end

            # Use trace_time_override if specified (for istep=0 parity where trace is at t=0)
            trace_time = isnothing(trace_time_override) ? (current_time_global + dt) : trace_time_override

            # Only write trace if output_config allows it at this time
            if should_write_trace(output_config, trace_time, dt)
                open(trace_filename, "a") do io
                    println(io, "$(i),$(trace_time),$(lat),$(lon),$(z_sigma_deposition),$(altitude_m),$(w_wind_trace),$(vg_sigma_trace),$(w_total_trace),$(in_surface_trace),0.0,0.0,0.0,0,0")
                end
            end
        end

        # Apply radioactive decay to airborne particles after transport
        for comp in 1:state.ensemble.ncomponents
            current_mass = get_rad(particle, comp)
            new_mass = current_mass * decay_params[comp].decayrate
            set_rad!(particle, comp, Float32(new_mass))
        end
    end

    return n_deposited
end

"""
    run_simulation!(state::SimulationState, met_files::Vector{String};
                   particle_size_config=ParticleSizeConfig(),
                   deposition_config=DepositionConfig{Float64}(),
                   decay_params=state.decay_params,
                   config=SimulationConfig{Float64}(),
                   dry_deposition_enabled=true,
                   wet_deposition_enabled=true)

Run complete atmospheric transport simulation.

# Arguments
- `state`: Initialized simulation state
- `met_files`: Vector of paths to ERA5 NetCDF files (sorted by time)

# Keyword Arguments
- `particle_size_config`: Particle size and settling configuration
- `deposition_config`: Deposition physics configuration
- `decay_params`: Radioactive decay parameters
- `config`: General simulation configuration
- `bomb_state`: Bomb decay state for Way-Wigner decay (optional)

# Returns
- `snapshots::Vector{SimulationSnapshot}`: Saved simulation snapshots

# Example
```julia
# Setup
domain = SimulationDomain(...)
sources = [ReleaseSource(...)]
state = initialize_simulation(domain, sources, ["Cs137"], [DecayParams(...)])

# Configure particle sizes
size_config = ParticleSizeConfig(
    size_bins = [ParticleProperties(diameter_μm=100.0, density_gcm3=2.5)],
    vgrav_tables = build_vgrav_tables([...])
)

# Get meteorological files
met_files = sort(filter(f -> endswith(f, ".nc"), readdir("met_data/", join=true)))

# Run
snapshots = run_simulation!(state, met_files,
                           particle_size_config=size_config)

# Save
save_netcdf(state, "results.nc")
```
"""
function run_simulation!(state::SimulationState{T},
                        met_files::Vector{String};
                        particle_size_config::ParticleSizeConfig=ParticleSizeConfig(),
                        deposition_config::DepositionConfig{T}=DepositionConfig{T}(),
                        diffusion_config::TurbulentDiffusionConfig{T}=TurbulentDiffusionConfig{T}(),
                        hanna_config::Union{Nothing,HannaTurbulenceConfig{T}}=nothing,
                        decay_params::Vector{DecayParams{T}}=DecayParams{T}[],
                        config::SimulationConfig{T}=SimulationConfig{T}(),
                        bomb_state::Union{Nothing,BombDecayState}=nothing,
                        advection_enabled::Bool=true,
                        settling_enabled::Bool=true,
                        dry_deposition_enabled::Bool=true,
                        wet_deposition_enabled::Bool=true,
                        release_height_m::Float64=10000.0,
                        numerical_config::Union{NumericalConfig, ERA5NumericalConfig, Nothing}=nothing,
                        trace_filename::String="julia_particles_trace.csv",
                        sigma_already_initialized::Bool=false,
                        met_data_cache::Union{Nothing, Dict}=nothing,
                        met_format_override::Union{Nothing, MetFormat}=nothing,
                        met_dimensions::Union{Nothing, Tuple{Int,Int,Int}}=nothing,
                        cache_init_file_idx::Int=1,
                        cache_init_time_idx::Int=1) where T<:Real

    if config.verbose
        println("="^70)
        println("TRANSPORT SIMULATION - RUN")
        println("="^70)
        println()
    end

    # TRACE: Initialize multi-particle trace file with diagnostic columns
    open(trace_filename, "w") do io
        println(io, "particle_id,time_s,lat,lon,z_sigma,altitude_m,w_meteo,vg_sigma,w_total,in_surface,dz,theta,deta,k1,k2")
    end

    # Use provided met files
    era5_files = met_files
    if isempty(era5_files)
        error("No meteorological files provided")
    end

    n_files = config.max_files > 0 ? min(config.max_files, length(era5_files)) : length(era5_files)

    if config.verbose
        println("Found $(length(era5_files)) met files, processing $n_files")
        println("Initial particle count: $(length(state.ensemble.particles))")
        if !isempty(state.ensemble.particles)
            p1 = state.ensemble.particles[1]
            println("Particle 1 active: $(is_active(p1)), rad=$(p1.rad)")
        end
        println()
    end

    # Initialize met field readers
    meteo_params = MeteoParams()
    init_meteo_params!(meteo_params, "era5_grib")

    # Detect meteorological data format (ERA5 or GFS) - use override if provided (thread-safe)
    met_format = if !isnothing(met_format_override)
        met_format_override
    else
        detect_met_format(era5_files[1])
    end

    if config.verbose
        println("Detected met format: $(typeof(met_format))")
    end

    # Get dimensions - use override if provided (thread-safe)
    nx_met, ny_met, nk_met = if !isnothing(met_dimensions)
        met_dimensions
    else
        NCDataset(era5_files[1]) do ds
            get_met_dimensions(met_format, ds)
        end
    end

    met_fields = MeteoFields(nx_met, ny_met, nk_met, T=Float32)
    domain_heights_initialized = false

    # Snapshots storage
    snapshots = SimulationSnapshot{T}[]
    current_time = 0.0
    snapshot_times = if !isnothing(config.saveat)
        sort(config.saveat)
    else
        []
    end
    snapshot_idx = 1
    next_snapshot_time = if !isempty(snapshot_times)
        snapshot_times[1]
    else
        config.dt_output
    end

    # TRACE: Write initial t=0 positions (FIX for 28km initial offset bug)
    # Write initial positions for ALL particles
    # NOTE: Defer writing t=0 trace until after met data is loaded
    # so we can compute altitude from sigma consistently

    # CRITICAL: Initialize met fields ONCE before the main loop
    # Find the file and time indices that bracket the simulation start time
    if config.verbose
        println("Initializing meteorological fields...")
    end

    # Find which file contains the simulation start time
    init_file_idx = 1
    init_time_idx1 = 1
    init_time_idx2 = 2
    found_start_time = false

    # If cache is provided, use specified init indices (caller must determine correct values)
    if !isnothing(met_data_cache) && !isempty(met_data_cache)
        # Use provided init indices for cache - caller is responsible for setting correct values
        init_file_idx = cache_init_file_idx
        init_time_idx1 = cache_init_time_idx
        init_time_idx2 = cache_init_time_idx + 1
        found_start_time = true
        if config.verbose
            println("Using cached met data: file_idx=$init_file_idx, time_idx=$init_time_idx1")
        end
    else
        # No cache - search through files to find start time
        if config.verbose
            println("DEBUG: state.domain.t_start = ", state.domain.t_start)
            println("DEBUG: typeof(state.domain.t_start) = ", typeof(state.domain.t_start))
        end

        # Search through ALL files to find the one containing start time
        # (not just n_files, since start time might be in a later file)
        for (file_idx, era5_file) in enumerate(era5_files)
        # Parse filename to get time range: era5_YYYYMMDD_HH-HH_snap.nc or era5_YYYYMMDD_snap.nc
        filename = basename(era5_file)
        
        # Try full format first: era5_YYYYMMDD_HH-HH_snap.nc
        m_full = match(r"era5_(\d{8})_(\d{2})-(\d{2})_snap\.nc", filename)
        # Try shorter format: era5_YYYYMMDD_snap.nc
        m_short = match(r"era5_(\d{8})_snap\.nc", filename)

        if m_full !== nothing || m_short !== nothing
            m = m_full !== nothing ? m_full : m_short
            file_date_str = m.captures[1]  # YYYYMMDD
            
            # Default to full day if hours not in filename
            file_hour_start = m_full !== nothing ? parse(Int, m.captures[2]) : 0
            file_hour_end = m_full !== nothing ? parse(Int, m.captures[3]) : 23

            # Construct DateTime for file start and end
            file_year = parse(Int, file_date_str[1:4])
            file_month = parse(Int, file_date_str[5:6])
            file_day = parse(Int, file_date_str[7:8])

            # Convert parsed filename times to Transport.DateTime for comparison
            # (state.domain.t_start is Transport.DateTime, not Dates.DateTime)
            file_start = Transport.DateTime(file_year, file_month, file_day, file_hour_start)
            file_end = Transport.DateTime(file_year, file_month, file_day, file_hour_end)

            if config.verbose && file_idx <= 6
                println("DEBUG: file_start=$file_start, file_end=$file_end")
                println("DEBUG: Checking: $file_start <= $(state.domain.t_start) <= $file_end")
                println("DEBUG: Result: $(state.domain.t_start >= file_start) && $(state.domain.t_start <= file_end)")
            end

            # Check if simulation start time falls in this file's range
            if state.domain.t_start >= file_start && state.domain.t_start <= file_end
                init_file_idx = file_idx

                # Load times to find which window brackets the start time
                NCDataset(era5_file) do ds
                    times = get_time_variable(met_format, ds)

                    # Find bracketing window by comparing durations
                    for i in 1:(length(times)-1)
                        # Compute time span of this window in hours
                        window_duration_hrs = Float64(value(times[i+1] - times[i])) / 3600000.0  # ms to hours

                        # Compute offset from window start in hours
                        # Both are Transport.DateTime now, so subtraction returns Transport.Duration with .hours field
                        offset_from_file_start_hrs = Float64((state.domain.t_start - file_start).hours)
                        window_i_start_offset_hrs = Float64(value(times[i] - times[1])) / 3600000.0
                        window_i_end_offset_hrs = Float64(value(times[i+1] - times[1])) / 3600000.0

                        if offset_from_file_start_hrs >= window_i_start_offset_hrs && offset_from_file_start_hrs <= window_i_end_offset_hrs
                            init_time_idx1 = i
                            init_time_idx2 = i + 1
                            found_start_time = true
                            break
                        end
                    end
                end

                if config.verbose
                    println("  Found start time in file $(filename)")
                    println("  Using time indices ($init_time_idx1, $init_time_idx2)")
                end
                break
            end
        end
        end  # end for loop
    end  # end else (no cache)

    if !found_start_time && config.verbose
        @warn "Could not find met file containing start time, using first file"
    end

    # Get time span for initial met window (needed for correct wind interpolation)
    init_time_diff = T(7200.0)  # Default 2 hours, will be updated from file
    if !isnothing(met_data_cache) && haskey(met_data_cache, (init_file_idx, init_time_idx1))
        # Use pre-loaded cached met data (thread-safe, no file I/O)
        cached_mf = met_data_cache[(init_file_idx, init_time_idx1)]
        copy_met_fields!(met_fields, cached_mf)
        update_domain_vertical!(state.domain, met_fields)
        state.domain.xm .= met_fields.xm
        state.domain.ym .= met_fields.ym
        domain_heights_initialized = true
        # Use default time diff for cached data (assumes 1-hour timesteps)
        init_time_diff = T(3600.0)
    else
        NCDataset(era5_files[init_file_idx]) do ds
            read_initial_met_fields!(met_format, met_fields, ds, init_time_idx1, init_time_idx2)
            # Initialize domain heights from first met data
            update_domain_vertical!(state.domain, met_fields)
            # Align domain map factors with met-derived ones (match reference mapfield)
            state.domain.xm .= met_fields.xm
            state.domain.ym .= met_fields.ym
            domain_heights_initialized = true
            # Get actual time span from met file
            times_init = get_time_variable(met_format, ds)
            if length(times_init) >= 2
                init_time_diff = T(Float64(value(times_init[init_time_idx2] - times_init[init_time_idx1])) / 1000.0)
            end
        end
    end

    # DEBUG: Disabled for performance (create_wind_interpolants check)

    # Proper initialization: set sigma from desired release height using hybrid profile at t=0
    # Then compute altitude from sigma consistently (no cheating) — this yields ~91 m exactly.
    # CRITICAL FIX (Issue #1): GFS needs w-wind negation to match reference sigma-dot convention
    negate_w_gfs = isa(met_format, GFSFormat)
    # CRITICAL: Use correct time span for winds0 (not 1.0s!) to enable proper time interpolation
    winds0 = create_wind_interpolants(met_fields, 0.0, init_time_diff,
                                      config=numerical_config,
                                      negate_v = false,  # ERA5 already has northward-positive v after lat flip
                                      negate_w = negate_w_gfs,  # GFS: true (fixes sigma), ERA5: false (unchanged)
                                      lon_min=state.domain.lon_min,
                                      lon_max=state.domain.lon_max,
                                      lat_min=state.domain.lat_min,
                                      lat_max=state.domain.lat_max)

    # Runtime sanity check: h_interp must match raw hlevel at grid knots
    try
        vlev_dbg = copy(met_fields.vlevel)
        level_perm_dbg = sortperm(vlev_dbg)
        z_grid_dbg = collect(vlev_dbg[level_perm_dbg])
        nx_dbg, ny_dbg, nk_dbg = met_fields.nx, met_fields.ny, met_fields.nk
        isamp_dbg = unique!(clamp.([1, max(1, Int(round(nx_dbg/2))), nx_dbg], 1, nx_dbg))
        jsamp_dbg = unique!(clamp.([1, max(1, Int(round(ny_dbg/2))), ny_dbg], 1, ny_dbg))
        ksamp_dbg = unique!(clamp.([2, max(2, Int(round(nk_dbg/2))), nk_dbg-3, nk_dbg-1], 1, nk_dbg))
        t1_dbg, t2_dbg = winds0.t_range
        maxd1 = 0.0
        maxd2 = 0.0
        for i_dbg in isamp_dbg, j_dbg in jsamp_dbg, k_dbg in ksamp_dbg
            σ_dbg = z_grid_dbg[k_dbg]
            # t1 vs hlevel1
            h_itp_1 = Float64(winds0.h_interp(Float64(i_dbg), Float64(j_dbg), Float64(σ_dbg), Float64(t1_dbg)))
            h_raw_1 = Float64(met_fields.hlevel1[i_dbg, j_dbg, level_perm_dbg[k_dbg]])
            maxd1 = max(maxd1, abs(h_itp_1 - h_raw_1))
            # t2 vs hlevel2
            h_itp_2 = Float64(winds0.h_interp(Float64(i_dbg), Float64(j_dbg), Float64(σ_dbg), Float64(t2_dbg)))
            h_raw_2 = Float64(met_fields.hlevel2[i_dbg, j_dbg, level_perm_dbg[k_dbg]])
            maxd2 = max(maxd2, abs(h_itp_2 - h_raw_2))
        end
        if config.verbose
            println("DEBUG alignment (winds0): max |Δ| t1=$(round(maxd1,digits=6)) m, t2=$(round(maxd2,digits=6)) m")
        end
        # Use tolerant threshold (supports cubic vertical mode); catches gross errors
        if !(maxd1 ≤ 0.5 && maxd2 ≤ 0.5)
            @warn "h_interp alignment check failed (winds0)" max_diff_t1=maxd1 max_diff_t2=maxd2
        end
    catch err
        @warn "h_interp alignment check (winds0) threw" err
    end

    for (i, particle) in enumerate(state.ensemble.particles)
        if is_active(particle)
            pos = state.ensemble.positions[i]
            x, y = pos[1], pos[2]

            if sigma_already_initialized
                # Bomb/cylinder release: particles already have correct sigma from initialization
                # Just use the existing sigma value, don't override with release_height_m
                sigma0 = pos[3]
            else
                # Point release: compute sigma from the single release_height_m value
                sigma0 = sigma_from_height(winds0, x, y, release_height_m, 0.0; fallback_sigma=pos[3])
                state.ensemble.positions[i] = SVector{3,Float32}(Float32(x), Float32(y), Float32(sigma0))
            end

            # Convert domain coords to lat/lon for trace and compute altitude back from sigma for consistency
            lon = state.domain.lon_min + (x - 1) * (state.domain.lon_max - state.domain.lon_min) / (state.domain.nx - 1)
            lat = state.domain.lat_min + (y - 1) * (state.domain.lat_max - state.domain.lat_min) / (state.domain.ny - 1)
            altitude_m = height_from_sigma(winds0, x, y, sigma0, 0.0; fallback_height=release_height_m)

            open(trace_filename, "a") do io
                println(io, "$(i),0.0,$(lat),$(lon),$(sigma0),$(altitude_m),0.0,0.0,0.0,false,0.0,0.0,0.0,0,0")
            end
        end
    end

    # PARITY FIX: Perform "istep=0" integration step before main loop
    # Reference integrates at istep=0 and writes trace at t=istep*tstep=0.
    # Without this step, Julia does N integrations while the reference does N+1.
    if config.verbose
        println("Performing istep=0 integration step (reference parity)")
    end
    _ = integrate_timestep!(state, winds0, T(config.dt_particle),
                           particle_size_config, deposition_config,
                           decay_params, config,
                           diffusion_config=diffusion_config,
                           hanna_config=hanna_config,
                           advection_enabled=advection_enabled,
                           settling_enabled=settling_enabled,
                           dry_enabled=dry_deposition_enabled,
                           wet_enabled=wet_deposition_enabled,
                           current_time_global=T(0.0),
                           local_time_offset=T(0.0),
                           numerical_config=numerical_config,
                           trace_filename=trace_filename,
                           is_era5=isa(met_format, ERA5Format),
                           trace_time_override=T(0.0),  # Write trace at t=0 like reference istep=0
                           output_config=config.output_config)

    # Main simulation loop - start from file containing the release time
    file_range_start = init_file_idx
    file_range_end = min(init_file_idx + n_files - 1, length(era5_files))
    actual_n_files = file_range_end - file_range_start + 1

    if config.verbose
        println("Processing files $file_range_start to $file_range_end ($actual_n_files files)")
    end

    for (loop_idx, file_idx) in enumerate(file_range_start:file_range_end)
        era5_file = era5_files[file_idx]
        if config.verbose
            println("[File $loop_idx/$actual_n_files (global #$file_idx)] $(basename(era5_file))")
        end

        # Determine number of time windows for this file
        n_time_windows_file = if !isnothing(met_data_cache)
            # Count cached timesteps for this file
            max_t = 0
            for k in keys(met_data_cache)
                if k[1] == file_idx
                    max_t = max(max_t, k[2])
                end
            end
            if max_t > 0
                max(0, max_t - 1)  # n_windows = n_times - 1
            else
                # Cache exists but doesn't have this file - read from file
                NCDataset(era5_file) do ds
                    length(get_time_variable(met_format, ds)) - 1
                end
            end
        else
            NCDataset(era5_file) do ds
                length(get_time_variable(met_format, ds)) - 1
            end
        end

        # Process time windows for this file
        for window_idx in 1:n_time_windows_file
            # Load met fields from cache or file
            if !isnothing(met_data_cache) && haskey(met_data_cache, (file_idx, window_idx))
                if !(file_idx == init_file_idx && window_idx == init_time_idx1)
                    cached_mf = met_data_cache[(file_idx, window_idx)]
                    copy_met_fields!(met_fields, cached_mf)
                    state.domain.xm .= met_fields.xm
                    state.domain.ym .= met_fields.ym
                end
                time_diff = 3600.0  # Default 1 hour for cached data
            else
                NCDataset(era5_file) do ds
                    met_fields.xm .= 1.0f0
                    met_fields.ym .= 1.0f0
                    if !(file_idx == init_file_idx && window_idx == init_time_idx1)
                        read_met_fields!(met_format, met_fields, ds, window_idx, window_idx + 1)
                        state.domain.xm .= met_fields.xm
                        state.domain.ym .= met_fields.ym
                    end
                end
                time_diff = NCDataset(era5_file) do ds
                    times = get_time_variable(met_format, ds)
                    Float64(value(times[window_idx + 1] - times[window_idx])) / 1000.0
                end
            end

            n_substeps = max(1, Int(ceil(time_diff / config.dt_particle)))
            dt_sub = time_diff / n_substeps
            all_particles_done = false

            # CRITICAL FIX: Create wind interpolants ONCE per met window with correct time domain
                # met_fields contains data at t=0 and t=time_diff (e.g., 0h and 3h)
                # The interpolator must know the actual time span to interpolate correctly
                # Use physical v for advection; y-grid increases northward (no negation)
                negate_v_for_advection = false  # Keep v as physical northward (no negation)
                # CRITICAL FIX (Issue #1): GFS needs w-wind negation, ERA5 does not
                negate_w_for_advection = isa(met_format, GFSFormat)
                winds = create_wind_interpolants(met_fields, 0.0, time_diff,
                                                config=numerical_config,
                                                negate_v=negate_v_for_advection,
                                                negate_w=negate_w_for_advection,
                                                lon_min=state.domain.lon_min,
                                                lon_max=state.domain.lon_max,
                                                lat_min=state.domain.lat_min,
                                                lat_max=state.domain.lat_max)

                # Runtime sanity check: h_interp must match raw hlevel at grid knots
                try
                    vlev_dbg = copy(met_fields.vlevel)
                    level_perm_dbg = sortperm(vlev_dbg)
                    z_grid_dbg = collect(vlev_dbg[level_perm_dbg])
                    nx_dbg, ny_dbg, nk_dbg = met_fields.nx, met_fields.ny, met_fields.nk
                    isamp_dbg = unique!(clamp.([1, max(1, Int(round(nx_dbg/2))), nx_dbg], 1, nx_dbg))
                    jsamp_dbg = unique!(clamp.([1, max(1, Int(round(ny_dbg/2))), ny_dbg], 1, ny_dbg))
                    ksamp_dbg = unique!(clamp.([2, max(2, Int(round(nk_dbg/2))), nk_dbg-3, nk_dbg-1], 1, nk_dbg))
                    t1_dbg, t2_dbg = winds.t_range
                    maxd1 = 0.0
                    maxd2 = 0.0
                    for i_dbg in isamp_dbg, j_dbg in jsamp_dbg, k_dbg in ksamp_dbg
                        σ_dbg = z_grid_dbg[k_dbg]
                        # t1 vs hlevel1
                        h_itp_1 = Float64(winds.h_interp(Float64(i_dbg), Float64(j_dbg), Float64(σ_dbg), Float64(t1_dbg)))
                        h_raw_1 = Float64(met_fields.hlevel1[i_dbg, j_dbg, level_perm_dbg[k_dbg]])
                        maxd1 = max(maxd1, abs(h_itp_1 - h_raw_1))
                        # t2 vs hlevel2
                        h_itp_2 = Float64(winds.h_interp(Float64(i_dbg), Float64(j_dbg), Float64(σ_dbg), Float64(t2_dbg)))
                        h_raw_2 = Float64(met_fields.hlevel2[i_dbg, j_dbg, level_perm_dbg[k_dbg]])
                        maxd2 = max(maxd2, abs(h_itp_2 - h_raw_2))
                    end
                    # Mid-time consistency: linear in time at grid knots
                    tmid = (t1_dbg + t2_dbg) * 0.5
                    rt2 = (tmid - t1_dbg) / (t2_dbg - t1_dbg)
                    rt1 = 1.0 - rt2
                    maxdt = 0.0
                    for i_dbg in isamp_dbg, j_dbg in jsamp_dbg, k_dbg in ksamp_dbg
                        σ_dbg = z_grid_dbg[k_dbg]
                        h_itp_mid = Float64(winds.h_interp(Float64(i_dbg), Float64(j_dbg), Float64(σ_dbg), Float64(tmid)))
                        h_lin_mid = rt1*Float64(met_fields.hlevel1[i_dbg, j_dbg, level_perm_dbg[k_dbg]]) +
                                    rt2*Float64(met_fields.hlevel2[i_dbg, j_dbg, level_perm_dbg[k_dbg]])
                        maxdt = max(maxdt, abs(h_itp_mid - h_lin_mid))
                    end
                    if config.verbose
                        println("DEBUG alignment (winds): max |Δ| t1=$(round(maxd1,digits=6)) m, t2=$(round(maxd2,digits=6)) m, tmid=$(round(maxdt,digits=6)) m")
                    end
                    if !(maxd1 ≤ 0.5 && maxd2 ≤ 0.5 && maxdt ≤ 0.5)
                        @warn "h_interp alignment check failed (winds)" max_diff_t1=maxd1 max_diff_t2=maxd2 max_diff_tmid=maxdt
                    end
                catch err
                    @warn "h_interp alignment check (winds) threw" err
                end

                # Track local time within this met window (resets to 0 for each window)
                local_time = 0.0

                # Define local scaling and orientation used by diagnostics
                nx_met = winds.nx
                ny_met = winds.ny
                grid_scale_x = (nx_met - 1) / (state.domain.nx - 1)
                grid_scale_y = (ny_met - 1) / (state.domain.ny - 1)
                lat_reversed = (winds.y_grid[end] < winds.y_grid[1])

                for sub_idx in 1:n_substeps

                    # Check if we've reached the maximum simulation duration
                    if config.max_duration > 0 && current_time >= config.max_duration
                        if config.verbose
                            println("  Reached maximum simulation duration: $(config.max_duration)s ($(config.max_duration/3600)h)")
                        end
                        all_particles_done = true
                        break
                    end

                    # Prepare decay rates per sub-step
                    if !isnothing(bomb_state)
                        bomb_state.total_time_s = current_time
                    end
                    prepare_decay_rates!(decay_params, dt_sub, bomb_state=bomb_state)

                    # Count active particles
                    n_active_before = count(is_active(p) for p in state.ensemble.particles)

                    if n_active_before == 0
                        if config.verbose
                            println("  All particles deposited or left domain!")
                        end
                        all_particles_done = true
                        break
                    end

                    # Integrate timestep with local time relative to met window
                    # ODE solver will integrate from local_time to local_time + dt_sub

                    # Optional per-step w diagnostic for particle 1
                    if get(ENV, "TRANSPORT_W_DIAG_STEP", "0") == "1"
                        # Only for particle 1 to keep file small
                        for i_diag in 1:1
                            if is_active(state.ensemble.particles[i_diag])
                                pos_diag = state.ensemble.positions[i_diag]
                                # Map domain coords to met coords
                                x_met = 1.0 + (pos_diag[1] - 1.0) * grid_scale_x
                                y_met = if lat_reversed
                                    (ny_met - (pos_diag[2] - 1.0) * grid_scale_y)
                                else
                                    1.0 + (pos_diag[2] - 1.0) * grid_scale_y
                                end
                                z_sigma = pos_diag[3]
                                # Compute reference-style w components at start of substep
                                ii = clamp(floor(Int, x_met), 1, winds.nx)
                                jj = clamp(floor(Int, y_met), 1, winds.ny)
                                dxf = clamp(Float64(x_met - ii), 0.0, 1.0)
                                dyf = clamp(Float64(y_met - jj), 0.0, 1.0)
                                c1 = (1.0 - dyf) * (1.0 - dxf)
                                c2 = (1.0 - dyf) * dxf
                                c3 = dyf * (1.0 - dxf)
                                c4 = dyf * dxf
                                # Time weights within window
                                t1w, t2w = winds.t_range
                                tloc = Float64(local_time)
                                tloc = clamp(tloc, t1w, t2w)
                                denom = max(t2w - t1w, eps(Float64))
                                rt1 = (t2w - tloc) / denom
                                rt2 = (tloc - t1w) / denom
                                # Vertical bracket in sigma space
                                zgrid = winds.z_grid
                                if z_sigma <= zgrid[1]
                                    k2 = 1; k1 = 2
                                elseif z_sigma >= zgrid[end]
                                    k1 = winds.nk; k2 = k1 - 1
                                else
                                    idxv = searchsortedlast(zgrid, z_sigma)
                                    k2 = clamp(idxv, 1, winds.nk - 1)
                                    k1 = k2 + 1
                                end
                                v1 = zgrid[k1]; v2 = zgrid[k2]
                                denomv = max(v1 - v2, eps(Float64))
                                dz1 = (z_sigma - v2) / denomv
                                dz2 = 1.0 - dz1
                                # Bilinear for w1_raw/w2_raw at k1/k2
                                w1_k1 = c1 * winds.w1_raw[ii, jj, k1] + c2 * winds.w1_raw[min(ii+1, winds.nx), jj, k1] +
                                         c3 * winds.w1_raw[ii, min(jj+1, winds.ny), k1] + c4 * winds.w1_raw[min(ii+1, winds.nx), min(jj+1, winds.ny), k1]
                                w1_k2 = c1 * winds.w1_raw[ii, jj, k2] + c2 * winds.w1_raw[min(ii+1, winds.nx), jj, k2] +
                                         c3 * winds.w1_raw[ii, min(jj+1, winds.ny), k2] + c4 * winds.w1_raw[min(ii+1, winds.nx), min(jj+1, winds.ny), k2]
                                w2_k1 = c1 * winds.w2_raw[ii, jj, k1] + c2 * winds.w2_raw[min(ii+1, winds.nx), jj, k1] +
                                         c3 * winds.w2_raw[ii, min(jj+1, winds.ny), k1] + c4 * winds.w2_raw[min(ii+1, winds.nx), min(jj+1, winds.ny), k1]
                                w2_k2 = c1 * winds.w2_raw[ii, jj, k2] + c2 * winds.w2_raw[min(ii+1, winds.nx), jj, k2] +
                                         c3 * winds.w2_raw[ii, min(jj+1, winds.ny), k2] + c4 * winds.w2_raw[min(ii+1, winds.nx), min(jj+1, winds.ny), k2]
                                wk1 = rt1 * w1_k1 + rt2 * w2_k1
                                wk2 = rt1 * w1_k2 + rt2 * w2_k2
                                w_blend = wk1 * dz1 + wk2 * dz2
                                w_interp_val = winds.w_interp(x_met, y_met, z_sigma, tloc)
                                dsigma_est = (w_blend) * dt_sub
                                # Write CSV into outputs/ to keep separate from hourly diagnostics
                                base_dir = dirname(trace_filename)
                                diag_dir = joinpath(base_dir, "outputs")
                                if !isdir(diag_dir)
                                    mkpath(diag_dir)
                                end
                                diag_path = joinpath(diag_dir, "w_diag_step_particle1.csv")
                                if !isfile(diag_path)
                                    open(diag_path, "w") do io
                                        println(io, "time_s,x_met,y_met,z_sigma,i,j,k1,k2,v1,v2,rt1,rt2,dz1,dz2,w1_k1,w1_k2,w2_k1,w2_k2,wk1,wk2,w_blend,w_interp,dsigma_est")
                                    end
                                end
                                open(diag_path, "a") do io
                                    println(io, join([current_time, x_met, y_met, z_sigma,
                                                      ii, jj, k1, k2, v1, v2, rt1, rt2, dz1, dz2,
                                                      w1_k1, w1_k2, w2_k1, w2_k2, wk1, wk2, w_blend, w_interp_val, dsigma_est], ","))
                                end
                            end
                        end
                    end

                    n_deposited = integrate_timestep!(state, winds, dt_sub,
                                                     particle_size_config, deposition_config,
                                                     decay_params, config,
                                                     diffusion_config=diffusion_config,
                                                     hanna_config=hanna_config,
                                                     advection_enabled=advection_enabled,
                                                     settling_enabled=settling_enabled,
                                                     dry_enabled=dry_deposition_enabled,
                                                     wet_enabled=wet_deposition_enabled,
                                                     current_time_global=current_time,
                                                     local_time_offset=local_time,
                                                     numerical_config=numerical_config,
                                                     trace_filename=trace_filename,
                                                     is_era5=isa(met_format, ERA5Format),
                                                     output_config=config.output_config)

                    current_time += dt_sub
                    local_time += dt_sub  # Advance local time within met window
                    # Note: state.current_time uses custom DateTime type, skip updating for now
                    state.timestep += 1

                    # FIXED: Count active particles directly instead of subtracting
                    # The old approach (n_active_before - n_deposited) was inaccurate because
                    # particles can become inactive for reasons other than deposition tracking
                    n_active_after = count(is_active(p) for p in state.ensemble.particles)

                    # Accumulate concentration at EVERY timestep (not just at snapshots)
                    accumulate_concentration!(state.fields, state.ensemble, state.domain, winds, dt_sub,
                                            use_trilinear=config.use_trilinear_gridding)

                    if config.verbose
                        @printf("  T+%.1fh: %d active, %d deposited, total_dep=%.2e Bq\n",
                               current_time/3600, n_active_after, n_deposited,
                               sum(state.total_deposited))
                    end

                    # Save snapshot if configured
                    if config.save_snapshots && current_time >= next_snapshot_time
                        active_positions = [state.ensemble.positions[i] for (i,p) in enumerate(state.ensemble.particles) if is_active(p)]

                        snapshot = SimulationSnapshot(
                            current_time,
                            active_positions,
                            copy(state.fields.atm_conc),
                            copy(state.fields.dry_deposition),
                            copy(state.fields.wet_deposition),
                            copy(state.fields.total_deposition),
                            n_active_after
                        )
                        push!(snapshots, snapshot)

                        if !isempty(snapshot_times)
                            snapshot_idx += 1
                            if snapshot_idx <= length(snapshot_times)
                                next_snapshot_time = snapshot_times[snapshot_idx]
                            else
                                next_snapshot_time = Inf  # No more snapshots
                            end
                        else
                            next_snapshot_time += config.dt_output
                        end
                    end
                end

                if all_particles_done
                    break
                end
            end
    end

    if config.verbose
        println()
        println("="^70)
        println("SIMULATION COMPLETE")
        println("="^70)
        println()
        @printf("Final time: T+%.1fh\n", current_time/3600)
        @printf("Total deposited: %.2e Bq\n", sum(state.total_deposited))
        @printf("Snapshots saved: %d\n", length(snapshots))
        println()

        # DEBUG: Show deposition by component/size class
        if !isempty(particle_size_config.size_bins)
            println("Deposition by size class:")
            for (ix, bin) in enumerate(particle_size_config.size_bins)
                if ix <= size(state.fields.dry_deposition, 3)
                    deposited = sum(state.fields.dry_deposition[:, :, ix])
                    @printf("  Size %6.1f μm: %.2e Bq\n", bin.diameter_μm, deposited)
                end
            end
            println()
        end
    end

    return snapshots
end

# Export public API
export OutputConfig, TraceFrequency, Verbosity
export TRACE_EVERY_TIMESTEP, TRACE_HOURLY, TRACE_DISABLED
export VERBOSITY_QUIET, VERBOSITY_NORMAL, VERBOSITY_DEBUG
export SimulationConfig, DepositionConfig, TurbulentDiffusionConfig, HannaTurbulenceConfig, ParticleSizeConfig, SimulationSnapshot
export run_simulation!, integrate_timestep!
