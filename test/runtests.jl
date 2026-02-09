# NuclearDetonation.jl Test Suite
# Tests for both Glasstone weapon effects and atmospheric transport

using Test
using NuclearDetonation

# Import all Transport submodule components for direct access in tests
import NuclearDetonation.Transport:
    # Date/Time types
    DateTime, Duration, add_duration, datetime_diff, monthdays,
    # Particle types and functions
    Particle, ExtraParticle, scale_rad!, set_rad!, get_rad, add_rad!, get_set_rad!,
    is_active, inactivate!, flush_away_denormal!,
    # Coordinate transformation functions
    earthr, sph2rot!, pol2sph!, lam2sph!, mer2sph!, xyconvert!,
    # Map field functions
    mapfield,
    # Met reader components
    MeteoParams, MeteoFields, init_meteo_params!, load_netcdf_variable,
    unit_conversion_factor, read_meteo_timestep!,
    # VGrav tables components
    air_viscosity, air_density, cunningham_factor,
    vgrav_corrected,
    VGravTables, ParticleProperties,
    ReferenceViscosity, SutherlandViscosity,
    build_vgrav_tables, interpolate_vgrav,
    # Particle dynamics types
    WindFields, ParticleParams,
    create_wind_interpolants, particle_velocity!,
    create_particle_problem,
    # Decay types and functions
    DecayType, NoDecay, ExponentialDecay, BombDecay,
    DecayParams, BombDecayState,
    prepare_decay_rates!, apply_decay, apply_decay!,
    # Release / source term types
    ReleaseProfile, ConstantRelease, BombRelease, LinearRelease, StepRelease,
    ReleaseGeometry, ColumnRelease, CylinderRelease, MushroomCloudRelease,
    ReleaseSource, Plume,
    compute_release_cylinders, sample_cylinder_position,
    generate_release_particles, compute_release_rate,
    create_mushroom_cloud_from_yield,
    # Deposition types and functions
    LandUseClass, SeasonCategory,
    WATER, DECIDUOUS_FOREST, CONIFEROUS_FOREST, MIXED_FOREST,
    GRASSLAND, CROPLAND, URBAN, BARE_SOIL, SNOW_ICE,
    WINTER, SPRING, SUMMER, AUTUMN,
    DryDepositionParams, WetDepositionParams,
    compute_air_density, compute_friction_velocity, compute_monin_obukhov_length,
    aerodynamic_resistance, cunningham_slip_factor, brownian_diffusivity,
    get_characteristic_radius, surface_resistance,
    compute_dry_deposition_velocity, compute_wet_scavenging_coefficient,
    apply_dry_deposition!, apply_wet_deposition!,
    # Simulation and time-stepping
    SimulationDomain, ParticleEnsemble, ConcentrationField, SimulationState,
    add_particle!, remove_inactive_particles!,
    initialize_simulation, accumulate_concentration!, clear_concentration!,
    TimeSteppingParams, PhysicsParams

@testset "NuclearDetonation.jl" begin
    println("\n" * "="^70)
    println("Running NuclearDetonation.jl Test Suite")
    println("="^70)

    # Phase 1 Tests: Core Data Structures
    @testset "Transport Phase 1: Core Data Structures" begin
        include("transport/test_datetime.jl")
        include("transport/test_particles.jl")
        include("transport/test_dimensions.jl")
    end

    # Phase 2 Tests: Meteorological Interface
    @testset "Transport Phase 2: Meteorological Interface" begin
        include("transport/test_milib.jl")
    end

    # Phase 3 Tests: Particle Physics
    @testset "Transport Phase 3: Particle Physics" begin
        include("transport/test_vgravtables.jl")
        include("transport/test_decay.jl")
    end

    # Phase 4 Tests: Source Term Modelling
    @testset "Transport Phase 4: Source Term Modelling" begin
        include("transport/test_release.jl")
    end

    # Phase 5 Tests: Deposition
    @testset "Transport Phase 5: Deposition" begin
        include("transport/test_deposition.jl")
    end

    # Phase 6 Tests: Main Simulation Loop
    @testset "Transport Phase 6: Main Simulation Loop" begin
        include("transport/test_simulation.jl")
    end

    println("\n" * "="^70)
    println("All tests completed!")
    println("="^70)
end
