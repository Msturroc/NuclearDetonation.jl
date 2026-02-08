# Tests for simulation.jl and timestepping.jl: Main Simulation Integration
# Integration test for full simulation

using Test
using Random
using StaticArrays

# Import simulation modules
using NuclearDetonation.Transport:
    # Date/time
    DateTime, Duration, add_duration,
    # Particles
    Particle, is_active, get_rad, set_rad!,
    # Release
    ReleaseSource, ColumnRelease, BombRelease,
    # Decay
    DecayParams, NoDecay, ExponentialDecay,
    # Deposition
    DryDepositionParams, WetDepositionParams, GRASSLAND, SUMMER,
    # Random walk
    RandomWalkParams,
    # VGrav
    VGravTables, ParticleProperties, build_vgrav_tables,
    # Simulation structures
    SimulationDomain, ParticleEnsemble, ConcentrationField, SimulationState,
    add_particle!, remove_inactive_particles!,
    initialize_simulation, accumulate_concentration!, clear_concentration!,
    # Time stepping
    TimeSteppingParams, PhysicsParams,
    # Wind fields (needed for accumulate_concentration!)
    MeteoFields, WindFields, create_wind_interpolants

@testset "simulation: Main Simulation Integration" begin

    @testset "Simulation Domain" begin
        # Create simple domain
        nx, ny, nz = 10, 10, 5
        dx, dy = 1000.0, 1000.0  # 1 km grid
        hlevel = [0.0, 100.0, 500.0, 1000.0, 2000.0]
        xm = ones(Float64, nx, ny)
        ym = ones(Float64, nx, ny)

        t_start = DateTime(2025, 1, 1, 0)
        t_end = DateTime(2025, 1, 1, 6)
        dt_output = Duration(0, 1, 0, 0)  # 1 hour
        dt_met = Duration(0, 1, 0, 0)  # 1 hour

        domain = SimulationDomain(nx, ny, nz, dx, dy, hlevel, xm, ym,
                                 t_start, t_end, dt_output, dt_met)

        @test domain.nx == 10
        @test domain.ny == 10
        @test domain.nz == 5
        @test domain.dx == 1000.0
        @test length(domain.hlevel) == 5

        # Invalid domain — negative nx causes ArgumentError in fill() before @assert
        @test_throws ArgumentError SimulationDomain(
            -1, ny, nz, dx, dy, hlevel, xm, ym,
            t_start, t_end, dt_output, dt_met
        )
    end

    @testset "Particle Ensemble" begin
        ensemble = ParticleEnsemble{Float64}(2, ["Cs137", "I131"])

        @test length(ensemble.particles) == 0
        @test ensemble.ncomponents == 2
        @test ensemble.component_names == ["Cs137", "I131"]

        # Add particle
        pos = SVector{3,Float64}(5.0, 5.0, 2.0)
        vel = SVector{3,Float64}(1.0, 0.0, 0.0)
        mass = [1e10, 5e9]
        age = 0.0

        add_particle!(ensemble, pos, vel, mass, age)

        @test length(ensemble.particles) == 1
        @test length(ensemble.positions) == 1
        @test ensemble.positions[1] == pos
        @test get_rad(ensemble.particles[1], 1) == 1e10
        @test get_rad(ensemble.particles[1], 2) == 5e9
    end

    @testset "Remove Inactive Particles" begin
        ensemble = ParticleEnsemble{Float64}(1, ["Cs137"])

        # Add 3 particles
        for i in 1:3
            pos = SVector{3,Float64}(Float64(i), Float64(i), 1.0)
            vel = SVector{3,Float64}(0.0, 0.0, 0.0)
            mass = [1e10]
            add_particle!(ensemble, pos, vel, mass, 0.0)
        end

        @test length(ensemble.particles) == 3

        # Set middle particle radioactivity to zero
        set_rad!(ensemble.particles[2], 1, 0.0)  # Zero mass
        @test !is_active(ensemble.particles[2])  # No longer active since radioactivity is zero

        # Remove inactive particles (the one with zero radioactivity)
        n_removed = remove_inactive_particles!(ensemble)
        @test n_removed == 1  # Should remove the particle with zero radioactivity
        @test length(ensemble.particles) == 2  # Now only 2 particles remaining
    end

    @testset "Concentration Field" begin
        nx, ny, nz, ncomp = 10, 10, 5, 2
        fields = ConcentrationField{Float64}(nx, ny, nz, ncomp)

        @test size(fields.atm_conc) == (nx, ny, nz, ncomp)
        @test size(fields.surf_conc) == (nx, ny, ncomp)
        @test size(fields.dry_deposition) == (nx, ny, ncomp)
        @test all(fields.atm_conc .== 0.0)
        @test all(fields.total_deposition .== 0.0)
    end

    @testset "Initialize Simulation" begin
        # Create domain
        nx, ny, nz = 10, 10, 5
        dx, dy = 1000.0, 1000.0
        hlevel = [0.0, 100.0, 500.0, 1000.0, 2000.0]
        xm = ones(Float64, nx, ny)
        ym = ones(Float64, nx, ny)

        t_start = DateTime(2025, 1, 1, 0)
        t_end = DateTime(2025, 1, 1, 6)
        dt_output = Duration(0, 1, 0, 0)
        dt_met = Duration(0, 1, 0, 0)

        domain = SimulationDomain(nx, ny, nz, dx, dy, hlevel, xm, ym,
                                 t_start, t_end, dt_output, dt_met)

        # Create release sources
        geom = ColumnRelease(0.0, 1000.0)
        prof = BombRelease(0.0)
        activity = [1e15]
        source = ReleaseSource((5.0, 5.0), geom, prof, activity, 100)
        sources = [source]

        # Create decay parameters
        component_names = ["Cs137"]
        decay_params = [DecayParams{Float64}(kdecay=ExponentialDecay, halftime_hours=30.17 * 365.25 * 24.0)]

        # Initialize
        state = initialize_simulation(domain, sources, component_names, decay_params)

        @test state.current_time == t_start
        @test state.timestep == 0
        @test length(state.ensemble.particles) == 0
        @test length(state.total_released) == 1
        @test state.total_released[1] == 0.0
    end

    @testset "Accumulate Concentration" begin
        # Simple test of concentration accumulation
        nx, ny, nz = 10, 10, 5
        nk = 5
        dx, dy = 1000.0, 1000.0
        hlevel = [0.0, 100.0, 500.0, 1000.0, 2000.0]
        xm = ones(Float64, nx, ny)
        ym = ones(Float64, nx, ny)

        t_start = DateTime(2025, 1, 1, 0)
        t_end = DateTime(2025, 1, 1, 6)
        dt_output = Duration(0, 1, 0, 0)
        dt_met = Duration(0, 1, 0, 0)

        domain = SimulationDomain(nx, ny, nz, dx, dy, hlevel, xm, ym,
                                 t_start, t_end, dt_output, dt_met)

        ensemble = ParticleEnsemble{Float64}(1, ["Cs137"])
        fields = ConcentrationField{Float64}(nx, ny, nz, 1)

        # Create mock MeteoFields and WindFields for accumulate_concentration!
        met = MeteoFields(nx, ny, nk; T=Float32)
        met.xm .= 1.0f0
        met.ym .= 1.0f0
        met.garea .= 1.0f0
        met.blevel .= 1.0f0
        met.vlevel .= collect(Float32, LinRange(0.0, 1.0, nk))
        met.bhalf .= 1.0f0
        met.vhalf .= collect(Float32, LinRange(0.0, 1.0, nk + 1))
        met.ps1 .= 1013.0f0
        met.ps2 .= 1013.0f0
        winds = create_wind_interpolants(met, 0.0, 3600.0)

        # Add particle at centre (sigma=0.95 near surface)
        pos = SVector{3,Float64}(5.5, 5.5, 0.95)
        vel = SVector{3,Float64}(0.0, 0.0, 0.0)
        mass = [1e10]
        add_particle!(ensemble, pos, vel, mass, 0.0)

        # Accumulate
        dt = 600.0
        accumulate_concentration!(fields, ensemble, domain, winds, dt)

        # Check that concentration is non-zero somewhere near particle location
        @test sum(fields.atm_conc) > 0.0

        # Check dose accumulated
        @test sum(fields.dose) > 0.0

        # Clear and check
        clear_concentration!(fields)
        @test all(fields.atm_conc .== 0.0)
        @test all(fields.surf_conc .== 0.0)
        @test sum(fields.dose) > 0.0  # Dose is not cleared
    end

    @testset "Time Stepping Parameters" begin
        dt = 600.0  # 10 minutes
        nsteps = 36  # 6 hours

        params = TimeSteppingParams(dt, nsteps)
        @test params.dt == 600.0
        @test params.nsteps == 36
        @test params.enable_turbulence == true
        @test params.enable_settling == true
        @test params.enable_decay == true

        # Custom parameters
        params_custom = TimeSteppingParams(dt, nsteps,
                                          advection_substeps=5,
                                          enable_turbulence=false,
                                          enable_decay=false)
        @test params_custom.advection_substeps == 5
        @test params_custom.enable_turbulence == false
        @test params_custom.enable_decay == false
    end

    @testset "Physics Parameters Construction" begin
        # Build vgrav tables
        particle_props = [ParticleProperties(diameter_μm=1e-6, density_gcm3=2500.0)]
        vgrav_tables = build_vgrav_tables(particle_props)

        # Decay params
        decay_params = [DecayParams{Float64}(kdecay=NoDecay)]

        # Dry deposition
        dry_dep_params = DryDepositionParams(
            1e-6, 2500.0, 30.0,
            fill(0.1, 10, 10),
            fill(GRASSLAND, 10, 10)
        )

        # Wet deposition
        wet_dep_params = WetDepositionParams(1e-6, 2500.0)

        # Random walk
        rwalk_params = RandomWalkParams(timestep=1.0, tmix_vertical=100.0, tmix_horizontal=1000.0)

        # Physics params
        physics = PhysicsParams(
            vgrav_tables,
            decay_params,
            dry_dep_params,
            wet_dep_params,
            rwalk_params,
            1000.0
        )

        @test physics.boundary_layer_height == 1000.0
        @test length(physics.decay_params) == 1
    end

    @testset "Integration Test: Simple Simulation" begin
        # This is a minimal end-to-end test
        # Full integration tests would use real meteorological data

        # Set up domain
        nx, ny, nz = 20, 20, 5
        dx, dy = 1000.0, 1000.0
        hlevel = [0.0, 100.0, 500.0, 1000.0, 2000.0]
        xm = ones(Float64, nx, ny)
        ym = ones(Float64, nx, ny)

        t_start = DateTime(2025, 1, 1, 0)
        t_end = DateTime(2025, 1, 1, 1)  # 1 hour
        dt_output = Duration(0, 1, 0, 0)
        dt_met = Duration(0, 1, 0, 0)

        domain = SimulationDomain(nx, ny, nz, dx, dy, hlevel, xm, ym,
                                 t_start, t_end, dt_output, dt_met)

        # Create source
        geom = ColumnRelease(0.0, 500.0)
        prof = BombRelease(0.0)  # Release at t=0
        activity = [1e15]  # 1 PBq
        source = ReleaseSource((10.0, 10.0), geom, prof, activity, 50)

        # Initialize simulation
        component_names = ["Cs137"]
        half_life = 30.17 * 365.25 * 24 * 3600  # seconds
        decay_params = [DecayParams{Float64}(kdecay=ExponentialDecay, halftime_hours=half_life/3600.0)]

        state = initialize_simulation(domain, [source], component_names, decay_params)

        @test state.timestep == 0
        @test length(state.ensemble.particles) == 0

        # Simulate initial release (simplified - no met fields)
        # In production, would call step!() with proper physics
        # For now, just verify state structure is correct

        @test isa(state, SimulationState)
        @test state.domain.nx == nx
        @test state.ensemble.ncomponents == 1
        @test size(state.fields.atm_conc) == (nx, ny, nz, 1)
    end
end

println("✓ All simulation integration tests passed!")
