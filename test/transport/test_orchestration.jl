# Tests for orchestration.jl: config defaults, trace scheduling, verbosity helpers

using Test

const _Tx = NuclearDetonation.Transport

@testset "orchestration: config and helpers" begin

    @testset "OutputConfig defaults" begin
        oc = _Tx.OutputConfig()
        @test oc.trace_frequency == _Tx.TRACE_EVERY_TIMESTEP
        @test oc.verbosity == _Tx.VERBOSITY_NORMAL
        @test oc.trace_enabled == true
        @test oc.progress_interval_hours > 0.0
        @test oc.settling_diagnostic_interval_hours >= 0.0

        # is_quiet / is_verbose / is_debug
        @test _Tx.is_quiet(_Tx.OutputConfig(verbosity=_Tx.VERBOSITY_QUIET))
        @test _Tx.is_verbose(_Tx.OutputConfig(verbosity=_Tx.VERBOSITY_NORMAL))
        @test _Tx.is_debug(_Tx.OutputConfig(verbosity=_Tx.VERBOSITY_DEBUG))
        @test !_Tx.is_quiet(_Tx.OutputConfig(verbosity=_Tx.VERBOSITY_DEBUG))
    end

    @testset "SimulationConfig defaults" begin
        sc = _Tx.SimulationConfig{Float64}()
        @test sc.dt_output > 0.0
        @test sc.dt_met > 0.0
        @test sc.reltol > 0.0 && sc.abstol > 0.0
        @test sc.reltol > sc.abstol  # tolerances ordered as expected
        @test sc.dt_particle > 0.0
        @test sc.save_snapshots == true
        @test sc.max_files == 0
        @test sc.max_duration == 0.0
        @test isa(sc.output_config, _Tx.OutputConfig)
        # use_reference_stepping default off (production uses ODE solver)
        @test sc.use_reference_stepping == false
    end

    @testset "DepositionConfig defaults" begin
        dc = _Tx.DepositionConfig{Float64}()
        @test dc.mixing_height > 0.0
        @test dc.surface_roughness > 0.0
        @test dc.friction_velocity > 0.0
        @test dc.monin_obukhov_length > 0.0
        @test dc.season == SUMMER
        @test dc.apply_dry_deposition == true
        @test dc.apply_wet_deposition == false
        @test dc.simple_deposition_velocity > 0.0
        @test dc.wet_deposition_precip_threshold >= 0.0
        @test isnothing(dc.land_use_map)
        @test isnothing(dc.roughness_length_map)
    end

    @testset "ParticleSizeConfig defaults" begin
        psc = _Tx.ParticleSizeConfig()
        @test isempty(psc.size_bins)
        @test isnothing(psc.vgrav_tables)
        @test isempty(psc.particle_radii)
        @test isempty(psc.particle_size_indices)
        @test isnothing(psc.fixed_gravity_cm_s)
    end

    @testset "should_write_trace dispatches on trace_frequency" begin
        # Disabled trace: never writes
        oc_off = _Tx.OutputConfig(trace_enabled=false)
        @test _Tx.should_write_trace(oc_off, 600.0, 60.0) == false

        oc_disabled = _Tx.OutputConfig(trace_frequency=_Tx.TRACE_DISABLED)
        @test _Tx.should_write_trace(oc_disabled, 600.0, 60.0) == false

        # Every-timestep: always writes
        oc_every = _Tx.OutputConfig(trace_frequency=_Tx.TRACE_EVERY_TIMESTEP)
        @test _Tx.should_write_trace(oc_every, 600.0, 60.0) == true
        @test _Tx.should_write_trace(oc_every, 0.0, 60.0) == true

        # Hourly: writes only when crossing an hour boundary
        oc_hourly = _Tx.OutputConfig(trace_frequency=_Tx.TRACE_HOURLY)
        # Stepping from t=3540 to t=3600 crosses 1-hour boundary
        @test _Tx.should_write_trace(oc_hourly, 3600.0, 60.0) == true
        # Stepping from t=3000 to t=3060 stays inside the same hour
        @test _Tx.should_write_trace(oc_hourly, 3060.0, 60.0) == false
        # Stepping from t=7140 to t=7200 crosses the 2-hour boundary
        @test _Tx.should_write_trace(oc_hourly, 7200.0, 60.0) == true
    end

    @testset "get_verbosity respects legacy verbose flag" begin
        # verbose=false short-circuits to QUIET regardless of output_config
        sc_quiet = _Tx.SimulationConfig{Float64}(verbose=false)
        @test _Tx.get_verbosity(sc_quiet) == _Tx.VERBOSITY_QUIET
        @test _Tx.should_print(sc_quiet) == false
        @test _Tx.should_print_debug(sc_quiet) == false

        # verbose=true defers to output_config
        sc_normal = _Tx.SimulationConfig{Float64}(verbose=true,
            output_config=_Tx.OutputConfig(verbosity=_Tx.VERBOSITY_NORMAL))
        @test _Tx.get_verbosity(sc_normal) == _Tx.VERBOSITY_NORMAL
        @test _Tx.should_print(sc_normal) == true
        @test _Tx.should_print_debug(sc_normal) == false

        sc_debug = _Tx.SimulationConfig{Float64}(verbose=true,
            output_config=_Tx.OutputConfig(verbosity=_Tx.VERBOSITY_DEBUG))
        @test _Tx.get_verbosity(sc_debug) == _Tx.VERBOSITY_DEBUG
        @test _Tx.should_print_debug(sc_debug) == true
    end
end

println("✓ All orchestration tests passed!")
