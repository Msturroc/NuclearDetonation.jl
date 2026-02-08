# Tests for random_walk.jl: Random Walk Turbulent Diffusion
# Based on SNAP rwalk.f90

using Test
using StochasticDiffEq
using Statistics

# Import necessary types from the main module
using NuclearDetonation.Transport:
    RandomWalkParams, RandomWalkState, initialize_random_walk,
    horizontal_diffusion_length, vertical_diffusion_coefficient,
    random_walk_noise!, apply_boundary_layer_reflection!,
    create_random_walk_sde_problem, particle_velocity_with_rwalk!,
    MeteoFields, WindFields, ParticleParams,
    create_wind_interpolants, particle_velocity!

@testset "random_walk: Turbulent Diffusion" begin

    @testset "RandomWalkParams Construction" begin
        # Default parameters
        params = RandomWalkParams(timestep=600.0)
        @test params.timestep == 600.0
        @test params.tmix_vertical == 900.0
        @test params.tmix_horizontal == 900.0
        @test params.lmax == 0.28
        @test params.labove == 0.03
        @test params.entrainment == 0.1
        @test params.hmax == 2500.0
        @test params.blfullmix == false
        @test params.horizontal_a_bl == 0.5
        @test params.horizontal_a_above == 0.25
        @test params.horizontal_b == 0.875
    end

    @testset "Initialize Random Walk State" begin
        params = RandomWalkParams(timestep=600.0)
        state = initialize_random_walk(params)

        # Check computed factors
        @test state.tfactor_v ≈ 600.0 / 900.0
        @test state.tsqrtfactor_v ≈ sqrt(600.0 / 900.0)
        @test state.tfactor_h ≈ 600.0 / 900.0
        @test state.tsqrtfactor_h ≈ sqrt(600.0 / 900.0)
        @test state.vrdbla ≈ 0.03 * sqrt(600.0 / 900.0)

        # Different timestep
        params2 = RandomWalkParams(timestep=300.0)
        state2 = initialize_random_walk(params2)
        @test state2.tfactor_v < state.tfactor_v  # Smaller timestep
        @test state2.tsqrtfactor_v < state.tsqrtfactor_v
    end

    @testset "Horizontal Diffusion Length" begin
        params = RandomWalkParams(timestep=600.0)
        state = initialize_random_walk(params)

        # In boundary layer (z=0.9, tbl=0.8)
        rl_bl = horizontal_diffusion_length(10.0, 5.0, 0.9, 0.8, params, state)
        @test rl_bl > 0.0

        # Above boundary layer (z=0.5, tbl=0.8)
        rl_above = horizontal_diffusion_length(10.0, 5.0, 0.5, 0.8, params, state)
        @test rl_above > 0.0

        # In BL should have larger coefficient (a=0.5 vs 0.25)
        @test rl_bl > rl_above

        # Should scale with wind speed
        rl_fast = horizontal_diffusion_length(20.0, 10.0, 0.9, 0.8, params, state)
        @test rl_fast > rl_bl  # Faster wind → larger diffusion

        # Zero wind
        rl_zero = horizontal_diffusion_length(0.0, 0.0, 0.9, 0.8, params, state)
        @test rl_zero ≈ 0.0 atol=1e-10
    end

    @testset "Vertical Diffusion Coefficient" begin
        params = RandomWalkParams(timestep=600.0)
        state = initialize_random_walk(params)

        # Above boundary layer (z=0.5, tbl=0.8)
        σ_above = vertical_diffusion_coefficient(0.5, 0.8, params, state)
        @test σ_above ≈ state.vrdbla  # Should use vrdbla

        # In boundary layer (z=0.9, tbl=0.8)
        σ_bl = vertical_diffusion_coefficient(0.9, 0.8, params, state)
        @test σ_bl > σ_above  # Larger in BL

        # Should scale with mixing height
        σ_shallow = vertical_diffusion_coefficient(0.95, 0.9, params, state)  # Shallow BL
        σ_deep = vertical_diffusion_coefficient(0.9, 0.7, params, state)  # Deep BL
        @test σ_deep > σ_shallow  # Deeper BL → more diffusion

        # At boundary layer top
        σ_top = vertical_diffusion_coefficient(0.8, 0.8, params, state)
        @test σ_top > 0.0
    end

    @testset "Random Walk Noise Function" begin
        # Create simple wind field
        # First create the arrays with proper increasing coordinate arrays
        u1 = fill(10.0f0, 10, 10, 5)
        u2 = fill(10.0f0, 10, 10, 5)
        v1 = fill(5.0f0, 10, 10, 5)
        v2 = fill(5.0f0, 10, 10, 5)
        w1 = zeros(Float32, 10, 10, 5)
        w2 = zeros(Float32, 10, 10, 5)
        t1 = fill(288.0f0, 10, 10, 5)
        t2 = fill(288.0f0, 10, 10, 5)
        t1_abs = nothing
        t2_abs = nothing
        ps1 = fill(1013.0f0, 10, 10)
        ps2 = fill(1013.0f0, 10, 10)
        pmsl1 = zeros(Float32, 10, 10)
        pmsl2 = zeros(Float32, 10, 10)
        hbl1 = zeros(Float32, 10, 10)
        hbl2 = zeros(Float32, 10, 10)
        bl1 = zeros(Float32, 10, 10)
        bl2 = zeros(Float32, 10, 10)
        precip1 = zeros(Float32, 10, 10)
        precip2 = zeros(Float32, 10, 10)
        xm = ones(Float32, 10, 10)
        ym = ones(Float32, 10, 10)
        garea = ones(Float32, 10, 10)
        alevel = zeros(Float32, 5)
        blevel = ones(Float32, 5)
        vlevel = collect(Float32, LinRange(0.0, 1.0, 5))  # Properly increasing coordinates
        ahalf = zeros(Float32, 5)
        bhalf = ones(Float32, 5)
        vhalf = collect(Float32, LinRange(0.0, 1.0, 5))  # Properly increasing coordinates
        hlevel1 = zeros(Float32, 10, 10, 5)
        hlevel2 = zeros(Float32, 10, 10, 5)
        hlayer1 = zeros(Float32, 10, 10, 5)
        hlayer2 = zeros(Float32, 10, 10, 5)

        # Create MeteoFields using positional constructor
        met_fields = MeteoFields{Float32}(
            10, 10, 5,  # nx, ny, nk
            u1, u2, v1, v2, w1, w2, t1, t2,
            t1_abs, t2_abs,
            ps1, ps2, pmsl1, pmsl2, hbl1, hbl2, bl1, bl2, precip1, precip2,
            xm, ym, garea,
            alevel, blevel, vlevel, ahalf, bhalf, vhalf,
            hlevel1, hlevel2, hlayer1, hlayer2
        )

        winds = create_wind_interpolants(met_fields, 0.0, 3600.0)
        particle_params = ParticleParams(grav_type=0)
        rwalk_params = RandomWalkParams(timestep=600.0)
        rwalk_state = initialize_random_walk(rwalk_params)
        tbl = 0.8

        p = (winds, particle_params, rwalk_params, rwalk_state, tbl)

        # Test noise coefficients
        u = [5.0, 5.0, 0.9]  # In boundary layer
        du = zeros(3)
        random_walk_noise!(du, u, p, 0.0)

        @test all(du .> 0.0)  # All noise coefficients positive
        @test du[1] > 0.0  # Horizontal x noise
        @test du[2] > 0.0  # Horizontal y noise
        @test du[3] > 0.0  # Vertical noise

        # Above boundary layer should have smaller vertical noise
        u_above = [5.0, 5.0, 0.5]
        du_above = zeros(3)
        random_walk_noise!(du_above, u_above, p, 0.0)
        @test du_above[3] < du[3]  # Less vertical diffusion above BL
    end

    @testset "Boundary Layer Reflection" begin
        # Mock integrator structure
        mutable struct MockIntegrator
            u::Vector{Float64}
            p::Tuple
        end

        params = RandomWalkParams(timestep=600.0)
        state = initialize_random_walk(params)
        tbl = 0.8

        # Create mock wind fields (not used in reflection)
        # First create the arrays
        u1 = zeros(Float32, 10, 10, 5)
        u2 = zeros(Float32, 10, 10, 5)
        v1 = zeros(Float32, 10, 10, 5)
        v2 = zeros(Float32, 10, 10, 5)
        w1 = zeros(Float32, 10, 10, 5)
        w2 = zeros(Float32, 10, 10, 5)
        t1 = zeros(Float32, 10, 10, 5)
        t2 = zeros(Float32, 10, 10, 5)
        t1_abs = nothing
        t2_abs = nothing
        ps1 = zeros(Float32, 10, 10)
        ps2 = zeros(Float32, 10, 10)
        pmsl1 = zeros(Float32, 10, 10)
        pmsl2 = zeros(Float32, 10, 10)
        hbl1 = zeros(Float32, 10, 10)
        hbl2 = zeros(Float32, 10, 10)
        bl1 = zeros(Float32, 10, 10)
        bl2 = zeros(Float32, 10, 10)
        precip1 = zeros(Float32, 10, 10)
        precip2 = zeros(Float32, 10, 10)
        xm = ones(Float32, 10, 10)
        ym = ones(Float32, 10, 10)
        garea = ones(Float32, 10, 10)
        alevel = zeros(Float32, 5)
        blevel = ones(Float32, 5)
        vlevel = collect(Float32, LinRange(0.0, 1.0, 5))  # Properly increasing coordinates
        ahalf = zeros(Float32, 5)
        bhalf = ones(Float32, 5)
        vhalf = collect(Float32, LinRange(0.0, 1.0, 5))  # Properly increasing coordinates
        hlevel1 = zeros(Float32, 10, 10, 5)
        hlevel2 = zeros(Float32, 10, 10, 5)
        hlayer1 = zeros(Float32, 10, 10, 5)
        hlayer2 = zeros(Float32, 10, 10, 5)

        # Create MeteoFields using positional constructor
        met_fields = MeteoFields{Float32}(
            10, 10, 5,  # nx, ny, nk
            u1, u2, v1, v2, w1, w2, t1, t2,
            t1_abs, t2_abs,
            ps1, ps2, pmsl1, pmsl2, hbl1, hbl2, bl1, bl2, precip1, precip2,
            xm, ym, garea,
            alevel, blevel, vlevel, ahalf, bhalf, vhalf,
            hlevel1, hlevel2, hlayer1, hlayer2
        )
        winds = create_wind_interpolants(met_fields, 0.0, 3600.0)
        particle_params = ParticleParams()

        p = (winds, particle_params, params, state, tbl)

        # Test reflection from surface (z > 1.0)
        integrator = MockIntegrator([5.0, 5.0, 1.05], p)
        apply_boundary_layer_reflection!(integrator)
        @test integrator.u[3] < 1.0  # Should be reflected back
        @test integrator.u[3] ≈ 2.0 - 1.05  # z_new = 2.0 - z_old

        # Test reflection from BL top (z < top_entrainment)
        bl_entrainment_thickness = (1.0 - tbl) * (1.0 + params.entrainment)
        top_entrainment = 1.0 - bl_entrainment_thickness
        integrator2 = MockIntegrator([5.0, 5.0, top_entrainment - 0.05], p)
        apply_boundary_layer_reflection!(integrator2)
        @test integrator2.u[3] > top_entrainment  # Should be reflected back

        # No reflection if in valid range
        integrator3 = MockIntegrator([5.0, 5.0, 0.9], p)
        z_before = integrator3.u[3]
        apply_boundary_layer_reflection!(integrator3)
        @test integrator3.u[3] == z_before  # Should not change

        # Above boundary layer: no reflection
        integrator4 = MockIntegrator([5.0, 5.0, 0.5], p)
        z_above = integrator4.u[3]
        apply_boundary_layer_reflection!(integrator4)
        @test integrator4.u[3] == z_above  # Should not change
    end

    @testset "Physical Consistency" begin
        params = RandomWalkParams(timestep=600.0)
        state = initialize_random_walk(params)

        # Diffusion should increase with timestep
        params_fast = RandomWalkParams(timestep=1200.0)
        state_fast = initialize_random_walk(params_fast)
        @test state_fast.tsqrtfactor_v > state.tsqrtfactor_v
        @test state_fast.vrdbla > state.vrdbla

        # Horizontal diffusion proportional to wind speed^0.875
        u1, v1 = 10.0, 0.0
        u2, v2 = 20.0, 0.0
        rl1 = horizontal_diffusion_length(u1, v1, 0.9, 0.8, params, state)
        rl2 = horizontal_diffusion_length(u2, v2, 0.9, 0.8, params, state)
        @test rl2 / rl1 ≈ 2.0^0.875 rtol=0.01

        # Vertical diffusion scales with BL depth
        σ_shallow = vertical_diffusion_coefficient(0.95, 0.9, params, state)
        σ_deep = vertical_diffusion_coefficient(0.9, 0.7, params, state)
        @test σ_deep / σ_shallow ≈ (1.0 - 0.7) / (1.0 - 0.9) rtol=0.01
    end

    @testset "Parameter Limits" begin
        params = RandomWalkParams(timestep=600.0)
        state = initialize_random_walk(params)

        # Very shallow boundary layer
        σ_min = vertical_diffusion_coefficient(0.99, 0.98, params, state)
        @test σ_min > 0.0
        @test σ_min < 0.1  # Should be small

        # Very deep boundary layer
        σ_max = vertical_diffusion_coefficient(0.9, 0.1, params, state)
        @test σ_max > σ_min  # Larger for deeper BL

        # Zero wind case
        rl_zero = horizontal_diffusion_length(0.0, 0.0, 0.9, 0.8, params, state)
        @test rl_zero == 0.0

        # Very high wind
        rl_extreme = horizontal_diffusion_length(100.0, 50.0, 0.9, 0.8, params, state)
        @test isfinite(rl_extreme)
        @test rl_extreme > 0.0
    end

    @testset "Time Scaling" begin
        # Test that diffusion scales with sqrt(time)
        params_short = RandomWalkParams(timestep=300.0)
        params_long = RandomWalkParams(timestep=1200.0)

        state_short = initialize_random_walk(params_short)
        state_long = initialize_random_walk(params_long)

        # sqrt(1200/300) = sqrt(4) = 2
        @test state_long.tsqrtfactor_v / state_short.tsqrtfactor_v ≈ 2.0 rtol=0.01
        @test state_long.tsqrtfactor_h / state_short.tsqrtfactor_h ≈ 2.0 rtol=0.01
        @test state_long.vrdbla / state_short.vrdbla ≈ 2.0 rtol=0.01
    end
end

println("✓ All random walk tests passed!")
