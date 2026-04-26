# Tests for particle_dynamics.jl: WindFields construction and particle ODE RHS

using Test
using OrdinaryDiffEq

# Build a minimal MeteoFields with uniform synthetic winds
function _make_uniform_met(; nx=10, ny=10, nk=3, u=5.0, v=0.0, w=0.0)
    met = MeteoFields(nx, ny, nk; T=Float32)
    met.u1 .= Float32(u);  met.u2 .= Float32(u)
    met.v1 .= Float32(v);  met.v2 .= Float32(v)
    met.w1 .= Float32(w);  met.w2 .= Float32(w)
    met.t1 .= 288.15f0;    met.t2 .= 288.15f0
    met.ps1 .= 1013.0f0;   met.ps2 .= 1013.0f0
    met.p1 .= 1013.0f0;    met.p2 .= 1013.0f0
    met.vlevel .= collect(Float32, range(0.0, 1.0; length=nk))
    met.vhalf  .= collect(Float32, range(0.0, 1.0; length=nk + 1))
    met.blevel .= 1.0f0
    met.bhalf  .= 1.0f0
    met.garea  .= 1.0f0
    met.xm     .= 1.0f0
    met.ym     .= 1.0f0
    return met
end

@testset "particle_dynamics: WindFields and ODE RHS" begin

    @testset "create_wind_interpolants on uniform field" begin
        met = _make_uniform_met(; u=5.0, v=0.0, w=0.0)
        winds = create_wind_interpolants(met, 0.0, 3600.0;
                                         lon_min=0.0, lon_max=10.0,
                                         lat_min=0.0, lat_max=10.0)

        @test isa(winds, WindFields)
        @test winds.nx == 10
        @test winds.ny == 10
        @test winds.nk == 3

        # Trilinear interpolation of a uniform field returns the same value at any interior point.
        @test winds.u_interp(5.0, 5.0, 0.5, 1800.0) ≈ 5.0    atol=1e-5
        @test winds.v_interp(5.0, 5.0, 0.5, 1800.0) ≈ 0.0    atol=1e-5
        @test winds.w_interp(5.0, 5.0, 0.5, 1800.0) ≈ 0.0    atol=1e-5

        # Grid-spacing metadata reflects the configured lon/lat box at lat_min=0.
        R_earth = 6.371e6
        expected_dlon_rad = (10.0 / 9.0) * π / 180.0
        @test winds.dx_m ≈ R_earth * expected_dlon_rad rtol=1e-6
    end

    @testset "particle_velocity! converts m/s wind to grid-unit velocity" begin
        met = _make_uniform_met(; u=5.0, v=0.0, w=0.0)
        winds = create_wind_interpolants(met, 0.0, 3600.0;
                                         lon_min=0.0, lon_max=10.0,
                                         lat_min=0.0, lat_max=10.0)

        params = ParticleParams{Float64}()
        ode_p = NuclearDetonation.Transport.ParticleODEParams(winds, params)

        du = zeros(Float64, 3)
        u  = [5.0, 5.0, 0.5]
        particle_velocity!(du, u, ode_p, 1800.0)

        # particle_velocity! does: du[1] = u_wind * xm(y) / dx_m, du[2] = v_wind / dy_m, du[3] = w_wind
        u_wind = winds.u_interp(u[1], u[2], u[3], 1800.0)
        xm_at_y = winds.xm_interp(u[2])
        @test du[1] ≈ u_wind * xm_at_y / winds.dx_m  rtol=1e-8
        @test du[2] ≈ 0.0  atol=1e-12
        @test du[3] ≈ 0.0  atol=1e-12

        # Sanity: with positive u-wind, particle drifts in +x; finite, non-zero rate.
        @test du[1] > 0.0
        @test isfinite(du[1])
    end

    @testset "Wind interpolants clamp out-of-domain queries" begin
        met = _make_uniform_met(; u=7.5, v=-2.0, w=0.0)
        winds = create_wind_interpolants(met, 0.0, 3600.0;
                                         lon_min=0.0, lon_max=10.0,
                                         lat_min=0.0, lat_max=10.0)

        # Inside-domain reference value
        u_in = winds.u_interp(5.0, 5.0, 0.5, 1800.0)
        v_in = winds.v_interp(5.0, 5.0, 0.5, 1800.0)

        # Querying outside the [1, nx] / [1, ny] grid box must remain finite (extrapolate Flat).
        u_out = winds.u_interp(-50.0, 5.0, 0.5, 1800.0)
        v_out = winds.v_interp(5.0, 500.0, 0.5, 1800.0)
        @test isfinite(u_out) && isfinite(v_out)
        @test u_out ≈ u_in atol=1e-5
        @test v_out ≈ v_in atol=1e-5

        # Out-of-time queries clamp too.
        u_t_low  = winds.u_interp(5.0, 5.0, 0.5, -10000.0)
        u_t_high = winds.u_interp(5.0, 5.0, 0.5,  10_000_000.0)
        @test isfinite(u_t_low) && isfinite(u_t_high)
        @test u_t_low  ≈ u_in atol=1e-5
        @test u_t_high ≈ u_in atol=1e-5
    end

    @testset "create_particle_problem advects under uniform u-wind" begin
        u_ms = 10.0
        met = _make_uniform_met(; u=u_ms, v=0.0, w=0.0)
        winds = create_wind_interpolants(met, 0.0, 3600.0;
                                         lon_min=0.0, lon_max=10.0,
                                         lat_min=0.0, lat_max=10.0)

        params = ParticleParams{Float64}()
        x0 = [5.0, 5.0, 0.5]
        prob = create_particle_problem(x0, (0.0, 3600.0), winds, params)
        sol = solve(prob, Tsit5(); reltol=1e-8, abstol=1e-10)

        # Expected drift in grid units: u_wind * t * xm(y) / dx_m
        xm_at_y = winds.xm_interp(x0[2])
        expected_dx_grid = u_ms * 3600.0 * xm_at_y / winds.dx_m

        @test sol.retcode == ReturnCode.Success || sol.retcode == ReturnCode.Terminated
        # Accept the trajectory either ending at t=3600 (within domain) or earlier (callback fired).
        Δx = sol.u[end][1] - x0[1]
        Δy = sol.u[end][2] - x0[2]
        Δz = sol.u[end][3] - x0[3]
        @test Δx ≈ expected_dx_grid * (sol.t[end] / 3600.0) rtol=1e-2
        @test isapprox(Δy, 0.0; atol=1e-8)
        @test isapprox(Δz, 0.0; atol=1e-8)
    end
end

println("✓ All particle_dynamics tests passed!")
