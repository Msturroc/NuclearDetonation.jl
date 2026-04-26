# Extra tests for particle_dynamics.jl: alternative interpolation orders and settling helpers.
# Reuses the _make_uniform_met fixture defined in test_particle_dynamics.jl.

using Test

const _Tx_pde = NuclearDetonation.Transport

# Build a fixture with realistic-ish hybrid half-level coefficients and a physically meaningful
# vertical pressure structure, so compute_settling_constant exercises a finite layer thickness.
function _make_settling_met(; nx=6, ny=6, nk=4, T_K=288.15, ps_hpa=1013.25)
    met = MeteoFields(nx, ny, nk; T=Float64)
    met.u1 .= 5.0;  met.u2 .= 5.0
    met.v1 .= 0.0;  met.v2 .= 0.0
    met.w1 .= 0.0;  met.w2 .= 0.0
    met.t1 .= T_K;  met.t2 .= T_K
    met.ps1 .= ps_hpa; met.ps2 .= ps_hpa

    # Half-level pressures from surface (1013 hPa) to TOA (50 hPa).
    p_half = collect(range(ps_hpa, 50.0; length=nk + 1))
    met.ahalf .= p_half  # pure pressure (b=0)
    met.bhalf .= 0.0
    # Mid-levels = arithmetic mean of bracketing half-levels.
    met.alevel .= [0.5 * (p_half[k] + p_half[k+1]) for k in 1:nk]
    met.blevel .= 0.0
    met.p1 .= reshape(met.alevel, 1, 1, nk)
    met.p2 .= reshape(met.alevel, 1, 1, nk)

    # Sigma-like vertical levels in ascending order (top → surface).
    met.vlevel .= collect(range(0.0, 1.0; length=nk))
    met.vhalf  .= collect(range(0.0, 1.0; length=nk + 1))

    met.garea .= 1.0
    met.xm    .= 1.0
    met.ym    .= 1.0
    return met
end

@testset "particle_dynamics extras" begin

    @testset "ReferenceTrilinearInterpolant evaluates uniform field" begin
        # The ReferenceInterp branch of create_wind_interpolants is currently incompatible with
        # the WindFields type constraints (u/v/w become ReferenceTrilinearInterpolant while
        # p/t/h stay as Interpolations.Extrapolation, violating the shared I4 type parameter).
        # Cover the interpolant call operator directly via constructed test data.
        nx, ny, nk = 6, 6, 4
        data1 = fill(7.5f0, nx, ny, nk)
        data2 = fill(7.5f0, nx, ny, nk)
        itp = _Tx_pde.ReferenceTrilinearInterpolant(data1, data2, 0.0f0, 3600.0f0, nx, ny, nk)

        # Constant field → constant output at any (x, y, z, t).
        @test itp(3.0, 3.0, 2.0, 0.0)    ≈ 7.5 atol=1e-4
        @test itp(3.0, 3.0, 2.0, 1800.0) ≈ 7.5 atol=1e-4
        @test itp(3.0, 3.0, 2.0, 3600.0) ≈ 7.5 atol=1e-4
        # Boundary clamping: out-of-domain queries also return the constant.
        @test itp(-50.0, 3.0, 2.0, 1800.0)   ≈ 7.5 atol=1e-4
        @test itp(3.0, 1000.0, 2.0, 1800.0)  ≈ 7.5 atol=1e-4
        @test itp(3.0, 3.0, 2.0, -1.0e9)     ≈ 7.5 atol=1e-4
        @test itp(3.0, 3.0, 2.0,  1.0e9)     ≈ 7.5 atol=1e-4

        # Equal-time degenerate case (t1 == t2) hits the dt<=0 branch (rt1=rt2=0.5).
        itp_eq = _Tx_pde.ReferenceTrilinearInterpolant(data1, data2, 1000.0f0, 1000.0f0,
                                                       nx, ny, nk)
        @test itp_eq(3.0, 3.0, 2.0, 1000.0) ≈ 7.5 atol=1e-4

        # Linear time blend: differing data1/data2 should interpolate to the midpoint at t = (t1+t2)/2.
        d1 = fill(2.0f0, nx, ny, nk)
        d2 = fill(8.0f0, nx, ny, nk)
        itp_lt = _Tx_pde.ReferenceTrilinearInterpolant(d1, d2, 0.0f0, 1000.0f0, nx, ny, nk)
        @test itp_lt(3.0, 3.0, 2.0, 0.0)     ≈ 2.0 atol=1e-4
        @test itp_lt(3.0, 3.0, 2.0, 500.0)   ≈ 5.0 atol=1e-4
        @test itp_lt(3.0, 3.0, 2.0, 1000.0)  ≈ 8.0 atol=1e-4

        # The ReferenceInterp config branch is documented as broken by type constraint
        # (u/v/w become ReferenceTrilinearInterpolant while p/t/h remain
        # Interpolations.Extrapolation — incompatible I4 parameter). This @test_throws
        # both exercises the early portion of the branch and pins the current behaviour.
        met = MeteoFields(6, 6, 4; T=Float64)
        met.u1 .= 1.0; met.u2 .= 1.0
        met.v1 .= 0.0; met.v2 .= 0.0
        met.w1 .= 0.0; met.w2 .= 0.0
        met.t1 .= 288.15; met.t2 .= 288.15
        met.ps1 .= 1013.0; met.ps2 .= 1013.0
        met.p1 .= 1013.0; met.p2 .= 1013.0
        met.vlevel .= collect(range(0.0, 1.0; length=4))
        met.vhalf  .= collect(range(0.0, 1.0; length=5))
        met.alevel .= 1.0; met.blevel .= 0.0
        met.ahalf  .= 1.0; met.bhalf  .= 0.0
        met.garea  .= 1.0; met.xm .= 1.0; met.ym .= 1.0
        cfg_ref = _Tx_pde.ERA5NumericalConfig{Float64}(
            interpolation_order=_Tx_pde.ReferenceInterp,
        )
        @test_throws MethodError create_wind_interpolants(met, 0.0, 3600.0;
                                                          config=cfg_ref,
                                                          lon_min=0.0, lon_max=10.0,
                                                          lat_min=0.0, lat_max=10.0)
        # Same with negate_v=true to cover the v1/v2 sign-flip lines inside the branch.
        @test_throws MethodError create_wind_interpolants(met, 0.0, 3600.0;
                                                          config=cfg_ref,
                                                          negate_v=true,
                                                          lon_min=0.0, lon_max=10.0,
                                                          lat_min=0.0, lat_max=10.0)
    end

    @testset "CubicInterp branch evaluates uniform field" begin
        met = MeteoFields(8, 8, 5; T=Float64)
        met.u1 .= 3.0;  met.u2 .= 3.0
        met.v1 .= 0.0;  met.v2 .= 0.0
        met.w1 .= 0.0;  met.w2 .= 0.0
        met.t1 .= 288.15; met.t2 .= 288.15
        met.ps1 .= 1013.0; met.ps2 .= 1013.0
        met.p1 .= 1013.0; met.p2 .= 1013.0
        met.vlevel .= collect(range(0.0, 1.0; length=5))
        met.vhalf  .= collect(range(0.0, 1.0; length=6))
        met.alevel .= 1.0; met.blevel .= 0.0
        met.ahalf  .= 1.0; met.bhalf  .= 0.0
        met.garea  .= 1.0; met.xm .= 1.0; met.ym .= 1.0

        cfg = _Tx_pde.ERA5NumericalConfig{Float64}(
            interpolation_order=_Tx_pde.CubicInterp,
        )
        winds = create_wind_interpolants(met, 0.0, 3600.0;
                                         config=cfg,
                                         lon_min=0.0, lon_max=10.0,
                                         lat_min=0.0, lat_max=10.0)

        @test isa(winds.u_interp, _Tx_pde.VerticalCubicInterpolant)
        # Constant input → constant output (interior + boundary).
        @test winds.u_interp(4.0, 4.0, 0.5, 1800.0) ≈ 3.0 atol=1e-6
        @test winds.u_interp(4.0, 4.0, 0.0, 1800.0) ≈ 3.0 atol=1e-6
        @test winds.u_interp(4.0, 4.0, 1.0, 1800.0) ≈ 3.0 atol=1e-6
    end

    @testset "compute_settling_constant returns finite positive sigma rate" begin
        met = _make_settling_met()
        winds = create_wind_interpolants(met, 0.0, 3600.0;
                                         lon_min=0.0, lon_max=10.0,
                                         lat_min=0.0, lat_max=10.0)

        params = ParticleParams{Float64}(grav_type=1, gravity_ms=0.01)
        sigma_rate = _Tx_pde.compute_settling_constant(params, winds, 3.0, 3.0, 0.5, 1800.0)
        @test isfinite(sigma_rate)
        @test sigma_rate > 0.0   # downward sigma (toward surface)

        # Zero gravity → zero settling.
        params0 = ParticleParams{Float64}(grav_type=1, gravity_ms=0.0)
        @test _Tx_pde.compute_settling_constant(params0, winds, 3.0, 3.0, 0.5, 1800.0) == 0.0
    end

    @testset "compute_settling_variable is a documented placeholder (returns 0)" begin
        met = _make_settling_met()
        winds = create_wind_interpolants(met, 0.0, 3600.0;
                                         lon_min=0.0, lon_max=10.0,
                                         lat_min=0.0, lat_max=10.0)
        params = ParticleParams{Float64}(grav_type=2)
        @test _Tx_pde.compute_settling_variable(params, winds, 3.0, 3.0, 0.5, 1800.0) == 0.0
    end

    @testset "particle_velocity! uses settling for grav_type=1" begin
        met = _make_settling_met()
        winds = create_wind_interpolants(met, 0.0, 3600.0;
                                         lon_min=0.0, lon_max=10.0,
                                         lat_min=0.0, lat_max=10.0)
        params = ParticleParams{Float64}(grav_type=1, gravity_ms=0.05)
        ode_p = _Tx_pde.ParticleODEParams(winds, params)
        du = zeros(Float64, 3)
        u  = [3.0, 3.0, 0.5]
        particle_velocity!(du, u, ode_p, 1800.0)
        # With u-wind = 5 m/s and zero v/w wind, du[3] is purely settling (positive sigma rate).
        @test du[3] > 0.0
        @test isfinite(du[1]) && isfinite(du[2]) && isfinite(du[3])
    end
end

println("✓ All particle_dynamics extras tests passed!")
