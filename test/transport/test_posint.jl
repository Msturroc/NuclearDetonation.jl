# Tests for posint.jl: bilinear/trilinear interpolation and per-particle met interpolation

using Test

const _Tx_p = NuclearDetonation.Transport

@testset "posint: spatial interpolation primitives" begin

    @testset "bilinear_interpolate" begin
        # Constant field → returns constant.
        field_const = fill(7.0, 5, 5)
        @test _Tx_p.bilinear_interpolate(field_const, 2.5, 3.5) == 7.0
        @test _Tx_p.bilinear_interpolate(field_const, 1.0, 1.0) == 7.0

        # Linear-in-x ramp: f(i, j) = i. At x=2.5, expected value is 2.5.
        field_x = Float64[i for i in 1:5, j in 1:5]
        @test _Tx_p.bilinear_interpolate(field_x, 2.5, 3.0) ≈ 2.5
        @test _Tx_p.bilinear_interpolate(field_x, 1.0, 1.0) ≈ 1.0
        @test _Tx_p.bilinear_interpolate(field_x, 4.0, 5.0) ≈ 4.0

        # Linear-in-y ramp: f(i, j) = j.
        field_y = Float64[j for i in 1:5, j in 1:5]
        @test _Tx_p.bilinear_interpolate(field_y, 3.0, 2.5) ≈ 2.5

        # Out-of-bounds queries clamp to nearest valid cell.
        @test isfinite(_Tx_p.bilinear_interpolate(field_x, -10.0, 3.0))
        @test isfinite(_Tx_p.bilinear_interpolate(field_x, 100.0, 3.0))
    end

    @testset "trilinear_interpolate" begin
        # Constant 3D field.
        field_const = fill(3.5, 4, 4, 4)
        @test _Tx_p.trilinear_interpolate(field_const, 2.0, 3.0, 1.5) == 3.5

        # Linear ramp in z: f(i, j, k) = k. At z=2.5, expected 2.5.
        field_z = Float64[k for i in 1:4, j in 1:4, k in 1:4]
        @test _Tx_p.trilinear_interpolate(field_z, 2.0, 2.0, 2.5) ≈ 2.5
        @test _Tx_p.trilinear_interpolate(field_z, 1.0, 1.0, 1.0) ≈ 1.0
        @test _Tx_p.trilinear_interpolate(field_z, 1.0, 1.0, 4.0) ≈ 4.0

        # Out-of-bounds clamps.
        @test isfinite(_Tx_p.trilinear_interpolate(field_z, 2.0, 2.0, -5.0))
        @test isfinite(_Tx_p.trilinear_interpolate(field_z, 2.0, 2.0, 99.0))
    end

    @testset "interpolate_met_to_particle! updates particle BL and map ratios" begin
        nx, ny, nk = 5, 5, 3
        met = MeteoFields(nx, ny, nk; T=Float32)
        met.hbl1   .= 1000.0f0
        met.hbl2   .= 1500.0f0   # advances to 1500 m at t2
        met.xm     .= 2.0f0
        met.ym     .= 1.0f0
        met.precip1 .= 0.0f0
        met.precip2 .= 4.0f0     # 4 mm/hr at t2

        particle = Particle(2.5, 3.5, 0.5, 0.0f0, 100.0f0, 1.0f0, 0.0f0, Int16(1))
        @test is_active(particle)
        pextra = ExtraParticle()

        # Halfway between t1=0 and t2=3600 → at t=1800.
        # Expected hbl: 0.5 * 1000 + 0.5 * 1500 = 1250.
        # Expected prc: 0.5 * 0    + 0.5 * 4    = 2.0.
        # Expected rmx = xm/dx = 2.0 / 4.0 = 0.5.  rmy = ym/dy = 1.0 / 2.0 = 0.5.
        _Tx_p.interpolate_met_to_particle!(particle, pextra, met,
                                           0.0, 3600.0, 1800.0, 4.0, 2.0)
        @test particle.hbl ≈ 1250.0f0 atol=1e-3
        @test pextra.prc  ≈ 2.0f0    atol=1e-5
        @test pextra.rmx  ≈ 0.5      atol=1e-10
        @test pextra.rmy  ≈ 0.5      atol=1e-10

        # Inactive particle: function returns immediately, leaving fields untouched.
        inactive = Particle(2.5, 3.5, 0.5, 0.0f0, 0.0f0, 1.0f0, 0.0f0, Int16(1))
        @test !is_active(inactive)
        ep2 = ExtraParticle()
        ep2.rmx = -42.0
        _Tx_p.interpolate_met_to_particle!(inactive, ep2, met,
                                           0.0, 3600.0, 1800.0, 4.0, 2.0)
        @test ep2.rmx == -42.0  # unchanged
    end
end

println("✓ All posint tests passed!")
