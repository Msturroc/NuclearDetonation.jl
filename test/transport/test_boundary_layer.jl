# Smoke tests for boundary_layer.jl: compute_boundary_layer! end-to-end on an isothermal fixture.

using Test

const _Tx_bl = NuclearDetonation.Transport

# Build a small isothermal atmosphere with realistic hybrid coordinates and a uniform wind shear.
# Surface (k=1) high pressure, TOA (k=nk) low pressure (ERA5 convention).
function _make_bl_met(; nx=3, ny=3, nk=8, T_K=288.15, ps_hpa=1013.25)
    met = MeteoFields(nx, ny, nk; T=Float64)

    # Mid-level pressures decreasing surface→top
    p_full = collect(range(ps_hpa - 10.0, 50.0; length=nk))
    p_half = vcat(ps_hpa, [(p_full[k] + p_full[k+1]) / 2 for k in 1:nk-1], 25.0)

    met.alevel .= p_full
    met.blevel .= 0.0
    met.ahalf  .= p_half
    met.bhalf  .= 0.0

    # vhalf: 1 at surface, decreasing toward 0 at TOA (sigma convention).
    met.vhalf  .= collect(range(1.0, 0.0; length=nk + 1))
    met.vlevel .= [0.5 * (met.vhalf[k] + met.vhalf[k+1]) for k in 1:nk]

    met.ps1 .= ps_hpa; met.ps2 .= ps_hpa
    met.t1  .= T_K;    met.t2  .= T_K

    # Uniform wind with weak vertical shear to give a non-trivial Richardson number.
    for k in 1:nk
        u_k = 5.0 + 0.5 * (k - 1)
        met.u1[:, :, k] .= u_k; met.u2[:, :, k] .= u_k
    end
    met.v1 .= 0.0; met.v2 .= 0.0

    met.garea .= 1.0; met.xm .= 1.0; met.ym .= 1.0
    return met
end

@testset "boundary_layer: compute_boundary_layer!" begin

    @testset "Populates bl2/hbl2 at time_level=2" begin
        met = _make_bl_met()
        # Both fields start at zero.
        @test all(met.bl2 .== 0.0)
        @test all(met.hbl2 .== 0.0)

        _Tx_bl.compute_boundary_layer!(met; time_level=2)

        @test all(isfinite, met.bl2)
        @test all(isfinite, met.hbl2)
        # At least one cell should produce a positive mixing height.
        @test any(met.hbl2 .> 0.0)
    end

    @testset "Populates bl1/hbl1 at time_level=1" begin
        met = _make_bl_met()
        _Tx_bl.compute_boundary_layer!(met; time_level=1)
        @test all(isfinite, met.bl1)
        @test all(isfinite, met.hbl1)
        @test any(met.hbl1 .> 0.0)
    end

    @testset "Skips cells with non-positive surface pressure" begin
        met = _make_bl_met()
        met.ps2[1, 1] = 0.0  # Force the no-data branch
        _Tx_bl.compute_boundary_layer!(met; time_level=2)
        # Cell should still be the initial value (zero); other cells populate normally.
        @test met.hbl2[1, 1] == 0.0
        @test any(met.hbl2 .> 0.0)
    end

    @testset "Format-explicit dispatch (ERA5Format)" begin
        met = _make_bl_met()
        _Tx_bl.compute_boundary_layer!(_Tx_bl.ERA5Format, met; time_level=2)
        @test all(isfinite, met.hbl2)
    end
end

println("✓ All boundary_layer tests passed!")
