# Tests for compheight.jl: hypsometric height integration over hybrid coordinates

using Test

const _Tx_h = NuclearDetonation.Transport

# Build a tiny isothermal atmosphere on a 2×2 horizontal grid with nk vertical levels.
# Hybrid coordinates are stored in hPa per project convention.
function _make_isothermal_met(; nx=2, ny=2, nk=4, T_K=288.15, ps_hpa=1013.25)
    met = MeteoFields(nx, ny, nk; T=Float64)

    # Mid-level pressures: linearly spaced from surface (1000 hPa) up to ~50 hPa.
    p_full = collect(range(1000.0, 50.0; length=nk))
    # Half-level pressures: nk+1 values bracketing the mid-levels.
    # p_half[1] = surface, p_half[nk+1] = top of atmosphere.
    p_half = vcat(ps_hpa, [(p_full[k] + p_full[k+1]) / 2 for k in 1:nk-1], 25.0)

    # Pure pressure coordinate: alevel = p_full, blevel = 0
    # (no surface-pressure dependence — simplest sanity case).
    met.alevel .= p_full
    met.blevel .= 0.0
    met.ahalf  .= p_half
    met.bhalf  .= 0.0

    met.ps1 .= ps_hpa
    met.ps2 .= ps_hpa
    met.t1  .= T_K
    met.t2  .= T_K
    return met
end

@testset "compheight: model height integration" begin

    @testset "compute_model_heights! produces monotonic, positive heights" begin
        met = _make_isothermal_met(; nk=5)

        # Default ERA5 path
        _Tx_h.compute_model_heights!(met, 2)

        # Surface (k=1) is exactly zero by construction.
        @test met.hlevel2[1, 1, 1] == 0.0
        # Subsequent levels strictly increasing in height.
        for k in 2:size(met.hlevel2, 3)
            @test met.hlevel2[1, 1, k] > met.hlevel2[1, 1, k-1]
            @test isfinite(met.hlevel2[1, 1, k])
        end
        # Layer thicknesses positive (last layer is sentinel 9999).
        for k in 1:size(met.hlayer2, 3) - 1
            @test met.hlayer2[1, 1, k] > 0.0
            @test isfinite(met.hlayer2[1, 1, k])
        end
        @test met.hlayer2[1, 1, end] == 9999.0

        # Top mid-level around 50 hPa should sit somewhere between 5 km and 50 km
        # for a 288 K isothermal atmosphere — sanity bound, not exact.
        top_h = met.hlevel2[1, 1, end]
        @test 5_000.0 < top_h < 50_000.0
    end

    @testset "compute_model_heights! time_level=1 path" begin
        met = _make_isothermal_met(; nk=4)
        _Tx_h.compute_model_heights!(met, 1)
        @test met.hlevel1[1, 1, 1] == 0.0
        for k in 2:size(met.hlevel1, 3)
            @test met.hlevel1[1, 1, k] > met.hlevel1[1, 1, k-1]
        end
    end

    @testset "compute_model_heights! errors on invalid time_level" begin
        met = _make_isothermal_met(; nk=3)
        @test_throws ErrorException _Tx_h.compute_model_heights!(met, 7)
    end

    @testset "compute_model_heights_simple! gives monotonic heights" begin
        met = _make_isothermal_met(; nk=4)
        _Tx_h.compute_model_heights_simple!(met)
        @test met.hlevel2[1, 1, 1] == 0.0
        for k in 2:size(met.hlevel2, 3)
            @test met.hlevel2[1, 1, k] > met.hlevel2[1, 1, k-1]
            @test isfinite(met.hlevel2[1, 1, k])
        end
        @test met.hlayer2[1, 1, end] == 9999.0

        # Hypsometric equation: dz = (R T / g) * ln(p_lower / p_upper).
        # Compare layer thickness between half levels [1] (surface) and [2].
        R = 287.0; g = 9.80665
        p_lower = met.ahalf[2] + met.bhalf[2] * met.ps2[1, 1]
        p_upper = met.ahalf[3] + met.bhalf[3] * met.ps2[1, 1]
        expected_dz = (R * met.t2[1, 1, 2] / g) * log(p_lower / p_upper)
        @test met.hlayer2[1, 1, 1] ≈ expected_dz rtol=1e-10
    end
end

println("✓ All compheight tests passed!")
