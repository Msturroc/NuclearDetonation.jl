# Smoke tests for om2edot.jl: omega→sigma-dot conversion and continuity-based etadot.

using Test

const _Tx_om = NuclearDetonation.Transport

@testset "om2edot: omega/etadot conversion" begin

    @testset "convert_omega_to_sigmadot! converts uniform omega" begin
        nx, ny, nk = 4, 4, 5
        # Half-level pressures in hPa (heuristic in source: ps < 2000 → hPa).
        ahalf = collect(range(1013.25, 50.0; length=nk + 1))
        bhalf = zeros(Float64, nk + 1)
        # Sigma half-levels: 1 at surface (k=1), 0 at TOA (k=nk+1).
        vhalf = collect(range(1.0, 0.0; length=nk + 1))
        ps    = fill(1013.25, nx, ny)
        # Uniform downward omega: 0.1 Pa/s (positive = descent in pressure coords).
        w     = fill(0.1, nx, ny, nk)

        _Tx_om.convert_omega_to_sigmadot!(w, ps, ahalf, bhalf, vhalf)

        # k=1 is left untouched; k=2..nk are converted.
        @test all(isfinite, w)
        # Layer thickness is positive (deta = vhalf[k-1] - vhalf[k] > 0 since vhalf decreases).
        # dp from p_lower - p_upper > 0 in Pa. So sigma-dot has the same sign as omega → positive.
        for k in 2:nk
            @test w[1, 1, k] > 0.0
        end
    end

    @testset "convert_omega_to_sigmadot! safely handles ~zero dp" begin
        nx, ny, nk = 3, 3, 3
        ahalf = fill(1000.0, nk + 1)   # constant → dp == 0
        bhalf = zeros(Float64, nk + 1)
        vhalf = collect(range(1.0, 0.0; length=nk + 1))
        ps    = fill(1013.25, nx, ny)
        w     = fill(0.1, nx, ny, nk)

        _Tx_om.convert_omega_to_sigmadot!(w, ps, ahalf, bhalf, vhalf)

        # k>=2 should fall through to the "set to zero" branch.
        @test all(w[:, :, 2] .== 0.0)
        @test all(w[:, :, 3] .== 0.0)
    end

    @testset "convert_omega_to_sigmadot! Pa-units branch" begin
        nx, ny, nk = 3, 3, 3
        ahalf = collect(range(101325.0, 5000.0; length=nk + 1))  # Pa
        bhalf = zeros(Float64, nk + 1)
        vhalf = collect(range(1.0, 0.0; length=nk + 1))
        ps    = fill(101325.0, nx, ny)  # ≥ 2000 → Pa branch
        w     = fill(0.1, nx, ny, nk)

        _Tx_om.convert_omega_to_sigmadot!(w, ps, ahalf, bhalf, vhalf)
        @test all(isfinite, w)
        @test w[1, 1, 2] > 0.0
    end

    @testset "compute_etadot_from_continuity! runs and writes output" begin
        nx, ny, nk = 5, 5, 4
        ahalf = collect(range(101325.0, 5000.0; length=nk + 1))  # Pa
        bhalf = zeros(Float64, nk + 1)
        vhalf = collect(range(1.0, 0.0; length=nk + 1))
        ps    = fill(101325.0, nx, ny)
        # Uniform u-wind that varies with j to give non-zero divergence after differencing.
        u = zeros(Float64, nx, ny, nk)
        v = zeros(Float64, nx, ny, nk)
        for k in 1:nk, j in 1:ny, i in 1:nx
            u[i, j, k] = 5.0 + 0.1 * i
            v[i, j, k] = 0.0
        end
        xm = ones(Float64, nx, ny)
        ym = ones(Float64, nx, ny)
        edot = zeros(Float64, nx, ny, nk)
        dx, dy = 1000.0, 1000.0

        _Tx_om.compute_etadot_from_continuity!(edot, u, v, ps, xm, ym,
                                                ahalf, bhalf, vhalf,
                                                dx, dy; averaging=true)
        @test all(isfinite, edot)
        # At least some interior point should be non-zero — diverging u-field gives non-trivial edot.
        @test any(abs.(edot) .> 0.0)

        # ERA5Format dispatch path
        edot2 = zeros(Float64, nx, ny, nk)
        _Tx_om.compute_etadot_from_continuity!(_Tx_om.ERA5Format(), edot2, u, v, ps, xm, ym,
                                                ahalf, bhalf, vhalf, dx, dy; averaging=false)
        @test all(isfinite, edot2)
    end

    @testset "compute_map_scale_factors! returns sensible values" begin
        nx, ny = 4, 5
        xm = zeros(Float64, nx, ny)
        ym = zeros(Float64, nx, ny)
        # Latitudes from 0° (equator, xm=1) to 60° (xm=2).
        lats = collect(range(0.0, 60.0; length=ny))

        _Tx_om.compute_map_scale_factors!(xm, ym, lats)
        @test all(ym .== 1.0)
        @test all(xm[:, 1] .≈ 1.0)              # equator
        @test all(xm[:, end] .≈ 1.0 / cos(deg2rad(60.0)))  # 60° N
        # xm strictly increases with latitude.
        @test all(diff(xm[1, :]) .>= 0.0)
    end
end

println("✓ All om2edot tests passed!")
