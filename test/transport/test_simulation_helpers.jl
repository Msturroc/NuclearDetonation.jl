# Tests for simulation.jl pure helpers: height/sigma/level conversions, lat/lon ↔ grid

using Test

const _Tx_s = NuclearDetonation.Transport

@testset "simulation: pure coordinate helpers" begin

    @testset "height_to_sigma / sigma_to_height" begin
        # Surface ↔ sigma=1, top ↔ sigma=0.
        @test _Tx_s.height_to_sigma(0.0, 10000.0) ≈ 1.0
        @test _Tx_s.height_to_sigma(10000.0, 10000.0) ≈ 0.0
        @test _Tx_s.height_to_sigma(5000.0, 10000.0) ≈ 0.5

        @test _Tx_s.sigma_to_height(1.0, 10000.0) ≈ 0.0
        @test _Tx_s.sigma_to_height(0.0, 10000.0) ≈ 10000.0
        @test _Tx_s.sigma_to_height(0.5, 10000.0) ≈ 5000.0

        # Round-trip identity.
        for z in (100.0, 2500.0, 8000.0)
            σ = _Tx_s.height_to_sigma(z, 10000.0)
            @test _Tx_s.sigma_to_height(σ, 10000.0) ≈ z atol=1e-9
        end

        # Out-of-range clamps.
        @test _Tx_s.height_to_sigma(20000.0, 10000.0) == 0.0
        @test _Tx_s.height_to_sigma(-100.0, 10000.0) == 1.0
        @test _Tx_s.sigma_to_height(2.0, 10000.0) == 0.0
        @test _Tx_s.sigma_to_height(-0.5, 10000.0) == 10000.0

        # Degenerate z_max.
        @test _Tx_s.height_to_sigma(123.0, 0.0) == 0.0
        @test _Tx_s.sigma_to_height(0.5, 0.0) == 0.0
        @test _Tx_s.height_to_sigma(123.0, -5.0) == 0.0
    end

    @testset "height_to_level on monotonic vertical grid" begin
        hlevel = [0.0, 100.0, 500.0, 1500.0, 5000.0]
        @test _Tx_s.height_to_level(0.0, hlevel) == 1.0       # at floor
        @test _Tx_s.height_to_level(-50.0, hlevel) == 1.0      # below floor clamps
        @test _Tx_s.height_to_level(5000.0, hlevel) == 5.0    # at top
        @test _Tx_s.height_to_level(10_000.0, hlevel) == 5.0  # above top clamps

        # Halfway between level 1 (100) and level 2 (500) → fractional level 2.5.
        @test _Tx_s.height_to_level(300.0, hlevel) ≈ 2.5

        # Below level 1 (between 0 and 100) → between 1 and 2.
        @test _Tx_s.height_to_level(50.0, hlevel) ≈ 1.5
    end

    @testset "compute_layer_thickness" begin
        # Single-level edge case.
        thickness1 = _Tx_s.compute_layer_thickness([10.0])
        @test length(thickness1) == 1
        @test thickness1[1] > 0

        # Multi-level grid: each layer has positive thickness, sum spans the range.
        hlevel = [0.0, 100.0, 500.0, 1500.0]
        t = _Tx_s.compute_layer_thickness(hlevel)
        @test length(t) == length(hlevel)
        @test all(t .> 0)
    end

    @testset "latlon_to_grid / grid_to_latlon round-trip" begin
        nx, ny, nz = 11, 11, 2
        hlevel = [0.0, 1000.0]
        xm = ones(Float64, nx, ny)
        ym = ones(Float64, nx, ny)
        cell_area = ones(Float64, nx, ny)
        t_start = DateTime(2025, 1, 1, 0)
        t_end   = DateTime(2025, 1, 1, 1)
        dt_out  = Duration(0, 1, 0, 0)
        dt_met  = Duration(0, 1, 0, 0)

        domain = SimulationDomain(
            nx, ny, nz, 1.0, 1.0, hlevel, xm, ym, cell_area,
            t_start, t_end, dt_out, dt_met;
            lon_min=-120.0, lon_max=-110.0,
            lat_min=35.0,   lat_max=45.0,
        )

        # Centre of the grid: x=6, y=6 ↔ lon midpoint, lat midpoint.
        xc, yc = _Tx_s.latlon_to_grid(domain, 40.0, -115.0)
        @test xc ≈ 6.0  rtol=1e-12
        @test yc ≈ 6.0  rtol=1e-12

        # Corner: lat=lat_min, lon=lon_min ↔ grid (1,1).
        xll, yll = _Tx_s.latlon_to_grid(domain, 35.0, -120.0)
        @test xll ≈ 1.0 atol=1e-10
        @test yll ≈ 1.0 atol=1e-10

        # Round-trip
        for (lat, lon) in ((36.5, -118.0), (44.0, -111.5), (38.7, -114.3))
            x, y = _Tx_s.latlon_to_grid(domain, lat, lon)
            lat2, lon2 = _Tx_s.grid_to_latlon(domain, x, y)
            @test lat2 ≈ lat rtol=1e-10
            @test lon2 ≈ lon rtol=1e-10
        end

        # Longitude convention auto-adjustment: domain in 0–360, query in -180–180.
        domain_360 = SimulationDomain(
            nx, ny, nz, 1.0, 1.0, hlevel, xm, ym, cell_area,
            t_start, t_end, dt_out, dt_met;
            lon_min=240.0, lon_max=250.0,
            lat_min=35.0,  lat_max=45.0,
        )
        x_neg, _ = _Tx_s.latlon_to_grid(domain_360, 40.0, -115.0)  # -115 → 245
        @test x_neg ≈ 6.0 rtol=1e-10
    end

    @testset "grid_cell_area on SimulationDomain" begin
        nx, ny, nz = 5, 5, 2
        hlevel = [0.0, 1000.0]
        xm = ones(Float64, nx, ny); ym = ones(Float64, nx, ny)
        cell_area = ones(Float64, nx, ny)
        t_start = DateTime(2025, 1, 1, 0); t_end = DateTime(2025, 1, 1, 1)
        dt_out = Duration(0, 1, 0, 0); dt_met = Duration(0, 1, 0, 0)

        # Equatorial 1° × 1° grid, 5×5 cells of 0.2° each.
        d_eq = SimulationDomain(
            nx, ny, nz, 1.0, 1.0, hlevel, xm, ym, cell_area,
            t_start, t_end, dt_out, dt_met;
            lon_min=0.0, lon_max=1.0, lat_min=0.0, lat_max=1.0,
        )
        a_specific = _Tx_s.grid_cell_area(d_eq, 1, 1)
        a_default  = _Tx_s.grid_cell_area(d_eq)  # mean / unspecified
        @test a_specific > 0 && isfinite(a_specific)
        @test a_default > 0 && isfinite(a_default)

        # 60°N domain returns ~half of equatorial cell area (cos(60°) ≈ 0.5).
        d_60 = SimulationDomain(
            nx, ny, nz, 1.0, 1.0, hlevel, xm, ym, cell_area,
            t_start, t_end, dt_out, dt_met;
            lon_min=0.0, lon_max=1.0, lat_min=60.0, lat_max=61.0,
        )
        a_60 = _Tx_s.grid_cell_area(d_60, 1, 1)
        @test 0.4 < a_60 / a_specific < 0.6
    end
end

println("✓ All simulation helper tests passed!")
