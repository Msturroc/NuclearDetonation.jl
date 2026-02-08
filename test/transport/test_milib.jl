# Tests for milib.jl: Coordinate Transformations and Map Field Calculations
# Based on DNMI/met.no Fortran code

using Test

@testset "milib: Coordinate Transformations" begin

    @testset "earthr()" begin
        @test earthr() == 6371000.0
        @test earthr() isa Float64
    end

    @testset "sph2rot!: Spherical to Rotated Coordinates" begin
        # Test case: Convert single point
        x = [0.0]
        y = [π/4]  # 45° N latitude
        xcen = 0.0
        ycen = 0.0

        # Identity transformation (xcen=0, ycen=0) should return same values
        ierror = sph2rot!(1, x, y, xcen, ycen)
        @test ierror == 0
        @test x[1] ≈ 0.0 atol=1e-6
        @test y[1] ≈ π/4 atol=1e-6

        # Test round-trip: sph → rot → sph
        x = [deg2rad(10.0)]
        y = [deg2rad(60.0)]
        x_orig = copy(x)
        y_orig = copy(y)
        xcen = deg2rad(15.0)
        ycen = deg2rad(50.0)

        ierror = sph2rot!(1, x, y, xcen, ycen)
        @test ierror == 0
        ierror = sph2rot!(-1, x, y, xcen, ycen)
        @test ierror == 0
        @test x[1] ≈ x_orig[1] atol=1e-10
        @test y[1] ≈ y_orig[1] atol=1e-10

        # Test invalid icall
        ierror = sph2rot!(999, x, y, xcen, ycen)
        @test ierror == 1
    end

    @testset "pol2sph!: Polar Stereographic to Spherical" begin
        # Test case: North pole position should give 90° N
        x = [100.0]  # x position of north pole
        y = [200.0]  # y position of north pole
        xp = 100.0
        yp = 200.0
        an = 100.0
        fi = 0.0
        fpol = 60.0

        ierror = pol2sph!(1, x, y, fpol, xp, yp, an, fi)
        @test ierror == 0
        @test y[1] ≈ π/2 atol=1e-6  # Should be at north pole

        # Test round-trip
        x = [150.0]
        y = [250.0]
        x_orig = copy(x)
        y_orig = copy(y)

        ierror = pol2sph!(1, x, y, fpol, xp, yp, an, fi)
        @test ierror == 0
        ierror = pol2sph!(-1, x, y, fpol, xp, yp, an, fi)
        @test ierror == 0
        @test x[1] ≈ x_orig[1] atol=1e-8
        @test y[1] ≈ y_orig[1] atol=1e-8

        # Test invalid icall
        ierror = pol2sph!(999, x, y, fpol, xp, yp, an, fi)
        @test ierror == 1
    end

    @testset "mer2sph!: Mercator to Spherical" begin
        # Test case: Origin point
        x = [1.0]
        y = [1.0]
        xw = 0.0
        ys = 0.0
        dx = 10000.0
        dy = 10000.0
        yc = 0.0

        x_grid_orig = copy(x)
        y_grid_orig = copy(y)

        # Convert from grid to spherical
        ierror = mer2sph!(1, x, y, xw, ys, dx, dy, yc)
        @test ierror == 0

        # Test round-trip: Convert back to grid
        ierror = mer2sph!(-1, x, y, xw, ys, dx, dy, yc)
        @test ierror == 0
        @test x[1] ≈ x_grid_orig[1] atol=1e-6
        @test y[1] ≈ y_grid_orig[1] atol=1e-6
    end

    @testset "lam2sph!: Lambert to Spherical" begin
        # Test case: Lambert conformal
        x = [50.0]
        y = [50.0]
        xw = deg2rad(10.0)
        ys = deg2rad(50.0)
        dx = 2500.0
        dy = 2500.0
        x0 = deg2rad(15.0)
        y1 = deg2rad(60.0)
        y2 = deg2rad(60.0)

        x_grid_orig = copy(x)
        y_grid_orig = copy(y)

        # Convert from grid to spherical
        ierror = lam2sph!(1, x, y, xw, ys, dx, dy, x0, y1, y2)
        @test ierror == 0

        # Test round-trip: Convert back to grid
        ierror = lam2sph!(-1, x, y, xw, ys, dx, dy, x0, y1, y2)
        @test ierror == 0
        @test x[1] ≈ x_grid_orig[1] atol=1e-6
        @test y[1] ≈ y_grid_orig[1] atol=1e-6
    end

    @testset "xyconvert!: General Coordinate Conversion" begin
        # Test: Geographic grid conversion (should be identity)
        npos = 5
        x = collect(range(1.0, 10.0, length=npos))
        y = collect(range(1.0, 10.0, length=npos))
        x_orig = copy(x)
        y_orig = copy(y)

        # Geographic grid parameters (igtype=2)
        ga = [0.0, 50.0, 0.5, 0.5, 0.0, 0.0]  # west, south, dlon, dlat, rot_lon, rot_lat
        gr = [0.0, 50.0, 0.5, 0.5, 0.0, 0.0]  # Same grid

        ierror = xyconvert!(npos, x, y, 2, ga, 2, gr)
        @test ierror == 0
        @test all(x .≈ x_orig)
        @test all(y .≈ y_orig)

        # Test polar to polar conversion
        x = collect(range(50.0, 150.0, length=npos))
        y = collect(range(50.0, 150.0, length=npos))
        ga = [100.0, 100.0, 100.0, 0.0, 60.0, 0.0]  # xp, yp, an, fi, fp, unused
        gr = [100.0, 100.0, 100.0, 0.0, 60.0, 0.0]  # Same grid

        ierror = xyconvert!(npos, x, y, 1, ga, 1, gr)
        @test ierror == 0
    end
end

@testset "mapfield: Map Ratios and Coriolis" begin

    @testset "Polar Stereographic Grid (igtype=1)" begin
        nx, ny = 50, 50
        grid = [25.0, 25.0, 100.0, 0.0, 60.0, 0.0]  # xp, yp, an, fi, fp, unused

        # Compute both map ratio and Coriolis
        xm, ym, fc, hx, hy, ierror = mapfield(1, 1, 1, grid, nx, ny)

        @test ierror == 0
        @test hx > 0
        @test hy > 0
        @test hx ≈ hy  # Should be equal for polar stereographic
        @test size(xm) == (nx, ny)
        @test size(ym) == (nx, ny)
        @test size(fc) == (nx, ny)

        # At pole (xp, yp), map ratio should be minimum
        @test xm[25, 25] < xm[50, 50]

        # Coriolis should be maximum at pole
        @test fc[25, 25] > fc[50, 25]  # Pole vs equator-ward
    end

    @testset "Geographic Grid (igtype=2)" begin
        nx, ny = 36, 18
        grid = [0.0, -90.0, 10.0, 10.0, 0.0, 0.0]  # west, south, dlon, dlat, rot_lon, rot_lat

        xm, ym, fc, hx, hy, ierror = mapfield(1, 1, 2, grid, nx, ny)

        @test ierror == 0
        @test hx > 0
        @test hy > 0
        @test size(xm) == (nx, ny)
        @test size(ym) == (nx, ny)
        @test size(fc) == (nx, ny)

        # Map ratio in x should increase toward poles (smaller cos(lat))
        @test xm[1, 1] > xm[1, ny÷2]  # South pole vs equator

        # Coriolis should be zero near equator
        mid_lat = ny ÷ 2
        @test abs(fc[1, mid_lat]) < abs(fc[1, 1])  # Equator vs pole
    end

    @testset "Mercator Grid (igtype=5)" begin
        nx, ny = 50, 50
        grid = [0.0, 0.0, 2500.0, 2500.0, 0.0, 0.0]  # west, south, dx, dy, yc, unused

        xm, ym, fc, hx, hy, ierror = mapfield(1, 1, 5, grid, nx, ny)

        @test ierror == 0
        @test hx > 0
        @test hy > 0
        @test size(xm) == (nx, ny)
        @test size(ym) == (nx, ny)
        @test size(fc) == (nx, ny)
    end

    @testset "Map Ratio Computation Modes" begin
        nx, ny = 10, 10
        grid = [5.0, 5.0, 50.0, 0.0, 60.0, 0.0]

        # Mode 0: Don't compute map ratio
        xm0, ym0, fc0, hx0, hy0, ierror0 = mapfield(0, 0, 1, grid, nx, ny)
        @test ierror0 == 0
        @test isnothing(xm0)
        @test isnothing(ym0)
        @test isnothing(fc0)

        # Mode 1: Compute map ratio
        xm1, ym1, fc1, hx1, hy1, ierror1 = mapfield(1, 1, 1, grid, nx, ny)
        @test ierror1 == 0
        @test !isnothing(xm1)
        @test !isnothing(ym1)
        @test !isnothing(fc1)

        # Mode 2: Divide by grid resolution
        xm2, ym2, fc2, hx2, hy2, ierror2 = mapfield(2, 1, 1, grid, nx, ny)
        @test ierror2 == 0
        @test all(xm2 .< xm1)  # Should be smaller due to division
    end

    @testset "Error Handling" begin
        nx, ny = 10, 10

        # Invalid grid type
        grid = [0.0, 0.0, 1.0, 0.0, 60.0, 0.0]
        xm, ym, fc, hx, hy, ierror = mapfield(1, 1, 999, grid, nx, ny)
        @test ierror == 2  # Unknown grid type

        # Invalid grid parameters (an=0 for polar)
        grid = [5.0, 5.0, 0.0, 0.0, 60.0, 0.0]
        xm, ym, fc, hx, hy, ierror = mapfield(1, 1, 1, grid, nx, ny)
        @test ierror == 1  # Bad grid value
    end
end

println("✓ All milib tests passed!")
