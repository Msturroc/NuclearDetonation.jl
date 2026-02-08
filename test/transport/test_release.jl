# Tests for release.jl: Source Term Modeling
# Based on SNAP release.f90

using Test
using Random

# Import release module
using NuclearDetonation.Transport:
    ReleaseProfile, ConstantRelease, BombRelease, LinearRelease, StepRelease,
    ReleaseGeometry, ColumnRelease, CylinderRelease, MushroomCloudRelease,
    ReleaseSource, Plume,
    compute_release_cylinders, sample_cylinder_position,
    generate_release_particles, compute_release_rate,
    create_mushroom_cloud_from_yield

@testset "release: Source Term Modeling" begin

    @testset "Release Profiles" begin
        # Constant release
        const_prof = ConstantRelease()
        @test const_prof isa ConstantRelease

        # Bomb release
        bomb_prof = BombRelease(0.5)  # 30 minutes after start
        @test bomb_prof.time_hours == 0.5

        # Linear release
        times = [0.0, 1.0, 2.0]
        rates = [1e10 1e9 1e8;
                 2e10 2e9 2e8]
        linear_prof = LinearRelease(times, rates)
        @test length(linear_prof.times_hours) == 3
        @test size(linear_prof.rates) == (2, 3)

        # Step release
        step_prof = StepRelease(times, rates)
        @test length(step_prof.times_hours) == 3
    end

    @testset "Release Geometries" begin
        # Column release
        col = ColumnRelease(0.0, 1000.0)
        @test col.hlower == 0.0
        @test col.hupper == 1000.0

        # Cylinder release
        cyl = CylinderRelease(100.0, 500.0, 50.0)
        @test cyl.hlower == 100.0
        @test cyl.hupper == 500.0
        @test cyl.radius == 50.0

        # Mushroom cloud
        mushroom = MushroomCloudRelease(2000.0, 8000.0, 500.0, 2000.0)
        @test mushroom.stem_height == 2000.0
        @test mushroom.cap_height == 8000.0
        @test mushroom.stem_radius == 500.0
        @test mushroom.cap_radius == 2000.0

        # Invalid mushroom cloud (cap height < stem height)
        @test_throws AssertionError MushroomCloudRelease(8000.0, 2000.0, 500.0, 2000.0)
    end

    @testset "Cylinder Decomposition" begin
        # Column: single cylinder with radius=0
        col = ColumnRelease(0.0, 1000.0)
        cyls = compute_release_cylinders(col)
        @test length(cyls) == 1
        @test cyls[1].radius == 0.0
        @test cyls[1].volume_fraction == 1.0

        # Cylinder: single cylinder
        cyl = CylinderRelease(100.0, 500.0, 50.0)
        cyls = compute_release_cylinders(cyl)
        @test length(cyls) == 1
        @test cyls[1].radius == 50.0
        @test cyls[1].volume_fraction == 1.0

        # Mushroom cloud: two cylinders
        mushroom = MushroomCloudRelease(2000.0, 8000.0, 500.0, 2000.0)
        cyls = compute_release_cylinders(mushroom)
        @test length(cyls) == 2

        # Stem
        @test cyls[1].hlower == 0.0
        @test cyls[1].hupper == 2000.0
        @test cyls[1].radius == 500.0

        # Cap
        @test cyls[2].hlower == 2000.0
        @test cyls[2].hupper == 8000.0
        @test cyls[2].radius == 2000.0

        # Volume fractions should sum to 1
        @test cyls[1].volume_fraction + cyls[2].volume_fraction ≈ 1.0

        # Volume should be proportional to π*r²*h
        stem_vol = π * 500.0^2 * 2000.0
        cap_vol = π * 2000.0^2 * (8000.0 - 2000.0)
        total_vol = stem_vol + cap_vol
        @test cyls[1].volume_fraction ≈ stem_vol / total_vol rtol=1e-10
        @test cyls[2].volume_fraction ≈ cap_vol / total_vol rtol=1e-10
    end

    @testset "Cylinder Position Sampling" begin
        rng = Random.MersenneTwister(42)

        # Column (radius=0): should return center
        x_center, y_center = 50.0, 75.0
        for _ in 1:10
            x, y, z = sample_cylinder_position(rng, x_center, y_center, 0.0,
                                              0.0, 1000.0, 1.0, 1.0, 1.0, 1.0)
            @test x == x_center
            @test y == y_center
            @test 0.0 <= z <= 1000.0
        end

        # Cylinder: should be within radius
        radius = 100.0
        xm, ym = 1.0, 1.0  # No map distortion
        dx_grid, dy_grid = 1000.0, 1000.0  # 1 km per grid unit

        positions = [sample_cylinder_position(rng, x_center, y_center, radius,
                                             0.0, 1000.0, xm, ym, dx_grid, dy_grid)
                    for _ in 1:100]

        for (x, y, z) in positions
            # Check height range
            @test 0.0 <= z <= 1000.0

            # Check within ellipse (should be circular with xm=ym=1)
            dx = abs(radius / (dx_grid / xm))
            dy = abs(radius / (dy_grid / ym))
            r_grid = sqrt(((x - x_center)/dx)^2 + ((y - y_center)/dy)^2)
            @test r_grid <= 1.0
        end

        # Check approximately uniform distribution in height
        z_vals = [pos[3] for pos in positions]
        @test minimum(z_vals) < 250.0  # Should sample low
        @test maximum(z_vals) > 750.0  # Should sample high
        @test abs(mean(z_vals) - 500.0) < 100.0  # Mean should be near center
    end

    @testset "Release Rate Computation" begin
        activity = [1e10, 2e10]
        nsteps_per_hour = 6  # 10 minute timesteps

        # Constant release
        const_prof = ConstantRelease()
        release, rates, dt = compute_release_rate(const_prof, 1, nsteps_per_hour, activity)
        @test release == true
        @test rates == activity
        @test dt == 600.0  # 3600/6

        # Bomb release - at release time
        bomb_prof = BombRelease(1.0)  # 1 hour = step 6
        release, rates, dt = compute_release_rate(bomb_prof, 6, nsteps_per_hour, activity)
        @test release == true
        @test rates == activity
        @test dt == 1.0  # Instantaneous

        # Bomb release - before release time
        release, rates, dt = compute_release_rate(bomb_prof, 5, nsteps_per_hour, activity)
        @test release == false
        @test all(rates .== 0.0)

        # Bomb release - after release time
        release, rates, dt = compute_release_rate(bomb_prof, 7, nsteps_per_hour, activity)
        @test release == false

        # Linear release
        times = [0.0, 1.0, 2.0]
        rate_matrix = [1e10 5e9 1e8;
                      2e10 1e10 2e8]
        linear_prof = LinearRelease(times, rate_matrix)

        # At t=0.5 (step 3), should interpolate between times[1] and times[2]
        release, rates, dt = compute_release_rate(linear_prof, 3, nsteps_per_hour, activity)
        @test release == true
        @test rates[1] ≈ 0.5 * 1e10 + 0.5 * 5e9
        @test rates[2] ≈ 0.5 * 2e10 + 0.5 * 1e10

        # Step release
        step_prof = StepRelease(times, rate_matrix)
        release, rates, dt = compute_release_rate(step_prof, 3, nsteps_per_hour, activity)
        @test release == true
        @test rates == rate_matrix[:, 1]  # Should use first step
    end

    @testset "Mushroom Cloud from Yield" begin
        # 15 kt surface burst (Hiroshima-sized)
        cloud = create_mushroom_cloud_from_yield(15.0, 0.0)
        @test cloud isa MushroomCloudRelease
        @test cloud.cap_height > cloud.stem_height > 0
        @test cloud.cap_radius > cloud.stem_radius > 0

        # Rough expected values (from Glasstone & Dolan scaling)
        # H ≈ 3.5 * 15^0.33 ≈ 8.6 km
        expected_height = 3.5 * 15.0^0.33 * 1000.0
        @test cloud.cap_height ≈ expected_height rtol=0.1

        # Stem should be ~65% of total
        @test cloud.stem_height / cloud.cap_height ≈ 0.65 rtol=0.1

        # 1 Mt burst (much larger)
        cloud_1mt = create_mushroom_cloud_from_yield(1000.0, 0.0)
        @test cloud_1mt.cap_height > cloud.cap_height  # Scales as W^0.33

        # Airburst (reduced height)
        cloud_air = create_mushroom_cloud_from_yield(15.0, 500.0)
        @test cloud_air.cap_height < cloud.cap_height
    end

    @testset "Release Source Construction" begin
        geom = CylinderRelease(0.0, 1000.0, 100.0)
        prof = BombRelease(0.0)
        activity = [1e15, 5e14]

        source = ReleaseSource((50.0, 75.0), geom, prof, activity, 1000)
        @test source.position == (50.0, 75.0)
        @test source.nparticles == 1000
        @test length(source.activity) == 2

        # Invalid: negative particles
        @test_throws AssertionError ReleaseSource((50.0, 75.0), geom, prof, activity, -10)

        # Invalid: negative activity
        @test_throws AssertionError ReleaseSource((50.0, 75.0), geom, prof, [-1e15, 5e14], 1000)
    end

    @testset "Particle Generation Integration" begin
        rng = Random.MersenneTwister(123)

        # Create simple release
        geom = CylinderRelease(0.0, 1000.0, 100.0)
        prof = BombRelease(0.0)
        activity = [1e15]  # Single component
        source = ReleaseSource((50.0, 75.0), geom, prof, activity, 100)

        # Mock meteorological fields
        xm_field = ones(Float64, 100, 100)
        ym_field = ones(Float64, 100, 100)
        dx_grid = 1000.0  # 1 km
        dy_grid = 1000.0
        hlevel = collect(LinRange(0.0, 10000.0, 10))

        # Generate at release time (step 0)
        positions, activities, released = generate_release_particles(
            rng, source, 0, 6, xm_field, ym_field, dx_grid, dy_grid, hlevel
        )

        @test released == true
        @test length(positions) == 100
        @test length(activities) == 100

        # All positions should be within cylinder
        for (x, y, z) in positions
            @test 0.0 <= z <= 1000.0
            # Check horizontal distance
            dx_m = (x - 50.0) * dx_grid / xm_field[50, 75]
            dy_m = (y - 75.0) * dy_grid / ym_field[50, 75]
            @test sqrt(dx_m^2 + dy_m^2) <= 100.0 + 1e-6
        end

        # Total activity should equal source activity
        total_activity = sum(activities)
        @test total_activity ≈ activity[1] rtol=1e-10

        # No release at other timesteps
        positions2, activities2, released2 = generate_release_particles(
            rng, source, 1, 6, xm_field, ym_field, dx_grid, dy_grid, hlevel
        )
        @test released2 == false
        @test isempty(positions2)
    end

    @testset "Physical Consistency" begin
        # Mushroom cloud volume partitioning
        mushroom = MushroomCloudRelease(3000.0, 10000.0, 800.0, 3000.0)
        cyls = compute_release_cylinders(mushroom)

        # Check conservation
        @test cyls[1].volume_fraction + cyls[2].volume_fraction ≈ 1.0

        # Larger radius should get more particles
        @test cyls[2].volume_fraction > cyls[1].volume_fraction

        # Yield scaling
        cloud_10kt = create_mushroom_cloud_from_yield(10.0)
        cloud_100kt = create_mushroom_cloud_from_yield(100.0)

        # Height should scale as W^0.33
        height_ratio = cloud_100kt.cap_height / cloud_10kt.cap_height
        expected_ratio = (100.0/10.0)^0.33
        @test height_ratio ≈ expected_ratio rtol=0.1
    end
end

println("✓ All release tests passed!")
