# Tests for vgravtables.jl: Gravitational Settling Velocity Tables
# Based on SNAP vgravtables.f90

using Test

@testset "vgravtables: Gravitational Settling" begin

    @testset "Physical Constants" begin
        @test G_GRAVITY_CM_S2 == 981.0
        @test R_SPECIFIC_J_KG_K == 287.04
        @test LAMBDA_FREE_PATH_μM == 0.0653
    end

    @testset "Air Viscosity" begin
        # At 0°C (273.15 K) - using Sutherland formula from FLEXPART
        η_0C = air_viscosity(273.15)
        @test η_0C ≈ 1.733e-4 rtol=0.01  # g/(cm·s)

        # At 20°C (293.15 K) - should be higher than 0°C
        η_20C = air_viscosity(293.15)
        @test η_20C > η_0C

        # At -40°C (233.15 K) - should be lower than 0°C
        η_minus40C = air_viscosity(233.15)
        @test η_minus40C < η_0C

        # Temperature dependence: viscosity increases with temperature
        @test air_viscosity(300.0) > air_viscosity(250.0)

        # At reference temperature T₀ = 291.15 K
        η_ref = air_viscosity(291.15)
        @test η_ref ≈ 1.827e-4 rtol=0.01  # Should match reference value
    end

    @testset "Air Density" begin
        # At sea level (1013.25 hPa) and 0°C (273.15 K)
        ρ_sealevel = air_density(1013.25, 273.15)
        @test ρ_sealevel ≈ 1.293e-3 rtol=0.01  # g/cm³

        # Ideal gas law: ρ ∝ P/T
        # Higher pressure → higher density
        @test air_density(1013.25, 273.15) > air_density(500.0, 273.15)

        # Higher temperature → lower density
        @test air_density(1013.25, 273.15) > air_density(1013.25, 300.0)

        # Low pressure (high altitude)
        ρ_high_alt = air_density(300.0, 250.0)
        @test ρ_high_alt < ρ_sealevel
    end

    @testset "Cunningham Slip Correction" begin
        # Large particles (>> mean free path): C ≈ 1
        C_large = cunningham_factor(100.0)  # 100 μm
        @test C_large ≈ 1.0 atol=0.1

        # Small particles (~ mean free path): significant correction
        C_small = cunningham_factor(0.1)  # 0.1 μm
        @test C_small > 2.0  # Strong slip effect

        # Intermediate particles
        C_mid = cunningham_factor(1.0)  # 1 μm
        @test C_mid > 1.0
        @test C_mid < 2.0

        # Monotonic: C decreases as particle size increases
        @test cunningham_factor(0.1) > cunningham_factor(1.0)
        @test cunningham_factor(1.0) > cunningham_factor(10.0)
        @test cunningham_factor(10.0) > cunningham_factor(100.0)

        # At dp = λ (mean free path), C should be fairly large
        C_at_lambda = cunningham_factor(LAMBDA_FREE_PATH_μM)
        @test C_at_lambda > 3.0  # Significant correction at mean free path
    end

    @testset "Stokes Settling Velocity" begin
        # Standard conditions: 1 μm particle, 2.5 g/cm³, sea level, 0°C
        vg = vgrav_stokes(1.0, 2.5, 1013.25, 273.15)

        # Should be positive (settling downward)
        @test vg > 0.0

        # Reasonable order of magnitude (few μm/s for 1 μm particle)
        @test vg > 1e-4  # cm/s
        @test vg < 1.0

        # Velocity proportional to dp²
        vg_1μm = vgrav_stokes(1.0, 2.5, 1013.25, 273.15)
        vg_2μm = vgrav_stokes(2.0, 2.5, 1013.25, 273.15)
        @test vg_2μm / vg_1μm ≈ 4.0 rtol=0.1  # Should be ~4x

        # Velocity proportional to (ρp - ρa)
        vg_light = vgrav_stokes(1.0, 1.5, 1013.25, 273.15)
        vg_heavy = vgrav_stokes(1.0, 3.0, 1013.25, 273.15)
        @test vg_heavy > vg_light

        # At high altitude (low pressure), air density lower → faster settling
        vg_sealevel = vgrav_stokes(1.0, 2.5, 1013.25, 273.15)
        vg_highalt = vgrav_stokes(1.0, 2.5, 300.0, 250.0)
        @test vg_highalt > vg_sealevel
    end

    @testset "Drag Coefficient" begin
        # Very low Reynolds number: should approach 24/Re (Stokes regime)
        Re_stokes = 0.01
        Cd_stokes = drag_coefficient(Re_stokes)
        @test Cd_stokes ≈ 24.0 / Re_stokes

        # Low Reynolds number: Stokes with correction
        Re_low = 0.1
        Cd_low = drag_coefficient(Re_low)
        @test Cd_low > 24.0 / Re_low  # Correction increases Cd slightly

        # Intermediate Reynolds number: both terms contribute
        Re_mid = 100.0
        Cd_mid = drag_coefficient(Re_mid)
        @test Cd_mid > 0.0
        @test isfinite(Cd_mid)

        # High Reynolds number: Newton regime dominates
        Re_high = 10000.0
        Cd_high = drag_coefficient(Re_high)
        @test Cd_high ≈ 0.42 rtol=0.1  # Approaches Newton drag coefficient

        # Drag coefficient should decrease with Reynolds number (initially)
        @test drag_coefficient(1.0) > drag_coefficient(10.0)
        @test drag_coefficient(10.0) > drag_coefficient(100.0)
    end

    @testset "Corrected Settling Velocity" begin
        # Small particle: correction should be negligible
        dp_small = 1.0  # 1 μm
        vg_stokes = vgrav_stokes(dp_small, 2.5, 1013.25, 273.15)
        vg_corrected = vgrav_corrected(dp_small, 2.5, 1013.25, 273.15)
        @test vg_corrected ≈ vg_stokes rtol=0.01  # Within 1%

        # Large particle: correction should be significant
        dp_large = 100.0  # 100 μm
        vg_stokes_large = vgrav_stokes(dp_large, 2.5, 1013.25, 273.15)
        vg_corrected_large = vgrav_corrected(dp_large, 2.5, 1013.25, 273.15)

        # Reynolds correction changes velocity (can increase or decrease depending on regime)
        @test vg_corrected_large != vg_stokes_large
        @test abs(vg_corrected_large - vg_stokes_large) / vg_stokes_large > 0.01  # >1% difference

        # Convergence test: should not error
        @test vg_corrected_large > 0.0
    end

    @testset "ParticleProperties Construction" begin
        props = ParticleProperties(diameter_μm=1.0, density_gcm3=2.5)
        @test props.diameter_μm == 1.0
        @test props.density_gcm3 == 2.5
    end

    @testset "Build VGrav Tables" begin
        # Single component
        props = [ParticleProperties(diameter_μm=10.0, density_gcm3=2.5)]
        tables = build_vgrav_tables(props)

        @test tables.numtemp == 41
        @test tables.numpres == 25
        @test size(tables.vgtable) == (41, 25, 1)

        # All velocities should be positive
        @test all(tables.vgtable .> 0.0)

        # Velocities should be in reasonable range (m/s)
        @test all(tables.vgtable .< 1.0)  # Less than 1 m/s for typical particles
        @test all(tables.vgtable .> 1e-6)  # Greater than 1 μm/s

        # Temperature range check
        T_min = tables.t_base + tables.t_incr
        T_max = tables.t_base + tables.numtemp * tables.t_incr
        @test T_min ≈ 153.0 rtol=0.01  # ~153 K
        @test T_max ≈ 353.0 rtol=0.01  # ~353 K

        # Pressure range check
        P_min = tables.p_base + tables.p_incr
        P_max = tables.p_base + tables.numpres * tables.p_incr
        @test P_min ≈ 0.0 rtol=0.1  # ~0 hPa (clamped to 1.0 in code)
        @test P_max ≈ 1200.0 rtol=0.01  # 1200 hPa
    end

    @testset "Build Tables: Multiple Components" begin
        props = [
            ParticleProperties(diameter_μm=1.0, density_gcm3=2.5),   # Fine
            ParticleProperties(diameter_μm=10.0, density_gcm3=2.5),  # Medium
            ParticleProperties(diameter_μm=100.0, density_gcm3=2.5)  # Coarse
        ]

        tables = build_vgrav_tables(props)

        @test size(tables.vgtable) == (41, 25, 3)

        # Larger particles settle faster at same conditions
        # Compare at mid-range T and P
        it_mid = 21  # Mid temperature
        ip_mid = 13  # Mid pressure

        vg_1μm = tables.vgtable[it_mid, ip_mid, 1]
        vg_10μm = tables.vgtable[it_mid, ip_mid, 2]
        vg_100μm = tables.vgtable[it_mid, ip_mid, 3]

        @test vg_10μm > vg_1μm
        @test vg_100μm > vg_10μm
    end

    @testset "Table Interpolation" begin
        props = [ParticleProperties(diameter_μm=10.0, density_gcm3=2.5)]
        tables = build_vgrav_tables(props)

        # Interpolate at grid point (should match table value)
        T_grid = tables.t_base + 20 * tables.t_incr
        P_grid = tables.p_base + 10 * tables.p_incr
        vg_interp = interpolate_vgrav(tables, 1, P_grid, T_grid)
        vg_table = tables.vgtable[20, 10, 1]
        @test vg_interp ≈ vg_table rtol=1e-6

        # Interpolate between grid points
        T_between = tables.t_base + 20.5 * tables.t_incr
        P_between = tables.p_base + 10.5 * tables.p_incr
        vg_between = interpolate_vgrav(tables, 1, P_between, T_between)

        # Should be between neighboring values
        v1 = tables.vgtable[20, 10, 1]
        v2 = tables.vgtable[21, 11, 1]
        @test vg_between >= min(v1, v2)
        @test vg_between <= max(v1, v2)

        # Test clamping at boundaries (out of range values should still return valid velocities)
        T_low = 100.0  # Below table range
        vg_clamped = interpolate_vgrav(tables, 1, 1000.0, T_low)
        # Should return a positive, finite velocity
        @test vg_clamped > 0.0
        @test isfinite(vg_clamped)
    end

    @testset "Physical Trends in Tables" begin
        props = [ParticleProperties(diameter_μm=10.0, density_gcm3=2.5)]
        tables = build_vgrav_tables(props)

        # At constant pressure, settling velocity should vary with temperature
        ip = 13  # Mid-pressure level
        vg_cold = tables.vgtable[5, ip, 1]   # Cold
        vg_warm = tables.vgtable[35, ip, 1]  # Warm

        # Temperature affects both density (↓ with T) and viscosity (↑ with T)
        # Net effect varies by particle size, but should be different
        @test vg_cold != vg_warm
        @test abs(vg_warm - vg_cold) / vg_cold > 0.1  # >10% difference

        # At constant temperature, settling velocity should vary with pressure
        it = 21  # Mid-temperature level
        vg_low_p = tables.vgtable[it, 5, 1]   # Low pressure (high altitude)
        vg_high_p = tables.vgtable[it, 20, 1]  # High pressure (low altitude)

        # Lower pressure → less dense air → faster settling
        @test vg_low_p > vg_high_p
    end

    @testset "Realistic Particle Examples" begin
        # Cs-137 aerosol: ~0.5 μm, density ~2.5 g/cm³
        vg_cs137 = vgrav_corrected(0.5, 2.5, 1013.25, 288.15)
        @test vg_cs137 > 0.0
        @test vg_cs137 < 0.01  # cm/s, very slow settling

        # Large dust particle: 50 μm, density ~2.65 g/cm³
        vg_dust = vgrav_corrected(50.0, 2.65, 1013.25, 288.15)
        @test vg_dust > vg_cs137 * 1000  # Much faster

        # Sand grain: 500 μm, density ~2.65 g/cm³
        vg_sand = vgrav_corrected(500.0, 2.65, 1013.25, 288.15)
        @test vg_sand > vg_dust * 10  # Even faster
    end

    @testset "Extreme Conditions" begin
        # Very high altitude (low pressure, cold)
        vg_stratosphere = vgrav_corrected(1.0, 2.5, 50.0, 220.0)
        @test vg_stratosphere > 0.0

        # Very hot conditions
        vg_hot = vgrav_corrected(1.0, 2.5, 1013.25, 350.0)
        @test vg_hot > 0.0

        # Very cold conditions
        vg_cold = vgrav_corrected(1.0, 2.5, 1013.25, 200.0)
        @test vg_cold > 0.0

        # All should be finite and positive
        @test isfinite(vg_stratosphere)
        @test isfinite(vg_hot)
        @test isfinite(vg_cold)
    end

    @testset "Edge Cases" begin
        # Very small particle (strong Cunningham correction)
        vg_tiny = vgrav_stokes(0.01, 2.5, 1013.25, 273.15)
        @test vg_tiny > 0.0
        @test isfinite(vg_tiny)

        # Very large particle (strong Reynolds correction)
        vg_huge = vgrav_corrected(1000.0, 2.5, 1013.25, 273.15, tol=0.01)
        @test vg_huge > 0.0
        @test isfinite(vg_huge)

        # Low density particle (barely denser than air)
        vg_light = vgrav_stokes(10.0, 0.002, 1013.25, 273.15)  # ρ ~ ρ_air
        @test vg_light > 0.0
        @test vg_light < 0.001  # Very slow settling
    end

    @testset "Table Size Variations" begin
        props = [ParticleProperties(diameter_μm=1.0, density_gcm3=2.5)]

        # Smaller table
        tables_small = build_vgrav_tables(props, numtemp=11, numpres=11)
        @test size(tables_small.vgtable) == (11, 11, 1)

        # Larger table
        tables_large = build_vgrav_tables(props, numtemp=81, numpres=51)
        @test size(tables_large.vgtable) == (81, 51, 1)

        # Interpolation should still work
        vg = interpolate_vgrav(tables_small, 1, 500.0, 273.0)
        @test vg > 0.0
    end

    @testset "Multiple Component Interpolation" begin
        props = [
            ParticleProperties(diameter_μm=1.0, density_gcm3=2.5),
            ParticleProperties(diameter_μm=10.0, density_gcm3=2.5),
            ParticleProperties(diameter_μm=100.0, density_gcm3=3.0)
        ]

        tables = build_vgrav_tables(props)

        # Interpolate each component
        T = 273.15
        P = 850.0

        vg1 = interpolate_vgrav(tables, 1, P, T)
        vg2 = interpolate_vgrav(tables, 2, P, T)
        vg3 = interpolate_vgrav(tables, 3, P, T)

        @test vg2 > vg1  # Larger particle settles faster
        @test vg3 > vg2  # Even larger and denser
    end
end

println("✓ All vgravtables tests passed!")
