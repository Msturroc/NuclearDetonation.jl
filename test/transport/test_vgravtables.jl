# Tests for vgravtables.jl: Gravitational Settling Velocity Tables
# Validates the Julia implementation of gravitational settling calculations.

using Test

@testset "vgravtables: Gravitational Settling" begin

    @testset "Air Viscosity (ReferenceViscosity)" begin
        model = ReferenceViscosity()

        # At 0 deg C (273.15 K) - reference formula
        eta_0C = air_viscosity(273.15, model)
        @test eta_0C ≈ 1.733e-4 rtol=0.01  # g/(cm*s) i.e. Poise

        # At 20 deg C (293.15 K) - should be higher than 0 deg C
        eta_20C = air_viscosity(293.15, model)
        @test eta_20C > eta_0C

        # At -40 deg C (233.15 K) - should be lower than 0 deg C
        eta_minus40C = air_viscosity(233.15, model)
        @test eta_minus40C < eta_0C

        # Temperature dependence: viscosity increases with temperature
        @test air_viscosity(300.0, model) > air_viscosity(250.0, model)
    end

    @testset "Air Viscosity (SutherlandViscosity)" begin
        model = SutherlandViscosity()

        # At reference temperature T0 = 273.15 K, should match mu0 * 10
        eta_ref = air_viscosity(273.15, model)
        @test eta_ref ≈ 1.716e-4 rtol=0.01  # Poise

        # Temperature dependence: viscosity increases with temperature
        @test air_viscosity(300.0, model) > air_viscosity(250.0, model)

        # Both models should agree within a few per cent at standard conditions
        eta_reference = air_viscosity(273.15, ReferenceViscosity())
        @test eta_ref ≈ eta_reference rtol=0.02
    end

    @testset "Air Density (ReferenceViscosity)" begin
        model = ReferenceViscosity()

        # At sea level (1013.25 hPa) and 0 deg C (273.15 K)
        # ReferenceViscosity expects P in hPa
        rho_sealevel = air_density(1013.25, 273.15, model)
        @test rho_sealevel ≈ 1.293e-3 rtol=0.01  # g/cm^3

        # Higher pressure -> higher density
        @test air_density(1013.25, 273.15, model) > air_density(500.0, 273.15, model)

        # Higher temperature -> lower density
        @test air_density(1013.25, 273.15, model) > air_density(1013.25, 300.0, model)

        # Low pressure (high altitude)
        rho_high_alt = air_density(300.0, 250.0, model)
        @test rho_high_alt < rho_sealevel
    end

    @testset "Air Density (SutherlandViscosity)" begin
        model = SutherlandViscosity()

        # At sea level (101325 Pa) and 0 deg C (273.15 K)
        # SutherlandViscosity expects P in Pa
        rho_sealevel = air_density(101325.0, 273.15, model)
        @test rho_sealevel ≈ 1.293e-3 rtol=0.01  # g/cm^3

        # Both models should give same density at equivalent conditions
        rho_reference = air_density(1013.25, 273.15, ReferenceViscosity())
        @test rho_sealevel ≈ rho_reference rtol=0.01
    end

    @testset "Cunningham Slip Correction" begin
        # Large particles (>> mean free path): C approx 1
        C_large = cunningham_factor(100.0)  # 100 um
        @test C_large ≈ 1.0 atol=0.1

        # Small particles (~ mean free path): significant correction
        C_small = cunningham_factor(0.1)  # 0.1 um
        @test C_small > 2.0  # Strong slip effect

        # Intermediate particles
        C_mid = cunningham_factor(1.0)  # 1 um
        @test C_mid > 1.0
        @test C_mid < 2.0

        # Monotonic: C decreases as particle size increases
        @test cunningham_factor(0.1) > cunningham_factor(1.0)
        @test cunningham_factor(1.0) > cunningham_factor(10.0)
        @test cunningham_factor(10.0) > cunningham_factor(100.0)

        # At very small dp, correction should be very large
        C_tiny = cunningham_factor(0.01)
        @test C_tiny > 10.0
    end

    @testset "Corrected Settling Velocity" begin
        model = ReferenceViscosity()

        # Small particle (1 um): Reynolds correction should be negligible
        # vgrav_corrected(dp, rp, P_pa, T, rho_p_kg_m3, model)
        dp_small = 1.0
        rp = 2.5           # g/cm^3
        rho_p = 2500.0      # kg/m^3
        P_pa = 101325.0     # Pa
        T = 273.15          # K

        vg_small = vgrav_corrected(dp_small, rp, P_pa, T, rho_p, model)
        @test vg_small > 0.0
        # For a 1 um particle, settling velocity should be very small (order of um/s -> m/s ~ 1e-5)
        @test vg_small < 0.001  # m/s

        # Large particle (100 um): correction should be more significant
        dp_large = 100.0
        vg_large = vgrav_corrected(dp_large, rp, P_pa, T, rho_p, model)
        @test vg_large > 0.0
        @test vg_large > vg_small  # Larger particles settle faster

        # Larger particles settle faster: monotonic ordering
        vg_10 = vgrav_corrected(10.0, rp, P_pa, T, rho_p, model)
        @test vg_10 > vg_small
        @test vg_large > vg_10

        # Convergence test: should not error
        @test isfinite(vg_large)
    end

    @testset "Corrected Velocity: Sutherland vs Reference" begin
        # Both models should give similar results at standard conditions
        dp = 10.0
        rp = 2.5
        rho_p = 2500.0
        P_pa = 101325.0
        T = 273.15

        vg_reference = vgrav_corrected(dp, rp, P_pa, T, rho_p, ReferenceViscosity())
        vg_sutherland = vgrav_corrected(dp, rp, P_pa, T, rho_p, SutherlandViscosity())

        # Should agree within a few per cent
        @test vg_reference ≈ vg_sutherland rtol=0.05
    end

    @testset "ParticleProperties Construction" begin
        props = ParticleProperties(diameter_μm=1.0, density_gcm3=2.5)
        @test props.diameter_μm == 1.0
        @test props.density_gcm3 == 2.5
    end

    @testset "Build VGrav Tables" begin
        # Single component
        props = [ParticleProperties(diameter_μm=10.0, density_gcm3=2.5)]
        tables = build_vgrav_tables(props; model=ReferenceViscosity())

        # Tables is a Dict{Int, Array{Float64, 2}}
        @test tables isa Dict{Int, Array{Float64, 2}}
        @test haskey(tables, 1)
        @test size(tables[1]) == (25, 41)  # (NUMPRES_VG, NUMTEMP_VG)

        # All velocities should be positive
        @test all(tables[1] .> 0.0)

        # Velocities should be in reasonable range (m/s)
        @test all(tables[1] .< 1.0)     # Less than 1 m/s for typical particles
        @test all(tables[1] .> 1e-6)    # Greater than 1 um/s
    end

    @testset "Build Tables: Multiple Components" begin
        props = [
            ParticleProperties(diameter_μm=1.0, density_gcm3=2.5),   # Fine
            ParticleProperties(diameter_μm=10.0, density_gcm3=2.5),  # Medium
            ParticleProperties(diameter_μm=100.0, density_gcm3=2.5)  # Coarse
        ]

        tables = build_vgrav_tables(props; model=ReferenceViscosity())

        @test length(tables) == 3
        @test haskey(tables, 1) && haskey(tables, 2) && haskey(tables, 3)

        # Larger particles settle faster at same conditions
        # Compare at mid-range indices: table[ip, it]
        ip_mid = 13  # Mid pressure
        it_mid = 21  # Mid temperature

        vg_1um = tables[1][ip_mid, it_mid]
        vg_10um = tables[2][ip_mid, it_mid]
        vg_100um = tables[3][ip_mid, it_mid]

        @test vg_10um > vg_1um
        @test vg_100um > vg_10um
    end

    @testset "Table Interpolation" begin
        props = [ParticleProperties(diameter_μm=10.0, density_gcm3=2.5)]
        tables = build_vgrav_tables(props; model=ReferenceViscosity())

        # Interpolate at a central grid point
        # The grid constants are internal to GravitationalSettling, so
        # we test with physically reasonable mid-range values.
        vg_mid = interpolate_vgrav(tables, 1, 600.0, 273.0)
        @test vg_mid > 0.0
        @test isfinite(vg_mid)

        # Interpolate between grid points (slightly off-grid values)
        vg_interp = interpolate_vgrav(tables, 1, 605.0, 274.0)
        @test vg_interp > 0.0
        @test isfinite(vg_interp)

        # Test clamping at boundaries (out of range values should still return valid velocities)
        vg_clamped = interpolate_vgrav(tables, 1, 1000.0, 100.0)
        @test vg_clamped > 0.0
        @test isfinite(vg_clamped)
    end

    @testset "Physical Trends in Tables" begin
        props = [ParticleProperties(diameter_μm=10.0, density_gcm3=2.5)]
        tables = build_vgrav_tables(props; model=ReferenceViscosity())

        # At constant pressure (mid-range index), settling velocity should vary with temperature
        ip = 13  # Mid-pressure level
        vg_cold = tables[1][ip, 5]   # Cold end
        vg_warm = tables[1][ip, 35]  # Warm end

        # Temperature affects both density and viscosity; net effect should differ
        @test vg_cold != vg_warm
        @test abs(vg_warm - vg_cold) / vg_cold > 0.1  # >10% difference

        # At constant temperature (mid-range index), settling velocity should vary with pressure
        it = 21  # Mid-temperature level
        vg_low_p = tables[1][5, it]    # Low pressure (high altitude)
        vg_high_p = tables[1][20, it]  # High pressure (low altitude)

        # Lower pressure -> less dense air -> faster settling
        @test vg_low_p > vg_high_p
    end

    @testset "Realistic Particle Examples" begin
        model = ReferenceViscosity()
        P_pa = 101325.0
        T = 288.15

        # Cs-137 aerosol: ~0.5 um, density ~2.5 g/cm^3
        vg_cs137 = vgrav_corrected(0.5, 2.5, P_pa, T, 2500.0, model)
        @test vg_cs137 > 0.0
        @test vg_cs137 < 1e-4  # m/s, very slow settling

        # Large dust particle: 50 um, density ~2.65 g/cm^3
        vg_dust = vgrav_corrected(50.0, 2.65, P_pa, T, 2650.0, model)
        @test vg_dust > vg_cs137 * 100  # Much faster

        # Sand grain: 500 um, density ~2.65 g/cm^3
        vg_sand = vgrav_corrected(500.0, 2.65, P_pa, T, 2650.0, model)
        @test vg_sand > vg_dust  # Even faster
    end

    @testset "Extreme Conditions" begin
        model = ReferenceViscosity()
        rp = 2.5
        rho_p = 2500.0

        # Very high altitude (low pressure, cold) - P in Pa
        vg_stratosphere = vgrav_corrected(1.0, rp, 5000.0, 220.0, rho_p, model)
        @test vg_stratosphere > 0.0
        @test isfinite(vg_stratosphere)

        # Very hot conditions
        vg_hot = vgrav_corrected(1.0, rp, 101325.0, 350.0, rho_p, model)
        @test vg_hot > 0.0
        @test isfinite(vg_hot)

        # Very cold conditions
        vg_cold = vgrav_corrected(1.0, rp, 101325.0, 200.0, rho_p, model)
        @test vg_cold > 0.0
        @test isfinite(vg_cold)
    end

    @testset "Edge Cases" begin
        model = ReferenceViscosity()
        rp = 2.5
        rho_p = 2500.0

        # Very small particle (strong Cunningham correction)
        vg_tiny = vgrav_corrected(0.01, rp, 101325.0, 273.15, rho_p, model)
        @test vg_tiny > 0.0
        @test isfinite(vg_tiny)

        # Very large particle (strong Reynolds correction)
        vg_huge = vgrav_corrected(1000.0, rp, 101325.0, 273.15, rho_p, model; tol=0.01)
        @test vg_huge > 0.0
        @test isfinite(vg_huge)

        # Low density particle (barely denser than air)
        vg_light = vgrav_corrected(10.0, 0.002, 101325.0, 273.15, 2.0, model)
        @test vg_light > 0.0
        @test vg_light < 0.0001  # Very slow settling (m/s)
    end

    @testset "Multiple Component Interpolation" begin
        props = [
            ParticleProperties(diameter_μm=1.0, density_gcm3=2.5),
            ParticleProperties(diameter_μm=10.0, density_gcm3=2.5),
            ParticleProperties(diameter_μm=100.0, density_gcm3=3.0)
        ]

        tables = build_vgrav_tables(props; model=ReferenceViscosity())

        # Interpolate each component at the same conditions
        T = 273.15  # K
        P = 850.0   # hPa (interpolate_vgrav takes hPa)

        vg1 = interpolate_vgrav(tables, 1, P, T)
        vg2 = interpolate_vgrav(tables, 2, P, T)
        vg3 = interpolate_vgrav(tables, 3, P, T)

        @test vg2 > vg1  # Larger particle settles faster
        @test vg3 > vg2  # Even larger and denser
    end
end
