# Tests for deposition.jl: Dry and Wet Deposition
# Based on SNAP drydep.f90 and wetdep.f90

using Test
using NuclearDetonation.Transport:
    LandUseClass, SeasonCategory,
    WATER, DECIDUOUS_FOREST, CONIFEROUS_FOREST, MIXED_FOREST,
    GRASSLAND, CROPLAND, URBAN, BARE_SOIL, SNOW_ICE,
    WINTER, SPRING, SUMMER, AUTUMN,
    DryDepositionParams, WetDepositionParams,
    compute_air_density, compute_friction_velocity, compute_monin_obukhov_length,
    aerodynamic_resistance, cunningham_slip_factor, brownian_diffusivity,
    get_characteristic_radius, surface_resistance,
    compute_dry_deposition_velocity, compute_wet_scavenging_coefficient,
    apply_dry_deposition!, apply_wet_deposition!

@testset "deposition: Dry and Wet Deposition" begin

    @testset "Deposition Parameters" begin
        # Dry deposition params
        dry_params = DryDepositionParams(
            1e-6,  # 1 μm diameter
            2500.0,  # 2.5 g/cm³ density
            30.0,  # 30 m reference height
            fill(0.1, 10, 10),  # 10 cm roughness (grassland)
            fill(GRASSLAND, 10, 10)
        )
        @test dry_params.particle_diameter == 1e-6
        @test dry_params.particle_density == 2500.0
        @test size(dry_params.roughness_length) == (10, 10)
        @test size(dry_params.land_use) == (10, 10)

        # Wet deposition params with defaults
        wet_params = WetDepositionParams(1e-6, 2500.0)
        @test wet_params.washout_coefficient == 2.0e-5
        @test wet_params.rainout_coefficient == 1.0e-4
        @test wet_params.precipitation_threshold == 0.01

        # Custom wet params
        wet_params_custom = WetDepositionParams(
            1e-5, 3000.0,
            washout_coef=1.0e-4,
            rainout_coef=5.0e-4,
            precip_threshold=0.1
        )
        @test wet_params_custom.washout_coefficient == 1.0e-4
        @test wet_params_custom.precipitation_threshold == 0.1
    end

    @testset "Air Density" begin
        # Standard conditions (15°C, 1013.25 hPa)
        T = 273.15 + 15.0  # K
        P = 101325.0  # Pa
        rho = compute_air_density(P, T)
        @test rho ≈ 1.225 rtol=0.01  # Standard air density

        # High altitude (reduced pressure and temperature)
        T_high = 250.0  # K
        P_high = 50000.0  # Pa (~5.5 km altitude)
        rho_high = compute_air_density(P_high, T_high)
        @test rho_high < rho  # Lower density at altitude
        @test rho_high ≈ 0.696 rtol=0.01
    end

    @testset "Friction Velocity" begin
        # Typical surface stress
        tau_x = 0.1  # N/m²
        tau_y = 0.0
        rho = 1.2  # kg/m³
        u_star = compute_friction_velocity(tau_x, tau_y, rho)
        @test u_star ≈ sqrt(0.1 / 1.2) rtol=1e-10

        # With both components
        tau_x2 = 0.1
        tau_y2 = 0.1
        u_star2 = compute_friction_velocity(tau_x2, tau_y2, rho)
        @test u_star2 ≈ sqrt(sqrt(0.1^2 + 0.1^2) / 1.2) rtol=1e-10
        @test u_star2 > u_star  # Magnitude increased

        # Zero stress
        u_star_zero = compute_friction_velocity(0.0, 0.0, rho)
        @test u_star_zero == 0.0
    end

    @testset "Monin-Obukhov Length" begin
        u_star = 0.4  # m/s
        T = 290.0  # K
        rho = 1.2  # kg/m³

        # Unstable (positive heat flux)
        H_unstable = 100.0  # W/m²
        L_unstable = compute_monin_obukhov_length(u_star, T, H_unstable, rho)
        @test L_unstable < 0  # Negative for unstable

        # Stable (negative heat flux)
        H_stable = -50.0  # W/m²
        L_stable = compute_monin_obukhov_length(u_star, T, H_stable, rho)
        @test L_stable > 0  # Positive for stable

        # Neutral (near-zero heat flux)
        H_neutral = 0.001  # W/m²
        L_neutral = compute_monin_obukhov_length(u_star, T, H_neutral, rho)
        @test isinf(L_neutral)  # Infinite for neutral
    end

    @testset "Aerodynamic Resistance" begin
        u_star = 0.5  # m/s
        z0 = 0.1  # m
        z_ref = 30.0  # m

        # Neutral conditions
        L_neutral = Inf
        Ra_neutral = aerodynamic_resistance(L_neutral, u_star, z0, z_ref)
        @test Ra_neutral > 0
        @test Ra_neutral ≈ log(z_ref/z0) / (0.4 * u_star) rtol=0.01

        # Unstable conditions (should have lower resistance)
        L_unstable = -100.0  # m
        Ra_unstable = aerodynamic_resistance(L_unstable, u_star, z0, z_ref)
        @test Ra_unstable > 0
        @test Ra_unstable < Ra_neutral  # Enhanced mixing in unstable atmosphere

        # Stable conditions (should have higher resistance)
        L_stable = 100.0  # m
        Ra_stable = aerodynamic_resistance(L_stable, u_star, z0, z_ref)
        @test Ra_stable > Ra_neutral  # Suppressed mixing in stable atmosphere

        # Very calm conditions
        Ra_calm = aerodynamic_resistance(Inf, 1e-11, z0, z_ref)
        @test Ra_calm ≈ 1e10  # Very large resistance
    end

    @testset "Cunningham Slip Factor" begin
        # Standard conditions
        T = 288.15  # K (15°C)

        # Small particle (strong slip effect)
        d_small = 0.01e-6  # 10 nm
        Cc_small = cunningham_slip_factor(d_small, T)
        @test Cc_small > 1.0
        @test Cc_small > 10.0  # Significant slip correction

        # Large particle (weak slip effect)
        d_large = 10e-6  # 10 μm
        Cc_large = cunningham_slip_factor(d_large, T)
        @test Cc_large > 1.0
        @test Cc_large < 1.1  # Small correction

        # Slip factor increases as particle size decreases
        d_mid = 1e-6  # 1 μm
        Cc_mid = cunningham_slip_factor(d_mid, T)
        @test Cc_small > Cc_mid > Cc_large
    end

    @testset "Brownian Diffusivity" begin
        T = 293.15  # K
        rho = 1.2  # kg/m³

        # Small particle (high diffusivity)
        d_small = 0.1e-6  # 100 nm
        D_small = brownian_diffusivity(d_small, T, rho)
        @test D_small > 0.0

        # Large particle (low diffusivity)
        d_large = 10e-6  # 10 μm
        D_large = brownian_diffusivity(d_large, T, rho)
        @test D_large > 0.0
        @test D_small > D_large  # Smaller particles diffuse faster

        # Diffusivity scales as 1/d (approximately, with Cunningham correction)
        d1 = 1e-6
        d2 = 2e-6
        D1 = brownian_diffusivity(d1, T, rho)
        D2 = brownian_diffusivity(d2, T, rho)
        @test D1 > D2
    end

    @testset "Characteristic Radius" begin
        # Forest types
        @test get_characteristic_radius(CONIFEROUS_FOREST, SUMMER) == 5.0
        @test get_characteristic_radius(DECIDUOUS_FOREST, SUMMER) == 2.0
        @test get_characteristic_radius(DECIDUOUS_FOREST, WINTER) == 10.0

        # Grassland/cropland seasonal variation
        @test get_characteristic_radius(GRASSLAND, SUMMER) == 2.0
        @test get_characteristic_radius(GRASSLAND, WINTER) == 10.0
        @test get_characteristic_radius(CROPLAND, SPRING) == 2.0

        # Urban and bare soil
        @test get_characteristic_radius(URBAN, SUMMER) == 10.0
        @test get_characteristic_radius(BARE_SOIL, SUMMER) == 50.0

        # Water (special case)
        @test isnan(get_characteristic_radius(WATER, SUMMER))
    end

    @testset "Surface Resistance" begin
        d = 1e-6  # 1 μm
        u_star = 0.5  # m/s
        vg = 0.01  # 1 cm/s settling velocity
        T = 293.15  # K
        rho = 1.2  # kg/m³
        D = brownian_diffusivity(d, T, rho)

        # Forest (high deposition)
        Rs_forest = surface_resistance(CONIFEROUS_FOREST, SUMMER, d, u_star, D, vg)
        @test Rs_forest > 0.0

        # Grassland
        Rs_grass = surface_resistance(GRASSLAND, SUMMER, d, u_star, D, vg)
        @test Rs_grass > 0.0

        # Bare soil (low deposition)
        Rs_bare = surface_resistance(BARE_SOIL, SUMMER, d, u_star, D, vg)
        @test Rs_bare > Rs_grass  # Higher resistance (lower deposition)

        # Water surface (special formulation)
        Rs_water = surface_resistance(WATER, SUMMER, d, u_star, D, vg)
        @test Rs_water > 0.0
        @test isfinite(Rs_water)

        # Calm conditions
        Rs_calm = surface_resistance(GRASSLAND, SUMMER, d, 1e-11, D, vg)
        @test Rs_calm ≈ 1e10  # Very large resistance
    end

    @testset "Dry Deposition Velocity" begin
        # Create simple test grid
        nx, ny = 5, 5
        params = DryDepositionParams(
            1e-6, 2500.0, 30.0,
            fill(0.1, nx, ny),
            fill(GRASSLAND, nx, ny)
        )

        u_star = fill(0.4, nx, ny)
        L = fill(Inf, nx, ny)  # Neutral
        T = fill(293.15, nx, ny)
        P = fill(101325.0, nx, ny)
        vg = 0.01  # m/s

        vd = compute_dry_deposition_velocity(params, u_star, L, T, P, SUMMER, vg)

        @test size(vd) == (nx, ny)
        @test all(vd .> 0.0)
        @test all(vd .>= vg)  # Should be at least settling velocity

        # Unstable atmosphere (enhanced deposition)
        L_unstable = fill(-50.0, nx, ny)
        vd_unstable = compute_dry_deposition_velocity(params, u_star, L_unstable, T, P, SUMMER, vg)
        @test mean(vd_unstable) > mean(vd)

        # Different land uses
        params_forest = DryDepositionParams(
            1e-6, 2500.0, 30.0,
            fill(0.5, nx, ny),  # Higher roughness
            fill(CONIFEROUS_FOREST, nx, ny)
        )
        vd_forest = compute_dry_deposition_velocity(params_forest, u_star, L, T, P, SUMMER, vg)
        @test mean(vd_forest) > mean(vd)  # Forest has higher deposition
    end

    @testset "Wet Scavenging Coefficient" begin
        params = WetDepositionParams(1e-6, 2500.0)

        # Below threshold
        Lambda_zero = compute_wet_scavenging_coefficient(0.001, params, false)
        @test Lambda_zero == 0.0

        # Below-cloud scavenging (washout)
        precip_low = 1.0  # mm/h
        Lambda_low = compute_wet_scavenging_coefficient(precip_low, params, false)
        @test Lambda_low > 0.0

        precip_high = 10.0  # mm/h
        Lambda_high = compute_wet_scavenging_coefficient(precip_high, params, false)
        @test Lambda_high > Lambda_low  # Scales with precipitation

        # Power law check: Λ ∝ P^0.79
        ratio_lambda = Lambda_high / Lambda_low
        ratio_precip = (precip_high / precip_low)^0.79
        @test ratio_lambda ≈ ratio_precip rtol=0.01

        # In-cloud scavenging (rainout)
        Lambda_cloud = compute_wet_scavenging_coefficient(5.0, params, true)
        @test Lambda_cloud == params.rainout_coefficient
        @test Lambda_cloud > 0.0
    end

    @testset "Apply Dry Deposition" begin
        # Single component
        mass = [1e10]
        vd = 0.01  # 1 cm/s
        dt = 600.0  # 10 minutes
        h_mix = 1000.0  # m

        mass_initial = copy(mass)
        deposited = apply_dry_deposition!(mass, vd, dt, h_mix)

        @test deposited > 0.0
        @test mass[1] < mass_initial[1]
        @test mass[1] + deposited ≈ mass_initial[1]  # Conservation

        # Exponential decay check
        k_dep = vd / h_mix
        expected_remaining = mass_initial[1] * exp(-k_dep * dt)
        @test mass[1] ≈ expected_remaining rtol=1e-10

        # Multiple components
        mass_multi = [1e10, 5e9, 2e9]
        mass_multi_initial = copy(mass_multi)
        deposited_multi = apply_dry_deposition!(mass_multi, vd, dt, h_mix)

        @test all(mass_multi .< mass_multi_initial)
        @test sum(mass_multi) + deposited_multi ≈ sum(mass_multi_initial)

        # Zero deposition
        mass_zero = [1e10]
        deposited_zero = apply_dry_deposition!(mass_zero, 0.0, dt, h_mix)
        @test deposited_zero == 0.0
        @test mass_zero[1] == 1e10
    end

    @testset "Apply Wet Deposition" begin
        # Single component
        mass = [1e10]
        Lambda = 1e-4  # s⁻¹
        dt = 600.0  # 10 minutes

        mass_initial = copy(mass)
        deposited = apply_wet_deposition!(mass, Lambda, dt)

        @test deposited > 0.0
        @test mass[1] < mass_initial[1]
        @test mass[1] + deposited ≈ mass_initial[1]  # Conservation

        # Exponential scavenging check
        expected_remaining = mass_initial[1] * exp(-Lambda * dt)
        @test mass[1] ≈ expected_remaining rtol=1e-10

        # Strong scavenging (heavy rain)
        mass_heavy = [1e10]
        Lambda_heavy = 1e-3  # s⁻¹ (10x stronger)
        deposited_heavy = apply_wet_deposition!(mass_heavy, Lambda_heavy, dt)
        @test deposited_heavy > deposited  # More removed

        # Multiple components
        mass_multi = [1e10, 5e9, 2e9]
        mass_multi_initial = copy(mass_multi)
        deposited_multi = apply_wet_deposition!(mass_multi, Lambda, dt)

        @test all(mass_multi .< mass_multi_initial)
        @test sum(mass_multi) + deposited_multi ≈ sum(mass_multi_initial)

        # Zero scavenging
        mass_zero = [1e10]
        deposited_zero = apply_wet_deposition!(mass_zero, 0.0, dt)
        @test deposited_zero == 0.0
        @test mass_zero[1] == 1e10
    end

    @testset "Physical Consistency" begin
        # Deposition velocity must be positive
        params = DryDepositionParams(
            1e-6, 2500.0, 30.0,
            fill(0.1, 3, 3),
            fill(GRASSLAND, 3, 3)
        )
        u_star = fill(0.3, 3, 3)
        L = fill(Inf, 3, 3)
        T = fill(293.15, 3, 3)
        P = fill(101325.0, 3, 3)
        vd = compute_dry_deposition_velocity(params, u_star, L, T, P, SUMMER, 0.001)
        @test all(vd .> 0.0)
        @test all(isfinite.(vd))

        # Scavenging coefficient must be non-negative
        wet_params = WetDepositionParams(1e-6, 2500.0)
        for precip in [0.0, 0.5, 1.0, 5.0, 20.0]
            Lambda = compute_wet_scavenging_coefficient(precip, wet_params, false)
            @test Lambda >= 0.0
            @test isfinite(Lambda)
        end

        # Deposition removes mass but doesn't create it
        mass = [1e10, 5e9]
        mass_initial = copy(mass)
        apply_dry_deposition!(mass, 0.01, 600.0, 1000.0)
        @test all(mass .<= mass_initial)
        @test all(mass .>= 0.0)

        # Longer time = more deposition
        mass1 = [1e10]
        mass2 = [1e10]
        apply_dry_deposition!(mass1, 0.01, 300.0, 1000.0)
        apply_dry_deposition!(mass2, 0.01, 600.0, 1000.0)
        @test mass2[1] < mass1[1]
    end

    @testset "Size Dependence" begin
        # Test deposition velocity vs particle size
        nx, ny = 3, 3
        u_star = fill(0.4, nx, ny)
        L = fill(Inf, nx, ny)
        T = fill(293.15, nx, ny)
        P = fill(101325.0, nx, ny)

        # Small particles (Brownian diffusion dominated)
        params_small = DryDepositionParams(
            0.1e-6, 2500.0, 30.0,  # 0.1 μm
            fill(0.1, nx, ny),
            fill(GRASSLAND, nx, ny)
        )
        vd_small = compute_dry_deposition_velocity(params_small, u_star, L, T, P, SUMMER, 0.0001)

        # Large particles (impaction/settling dominated)
        params_large = DryDepositionParams(
            10e-6, 2500.0, 30.0,  # 10 μm
            fill(0.1, nx, ny),
            fill(GRASSLAND, nx, ny)
        )
        vd_large = compute_dry_deposition_velocity(params_large, u_star, L, T, P, SUMMER, 0.1)

        # Both should have positive deposition
        @test mean(vd_small) > 0.0
        @test mean(vd_large) > 0.0

        # Large particles typically have higher deposition (in submicron-to-supermicron range)
        @test mean(vd_large) > mean(vd_small)
    end

    @testset "Land Use Dependence" begin
        # Test different land uses
        nx, ny = 3, 3
        params_base = (1e-6, 2500.0, 30.0)
        u_star = fill(0.4, nx, ny)
        L = fill(Inf, nx, ny)
        T = fill(293.15, nx, ny)
        P = fill(101325.0, nx, ny)
        vg = 0.01

        land_uses = [CONIFEROUS_FOREST, GRASSLAND, BARE_SOIL]
        roughness = [0.5, 0.1, 0.01]  # Typical roughness lengths

        vd_values = Float64[]

        for (lu, z0) in zip(land_uses, roughness)
            params = DryDepositionParams(
                params_base...,
                fill(z0, nx, ny),
                fill(lu, nx, ny)
            )
            vd = compute_dry_deposition_velocity(params, u_star, L, T, P, SUMMER, vg)
            push!(vd_values, mean(vd))
        end

        # Forest should have highest deposition (rough surface, good collection)
        @test vd_values[1] > vd_values[2]  # Forest > Grassland
        @test vd_values[2] > vd_values[3]  # Grassland > Bare soil
    end

    @testset "Seasonal Dependence" begin
        # Deciduous forest in different seasons
        nx, ny = 3, 3
        params = DryDepositionParams(
            1e-6, 2500.0, 30.0,
            fill(0.5, nx, ny),
            fill(DECIDUOUS_FOREST, nx, ny)
        )
        u_star = fill(0.4, nx, ny)
        L = fill(Inf, nx, ny)
        T = fill(293.15, nx, ny)
        P = fill(101325.0, nx, ny)
        vg = 0.01

        vd_summer = compute_dry_deposition_velocity(params, u_star, L, T, P, SUMMER, vg)
        vd_winter = compute_dry_deposition_velocity(params, u_star, L, T, P, WINTER, vg)

        # Summer (with leaves) should have higher deposition than winter (bare branches)
        @test mean(vd_summer) > mean(vd_winter)
    end
end

println("✓ All deposition tests passed!")
