# Tests for defaults.jl: optimised parameter presets

using Test

const _Tx_d = NuclearDetonation.Transport

@testset "defaults: optimised configuration presets" begin

    @testset "nancy_optimised_config" begin
        params = _Tx_d.nancy_optimised_config()
        @test params.hanna_config.apply_turbulence == true
        @test params.hanna_config.use_cbl == true

        # Particle size distribution: bimodal, fine fraction ~86.5%.
        psd = params.particle_size_config
        @test 0 < psd.frac_fine < 1
        @test psd.d_median_fine_μm > 0
        @test psd.d_median_coarse_μm > 0
        @test psd.sigma_g_fine > 1.0   # geometric std must be > 1
        @test psd.sigma_g_coarse > 1.0

        # Layer fractions sum to 1 (by construction) and are positive.
        lf = params.layer_fractions
        @test lf.lower > 0 && lf.middle > 0 && lf.upper > 0
        @test lf.lower + lf.middle + lf.upper ≈ 1.0 atol=1e-12

        # Physics scales positive and finite.
        ps = params.physics_scales
        for k in propertynames(ps)
            v = getproperty(ps, k)
            @test isfinite(v) && v > 0
        end

        # Activity ~48 PBq for 24 kT Nancy yield.
        @test params.activity_Bq > 1e15
    end

    @testset "etex_optimised_config" begin
        params = _Tx_d.etex_optimised_config()
        @test params.hanna_config.apply_turbulence == true
        @test params.hanna_config.use_cbl == true

        ps = params.physics_scales
        for k in propertynames(ps)
            v = getproperty(ps, k)
            @test isfinite(v) && v > 0
        end

        # ETEX is a gas tracer — no particle size or activity fields.
        @test !hasproperty(params, :particle_size_config)
        @test !hasproperty(params, :activity_Bq)
    end
end

println("✓ All defaults tests passed!")
