# Supplemental tests for particles.jl: set_rad!, component-aware radioactive ops,
# get_set_rad!, flush_away_denormal!.

using Test

@testset "Particle supplemental: component ops + denormal flush" begin

    @testset "Particle component-aware ops" begin
        p = Particle(0.0, 0.0, 0.5, 0.0f0, 100.0f0, 1.0f0, 0.0f0, Int16(1))
        @test get_rad(p, 1) == 100.0f0

        # set_rad! returns the new value (per docstring).
        new_val = set_rad!(p, 250.0f0)
        @test new_val == 250.0f0
        @test get_rad(p) == 250.0f0

        # add_rad! returns the new total.
        total = add_rad!(p, 10.0f0)
        @test total == 260.0f0
        @test get_rad(p) == 260.0f0

        # scale_rad! returns absolute amount removed.
        removed = scale_rad!(p, 0.25f0)
        @test get_rad(p) == 65.0f0
        @test removed ≈ 195.0f0

        # get_set_rad! returns previous value and writes the new one.
        prev_total = get_set_rad!(p, 0.0f0)
        @test prev_total == 65.0f0
        @test get_rad(p) == 0.0f0
        @test !is_active(p)
    end

    @testset "set_rad!/add_rad!/scale_rad! component-indexed forms" begin
        p = Particle(0.0, 0.0, 0.5, 0.0f0, 50.0f0, 1.0f0, 0.0f0, Int16(1))

        new_val = set_rad!(p, 1, 200.0f0)
        @test new_val == 200.0f0
        @test get_rad(p, 1) == 200.0f0

        total = add_rad!(p, 1, 25.0f0)
        @test total == 225.0f0
        @test get_rad(p, 1) == 225.0f0

        removed = scale_rad!(p, 1, 0.5f0)
        @test removed ≈ 112.5f0
        @test get_rad(p, 1) ≈ 112.5f0
    end

    @testset "inactivate! returns previous active state" begin
        p_live = Particle(0.0, 0.0, 0.0, 0.0f0, 100.0f0, 1.0f0, 0.0f0, Int16(1))
        @test is_active(p_live)
        was_active = inactivate!(p_live)
        @test was_active == true
        @test !is_active(p_live)

        # Inactivating an already-inactive particle returns false.
        again = inactivate!(p_live)
        @test again == false
    end

    @testset "flush_away_denormal! zeroes very small radioactivity" begin
        # Threshold is NUMERIC_LIMIT_RAD = 1e-35.
        # 1e-37 is below the threshold and should be flushed to zero.
        p = Particle(0.0, 0.0, 0.0, 0.0f0, Float32(1e-37), 1.0f0, 0.0f0, Int16(1))
        flush_away_denormal!(p)
        @test get_rad(p) == 0.0f0

        # Normal-magnitude rad is preserved.
        p_normal = Particle(0.0, 0.0, 0.0, 0.0f0, 100.0f0, 1.0f0, 0.0f0, Int16(1))
        flush_away_denormal!(p_normal)
        @test get_rad(p_normal) == 100.0f0
    end
end

println("✓ All Particle supplemental tests passed!")
