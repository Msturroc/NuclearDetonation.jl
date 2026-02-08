# Tests for decay.jl: Radioactive Decay Module
# Based on SNAP decay.f90

using Test

@testset "decay: Radioactive Decay" begin

    @testset "DecayParams Construction" begin
        # No decay (default)
        p1 = DecayParams()
        @test p1.kdecay == NoDecay
        @test p1.halftime_hours == 0.0
        @test p1.decayrate == 1.0

        # Exponential decay with half-life
        p2 = DecayParams(kdecay=ExponentialDecay, halftime_hours=8.02 * 24.0)  # I-131
        @test p2.kdecay == ExponentialDecay
        @test p2.halftime_hours ≈ 192.48

        # Bomb decay
        p3 = DecayParams(kdecay=BombDecay, halftime_hours=0.0)
        @test p3.kdecay == BombDecay
    end

    @testset "BombDecayState Construction" begin
        state = BombDecayState(total_time_s=0.0, bomb_time_s=0.0, has_components=false)
        @test state.total_time_s == 0.0
        @test state.bomb_time_s == 0.0
        @test state.has_components == false
    end

    @testset "Exponential Decay Rate Computation" begin
        # I-131: 8.02 day half-life
        params = [DecayParams(kdecay=ExponentialDecay, halftime_hours=8.02 * 24.0)]
        timestep_s = 3600.0  # 1 hour

        prepare_decay_rates!(params, timestep_s)

        # After 1 hour, activity should be: exp(-ln(2) * 1 / (8.02*24))
        expected_rate = exp(-log(2.0) * 1.0 / (8.02 * 24.0))
        @test params[1].decayrate ≈ expected_rate rtol=1e-10

        # Decay rate should be less than 1.0 (activity decreases)
        @test params[1].decayrate < 1.0

        # After multiple applications, should approach 0
        activity = 1.0e6  # 1 MBq
        for _ in 1:1000  # 1000 hours
            activity = apply_decay(activity, params[1])
        end
        # After ~1000 hours (5.2 half-lives), should be << initial
        @test activity < 1.0e6 * 0.5^5
    end

    @testset "Exponential Decay: Known Half-Lives" begin
        timestep_s = 3600.0  # 1 hour

        # Test several isotopes
        isotopes = [
            ("I-131", 8.02 * 24.0),          # 8.02 days
            ("Cs-137", 30.17 * 365.25 * 24.0),  # 30.17 years
            ("Xe-133", 5.243 * 24.0),        # 5.243 days
            ("Te-132", 3.204 * 24.0),        # 3.204 days
        ]

        for (name, halftime_hours) in isotopes
            params = [DecayParams(kdecay=ExponentialDecay, halftime_hours=halftime_hours)]
            prepare_decay_rates!(params, timestep_s)

            # After one half-life, activity should be 50%
            n_steps = Int(round(halftime_hours))  # Number of 1-hour steps
            activity = 1.0
            for _ in 1:n_steps
                prepare_decay_rates!(params, timestep_s)
                activity = apply_decay(activity, params[1])
            end

            @test activity ≈ 0.5 rtol=0.01  # Within 1% (timestep discretization)
        end
    end

    @testset "No Decay" begin
        params = [DecayParams(kdecay=NoDecay)]
        timestep_s = 3600.0

        prepare_decay_rates!(params, timestep_s)
        @test params[1].decayrate == 1.0

        # Activity should remain constant
        activity = 1.0e6
        for _ in 1:100
            activity = apply_decay(activity, params[1])
        end
        @test activity == 1.0e6
    end

    @testset "Bomb Decay: t^-1.2 Power Law" begin
        params = [DecayParams(kdecay=BombDecay)]
        bomb_state = BombDecayState(total_time_s=0.0, bomb_time_s=0.0)
        timestep_s = 3600.0  # 1 hour

        # Before H+1, no decay should occur
        for _ in 1:1  # First hour
            prepare_decay_rates!(params, timestep_s, bomb_state=bomb_state)
            @test params[1].decayrate == 1.0
        end

        # After H+1, decay should follow t^-1.2
        # At t=2hr (first step after H+1): rate = (2/1)^-1.2 = 0.435...
        prepare_decay_rates!(params, timestep_s, bomb_state=bomb_state)

        # Check that decay is happening (rate < 1.0)
        @test params[1].decayrate < 1.0
        @test params[1].decayrate > 0.0

        # Verify power law behavior
        # This second call (after H+1) computes decay from t=1hr to t=2hr
        # rate = (2/1)^-1.2 = 0.435...
        t1_hrs = 1.0
        t2_hrs = 2.0
        expected_rate = (t2_hrs / t1_hrs)^(-1.2)
        @test params[1].decayrate ≈ expected_rate rtol=1e-6
    end

    @testset "Bomb Decay: Glasstone 7-10 Rule" begin
        # The "7-10 rule": dose rate decreases by factor of 10 for every 7-fold time increase
        # R(t) = R₀ * t^-1.2
        # R(7t)/R(t) = (7t)^-1.2 / t^-1.2 = 7^-1.2 ≈ 0.1

        params = [DecayParams(kdecay=BombDecay)]
        bomb_state = BombDecayState(total_time_s=3600.0, bomb_time_s=0.0)  # Start at H+1
        timestep_s = 3600.0

        # Simulate 7 hours
        for _ in 1:7
            prepare_decay_rates!(params, timestep_s, bomb_state=bomb_state)
        end

        # Calculate total decay factor after 7x time increase (H+1 to H+7)
        # Should be approximately 10x decrease
        total_factor = 1.0
        bomb_state = BombDecayState(total_time_s=3600.0, bomb_time_s=0.0)
        for _ in 1:6  # 6 steps from t=1hr to t=7hr
            prepare_decay_rates!(params, timestep_s, bomb_state=bomb_state)
            total_factor *= params[1].decayrate
        end

        # 7^-1.2 ≈ 0.1 (the 7-10 rule)
        @test total_factor ≈ 7.0^(-1.2) rtol=0.01
        @test total_factor ≈ 0.1 rtol=0.05  # Within 5% of 1/10
    end

    @testset "apply_decay: Activity Reduction" begin
        param = DecayParams(kdecay=ExponentialDecay, halftime_hours=1.0)  # 1 hour half-life
        prepare_decay_rates!([param], 3600.0)  # 1 hour timestep

        activity = 1.0e6
        new_activity = apply_decay(activity, param)

        # After 1 half-life, should be 50%
        @test new_activity ≈ 5.0e5 rtol=1e-6
    end

    @testset "apply_decay!: Field Decay (In-Place)" begin
        param = DecayParams(kdecay=ExponentialDecay, halftime_hours=24.0)
        prepare_decay_rates!([param], 3600.0)  # 1 hour timestep

        # 2D deposition field
        field = ones(Float64, 10, 10) * 1.0e6

        apply_decay!(field, param)

        # All values should be multiplied by decayrate
        expected = exp(-log(2.0) * 1.0 / 24.0) * 1.0e6
        @test all(field .≈ expected)

        # Test with no decay
        param_nodecay = DecayParams(kdecay=NoDecay)
        field2 = ones(Float64, 10, 10) * 1.0e6
        apply_decay!(field2, param_nodecay)
        @test all(field2 .== 1.0e6)  # Unchanged
    end

    @testset "Multiple Components with Different Decay" begin
        # Simulate multiple isotopes
        params = [
            DecayParams(kdecay=ExponentialDecay, halftime_hours=8.02 * 24.0),  # I-131
            DecayParams(kdecay=ExponentialDecay, halftime_hours=30.17 * 365.25 * 24.0),  # Cs-137
            DecayParams(kdecay=NoDecay),  # Stable isotope
            DecayParams(kdecay=BombDecay),  # Bomb fallout
        ]

        bomb_state = BombDecayState(total_time_s=3600.0, bomb_time_s=0.0)
        timestep_s = 3600.0

        prepare_decay_rates!(params, timestep_s, bomb_state=bomb_state)

        # I-131 should decay faster than Cs-137
        @test params[1].decayrate < params[2].decayrate

        # Stable isotope should not decay
        @test params[3].decayrate == 1.0

        # Bomb fallout should decay (after H+1)
        @test params[4].decayrate < 1.0
    end

    @testset "Decay Rate Consistency" begin
        # Decay rate should be multiplicative: rate(2Δt) = rate(Δt)²
        param = DecayParams(kdecay=ExponentialDecay, halftime_hours=10.0)

        # Compute rate for Δt = 1 hour
        prepare_decay_rates!([param], 3600.0)
        rate_1hr = param.decayrate

        # Compute rate for Δt = 2 hours
        prepare_decay_rates!([param], 7200.0)
        rate_2hr = param.decayrate

        # Should satisfy: rate_2hr ≈ rate_1hr²
        @test rate_2hr ≈ rate_1hr^2 rtol=1e-10
    end

    @testset "Long-Term Decay Simulation" begin
        # Simulate 10 half-lives of I-131 (80.2 days)
        param = DecayParams(kdecay=ExponentialDecay, halftime_hours=8.02 * 24.0)
        timestep_s = 3600.0  # 1 hour
        n_steps = Int(round(10 * 8.02 * 24))  # 10 half-lives

        activity = 1.0e9  # 1 GBq initial
        for _ in 1:n_steps
            prepare_decay_rates!([param], timestep_s)
            activity = apply_decay(activity, param)
        end

        # After 10 half-lives: N = N₀ * (1/2)^10 = N₀ / 1024
        expected = 1.0e9 / 1024.0
        @test activity ≈ expected rtol=0.01
    end

    @testset "Bomb Decay: Delayed Start" begin
        # Test bomb detonation at t=10000s (not at t=0)
        params = [DecayParams(kdecay=BombDecay)]
        bomb_state = BombDecayState(total_time_s=0.0, bomb_time_s=10000.0)
        timestep_s = 3600.0

        # Before bomb time + 1 hour (H+1 = 13600s), no decay
        # After 3 steps: t=10800s < 13600s, so still no decay
        for _ in 1:3
            prepare_decay_rates!(params, timestep_s, bomb_state=bomb_state)
            @test params[1].decayrate == 1.0
        end

        # After one more step, we're at t=10800s checking if >= 13600s, still NO
        prepare_decay_rates!(params, timestep_s, bomb_state=bomb_state)
        @test params[1].decayrate == 1.0  # Still at t=14400s, but started at 10800s

        # After one MORE step: now at t=14400s >= 13600s, decay starts!
        prepare_decay_rates!(params, timestep_s, bomb_state=bomb_state)
        @test params[1].decayrate < 1.0
    end

    @testset "Edge Cases" begin
        # Very short half-life
        param_short = DecayParams(kdecay=ExponentialDecay, halftime_hours=0.01)
        prepare_decay_rates!([param_short], 3600.0)
        @test param_short.decayrate < 0.1  # Rapid decay

        # Very long half-life
        param_long = DecayParams(kdecay=ExponentialDecay, halftime_hours=1e6)
        prepare_decay_rates!([param_long], 3600.0)
        @test param_long.decayrate ≈ 1.0 atol=1e-6  # Negligible decay

        # Zero activity
        param = DecayParams(kdecay=ExponentialDecay, halftime_hours=10.0)
        prepare_decay_rates!([param], 3600.0)
        @test apply_decay(0.0, param) == 0.0

        # Negative activity (should still work mathematically)
        @test apply_decay(-1.0, param) < 0.0
    end
end

println("✓ All decay tests passed!")
