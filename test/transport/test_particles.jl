# Tests for particles.jl: Particle Data Structures

using Test

@testset "Particle System" begin

    @testset "Particle Construction" begin
        p = Particle()
        @test p.x == 0.0
        @test p.y == 0.0
        @test p.z == 0.0
        @test p.tbl == 0.0f0  # tbl is Float32
        @test is_active(p) == false  # Use is_active function instead of active field
    end

    @testset "Particle Field Access" begin
        p = Particle(10.0, 20.0, 0.5, 0.8f0, 100.0f0, 1000.0f0, 0.01f0, Int16(1))

        @test p.x == 10.0
        @test p.y == 20.0
        @test p.z == 0.5
        @test p.tbl == 0.8f0
        @test get_rad(p) == 100.0f0  # Use accessor function for rad field

        # Test radioactive content operations
        @test is_active(p) == true
        removed = scale_rad!(p, 0.5f0)
        @test get_rad(p) == 50.0f0
        @test removed == 50.0f0  # Should have removed 50.0

        # Test adding content
        add_rad!(p, 25.0f0)
        @test get_rad(p) == 75.0f0
        
        # Test inactivate
        inactivate!(p)
        @test is_active(p) == false
    end

    @testset "ExtraParticle Construction" begin
        ep = ExtraParticle()
        @test ep.u == 0.0f0  # u velocity
        @test ep.v == 0.0f0  # v velocity
        @test ep.rmx == 0.0  # x map ratio
        @test ep.rmy == 0.0  # y map ratio
        @test ep.prc == 0.0f0  # precipitation
    end

    @testset "ExtraParticle Field Access" begin
        ep = ExtraParticle()
        
        # Test mutable fields directly since they're public
        ep.u = 5.0f0
        ep.v = 3.0f0
        ep.rmx = 1.2
        ep.rmy = 1.1
        ep.prc = 2.5f0

        @test ep.u == 5.0f0
        @test ep.v == 3.0f0
        @test ep.rmx == 1.2
        @test ep.rmy == 1.1
        @test ep.prc == 2.5f0
    end

    println("✓ All Particle tests passed!")
end
