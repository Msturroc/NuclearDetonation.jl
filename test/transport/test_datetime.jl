# Tests for datetime.jl: DateTime and Duration System

using Test

@testset "DateTime System" begin

    @testset "DateTime Construction" begin
        dt = DateTime(2025, 10, 10, 12)
        @test dt.year == 2025
        @test dt.month == 10
        @test dt.day == 10
        @test dt.hour == 12
    end

    @testset "Duration Construction" begin
        dur = Duration(24)  # 24 hours
        @test dur.hours == 24
    end

    @testset "DateTime Arithmetic" begin
        dt1 = DateTime(2025, 1, 1, 0)
        dur = Duration(24)

        # Addition
        dt2 = dt1 + dur
        @test dt2.day == 2

        # Subtraction
        dt3 = dt2 - dur
        @test dt3.day == 1
    end

    @testset "DateTime Comparison" begin
        dt1 = DateTime(2025, 1, 1, 0)
        dt2 = DateTime(2025, 1, 2, 0)
        dt3 = DateTime(2025, 1, 1, 0)

        @test dt1 < dt2
        @test dt2 > dt1
        @test dt1 == dt3
        @test dt1 <= dt3
        @test dt1 >= dt3
    end

    println("✓ All DateTime tests passed!")
end
