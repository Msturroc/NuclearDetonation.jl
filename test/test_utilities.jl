# Tests for src/utilities.jl: unit conversions, exception types, dict helpers

using Test

import NuclearDetonation.Utilities:
    ValueOutsideGraphError, UnknownUnitError,
    convert_units, dict_reverse

@testset "Utilities: unit conversion and helpers" begin

    @testset "convert_units identity" begin
        @test convert_units(42.0, "m", "m") == 42.0
        @test convert_units(7,    "kT", "kT") == 7
    end

    @testset "convert_units yield" begin
        @test convert_units(2000.0, "kT", "MT") ≈ 2.0
        @test convert_units(1.5,    "MT", "kT") ≈ 1500.0
    end

    @testset "convert_units distance" begin
        @test convert_units(1000.0, "m",       "km") ≈ 1.0
        @test convert_units(2.0,    "km",      "m") ≈ 2000.0
        @test convert_units(304.8,  "m",       "kilofeet") ≈ 1.0
        @test convert_units(1.0,    "kilofeet","m") ≈ 304.8
        @test convert_units(1.0,    "kilofeet","km") ≈ 0.3048
        @test convert_units(1.0,    "ft",      "m") ≈ 0.3048
        @test convert_units(1.0,    "m",       "ft") ≈ 1 / 0.3048
        @test convert_units(1.09361,"m",       "yards") ≈ 1.09361 * 1.09361
        @test convert_units(1.0,    "yards",   "m") ≈ 1 / 1.09361
        @test convert_units(1.0,    "kilofeet","mi") ≈ 1 / 5.28
        @test convert_units(1.0,    "mi",      "km") ≈ 1.60934
        @test convert_units(1.0,    "km",      "mi") ≈ 1 / 1.60934
        @test convert_units(1.0,    "km",      "kilofeet") ≈ 1 / 0.3048
        @test convert_units(1.0,    "yards",   "meters") ≈ 0.9144
        @test convert_units(1.0,    "meters",  "yards") ≈ 1 / 0.9144
        @test convert_units(1.0,    "yards",   "km") ≈ 0.0009144
        @test convert_units(1.0,    "km",      "yards") ≈ 1 / 0.0009144
    end

    @testset "convert_units pressure" begin
        @test convert_units(1.0, "psi",     "kg/cm^2") ≈ 0.070307
        @test convert_units(1.0, "kg/cm^2", "psi") ≈ 1 / 0.070307
        @test convert_units(1.0, "MPa",     "psi") ≈ 145.037738
        @test convert_units(1.0, "psi",     "MPa") ≈ 1 / 145.037738
        @test convert_units(1.0, "kg/cm^2", "MPa") > 0   # composite path
        @test convert_units(1.0, "MPa",     "kg/cm^2") > 0
        # Pa fallthrough paths
        @test convert_units(1e6, "Pa",  "MPa") ≈ 1.0
        @test convert_units(1.0, "MPa", "Pa") ≈ 1e6
    end

    @testset "convert_units speed" begin
        @test convert_units(1.0, "m/s",  "mph") ≈ 2.23694
        @test convert_units(1.0, "mph",  "m/s") ≈ 1 / 2.23694
        @test convert_units(1.0, "m/s",  "km/h") ≈ 3.6
        @test convert_units(1.0, "km/h", "m/s") ≈ 1 / 3.6
        @test convert_units(1.0, "mph",  "km/h") ≈ 1.60934
        @test convert_units(1.0, "km/h", "mph") ≈ 1 / 1.60934
    end

    @testset "convert_units shear and dose" begin
        @test convert_units(1.0, "m/s-km",   "mph/kilofoot") ≈ 0.13625756613945836
        @test convert_units(100.0, "Roentgen","Sv") ≈ 1.0
    end

    @testset "convert_units throws on unknown pair" begin
        @test_throws UnknownUnitError convert_units(1.0, "furlongs", "parsecs")
    end

    @testset "Exception printing" begin
        io = IOBuffer()
        Base.showerror(io, ValueOutsideGraphError(42))
        s = String(take!(io))
        @test occursin("ValueOutsideGraphError", s)
        @test occursin("42", s)

        io2 = IOBuffer()
        Base.showerror(io2, UnknownUnitError(("furlongs", "parsecs")))
        s2 = String(take!(io2))
        @test occursin("UnknownUnitError", s2)
        @test occursin("furlongs", s2)
        @test occursin("parsecs", s2)
    end

    @testset "dict_reverse reverses each value" begin
        d = Dict("a" => [1, 2, 3], "b" => [10, 20])
        r = dict_reverse(d)
        @test r["a"] == [3, 2, 1]
        @test r["b"] == [20, 10]
    end
end

println("✓ All Utilities tests passed!")
