#!/usr/bin/env julia
# Unit tests for gpu_transport/calibration_shared.jl
# Run from repo root:  julia --project gpu_transport/test_calibration_shared.jl
using Test
include(joinpath(@__DIR__, "calibration_shared.jl"))
using .CalibrationShared

@testset "CalibrationShared" begin

    @testset "log_mask" begin
        m20 = log_mask(20); m23 = log_mask(23)
        @test count(m20) == 12 && count(m23) == 12        # slots 8..19
        @test all(m20[8:19]) && all(m23[8:19])
        @test !m20[7] && !m20[20] && !m23[7] && !m23[20] && !m23[23]
    end

    @testset "decode∘encode identity (1e-15)" begin
        for n in (20, 23)
            mask = log_mask(n)
            x = collect(range(0.3, 4.0; length = n))          # all positive
            x[8:19] .= [0.01, 0.05, 0.2, 0.5, 1.0, 1.7, 2.3, 3.0, 5.0, 8.0, 12.0, 30.0]
            rt = decode_params(encode_params(x, mask), mask)
            @test maximum(abs.(rt .- x)) < 1e-13
        end
    end

    @testset "make_logspace bounds" begin
        n = 23
        lb = fill(0.1, n); ub = fill(10.0, n)
        L = make_logspace(n, lb, ub)
        @test L.LB_S[8]  ≈ log10(0.1)
        @test L.UB_S[19] ≈ log10(10.0)
        @test L.LB_S[1]  ≈ 0.1          # linear slot unchanged
        @test L.decode(L.encode(ub)) ≈ ub
    end

    @testset "combined_loss weights (one formula for all tests)" begin
        r = combined_loss(fms=0.4, shape=0.5, bearing=0.8, extent=0.6, toa=0.9)
        @test r.loss ≈ 1.0 - 0.665 atol=1e-12      # 0.25,0.15,0.20,0.10,0.30
        @test r.combined_old ≈ 0.60 atol=1e-12     # 0.35,0.20,0.15,0.30
        @test r.bearing == 0.8                     # bearing always carried in the score
        # no per-test override exists: combined_loss takes no `doppler`/test kwarg
        @test !any(==(:doppler), Base.kwarg_decl(first(methods(combined_loss))))
    end

    @testset "hard bearing gate" begin
        g = combined_loss(fms=1.0, shape=1.0, bearing=0.3, extent=1.0, toa=1.0)
        @test g.loss == 2.0                          # gated regardless of others
        @test g.bearing == 0.3
        # boundary: bearing == 0.5 is NOT gated (gate is strict <)
        b = combined_loss(fms=1.0, shape=1.0, bearing=0.5, extent=1.0, toa=1.0)
        @test b.loss < 2.0
    end

    @testset "geo_mean" begin
        @test geo_mean(Float64[]) == 0.0
        @test geo_mean([0.5, 0.5]) ≈ 0.5
        @test geo_mean([0.0, 1.0]) ≈ sqrt(0.005)   # zero floored to 0.005: exp(mean(log([0.005,1])))
    end

    @testset "centroid_bearing cardinal directions" begin
        lon = collect(-2.0:0.5:2.0); lat = collect(-2.0:0.5:2.0)
        nx, ny = length(lon), length(lat)
        j0 = findfirst(==(0.0), lat); i0 = findfirst(==(0.0), lon)
        # due east: cells at lat=0, lon>0
        east = falses(nx, ny); for i in (i0+1):nx; east[i, j0] = true; end
        @test centroid_bearing(east, lat, lon, 0.0, 0.0; min_cells=1) ≈ 90.0 atol=1e-6
        # due north: cells at lon=0, lat>0
        north = falses(nx, ny); for j in (j0+1):ny; north[i0, j] = true; end
        @test centroid_bearing(north, lat, lon, 0.0, 0.0; min_cells=1) ≈ 0.0 atol=1e-6
    end

    @testset "bearing_score aligned vs orthogonal" begin
        lon = collect(-2.0:0.5:2.0); lat = collect(-2.0:0.5:2.0)
        nx, ny = length(lon), length(lat)
        j0 = findfirst(==(0.0), lat); i0 = findfirst(==(0.0), lon)
        dose = zeros(nx, ny); for i in (i0+1):nx; dose[i, j0] = 200.0; end   # plume due east
        obs_masks = [(100.0, falses(nx, ny))]
        # obs also east -> aligned -> score ~1
        s_aligned = bearing_score(dose, obs_masks, Dict(100.0 => 90.0),
                                  lat, lon, 0.0, 0.0; min_cells=1)
        @test s_aligned ≈ 1.0 atol=1e-6
        # obs north, model east -> 90° off -> score ~0 -> would gate
        s_ortho = bearing_score(dose, obs_masks, Dict(100.0 => 0.0),
                                lat, lon, 0.0, 0.0; min_cells=1)
        @test s_ortho < 1e-6
    end
end
