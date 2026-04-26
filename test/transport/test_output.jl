# Tests for output.jl: dose-rate calculation, grid-cell area, NetCDF export

using Test
using Dates: DateTime as DatesDT
using NCDatasets

const _Tx = NuclearDetonation.Transport

@testset "output: dose, area and NetCDF export" begin

    @testset "compute_dose_rate applies K_DOSE × deposition × t^-1.2" begin
        deposition = zeros(Float64, 2, 2)
        deposition[1, 1] = 1e6  # 1e6 Bq/m² in one cell

        decay = BombDecayState{Float64}()  # state object only carries time, not used in formula
        dose = _Tx.compute_dose_rate(deposition, decay; hours_after=12)

        K_DOSE = 1.9e-6  # mSv·hr⁻¹ per Bq·m⁻²
        expected_hot = K_DOSE * 1e6 * 12.0^(-1.2)

        @test size(dose) == size(deposition)
        @test dose[1, 1] ≈ expected_hot rtol=1e-12
        @test dose[1, 2] ≈ 0.0
        @test dose[2, 1] ≈ 0.0
        @test dose[2, 2] ≈ 0.0

        # H+1 reference: decay factor = 1, so dose at H+1 equals K_DOSE × deposition.
        dose_h1 = _Tx.compute_dose_rate(deposition, decay; hours_after=1)
        @test dose_h1[1, 1] ≈ K_DOSE * 1e6 rtol=1e-12

        # Sub-hour clamp: decay factor capped at 10× to avoid singularity.
        dose_early = _Tx.compute_dose_rate(deposition, decay; hours_after=0.001)
        @test dose_early[1, 1] ≈ K_DOSE * 1e6 * 10.0 rtol=1e-12
    end

    @testset "grid_cell_area shrinks with latitude" begin
        # Helper to build a minimal SimulationDomain with given lat/lon bounds.
        # Uses the explicit constructor so we control nx/ny and the geographic box.
        nx, ny, nz = 1, 1, 2
        hlevel = [0.0, 1000.0]
        xm = ones(Float64, nx, ny)
        ym = ones(Float64, nx, ny)
        cell_area = ones(Float64, nx, ny)
        t_start = DateTime(2025, 1, 1, 0)
        t_end   = DateTime(2025, 1, 1, 1)
        dt_out  = Duration(0, 1, 0, 0)
        dt_met  = Duration(0, 1, 0, 0)

        # Equator: 1°×1° box
        dom_eq = SimulationDomain(
            nx, ny, nz, 1.0, 1.0, hlevel, xm, ym, cell_area,
            t_start, t_end, dt_out, dt_met;
            lon_min=0.0, lon_max=1.0, lat_min=0.0, lat_max=1.0,
        )
        a_eq = _Tx.grid_cell_area(dom_eq, 1, 1)

        R = 6_371_000.0
        dlon_rad = deg2rad(1.0)
        expected_eq = R^2 * dlon_rad * (sin(deg2rad(1.0)) - sin(0.0))
        @test a_eq ≈ expected_eq rtol=1e-12

        # 60°N: 1°×1° box at higher latitude → roughly half the equatorial area.
        dom_60 = SimulationDomain(
            nx, ny, nz, 1.0, 1.0, hlevel, xm, ym, cell_area,
            t_start, t_end, dt_out, dt_met;
            lon_min=0.0, lon_max=1.0, lat_min=60.0, lat_max=61.0,
        )
        a_60 = _Tx.grid_cell_area(dom_60, 1, 1)
        expected_60 = R^2 * dlon_rad * (sin(deg2rad(61.0)) - sin(deg2rad(60.0)))
        @test a_60 ≈ expected_60 rtol=1e-12

        # Sanity: cell at 60°N should be ~half of equatorial.
        @test 0.4 < a_60 / a_eq < 0.6

        # The sum-of-cells overload returns the average cell area for the whole domain.
        a_total_avg = _Tx.grid_cell_area(dom_eq)
        @test a_total_avg > 0.0 && isfinite(a_total_avg)
    end

    @testset "export_dose_fields writes a readable NetCDF" begin
        nx, ny, nz = 4, 3, 2
        hlevel = [0.0, 1000.0]
        xm = ones(Float64, nx, ny)
        ym = ones(Float64, nx, ny)
        cell_area = ones(Float64, nx, ny)
        t_start = DateTime(2025, 1, 1, 0)
        t_end   = DateTime(2025, 1, 1, 1)
        dt_out  = Duration(0, 1, 0, 0)
        dt_met  = Duration(0, 1, 0, 0)

        domain = SimulationDomain(
            nx, ny, nz, 1.0, 1.0, hlevel, xm, ym, cell_area,
            t_start, t_end, dt_out, dt_met;
            lon_min=-1.0, lon_max=1.0, lat_min=50.0, lat_max=52.0,
        )

        dose_rate  = fill(0.5, nx, ny)        # 0.5 mSv/hr everywhere
        deposition = fill(1.0e3, nx, ny)      # 1000 Bq/m²
        dose_rate[2, 2]  = 2.5
        deposition[2, 2] = 9.9e4

        path = tempname() * ".nc"
        ref_time = DatesDT(2025, 1, 1, 12)
        _Tx.export_dose_fields(path, domain, dose_rate, deposition, ref_time)
        @test isfile(path)

        try
            ds = NCDataset(path, "r")
            try
                # Variable presence
                @test haskey(ds, "longitude")
                @test haskey(ds, "latitude")
                @test haskey(ds, "dose_rate_mSv_hr")
                @test haskey(ds, "dose_rate_mR_hr")
                @test haskey(ds, "deposition_Bq_m2")

                # Dimensions match
                @test size(ds["dose_rate_mSv_hr"]) == (nx, ny)
                @test size(ds["deposition_Bq_m2"]) == (nx, ny)

                # Sample values round-trip
                dose_back = Array(ds["dose_rate_mSv_hr"])
                @test dose_back[2, 2] ≈ Float32(2.5)
                @test dose_back[1, 1] ≈ Float32(0.5)

                dep_back = Array(ds["deposition_Bq_m2"])
                @test dep_back[2, 2] ≈ Float32(9.9e4)

                # mR/hr should be 100 × mSv/hr
                dose_mR = Array(ds["dose_rate_mR_hr"])
                @test dose_mR[2, 2] ≈ Float32(2.5 * 100.0) rtol=1e-5

                # Attributes
                @test ds["dose_rate_mSv_hr"].attrib["units"] == "mSv/hr"
                @test ds["deposition_Bq_m2"].attrib["units"] == "Bq/m2"
                @test ds["longitude"].attrib["units"] == "degrees_east"
                @test ds["latitude"].attrib["units"] == "degrees_north"
                @test ds.attrib["Conventions"] == "CF-1.8"
            finally
                close(ds)
            end
        finally
            isfile(path) && rm(path; force=true)
        end
    end
end

println("✓ All output tests passed!")
