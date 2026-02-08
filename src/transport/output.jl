# SNAP: Severe Nuclear Accident Programme
# Output and Dose Calculation Functions
#
# Phase 7: NetCDF output, dose calculations, and export utilities

using NCDatasets
using Printf
using Dates: format, now

"""
    compute_dose_rate(deposition::Matrix{T}, decay::BombDecay; hours_after=12) where T

Compute dose rate (mSv/hr) from surface deposition using bomb decay model.

# Arguments
- `deposition`: Surface deposition density (Bq/m²)
- `decay`: BombDecay parameters
- `hours_after`: Hours after detonation for dose rate calculation (default: 12)

# Returns
- Dose rate field (mSv/hr) at specified time

# Physics
For nuclear weapon fallout, dose rate from surface deposition follows:
    D(t) = D₀ × t^(-1.2)

Where:
- D(t) is dose rate at time t (hours after detonation)
- D₀ is initial dose rate at H+1
- The t^(-1.2) is the Glasstone/Dolan "7-10 rule" decay

Conversion from deposition to dose rate:
    D₀ (mSv/hr) = k × σ (Bq/m²)

where k ≈ 1.9×10⁻⁶ mSv·hr⁻¹·Bq⁻¹·m² for mixed fission products
at H+1 (this factor varies with isotope composition and geometry)

# References
- Glasstone & Dolan (1977), "The Effects of Nuclear Weapons", Ch. 9
- IAEA Safety Report No. 46 (2006), "Radiation Protection in Nuclear Accidents"
"""
function compute_dose_rate(deposition::Matrix{T}, decay::BombDecayState; hours_after=12) where T
    # Dose conversion factor: Bq/m² → mSv/hr at H+1
    # This is for mixed fission products, 1m above infinite smooth surface
    K_DOSE = 1.9e-6  # mSv·hr⁻¹ per Bq·m⁻²

    # Time since detonation (hours)
    t_hours = Float64(hours_after)

    # Initial dose rate at H+1 from deposition
    dose_h1 = K_DOSE .* deposition

    # Apply bomb decay: D(t) = D(H+1) × (t/1)^(-1.2)
    # For t < 1 hour, we extrapolate (dose increases, but capped for safety)
    if t_hours < 1.0
        decay_factor = (1.0 / t_hours)^1.2  # Higher dose earlier
        decay_factor = min(decay_factor, 10.0)  # Cap at 10× to avoid numerical issues
    else
        decay_factor = t_hours^(-1.2)
    end

    dose_rate = dose_h1 .* decay_factor

    return dose_rate
end

"""
    grid_cell_area(domain::SimulationDomain, i::Int, j::Int)

Compute the area of grid cell (i, j) in m².

# Arguments
- `domain`: Simulation domain with lat/lon bounds
- `i`: Longitude index
- `j`: Latitude index

# Returns
- Cell area in m²

# Notes
Uses spherical Earth approximation:
    Area = R² × Δλ × (sin(φ₂) - sin(φ₁))

where R is Earth radius, Δλ is longitude width (radians),
φ₁, φ₂ are latitude bounds (radians)
"""
function grid_cell_area(domain::SimulationDomain, i::Int, j::Int)
    R_EARTH = 6371000.0  # Earth radius (m)

    # Grid spacing
    dlon = (domain.lon_max - domain.lon_min) / domain.nx
    dlat = (domain.lat_max - domain.lat_min) / domain.ny

    # Cell bounds
    lon1 = domain.lon_min + (i - 1) * dlon
    lon2 = lon1 + dlon
    lat1 = domain.lat_min + (j - 1) * dlat
    lat2 = lat1 + dlat

    # Convert to radians
    dlon_rad = deg2rad(dlon)
    lat1_rad = deg2rad(lat1)
    lat2_rad = deg2rad(lat2)

    # Spherical area formula
    area = R_EARTH^2 * dlon_rad * abs(sin(lat2_rad) - sin(lat1_rad))

    return area
end

"""
    grid_cell_area(domain::SimulationDomain)

Compute total area of the simulation domain in m².

# Arguments
- `domain`: Simulation domain

# Returns
- Total domain area in m²
"""
function grid_cell_area(domain::SimulationDomain)
    R_EARTH = 6371000.0  # Earth radius (m)

    # Total longitude/latitude spans
    dlon_total = deg2rad(domain.lon_max - domain.lon_min)
    lat1_rad = deg2rad(domain.lat_min)
    lat2_rad = deg2rad(domain.lat_max)

    # Total spherical area
    area = R_EARTH^2 * dlon_total * abs(sin(lat2_rad) - sin(lat1_rad))

    return area / (domain.nx * domain.ny)  # Average cell area
end

"""
    export_dose_fields(filename::String, domain::SimulationDomain,
                       dose_rate::Matrix{T}, deposition::Matrix{T},
                       reference_time::DateTime) where T

Export dose rate and deposition fields to NetCDF file for QGIS visualization.

# Arguments
- `filename`: Output NetCDF file path
- `domain`: Simulation domain (defines grid)
- `dose_rate`: Dose rate field (mSv/hr) at reference_time
- `deposition`: Total surface deposition (Bq/m²)
- `reference_time`: Time for dose rate calculation (e.g., H+12)

# Output Format
NetCDF file with:
- Dimensions: longitude, latitude
- Variables: dose_rate_mSv_hr, deposition_Bq_m2, dose_rate_mR_hr (for compatibility)
- Attributes: CF-compliant metadata, CRS information

# Notes
- Includes conversion to mR/hr for compatibility with legacy tools
- 1 mSv/hr ≈ 100 mR/hr (more precisely 100.7 mR/hr)
"""
function export_dose_fields(filename::String, domain::SimulationDomain,
                             dose_rate::Matrix{T}, deposition::Matrix{T},
                             reference_time::DateTime) where T

    @info "Exporting dose fields to $filename"

    # Create coordinate arrays
    lons = range(domain.lon_min, domain.lon_max, length=domain.nx)
    lats = range(domain.lat_min, domain.lat_max, length=domain.ny)

    # Convert dose rate to mR/hr for compatibility
    dose_rate_mR = dose_rate .* 100.0  # 1 mSv/hr ≈ 100 mR/hr

    # Create NetCDF file
    NCDataset(filename, "c") do ds
        # Define dimensions
        defDim(ds, "longitude", domain.nx)
        defDim(ds, "latitude", domain.ny)

        # Define coordinate variables
        lon_var = defVar(ds, "longitude", Float64, ("longitude",))
        lon_var[:] = collect(lons)
        lon_var.attrib["units"] = "degrees_east"
        lon_var.attrib["long_name"] = "Longitude"
        lon_var.attrib["standard_name"] = "longitude"

        lat_var = defVar(ds, "latitude", Float64, ("latitude",))
        lat_var[:] = collect(lats)
        lat_var.attrib["units"] = "degrees_north"
        lat_var.attrib["long_name"] = "Latitude"
        lat_var.attrib["standard_name"] = "latitude"

        # Define data variables
        # 1. Dose rate (mSv/hr)
        dose_var = defVar(ds, "dose_rate_mSv_hr", Float32, ("longitude", "latitude"))
        dose_var[:, :] = Float32.(dose_rate)
        dose_var.attrib["units"] = "mSv/hr"
        dose_var.attrib["long_name"] = "Radiation dose rate"
        dose_var.attrib["standard_name"] = "dose_rate"
        dose_var.attrib["reference_time"] = format(reference_time, "yyyy-mm-dd HH:MM:SS UTC")
        dose_var.attrib["_FillValue"] = Float32(NaN)
        dose_var.attrib["valid_min"] = Float32(0.0)

        # 2. Dose rate (mR/hr) - for legacy compatibility
        dose_mR_var = defVar(ds, "dose_rate_mR_hr", Float32, ("longitude", "latitude"))
        dose_mR_var[:, :] = Float32.(dose_rate_mR)
        dose_mR_var.attrib["units"] = "mR/hr"
        dose_mR_var.attrib["long_name"] = "Radiation dose rate (roentgen per hour)"
        dose_mR_var.attrib["reference_time"] = format(reference_time, "yyyy-mm-dd HH:MM:SS UTC")
        dose_mR_var.attrib["_FillValue"] = Float32(NaN)
        dose_mR_var.attrib["valid_min"] = Float32(0.0)
        dose_mR_var.attrib["note"] = "1 mSv/hr ≈ 100 mR/hr (conversion factor 100.7)"

        # 3. Surface deposition (Bq/m²)
        dep_var = defVar(ds, "deposition_Bq_m2", Float32, ("longitude", "latitude"))
        dep_var[:, :] = Float32.(deposition)
        dep_var.attrib["units"] = "Bq/m2"
        dep_var.attrib["long_name"] = "Total surface deposition"
        dep_var.attrib["standard_name"] = "surface_deposition"
        dep_var.attrib["_FillValue"] = Float32(NaN)
        dep_var.attrib["valid_min"] = Float32(0.0)

        # Global attributes
        ds.attrib["title"] = "Nuclear fallout simulation results"
        ds.attrib["institution"] = "NuclearDetonation.jl"
        ds.attrib["source"] = "NuclearDetonation.jl atmospheric transport model"
        ds.attrib["history"] = format(now(), "yyyy-mm-dd HH:MM:SS") * " - Created by NuclearDetonation.jl"
        ds.attrib["Conventions"] = "CF-1.8"
        ds.attrib["reference_time"] = format(reference_time, "yyyy-mm-dd HH:MM:SS UTC")

        # Coordinate Reference System (WGS84)
        crs_var = defVar(ds, "crs", Int32, ())
        crs_var.attrib["grid_mapping_name"] = "latitude_longitude"
        crs_var.attrib["long_name"] = "WGS84"
        crs_var.attrib["semi_major_axis"] = 6378137.0
        crs_var.attrib["inverse_flattening"] = 298.257223563
        crs_var.attrib["crs_wkt"] = "GEOGCS[\"WGS 84\",DATUM[\"WGS_1984\",SPHEROID[\"WGS 84\",6378137,298.257223563]],PRIMEM[\"Greenwich\",0],UNIT[\"degree\",0.0174532925199433]]"

        # Add CRS reference to data variables
        dose_var.attrib["grid_mapping"] = "crs"
        dose_mR_var.attrib["grid_mapping"] = "crs"
        dep_var.attrib["grid_mapping"] = "crs"
    end

    @info "  ✓ Exported dose_rate_mSv_hr, dose_rate_mR_hr, deposition_Bq_m2"
    @info "  ✓ File ready for QGIS: $filename"
end

"""
    export_concentration_field(filename::String, domain::SimulationDomain,
                                concentration::Array{T,3}, valid_time::DateTime,
                                varname::String="concentration") where T

Export 3D concentration field to NetCDF.

# Arguments
- `filename`: Output NetCDF file path
- `domain`: Simulation domain
- `concentration`: 3D concentration field (Bq/m³)
- `valid_time`: Time of the concentration field
- `varname`: Variable name in NetCDF (default: "concentration")
"""
function export_concentration_field(filename::String, domain::SimulationDomain,
                                    concentration::Array{T,3}, valid_time::DateTime,
                                    varname::String="concentration") where T

    @info "Exporting 3D concentration field to $filename"

    # Create coordinate arrays
    lons = range(domain.lon_min, domain.lon_max, length=domain.nx)
    lats = range(domain.lat_min, domain.lat_max, length=domain.ny)
    heights = range(domain.z_min, domain.z_max, length=domain.nz)

    NCDataset(filename, "c") do ds
        # Define dimensions
        defDim(ds, "longitude", domain.nx)
        defDim(ds, "latitude", domain.ny)
        defDim(ds, "height", domain.nz)

        # Coordinate variables
        defVar(ds, "longitude", collect(lons), ("longitude",), attrib=Dict(
            "units" => "degrees_east",
            "long_name" => "Longitude",
            "standard_name" => "longitude"
        ))

        defVar(ds, "latitude", collect(lats), ("latitude",), attrib=Dict(
            "units" => "degrees_north",
            "long_name" => "Latitude",
            "standard_name" => "latitude"
        ))

        defVar(ds, "height", collect(heights), ("height",), attrib=Dict(
            "units" => "m",
            "long_name" => "Height above ground level",
            "standard_name" => "height",
            "positive" => "up"
        ))

        # Concentration variable
        conc_var = defVar(ds, varname, Float32, ("longitude", "latitude", "height"))
        conc_var[:, :, :] = Float32.(concentration)
        conc_var.attrib["units"] = "Bq/m3"
        conc_var.attrib["long_name"] = "Atmospheric concentration"
        conc_var.attrib["standard_name"] = "atmosphere_mole_content_of_radionuclide"
        conc_var.attrib["valid_time"] = format(valid_time, "yyyy-mm-dd HH:MM:SS UTC")
        conc_var.attrib["_FillValue"] = Float32(NaN)

        # Global attributes
        ds.attrib["title"] = "3D atmospheric concentration field"
        ds.attrib["institution"] = "NuclearDetonation.jl"
        ds.attrib["source"] = "NuclearDetonation.jl atmospheric transport model"
        ds.attrib["Conventions"] = "CF-1.8"
        ds.attrib["valid_time"] = format(valid_time, "yyyy-mm-dd HH:MM:SS UTC")
    end

    @info "  ✓ Exported 3D concentration field"
end

# Export public API
export compute_dose_rate, grid_cell_area
export export_dose_fields, export_concentration_field
