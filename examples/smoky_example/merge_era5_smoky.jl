#!/usr/bin/env julia
"""
Merge ERA5 Model-Level and Surface Files for SNAP — Shot Smoky

This script combines:
1. Model-level files (t, u, v, omega) — era5_YYYYMMDD_HH-HH_ml.nc
2. Surface files (sp, u10, v10, tp) — era5_YYYYMMDD_HH-HH_sfc.nc
3. ERA5 L137 hybrid level coefficients (ap, b)

Into final SNAP-ready files: era5_smoky_YYYYMMDD_HH-HH_snap.nc
"""

using NCDatasets
using Printf

# ERA5 L137 hybrid level coefficients
# Source: https://confluence.ecmwf.int/display/UDOC/L137+model+level+definitions
# Format: (a [Pa], b [dimensionless]) at half levels (interfaces)
# Index 1 = model top, Index 138 = surface
const ERA5_L137_COEFFICIENTS = [
    (0.000000, 0.000000),      # Level 1 top
    (2.000365, 0.000000),      # Level 1 bottom / Level 2 top
    (3.102241, 0.000000),
    (4.666084, 0.000000),
    (6.827977, 0.000000),
    (9.746966, 0.000000),
    (13.605424, 0.000000),
    (18.608931, 0.000000),
    (24.985718, 0.000000),
    (32.985710, 0.000000),
    (42.879242, 0.000000),
    (54.955463, 0.000000),
    (69.520576, 0.000000),
    (86.895882, 0.000000),
    (107.415741, 0.000000),
    (131.425507, 0.000000),
    (159.279404, 0.000000),
    (191.338562, 0.000000),
    (227.968948, 0.000000),
    (269.539581, 0.000000),
    (316.420746, 0.000000),
    (368.982361, 0.000000),
    (427.592499, 0.000000),
    (492.616028, 0.000000),
    (564.413452, 0.000000),
    (643.339905, 0.000000),
    (729.744141, 0.000000),
    (823.967834, 0.000000),
    (926.344910, 0.000000),
    (1037.201172, 0.000000),
    (1156.853638, 0.000000),
    (1285.610352, 0.000000),
    (1423.770142, 0.000000),
    (1571.622925, 0.000000),
    (1729.448975, 0.000000),
    (1897.519287, 0.000000),
    (2076.095947, 0.000000),
    (2265.431641, 0.000000),
    (2465.770508, 0.000000),
    (2677.348145, 0.000000),
    (2900.391357, 0.000000),
    (3135.119385, 0.000000),
    (3381.743652, 0.000000),
    (3640.468262, 0.000000),
    (3911.490479, 0.000000),
    (4194.930664, 0.000000),
    (4490.817383, 0.000000),
    (4799.149414, 0.000000),
    (5119.895020, 0.000000),
    (5452.990723, 0.000000),
    (5798.344727, 0.000000),
    (6156.074219, 0.000000),
    (6526.946777, 0.000000),
    (6911.870605, 0.000000),
    (7311.869141, 0.000000),
    (7727.412109, 0.000700),
    (8159.354004, 0.002400),
    (8608.525391, 0.005300),
    (9076.400391, 0.011200),
    (9562.682617, 0.020000),
    (10065.978516, 0.032400),
    (10584.631836, 0.049300),
    (11116.662109, 0.070800),
    (11660.067383, 0.098000),
    (12211.547852, 0.131800),
    (12766.873047, 0.172900),
    (13324.668945, 0.221500),
    (13881.331055, 0.278200),
    (14432.139648, 0.343500),
    (14975.615234, 0.418000),
    (15508.256836, 0.502300),
    (16026.115234, 0.596300),
    (16527.322266, 0.700000),
    (17008.789063, 0.813400),
    (17467.613281, 0.936500),
    (17901.621094, 1.069300),
    (18308.433594, 1.211100),
    (18685.718750, 1.361200),
    (19031.289063, 1.519300),
    (19343.511719, 1.684700),
    (19620.042969, 1.856900),
    (19859.390625, 2.035400),
    (20059.931641, 2.219600),
    (20219.664063, 2.409100),
    (20337.863281, 2.603400),
    (20412.308594, 2.802000),
    (20442.078125, 3.004500),
    (20425.718750, 3.210200),
    (20361.816406, 3.418600),
    (20249.511719, 3.629200),
    (20087.085938, 3.841400),
    (19874.025391, 4.054700),
    (19608.572266, 4.268500),
    (19290.226563, 4.482100),
    (18917.460938, 4.695100),
    (18489.707031, 4.906900),
    (18006.925781, 5.117100),
    (17471.839844, 5.325000),
    (16888.687500, 5.530200),
    (16262.046875, 5.732100),
    (15596.695313, 5.930200),
    (14898.453125, 6.124100),
    (14173.324219, 6.313500),
    (13427.769531, 6.498100),
    (12668.257813, 6.677600),
    (11901.339844, 6.851800),
    (11133.304688, 7.020500),
    (10370.175781, 7.183500),
    (9617.515625, 7.340500),
    (8880.453125, 7.491300),
    (8163.375000, 7.635800),
    (7470.343750, 7.774000),
    (6804.421875, 7.905700),
    (6168.531250, 8.030800),
    (5564.382813, 8.149200),
    (4993.796875, 8.261000),
    (4457.375000, 8.366100),
    (3955.960938, 8.464500),
    (3489.234375, 8.556100),
    (3057.265625, 8.642000),
    (2659.140625, 8.721100),
    (2294.242188, 8.794200),
    (1961.640625, 8.861400),
    (1659.476563, 8.922600),
    (1387.546875, 8.978100),
    (1143.250000, 9.027800),
    (926.507813, 9.072100),
    (734.992188, 9.111200),
    (568.062500, 9.145400),
    (424.414063, 9.174800),
    (302.476563, 9.199500),
    (202.484375, 9.220100),
    (122.101563, 9.236600),
    (62.781250, 9.249400),
    (22.835938, 9.258600),
    (3.757813, 9.264400),
    (0.000000, 9.267100),
    (0.000000, 9.267100),  # Surface (repeated for consistency)
]

"""
    get_midpoint_coefficients(ap_half, b_half)

Compute midpoint (layer center) coefficients from half-level (interface) coefficients.
"""
function get_midpoint_coefficients(ap_half, b_half)
    n_half = length(ap_half)
    n_mid = n_half - 1

    ap_mid = zeros(n_mid)
    b_mid = zeros(n_mid)

    for i in 1:n_mid
        ap_mid[i] = (ap_half[i] + ap_half[i+1]) / 2.0
        b_mid[i] = (b_half[i] + b_half[i+1]) / 2.0
    end

    return ap_mid, b_mid
end

"""
    merge_era5_files(ml_file, sfc_file, output_file)

Merge model-level and surface ERA5 files into SNAP-ready format.
"""
function merge_era5_files(ml_file::String, sfc_file::String, output_file::String)
    println("Merging:")
    println("  Model-level: $ml_file")
    println("  Surface: $sfc_file")
    println("  -> Output: $output_file")

    # Extract hybrid coefficients
    ap_half = [c[1] for c in ERA5_L137_COEFFICIENTS]
    b_half = [c[2] for c in ERA5_L137_COEFFICIENTS]
    ap_mid, b_mid = get_midpoint_coefficients(ap_half, b_half)

    # Open input files
    ds_ml = NCDataset(ml_file, "r")
    ds_sfc = NCDataset(sfc_file, "r")

    # Read raw data arrays before creating output
    lon_data = ds_ml["longitude"][:]
    lat_data = ds_ml["latitude"][:]
    time_data = ds_ml["valid_time"][:]
    nk = length(ds_ml["model_level"])

    # Create output file with ERA5Format-compatible names
    NCDataset(output_file, "c") do ds_out
        # Dimensions: use "hybrid" (not "model_level") and "time" (not "valid_time")
        defDim(ds_out, "longitude", length(lon_data))
        defDim(ds_out, "latitude", length(lat_data))
        defDim(ds_out, "hybrid", nk)
        defDim(ds_out, "time", length(time_data))

        # Coordinate variables
        defVar(ds_out, "longitude", lon_data, ("longitude",))
        defVar(ds_out, "latitude", lat_data, ("latitude",))
        defVar(ds_out, "hybrid", Float32.(1:nk), ("hybrid",))
        defVar(ds_out, "time", time_data, ("time",))

        # Hybrid coefficients on "hybrid" dimension
        defVar(ds_out, "ap", Float32.(ap_mid), ("hybrid",), attrib=Dict(
            "long_name" => "hybrid A coefficient at layer midpoints",
            "units" => "Pa"
        ))
        defVar(ds_out, "b", Float32.(b_mid), ("hybrid",), attrib=Dict(
            "long_name" => "hybrid B coefficient at layer midpoints",
            "units" => "1"
        ))

        # Model-level 3D variables: rename to ERA5Format convention
        # u → x_wind_ml, v → y_wind_ml, t → air_temperature_ml
        ml_var_map = Dict("u" => "x_wind_ml", "v" => "y_wind_ml", "t" => "air_temperature_ml")
        for (raw_name, snap_name) in ml_var_map
            if haskey(ds_ml, raw_name)
                v = defVar(ds_out, snap_name, Float32,
                           ("longitude", "latitude", "hybrid", "time"))
                v[:,:,:,:] = nomissing(ds_ml[raw_name][:,:,:,:], 0f0)
            end
        end

        # Surface variables: rename to ERA5Format convention
        # sp → surface_air_pressure
        if haskey(ds_sfc, "sp")
            v = defVar(ds_out, "surface_air_pressure", Float32,
                       ("longitude", "latitude", "time"))
            v[:,:,:] = nomissing(ds_sfc["sp"][:,:,:], 0f0)
        end
        if haskey(ds_sfc, "tp")
            v = defVar(ds_out, "precipitation_amount", Float32,
                       ("longitude", "latitude", "time"))
            v[:,:,:] = nomissing(ds_sfc["tp"][:,:,:], 0f0)
        end

        # Add metadata
        ds_out.attrib["title"] = "ERA5 data for SNAP (Plumbbob Smoky, 44 kT)"
        ds_out.attrib["source"] = "ERA5 Back Extension (1950-1978)"
        ds_out.attrib["vertical_coordinate"] = "hybrid sigma-pressure (ECMWF L137)"
        ds_out.attrib["history"] = "Merged from model-level and surface files, variables renamed to ERA5Format convention"
    end

    close(ds_ml)
    close(ds_sfc)

    println("  Merged successfully\n")
end

"""
    merge_all_smoky_files(data_dir)

Merge all ERA5 file pairs in the Smoky ERA5 data directory.
"""
function merge_all_smoky_files(data_dir::String)
    # Find all model-level files
    ml_files = filter(f -> occursin(r"era5_\d{8}_\d{2}-\d{2}_ml\.nc$", f), readdir(data_dir))

    if isempty(ml_files)
        @error "No model-level files found in $data_dir"
        return
    end

    println("="^60)
    println("Merging ERA5 Files for Smoky Simulation")
    println("="^60)
    println("Found $(length(ml_files)) model-level files\n")

    merged_count = 0
    for ml_file in sort(ml_files)
        # Construct corresponding surface file name
        sfc_file = replace(ml_file, "_ml.nc" => "_sfc.nc")
        output_file = replace(ml_file, r"era5_(\d{8}_\d{2}-\d{2})_ml\.nc" => s"era5_smoky_\1_snap.nc")

        ml_path = joinpath(data_dir, ml_file)
        sfc_path = joinpath(data_dir, sfc_file)
        output_path = joinpath(data_dir, output_file)

        # Check if surface file exists
        if !isfile(sfc_path)
            @warn "Missing surface file: $sfc_file (skipping)"
            continue
        end

        # Overwrite if output already exists (variable naming may have changed)
        if isfile(output_path)
            println("  $output_file exists, regenerating...")
        end

        # Merge
        try
            merge_era5_files(ml_path, sfc_path, output_path)
            merged_count += 1
        catch e
            @error "Failed to merge $ml_file" exception=(e, catch_backtrace())
        end
    end

    println("="^60)
    println("Merge complete: $merged_count/$(length(ml_files)) files")
    println("="^60)
end

# Main execution
if abspath(PROGRAM_FILE) == @__FILE__
    data_dir = joinpath(@__DIR__, "ERA5_data")
    merge_all_smoky_files(data_dir)
end
