#!/usr/bin/env julia
"""
Repair b coefficients in existing Smoky snap.nc files.

Copies snap.nc files from the Zenodo artifact to local ERA5_data/,
then overwrites the `ap` and `b` variables with correct ECMWF L137 values.
The meteorological data (wind, temperature, pressure) is unchanged.
"""

using NCDatasets

# Load the fixed coefficient table from the merge script
include(joinpath(@__DIR__, "merge_era5_smoky.jl"))

# Compute correct midpoint coefficients
b_half = Float32[c[2] for c in ERA5_L137_COEFFICIENTS]
ap_half = Float32[c[1] for c in ERA5_L137_COEFFICIENTS]
n_mid = length(b_half) - 1
b_mid = Float32[(b_half[i] + b_half[i+1]) / 2.0f0 for i in 1:n_mid]
ap_mid = Float32[(ap_half[i] + ap_half[i+1]) / 2.0f0 for i in 1:n_mid]

println("Correct b coefficients (midpoint):")
println("  b[1] = $(b_mid[1])  (model top)")
println("  b[end] = $(b_mid[end])  (near surface, should be ~0.999)")
println("  b range: [$(minimum(b_mid)), $(maximum(b_mid))]")
println()

# Source: Zenodo artifact
artifact_dir = "/home/marc/.julia/artifacts/8966357b72a8f3ccfcdbddb7c083181719eec1a8/smoky_era5_data"
if !isdir(artifact_dir)
    error("Artifact not cached at $artifact_dir — run smoky_era5_files() first")
end

# Destination: local ERA5_data/
output_dir = joinpath(@__DIR__, "ERA5_data")
mkpath(output_dir)

snap_files = sort(filter(f -> endswith(f, "_snap.nc"), readdir(artifact_dir)))
println("Found $(length(snap_files)) snap files to repair\n")

for fname in snap_files
    src = joinpath(artifact_dir, fname)
    dst = joinpath(output_dir, fname)
    print("Repairing: $fname ... ")

    # Copy file verbatim and make writable
    cp(src, dst, force=true)
    chmod(dst, 0o644)

    # Overwrite ap and b in the copy
    NCDataset(dst, "a") do ds  # "a" = append mode
        ds["b"][:] = b_mid
        ds["ap"][:] = ap_mid
    end

    println("done")
end

# Verify
println("\nVerification:")
test_file = joinpath(output_dir, snap_files[1])
NCDataset(test_file) do ds
    b = ds["b"][:]
    ap = ds["ap"][:]
    println("  b[1] = $(b[1])  (model top)")
    println("  b[end] = $(b[end])  (near surface)")
    println("  b range: [$(minimum(b)), $(maximum(b))]")
    println("  ap[1] = $(ap[1])  (model top)")
    println("  ap[end] = $(ap[end])  (near surface)")
    println("  Surface b ≈ 1.0: $(abs(b[end] - 0.99881) < 0.01 ? "PASS" : "FAIL")")
end

println("\nRepair complete! $(length(snap_files)) files written to $output_dir")
