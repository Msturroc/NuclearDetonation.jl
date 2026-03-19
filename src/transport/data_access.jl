# Data access helpers for ERA5 meteorological data via Julia Artifacts
# Provides convenience functions for downloading and locating met data files.

using Pkg.Artifacts

export nancy_era5_files, smoky_era5_files, etex_era5_files

const _ARTIFACTS_TOML = joinpath(@__DIR__, "..", "..", "Artifacts.toml")

"""
    nancy_era5_files()

Return a sorted vector of file paths to the Nancy ERA5 meteorological data files.

On first call, triggers a download from Zenodo (~96 MB). Subsequent calls
use the cached artifact.

# Returns
- `Vector{String}` — sorted paths to 24 ERA5 NetCDF files covering 24–27 March 1953

# Example
```julia
met_files = nancy_era5_files()
results = run_simulation!(state, met_files, ...)
```
"""
function nancy_era5_files()
    hash = artifact_hash("nancy_era5_data", _ARTIFACTS_TOML)
    if hash === nothing
        error("Artifact 'nancy_era5_data' not found in Artifacts.toml. " *
              "ERA5 data must be uploaded to Zenodo and the artifact registered first.")
    end
    if !artifact_exists(hash)
        ensure_artifact_installed("nancy_era5_data", _ARTIFACTS_TOML)
    end
    rootpath = artifact_path(hash)
    # Tarball extracts with a nancy_era5_data/ subdirectory
    datadir = joinpath(rootpath, "nancy_era5_data")
    if !isdir(datadir)
        datadir = rootpath
    end
    sort(filter(f -> endswith(f, ".nc"), readdir(datadir, join=true)))
end

"""
    smoky_era5_files()

Return a sorted vector of file paths to the Smoky ERA5 meteorological data files.

On first call, triggers a download from Zenodo. Subsequent calls use the cached artifact.

# Returns
- `Vector{String}` — sorted paths to ERA5 NetCDF files covering 31 Aug – 2 Sep 1957

# Example
```julia
met_files = smoky_era5_files()
results = run_simulation!(state, met_files, ...)
```
"""
function smoky_era5_files(; local_dir::Union{String,Nothing}=nothing)
    # If a local directory is provided and exists, use it directly
    if local_dir !== nothing && isdir(local_dir)
        files = sort(filter(f -> endswith(f, ".nc"), readdir(local_dir, join=true)))
        if !isempty(files)
            return files
        end
    end
    hash = artifact_hash("smoky_era5_data", _ARTIFACTS_TOML)
    if hash === nothing
        error("Artifact 'smoky_era5_data' not found in Artifacts.toml. " *
              "ERA5 data must be downloaded, merged, and uploaded to Zenodo first.\n" *
              "See examples/smoky_example/ for download and merge scripts.")
    end
    if !artifact_exists(hash)
        ensure_artifact_installed("smoky_era5_data", _ARTIFACTS_TOML)
    end
    rootpath = artifact_path(hash)
    datadir = joinpath(rootpath, "smoky_era5_data")
    if !isdir(datadir)
        datadir = rootpath
    end
    sort(filter(f -> endswith(f, ".nc"), readdir(datadir, join=true)))
end

"""
    etex_era5_files()

Return a sorted vector of file paths to the ETEX-1 ERA5 meteorological data files.

On first call, triggers a download from Zenodo (~1 GB). Subsequent calls
use the cached artifact.

# Returns
- `Vector{String}` — sorted paths to 35 ERA5 NetCDF files covering 23–27 October 1994

# Example
```julia
met_files = etex_era5_files()
results = run_simulation!(state, met_files, ...)
```
"""
function etex_era5_files()
    hash = artifact_hash("etex_era5_data", _ARTIFACTS_TOML)
    if hash === nothing
        error("Artifact 'etex_era5_data' not found in Artifacts.toml. " *
              "ERA5 data must be uploaded to Zenodo and the artifact registered first.")
    end
    if !artifact_exists(hash)
        ensure_artifact_installed("etex_era5_data", _ARTIFACTS_TOML)
    end
    rootpath = artifact_path(hash)
    datadir = joinpath(rootpath, "etex_era5_data")
    if !isdir(datadir)
        datadir = rootpath
    end
    sort(filter(f -> endswith(f, "_snap.nc"), readdir(datadir, join=true)))
end
