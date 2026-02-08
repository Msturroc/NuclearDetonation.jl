# Data access helpers for ERA5 meteorological data via Julia Artifacts
# Provides convenience functions for downloading and locating met data files.

using Pkg.Artifacts

export nancy_era5_files

const _ARTIFACTS_TOML = joinpath(@__DIR__, "..", "..", "Artifacts.toml")

"""
    nancy_era5_files()

Return a sorted vector of file paths to the Nancy ERA5 meteorological data files.

On first call, triggers a lazy download from Zenodo (~300 MB). Subsequent calls
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
        ensure_artifact_installed("nancy_era5_data", _ARTIFACTS_TOML; lazy_download=true)
    end
    rootpath = artifact_path(hash)
    sort(filter(f -> endswith(f, ".nc"), readdir(rootpath, join=true)))
end
