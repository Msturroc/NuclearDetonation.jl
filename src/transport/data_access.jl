# Data access helpers for ERA5 meteorological data via Julia Artifacts
# Provides convenience functions for downloading and locating met data files.

using Pkg.Artifacts

export nancy_era5_files

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
snapshots = run_simulation!(state, met_files, ...)
```
"""
function nancy_era5_files()
    rootpath = artifact"nancy_era5_data"
    sort(filter(f -> endswith(f, ".nc"), readdir(rootpath, join=true)))
end
