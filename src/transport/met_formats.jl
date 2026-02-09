# Met Data Format Dispatch
#
# Support for different meteorological data formats (ERA5 reanalysis)
# using Julia's multiple dispatch.

using NCDatasets

# Debug flag for EDCOMP vertical velocity computation
# Set to true to print detailed diagnostic output
const DEBUG_EDCOMP = false

"""
    MetFormat

Abstract type for meteorological data formats.

Concrete implementations:
- `ERA5Format`: ECMWF ERA5 reanalysis (model levels)
- `ERA5RawFormat`: Raw CDS ERA5 (model levels)
"""
abstract type MetFormat end

"""
    ERA5Format <: MetFormat

ERA5 reanalysis data preprocessed via fimex.

Variable names:
- Dimensions: `longitude`, `latitude`, `hybrid`
- Wind: `x_wind_ml`, `y_wind_ml`
- Temperature: `air_temperature_ml`
- Surface pressure: `surface_air_pressure`
- Vertical: `ap`, `b` coefficients
"""
struct ERA5Format <: MetFormat end

"""
    ERA5RawFormat <: MetFormat

ERA5 reanalysis data in raw CDS (Copernicus) format.

Variable names:
- Dimensions: `longitude`, `latitude`, `model_level`
- Wind: `u`, `v`
- Temperature: `t`
- Vertical velocity: `w` (optional)
- Surface pressure: `sp` (in separate surface file)
- Time: `valid_time`

Note: Requires separate surface file for `sp` (surface pressure).
The raw format uses `model_level` dimension instead of `hybrid`.
"""
struct ERA5RawFormat <: MetFormat end


"""
    detect_met_format(filepath::String) -> MetFormat

Automatically detect meteorological data format from NetCDF file.

Supported formats:
- ERA5Format: ERA5 preprocessed via fimex (x_wind_ml, y_wind_ml, air_temperature_ml)
- ERA5RawFormat: Raw CDS ERA5 (u, v, t with model_level dimension)
"""
function detect_met_format(filepath::String)
    NCDataset(filepath) do ds
        # ERA5 preprocessed uses "hybrid" dimension
        # Raw ERA5 uses "model_level" dimension
        # Distinguish by checking variable names:
        # - ERA5 preprocessed: x_wind_ml, y_wind_ml, air_temperature_ml (model levels)
        # - ERA5 Raw: u, v, t (with model_level dimension)
        if haskey(ds.dim, "hybrid") || haskey(ds.dim, "model_level")
            # Check for ERA5-specific preprocessed variable names
            if haskey(ds, "x_wind_ml") || haskey(ds, "air_temperature_ml")
                return ERA5Format()
            # Check for raw CDS ERA5 format (model_level dimension with u, v, t variables)
            elseif haskey(ds.dim, "model_level") && haskey(ds, "u") && haskey(ds, "v") && haskey(ds, "t")
                return ERA5RawFormat()
            else
                error("Unknown met format: found hybrid/model_level dimension but cannot identify ERA5 variables")
            end
        else
            error("Unknown met format: file must have 'model_level' or 'hybrid' dimension")
        end
    end
end

"""
    get_met_dimensions(format::MetFormat, ds::NCDataset) -> (nx, ny, nk)

Get grid dimensions for the given format.
"""
function get_met_dimensions(::ERA5Format, ds::NCDataset)
    # ERA5 preprocessed files use "hybrid" dimension
    return (
        length(ds["longitude"]),
        length(ds["latitude"]),
        length(ds["hybrid"])
    )
end

function get_met_dimensions(::ERA5RawFormat, ds::NCDataset)
    # Raw CDS ERA5 files use "model_level" dimension
    return (
        length(ds["longitude"]),
        length(ds["latitude"]),
        length(ds["model_level"])
    )
end


"""
    get_vertical_levels(format::MetFormat, ds::NCDataset) -> (vlevel, blevel)

Get vertical level coefficients for the given format.

Returns (vlevel, blevel) where:
- vlevel = (ap/p0 + b) for ERA5 sigma-hybrid coordinates
- blevel = b coefficients
"""
function get_vertical_levels(::ERA5Format, ds::NCDataset)
    ap = ds["ap"][:]
    b = ds["b"][:]
    p0_pa = 100000.0
    vlevel = (ap ./ p0_pa .+ b)
    return (vlevel, b)
end

function get_vertical_levels(::ERA5RawFormat, ds::NCDataset)
    # Raw CDS ERA5 doesn't have ap/b in the same file
    # Need to load from a separate surface file or use ERA5 L137 standard coefficients
    # For now, use hardcoded ERA5 L137 hybrid coefficients
    nk = length(ds["model_level"])
    # ERA5 uses 137 model levels with standard hybrid coefficients
    # These are the standard ECMWF L137 coefficients
    if nk == 137
        # Use standard L137 coefficients (would need full table here)
        # For now, return placeholder - full implementation would load from constants
        ap = zeros(Float64, nk)
        b = zeros(Float64, nk)
        # Approximate: surface b=1, top b=0
        for k in 1:nk
            b[k] = 1.0 - (k - 1) / (nk - 1)
        end
        p0_pa = 100000.0
        vlevel = b  # Simplified approximation
        return (vlevel, b)
    else
        error("ERA5RawFormat only supports 137 model levels, got $nk")
    end
end


"""
    get_time_variable(format::MetFormat, ds::NCDataset) -> Vector

Get time coordinate variable for the given format.
"""
function get_time_variable(::ERA5Format, ds::NCDataset)
    # ERA5 preprocessed files use "time" dimension
    return ds["time"][:]
end

function get_time_variable(::ERA5RawFormat, ds::NCDataset)
    # Raw CDS ERA5 uses "valid_time" dimension
    return ds["valid_time"][:]
end


"""
    read_initial_met_fields!(format::MetFormat, met_fields, ds::NCDataset,
                             window_idx::Int, next_idx::Int)

Initialize meteorological fields for the FIRST time (before simulation loop).

Loads TWO time slices to bracket the simulation start time:
- *1 fields: earlier timestep (e.g., 13:00 for 13:10 release)
- *2 fields: later timestep (e.g., 14:00 for 13:10 release)

This enables temporal interpolation from the start. NO SWAPPING occurs.
Call this ONCE before entering the main simulation loop.
"""
function read_initial_met_fields! end

"""
    read_met_fields!(format::MetFormat, met_fields, ds::NCDataset,
                     window_idx::Int, next_idx::Int)

Update meteorological fields for the NEXT time window (during simulation loop).

ALWAYS performs these steps:
1. Swap *2 → *1 (preserve temporal continuity)
2. Read new data into *2 fields only

Call this for ALL timesteps AFTER initialization.
"""

"""
    is_latitude_reversed(ds::NCDataset) -> Bool

Check if latitude coordinates are in reverse order (north-to-south).
ERA5 reanalysis data often has latitudes ordered from north to south.
"""
function is_latitude_reversed(ds::NCDataset)
    if !haskey(ds, "latitude")
        return false
    end
    lat = ds["latitude"][:]
    if length(lat) < 2
        return false
    end
    # Reversed if first latitude > last latitude (north-to-south ordering)
    return lat[1] > lat[end]
end

function read_initial_met_fields!(::ERA5Format, met_fields, ds::NCDataset,
                                  window_idx::Int, next_idx::Int)
    T = eltype(met_fields.u1)

    # Load hybrid coefficients (same for both timesteps)
    ap = T.(ds["ap"][:]) ./ T(100.0)  # Pa → hPa
    b = T.(ds["b"][:])
    p0_hpa = haskey(ds, "p0") ? T(ds["p0"][]) / T(100.0) : T(1000.0)
    nk = length(ap)

    # Treatment of hybrid coefficients:
    # ERA5's ap/bp are loaded and reversed so that alevel[1]=surface, alevel[nk]=TOA.
    # ahalf is then computed as half-levels by averaging adjacent full-levels.
    #
    # Convention: alevel(1)=surface, alevel(nk)=TOA
    # ERA5 native: ap[1]=TOA, ap[137]=surface
    # Reversal: klevel(k) = nk - k + 1

    # Step 1: Load reversed ap into alevel array
    # alevel[1] = ap[nk] (near surface), alevel[nk] = ap[1] (TOA)
    for k in 1:nk
        met_fields.alevel[k] = ap[nk + 1 - k]
        met_fields.blevel[k] = b[nk + 1 - k]
    end

    # Step 2: Compute ahalf from alevel (half-level averaging)
    met_fields.ahalf[1] = met_fields.alevel[1]  # Surface
    met_fields.bhalf[1] = met_fields.blevel[1]

    # ahalf(k) = (alevel(k) + alevel(k+1)) / 2  (average forward)
    for k in 2:nk-1
        met_fields.ahalf[k] = (met_fields.alevel[k] + met_fields.alevel[k + 1]) * T(0.5)
        met_fields.bhalf[k] = (met_fields.blevel[k] + met_fields.blevel[k + 1]) * T(0.5)
    end

    # After reversal: alevel[1]=surface, alevel[nk]=TOA
    # TOA boundary
    met_fields.ahalf[nk] = met_fields.alevel[nk]
    met_fields.bhalf[nk] = met_fields.blevel[nk]
    # Convenience extra slot for edcomp indexing (nz+1): equals TOA boundary
    met_fields.ahalf[nk + 1] = met_fields.alevel[nk]
    met_fields.bhalf[nk + 1] = met_fields.blevel[nk]

    # Step 3: Compute vlevel and vhalf
    # NOTE: vlevel is used for height↔sigma conversion and must match alevel/blevel indices.
    # The wind shift (reverse_and_shift_levels) is handled separately in particle_dynamics.jl
    for k in 1:nk
        met_fields.vlevel[k] = met_fields.alevel[k] / p0_hpa + met_fields.blevel[k]
    end
    for k in 1:(nk + 1)
        met_fields.vhalf[k] = met_fields.ahalf[k] / p0_hpa + met_fields.bhalf[k]
    end

    # Load BOTH timesteps (NO swapping - this is initialization)
    # CRITICAL: Reverse vertical dimension to match reversed hybrid coefficients!
    # ERA5 native: [:,:,1] = TOA, [:,:,137] = surface
    # After reversal: [:,:,1] = surface, [:,:,137] = TOA (matches alevel ordering)
    #
    # Store PHYSICAL v-wind (no negation) for correct EDCOMP divergence calculation
    # Negation for grid convention is applied later in create_wind_interpolants

    # Reverse wind arrays to match vlevel ordering (surface at k=1, TOA at k=nk)
    function reverse_levels!(dest::AbstractArray{T,3}, src::AbstractArray{T,3}) where T
        dest .= reverse(src, dims=3)
        return dest
    end

    # Apply level reversal to WIND fields
    reverse_levels!(met_fields.u1, ds["x_wind_ml"][:, :, :, window_idx])
    reverse_levels!(met_fields.v1, ds["y_wind_ml"][:, :, :, window_idx])
    # Temperature: simple reverse WITHOUT shift (preserves BL calculation gradients)
    met_fields.t1 .= reverse(ds["air_temperature_ml"][:, :, :, window_idx], dims=3)
    met_fields.ps1 .= ds["surface_air_pressure"][:, :, window_idx] ./ 100.0f0

    # Apply k-level shift to WIND fields only (for advection parity)
    reverse_levels!(met_fields.u2, ds["x_wind_ml"][:, :, :, next_idx])
    reverse_levels!(met_fields.v2, ds["y_wind_ml"][:, :, :, next_idx])
    # Temperature: simple reverse WITHOUT shift (preserves BL calculation gradients)
    met_fields.t2 .= reverse(ds["air_temperature_ml"][:, :, :, next_idx], dims=3)
    met_fields.ps2 .= ds["surface_air_pressure"][:, :, next_idx] ./ 100.0f0

    # ERA5 provides absolute temperature T (K). Do not convert to θ here.

    # Latitude orientation and map factors
    # ERA5 often stores latitude from north->south; reverse Y dimension to make j increase northward
    lat_rev = is_latitude_reversed(ds)
    if lat_rev
        met_fields.u1 .= reverse(met_fields.u1, dims=2)
        met_fields.v1 .= reverse(met_fields.v1, dims=2)
        met_fields.t1 .= reverse(met_fields.t1, dims=2)
        met_fields.ps1 .= reverse(met_fields.ps1, dims=2)
        met_fields.u2 .= reverse(met_fields.u2, dims=2)
        met_fields.v2 .= reverse(met_fields.v2, dims=2)
        met_fields.t2 .= reverse(met_fields.t2, dims=2)
        met_fields.ps2 .= reverse(met_fields.ps2, dims=2)
    end

    # Use actual lon/lat increments from file to compute constant dx, dy
    R_earth = T(6.371e6)
    lons = collect(T.(ds["longitude"]))
    lats = collect(T.(ds["latitude"]))
    if lat_rev
        lats = reverse(lats)
    end
    dlon_deg = abs(lons[2] - lons[1])
    dlat_deg = abs(lats[2] - lats[1])
    dx_m = R_earth * dlon_deg * T(π) / T(180.0)
    dy_m = R_earth * dlat_deg * T(π) / T(180.0)

    # Map scale factors per latitude row
    Transport.compute_map_scale_factors!(met_fields.xm, met_fields.ym, lats)

    # Compute vertical velocity for BOTH timesteps (edcomp separately for each time slice)
    era5_format = ERA5Format()

    # 1a) Compute w1 from u1/v1/ps1 (window_idx time slice)
    fill!(met_fields.w1, zero(T))
    has_omega = haskey(ds, "omega_ml")
    if has_omega
        # Load and reverse omega levels (surface at k=1)
        reverse_levels!(met_fields.w1, ds["omega_ml"][:, :, :, window_idx])
        if lat_rev
            met_fields.w1 .= reverse(met_fields.w1, dims=2)
        end
        # Convert omega (Pa/s) to sigma-dot
        # Use ps1 in hPa (already converted above)
        convert_omega_to_sigmadot!(met_fields.w1, met_fields.ps1,
                                  met_fields.ahalf, met_fields.bhalf, met_fields.vhalf)
    end

    compute_etadot_from_continuity!(
        era5_format,
        met_fields.w1, met_fields.u1, met_fields.v1, met_fields.ps1,
        met_fields.xm, met_fields.ym, met_fields.ahalf, met_fields.bhalf, met_fields.vhalf,
        dx_m, dy_m,
        averaging = !has_omega
    )
    if !has_omega
        # Continuity-only branch: compensate for internal 0.5 averaging
        met_fields.w1 .*= T(2.0)
    end
    # Enforce w=0 at surface level
    met_fields.w1[:, :, 1] .= zero(T)

    # 1b) Compute w2 from u2/v2/ps2 (next_idx time slice)
    fill!(met_fields.w2, zero(T))
    if has_omega
        # Load and reverse omega levels (surface at k=1)
        reverse_levels!(met_fields.w2, ds["omega_ml"][:, :, :, next_idx])
        if lat_rev
            met_fields.w2 .= reverse(met_fields.w2, dims=2)
        end
        # Convert omega (Pa/s) to sigma-dot
        convert_omega_to_sigmadot!(met_fields.w2, met_fields.ps2,
                                  met_fields.ahalf, met_fields.bhalf, met_fields.vhalf)
    end

    compute_etadot_from_continuity!(
        era5_format,
        met_fields.w2, met_fields.u2, met_fields.v2, met_fields.ps2,
        met_fields.xm, met_fields.ym, met_fields.ahalf, met_fields.bhalf, met_fields.vhalf,
        dx_m, dy_m,
        averaging = !has_omega
    )
    if !has_omega
        # Continuity-only branch: compensate for internal 0.5 averaging
        met_fields.w2 .*= T(2.0)
    end
    # Enforce w=0 at surface level
    met_fields.w2[:, :, 1] .= zero(T)

    # DEBUG: Print final w values at particle location for comparison
    if DEBUG_EDCOMP
        println("=== w1 FINAL (continuity-only, from window_idx=$window_idx) ===")
        println("  w1[68,79,4] = $(met_fields.w1[68,79,4])")
        println("  w1[68,79,5] = $(met_fields.w1[68,79,5])")
        println("=== w2 AFTER *2.0 (from next_idx=$next_idx) ===")
        println("  w2[68,79,4] = $(met_fields.w2[68,79,4])")
        println("  w2[68,79,5] = $(met_fields.w2[68,79,5])")
        println("===============================================")
    end

    # Load precipitation for both timesteps (check both precipitation_flux and precipitation_rate)
    precip_var = haskey(ds, "precipitation_flux") ? "precipitation_flux" :
                 haskey(ds, "precipitation_rate") ? "precipitation_rate" : nothing
    if precip_var !== nothing
        precip = ds[precip_var]
        units = haskey(precip.attrib, "units") ? String(precip.attrib["units"]) : ""
        # Convert kg/m²/s to mm/hr: 1 kg/m²/s = 3600 mm/hr
        scale = occursin("kg", lowercase(units)) && occursin("s", lowercase(units)) ? T(3600.0) : T(1.0)
        p1 = precip[:, :, window_idx] .* scale
        p2 = precip[:, :, next_idx] .* scale
        if lat_rev
            met_fields.precip1 .= reverse(p1, dims=2)
            met_fields.precip2 .= reverse(p2, dims=2)
        else
            met_fields.precip1 .= p1
            met_fields.precip2 .= p2
        end
    else
        fill!(met_fields.precip1, zero(T))
        fill!(met_fields.precip2, zero(T))
    end

    # Convert absolute temperature T to potential temperature θ.
    # The hypsometric height formula requires θ, not T.
    # θ = T × (p0/p)^(R/cp) where p0 = 1000 hPa, R/cp ≈ 0.286
    R_CP = T(287.058 / 1005.0)  # R/cp ≈ 0.286
    for k in 2:size(met_fields.t1, 3)  # Skip k=1 (surface)
        for j in 1:size(met_fields.t1, 2)
            for i in 1:size(met_fields.t1, 1)
                # Compute pressure at this grid point (hPa)
                p1 = met_fields.alevel[k] + met_fields.blevel[k] * met_fields.ps1[i, j]
                p2 = met_fields.alevel[k] + met_fields.blevel[k] * met_fields.ps2[i, j]

                # t2thetafac = (p0/p)^(R/cp) where p0 = 1000 hPa
                t2thetafac1 = (T(1000.0) / p1)^R_CP
                t2thetafac2 = (T(1000.0) / p2)^R_CP

                # Convert absolute temperature T to potential temperature θ
                met_fields.t1[i, j, k] *= t2thetafac1
                met_fields.t2[i, j, k] *= t2thetafac2
            end
        end
    end

    # Compute pressure and heights for both timesteps
    compute_pressure_from_hybrid!(met_fields, 1)
    compute_pressure_from_hybrid!(met_fields, 2)

    compute_model_heights!(ERA5Format, met_fields, 1)
    compute_model_heights!(ERA5Format, met_fields, 2)

    compute_boundary_layer!(ERA5Format, met_fields, time_level=1)
    compute_boundary_layer!(ERA5Format, met_fields, time_level=2)
end


function read_met_fields!(::ERA5Format, met_fields, ds::NCDataset,
                         window_idx::Int, next_idx::Int)
    T = eltype(met_fields.u1)

    # STEP 1: ALWAYS swap *2 → *1 to maintain temporal continuity
    # This preserves the previous timestep's data across file boundaries
    met_fields.u1 .= met_fields.u2
    met_fields.v1 .= met_fields.v2
    met_fields.t1 .= met_fields.t2
    met_fields.ps1 .= met_fields.ps2
    met_fields.w1 .= met_fields.w2
    met_fields.p1 .= met_fields.p2
    met_fields.hlevel1 .= met_fields.hlevel2
    met_fields.precip1 .= met_fields.precip2
    met_fields.hbl1 .= met_fields.hbl2
    met_fields.bl1 .= met_fields.bl2

    # ERA5 hybrid coefficients are time-invariant but we reload for consistency
    ap = T.(ds["ap"][:]) ./ T(100.0)  # Pa → hPa
    b = T.(ds["b"][:])
    p0_hpa = haskey(ds, "p0") ? T(ds["p0"][]) / T(100.0) : T(1000.0)
    nk = length(ap)

    # Reversed hybrid coefficient loading (same as initialisation)
    for k in 1:nk
        met_fields.alevel[k] = ap[nk + 1 - k]
        met_fields.blevel[k] = b[nk + 1 - k]
    end

    # ahalf computation (average) with TOA boundary
    met_fields.ahalf[1] = met_fields.alevel[1]
    met_fields.bhalf[1] = met_fields.blevel[1]

    for k in 2:nk-1
        met_fields.ahalf[k] = (met_fields.alevel[k] + met_fields.alevel[k + 1]) * T(0.5)
        met_fields.bhalf[k] = (met_fields.blevel[k] + met_fields.blevel[k + 1]) * T(0.5)
    end

    met_fields.ahalf[nk] = met_fields.alevel[nk]
    met_fields.bhalf[nk] = met_fields.blevel[nk]
    met_fields.ahalf[nk + 1] = met_fields.alevel[nk]
    met_fields.bhalf[nk + 1] = met_fields.blevel[nk]

    for k in 1:nk
        met_fields.vlevel[k] = met_fields.alevel[k] / p0_hpa + met_fields.blevel[k]
    end
    for k in 1:(nk + 1)
        met_fields.vhalf[k] = met_fields.ahalf[k] / p0_hpa + met_fields.bhalf[k]
    end

    # STEP 2: ALWAYS read ONLY next_idx into *2 fields
    # The *1 fields already have the previous *2 data from the swap above
    # Reverse vertical dimension to match vlevel ordering (surface at k=1)
    function reverse_levels!(dest::AbstractArray{T,3}, src::AbstractArray{T,3}) where T
        dest .= reverse(src, dims=3)
        return dest
    end
    # Apply level reversal to WIND fields
    reverse_levels!(met_fields.u2, ds["x_wind_ml"][:, :, :, next_idx])
    reverse_levels!(met_fields.v2, ds["y_wind_ml"][:, :, :, next_idx])
    # Temperature: simple reverse WITHOUT shift (preserves BL calculation gradients)
    met_fields.t2 .= reverse(ds["air_temperature_ml"][:, :, :, next_idx], dims=3)
    met_fields.ps2 .= ds["surface_air_pressure"][:, :, next_idx] ./ 100.0f0

    # ERA5 provides absolute temperature T (K). Do not convert to θ here.

    # STEP 3: Compute vertical velocity for new *2 data only
    # w1 already has the swapped w2 from previous timestep
    fill!(met_fields.w2, zero(T))

    # Latitude orientation and grid spacing: use file coordinates
    lat_rev = is_latitude_reversed(ds)
    if lat_rev
        met_fields.u2 .= reverse(met_fields.u2, dims=2)
        met_fields.v2 .= reverse(met_fields.v2, dims=2)
        met_fields.t2 .= reverse(met_fields.t2, dims=2)
        met_fields.ps2 .= reverse(met_fields.ps2, dims=2)
    end

    # Grid spacing: arc-length on sphere (constant per window)
    R_earth = T(6.371e6)
    lons = collect(T.(ds["longitude"]))
    lats = collect(T.(ds["latitude"]))
    if lat_rev
        lats = reverse(lats)
    end
    dlon_deg = abs(lons[2] - lons[1])
    dlat_deg = abs(lats[2] - lats[1])
    dx_m = R_earth * dlon_deg * T(π) / T(180.0)
    dy_m = R_earth * dlat_deg * T(π) / T(180.0)

    # Map scale factors
    Transport.compute_map_scale_factors!(met_fields.xm, met_fields.ym, lats)

    has_omega = haskey(ds, "omega_ml")
    if has_omega
        # Load and reverse omega levels (surface at k=1)
        reverse_levels!(met_fields.w2, ds["omega_ml"][:, :, :, next_idx])
        if lat_rev
            met_fields.w2 .= reverse(met_fields.w2, dims=2)
        end
        # Convert omega (Pa/s) to sigma-dot
        convert_omega_to_sigmadot!(met_fields.w2, met_fields.ps2,
                                  met_fields.ahalf, met_fields.bhalf, met_fields.vhalf)
    end

    # Compute vertical velocity from continuity equation
    era5_format = ERA5Format()
    compute_etadot_from_continuity!(
        era5_format,
        met_fields.w2, met_fields.u2, met_fields.v2, met_fields.ps2,
        met_fields.xm, met_fields.ym, met_fields.ahalf, met_fields.bhalf, met_fields.vhalf,
        dx_m, dy_m,
        averaging = !has_omega
    )
    if !has_omega
        # Continuity-only branch: compensate for internal 0.5 averaging
        met_fields.w2 .*= T(2.0)
    end
    # Enforce w=0 at surface level
    met_fields.w2[:, :, 1] .= zero(T)

    # STEP 4: Load precipitation for new *2 data only
    if haskey(ds, "precipitation_flux")
        precip = ds["precipitation_flux"]
        units = haskey(precip.attrib, "units") ? String(precip.attrib["units"]) : ""
        scale = units == "kg/m^2/s" ? T(3600.0) : T(1.0)
        p2 = precip[:, :, next_idx] .* scale
        met_fields.precip2 .= lat_rev ? reverse(p2, dims=2) : p2
    elseif haskey(ds, "precipitation_rate")
        precip = ds["precipitation_rate"]
        units = haskey(precip.attrib, "units") ? String(precip.attrib["units"]) : ""
        scale = units == "kg/m^2/s" ? T(3600.0) : T(1.0)
        p2 = precip[:, :, next_idx] .* scale
        met_fields.precip2 .= lat_rev ? reverse(p2, dims=2) : p2
    else
        fill!(met_fields.precip2, zero(T))
    end

    # Compute 3D pressure and heights (all in file order: k=1 is TOA)
    compute_pressure_from_hybrid!(met_fields, 1)
    compute_pressure_from_hybrid!(met_fields, 2)
    compute_model_heights!(ERA5Format, met_fields, 1)
    compute_model_heights!(ERA5Format, met_fields, 2)

    compute_boundary_layer!(ERA5Format, met_fields, time_level=1)
    compute_boundary_layer!(ERA5Format, met_fields, time_level=2)
end


export MetFormat, ERA5Format, ERA5RawFormat
export detect_met_format, get_met_dimensions, get_vertical_levels
export get_time_variable, read_initial_met_fields!, read_met_fields!
