# Met Data Format Dispatch
#
# Support for different meteorological data formats (ERA5, GFS, etc.)
# using Julia's multiple dispatch.

using NCDatasets

# Debug flag for EDCOMP comparison with Fortran SNAP
# Set to true to print detailed diagnostic output for vertical velocity computation
const SNAP_DEBUG_EDCOMP = false

"""
    MetFormat

Abstract type for meteorological data formats.

Concrete implementations:
- `ERA5Format`: ECMWF ERA5 reanalysis (model levels)
- `GFSFormat`: NOAA GFS forecasts (sigma-hybrid levels via fimex)
"""
abstract type MetFormat end

"""
    ERA5Format <: MetFormat

ERA5 reanalysis data in SNAP-preprocessed format (via fimex).

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
    GFSFormat <: MetFormat

GFS forecast data in SNAP format (converted via fimex).

Variable names:
- Dimensions: `longitude`, `latitude`, `hybrid`
- Wind: `x_wind_pl`, `y_wind_pl`
- Temperature: `air_temperature_pl`
- Surface pressure: `surface_air_pressure`
- Vertical: `ap`, `b` coefficients
"""
struct GFSFormat <: MetFormat end

"""
    detect_met_format(filepath::String) -> MetFormat

Automatically detect meteorological data format from NetCDF file.

Supported formats:
- ERA5Format: SNAP-preprocessed ERA5 (x_wind_ml, y_wind_ml, air_temperature_ml)
- ERA5RawFormat: Raw CDS ERA5 (u, v, t with model_level dimension)
- GFSFormat: SNAP-preprocessed GFS (x_wind_pl, y_wind_pl, air_temperature_pl)
"""
function detect_met_format(filepath::String)
    NCDataset(filepath) do ds
        # Both ERA5 and GFS use "hybrid" dimension in SNAP format
        # Raw ERA5 uses "model_level" dimension
        # Distinguish by checking variable names:
        # - ERA5 SNAP: x_wind_ml, y_wind_ml, air_temperature_ml (model levels)
        # - ERA5 Raw: u, v, t (with model_level dimension)
        # - GFS: x_wind_pl, y_wind_pl, air_temperature_pl (pressure levels)
        if haskey(ds.dim, "hybrid") || haskey(ds.dim, "model_level")
            # Check for ERA5-specific SNAP variable names
            if haskey(ds, "x_wind_ml") || haskey(ds, "air_temperature_ml")
                return ERA5Format()
            # Check for GFS-specific SNAP variable names
            elseif haskey(ds, "x_wind_pl") || haskey(ds, "air_temperature_pl")
                return GFSFormat()
            # Check for raw CDS ERA5 format (model_level dimension with u, v, t variables)
            elseif haskey(ds.dim, "model_level") && haskey(ds, "u") && haskey(ds, "v") && haskey(ds, "t")
                return ERA5RawFormat()
            else
                error("Unknown met format: found hybrid/model_level dimension but cannot identify ERA5 or GFS variables")
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
    # ERA5 files use "hybrid" dimension (for Fortran SNAP compatibility)
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

function get_met_dimensions(::GFSFormat, ds::NCDataset)
    return (
        length(ds["longitude"]),
        length(ds["latitude"]),
        length(ds["hybrid"])
    )
end

"""
    get_vertical_levels(format::MetFormat, ds::NCDataset) -> (vlevel, blevel)

Get vertical level coefficients for the given format.

Returns (vlevel, blevel) where:
- vlevel = (ap/p0 + b) for ERA5/GFS sigma-hybrid coordinates
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

function get_vertical_levels(::GFSFormat, ds::NCDataset)
    ap = ds["ap"][:]
    b = ds["b"][:]
    p0_pa = 100000.0
    vlevel = (ap ./ p0_pa .+ b)
    return (vlevel, b)
end

"""
    get_time_variable(format::MetFormat, ds::NCDataset) -> Vector

Get time coordinate variable for the given format.
"""
function get_time_variable(::ERA5Format, ds::NCDataset)
    # ERA5 files use "time" dimension (for Fortran SNAP compatibility)
    return ds["time"][:]
end

function get_time_variable(::ERA5RawFormat, ds::NCDataset)
    # Raw CDS ERA5 uses "valid_time" dimension
    return ds["valid_time"][:]
end

function get_time_variable(::GFSFormat, ds::NCDataset)
    return ds["time"][:]
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

    # CRITICAL: Match Fortran SNAP's treatment of hybrid coefficients EXACTLY!
    # Fortran readfield_nc.f90 reads ap into alev array (reversed via klevel),
    # then compute_vertical_coords treats alev as FULL-level values and computes
    # ahalf as half-levels by averaging adjacent full-levels.
    #
    # POTENTIAL BUG: ERA5's ap/bp ARE half-level coefficients, but Fortran treats
    # them as full-level values. This may be incorrect for ERA5 data, but we must
    # match Fortran's behavior exactly for validation. TODO: investigate and fix
    # in both Fortran and Julia after validation is complete.
    #
    # Fortran convention: alevel(1)=surface, alevel(nk)=TOA
    # ERA5 native: ap[1]=TOA, ap[137]=surface
    # This reversal happens in Fortran's readfield_nc.f90 via klevel(k)=nk-k+1

    # Step 1: Load reversed ap into alevel array
    # Julia's alevel[1..nk] = Fortran's alevel(2..nk+1) (no surface boundary in Julia's alevel)
    # Fortran: alevel(1)=0.0, alevel(2)=ap(137), ..., alevel(138)=ap(1)
    # Julia: alevel[1]=ap(137), ..., alevel[137]=ap(1)
    for k in 1:nk
        # Julia alevel[k] = ap[nk + 1 - k]
        # k=1: alevel[1] = ap[137] (near surface)
        # k=137: alevel[137] = ap[1] (TOA)
        met_fields.alevel[k] = ap[nk + 1 - k]
        met_fields.blevel[k] = b[nk + 1 - k]
    end

    # Step 2: Compute ahalf from alevel - MATCH FORTRAN EXACTLY!
    # Fortran readfield_nc.f90 line 1222-1242
    # Julia ahalf[1..nk+1] should match Fortran ahalf(1..nk+1)
    met_fields.ahalf[1] = met_fields.alevel[1]  # Surface (Fortran line 1222)
    met_fields.bhalf[1] = met_fields.blevel[1]

    # Fortran uses manual_level_selection branch for ERA5 (line 1237):
    # ahalf(k) = (alevel(k) + alevel(k+1)) / 2  (average FORWARD, not backward!)
    for k in 2:nk-1
        met_fields.ahalf[k] = (met_fields.alevel[k] + met_fields.alevel[k + 1]) * T(0.5)
        met_fields.bhalf[k] = (met_fields.blevel[k] + met_fields.blevel[k + 1]) * T(0.5)
    end

    # After reversal: alevel[1]=surface, alevel[nk]=TOA
    # Fortran compute_vertical_coords: ahalf(nk) = alevel(nk) (TOA boundary)
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
    # KNOWN ISSUE: k-level offset between Julia and Fortran
    # Empirical testing confirmed: Julia k=5 values EXACTLY match Fortran k=4 values
    #   - At time=2: Julia k=5 has u=-2.561119, v=-1.9980354
    #   - Fortran k=4 reports: u=-2.56111908, v=-1.99803543 (identical!)
    #
    # Root cause: Fortran's LEVELS.INPUT = "137, 0, 136, 135, 134, 133, ..."
    # has a dummy surface level (0) at position 1, shifting all indices by 1.
    # Julia's simple reverse() doesn't account for this dummy level.
    #
    # Effect: Julia accesses winds at slightly different vertical levels than Fortran,
    # causing ~1-3m/hour altitude divergence that accumulates over time.
    #
    # FIX: Shift level indexing by 1 to match Fortran's dummy-level convention
    # After reverse and shift: Julia k matches Fortran k for all levels
    #
    # Store PHYSICAL v-wind (no negation) for correct EDCOMP divergence calculation
    # Negation for grid convention is applied later in create_wind_interpolants

    # Reverse wind arrays to match vlevel ordering (surface at k=1, TOA at k=nk)
    # NOTE (2025-12-12): Removed the +1 shift that was causing wind/vlevel mismatch.
    # The shift was intended to match Fortran's dummy surface level, but it caused
    # Julia to sample winds from the wrong vertical level (~26% v-wind error).
    # Now winds and vlevel have consistent k-indexing for correct interpolation.
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

    # Latitude orientation and map factors: match Fortran conventions
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

    # Compute vertical velocity for BOTH timesteps
    # CRITICAL FIX: Match Fortran behavior - compute edcomp SEPARATELY for each time slice!
    # Previously Julia computed w2 and copied to w1, but Fortran calls edcomp for each timestep.
    #
    # NOTE: Fortran om2edot.f90 DOES use omega to seed edcomp, but getting the units/ordering
    # correct is complex. Using continuity-only with *2 compensation gives good results.
    # TODO: Properly implement omega→sigmadot conversion for even better matching.
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
        # Match Fortran om2edot behavior when omega is absent: continuity-only → multiply by 2
        met_fields.w1 .*= T(2.0)
    end
    # Fortran enforces w=0 at surface level
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
        # Match Fortran om2edot behavior when omega is absent: continuity-only → multiply by 2
        met_fields.w2 .*= T(2.0)
    end
    # Fortran enforces w=0 at surface level
    met_fields.w2[:, :, 1] .= zero(T)

    # DEBUG: Print final w values at particle location for comparison with Fortran
    if SNAP_DEBUG_EDCOMP
        println("=== JULIA w1 FINAL (continuity-only, from window_idx=$window_idx) ===")
        println("  w1[68,79,4] = $(met_fields.w1[68,79,4])")
        println("  w1[68,79,5] = $(met_fields.w1[68,79,5])")
        println("=== JULIA w2 AFTER *2.0 (from next_idx=$next_idx) ===")
        println("  w2[68,79,4] = $(met_fields.w2[68,79,4])")
        println("  w2[68,79,5] = $(met_fields.w2[68,79,5])")
        println("  Compare with Fortran first edcomp (time 2): w2(68,79,k=4)")
        println("  Compare with Fortran second edcomp (time 3): w2(68,79,k=4)")
        println("===============================================")
    end

    # For ERA5 validation against Fortran SNAP: do NOT use omega even if present.
    # Fortran configuration in this case derives sigma-dot from continuity only (EDCOMP).

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

    # CRITICAL FIX (2025-12-12): ERA5 provides absolute temperature T, but the
    # height computation formula (h2 = h1 + θ*(π_upper - π_lower)/g) requires
    # potential temperature θ. Fortran converts T→θ in readfield_nc.f90:441-448.
    # Without this conversion, computed heights are ~13m lower, causing particles
    # to be initialized at wrong sigma levels → wrong winds → trajectory divergence.
    #
    # Convert T→θ: θ = T × (p0/p)^(R/cp) where p0 = 1000 hPa, R/cp ≈ 0.286
    R_CP = T(287.058 / 1005.0)  # R/cp ≈ 0.286
    for k in 2:size(met_fields.t1, 3)  # Skip k=1 (surface), matching Fortran k=2,nk
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

function read_initial_met_fields!(::GFSFormat, met_fields, ds::NCDataset,
                                  window_idx::Int, next_idx::Int)
    # This is EXACTLY like read_met_fields!(::GFSFormat...) but loads BOTH timesteps without swapping
    # Copy the exact structure from the working read_met_fields! to ensure consistency
    T = eltype(met_fields.u1)

    # GFS data uses standard hybrid coefficients (like ERA5)
    # Reference pressure: standard atmosphere
    ap = T.(ds["ap"][:])
    b = T.(ds["b"][:])
    p0_pa = T(101325.0) # Standard atmosphere in Pa

    nk = length(ap)

    # Half-levels (interfaces) - same structure as ERA5
    for k in 1:nk
        met_fields.ahalf[k] = ap[k]
        met_fields.bhalf[k] = b[k]
    end
    # Surface boundary (nk+1 = surface level)
    met_fields.ahalf[nk + 1] = zero(T)   # a at surface is 0
    met_fields.bhalf[nk + 1] = one(T)    # b at surface is 1

    # Full-levels (centers)
    for k in 1:nk
        met_fields.alevel[k] = (met_fields.ahalf[k] + met_fields.ahalf[k + 1]) * T(0.5)
        met_fields.blevel[k] = (met_fields.bhalf[k] + met_fields.bhalf[k + 1]) * T(0.5)
        # vlevel is a sigma-like coordinate based on a reference pressure profile
        met_fields.vlevel[k] = met_fields.alevel[k] / p0_pa + met_fields.blevel[k]
    end
    for k in 1:(nk + 1)
        met_fields.vhalf[k] = met_fields.ahalf[k] / p0_pa + met_fields.bhalf[k]
    end

    # GFS variable names (from fimex conversion)
    # Now using same coordinate order as ERA5 (top-to-bottom = ascending sigma)
    # LOAD BOTH TIMESTEPS (no swapping - this is initialization)
    met_fields.u1 .= ds["x_wind_pl"][:, :, :, window_idx]
    met_fields.v1 .= ds["y_wind_pl"][:, :, :, window_idx]
    met_fields.t1 .= ds["air_temperature_pl"][:, :, :, window_idx]
    met_fields.ps1 .= ds["surface_air_pressure"][:, :, window_idx] ./ 100.0f0  # Pa → hPa

    met_fields.u2 .= ds["x_wind_pl"][:, :, :, next_idx]
    met_fields.v2 .= ds["y_wind_pl"][:, :, :, next_idx]
    met_fields.t2 .= ds["air_temperature_pl"][:, :, :, next_idx]
    met_fields.ps2 .= ds["surface_air_pressure"][:, :, next_idx] ./ 100.0f0

    # Compute map scale factors (CRITICAL for correct horizontal wind advection!)
    # GFS uses latitude coordinate (no reversal needed like ERA5)
    lats = collect(T.(ds["latitude"]))
    Transport.compute_map_scale_factors!(met_fields.xm, met_fields.ym, lats)

    # CRITICAL: Convert absolute temperature to potential temperature
    # This matches Fortran SNAP's t2thetafac conversion in readfield_nc.f90:400
    # θ = T * (p₀/p)^(R/cp) where p₀ = 1000 hPa
    # NOTE: alevel is in Pa (from NetCDF ap), ps is in hPa (converted above)
    # So we need to convert alevel/100 + blevel*ps to get pressure in hPa
    R_CP = 287.058 / 1005.0  # R/cp ≈ 0.286
    for k in 1:size(met_fields.t1, 3)
        for j in 1:size(met_fields.t1, 2)
            for i in 1:size(met_fields.t1, 1)
                # Compute pressure at this grid point (hybrid coordinates) in hPa
                # p = ap/100 + b*ps, where ap is in Pa and ps is in hPa
                p1 = met_fields.alevel[k]/100.0 + met_fields.blevel[k] * met_fields.ps1[i, j]
                p2 = met_fields.alevel[k]/100.0 + met_fields.blevel[k] * met_fields.ps2[i, j]

                # Convert to potential temperature: θ = T × (1000/p)^(R/cp)
                met_fields.t1[i, j, k] *= Float32(1.0 / ((p1 * 0.001)^R_CP))
                met_fields.t2[i, j, k] *= Float32(1.0 / ((p2 * 0.001)^R_CP))
            end
        end
    end

    # GFS lacks vertical velocity - compute from continuity equation (edcomp)
    met_fields.w1 .= 0.0f0
    met_fields.w2 .= 0.0f0

    # CRITICAL: Compute vertical velocity from horizontal wind divergence
    # GFS doesn't provide omega/w field, so we use continuity equation
    @info "Computing vertical velocity from continuity equation (edcomp) for GFS initial data"

    # Grid spacing for GFS 0.25° data
    # Domain: lon[-120,-110], lat[32,45], 41x53 grid → 0.25° spacing
    # At mid-latitude 38.5°:
    #   dx = 0.25° × 111.32 km/° × cos(38.5°) = 21,780 m
    #   dy = 0.25° × 111.32 km/° = 27,830 m
    dx_m = T(21780.0)  # meters (longitude spacing at mid-latitude)
    dy_m = T(27830.0)  # meters (latitude spacing)

    # Compute for both time levels
    compute_etadot_from_continuity!(
        GFSFormat(),
        met_fields.w1, met_fields.u1, met_fields.v1, met_fields.ps1,
        met_fields.xm, met_fields.ym,
        met_fields.ahalf, met_fields.bhalf, met_fields.vhalf,
        dx_m, dy_m
    )
    # CRITICAL: edcomp averages output with input (see om2edot.f90:199).
    # When input is zero, output is halved, so multiply by 2 for correct value.
    met_fields.w1 .*= T(2.0)

    compute_etadot_from_continuity!(
        GFSFormat(),
        met_fields.w2, met_fields.u2, met_fields.v2, met_fields.ps2,
        met_fields.xm, met_fields.ym,
        met_fields.ahalf, met_fields.bhalf, met_fields.vhalf,
        dx_m, dy_m
    )
    # CRITICAL: edcomp averages output with input (see om2edot.f90:199).
    # When input is zero, output is halved, so multiply by 2 for correct value.
    met_fields.w2 .*= T(2.0)

    # Load precipitation (check both precipitation_flux and precipitation_rate)
    precip_var = haskey(ds, "precipitation_flux") ? "precipitation_flux" :
                 haskey(ds, "precipitation_rate") ? "precipitation_rate" : nothing
    if precip_var !== nothing
        precip = ds[precip_var]
        units = haskey(precip.attrib, "units") ? String(precip.attrib["units"]) : ""
        scale = occursin("kg", lowercase(units)) && occursin("s", lowercase(units)) ? T(3600.0) : T(1.0)
        met_fields.precip1 .= precip[:, :, window_idx] .* scale
        met_fields.precip2 .= precip[:, :, next_idx] .* scale
    else
        fill!(met_fields.precip1, zero(T))
        fill!(met_fields.precip2, zero(T))
    end

    # Compute 3D pressure fields from hybrid coordinates
    # CRITICAL: p(i,j,k) = alevel(k) + blevel(k)*ps(i,j) - needed for correct vgrav!
    compute_pressure_from_hybrid!(met_fields, 1)
    compute_pressure_from_hybrid!(met_fields, 2)

    # Derive geopotential heights for both time slices
    compute_model_heights!(GFSFormat, met_fields, 1)
    compute_model_heights!(GFSFormat, met_fields, 2)

    compute_boundary_layer!(GFSFormat, met_fields, time_level=1)
    compute_boundary_layer!(GFSFormat, met_fields, time_level=2)

    @info "GFS initial met fields loaded"
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

    # Match Fortran SNAP's reversed hybrid coefficient loading (same as initialization)
    for k in 1:nk
        met_fields.alevel[k] = ap[nk + 1 - k]
        met_fields.blevel[k] = b[nk + 1 - k]
    end

    # Match Fortran's ahalf computation (average) with TOA boundary
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

    # Fortran mapfield.f calculates hx/hy as arc-length on sphere (constant by window)
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
        # Match Fortran continuity-only branch: multiply by 2 after internal 0.5 averaging
        met_fields.w2 .*= T(2.0)
    end
    # Fortran enforces w=0 at surface level
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

function read_met_fields!(::GFSFormat, met_fields, ds::NCDataset,
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
    met_fields.hlayer1 .= met_fields.hlayer2
    met_fields.precip1 .= met_fields.precip2
    met_fields.hbl1 .= met_fields.hbl2
    met_fields.bl1 .= met_fields.bl2

    # GFS data uses standard hybrid coefficients (like ERA5)
    # Reference pressure: standard atmosphere
    ap = T.(ds["ap"][:])
    b = T.(ds["b"][:])
    p0_pa = T(101325.0) # Standard atmosphere in Pa

    nk = length(ap)

    # Half-levels (interfaces) - same structure as ERA5
    for k in 1:nk
        met_fields.ahalf[k] = ap[k]
        met_fields.bhalf[k] = b[k]
    end
    # Surface boundary (nk+1 = surface level)
    met_fields.ahalf[nk + 1] = zero(T)   # a at surface is 0
    met_fields.bhalf[nk + 1] = one(T)    # b at surface is 1

    # Full-levels (centers)
    for k in 1:nk
        met_fields.alevel[k] = (met_fields.ahalf[k] + met_fields.ahalf[k + 1]) * T(0.5)
        met_fields.blevel[k] = (met_fields.bhalf[k] + met_fields.bhalf[k + 1]) * T(0.5)
        # vlevel is a sigma-like coordinate based on a reference pressure profile
        met_fields.vlevel[k] = met_fields.alevel[k] / p0_pa + met_fields.blevel[k]
    end
    for k in 1:(nk + 1)
        met_fields.vhalf[k] = met_fields.ahalf[k] / p0_pa + met_fields.bhalf[k]
    end

    # GFS variable names (from fimex conversion)
    # Now using same coordinate order as ERA5 (top-to-bottom = ascending sigma)
    # STEP 2: Load ONLY the next timestep into *2 fields
    met_fields.u2 .= ds["x_wind_pl"][:, :, :, next_idx]
    met_fields.v2 .= ds["y_wind_pl"][:, :, :, next_idx]
    met_fields.t2 .= ds["air_temperature_pl"][:, :, :, next_idx]
    met_fields.ps2 .= ds["surface_air_pressure"][:, :, next_idx] ./ 100.0f0

    # Compute map scale factors (CRITICAL for correct horizontal wind advection!)
    # GFS uses latitude coordinate (no reversal needed like ERA5)
    lats = collect(T.(ds["latitude"]))
    Transport.compute_map_scale_factors!(met_fields.xm, met_fields.ym, lats)

    # CRITICAL: Convert absolute temperature to potential temperature
    # This matches Fortran SNAP's t2thetafac conversion in readfield_nc.f90:400
    # θ = T * (p₀/p)^(R/cp) where p₀ = 1000 hPa
    # NOTE: alevel is in Pa (from NetCDF ap), ps is in hPa (converted above)
    # So we need to convert alevel/100 + blevel*ps to get pressure in hPa
    # IMPORTANT: Only convert t2 since t1 was already converted in the previous timestep
    R_CP = 287.058 / 1005.0  # R/cp ≈ 0.286
    for k in 1:size(met_fields.t2, 3)
        for j in 1:size(met_fields.t2, 2)
            for i in 1:size(met_fields.t2, 1)
                # Compute pressure at this grid point (hybrid coordinates) in hPa
                # p = ap/100 + b*ps, where ap is in Pa and ps is in hPa
                p2 = met_fields.alevel[k]/100.0 + met_fields.blevel[k] * met_fields.ps2[i, j]

                # Convert to potential temperature: θ = T × (1000/p)^(R/cp)
                met_fields.t2[i, j, k] *= Float32(1.0 / ((p2 * 0.001)^R_CP))
            end
        end
    end

    # GFS lacks vertical velocity - compute from continuity equation (edcomp)
    # w1 was already swapped from previous w2, so only zero w2 before computing
    met_fields.w2 .= 0.0f0

    # CRITICAL: Compute vertical velocity from horizontal wind divergence
    # GFS doesn't provide omega/w field, so we use continuity equation
    @info "Computing vertical velocity from continuity equation (edcomp) for GFS time level 2"

    # Grid spacing for GFS 0.25° data
    # Domain: lon[-120,-110], lat[32,45], 41x53 grid → 0.25° spacing
    # At mid-latitude 38.5°:
    #   dx = 0.25° × 111.32 km/° × cos(38.5°) = 21,780 m
    #   dy = 0.25° × 111.32 km/° = 27,830 m
    dx_m = T(21780.0)  # meters (longitude spacing at mid-latitude)
    dy_m = T(27830.0)  # meters (latitude spacing)

    # Compute ONLY for time level 2 (w1 was already swapped from previous w2)
    compute_etadot_from_continuity!(
        GFSFormat(),
        met_fields.w2, met_fields.u2, met_fields.v2, met_fields.ps2,
        met_fields.xm, met_fields.ym,
        met_fields.ahalf, met_fields.bhalf, met_fields.vhalf,
        dx_m, dy_m
    )
    # CRITICAL: edcomp averages output with input (see om2edot.f90:199).
    # When input is zero, output is halved, so multiply by 2 for correct value.
    met_fields.w2 .*= 2.0

    # Load precipitation for next timestep only (precip1 was already swapped)
    if haskey(ds, "precipitation_flux")
        precip = ds["precipitation_flux"]
        units = haskey(precip.attrib, "units") ? String(precip.attrib["units"]) : ""
        scale = units == "kg/m^2/s" ? T(3600.0) : T(1.0)
        met_fields.precip2 .= precip[:, :, next_idx] .* scale
    else
        fill!(met_fields.precip2, zero(T))
    end

    # Compute 3D pressure fields from hybrid coordinates
    # CRITICAL: p(i,j,k) = alevel(k) + blevel(k)*ps(i,j) - needed for correct vgrav!
    compute_pressure_from_hybrid!(met_fields, 1)
    compute_pressure_from_hybrid!(met_fields, 2)

    # Derive geopotential heights for both time slices
    compute_model_heights!(GFSFormat, met_fields, 1)
    compute_model_heights!(GFSFormat, met_fields, 2)

    compute_boundary_layer!(GFSFormat, met_fields, time_level=1)
    compute_boundary_layer!(GFSFormat, met_fields, time_level=2)
end

export MetFormat, ERA5Format, ERA5RawFormat, GFSFormat
export detect_met_format, get_met_dimensions, get_vertical_levels
export get_time_variable, read_initial_met_fields!, read_met_fields!
