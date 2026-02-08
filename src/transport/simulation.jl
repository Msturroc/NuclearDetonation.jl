# Main Simulation Loop and State Management
#
# Integrates all transport components into a working atmospheric dispersion simulation

using StaticArrays
using Statistics
import Dates

@inline function height_to_sigma(z::Real, z_max::Real)
    if z_max <= 0
        return 0.0
    end
    return clamp(1.0 - Float64(z) / Float64(z_max), 0.0, 1.0)
end

@inline function sigma_to_height(sigma::Real, z_max::Real)
    if z_max <= 0
        return 0.0
    end
    return clamp((1.0 - Float64(sigma)) * Float64(z_max), 0.0, Float64(z_max))
end

@inline function height_to_level(z::Real, hlevel::AbstractVector{<:Real})
    zf = Float64(z)
    h_min = Float64(first(hlevel))
    h_max = Float64(last(hlevel))

    if zf <= h_min
        return 1.0
    elseif zf >= h_max
        return Float64(length(hlevel))
    end

    k = searchsortedlast(hlevel, zf)
    k = max(k, 1)
    if k >= length(hlevel)
        return Float64(length(hlevel))
    end

    h_low = Float64(hlevel[k])
    h_high = Float64(hlevel[k + 1])
    frac = (zf - h_low) / (h_high - h_low)
    return Float64(k) + frac
end

"""
    height_to_sigma_hybrid(x, y, height_m, met_fields, t)

Convert height in meters to sigma/eta coordinate using hybrid coordinate system.

This function inverts the height interpolation from met_fields.hlevel2 to find
the sigma coordinate that corresponds to the given height at position (x, y, t).

# Arguments
- `x, y`: Grid position
- `height_m`: Height in meters above surface
- `met_fields`: MeteoFields with hlevel2 data
- `t`: Time (seconds)

# Returns
- `sigma`: Sigma/eta coordinate (0 = top of atmosphere, 1 = surface)

# Notes
- Uses binary search to invert the height→sigma mapping
- Clamps to valid sigma range [z_grid[1], z_grid[end]]
- Handles surface (height ≈ 0) and top of atmosphere cases
"""
function height_to_sigma_hybrid(x::Real, y::Real, height_m::Real,
                               met_fields, t::Real=0.0)
    # CRITICAL: We must interpolate TEMPERATURE first, then compute heights using the
    # interpolated T profile. Interpolating pre-computed heights gives different results
    # due to nonlinearity of the hypsometric equation.

    # Get grid indices using floor (truncation) to match int(x) behaviour
    i = clamp(floor(Int, x), 1, met_fields.nx - 1)
    j = clamp(floor(Int, y), 1, met_fields.ny - 1)

    # Bilinear interpolation weights
    dxx = x - i
    dyy = y - j
    c1 = (1.0 - dyy) * (1.0 - dxx)  # (i,j)
    c2 = (1.0 - dyy) * dxx           # (i+1,j)
    c3 = dyy * (1.0 - dxx)           # (i,j+1)
    c4 = dyy * dxx                   # (i+1,j+1)

    # Temporal interpolation weights
    # t=0 means use t1 only, t=1 means use t2 only
    # For t between 0 and 1: rt1 = (1-t), rt2 = t
    rt1 = 1.0 - t
    rt2 = t

    # DEBUG: Print temporal weights and height profile
    if false  # debug disabled
        println("\nDEBUG height_to_sigma_hybrid: Converting $(height_m)m to sigma")
        println("  x=$x, y=$y")
        println("  i=$i, j=$j, t=$t")
        println("  rt1=$(rt1), rt2=$(rt2)")
        println("  t=$t → rt1=$rt1, rt2=$rt2")
    end

    # Interpolate TEMPERATURE at release location (spatial + temporal)
    nk = met_fields.nk
    T_interp = zeros(Float64, nk)
    for k in 1:nk
        # Spatial interpolation of t1
        T1_spatial = c1 * met_fields.t1[i, j, k] +
                    c2 * met_fields.t1[i+1, j, k] +
                    c3 * met_fields.t1[i, j+1, k] +
                    c4 * met_fields.t1[i+1, j+1, k]

        # Spatial interpolation of t2
        T2_spatial = c1 * met_fields.t2[i, j, k] +
                    c2 * met_fields.t2[i+1, j, k] +
                    c3 * met_fields.t2[i, j+1, k] +
                    c4 * met_fields.t2[i+1, j+1, k]

        # Temporal interpolation
        T_interp[k] = rt1 * T1_spatial + rt2 * T2_spatial
    end

    # DEBUG: Check what values we're reading from ps1 array
    if false  # debug disabled
        println("  DEBUG bilinear interpolation:")
        println("    ps1[i,j]=$(met_fields.ps1[i,j]), ps1[i+1,j]=$(met_fields.ps1[i+1,j])")
        println("    ps1[i,j+1]=$(met_fields.ps1[i,j+1]), ps1[i+1,j+1]=$(met_fields.ps1[i+1,j+1])")
        println("    c1=$c1, c2=$c2, c3=$c3, c4=$c4")
    end

    # Interpolate surface pressure (spatial + temporal)
    ps1_spatial = c1 * met_fields.ps1[i, j] +
                  c2 * met_fields.ps1[i+1, j] +
                  c3 * met_fields.ps1[i, j+1] +
                  c4 * met_fields.ps1[i+1, j+1]

    ps2_spatial = c1 * met_fields.ps2[i, j] +
                  c2 * met_fields.ps2[i+1, j] +
                  c3 * met_fields.ps2[i, j+1] +
                  c4 * met_fields.ps2[i+1, j+1]

    ps_interp = rt1 * ps1_spatial + rt2 * ps2_spatial

    # DEBUG: Print some key values
    if false  # debug disabled
        println("  ps1_spatial=$ps1_spatial, ps2_spatial=$ps2_spatial")
        println("  ps_interp=$ps_interp")
        println("  T_interp[nk]=$(T_interp[nk]) (surface)")
        println("  T_interp[1]=$(T_interp[1]) (TOA)")
        # Sample a few levels to see temperature profiles
        T1_surf = c1 * met_fields.t1[i, j, nk] + c2 * met_fields.t1[i+1, j, nk] + c3 * met_fields.t1[i, j+1, nk] + c4 * met_fields.t1[i+1, j+1, nk]
        T2_surf = c1 * met_fields.t2[i, j, nk] + c2 * met_fields.t2[i+1, j, nk] + c3 * met_fields.t2[i, j+1, nk] + c4 * met_fields.t2[i+1, j+1, nk]
        println("  T1_spatial[nk]=$T1_surf, T2_spatial[nk]=$T2_surf")
        println("  Check: rt1*T1 + rt2*T2 = $(rt1*T1_surf + rt2*T2_surf) == T_interp[nk]=$(T_interp[nk])")
        println("  ahalf[nk+1]=$(met_fields.ahalf[nk+1]), bhalf[nk+1]=$(met_fields.bhalf[nk+1])")
        println("  alevel[nk]=$(met_fields.alevel[nk]), blevel[nk]=$(met_fields.blevel[nk])")
    end

    # Now compute heights using interpolated T (hypsometric integration)
    g = 9.81
    ginv = 1.0 / g
    p0 = 1000.0
    cp = 1004.0
    r = 287.0
    rcp = r / cp
    exner_func(p) = cp * (p / p0)^rcp
    # NOTE: ahalf/alevel already in hPa - do NOT apply Pa→hPa conversion!

    height_levels = zeros(Float64, nk)
    height_levels[nk] = 0.0  # Surface

    # Surface pihu
    # CRITICAL: After reversal, ahalf[1]=surface (not ahalf[nk+1]!)
    # ahalf already in hPa, ps_interp in hPa - no conversion needed
    p_surf = met_fields.ahalf[1] + met_fields.bhalf[1] * ps_interp
    pihu = exner_func(p_surf)
    hhu = 0.0

    # Integrate upward from surface - MUST loop UPWARD k=2 to nk, not downward!
    for k in 2:nk
        # CRITICAL: ahalf/alevel already in hPa - no double conversion!
        # SPECIAL CASE: At k=nk (TOA), use ahalf[nk+1] which is the actual TOA boundary!
        # (ahalf[nk] is just an average and would give pihu==pih, causing NaN)
        if k == nk
            p_half = met_fields.ahalf[nk + 1] + met_fields.bhalf[nk + 1] * ps_interp
        else
            p_half = met_fields.ahalf[k] + met_fields.bhalf[k] * ps_interp
        end
        pih = exner_func(p_half)

        p_full = met_fields.alevel[k] + met_fields.blevel[k] * ps_interp
        pif = exner_func(p_full)

        # T_interp already contains absolute temperature (ERA5 provides T, not θ)
        h1 = hhu
        h2 = h1 + T_interp[k] * (pihu - pih) * ginv

        # DEBUG: Print details for k=nk (TOA) to find NaN source
        if false && k == nk
            println("  DEBUG at k=nk=$k (TOA):")
            println("    ahalf[k]=$(met_fields.ahalf[k]) hPa, bhalf[k]=$(met_fields.bhalf[k])")
            println("    alevel[k]=$(met_fields.alevel[k]) hPa, blevel[k]=$(met_fields.blevel[k])")
            println("    ps_interp=$ps_interp hPa")
            println("    p_half=$p_half hPa, p_full=$p_full hPa")
            println("    pih=$pih, pif=$pif")
            println("    T_interp[k]=$(T_interp[k]) K")
            println("    pihu=$pihu (Exner at lower interface)")
            println("    pihu - pih = $(pihu - pih)")
            println("    pihu - pif = $(pihu - pif)")
            println("    h1=$h1 m, h2=$h2 m")
            println("    (h2 - h1) = $(h2 - h1)")
            println("    (pihu - pif) / (pihu - pih) = $((pihu - pif) / (pihu - pih))")
            println("    height_levels[k] will be: $(h1 + (h2 - h1) * (pihu - pif) / (pihu - pih))")
        end

        height_levels[k] = h1 + (h2 - h1) * (pihu - pif) / (pihu - pih)

        hhu = h2
        pihu = pih
    end

    # Height at k=1 never computed, set to surface
    height_levels[1] = 0.0

    # DEBUG: Show height_levels BEFORE reverse
    if false
        println("  DEBUG BEFORE reverse:")
        println("    height_levels[1]=$(height_levels[1])m (should be surface=0)")
        println("    height_levels[2]=$(height_levels[2])m (first level above surface)")
        println("    height_levels[3]=$(height_levels[3])m")
        println("    height_levels[nk-2]=$(height_levels[nk-2])m")
        println("    height_levels[nk-1]=$(height_levels[nk-1])m")
        println("    height_levels[nk]=$(height_levels[nk])m (should be TOA, high altitude)")
    end

    # Now height_levels is ASCENDING: [0 (surface), ..., high (TOA)]
    # But search expects DESCENDING, so reverse it!
    height_levels = reverse(height_levels)
    # After reverse: height_levels[1] = TOA (high), height_levels[nk] = 0 (surface)

    # Also need sigma_levels in same order as height_levels (both descending)
    # After alevel reversal, vlevel[1]=surface, vlevel[nk]=TOA (ascending from surface to TOA)
    # Reverse to get: sigma_levels[1]=TOA (small), sigma_levels[nk]=surface (large)
    sigma_levels = reverse(Float64.(met_fields.vlevel))

    # DEBUG: Disabled for performance
    # if height_m == 91.0
    #     println("  DEBUG AFTER reverse: ...")
    # end

    # Now both arrays have same ordering:
    # height_levels: [highest (TOA), ..., 0 (surface)] - DESCENDING
    # sigma_levels:  [~0 (TOA), ..., ~1 (surface)] - ASCENDING

    # DEBUG: Disabled for performance
    # if height_m == 91.0
    #     println("  Searching for height_m=$height_m ...")
    # end

    nk = length(height_levels)

    # Guard against empty profiles
    nk == 0 && return Float64(height_m <= 0 ? 1.0 : 0.0)

    if nk == 1
        return sigma_levels[1]
    end

    # CRITICAL FIX: Search in DESCENDING height array
    # height_levels is descending: [high (TOA), ..., low (surface)]
    # Find first index where height_levels[idx] <= height_m
    idx_upper = findfirst(h -> h <= height_m, height_levels)

    if isnothing(idx_upper)  # height_m is below all levels (below surface)
        idx_upper = nk
    end

    idx_upper = max(idx_upper, 2)  # Ensure we have room for idx_lower
    idx_lower = idx_upper - 1

    # In descending height array:
    # idx_lower points to HIGHER height (closer to TOA)
    # idx_upper points to LOWER height (closer to surface)
    h_high = height_levels[idx_lower]  # Higher height
    h_low = height_levels[idx_upper]   # Lower height
    σ_high = sigma_levels[idx_lower]   # Sigma at higher height (smaller sigma, TOA)
    σ_low = sigma_levels[idx_upper]    # Sigma at lower height (larger sigma, surface)

    # Avoid division by ~0 thickness layers
    Δh = h_high - h_low
    if abs(Δh) < 1e-6
        return (σ_low + σ_high) / 2.0
    end

    w = (Float64(height_m) - h_low) / Δh
    σ = σ_low + w * (σ_high - σ_low)

    # DEBUG: Disabled for performance
    # if height_m == 91.0
    #     println("  Found bracket: ...")
    # end

    return Float64(clamp(σ, 0.0, 1.0))
end

"""
    SimulationDomain{T<:Real}

Defines the spatial and temporal domain for the simulation.

# Fields
- `nx, ny, nz::Int`: Grid dimensions
- `dx, dy::T`: Horizontal grid spacing (meters)
- `hlevel::Vector{T}`: Vertical levels (meters MSL)
- `xm, ym::Matrix{T}`: Map scale factors for each grid cell
- `t_start::DateTime`: Simulation start time
- `t_end::DateTime`: Simulation end time
- `dt_output::Duration`: Output interval
- `dt_met::Duration`: Meteorological data time resolution
- `lon_min, lon_max, lat_min, lat_max::T`: Geographic bounds (degrees)
"""
mutable struct SimulationDomain{T<:Real}
    nx::Int
    ny::Int
    nz::Int
    dx::T
    dy::T
    hlevel::Vector{T}
    xm::Matrix{T}
    ym::Matrix{T}
    cell_area::Matrix{T}
    layer_thickness::Vector{T}
    surface_layer_height::T
    t_start::DateTime
    t_end::DateTime
    dt_output::Duration
    dt_met::Duration
    lon_min::T
    lon_max::T
    lat_min::T
    lat_max::T

    function SimulationDomain(nx::Int, ny::Int, nz::Int,
                             dx::T, dy::T, hlevel::Vector{T},
                             xm::Matrix{T}, ym::Matrix{T}, cell_area::Matrix{T},
                             t_start::Union{DateTime,Dates.DateTime},
                             t_end::Union{DateTime,Dates.DateTime},
                             dt_output::Duration, dt_met::Duration;
                             lon_min::T=T(-180), lon_max::T=T(180),
                             lat_min::T=T(-90), lat_max::T=T(90)) where T<:Real
        @assert nx > 0 && ny > 0 && nz > 0
        @assert dx > 0 && dy > 0
        @assert length(hlevel) == nz
        @assert size(xm) == (nx, ny)
        @assert size(ym) == (nx, ny)
        @assert size(cell_area) == (nx, ny)
        @assert t_start < t_end
        layer_thickness = compute_layer_thickness(hlevel)
        surface_layer_height = T(30.0)
        new{T}(nx, ny, nz, dx, dy, hlevel, xm, ym, cell_area,
               layer_thickness, surface_layer_height,
               t_start, t_end, dt_output, dt_met,
               lon_min, lon_max, lat_min, lat_max)
    end
end

function SimulationDomain(nx::Int, ny::Int, nz::Int,
                          dx::T, dy::T, hlevel::Vector{T},
                          xm::Matrix{T}, ym::Matrix{T},
                          t_start::Union{DateTime,Dates.DateTime},
                          t_end::Union{DateTime,Dates.DateTime},
                          dt_output::Duration, dt_met::Duration;
                          lon_min::T=T(-180), lon_max::T=T(180),
                          lat_min::T=T(-90), lat_max::T=T(90)) where T<:Real
    cell_area = fill(dx * dy, nx, ny)
    return SimulationDomain(nx, ny, nz, dx, dy, hlevel, xm, ym, cell_area,
                             t_start, t_end, dt_output, dt_met;
                             lon_min=lon_min, lon_max=lon_max,
                             lat_min=lat_min, lat_max=lat_max)
end

"""
    SimulationDomain(; lon_min, lon_max, lat_min, lat_max, z_min, z_max,
                      nx, ny, nz, start_time, end_time, kwargs...)

Create simulation domain from geographic coordinates (convenience constructor).

# Arguments
- `lon_min, lon_max`: Longitude bounds (degrees East, -180 to 180)
- `lat_min, lat_max`: Latitude bounds (degrees North, -90 to 90)
- `z_min, z_max`: Height bounds (meters MSL)
- `nx, ny, nz`: Grid dimensions
- `start_time, end_time`: DateTime bounds
- `dt_output`: Output interval (default: 1 hour)
- `dt_met`: Met data interval (default: 1 hour for ERA5 hourly data)

# Returns
- `SimulationDomain{Float64}`: Configured domain

# Example
```julia
domain = SimulationDomain(
    lon_min=-120.0, lon_max=-110.0,
    lat_min=34.0, lat_max=42.0,
    z_min=0.0, z_max=15000.0,
    nx=100, ny=80, nz=30,
    start_time=DateTime(1953, 3, 24, 13, 10),
    end_time=DateTime(1953, 3, 26, 13, 10)
)
```
"""
function SimulationDomain(; lon_min::Real, lon_max::Real,
                           lat_min::Real, lat_max::Real,
                           z_min::Real, z_max::Real,
                           nx::Int, ny::Int, nz::Int,
                           start_time::Union{DateTime,Dates.DateTime},
                           end_time::Union{DateTime,Dates.DateTime},
                           dt_output::Union{Duration,Dates.Hour}=Dates.Hour(1),
                           dt_met::Union{Duration,Dates.Hour}=Dates.Hour(1))
    T = Float64

    # Convert to typed values
    lon_min, lon_max = T(lon_min), T(lon_max)
    lat_min, lat_max = T(lat_min), T(lat_max)
    z_min, z_max = T(z_min), T(z_max)

    # Compute grid spacing (approximate, using mid-latitude)
    lat_mid = (lat_min + lat_max) / 2

    # Degrees to meters at mid-latitude
    deg_to_m_lon = 111320.0 * cos(deg2rad(lat_mid))  # meters per degree longitude
    deg_to_m_lat = 110540.0                          # meters per degree latitude

    dx = (lon_max - lon_min) * deg_to_m_lon / nx
    dy = (lat_max - lat_min) * deg_to_m_lat / ny

    # Vertical levels (linearly spaced for now)
    hlevel = range(z_min, z_max, length=nz) |> collect

    # Map scale factors (unity for simple lat/lon projection)
    xm = ones(T, nx, ny)
    ym = ones(T, nx, ny)

    # Compute latitude-dependent cell areas (m²)
    cell_area = zeros(T, nx, ny)
    if nx > 1 && ny > 1
        lon_step = (lon_max - lon_min) / (nx - 1)
        lat_step = (lat_max - lat_min) / (ny - 1)
        lon_step_rad = deg2rad(lon_step)
        R = T(6_371_000.0)
        for j in 1:ny
            lat_center = lat_min + (j - 1) * lat_step
            lat_lower_rad = deg2rad(lat_center - lat_step / 2)
            lat_upper_rad = deg2rad(lat_center + lat_step / 2)
            band_area = R^2 * lon_step_rad * (sin(lat_upper_rad) - sin(lat_lower_rad))
            band_area_T = T(band_area)
            @inbounds cell_area[:, j] .= band_area_T
        end
    else
        fill!(cell_area, dx * dy)
    end

    # Convert Dates types to internal types if needed
    # Note: Internal DateTime only supports hourly granularity (no minutes/seconds)
    if start_time isa Dates.DateTime
        start_time = DateTime(Dates.year(start_time), Dates.month(start_time),
                             Dates.day(start_time), Dates.hour(start_time))
    end
    if end_time isa Dates.DateTime
        end_time = DateTime(Dates.year(end_time), Dates.month(end_time),
                           Dates.day(end_time), Dates.hour(end_time))
    end
    if dt_output isa Dates.Hour
        dt_output = Duration(0, 0, 0, Dates.value(dt_output))
    end
    if dt_met isa Dates.Hour
        dt_met = Duration(0, 0, 0, Dates.value(dt_met))
    end

    return SimulationDomain(nx, ny, nz, dx, dy, hlevel, xm, ym, cell_area,
                            start_time, end_time, dt_output, dt_met,
                            lon_min=lon_min, lon_max=lon_max,
                            lat_min=lat_min, lat_max=lat_max)
end

@inline function compute_layer_thickness(hlevel::AbstractVector{T}) where T
    nz = length(hlevel)
    layer_edges = Vector{T}(undef, nz + 1)
    if nz == 1
        layer_edges[1] = zero(T)
        layer_edges[2] = hlevel[1] > zero(T) ? hlevel[1] : one(T)
    else
        layer_edges[1] = max(zero(T), hlevel[1] - (hlevel[2] - hlevel[1]) / T(2))
        for k in 2:nz
            layer_edges[k] = (hlevel[k - 1] + hlevel[k]) / T(2)
        end
        layer_edges[nz + 1] = hlevel[end] + (hlevel[end] - hlevel[end - 1]) / T(2)
    end
    layer_thickness = Vector{T}(undef, nz)
    for k in 1:nz
        layer_thickness[k] = max(layer_edges[k + 1] - layer_edges[k], T(1.0))
    end
    layer_thickness
end

function update_domain_vertical!(domain::SimulationDomain{T}, met_fields) where {T<:Real}
    nk = domain.nz
    @assert nk == size(met_fields.hlevel1, 3) "Domain vertical levels ($(domain.nz)) must match meteorology ($(size(met_fields.hlevel1, 3)))"

    raw_hlevel = Vector{T}(undef, nk)
    raw_hlayer = Vector{T}(undef, nk)
    for k in 1:nk
        raw_hlevel[k] = T(mean(Float64.(met_fields.hlevel1[:, :, k])))
        raw_hlayer[k] = T(mean(Float64.(met_fields.hlayer1[:, :, k])))
    end

    # CRITICAL FIX (Issue #16): Check if data needs reversal
    # ERA5 data is already reversed (k=1 surface, k=nk TOA) during loading
    # GFS data may be in native order (k=1 TOA, k=nk surface)
    # Only reverse if data is in descending order (TOA→surface)
    needs_reversal = raw_hlevel[1] > raw_hlevel[end]

    if needs_reversal
        domain.hlevel .= reverse(raw_hlevel)
        domain.layer_thickness .= reverse(raw_hlayer)
    else
        domain.hlevel .= raw_hlevel
        domain.layer_thickness .= raw_hlayer
    end

    domain.surface_layer_height = T(30.0)
    return nothing
end

"""
    ParticleEnsemble{T<:Real}

Container for all active particles in the simulation.

# Fields
- `particles::Vector{Particle}`: Active particle states
- `ncomponents::Int`: Number of radionuclide components
- `component_names::Vector{String}`: Names of components (e.g., "Cs137", "I131")
- `positions::Vector{SVector{3,T}}`: (x, y, z) positions in grid coordinates
- `velocities::Vector{SVector{3,T}}`: (u, v, w) velocities (m/s)
- `ages::Vector{T}`: Particle age since release (seconds)
"""
mutable struct ParticleEnsemble{T<:Real}
    particles::Vector{Particle}
    ncomponents::Int
    component_names::Vector{String}
    positions::Vector{SVector{3,T}}
    velocities::Vector{SVector{3,T}}
    ages::Vector{T}
    initial_total_masses::Vector{Float32}  # Track initial mass for 1% removal threshold

    function ParticleEnsemble{T}(ncomponents::Int, component_names::Vector{String}) where T<:Real
        @assert ncomponents > 0
        @assert length(component_names) == ncomponents
        new{T}(Particle[], ncomponents, component_names,
               SVector{3,T}[], SVector{3,T}[], T[], Float32[])
    end
end

"""
    ConcentrationField{T<:Real}

Accumulated concentration and deposition fields for output.

# Fields
- `atm_conc::Array{T,4}`: 3D+time atmospheric concentration (Bq/m³)
- `surf_conc::Array{T,3}`: 2D+time surface concentration (Bq/m³)
- `dry_deposition::Array{T,3}`: Accumulated dry deposition (Bq/m²)
- `wet_deposition::Array{T,3}`: Accumulated wet deposition (Bq/m²)
- `total_deposition::Array{T,3}`: Total deposition (Bq/m²)
- `dose::Array{T,3}`: Time-integrated dose (Bq·s/m³)
"""
mutable struct ConcentrationField{T<:Real}
    atm_conc::Array{T,4}         # (nx, ny, nz, ncomp)
    surf_conc::Array{T,3}        # (nx, ny, ncomp)
    dry_deposition::Array{T,3}   # (nx, ny, ncomp)
    wet_deposition::Array{T,3}   # (nx, ny, ncomp)
    total_deposition::Array{T,3} # (nx, ny, ncomp)
    dose::Array{T,3}             # (nx, ny, ncomp) - integrated air concentration

    function ConcentrationField{T}(nx::Int, ny::Int, nz::Int, ncomp::Int) where T<:Real
        new{T}(
            zeros(T, nx, ny, nz, ncomp),
            zeros(T, nx, ny, ncomp),
            zeros(T, nx, ny, ncomp),
            zeros(T, nx, ny, ncomp),
            zeros(T, nx, ny, ncomp),
            zeros(T, nx, ny, ncomp)
        )
    end
end

"""
    DepositionEvent{T<:Real}

Record of a single deposition event with position and mass.

# Fields
- `x::T`: Grid x position
- `y::T`: Grid y position
- `mass::T`: Deposited mass (Bq)
- `time::T`: Time since start (s)
- `component::Int`: Component index
"""
struct DepositionEvent{T<:Real}
    x::T           # Grid x position
    y::T           # Grid y position
    mass::T        # Deposited mass (Bq)
    time::T        # Time since start (s)
    component::Int # Component index
end

"""
    SimulationState{T<:Real}

Complete state of the transport simulation at a given time.

# Fields
- `domain::SimulationDomain{T}`: Spatial-temporal domain
- `ensemble::ParticleEnsemble{T}`: Particle ensemble
- `fields::ConcentrationField{T}`: Accumulated output fields
- `current_time::DateTime`: Current simulation time
- `timestep::Int`: Current timestep number
- `total_released::Vector{T}`: Total activity released per component (Bq)
- `total_deposited::Vector{T}`: Total activity deposited per component (Bq)
- `deposition_log::Vector{DepositionEvent{T}}`: Log of individual deposition events
- `log_depositions::Bool`: Whether to log depositions
"""
mutable struct SimulationState{T<:Real}
    domain::SimulationDomain{T}
    ensemble::ParticleEnsemble{T}
    fields::ConcentrationField{T}
    current_time::DateTime
    timestep::Int
    total_released::Vector{T}
    total_deposited::Vector{T}
    deposition_log::Vector{DepositionEvent{T}}  # Track individual deposition events
    log_depositions::Bool  # Whether to log individual depositions

    function SimulationState(domain::SimulationDomain{T},
                            ensemble::ParticleEnsemble{T},
                            fields::ConcentrationField{T};
                            log_depositions::Bool=false) where T<:Real
        ncomp = ensemble.ncomponents
        new{T}(domain, ensemble, fields, domain.t_start, 0,
               zeros(T, ncomp), zeros(T, ncomp),
               DepositionEvent{T}[], log_depositions)
    end
end

"""
    add_particle!(ensemble::ParticleEnsemble, position, velocity, mass::Vector, age::Real)

Add a new particle to the ensemble.

# Arguments
- `ensemble`: Particle ensemble
- `position`: (x, y, z) position in grid coordinates
- `velocity`: (u, v, w) velocity (m/s)
- `mass`: Component masses (Bq)
- `age`: Particle age since release (seconds)
"""
function add_particle!(ensemble::ParticleEnsemble{T},
                      position::SVector{3,T},
                      velocity::SVector{3,T},
                      mass::Vector{T},
                      age::T;
                      icomp::Int=0) where T<:Real
    @assert length(mass) == ensemble.ncomponents

    # Create particle with multi-component radioactive content
    # particle.z stores SIGMA coordinates (0-1 range)
    # Caller is responsible for converting height→sigma before calling add_particle!
    rad_content = [Float32(m) for m in mass]
    particle = Particle(position[1], position[2], position[3], 0.0f0, rad_content, 0.0f0, 0.0f0, Int16(icomp))

    push!(ensemble.particles, particle)
    push!(ensemble.positions, position)
    push!(ensemble.velocities, velocity)
    push!(ensemble.ages, age)

    # Track initial total mass for 1% removal threshold
    push!(ensemble.initial_total_masses, Float32(sum(mass)))
end

"""
    remove_inactive_particles!(ensemble::ParticleEnsemble)

Remove particles that have become inactive (deposited or decayed away).

Returns the number of particles removed.
"""
function remove_inactive_particles!(ensemble::ParticleEnsemble{T}; mass_threshold::Float32=0.01f0) where T<:Real
    # Remove particles based on inactivity criteria:
    # 1. Explicitly inactivated (negative radioactivity)
    # 2. Below relative mass threshold (default 1% of initial mass)
    removal_mask = [begin
        total_radioactivity = sum(p.rad)
        initial_mass = ensemble.initial_total_masses[i]

        # Remove if:
        # - Explicitly inactivated (any component negative)
        # - OR below mass threshold (< 1% of initial)
        any(<(0.0f0), p.rad) || (total_radioactivity < mass_threshold * initial_mass)
    end for (i, p) in enumerate(ensemble.particles)]

    n_removed = count(removal_mask)

    if n_removed > 0
        # Keep particles that are NOT in removal_mask
        keep_mask = .!removal_mask
        ensemble.particles = ensemble.particles[keep_mask]
        ensemble.positions = ensemble.positions[keep_mask]
        ensemble.velocities = ensemble.velocities[keep_mask]
        ensemble.ages = ensemble.ages[keep_mask]
        ensemble.initial_total_masses = ensemble.initial_total_masses[keep_mask]
    end

    return n_removed
end

"""
    initialize_simulation(domain::SimulationDomain, sources::Vector{ReleaseSource},
                         component_names::Vector{String}, decay_params::Vector{DecayParams})

Initialise a transport simulation with release sources.

# Arguments
- `domain`: Simulation domain configuration
- `sources`: Vector of release sources
- `component_names`: Names of radionuclide components
- `decay_params`: Decay parameters for each component

# Returns
- `SimulationState`: Initialized simulation state
"""
function initialize_simulation(domain::SimulationDomain{T},
                              sources::Vector{ReleaseSource{T}},
                              component_names::Vector{String},
                              decay_params::Vector{DecayParams{T}};
                              log_depositions::Bool=false) where T<:Real
    ncomp = length(component_names)
    @assert ncomp > 0
    @assert length(decay_params) == ncomp

    # Create ensemble
    ensemble = ParticleEnsemble{T}(ncomp, component_names)

    # Create concentration fields
    fields = ConcentrationField{T}(domain.nx, domain.ny, domain.nz, ncomp)

    # Create state with optional deposition logging
    state = SimulationState(domain, ensemble, fields; log_depositions=log_depositions)

    return state
end

"""
    accumulate_concentration!(fields::ConcentrationField, ensemble::ParticleEnsemble,
                             domain::SimulationDomain, winds::WindFields, dt::Real)

Accumulate particle mass into concentration fields.

Uses kernel density estimation with trilinear interpolation to distribute
particle mass to the surrounding 8 grid cells.

# Arguments
- `fields`: Concentration fields to update
- `ensemble`: Current particle ensemble
- `domain`: Simulation domain
- `dt`: Time step for dose accumulation (seconds)
- `reset_fields`: When `true` (default), zero the instantaneous fields before distributing particle mass.

# Performance
- Uses @views to eliminate array slice copies
"""
@views function accumulate_concentration!(fields::ConcentrationField{T},
                                  ensemble::ParticleEnsemble{T},
                                  domain::SimulationDomain{T},
                                  winds::WindFields,
                                  dt::T;
                                  reset_fields::Bool=true,
                                  use_trilinear::Bool=false) where T<:Real
    if reset_fields
        clear_concentration!(fields)
    end

    # DIAGNOSTIC COUNTERS
    n_total = 0
    n_inactive = 0
    n_out_xy = 0
    n_out_z = 0
    n_accumulated = 0

    for (pidx, particle) in enumerate(ensemble.particles)
        n_total += 1

        if !is_active(particle)
            n_inactive += 1
            continue
        end

        pos = ensemble.positions[pidx]
        x, y = pos[1], pos[2]
        σ = clamp(Float64(pos[3]), 0.0, 1.0)

        # CRITICAL FIX (2025-10-24): Particles store σ in HYBRID PRESSURE COORDINATES!
        # They are initialized using height_to_sigma_hybrid(), so we must convert back
        # using the matching hybrid→height function, not simple terrain-following.
        #
        # Previous bug: Used sigma_to_height(σ, 35000) which treats σ as simple terrain-following.
        # This gave z=27335m for σ=0.219, exceeding grid bounds (~14km), rejecting all particles!
        #
        # Correct: Use hybrid_profile to convert σ→height matching the meteorological grid.
        # Note: x, y are in domain coordinates (from ensemble.positions), but hybrid_profile
        # expects met coordinates. However, if domain and met grids have the same resolution
        # and bounds, this works out correctly.
        profile = hybrid_profile(winds, x, y, T(0.0))
        z_height = height_from_sigma(profile, σ; fallback_height=T(5000.0))

        # Check bounds
        if x < 1 || x > domain.nx || y < 1 || y > domain.ny
            n_out_xy += 1
            continue
        end

        if z_height < domain.hlevel[1] || z_height > domain.hlevel[end]
            n_out_z += 1
            continue
        end

        n_accumulated += 1

        # Grid assignment: trilinear interpolation (smooth, default) or nearest-neighbour
        if use_trilinear
            # TRILINEAR INTERPOLATION: Distribute to 8 cells (more accurate, smoother fields)
            i0 = floor(Int, x)
            j0 = floor(Int, y)

            z_idx = clamp(height_to_level(z_height, domain.hlevel), 1.0, Float64(domain.nz))

            if domain.nz == 1
                k0 = 1
                k1 = 1
                fz = 0.0
            else
                if z_idx >= domain.nz
                    k0 = domain.nz - 1
                    k1 = domain.nz
                    fz = 1.0
                else
                    k0 = max(1, floor(Int, z_idx))
                    k1 = min(k0 + 1, domain.nz)
                    fz = z_idx - k0
                end
            end

            i1 = min(i0 + 1, domain.nx)
            j1 = min(j0 + 1, domain.ny)
            if domain.nz > 1
                k1 = min(k1, domain.nz)
            end

            # Interpolation weights
            fx = x - i0
            fy = y - j0

            # Distribute particle mass to 8 surrounding cells
            weights = [
                (1-fx) * (1-fy) * (1-fz),  # i0, j0, k0
                (1-fx) * (1-fy) * fz,      # i0, j0, k1
                (1-fx) * fy * (1-fz),      # i0, j1, k0
                (1-fx) * fy * fz,          # i0, j1, k1
                fx * (1-fy) * (1-fz),      # i1, j0, k0
                fx * (1-fy) * fz,          # i1, j0, k1
                fx * fy * (1-fz),          # i1, j1, k0
                fx * fy * fz               # i1, j1, k1
            ]

            cells = [
                (i0, j0, k0), (i0, j0, k1), (i0, j1, k0), (i0, j1, k1),
                (i1, j0, k0), (i1, j0, k1), (i1, j1, k0), (i1, j1, k1)
            ]
        else
            # NEAREST-NEIGHBOUR: Assign to 1 cell
            i0 = round(Int, x)
            j0 = round(Int, y)

            z_idx = clamp(height_to_level(z_height, domain.hlevel), 1.0, Float64(domain.nz))
            k0 = round(Int, z_idx)

            # Clamp to valid range
            i0 = clamp(i0, 1, domain.nx)
            j0 = clamp(j0, 1, domain.ny)
            k0 = clamp(k0, 1, domain.nz)

            # Single cell, full weight
            weights = [1.0]
            cells = [(i0, j0, k0)]
        end

        surface_height = domain.surface_layer_height
        for comp in 1:ensemble.ncomponents
            mass = get_rad(particle, comp)

            if mass > 0
                for (w, (i, j, k)) in zip(weights, cells)
                    if i >= 1 && i <= domain.nx && j >= 1 && j <= domain.ny && k >= 1 && k <= domain.nz
                        area = domain.cell_area[i, j]
                        layer_dz = domain.layer_thickness[k]
                        layer_dz = layer_dz <= 0 ? one(T) : layer_dz
                        mass_contrib = mass * w
                        conc = mass_contrib / (area * layer_dz)  # Bq/m³
                        fields.atm_conc[i, j, k, comp] += conc
                        fields.dose[i, j, comp] += conc * dt  # Time-integrated atmosphere concentration

                        if k == 1 && z_height <= surface_height
                            fields.surf_conc[i, j, comp] += conc
                        end
                    end
                end
            end
        end
    end
end

"""
    clear_concentration!(fields::ConcentrationField)

Clear instantaneous concentration fields (not deposition or dose).
"""
function clear_concentration!(fields::ConcentrationField{T}) where T<:Real
    fill!(fields.atm_conc, zero(T))
    fill!(fields.surf_conc, zero(T))
end

"""
    print_simulation_status(state::SimulationState)

Print current simulation status to console.
"""
function print_simulation_status(state::SimulationState{T}) where T<:Real
    nactive = count(is_active(p) for p in state.ensemble.particles)
    total_mass = sum(state.total_released)
    deposited_mass = sum(state.total_deposited)

    println("="^70)
    println("Timestep: $(state.timestep)")
    println("Time: $(state.current_time)")
    println("Active particles: $nactive / $(length(state.ensemble.particles))")
    println("Total released: $(total_mass) Bq")
    println("Total deposited: $(deposited_mass) Bq ($(100*deposited_mass/total_mass)%)")
    println("="^70)
end

"""
    grid_cell_area(domain::SimulationDomain{T}, i::Int=-1, j::Int=-1) where T

Compute area of grid cell(s) on spherical Earth.

# Arguments
- `domain`: Simulation domain
- `i, j`: Grid cell indices (optional). If not provided, returns mean cell area.

# Returns
- `T`: Cell area in m²

# Notes
Uses spherical Earth approximation:
```
A = R² × Δλ × |sin(φ₂) - sin(φ₁)|
```
where R = 6371 km, Δλ is longitude spacing (radians), φ is latitude (radians).
"""
function grid_cell_area(domain::SimulationDomain{T}, i::Int=-1, j::Int=-1) where T
    R_earth = T(6371000.0)  # Earth radius (meters)

    # Compute lat/lon of cell corners
    if i > 0 && j > 0
        # Specific cell
        @assert 1 <= i <= domain.nx && 1 <= j <= domain.ny

        dlon = (domain.lon_max - domain.lon_min) / domain.nx
        dlat = (domain.lat_max - domain.lat_min) / domain.ny

        lon1 = domain.lon_min + (i - 1) * dlon
        lon2 = lon1 + dlon
        lat1 = domain.lat_min + (j - 1) * dlat
        lat2 = lat1 + dlat
    else
        # Mean cell area
        dlon = (domain.lon_max - domain.lon_min) / domain.nx
        dlat = (domain.lat_max - domain.lat_min) / domain.ny

        lon1 = domain.lon_min
        lon2 = domain.lon_min + dlon
        lat1 = (domain.lat_min + domain.lat_max) / 2 - dlat / 2
        lat2 = lat1 + dlat
    end

    # Convert to radians
    dλ = deg2rad(lon2 - lon1)
    φ1 = deg2rad(lat1)
    φ2 = deg2rad(lat2)

    # Spherical Earth area formula
    area = R_earth^2 * dλ * abs(sin(φ2) - sin(φ1))

    return area
end

"""
    latlon_to_grid(domain::SimulationDomain{T}, lat::Real, lon::Real) where T

Convert geographic coordinates to grid coordinates.

# Arguments
- `domain`: Simulation domain
- `lat`: Latitude (degrees North)
- `lon`: Longitude (degrees East)

# Returns
- `(x, y)`: Grid coordinates (fractional, 1-based)

# Example
```julia
x, y = latlon_to_grid(domain, 37.0956, -116.1028)
```
"""
function latlon_to_grid(domain::SimulationDomain{T}, lat::Real, lon::Real) where T
    # Fractional grid coordinates
    # CRITICAL FIX (2025-10-25): Must use (nx-1) and (ny-1) for grid spacing!
    # Grid cells span from 1 to nx, so there are (nx-1) intervals between them.
    # Previous bug: multiplied by nx instead of (nx-1), causing 0.4-cell offset.

    # Handle longitude convention mismatch (2026-01-28):
    # Domain may use 0-360° but input may be -180 to 180° (or vice versa)
    lon_adj = lon
    if domain.lon_min >= 0 && lon < 0
        # Domain uses 0-360°, input uses -180 to 180°
        lon_adj = lon + 360.0
    elseif domain.lon_max <= 180 && lon > 180
        # Domain uses -180 to 180°, input uses 0-360°
        lon_adj = lon - 360.0
    end

    x = 1 + (lon_adj - domain.lon_min) / (domain.lon_max - domain.lon_min) * (domain.nx - 1)
    y = 1 + (lat - domain.lat_min) / (domain.lat_max - domain.lat_min) * (domain.ny - 1)

    return (T(x), T(y))
end

"""
    grid_to_latlon(domain::SimulationDomain{T}, x::Real, y::Real) where T

Convert grid coordinates to geographic coordinates.

# Arguments
- `domain`: Simulation domain
- `x, y`: Grid coordinates (1-based, can be fractional)

# Returns
- `(lat, lon)`: Geographic coordinates (degrees)

# Example
```julia
lat, lon = grid_to_latlon(domain, 50.5, 40.3)
```
"""
function grid_to_latlon(domain::SimulationDomain{T}, x::Real, y::Real) where T
    # CRITICAL FIX (2025-12-12): Must use (nx-1) and (ny-1) to match latlon_to_grid!
    # Previous bug: used nx/ny instead of (nx-1)/(ny-1), causing ~0.8% coordinate error
    # that accumulated to ~0.5° divergence over 12 hours.
    lon = domain.lon_min + (x - 1) / (domain.nx - 1) * (domain.lon_max - domain.lon_min)
    lat = domain.lat_min + (y - 1) / (domain.ny - 1) * (domain.lat_max - domain.lat_min)

    return (T(lat), T(lon))
end

# Export public API
export SimulationDomain, ParticleEnsemble, ConcentrationField, SimulationState, DepositionEvent
export add_particle!, remove_inactive_particles!
export height_to_sigma_hybrid
export initialize_simulation
export accumulate_concentration!, clear_concentration!
export print_simulation_status
export grid_cell_area, latlon_to_grid, grid_to_latlon
