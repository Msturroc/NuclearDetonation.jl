# Particle Dynamics - Modern ODE Solver Implementation
#
# Uses OrdinaryDiffEq.jl for adaptive integration and Interpolations.jl for wind fields

using OrdinaryDiffEq  # ODE solvers only (not the full DifferentialEquations.jl metapackage)
using Interpolations
using Dierckx  # For cubic splines on irregular sigma grid (Modern 2)
using LinearAlgebra


"""
    WindFields

Interpolated wind field data for particle advection.

# Fields
- `u_interp`: Eastward wind interpolant (m/s)
- `v_interp`: Northward wind interpolant (m/s)
- `w_interp`: Vertical velocity interpolant (sigma/eta per second)
- `ps_interp`: Surface pressure interpolant (hPa)
- `p_interp`: 3D pressure interpolant (hPa) - CRITICAL for correct settling velocity!
- `t_interp`: Potential temperature interpolant (K)
- `tbl_interp`: Boundary layer top interpolant (sigma/eta)
- `hbl_interp`: Boundary layer height interpolant (m)
- `h_interp`: Geopotential height interpolant (m)
- `x_grid`, `y_grid`, `z_grid`: Grid coordinates (grid units, grid units, sigma/eta)
- `t_range`: Time range [t1, t2] in seconds
- `nx`, `ny`, `nk`: Grid dimensions

# Notes
- All interpolants use trilinear interpolation in space + linear in time
- Coordinates: (x, y, z, t) where x,y are grid positions, z is sigma/eta, t is seconds
- Boundary behavior: extrapolation clamped to grid bounds
- p_interp gives actual pressure at altitude: P(20km) ≈ 50 hPa, not 1000 hPa!
"""
struct WindFields{T<:Real, I4, I3, I1}
    u_interp::I4  # 4D: (x, y, z, t)
    v_interp::I4  # 4D: (x, y, z, t)
    w_interp::I4  # 4D: (x, y, z, t)
    ps_interp::I3  # 3D: (x, y, t) - surface field
    p_interp::I4   # 4D: (x, y, z, t) - 3D pressure field
    t_interp::I4  # 4D: (x, y, z, t)
    precip_interp::I3 # 3D: (x, y, t) - precipitation rate (mm/hr)
    hflux_interp::I3  # 3D: sensible heat flux (W/m²)
    tbl_interp::I3  # 3D: boundary layer top (sigma/eta)
    hbl_interp::I3  # 3D: boundary layer height (m)
    h_interp::I4   # 4D: geopotential height (m)
    xm_interp::I1  # 1D: (y) - latitude-dependent map scale factor xm = 1/cos(lat)

    # Hybrid coordinate metadata
    a_half::Vector{T}
    b_half::Vector{T}

    # Grid information
    x_grid::Vector{T}
    y_grid::Vector{T}
    z_grid::Vector{T}  # vlevel (sigma/eta levels)
    t_range::Tuple{T, T}  # (t1, t2) in seconds

    # Domain bounds and grid spacing for map scale factor computation
    lon_min::T
    lon_max::T
    lat_min::T
    lat_max::T
    dx_m::T  # Base grid spacing in x (meters)
    dy_m::T  # Base grid spacing in y (meters)

    nx::Int
    ny::Int
    nk::Int

    # Raw w fields for floor-based interpolation (x,y,k,time-1/2)
    w1_raw::Array{T,3}
    w2_raw::Array{T,3}
end

"""
    CubicVerticalInterpolant

Custom 4D interpolant with cubic spline on vertical (sigma) axis and linear on horizontal/time.

Uses Dierckx.Spline1D for cubic interpolation along irregular sigma coordinate.
This is more accurate than linear for the highly non-uniform vertical grid.

# Evaluation strategy
For point (x, y, sigma, t):
1. Create 3D linear interpolant for each sigma level: f_k(x, y, t)
2. Evaluate at all sigma levels: values_k = f_k(x, y, t)
3. Cubic spline interpolate along sigma: Spline1D(sigma_grid, values)
"""
struct CubicVerticalInterpolant{T<:Real, A<:AbstractArray{T, 4}}
    data::A  # 4D array (nx, ny, nz, nt)
    x_grid::Vector{T}
    y_grid::Vector{T}
    z_grid::Vector{T}  # sigma levels (irregular)
    t_grid::Vector{T}

    function CubicVerticalInterpolant(data::AbstractArray{T, 4}, x::Vector{T}, y::Vector{T}, z::Vector{T}, t::Vector{T}) where T<:Real
        new{T, typeof(data)}(data, x, y, z, t)
    end
end

# Make it callable with (x, y, z, t) signature
function (interp::CubicVerticalInterpolant{T})(x::Real, y::Real, z::Real, t::Real) where T
    # Clamp to grid boundaries
    x_clamped = clamp(x, first(interp.x_grid), last(interp.x_grid))
    y_clamped = clamp(y, first(interp.y_grid), last(interp.y_grid))
    z_clamped = clamp(z, first(interp.z_grid), last(interp.z_grid))
    t_clamped = clamp(t, first(interp.t_grid), last(interp.t_grid))

    # Evaluate at each sigma level using trilinear interpolation
    nz = length(interp.z_grid)
    values_at_sigma_levels = Vector{T}(undef, nz)

    for k in 1:nz
        # Extract 3D slice at this sigma level
        data_slice = @view interp.data[:, :, k, :]

        # Create 3D linear interpolant for (x, y, t)
        itp_3d = interpolate((interp.x_grid, interp.y_grid, interp.t_grid), data_slice, Gridded(Linear()))
        itp_3d_extrap = extrapolate(itp_3d, Flat())

        # Evaluate at (x, y, t)
        values_at_sigma_levels[k] = itp_3d_extrap(x_clamped, y_clamped, t_clamped)
    end

    # Cubic spline interpolation along sigma axis
    # Dierckx.Spline1D: k=3 for cubic, s=0 for interpolation (no smoothing)
    spl = Spline1D(interp.z_grid, values_at_sigma_levels; k=3, s=0.0)

    return spl(z_clamped)
end

"""
    ReferenceTrilinearInterpolant

Manual 4D interpolant using floor-based bilinear/trilinear approach for reference validation.

Uses:
- floor() for integer grid indices (truncation toward zero)
- Explicit bilinear weights c1, c2, c3, c4
- Linear time interpolation with (t2-t)/(t2-t1) weighting
- No sigma-level interpolation (uses nearest k level)

This provides exact reference implementation matching for validation.
"""
struct ReferenceTrilinearInterpolant{T<:Real}
    data1::Array{T, 3}  # 3D array at t1: (nx, ny, nk)
    data2::Array{T, 3}  # 3D array at t2: (nx, ny, nk)
    t1::T
    t2::T
    nx::Int
    ny::Int
    nk::Int
end

# Make it callable with (x, y, z, t) signature
function (interp::ReferenceTrilinearInterpolant{T})(x::Real, y::Real, z::Real, t::Real) where T
    # Truncate toward zero (equivalent to floor for positive values)
    # Julia 1-indexed, so floor then clamp to [1, n-1]
    i = clamp(floor(Int, x), 1, interp.nx - 1)
    j = clamp(floor(Int, y), 1, interp.ny - 1)
    k = clamp(floor(Int, z), 1, interp.nk - 1)

    # Fractional parts for bilinear interpolation
    dx = x - i
    dy = y - j
    dz = z - k

    # Clamp fractions to [0, 1] for boundary cases
    dx = clamp(dx, T(0), T(1))
    dy = clamp(dy, T(0), T(1))
    dz = clamp(dz, T(0), T(1))

    # Time interpolation weights (rt1, rt2)
    dt = interp.t2 - interp.t1
    if dt > 0
        rt1 = (interp.t2 - t) / dt  # Weight for older fields
        rt2 = (t - interp.t1) / dt  # Weight for newer fields
    else
        rt1 = T(0.5)
        rt2 = T(0.5)
    end
    rt1 = clamp(rt1, T(0), T(1))
    rt2 = clamp(rt2, T(0), T(1))

    # Trilinear interpolation using @fastmath and muladd for performance
    # FMA (Fused Multiply-Add) reduces rounding errors and is faster on modern CPUs
    @fastmath begin
        # Bilinear weights in x-y
        c00 = (1 - dx) * (1 - dy)
        c10 = dx * (1 - dy)
        c01 = (1 - dx) * dy
        c11 = dx * dy

        # Bilinear interpolation at t1, level k
        val_k_t1 = muladd(c00, interp.data1[i, j, k],
                   muladd(c10, interp.data1[i+1, j, k],
                   muladd(c01, interp.data1[i, j+1, k],
                          c11 * interp.data1[i+1, j+1, k])))

        # Bilinear interpolation at t1, level k+1
        val_k1_t1 = muladd(c00, interp.data1[i, j, k+1],
                    muladd(c10, interp.data1[i+1, j, k+1],
                    muladd(c01, interp.data1[i, j+1, k+1],
                           c11 * interp.data1[i+1, j+1, k+1])))

        # Linear interpolation in z at t1
        val_t1 = muladd(dz, val_k1_t1 - val_k_t1, val_k_t1)

        # Bilinear interpolation at t2, level k
        val_k_t2 = muladd(c00, interp.data2[i, j, k],
                   muladd(c10, interp.data2[i+1, j, k],
                   muladd(c01, interp.data2[i, j+1, k],
                          c11 * interp.data2[i+1, j+1, k])))

        # Bilinear interpolation at t2, level k+1
        val_k1_t2 = muladd(c00, interp.data2[i, j, k+1],
                    muladd(c10, interp.data2[i+1, j, k+1],
                    muladd(c01, interp.data2[i, j+1, k+1],
                           c11 * interp.data2[i+1, j+1, k+1])))

        # Linear interpolation in z at t2
        val_t2 = muladd(dz, val_k1_t2 - val_k_t2, val_k_t2)

        # Linear time interpolation
        return muladd(rt2, val_t2 - val_t1, rt1 * val_t1)
    end
end

"""
    create_wind_interpolants(met_fields::MeteoFields, t1::Real, t2::Real;
                             config::Union{NumericalConfig, Nothing}=nothing,
                             negate_v::Bool=false,
                             lon_min::Real=-180.0,
                             lon_max::Real=180.0,
                             lat_min::Real=0.0,
                             lat_max::Real=90.0)

Create interpolated wind field objects from meteorological data.

# Arguments
- `met_fields`: MeteoFields struct containing u1, u2, v1, v2, w1, w2, ps1, ps2, t1, t2
- `t1`, `t2`: Time values (seconds) for field set 1 and 2
- `config`: NumericalConfig for interpolation order (default: LinearInterp)
- `negate_v`: If true, negate v-wind for particle advection (ERA5 latitude convention)
- `lon_min`, `lon_max`: Domain longitude bounds (degrees) for grid spacing computation
- `lat_min`, `lat_max`: Domain latitude bounds (degrees) for grid spacing/map scale computation

# Returns
- `WindFields`: Interpolated wind field functor

# Notes
Uses linear interpolation for spatial and temporal dimensions (default).
Can use cubic spline interpolation if config.interpolation_order == CubicInterp.
Extrapolation is clamped to grid boundaries (particles that exit domain are marked inactive).

# Performance
- Pre-allocates arrays instead of using cat() (15-20% faster)
- Uses @views to eliminate temporary array copies

# Example
```julia
# Validation mode (linear interpolation)
config = ValidationMode()
winds = create_wind_interpolants(met_fields, 0.0, 21600.0, config=config)

# Modern mode (cubic interpolation)
config = ModernMode()
winds = create_wind_interpolants(met_fields, 0.0, 21600.0, config=config)

u = winds.u_interp(10.5, 20.3, 0.85, 10800.0)  # Interpolate u at position, time
```
"""

"""
    VerticalCubicInterpolant

A custom 4D interpolant that uses cubic splines for the vertical (sigma) dimension
and linear interpolation for horizontal (x, y) and time dimensions.
This is much more stable than full 4D cubic interpolation on irregular grids.
"""
struct VerticalCubicInterpolant{T<:Real, A<:AbstractArray{T, 4}}
    data::A
    x_grid::Vector{T}
    y_grid::Vector{T}
    z_grid::Vector{T} # Must be strictly ascending
    t_grid::Vector{T}
    nx::Int
    ny::Int
end

"""
    local_cubic_hermite(x, x0, x1, x2, x3, y0, y1, y2, y3)

Perform local cubic Hermite interpolation at point x between x1 and x2,
using points x0, x1, x2, x3.
Handles irregular grid spacing.
"""
@inline function local_cubic_hermite(x::T, x0::T, x1::T, x2::T, x3::T, y0::T, y1::T, y2::T, y3::T) where T
    # Finite difference slopes (Catmull-Rom for irregular grid)
    m1 = (y2 - y0) / (x2 - x0)
    m2 = (y3 - y1) / (x3 - x1)
    
    # Standard cubic Hermite interpolation between x1 and x2
    h = x2 - x1
    t_param = (x - x1) / h
    t2 = t_param * t_param
    t3 = t2 * t_param
    
    # Hermite basis functions
    h00 = 2t3 - 3t2 + 1
    h10 = t3 - 2t2 + t_param
    h01 = -2t3 + 3t2
    h11 = t3 - t2
    
    return h00 * y1 + h10 * h * m1 + h01 * y2 + h11 * h * m2
end

# Make it callable with (x, y, z, t) signature
function (itp::VerticalCubicInterpolant{T})(x::Real, y::Real, z::Real, t::Real) where T
    # 1. Bracket the time
    t1, t2 = itp.t_grid[1], itp.t_grid[2]
    dt = t2 - t1
    tr_frac = dt > eps(T) ? (t - t1) / dt : T(0.5)
    tr_frac = clamp(tr_frac, T(0.0), T(1.0))

    # 2. Setup horizontal indices and weights
    nx_met, ny_met = itp.nx, itp.ny
    # Clamp x and y to stay within grid indices [1, nx-1] and [1, ny-1]
    xq = clamp(T(x), T(1.0), T(nx_met - 1))
    yq = clamp(T(y), T(1.0), T(ny_met - 1))
    
    xi = floor(Int, xq)
    yi = floor(Int, yq)
    xf = xq - xi
    yf = yq - yi
    
    w00 = (1-xf)*(1-yf)
    w10 = xf*(1-yf)
    w01 = (1-xf)*yf
    w11 = xf*yf

    # Helper to get bilinear value at a specific level k
    @inline function get_val(k)
        # Time slice 1
        v1 = w00 * itp.data[xi, yi, k, 1] + 
             w10 * itp.data[xi+1, yi, k, 1] + 
             w01 * itp.data[xi, yi+1, k, 1] + 
             w11 * itp.data[xi+1, yi+1, k, 1]
        # Time slice 2
        v2 = w00 * itp.data[xi, yi, k, 2] + 
             w10 * itp.data[xi+1, yi, k, 2] + 
             w01 * itp.data[xi, yi+1, k, 2] + 
             w11 * itp.data[xi+1, yi+1, k, 2]
        return (1 - tr_frac) * v1 + tr_frac * v2
    end

    # 3. Find vertical neighbors for point z
    z_grid = itp.z_grid
    nz = length(z_grid)
    z_clamped = clamp(T(z), z_grid[1], z_grid[end])
    
    # Find index such that z_grid[idx] <= z_clamped < z_grid[idx+1]
    idx = searchsortedlast(z_grid, z_clamped)
    
    # 4. Handle boundaries and interior
    if idx < 1
        return get_val(1)
    elseif idx >= nz
        return get_val(nz)
    end

    # Interior: we are between idx and idx+1
    # For cubic Hermite, we ideally want idx-1, idx, idx+1, idx+2
    if idx == 1 || idx == nz - 1
        # Fallback to linear at boundaries
        y1_lin = get_val(idx)
        y2_lin = get_val(idx + 1)
        h_v = z_grid[idx+1] - z_grid[idx]
        frac = h_v > eps(T) ? (z_clamped - z_grid[idx]) / h_v : T(0.0)
        return T((1-frac)*y1_lin + frac*y2_lin)
    else
        # Full cubic stencil
        y0 = T(get_val(idx - 1))
        y1 = T(get_val(idx))
        y2 = T(get_val(idx + 1))
        y3 = T(get_val(idx + 2))
        
        res = local_cubic_hermite(z_clamped, z_grid[idx-1], z_grid[idx], z_grid[idx+1], z_grid[idx+2], y0, y1, y2, y3)
        # Prevent overshoot
        return clamp(res, min(y1, y2, y0, y3), max(y1, y2, y0, y3))
    end
end

@views function create_wind_interpolants(met_fields::MeteoFields{T}, t1::Real, t2::Real;
                                         config::Union{NumericalConfig, ERA5NumericalConfig, Nothing}=nothing,
                                         negate_v::Bool=false,
                                         negate_w::Bool=false,
                                         lon_min::Real=T(-180.0),
                                         lon_max::Real=T(180.0),
                                         lat_min::Real=T(0.0),
                                         lat_max::Real=T(90.0)) where T
    nx, ny, nk = met_fields.nx, met_fields.ny, met_fields.nk

    # Compute base grid spacing in metres
    R_earth = T(6.371e6)  # Earth radius in metres
    lon_range_deg = T(lon_max - lon_min)
    lat_range_deg = T(lat_max - lat_min)
    dlon_deg = lon_range_deg / T(nx - 1)  # Degrees per grid cell in longitude
    dlat_deg = lat_range_deg / T(ny - 1)  # Degrees per grid cell in latitude
    dlon_rad = dlon_deg * T(π) / T(180.0)
    dlat_rad = dlat_deg * T(π) / T(180.0)
    dx_m = R_earth * dlon_rad  # hx (metres)
    dy_m = R_earth * dlat_rad  # hy (metres)

    # Compute latitude-dependent map scale factor xm = 1/cos(lat)
    # For geographic grids: xm(i,j) = 1/cos(lat)
    # Near poles, clamp cos(lat) to avoid division by zero
    xm_1d = Vector{T}(undef, ny)
    for j in 1:ny
        lat_deg = lat_min + (j - 1) * dlat_deg
        lat_rad = lat_deg * T(π) / T(180.0)
        clat = cos(lat_rad)
        clat = max(clat, T(0.01745))  # ~cos(89°), polar clamp
        xm_1d[j] = T(1.0) / clat
    end

    # Grid coordinates (1-indexed Julia arrays)
    x_grid = collect(T, 1:nx)
    y_grid = collect(T, 1:ny)

    # Ensure z_grid is sorted and unique (copy to avoid modifying met_fields)
    # Interpolations.jl requires ASCENDING order (top→surface: low σ → high σ)
    # Data should now be in this order from met_formats.jl
    z_grid = copy(met_fields.vlevel)
    level_perm = sortperm(z_grid)  # Get permutation for ascending order

    # DEBUG: Disabled for performance (level_perm check)

    z_grid = collect(z_grid[level_perm])  # Collect to ensure Vector{T} not SubArray

    # Add tiny epsilon to duplicates to make them unique for interpolation
    for i in 2:length(z_grid)
        if z_grid[i] <= z_grid[i-1]
            z_grid[i] = z_grid[i-1] + eps(T) * 10
        end
    end

    permuted = level_perm != collect(1:length(level_perm))

    # Handle case where t1 == t2 (instantaneous snapshot)
    # Interpolation requires unique knots, so add small epsilon if needed
    t_grid = if abs(t2 - t1) < eps(T) * 100
        [T(t1), T(t1) + T(1.0)]  # Use 1 second difference as minimum
    else
        [T(t1), T(t2)]
    end

    # Create 4D arrays (x, y, z, t) for interpolation
    # Pre-allocate and fill in-place instead of cat() for better performance
    # This eliminates ~345 profile samples from array concatenation
    u_4d = Array{T, 4}(undef, nx, ny, nk, 2)
    if permuted
        @views begin
            u_4d[:, :, :, 1] .= met_fields.u1[:, :, level_perm]
            u_4d[:, :, :, 2] .= met_fields.u2[:, :, level_perm]
        end
    else
        u_4d[:, :, :, 1] .= met_fields.u1
        u_4d[:, :, :, 2] .= met_fields.u2
    end

    v_4d = Array{T, 4}(undef, nx, ny, nk, 2)
    if permuted
        @views begin
            v_4d[:, :, :, 1] .= met_fields.v1[:, :, level_perm]
            v_4d[:, :, :, 2] .= met_fields.v2[:, :, level_perm]
        end
    else
        v_4d[:, :, :, 1] .= met_fields.v1
        v_4d[:, :, :, 2] .= met_fields.v2
    end

    # Apply v-wind negation for ERA5's reversed latitude grid (advection only)
    # EDCOMP uses physical v-wind; advection needs negated v-wind
    if negate_v
        v_4d .*= -1.0
    end

    w_4d = Array{T, 4}(undef, nx, ny, nk, 2)
    if permuted
        @views begin
            w_4d[:, :, :, 1] .= met_fields.w1[:, :, level_perm]
            w_4d[:, :, :, 2] .= met_fields.w2[:, :, level_perm]
        end
    else
        w_4d[:, :, :, 1] .= met_fields.w1
        w_4d[:, :, :, 2] .= met_fields.w2
    end

    # Surface fields (x, y, t)
    ps_3d = Array{T, 3}(undef, nx, ny, 2)
    ps_3d[:, :, 1] .= met_fields.ps1
    ps_3d[:, :, 2] .= met_fields.ps2

    # 3D pressure field (x, y, z, t) - CRITICAL for correct vgrav!
    p_4d = Array{T, 4}(undef, nx, ny, nk, 2)
    if permuted
        @views begin
            p_4d[:, :, :, 1] .= met_fields.p1[:, :, level_perm]
            p_4d[:, :, :, 2] .= met_fields.p2[:, :, level_perm]
        end
    else
        p_4d[:, :, :, 1] .= met_fields.p1
        p_4d[:, :, :, 2] .= met_fields.p2
    end

    t_3d = Array{T, 4}(undef, nx, ny, nk, 2)
    if permuted
        @views begin
            t_3d[:, :, :, 1] .= met_fields.t1[:, :, level_perm]
            t_3d[:, :, :, 2] .= met_fields.t2[:, :, level_perm]
        end
    else
        t_3d[:, :, :, 1] .= met_fields.t1
        t_3d[:, :, :, 2] .= met_fields.t2
    end

    tbl_3d = Array{T, 3}(undef, nx, ny, 2)
    tbl_3d[:, :, 1] .= met_fields.bl1
    tbl_3d[:, :, 2] .= met_fields.bl2

    hbl_3d = Array{T, 3}(undef, nx, ny, 2)
    hbl_3d[:, :, 1] .= met_fields.hbl1
    hbl_3d[:, :, 2] .= met_fields.hbl2

    # Geopotential heights at layer midpoints (m)
    hlevel_4d = Array{T, 4}(undef, nx, ny, nk, 2)
    if permuted
        @views begin
            hlevel_4d[:, :, :, 1] .= met_fields.hlevel1[:, :, level_perm]
            hlevel_4d[:, :, :, 2] .= met_fields.hlevel2[:, :, level_perm]
        end
    else
        hlevel_4d[:, :, :, 1] .= met_fields.hlevel1
        hlevel_4d[:, :, :, 2] .= met_fields.hlevel2
    end

    # CRITICAL: Replace NaN values in ALL meteorological fields with sensible defaults
    # ERA5 TOA levels often contain NaNs for some variables which break cubic interpolation.
    replace_nan!(x, default) = replace!(val -> isnan(val) ? T(default) : val, x)
    
    replace_nan!(u_4d, 0.0)
    replace_nan!(v_4d, 0.0)
    replace_nan!(w_4d, 0.0)
    replace_nan!(p_4d, 1013.25) # Standard surface pressure
    replace_nan!(t_3d, 288.15)  # Standard surface temperature
    replace_nan!(hlevel_4d, 9999.0) # High altitude default for TOA
    replace_nan!(ps_3d, 1013.25)

    # Determine interpolation scheme from config
    # Options: ReferenceInterp (exact matching), LinearInterp (Interpolations.jl), CubicInterp (modern)
    interp_order = config !== nothing ? config.interpolation_order : LinearInterp
    use_reference_interp = interp_order == ReferenceInterp
    use_cubic_vertical = interp_order == CubicInterp

    if use_reference_interp
        # Exact reference matching: floor-based trilinear + time interpolation
        # Uses 3D arrays at each time level (not 4D)
        u1_3d = permuted ? u_4d[:, :, :, 1] : met_fields.u1
        u2_3d = permuted ? u_4d[:, :, :, 2] : met_fields.u2
        v1_3d = permuted ? v_4d[:, :, :, 1] : met_fields.v1
        v2_3d = permuted ? v_4d[:, :, :, 2] : met_fields.v2
        w1_3d = permuted ? w_4d[:, :, :, 1] : met_fields.w1
        w2_3d = permuted ? w_4d[:, :, :, 2] : met_fields.w2

        # Apply v-wind negation for ERA5's reversed latitude grid (advection only)
        if negate_v
            v1_3d = -v1_3d
            v2_3d = -v2_3d
        end

        u_interp = ReferenceTrilinearInterpolant(copy(u1_3d), copy(u2_3d), T(t1), T(t2), nx, ny, nk)
        v_interp = ReferenceTrilinearInterpolant(copy(v1_3d), copy(v2_3d), T(t1), T(t2), nx, ny, nk)
        w_interp = ReferenceTrilinearInterpolant(copy(w1_3d), copy(w2_3d), T(t1), T(t2), nx, ny, nk)
    elseif use_cubic_vertical
        # Use our custom VerticalCubicInterpolant for stability on irregular vertical grids
        # It uses a local cubic Hermite scheme (O(1) evaluation, no object creation)
        u_interp = VerticalCubicInterpolant(u_4d, x_grid, y_grid, z_grid, t_grid, nx, ny)
        v_interp = VerticalCubicInterpolant(v_4d, x_grid, y_grid, z_grid, t_grid, nx, ny)
        w_interp = VerticalCubicInterpolant(w_4d, x_grid, y_grid, z_grid, t_grid, nx, ny)
        
        # 3D pressure field
        p_interp = VerticalCubicInterpolant(p_4d, x_grid, y_grid, z_grid, t_grid, nx, ny)

        # Temperature field
        t_interp = VerticalCubicInterpolant(t_3d, x_grid, y_grid, z_grid, t_grid, nx, ny)

        # Geopotential height field
        h_interp = VerticalCubicInterpolant(hlevel_4d, x_grid, y_grid, z_grid, t_grid, nx, ny)
    else
        # LinearInterp: Interpolations.jl linear on all axes
        u_interp = interpolate((x_grid, y_grid, z_grid, t_grid), u_4d,
                              Gridded(Linear()))
        v_interp = interpolate((x_grid, y_grid, z_grid, t_grid), v_4d,
                              Gridded(Linear()))
        w_interp = interpolate((x_grid, y_grid, z_grid, t_grid), w_4d,
                              Gridded(Linear()))

        # Apply extrapolation (clamped to boundary values)
        u_interp = extrapolate(u_interp, Flat())
        v_interp = extrapolate(v_interp, Flat())
        w_interp = extrapolate(w_interp, Flat())
    end
    # Note: CubicVerticalInterpolant (Dierckx) handles boundary clamping internally

    # Surface fields (3D: x, y, t) - always linear
    ps_interp = interpolate((x_grid, y_grid, t_grid), ps_3d, Gridded(Linear()))
    ps_interp = extrapolate(ps_interp, Flat())

    # Temperature boundary layer fields (3D) - linear
    precip_3d = Array{T, 3}(undef, nx, ny, 2)
    precip_3d[:, :, 1] .= met_fields.precip1
    precip_3d[:, :, 2] .= met_fields.precip2
    precip_interp = interpolate((x_grid, y_grid, t_grid), precip_3d, Gridded(Linear()))
    precip_interp = extrapolate(precip_interp, Flat())

    tbl_interp = interpolate((x_grid, y_grid, t_grid), tbl_3d, Gridded(Linear()))
    tbl_interp = extrapolate(tbl_interp, Flat())

    hbl_interp = interpolate((x_grid, y_grid, t_grid), hbl_3d, Gridded(Linear()))
    hbl_interp = extrapolate(hbl_interp, Flat())

    hflux_3d = Array{T, 3}(undef, nx, ny, 2)
    hflux_3d[:, :, 1] .= met_fields.hflux1
    hflux_3d[:, :, 2] .= met_fields.hflux2
    hflux_interp = interpolate((x_grid, y_grid, t_grid), hflux_3d, Gridded(Linear()))
    hflux_interp = extrapolate(hflux_interp, Flat())

    # Map scale factor (1D: y only) - latitude-dependent xm = 1/cos(lat)
    # Used for converting u-wind (m/s) to grid units/s in longitude direction
    xm_interp = interpolate((y_grid,), xm_1d, Gridded(Linear()))
    xm_interp = extrapolate(xm_interp, Flat())

    # 4D fields (x, y, sigma, t) for pressure, temperature, height
    # Note: When use_cubic_vertical=true, these are already created above with VerticalCubicInterpolant
    # Only need to create them here for LinearInterp mode
    if !use_cubic_vertical
        p_interp = interpolate((x_grid, y_grid, z_grid, t_grid), p_4d, Gridded(Linear()))
        t_interp = interpolate((x_grid, y_grid, z_grid, t_grid), t_3d, Gridded(Linear()))
        h_interp = interpolate((x_grid, y_grid, z_grid, t_grid), hlevel_4d, Gridded(Linear()))

        p_interp = extrapolate(p_interp, Flat())
        t_interp = extrapolate(t_interp, Flat())
        h_interp = extrapolate(h_interp, Flat())
    end

    # Align hybrid half-level coefficients with the permuted vertical grid ordering.
    # met_fields.ahalf/bhalf have length nk+1 (include top/surface boundary). After applying
    # level_perm (length nk) to the mid-layer ordering, append the extra boundary unchanged.
    a_half_vec = permuted ? vcat(met_fields.ahalf[level_perm], met_fields.ahalf[end]) : copy(met_fields.ahalf)
    b_half_vec = permuted ? vcat(met_fields.bhalf[level_perm], met_fields.bhalf[end]) : copy(met_fields.bhalf)

    return WindFields(
        u_interp, v_interp, w_interp, ps_interp, p_interp, t_interp, precip_interp, hflux_interp, tbl_interp, hbl_interp, h_interp, xm_interp,
        Vector{T}(a_half_vec), Vector{T}(b_half_vec),
        x_grid, y_grid, z_grid, (T(t1), T(t2)),
        T(lon_min), T(lon_max), T(lat_min), T(lat_max), dx_m, dy_m,
        nx, ny, nk,
        # Store raw w arrays aligned with z_grid ordering for floor-based interpolation
        # Materialize the views to Array{T,3}
        (permuted ? collect(met_fields.w1[:, :, level_perm]) : met_fields.w1),
        (permuted ? collect(met_fields.w2[:, :, level_perm]) : met_fields.w2)
    )
end

"""
    ParticleParams

Parameters for a single particle or particle component.

# Fields
- `icomp`: Component index (for multi-component simulations)
- `radius_m`: Particle radius (meters)
- `density_kg_m3`: Particle density (kg/m³)
- `grav_type`: Gravitational settling type
  - 0: No settling
  - 1: Constant settling velocity
  - 2: Variable settling (Stokes + slip correction from vgravtables)
- `gravity_ms`: Constant settling velocity (m/s) if grav_type == 1
- `map_ratio_x`: Map scale factor in x direction (accounts for projection distortion)
- `map_ratio_y`: Map scale factor in y direction
"""
@kwdef struct ParticleParams{T<:Real}
    icomp::Int = 1
    radius_m::T = 1.0e-6
    density_kg_m3::T = 2500.0
    grav_type::Int = 0
    gravity_ms::T = 0.0
    map_ratio_x::T = 1.0
    map_ratio_y::T = 1.0
end

"""
    ParticleODEParams{T,W}

Combined parameters struct for particle ODE integration.
Wraps both wind fields and particle parameters in a struct (not tuple)
to work properly with ContinuousCallback and ForwardDiff.

# Fields
- `winds::W`: WindFields interpolation object
- `params::ParticleParams{T}`: Particle parameters
"""
struct ParticleODEParams{T<:Real, W<:WindFields}
    winds::W
    params::ParticleParams{T}
end

"""
    particle_velocity!(du, u, p::ParticleODEParams, t)

ODE right-hand side: compute particle velocity at position u, time t.

# Arguments
- `du`: Output derivative [dx/dt, dy/dt, dz/dt]
- `u`: Current state [x, y, z] in grid units (x,y) and sigma/eta (z)
- `p`: Parameters struct (ParticleODEParams containing winds and params)
- `t`: Current time (seconds)

# State vector
- `u[1]`: x position (grid units, 1-indexed)
- `u[2]`: y position (grid units, 1-indexed)
- `u[3]`: z position (sigma/eta, 0 at top, 1 at surface)

# Velocity computation
1. Interpolate u,v,w winds to particle position
2. Add gravitational settling to vertical velocity (if enabled)
3. Apply map ratio corrections to horizontal velocities
4. Return du/dt = [u_wind, v_wind, w_wind + w_gravity]

# Notes
- Horizontal winds in m/s converted to grid units/s using map ratios
- Vertical velocity already in sigma/eta per second
- Gravitational settling computed using Stokes law with slip correction
- Uses struct parameters for ForwardDiff compatibility with callbacks
"""
function particle_velocity!(du, u, p::ParticleODEParams, t)
    # Access winds and params from struct (no unpacking needed)
    x, y, z = u

    # Interpolate wind components at particle position
    u_wind = p.winds.u_interp(x, y, z, t)
    v_wind = p.winds.v_interp(x, y, z, t)
    w_wind = p.winds.w_interp(x, y, z, t)

    # DEBUG: Print w_wind for particle 1 (first few evaluations only)
    global debug_call_count = get(task_local_storage(), :debug_w_count, 0)
    if p.particle_idx == 1 && debug_call_count < 5
        println("JULIA W_WIND DIAGNOSTIC #$(debug_call_count+1) - Particle 1 at t=$t s: w_wind=$w_wind sigma/s, z=$z")
        flush(stdout)
        task_local_storage(:debug_w_count, debug_call_count + 1)
    end

    # Gravitational settling (if enabled)
    w_grav = 0.0
    if p.params.grav_type > 0
        if p.params.grav_type == 1
            # Constant settling velocity
            w_grav = compute_settling_constant(p.params, p.winds, x, y, z, t)
        elseif p.params.grav_type == 2
            # Variable settling (pressure/temperature dependent)
            w_grav = compute_settling_variable(p.params, p.winds, x, y, z, t)
        end
    end

    # CRITICAL: Use latitude-dependent map scale factor for x-direction (longitude)
    # Map scale: xm(i,j) = 1/cos(lat)
    # Advection: rmx = xm/dxgrid, rmy = ym/dygrid; x = x + u*dt*rmx
    #
    # For geographic (lat/lon) grids, one degree of longitude shrinks toward poles:
    #   physical_distance_x = grid_distance_x * cos(lat)
    # So to convert m/s to grid_units/s: dx/dt = u * xm / dx_m where xm = 1/cos(lat)
    #
    # The y-direction (latitude) has constant spacing: ym = 1.0

    # Interpolate xm at current y-position (latitude-dependent)
    xm_at_particle = p.winds.xm_interp(y)
    map_ratio_x = xm_at_particle / p.winds.dx_m
    map_ratio_y = 1.0 / p.winds.dy_m  # ym = 1.0 for geographic grids

    # Convert winds from m/s to grid units/s
    # du/dt in grid units per second
    du[1] = u_wind * map_ratio_x
    du[2] = v_wind * map_ratio_y
    du[3] = w_wind + w_grav  # Vertical already in sigma/eta per second

    return nothing
end

"""
    compute_settling_constant(params, winds, x, y, z, t)

Compute constant gravitational settling velocity in sigma/eta coordinates.

Simple conversion from m/s to sigma/eta per second using local layer thickness.
"""
function compute_settling_constant(params::ParticleParams, winds::WindFields,
                                   x, y, z, t)
    gravity_ms = params.gravity_ms
    gravity_ms == 0 && return zero(gravity_ms)

    # Clamp the particle to the valid sigma/eta range and locate the surrounding layer.
    z_min = winds.z_grid[1]
    z_max = winds.z_grid[end]
    z_clamped = clamp(float(z), z_min, z_max)

    levels = winds.z_grid
    idx = searchsortedlast(levels, z_clamped)
    idx = clamp(idx, 1, length(levels) - 1)
    k1 = idx
    k2 = idx + 1

    # Surface pressure interpolant stored in hPa; convert to Pa for hybrid formula.
    ps_hpa = float(winds.ps_interp(float(x), float(y), float(t)))

    # Hybrid coefficients stored in hPa (ERA5 loader converts Pa → hPa).
    # Combine directly with surface pressure in hPa.
    p_half1_hpa = winds.a_half[k1] + winds.b_half[k1] * ps_hpa
    p_half2_hpa = winds.a_half[k2] + winds.b_half[k2] * ps_hpa

    # Exner function π = c_p (p/p₀)^κ with p in hPa and p₀ = 1000 hPa.
    cp_air = 1004.0
    kappa = 0.286
    p0_hpa = 1000.0
    exner(p_hpa) = cp_air * (p_hpa / p0_hpa)^kappa
    pi1 = exner(p_half1_hpa)
    pi2 = exner(p_half2_hpa)

    # Approximate potential temperature from the interpolated actual temperature.
    temp_actual = float(winds.t_interp(float(x), float(y), z_clamped, float(t)))
    pi_mid = 0.5 * (pi1 + pi2)
    if pi_mid == 0
        return zero(gravity_ms)
    end
    theta = temp_actual * (cp_air / pi_mid)

    g = float(GravitationalSettling.G_GRAVITY_M_S2)
    # Use abs() to ensure positive layer thickness (k2 > k1 means p2 > p1, so pi2 > pi1)
    dz = abs(theta * (pi1 - pi2) / g)
    if !isfinite(dz) || dz == 0
        return zero(gravity_ms)
    end

    # Hybrid coordinate difference: use actual sigma grid spacing between interfaces
    dsigma = abs(winds.z_grid[k2] - winds.z_grid[k1])

    # Both dz and dsigma are positive, so sigma_rate is positive (downward)
    sigma_rate = gravity_ms * dsigma / dz
    return convert(typeof(gravity_ms), sigma_rate)
end

"""
    compute_settling_variable(params, winds, x, y, z, t)

Compute variable gravitational settling using pressure/temperature-dependent drag.

Uses Stokes law with Cunningham slip correction factor.
Interpolates from pre-computed vgravtables (to be implemented).
"""
function compute_settling_variable(params::ParticleParams, winds::WindFields,
                                   x, y, z, t)
    # TODO: Implement variable settling using vgravtables
    # This requires:
    # 1. Interpolate pressure and temperature at particle position
    # 2. Look up settling velocity from pre-computed table
    # 3. Convert from m/s to sigma/eta per second

    # Placeholder: return zero for now
    return 0.0
end

"""
    check_domain_bounds(u, t, integrator)

Condition function for domain boundary callback.

Returns negative value when particle exits domain or hits ground.
Access parameters from integrator.p (ParticleODEParams struct).
"""
function check_domain_bounds(u, t, integrator)
    # Access winds from integrator parameters
    p = integrator.p
    x, y, z = u

    # Check horizontal domain bounds
    if x < 1.0 || x > p.winds.nx || y < 1.0 || y > p.winds.ny
        return -1.0  # Exited domain
    end

    # Check vertical bounds
    # In sigma coordinates: 0.0 = top of atmosphere, 1.0 = surface
    # z_grid is sorted in increasing order: [0.0, 0.05, ..., 0.95, 1.0]
    z_top = p.winds.z_grid[1]      # First element (top, ~0.0)
    z_surface = p.winds.z_grid[end] # Last element (surface, ~1.0)

    # Particle exits if it goes above top (z < z_top) or below surface (z > z_surface)
    if z < z_top || z > z_surface
        return -1.0  # Exited vertical domain
    end

    return 1.0  # Still in domain
end

"""
    deactivate_particle!(integrator)

Affect function for domain boundary callback.

Marks particle as inactive and terminates integration when it exits domain.
"""
function deactivate_particle!(integrator)
    terminate!(integrator)
end

"""
    create_particle_problem(initial_position, tspan, winds::WindFields,
                           params::ParticleParams)

Create an ODEProblem for a single particle trajectory.

# Arguments
- `initial_position`: [x0, y0, z0] initial position (grid units, grid units, sigma/eta)
- `tspan`: Time span (t_start, t_end) in seconds
- `winds`: WindFields interpolation object
- `params`: ParticleParams for this particle

# Returns
- `ODEProblem`: Problem ready to solve with DifferentialEquations.jl

# Example
```julia
winds = create_wind_interpolants(met_fields, 0.0, 21600.0)
params = ParticleParams(grav_type=1, gravity_ms=0.01)
prob = create_particle_problem([50.0, 50.0, 0.5], (0.0, 3600.0), winds, params)
sol = solve(prob, Tsit5())
```
"""
function create_particle_problem(initial_position::AbstractVector,
                                 tspan::Tuple,
                                 winds::WindFields,
                                 params::ParticleParams)

    # Create callback for domain boundary
    boundary_cb = ContinuousCallback(check_domain_bounds, deactivate_particle!)

    # Create ODE problem with struct parameters (required for ForwardDiff compatibility)
    prob = ODEProblem(particle_velocity!,
                     initial_position,
                     tspan,
                     ParticleODEParams(winds, params),
                     callback = boundary_cb)

    return prob
end

"""
    simulate_particles(initial_positions::Vector, tspan, winds::WindFields,
                      params::Vector{ParticleParams};
                      alg=Tsit5(), parallel=EnsembleThreads())

Simulate multiple particles in parallel using EnsembleProblem.

# Arguments
- `initial_positions`: Vector of [x, y, z] initial positions for each particle
- `tspan`: Time span (t_start, t_end) in seconds
- `winds`: WindFields interpolation object
- `params`: Vector of ParticleParams, one per particle (or single params for all)
- `alg`: ODE solver algorithm (default: Tsit5() - adaptive 5th order Runge-Kutta)
- `parallel`: Parallelization strategy (default: EnsembleThreads())

# Returns
- `EnsembleSolution`: Collection of particle trajectories

# Example
```julia
# Simulate 1000 particles
n_particles = 1000
positions = [rand(3) .* [100.0, 100.0, 1.0] for _ in 1:n_particles]
params_vec = [ParticleParams() for _ in 1:n_particles]

sols = simulate_particles(positions, (0.0, 3600.0), winds, params_vec)

# Access individual trajectories
# for (i, sol) in enumerate(sols)
#     println("Particle ", i, " final position: ", sol[end])
# end
```

# Notes
- Uses EnsembleThreads() for automatic parallel execution across CPU cores
- For GPU acceleration, use `parallel=EnsembleGPUArray()` (requires DiffEqGPU.jl)
- Adaptive time stepping means each particle uses optimal dt
"""
function simulate_particles(initial_positions::Vector,
                           tspan::Tuple,
                           winds::WindFields,
                           params::Vector{<:ParticleParams};
                           alg=Tsit5(),
                           parallel=EnsembleThreads(),
                           kwargs...)

    n_particles = length(initial_positions)
    @assert length(params) == n_particles "Must have one ParticleParams per particle"

    # Create a problem function that generates a problem for each particle
    # Use let block to properly capture the scope
    prob_func = let positions = initial_positions, particle_params = params
        function (prob, i, repeat)
            remake(prob, u0 = positions[i], p = ParticleODEParams(winds, particle_params[i]))
        end
    end

    # Create base problem (will be modified by prob_func)
    base_prob = create_particle_problem(initial_positions[1], tspan, winds, params[1])

    # Create ensemble problem
    ensemble_prob = EnsembleProblem(base_prob,
                                    prob_func = prob_func,
                                    safetycopy = false)  # winds is immutable, safe

    # Solve ensemble
    sol = solve(ensemble_prob, alg, parallel, trajectories = n_particles; kwargs...)

    return sol
end

"""
    simulate_particles(initial_positions::Vector, tspan, winds::WindFields,
                      params::ParticleParams; kwargs...)

Simulate multiple particles with same parameters (convenience method).
"""
function simulate_particles(initial_positions::Vector,
                           tspan::Tuple,
                           winds::WindFields,
                           params::ParticleParams;
                           kwargs...)
    n_particles = length(initial_positions)
    params_vec = [params for _ in 1:n_particles]
    return simulate_particles(initial_positions, tspan, winds, params_vec; kwargs...)
end

# Export public API
export WindFields, create_wind_interpolants
export ParticleParams, particle_velocity!
export create_particle_problem
# export simulate_particles  # Temporarily removed due to ensemble problem issue
