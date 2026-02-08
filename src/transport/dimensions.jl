# SNAP: Severe Nuclear Accident Programme
# Julia port of snapdimML.f90
# Original Copyright (C) 1992-2023 Norwegian Meteorological Institute
#
# Model dimensions and resolution parameters

module TransportDimensions

export nx, ny, nk, surface_index, maxsiz, ldata
export output_resolution_factor
export mcomp, mavail
export hres_field, hres_pos, lres_pos
export set_dimensions!

# Default grid sizes (when not specified in input)
const NXPRE = 864
const NYPRE = 698
const NKPRE = 61

# Grid dimensions (mutable, set during initialization)
"""Horizontal field dimension (x-direction)"""
nx = Ref(NXPRE)

"""Horizontal field dimension (y-direction)"""
ny = Ref(NYPRE)

"""Number of vertical levels"""
nk = Ref(NKPRE)

"""Index of the surface level in meteorology (1-based)"""
surface_index = Ref(-1)

"""Maximum input field size (possibly larger than nx*ny)"""
maxsiz = Ref(0)

"""Length of buffer for field input/output"""
ldata = Ref(0)

"""
Output grids can have finer resolution than input by this factor.
Default is 1 (same resolution as input).
"""
output_resolution_factor = Ref(1)

# Constants for array sizes
"""Maximum number of components in one run"""
const MCOMP = 1024

"""Maximum number of available timesteps with meteorological data"""
const MAVAIL = 8192

"""Convenience constants for module users"""
const mcomp = MCOMP
const mavail = MAVAIL

"""
    set_dimensions!(nx_new::Int, ny_new::Int, nk_new::Int;
                   surface_idx::Int=-1, output_res_factor::Int=1)

Set the grid dimensions and related parameters.

# Arguments
- `nx_new::Int` - Grid size in x-direction
- `ny_new::Int` - Grid size in y-direction
- `nk_new::Int` - Number of vertical levels
- `surface_idx::Int` - Index of surface level (default: -1)
- `output_res_factor::Int` - Output resolution enhancement factor (default: 1)
"""
function set_dimensions!(nx_new::Int, ny_new::Int, nk_new::Int;
                        surface_idx::Int=-1, output_res_factor::Int=1)
    nx[] = nx_new
    ny[] = ny_new
    nk[] = nk_new
    surface_index[] = surface_idx
    output_resolution_factor[] = output_res_factor

    # Set derived quantities
    maxsiz[] = nx_new * ny_new
    ldata[] = maxsiz[] * output_res_factor^2
end

"""
    hres_field(field::AbstractMatrix{T}, bilinear::Bool=false) where T <: Real

Translate a field from normal resolution to high output resolution.

# Arguments
- `field::AbstractMatrix` - Input field at normal resolution (nx × ny)
- `bilinear::Bool` - Use bilinear interpolation (default: false, uses nearest neighbor)

# Returns
- `Matrix` - High-resolution field (nx*factor × ny*factor)

# Notes
- If `output_resolution_factor == 1`, always uses nearest neighbor
- Bilinear interpolation extrapolates slightly at borders
"""
function hres_field(field::AbstractMatrix{T}, bilinear::Bool=false) where T <: Real
    factor = output_resolution_factor[]

    if factor == 1
        return copy(field)
    end

    n_x = nx[]
    n_y = ny[]

    # Initialize output array
    field_hres = zeros(T, n_x * factor, n_y * factor)

    # Nearest neighbor (always do this to handle borders)
    for j in 1:n_y
        for l in 1:factor
            for i in 1:n_x
                for k in 1:factor
                    field_hres[factor*(i-1)+k, factor*(j-1)+l] = field[i, j]
                end
            end
        end
    end

    # Bilinear interpolation for interior points
    if bilinear && factor > 1
        or_2 = div(factor, 2)
        dd = 1.0 / factor

        for j in 1:(n_y-1)
            for l in 1:factor
                for i in 1:(n_x-1)
                    for k in 1:factor
                        # Calculate interpolation weights
                        dx = (k - 1 - or_2) * dd
                        dy = (l - 1 - or_2) * dd
                        c1 = (1.0 - dy) * (1.0 - dx)
                        c2 = (1.0 - dy) * dx
                        c3 = dy * (1.0 - dx)
                        c4 = dy * dx

                        # Interpolate
                        field_hres[factor*i+k-or_2, factor*j+l-or_2] =
                            c1 * field[i, j] + c2 * field[i+1, j] +
                            c3 * field[i, j+1] + c4 * field[i+1, j+1]
                    end
                end
            end
        end
    end

    return field_hres
end

"""
    hres_field(field::AbstractMatrix{Int8}, bilinear::Bool=false)

Specialized version for Int8 arrays (e.g., land use categories).
Always uses nearest neighbor interpolation.
"""
function hres_field(field::AbstractMatrix{Int8}, bilinear::Bool=false)
    factor = output_resolution_factor[]

    if factor == 1
        return copy(field)
    end

    n_x = nx[]
    n_y = ny[]

    field_hres = zeros(Int8, n_x * factor, n_y * factor)

    for j in 1:n_y
        for l in 1:factor
            for i in 1:n_x
                for k in 1:factor
                    field_hres[factor*(i-1)+k, factor*(j-1)+l] = field[i, j]
                end
            end
        end
    end

    return field_hres
end

"""
    hres_pos(lres_position::Real) -> Int

Translate an x or y position from the input grid to the high-resolution output grid.

# Arguments
- `lres_position::Real` - Position in low-resolution grid (1-based, can be fractional)

# Returns
- `Int` - Position in high-resolution grid (1-based)

# Notes
- Cells are assumed to be centered (cell 1 covers [0.5, 1.5))
"""
function hres_pos(lres_position::Real)::Int
    factor = output_resolution_factor[]
    # Convert to 0.5-starting position, scale to new range, convert to 1-start
    return round(Int, (lres_position - 0.5) * factor + 1.0)
end

"""
    lres_pos(hres_position::Int) -> Int

Translate an x or y position from the output grid to the low-resolution input grid.

# Arguments
- `hres_position::Int` - Position in high-resolution grid (1-based)

# Returns
- `Int` - Position in low-resolution grid (1-based)
"""
function lres_pos(hres_position::Int)::Int
    factor = output_resolution_factor[]
    # Convert to 0-starting position, scale to new range, convert to 1-start
    return round(Int, (hres_position - 1.0) / factor + 0.5)
end

end  # module TransportDimensions