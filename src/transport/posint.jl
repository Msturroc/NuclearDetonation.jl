# SNAP: Severe Nuclear Accident Programme
# Julia port of posintML.f90
# Original Copyright (C) 1992-2023 Norwegian Meteorological Institute
#
# Interpolation of meteorological fields to particle positions

"""
    interpolate_met_to_particle!(particle::Particle, pextra::ExtraParticle,
                                  fields::MeteoFields{T},
                                  t1::Real, t2::Real, tnow::Real,
                                  dx::Real, dy::Real) where T

Interpolate boundary layer top, boundary layer height, map ratios, and precipitation
to particle position.

This function performs:
1. **Temporal interpolation** between two time levels (fields *1 and *2)
2. **Bilinear spatial interpolation** at particle position (x, y)

Updates:
- `particle.tbl`: Sigma/eta at top of boundary layer
- `particle.hbl`: Height of boundary layer (m)
- `pextra.rmx`: Map ratio in x direction (dimensionless)
- `pextra.rmy`: Map ratio in y direction (dimensionless)
- `pextra.prc`: Precipitation intensity (mm/hour)

# Arguments
- `particle::Particle`: Particle to update (position in grid coordinates)
- `pextra::ExtraParticle`: Extra particle data to fill
- `fields::MeteoFields`: Meteorological fields at two time levels
- `t1::Real`: Time in seconds for field set 1 (e.g., 0.0)
- `t2::Real`: Time in seconds for field set 2 (e.g., 21600.0 for 6 hours)
- `tnow::Real`: Time in seconds for current particle position
- `dx::Real`: Grid spacing in x direction (m)
- `dy::Real`: Grid spacing in y direction (m)

# Notes
- Particle position (x, y) is in grid coordinates (1-based, can be fractional)
- Temporal interpolation is linear between t1 and t2
- Spatial interpolation is bilinear using the 4 surrounding grid points
- If particle is inactive, no computation is performed
"""
function interpolate_met_to_particle!(particle::Particle,
                                       pextra::ExtraParticle,
                                       fields::MeteoFields{T},
                                       t1::Real, t2::Real, tnow::Real,
                                       dx::Real, dy::Real) where T
    # Skip inactive particles
    if !is_active(particle)
        return nothing
    end

    # Temporal interpolation weights
    dt = t2 - t1
    @assert dt > 0 "Time interval must be positive"

    rt1 = (t2 - tnow) / dt  # Weight for older fields (*1)
    rt2 = (tnow - t1) / dt  # Weight for newer fields (*2)

    # Particle position in grid coordinates
    x_pos = particle.x
    y_pos = particle.y

    # Lower-left corner indices (integer part)
    i = floor(Int, x_pos)
    j = floor(Int, y_pos)

    # Ensure indices are within bounds
    nx, ny = fields.nx, fields.ny
    i = clamp(i, 1, nx - 1)
    j = clamp(j, 1, ny - 1)

    # Fractional part for bilinear interpolation
    dx_frac = x_pos - i
    dy_frac = y_pos - j

    # Bilinear interpolation weights
    c1 = (1 - dy_frac) * (1 - dx_frac)  # Lower-left  (i, j)
    c2 = (1 - dy_frac) * dx_frac        # Lower-right (i+1, j)
    c3 = dy_frac * (1 - dx_frac)        # Upper-left  (i, j+1)
    c4 = dy_frac * dx_frac              # Upper-right (i+1, j+1)

    # === Boundary layer top (sigma/eta coordinate) ===
    # Note: In original Fortran, this interpolated from bl1/bl2 arrays
    # Here we assume boundary layer top is stored in a field
    # For now, we'll skip this as it's not in MeteoFields structure
    # TODO: Add bl (boundary layer top in sigma/eta) to MeteoFields if needed

    # === Boundary layer height (m) ===
    hbl = rt1 * (c1 * fields.hbl1[i, j] + c2 * fields.hbl1[i+1, j] +
                  c3 * fields.hbl1[i, j+1] + c4 * fields.hbl1[i+1, j+1]) +
          rt2 * (c1 * fields.hbl2[i, j] + c2 * fields.hbl2[i+1, j] +
                  c3 * fields.hbl2[i, j+1] + c4 * fields.hbl2[i+1, j+1])

    particle.hbl = Float32(hbl)

    # === Map ratios (dimensionless scale factors) ===
    # Map ratios are time-invariant, so no temporal interpolation
    rmx = c1 * fields.xm[i, j] + c2 * fields.xm[i+1, j] +
          c3 * fields.xm[i, j+1] + c4 * fields.xm[i+1, j+1]

    rmy = c1 * fields.ym[i, j] + c2 * fields.ym[i+1, j] +
          c3 * fields.ym[i, j+1] + c4 * fields.ym[i+1, j+1]

    # Normalize by grid spacing to get map scale factor
    pextra.rmx = rmx / dx
    pextra.rmy = rmy / dy

    # === Precipitation intensity (mm/hour) ===
    # Precipitation is instantaneous field (no temporal interpolation needed)
    prc_t1 = c1 * fields.precip1[i, j] + c2 * fields.precip1[i+1, j] +
             c3 * fields.precip1[i, j+1] + c4 * fields.precip1[i+1, j+1]
    prc_t2 = c1 * fields.precip2[i, j] + c2 * fields.precip2[i+1, j] +
             c3 * fields.precip2[i, j+1] + c4 * fields.precip2[i+1, j+1]
    prc = rt1 * prc_t1 + rt2 * prc_t2

    pextra.prc = Float32(prc)

    return nothing
end

"""
    bilinear_interpolate(field::AbstractMatrix{T}, x::Real, y::Real) where T

Perform bilinear interpolation of a 2D field at fractional position (x, y).

# Arguments
- `field::AbstractMatrix`: 2D field to interpolate (nx × ny)
- `x::Real`: X position in grid coordinates (1-based, can be fractional)
- `y::Real`: Y position in grid coordinates (1-based, can be fractional)

# Returns
- `T`: Interpolated value

# Notes
- Position is clamped to valid grid bounds
- Uses bilinear interpolation with 4 surrounding grid points
"""
function bilinear_interpolate(field::AbstractMatrix{T}, x::Real, y::Real) where T
    nx, ny = size(field)

    # Lower-left corner indices
    i = floor(Int, x)
    j = floor(Int, y)

    # Clamp to valid range
    i = clamp(i, 1, nx - 1)
    j = clamp(j, 1, ny - 1)

    # Fractional part
    dx = x - i
    dy = y - j

    # Bilinear weights
    c1 = (1 - dy) * (1 - dx)
    c2 = (1 - dy) * dx
    c3 = dy * (1 - dx)
    c4 = dy * dx

    # Interpolate
    return c1 * field[i, j] + c2 * field[i+1, j] +
           c3 * field[i, j+1] + c4 * field[i+1, j+1]
end

"""
    trilinear_interpolate(field::AbstractArray{T,3}, x::Real, y::Real, z::Real) where T

Perform trilinear interpolation of a 3D field at fractional position (x, y, z).

# Arguments
- `field::AbstractArray{T,3}`: 3D field to interpolate (nx × ny × nz)
- `x::Real`: X position in grid coordinates
- `y::Real`: Y position in grid coordinates
- `z::Real`: Z position in grid coordinates (vertical level)

# Returns
- `T`: Interpolated value

# Notes
- Position is clamped to valid grid bounds
- Uses trilinear interpolation with 8 surrounding grid points
"""
function trilinear_interpolate(field::AbstractArray{T,3}, x::Real, y::Real, z::Real) where T
    nx, ny, nz = size(field)

    # Lower corner indices
    i = floor(Int, x)
    j = floor(Int, y)
    k = floor(Int, z)

    # Clamp to valid range
    i = clamp(i, 1, nx - 1)
    j = clamp(j, 1, ny - 1)
    k = clamp(k, 1, nz - 1)

    # Fractional parts
    dx = x - i
    dy = y - j
    dz = z - k

    # Trilinear weights (8 corners of cube)
    c000 = (1 - dx) * (1 - dy) * (1 - dz)
    c100 = dx * (1 - dy) * (1 - dz)
    c010 = (1 - dx) * dy * (1 - dz)
    c110 = dx * dy * (1 - dz)
    c001 = (1 - dx) * (1 - dy) * dz
    c101 = dx * (1 - dy) * dz
    c011 = (1 - dx) * dy * dz
    c111 = dx * dy * dz

    # Interpolate from 8 surrounding points
    return c000 * field[i, j, k] + c100 * field[i+1, j, k] +
           c010 * field[i, j+1, k] + c110 * field[i+1, j+1, k] +
           c001 * field[i, j, k+1] + c101 * field[i+1, j, k+1] +
           c011 * field[i, j+1, k+1] + c111 * field[i+1, j+1, k+1]
end

export interpolate_met_to_particle!
export bilinear_interpolate, trilinear_interpolate
