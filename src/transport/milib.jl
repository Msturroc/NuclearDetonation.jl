# Meteorological Interpolation Library (milib)
#
# Coordinate conversion routines for various map projections.

"""
    earthr()

Return Earth radius in metres (standard value).

# Returns
- `Float64`: Earth radius = 6,371,000 m
"""
function earthr()
    return 6371000.0
end

"""
    sph2rot!(icall::Int, x::Vector{Float64}, y::Vector{Float64},
             xcen::Float64, ycen::Float64)

Convert between spherical (xsph, ysph) and spherical rotated (xrot, yrot) coordinates.
All values are in radians. Arrays are modified in-place.

# Arguments
- `icall`: +1 = spherical → rotated, -1 = rotated → spherical
- `x`: Longitude values (radians), modified in-place
- `y`: Latitude values (radians), modified in-place
- `xcen`: Longitude position of rotated equator/Greenwich (radians)
- `ycen`: Latitude position of rotated equator (radians)

# Returns
- `ierror`: 0 = success, 1 = invalid icall

# Based on
Hirlam code by Anstein Foss (1995)
"""
function sph2rot!(icall::Int, x::Vector{Float64}, y::Vector{Float64},
                  xcen::Float64, ycen::Float64)
    n = length(x)
    @assert length(y) == n "x and y must have same length"

    ierror = 0

    zsycen = sin(ycen)
    zcycen = cos(ycen)

    if icall == 1
        # Compute spherical rotated coordinates from spherical coordinates
        for j in 1:n
            xsph = x[j]
            ysph = y[j]
            zxmxc = xsph - xcen
            zsxmxc = sin(zxmxc)
            zcxmxc = cos(zxmxc)
            zsysph = sin(ysph)
            zcysph = cos(ysph)

            zsyrot = zcycen * zsysph - zsycen * zcysph * zcxmxc
            zsyrot = clamp(zsyrot, -1.0, 1.0)
            yrot = asin(zsyrot)
            zcyrot = cos(yrot)

            zcxrot = (zcycen * zcysph * zcxmxc + zsycen * zsysph) / zcyrot
            zcxrot = clamp(zcxrot, -1.0, 1.0)
            zsxrot = zcysph * zsxmxc / zcyrot
            xrot = acos(zcxrot)
            if zsxrot < 0.0
                xrot = -xrot
            end

            x[j] = xrot
            y[j] = yrot
        end

    elseif icall == -1
        # Compute spherical coordinates from spherical rotated coordinates
        for j in 1:n
            xrot = x[j]
            yrot = y[j]
            zsxrot = sin(xrot)
            zcxrot = cos(xrot)
            zsyrot = sin(yrot)
            zcyrot = cos(yrot)

            zsysph = zcycen * zsyrot + zsycen * zcyrot * zcxrot
            zsysph = clamp(zsysph, -1.0, 1.0)
            ysph = asin(zsysph)
            zcysph = cos(ysph)

            zcxmxc = (zcycen * zcyrot * zcxrot - zsycen * zsyrot) / zcysph
            zcxmxc = clamp(zcxmxc, -1.0, 1.0)
            zsxmxc = zcyrot * zsxrot / zcysph
            zxmxc = acos(zcxmxc)
            if zsxmxc < 0.0
                zxmxc = -zxmxc
            end
            xsph = zxmxc + xcen

            x[j] = xsph
            y[j] = ysph
        end
    else
        ierror = 1
    end

    return ierror
end

"""
    pol2sph!(icall::Int, x::Vector{Float64}, y::Vector{Float64},
             fpol::Float64, xp::Float64, yp::Float64, an::Float64, fi::Float64)

Convert between polar stereographic and spherical coordinates.
Polar stereographic coordinates are grid positions (starting at 1.0, 1.0 for lower left).
Spherical coordinates are in radians.

# Arguments
- `icall`: +1 = polar → spherical, -1 = spherical → polar
- `x`: X coordinates (grid positions or longitude in radians), modified in-place
- `y`: Y coordinates (grid positions or latitude in radians), modified in-place
- `fpol`: Projection latitude (degrees), negative for southern hemisphere
- `xp`: X position of north pole in grid coordinates
- `yp`: Y position of north pole in grid coordinates
- `an`: Grid units between North Pole and Equator
- `fi`: Grid rotation angle (radians)

# Returns
- `ierror`: 0 = success, 1 = invalid icall

# Based on
Hirlam code by Anstein Foss (1995-1997)
"""
function pol2sph!(icall::Int, x::Vector{Float64}, y::Vector{Float64},
                  fpol::Float64, xp::Float64, yp::Float64, an::Float64, fi::Float64)
    n = length(x)
    @assert length(y) == n "x and y must have same length"

    ierror = 0

    rearth = earthr()

    zfpol = fpol
    zxp = xp
    zyp = yp
    zan = an
    zfi = fi

    # Constants
    zpi = π
    ztwopi = 2π
    zns = 1.0
    if zfpol < 0.0
        zns = -1.0
        zfi = -zfi
    end
    zpihal = π / 2

    if icall == 1
        # Compute spherical coordinates from polar stereographic coordinates
        for j in 1:n
            xpol = x[j]
            ypol = y[j]

            zdx = xpol - zxp
            zdy = zyp - ypol
            zr = sqrt(zdx * zdx + zdy * zdy)

            ysph = zpihal - 2.0 * atan(zr / zan)
            xsph = 0.0
            if zr > 1.0e-10
                xsph = zfi + atan(zdx, zdy)
            end

            # Normalize longitude to [-π, π]
            while xsph <= -zpi
                xsph += ztwopi
            end
            while xsph > zpi
                xsph -= ztwopi
            end

            x[j] = xsph * zns
            y[j] = ysph * zns
        end

    elseif icall == -1
        # Compute polar stereographic coordinates from spherical coordinates
        alfa = sin(zpihal + zfi)
        beta = cos(zpihal + zfi)

        for j in 1:n
            xsph = x[j] * zns
            ysph = y[j] * zns

            zr = zan * cos(ysph) / (1.0 + sin(ysph))
            xr = zr * sin(xsph)
            yr = -zr * cos(xsph)

            xpol = xr * alfa - yr * beta + zxp
            ypol = yr * alfa + xr * beta + zyp

            x[j] = xpol
            y[j] = ypol
        end
    else
        ierror = 1
    end

    return ierror
end

"""
    lam2sph!(icall::Int, x::Vector{Float64}, y::Vector{Float64},
             xw::Float64, ys::Float64, dx::Float64, dy::Float64,
             x0::Float64, y1::Float64, y2::Float64)

Convert between Lambert (non-oblique) and spherical coordinates.
Lambert coordinates are grid positions (starting at 1.0, 1.0 for lower left).
Spherical coordinates are in radians.

# Arguments
- `icall`: +1 = Lambert → spherical, -1 = spherical → Lambert
- `x`: X coordinates (grid positions or longitude in radians), modified in-place
- `y`: Y coordinates (grid positions or latitude in radians), modified in-place
- `xw`: Western boundary longitude (radians)
- `ys`: Southern boundary latitude (radians)
- `dx`: X resolution at reference latitude (meters)
- `dy`: Y resolution at reference latitude (meters)
- `x0`: Reference longitude (radians)
- `y1`: First secant latitude (radians), or tangent latitude if y1 ≈ y2
- `y2`: Second secant latitude (radians)

# Returns
- `ierror`: 0 = success, 1 = invalid icall

# Based on
Lambert projection code (2008)
"""
function lam2sph!(icall::Int, x::Vector{Float64}, y::Vector{Float64},
                  xw::Float64, ys::Float64, dx::Float64, dy::Float64,
                  x0::Float64, y1::Float64, y2::Float64)
    n = length(x)
    @assert length(y) == n "x and y must have same length"

    ierror = 0

    # Utility function
    ftan(yy) = tan(π/4 + 0.5 * yy)

    zRe = earthr()
    zxw = xw
    zys = ys
    zdx = dx
    zdy = dy
    zx0 = x0
    zy1 = y1
    zy2 = y2

    # Test if tangent or secant projection
    if abs(zy1 - zy2) < 1.0e-5
        # Tangent version
        zn = sin(zy1)
    else
        # Secant version
        zn = log(cos(zy1) / cos(zy2)) / log(ftan(zy2) / ftan(zy1))
    end

    # Projection constants
    zninv = 1.0 / zn
    ztn = ftan(zy1)^zn
    zF = cos(zy1) * ztn * zninv
    zR0 = zRe * zF / ztn

    # Find Lambert coordinates of lower left corner (0,0) in grid
    zR = zRe * zF / (ftan(zys)^zn)
    ztheta = zn * (zxw - zx0)
    zxlam0 = zR * sin(ztheta) - zdx
    zylam0 = zR0 - zR * cos(ztheta) - zdy

    if icall == 1
        # Compute spherical coordinates from Lambert coordinates
        for j in 1:n
            xlam = zxlam0 + x[j] * zdx
            ylam = zylam0 + y[j] * zdy

            zR = sign(zn) * sqrt(xlam * xlam + (zR0 - ylam) * (zR0 - ylam))
            ztheta = atan(xlam, zR0 - ylam)

            xsph = zx0 + ztheta * zninv
            ysph = 2.0 * (atan((zRe * zF / zR)^zninv) - π/4)

            # Normalize longitude to [-π, π]
            if xsph < -π
                xsph += 2π
            end
            if xsph > π
                xsph -= 2π
            end

            x[j] = xsph
            y[j] = ysph
        end

    elseif icall == -1
        # Compute Lambert coordinates from spherical coordinates
        for j in 1:n
            xsph = x[j]
            ysph = y[j]

            zR = zRe * zF / (ftan(ysph)^zn)
            ztheta = zn * (xsph - zx0)

            xlam = zR * sin(ztheta)
            ylam = zR0 - zR * cos(ztheta)

            x[j] = (xlam - zxlam0) / zdx
            y[j] = (ylam - zylam0) / zdy
        end
    else
        ierror = 1
    end

    return ierror
end

"""
    mer2sph!(icall::Int, x::Vector{Float64}, y::Vector{Float64},
             xw::Float64, ys::Float64, dx::Float64, dy::Float64, yc::Float64)

Convert between Mercator (unrotated) and spherical coordinates.
Mercator coordinates are grid positions (starting at 1.0, 1.0 for lower left).
Spherical coordinates are in radians.

# Arguments
- `icall`: +1 = Mercator → spherical, -1 = spherical → Mercator
- `x`: X coordinates (grid positions or longitude in radians), modified in-place
- `y`: Y coordinates (grid positions or latitude in radians), modified in-place
- `xw`: Western boundary longitude (radians)
- `ys`: Southern boundary latitude (radians)
- `dx`: X resolution at construction latitude (meters)
- `dy`: Y resolution at construction latitude (meters)
- `yc`: Construction latitude (radians)

# Returns
- `ierror`: 0 = success, 1 = invalid icall

# Based on
Mercator projection code by Anstein Foss (1996)
"""
function mer2sph!(icall::Int, x::Vector{Float64}, y::Vector{Float64},
                  xw::Float64, ys::Float64, dx::Float64, dy::Float64, yc::Float64)
    n = length(x)
    @assert length(y) == n "x and y must have same length"

    ierror = 0

    zxw = xw
    zys = ys
    zdx = dx
    zdy = dy
    zyc = yc

    rearth = earthr()
    zpih = π / 2
    zrcos = rearth * cos(zyc)
    zxmerc = zrcos * zxw - zdx
    zymerc = zrcos * log((1.0 + sin(zys)) / cos(zys)) - zdy

    # Test if possible to avoid heavy computations dependent on y
    ieq = if n < 4
        0
    elseif y[1] == y[2] || y[2] == y[3] || y[3] == y[4]
        1
    else
        0
    end

    if icall == 1
        # Compute spherical coordinates from Mercator coordinates
        if ieq == 0
            for j in 1:n
                xmer = zxmerc + x[j] * zdx
                ymer = zymerc + y[j] * zdy

                xsph = xmer / zrcos
                ysph = 2.0 * atan(exp(ymer / zrcos)) - zpih

                x[j] = xsph
                y[j] = ysph
            end
        else
            # Optimization when y values repeat
            ypos = -1.0e35
            if ypos == y[1]
                ypos = 0.0
            end
            ysph = 0.0

            for j in 1:n
                xmer = zxmerc + x[j] * zdx
                xsph = xmer / zrcos
                x[j] = xsph

                if y[j] != ypos
                    ypos = y[j]
                    ymer = zymerc + y[j] * zdy
                    ysph = 2.0 * atan(exp(ymer / zrcos)) - zpih
                end
                y[j] = ysph
            end
        end

    elseif icall == -1
        # Compute Mercator coordinates from spherical coordinates
        if ieq == 0
            for j in 1:n
                xsph = x[j]
                ysph = y[j]

                xmer = zrcos * xsph
                ymer = zrcos * log((1.0 + sin(ysph)) / cos(ysph))

                x[j] = (xmer - zxmerc) / zdx
                y[j] = (ymer - zymerc) / zdy
            end
        else
            # Optimization when y values repeat
            ysph = -1.0e35
            if ysph == y[1]
                ysph = 0.0
            end
            ypos = 0.0

            for j in 1:n
                xsph = x[j]
                xmer = zrcos * xsph
                x[j] = (xmer - zxmerc) / zdx

                if y[j] != ysph
                    ysph = y[j]
                    ymer = zrcos * log((1.0 + sin(ysph)) / cos(ysph))
                    ypos = (ymer - zymerc) / zdy
                end
                y[j] = ypos
            end
        end
    else
        ierror = 1
    end

    return ierror
end

"""
    xyconvert!(npos::Int, x::Vector{Float64}, y::Vector{Float64},
               igtypa::Int, ga::Vector{Float64}, igtypr::Int, gr::Vector{Float64})

Convert coordinates from one grid type to another. This is a two-phase conversion:
1. Convert from input grid to geographic (spherical) coordinates
2. Convert from geographic to output grid coordinates

# Arguments
- `npos`: Number of positions to convert
- `x`: X coordinates (modified in-place)
- `y`: Y coordinates (modified in-place)
- `igtypa`: Input grid type (1=polar60°, 2=geographic, 3=rotated, 4=polar, 5=mercator, 6=lambert)
- `ga`: Input grid parameters (6 elements)
- `igtypr`: Output grid type
- `gr`: Output grid parameters (6 elements)

# Grid Parameter Description

For all grid types, parameters are stored in a 6-element array:

**Spherical/Rotated (igtype=2,3):**
- `g[1]`: Western boundary (degrees)
- `g[2]`: Southern boundary (degrees)
- `g[3]`: Longitude increment (degrees)
- `g[4]`: Latitude increment (degrees)
- `g[5]`: Longitude of rotated equator (degrees, 0 for geographic)
- `g[6]`: Latitude of rotated equator (degrees, 0 for geographic)

**Polar Stereographic (igtype=1,4):**
- `g[1]`: X position of north pole (grid units)
- `g[2]`: Y position of north pole (grid units)
- `g[3]`: Number of grid units between pole and equator
- `g[4]`: Grid rotation angle (degrees)
- `g[5]`: Projection latitude (degrees, 60 for igtype=1)
- `g[6]`: Not used

**Mercator (igtype=5):**
- `g[1]`: Western boundary (degrees)
- `g[2]`: Southern boundary (degrees)
- `g[3]`: X increment (km)
- `g[4]`: Y increment (km)
- `g[5]`: Construction latitude (degrees)
- `g[6]`: Not used

**Lambert (igtype=6):**
- `g[1]`: Western boundary (degrees)
- `g[2]`: Southern boundary (degrees)
- `g[3]`: X increment (km)
- `g[4]`: Y increment (km)
- `g[5]`: Reference longitude (degrees)
- `g[6]`: Reference latitude (degrees)

# Returns
- `ierror`: 0 = success, 1 = invalid grid type/parameters

# Based on
Coordinate conversion code by J.E. Haugen and Anstein Foss (1994-1996)
"""
function xyconvert!(npos::Int, x::Vector{Float64}, y::Vector{Float64},
                    igtypa::Int, ga::Vector{Float64}, igtypr::Int, gr::Vector{Float64})
    @assert length(x) >= npos "x must have at least npos elements"
    @assert length(y) >= npos "y must have at least npos elements"
    @assert length(ga) == 6 "ga must have 6 elements"
    @assert length(gr) == 6 "gr must have 6 elements"

    zpir18 = π / 180.0
    rearth = earthr()

    ierror = 0
    iconv2 = 1

    # Phase 1: Convert from input to geographic coordinates
    if igtypa == 2 || igtypa == 3
        # Spherical -> geographic
        xwa = ga[1] * zpir18
        ysa = ga[2] * zpir18
        dxa = ga[3] * zpir18
        dya = ga[4] * zpir18
        xca = ga[5] * zpir18
        yca = ga[6] * zpir18

        for j in 1:npos
            x[j] = xwa + (x[j] - 1.0) * dxa
            y[j] = ysa + (y[j] - 1.0) * dya
        end

        if xca != 0.0 || yca != 0.0
            ierror = sph2rot!(-1, x, y, xca, yca)
        end

    elseif igtypa == 1 || igtypa == 4
        # Polar -> geographic (unless pol->pol with same projection latitude)
        xpa = ga[1]
        ypa = ga[2]
        ana = ga[3]
        fia = ga[4] * zpir18
        fpa = ga[5] * zpir18

        if (igtypr == 1 || igtypr == 4) && ga[5] == gr[5]
            # Optimization: direct pol->pol transformation
            xpr = gr[1]
            ypr = gr[2]
            anr = gr[3]
            fir = gr[4] * zpir18
            fpr = gr[5] * zpir18

            zcrot = cos(fia - fir)
            zsrot = sin(fia - fir)
            zx2 = zcrot * (anr / ana)
            zx3 = -zsrot * (anr / ana)
            zy2 = zsrot * (anr / ana)
            zy3 = zcrot * (anr / ana)
            zx1 = xpr - zx2 * xpa - zx3 * ypa
            zy1 = ypr - zy2 * xpa - zy3 * ypa

            for j in 1:npos
                xa = x[j]
                ya = y[j]
                x[j] = zx1 + zx2 * xa + zx3 * ya
                y[j] = zy1 + zy2 * xa + zy3 * ya
            end

            iconv2 = 0
        else
            ierror = pol2sph!(1, x, y, fpa, xpa, ypa, ana, fia)
        end

    elseif igtypa == 5
        # Mercator -> geographic
        xwa = ga[1] * zpir18
        ysa = ga[2] * zpir18
        dxa = ga[3] * 1000.0
        dya = ga[4] * 1000.0
        yca = ga[5] * zpir18

        ierror = mer2sph!(1, x, y, xwa, ysa, dxa, dya, yca)

    elseif igtypa == 6
        # Lambert -> geographic
        xwa = ga[1] * zpir18
        ysa = ga[2] * zpir18
        dxa = ga[3] * 1000.0
        dya = ga[4] * 1000.0
        xca = ga[5] * zpir18
        yca = ga[6] * zpir18

        ierror = lam2sph!(1, x, y, xwa, ysa, dxa, dya, xca, yca, yca)
    else
        ierror = 1
    end

    if iconv2 == 0 || ierror != 0
        return ierror
    end

    # Phase 2: Convert from geographic to output coordinates
    if igtypr == 2 || igtypr == 3
        # Geographic -> spherical
        xwr = gr[1] * zpir18
        ysr = gr[2] * zpir18
        dxr = gr[3] * zpir18
        dyr = gr[4] * zpir18
        xcr = gr[5] * zpir18
        ycr = gr[6] * zpir18

        if xcr != 0.0 || ycr != 0.0
            ierror = sph2rot!(1, x, y, xcr, ycr)
            if ierror != 0
                return ierror
            end
        end

        for j in 1:npos
            x[j] = (x[j] - xwr) / dxr + 1.0
            y[j] = (y[j] - ysr) / dyr + 1.0
        end

    elseif igtypr == 1 || igtypr == 4
        # Geographic -> polar
        xpr = gr[1]
        ypr = gr[2]
        anr = gr[3]
        fir = gr[4] * zpir18
        fpr = gr[5] * zpir18

        ierror = pol2sph!(-1, x, y, fpr, xpr, ypr, anr, fir)

    elseif igtypr == 5
        # Geographic -> mercator
        xwr = gr[1] * zpir18
        ysr = gr[2] * zpir18
        dxr = gr[3] * 1000.0
        dyr = gr[4] * 1000.0
        ycr = gr[5] * zpir18

        ierror = mer2sph!(-1, x, y, xwr, ysr, dxr, dyr, ycr)

    elseif igtypr == 6
        # Geographic -> lambert
        xwr = gr[1] * zpir18
        ysr = gr[2] * zpir18
        dxr = gr[3] * 1000.0
        dyr = gr[4] * 1000.0
        xcr = gr[5] * zpir18
        ycr = gr[6] * zpir18

        ierror = lam2sph!(-1, x, y, xwr, ysr, dxr, dyr, xcr, ycr, ycr)
    else
        ierror = 1
    end

    return ierror
end

"""
    mapfield(imapr::Int, icori::Int, igtype::Int, grid::Vector{Float64},
             nx::Int, ny::Int)

Compute parameters (fields) dependent on the map projection: map ratio (in x and y direction)
and Coriolis parameter.

# Arguments
- `imapr`: Map ratio computation mode
  - 0: Do not compute map ratio
  - 1: Compute map ratio
  - 2+: Compute map ratio divided by grid resolution in meters times (imapr-1)
- `icori`: Coriolis parameter computation mode
  - 0: Do not compute Coriolis parameter
  - 1: Compute Coriolis parameter
- `igtype`: Grid type
  - 1: Polar stereographic (true at 60° N)
  - 2: Geographic
  - 3: Spherical rotated
  - 4: Polar stereographic (custom latitude)
  - 5: Mercator (unrotated)
  - 6: Lambert (tangent, non-oblique)
- `grid`: Grid parameters (6-element vector, see below)
- `nx`, `ny`: Grid dimensions

# Grid Parameters by Type

**Polar stereographic (igtype=1,4):**
- grid[1]: X position of north pole (grid units)
- grid[2]: Y position of north pole (grid units)
- grid[3]: Grid units between North Pole and Equator
- grid[4]: Grid rotation angle (degrees)
- grid[5]: Projection latitude (degrees), standard is 60
- grid[6]: Not used

**Geographic/rotated spherical (igtype=2,3):**
- grid[1]: Western boundary (degrees, longitude for x=1)
- grid[2]: Southern boundary (degrees, latitude for y=1)
- grid[3]: Longitude increment (degrees)
- grid[4]: Latitude increment (degrees)
- grid[5]: Longitude position of rotated equator (degrees, 0 for geographic)
- grid[6]: Latitude position of rotated equator (degrees, 0 for geographic)

**Mercator (igtype=5):**
- grid[1]: Western boundary (degrees, longitude for x=1)
- grid[2]: Southern boundary (degrees, latitude for y=1)
- grid[3]: X (longitude) increment (km)
- grid[4]: Y (latitude) increment (km)
- grid[5]: Reference (construction) latitude (degrees)
- grid[6]: Not used

**Lambert (igtype=6):**
- grid[1]: West (degrees, longitude for x=1, y=1)
- grid[2]: South (degrees, latitude for x=1, y=1)
- grid[3]: X (easting) increment (km)
- grid[4]: Y (northing) increment (km)
- grid[5]: Reference longitude (degrees)
- grid[6]: Reference (tangent) latitude (degrees)

# Returns
- `xm`: Map ratio in x direction (nx×ny matrix, if imapr>0)
- `ym`: Map ratio in y direction (nx×ny matrix, if imapr>0)
- `fc`: Coriolis parameter (nx×ny matrix, if icori>0)
- `hx`: Grid resolution in meters in x direction (at map ratio = 1)
- `hy`: Grid resolution in meters in y direction (at map ratio = 1)
- `ierror`: Error status (0=success, 1=bad grid value, 2=unknown grid type, 3=conversion error)

# Notes
To avoid division by zero in calling routines, some incorrect values may be returned:
1. Coriolis parameter: minimum value computed 1/100 grid unit from equator (correct value is 0)
2. Map ratio (geographic/rotated): minimum value computed 1/100 grid unit from pole (correct is infinite)

Example usage with imapr=1:
```julia
x_gradient = xm[i,j] * (field[i+1,j] - field[i-1,j]) / (hx * 2.0)
y_gradient = ym[i,j] * (field[i,j+1] - field[i,j-1]) / (hy * 2.0)
```

# Based on
Map projection code by Anstein Foss (1995-1996)
"""
function mapfield(imapr::Int, icori::Int, igtype::Int, grid::Vector{Float64},
                  nx::Int, ny::Int)
    @assert length(grid) == 6 "grid must have 6 elements"

    # Maximum block size for coordinate transformations
    NMAX = 1000

    ierror = 0

    # Validate grid parameters
    if (igtype == 1 || igtype == 4) && (grid[3] == 0.0 || grid[5] == 0.0)
        ierror = 1
    end
    if (igtype == 2 || igtype == 3) && (grid[3] == 0.0 || grid[4] == 0.0)
        ierror = 1
    end
    if igtype == 5 && (grid[3] == 0.0 || grid[4] == 0.0)
        ierror = 1
    end
    if igtype == 5 && grid[6] != 0.0
        ierror = 1
    end

    if ierror != 0
        return (nothing, nothing, nothing, 0.0, 0.0, ierror)
    end

    # Initialize output arrays
    xm = imapr > 0 ? zeros(Float64, nx, ny) : nothing
    ym = imapr > 0 ? zeros(Float64, nx, ny) : nothing
    fc = icori > 0 ? zeros(Float64, nx, ny) : nothing

    # Earth radius (m)
    rearth = earthr()

    # Constants
    zpir18 = π / 180.0
    zfc = 2.0 * 0.7292e-4  # Coriolis coefficient (2 * Earth's angular velocity)

    hx = 0.0
    hy = 0.0

    # ===== Polar stereographic grid =====
    if igtype == 1 || igtype == 4
        xp = grid[1]
        yp = grid[2]
        an = grid[3]
        fi = grid[4]
        fp = grid[5]
        fq = 1.0 + sin(fp * zpir18)

        # Resolution
        dh = rearth * fq / an
        an2 = an * an
        xm1 = fq * 0.5 / an2

        # Avoiding problems in calling routines (1/fc), coriolis at equator
        rpol2_min = (an - 0.01) * (an - 0.01)
        fc00 = zfc * (an2 - rpol2_min) / (an2 + rpol2_min)

        if imapr > 0
            # Map ratio (possibly divided by n*dh)
            s = xm1
            if imapr > 1
                s = s / (dh * Float64(imapr - 1))
            end

            for j in 1:ny
                for i in 1:nx
                    rpol2 = (i - xp)^2 + (j - yp)^2
                    xm[i, j] = s * (an2 + rpol2)
                    ym[i, j] = xm[i, j]
                end
            end
        end

        if icori > 0
            # Coriolis parameter
            for j in 1:ny
                for i in 1:nx
                    rpol2 = (i - xp)^2 + (j - yp)^2
                    fc[i, j] = zfc * (an2 - rpol2) / (an2 + rpol2)
                    fc[i, j] = max(fc[i, j], fc00)
                end
            end
        end

        hx = dh
        hy = dh

    # ===== Geographic or spherical rotated grid =====
    elseif igtype == 2 || igtype == 3
        west = grid[1] * zpir18
        south = grid[2] * zpir18
        dlon = grid[3] * zpir18
        dlat = grid[4] * zpir18
        xcen = grid[5] * zpir18
        ycen = grid[6] * zpir18
        hlon = rearth * dlon
        hlat = rearth * dlat

        sx = 1.0
        sy = 1.0
        if imapr > 1
            sx = sx / (hlon * Float64(imapr - 1))
            sy = sy / (hlat * Float64(imapr - 1))
        end

        # Avoiding problems at poles and equator
        clat90 = cos((90.0 - 0.01 * abs(grid[4])) * zpir18)
        slat00 = sin((0.0 + 0.01 * abs(grid[4])) * zpir18)

        # Process in blocks for efficiency
        for j in 1:ny
            i1 = 1
            while i1 <= nx
                i0 = i1 - 1
                i2 = min(i0 + NMAX, nx)
                nblock = i2 - i0

                # Prepare coordinate arrays
                flon = [west + (i - 1) * dlon for i in i1:i2]
                flat = [south + (j - 1) * dlat for _ in i1:i2]

                if imapr > 0
                    # Map ratio (possibly divided by n*hlon and n*hlat)
                    for i in i1:i2
                        idx = i - i0
                        clat = cos(flat[idx])
                        clat = max(clat, clat90)
                        xm[i, j] = sx / clat
                        ym[i, j] = sy
                    end
                end

                if icori > 0
                    # Coriolis parameter
                    if xcen != 0.0 || ycen != 0.0
                        ierror = sph2rot!(-1, flon, flat, xcen, ycen)
                        if ierror != 0
                            ierror = 3
                            return (xm, ym, fc, hx, hy, ierror)
                        end
                    end

                    for i in i1:i2
                        idx = i - i0
                        slat = sin(flat[idx])
                        if abs(slat) < slat00
                            slat = slat >= 0.0 ? slat00 : -slat00
                        end
                        fc[i, j] = zfc * slat
                    end
                end

                i1 += NMAX
            end
        end

        hx = hlon
        hy = hlat

    # ===== Mercator or Lambert grid =====
    elseif igtype == 5 || igtype == 6
        xw = grid[1] * zpir18
        ys = grid[2] * zpir18
        dx = grid[3] * 1000.0
        dy = grid[4] * 1000.0

        if igtype == 5
            yc = grid[5] * zpir18
            xc = 0.0  # Not used for mercator
        else
            xc = grid[5] * zpir18
            yc = grid[6] * zpir18
        end

        sx = cos(yc)
        sy = sx
        if imapr > 1
            sx = sx / (dx * Float64(imapr - 1))
            sy = sy / (dy * Float64(imapr - 1))
        end

        # Avoiding problems at poles and equator
        clat90 = cos(89.999 * zpir18)
        slat00 = sin(0.001 * zpir18)

        # Process in blocks
        for j in 1:ny
            i1 = 1
            while i1 <= nx
                i0 = i1 - 1
                i2 = min(i0 + NMAX, nx)
                nblock = i2 - i0

                # Prepare grid position arrays
                flon = Float64[Float64(i) for i in i1:i2]
                flat = Float64[Float64(j) for _ in i1:i2]

                # Convert to spherical coordinates
                if igtype == 5
                    ierror = mer2sph!(1, flon, flat, xw, ys, dx, dy, yc)
                else
                    ierror = lam2sph!(1, flon, flat, xw, ys, dx, dy, xc, yc, yc)
                end

                if ierror != 0
                    ierror = 3
                    return (xm, ym, fc, hx, hy, ierror)
                end

                if imapr > 0
                    # Map ratio (possibly divided by n*dx and n*dy)
                    for i in i1:i2
                        idx = i - i0
                        clat = cos(flat[idx])
                        clat = max(clat, clat90)
                        xm[i, j] = sx / clat
                        ym[i, j] = sy / clat
                    end
                end

                if icori > 0
                    # Coriolis parameter
                    for i in i1:i2
                        idx = i - i0
                        slat = sin(flat[idx])
                        if abs(slat) < slat00
                            slat = slat >= 0.0 ? slat00 : -slat00
                        end
                        fc[i, j] = zfc * slat
                    end
                end

                i1 += NMAX
            end
        end

        hx = dx
        hy = dy

    else
        ierror = 2
    end

    return (xm, ym, fc, hx, hy, ierror)
end
