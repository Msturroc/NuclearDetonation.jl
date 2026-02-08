"""
Convert omega (Pa/s) to sigma-dot (dσ/dt) for hybrid coordinates.

This implements the critical conversion from SNAP's om2edot.f90.
Meteorological models provide vertical velocity in pressure coordinates (omega, Pa/s),
but SNAP's transport code requires it in sigma coordinates (dσ/dt).

The conversion is: σ̇ = ω * (dσ/dp)

where dσ/dp is computed from the hybrid coordinate coefficients and surface pressure.
"""
function convert_omega_to_sigmadot!(w_field::Array{T,3},
                                    ps_field::Matrix{T},
                                    ahalf::Vector{T},
                                    bhalf::Vector{T},
                                    vhalf::Vector{T}) where T<:Real
    nx, ny, nk = size(w_field)

    # Convert each model level
    for k in 2:nk
        # Change in sigma/eta across the layer
        deta = vhalf[k-1] - vhalf[k]

        for j in 1:ny
            for i in 1:nx
                omega = w_field[i, j, k]  # Original omega value (Pa/s)

                # Pressure at layer interfaces (Pa)
                # Note: Handle both Pa (GFS) and hPa (ERA5) units for coefficients
                p_upper = ahalf[k] + bhalf[k] * ps_field[i, j]
                p_lower = ahalf[k-1] + bhalf[k-1] * ps_field[i, j]

                # Convert to Pa if needed (heuristic: if ps < 2000, it's likely hPa)
                if ps_field[i, j] < 2000.0
                    dp = (p_lower - p_upper) * 100.0  # hPa -> Pa
                else
                    dp = p_lower - p_upper  # Already in Pa
                end

                if abs(dp) > 1e-6
                    # Convert omega to sigma-dot: σ̇ = ω * (dσ/dp)
                    # Overwrite the w field with converted value
                    w_field[i, j, k] = omega * deta / dp
                else
                    # Avoid division by zero
                    w_field[i, j, k] = zero(T)
                end
            end
        end
    end

    return nothing
end

"""
    compute_etadot_from_continuity!(::ERA5Format, ...)

ERA5-specific vertical velocity computation using continuity equation.
Keeps data in file order (TOA first, surface last) matching Fortran SNAP.
"""
function compute_etadot_from_continuity!(::ERA5Format,
                                         edot::Array{T,3},
                                         u::Array{T,3},
                                         v::Array{T,3},
                                         ps::Matrix{T},
                                         xm::Matrix{T},
                                         ym::Matrix{T},
                                         ahalf::Vector{T},
                                         bhalf::Vector{T},
                                         vhalf::Vector{T},
                                         dx::T,
                                         dy::T;
                                         averaging::Bool=true) where T<:Real
    # Use fields as stored (k=1 = surface) — arrays and coefficients are aligned.
    compute_etadot_from_continuity!(edot, u, v, ps, xm, ym, ahalf, bhalf, vhalf, dx, dy, averaging=averaging)
end

"""
    compute_etadot_from_continuity!(::GFSFormat, ...)

GFS-specific vertical velocity computation.
"""
function compute_etadot_from_continuity!(::GFSFormat,
                                         edot::Array{T,3},
                                         u::Array{T,3},
                                         v::Array{T,3},
                                         ps::Matrix{T},
                                         xm::Matrix{T},
                                         ym::Matrix{T},
                                         ahalf::Vector{T},
                                         bhalf::Vector{T},
                                         vhalf::Vector{T},
                                         dx::T,
                                         dy::T;
                                         averaging::Bool=true) where T<:Real
    compute_etadot_from_continuity!(edot, u, v, ps, xm, ym, ahalf, bhalf, vhalf, dx, dy, averaging=averaging)
end

"""
Compute etadot (sigma-dot) from horizontal winds using continuity equation.

This is a critical fallback when meteorological data lacks vertical velocity.
Implements SNAP's edcomp subroutine from om2edot.f90.

Based on Jan Erik Haugen's Hirlam routine EDCOMP.
"""
function compute_etadot_from_continuity!(edot::Array{T,3},
                                         u::Array{T,3},
                                         v::Array{T,3},
                                         ps::Matrix{T},
                                         xm::Matrix{T},
                                         ym::Matrix{T},
                                         ahalf::Vector{T},
                                         bhalf::Vector{T},
                                         vhalf::Vector{T},
                                         dx::T,
                                         dy::T;
                                         averaging::Bool=true) where T<:Real
    nx, ny, nz = size(edot)

    # Map factor divided by 2*grid_spacing
    d2hx = 1.0 / (dx * 2.0)
    d2hy = 1.0 / (dy * 2.0)
    xmd2h = xm .* d2hx
    ymd2h = ym .* d2hy

    # Temporary work arrays
    uu = zeros(T, nx, ny)
    vv = zeros(T, nx, ny)
    dpsdt = zeros(T, nx, ny)
    edoth = zeros(T, nx, ny)

    # Step 1: Compute surface pressure tendency
    # Vertically integrate mass-weighted winds
    uu .= 0.0
    vv .= 0.0

    for k in 1:nz
        da = ahalf[k] - ahalf[k+1]
        db = bhalf[k] - bhalf[k+1]

        for j in 1:ny
            for i in 1:nx
                dp = da + db * ps[i, j]
                uu[i, j] -= u[i, j, k] * dp
                vv[i, j] -= v[i, j, k] * dp
            end
        end
    end

    # Compute divergence of column-integrated momentum → dps/dt
    for j in 2:ny-1
        for i in 2:nx-1
            dpsdt[i, j] = (uu[i+1, j] - uu[i-1, j]) * xmd2h[i, j] +
                         (vv[i, j+1] - vv[i, j-1]) * ymd2h[i, j]
        end
    end

    # Step 2: Vertical integration of continuity equation
    edoth .= 0.0

    for k in 1:nz-1
        da = ahalf[k] - ahalf[k+1]
        db = bhalf[k] - bhalf[k+1]
        deta = vhalf[k] - vhalf[k+1]

        # Mass-weighted winds at this level
        for j in 1:ny
            for i in 1:nx
                dp = da + db * ps[i, j]
                uu[i, j] = dp * u[i, j, k]
                vv[i, j] = dp * v[i, j, k]
            end
        end

        # Integrate continuity equation
        for j in 2:ny-1
            for i in 2:nx-1
                # Horizontal divergence of mass flux
                div = (uu[i+1, j] - uu[i-1, j]) * xmd2h[i, j] +
                     (vv[i, j+1] - vv[i, j-1]) * ymd2h[i, j]

                # etadot at half-level below full level k
                edothu = edoth[i, j]

                # etadot at half-level above full level k
                edoth[i, j] = edoth[i, j] + db * dpsdt[i, j] + div

                # etadot at full level k (mean of half-levels)
                etadot = (edothu + edoth[i, j]) * 0.5

                # Convert to sigma space: etadot * deta/dp
                dp = da + db * ps[i, j]

                # DEBUG: Print at sample point (80, 63) for level 100
                if SNAP_DEBUG_EDCOMP && i == 80 && j == 63 && k == 100
                    println("JULIA EDCOMP DEBUG k=$k i=$i j=$j:")
                    println("  da=$da db=$db deta=$deta")
                    println("  ps[i,j]=$(ps[i,j]) Pa")
                    println("  dp=$dp Pa")
                    println("  div=$div")
                    println("  db*dpsdt[i,j]=$(db * dpsdt[i, j])")
                      println("  edothu=$edothu")
                      println("  edoth[i,j]=$(edoth[i, j])")
                    println("  etadot (before deta/dp)=$etadot")
                    println("  etadot*deta/dp=$(etadot * deta / dp)")
                end

                # CRITICAL: Match Fortran EXACTLY - no dp check!
                # Fortran om2edot.f90:220 does: etadot = etadot*deta/dp
                # without any guard, so Julia must do the same
                etadot = etadot * deta / dp

                # Mean with input value (will be zero for GFS case)
                if averaging
                    edot[i, j, k] = (edot[i, j, k] + etadot) * 0.5
                else
                    edot[i, j, k] = edot[i, j, k] + etadot
                end

                # DEBUG: Print final value
                if SNAP_DEBUG_EDCOMP && i == 80 && j == 63 && k == 100
                    println("  edot[i,j,k] (after averaging=$(averaging))=$(edot[i, j, k])")
                end

                # DEBUG: Also print values at particle 1 location (i=68, j=79)
                # After reversal: particle near surface (sigma~0.989) is at k=4-5, not k=133-134!
                if SNAP_DEBUG_EDCOMP && i == 68 && j == 79 && (k == 4 || k == 5)
                    println("JULIA EDCOMP DEBUG k=$k i=$i j=$j:")
                    println("  === U/V FIELD VALUES ===")
                    println("  u[i,j,k]=$(u[i,j,k]) v[i,j,k]=$(v[i,j,k])")
                    println("  u[i-1,j,k]=$(u[i-1,j,k]) u[i+1,j,k]=$(u[i+1,j,k])")
                    println("  v[i,j-1,k]=$(v[i,j-1,k]) v[i,j+1,k]=$(v[i,j+1,k])")
                    println("  === STENCIL COEFFICIENTS ===")
                    println("  xmd2h[i,j]=$(xmd2h[i,j]) ymd2h[i,j]=$(ymd2h[i,j])")
                    println("  === FLUX VALUES (uu=dp*u, vv=dp*v) ===")
                    println("  uu[i-1,j]=$(uu[i-1,j]) uu[i+1,j]=$(uu[i+1,j])")
                    println("  vv[i,j-1]=$(vv[i,j-1]) vv[i,j+1]=$(vv[i,j+1])")
                    println("  === COMPUTED VALUES ===")
                    println("  SIGMA LEVELS: vhalf[k]=$(vhalf[k]), vhalf[k+1]=$(vhalf[k+1])")
                    println("  HYBRID COEFFS: ahalf[k]=$(ahalf[k]), bhalf[k]=$(bhalf[k])")
                    println("  HYBRID COEFFS: ahalf[k+1]=$(ahalf[k+1]), bhalf[k+1]=$(bhalf[k+1])")
                    println("  da=$da db=$db deta=$deta")
                    println("  ps[i,j]=$(ps[i,j]) Pa")
                    println("  dp=$dp Pa  ← $(dp < 0 ? "NEGATIVE! BUG!" : "positive OK")")
                    println("  div=$div")
                    println("  db*dpsdt[i,j]=$(db * dpsdt[i, j])")
                    println("  edothu=$edothu")
                    println("  edoth[i,j]=$(edoth[i, j])")
                    println("  etadot (before deta/dp)=$etadot")
                    println("  etadot*deta/dp=$(etadot * deta / dp)")
                    println("  edot[i,j,k] (after averaging=$(averaging))=$(edot[i, j, k])")
                end
            end
        end
    end

    # Step 3: Upper-most level
    da = ahalf[nz] - ahalf[nz+1]
    db = bhalf[nz] - bhalf[nz+1]
    deta = vhalf[nz] - vhalf[nz+1]

    for j in 2:ny-1
        for i in 2:nx-1
            dp = da + db * ps[i, j]
            if averaging
                edot[i, j, nz] = (edot[i, j, nz] + 0.5 * edoth[i, j] * deta / dp) * 0.5
            else
                edot[i, j, nz] = edot[i, j, nz] + 0.5 * edoth[i, j] * deta / dp
            end
        end
    end

    # Step 4: Boundary treatment (halve edges)
    for k in 1:nz
        edot[:, 1, k] .*= 0.5
        edot[:, ny, k] .*= 0.5
        edot[1, 2:ny-1, k] .*= 0.5
        edot[nx, 2:ny-1, k] .*= 0.5
    end

    return nothing
end

"""
Compute latitude-dependent map scale factors for geographic (lat/lon) grids.

For a latitude-longitude grid, the physical distance represented by one degree
of longitude varies with latitude due to the convergence of meridians toward the poles.

Map scale factors correct for this distortion:
- xm(i,j) = 1 / cos(latitude) - corrects for longitudinal convergence
- ym(i,j) = 1 - constant for all latitudes

Based on SNAP's mapfield.f90 for igtype=2 (geographic grids).
"""
function compute_map_scale_factors!(xm::Matrix{T},
                                     ym::Matrix{T},
                                     latitude::Vector{T}) where T<:Real
    nx, ny = size(xm)

    # Sanity check
    @assert length(latitude) == ny "Latitude vector length must match grid ny dimension"

    # Compute map factors for each grid point
    for j in 1:ny
        # Convert latitude to radians
        lat_rad = deg2rad(latitude[j])

        # Map scale factor for longitude (corrects for meridian convergence)
        # At the equator: cos(0) = 1, so xm = 1
        # At poles: cos(90°) → 0, so xm → ∞ (grid becomes singular)
        xm_factor = 1.0 / cos(lat_rad)

        for i in 1:nx
            xm[i, j] = T(xm_factor)
            ym[i, j] = one(T)  # Latitude spacing is constant
        end
    end

    return nothing
end
const SNAP_DEBUG_EDCOMP = false
