# SNAP: Severe Nuclear Accident Programme
# Julia port of compheightML.f90
# Original Copyright (C) 1992-2023 Norwegian Meteorological Institute
#
# Compute height of model levels and thickness of model layers from hybrid coordinates

function _compute_model_heights!(
    fields::MeteoFields{T},
    ps::AbstractArray{T,2},
    temperature::AbstractArray{T,3},
    hlevel::AbstractArray{T,3},
    hlayer::AbstractArray{T,3},
    alevel::AbstractVector{T},
    blevel::AbstractVector{T},
    ahalf::AbstractVector{T},
    bhalf::AbstractVector{T},
    ::Type{ERA5Format}
) where T
    nx, ny, nk = fields.nx, fields.ny, fields.nk

    # Physical constants (MUST match Fortran snaptabML.f90:32-33)
    g = T(9.81)         # Gravitational acceleration (m/s²) - Fortran value
    ginv = T(1.0) / g
    p0 = T(1000.0)      # Reference pressure (hPa)
    cp = T(1004.0)      # Specific heat at constant pressure (J/kg·K)
    r = T(287.0)        # Gas constant for dry air (J/kg·K)
    rcp = r / cp        # R/cp = 287/1004 ≈ 0.285856 (exact, not rounded!)
    # NOTE: ahalf/alevel are already in hPa (converted when reading ERA5 data)
    # Do NOT apply Pa→hPa conversion here (that would be a double conversion!)

    # Exner function: π = cp * (p/p₀)^(R/cp) (matches Fortran snaptabML.f90:71)
    # NOTE: @fastmath provides significant speedup for the power operation
    @inline @fastmath exner(p::T) = cp * (p / p0)^rcp

    # Temporary 2D arrays to hold the height and Exner pressure at the current lower interface
    # during the upward integration.
    hlev_temp = zeros(T, nx, ny)  # Initialized to 0 (surface height)
    pihl_temp = zeros(T, nx, ny)

    # Initialize with values at the surface - MATCH FORTRAN EXACTLY!
    # Fortran: pihl(:,:) = exner(ps2)
    for j in 1:ny, i in 1:nx
        pihl_temp[i, j] = exner(ps[i, j])
    end

    # Initialize surface boundary values
    hlev_temp[:, :] .= zero(T)
    hlayer[:, :, nk] .= T(9999.0)
    hlevel[:, :, 1] .= zero(T)

    # Integrate from the bottom-up - MATCH FORTRAN LOOP EXACTLY!
    # Fortran: do k=2,nk
    # After reversal, Julia's coefficients are: alevel[1]=surface, alevel[nk]=TOA
    # So we loop k=2 to nk just like Fortran!
    for k in 2:nk
        for j in 1:ny
            for i in 1:nx
                # Pressure at the upper interface - USE k DIRECTLY like Fortran
                # CRITICAL: At k=nk (TOA layer), ahalf[nk] is just an average,
                # we need ahalf[nk+1] which is the actual TOA boundary!
                if k == nk
                    p_half = ahalf[nk+1] + bhalf[nk+1] * ps[i, j]
                else
                    p_half = ahalf[k] + bhalf[k] * ps[i, j]
                end
                pih = exner(p_half)

                # Pressure at the midpoint - USE k DIRECTLY like Fortran
                p_full = alevel[k] + blevel[k] * ps[i, j]
                pif = exner(p_full)

                # h1 is the height of the lower interface (from previous iteration or surface)
                h1 = hlev_temp[i, j]

                # h2 is the height of the upper interface via hypsometric equation
                h2 = h1 + temperature[i, j, k] * (pihl_temp[i, j] - pih) * ginv

                # DEBUG: Disabled for performance - uncomment when debugging height computation
                # if i == 1 && j == 1 && (k <= 4 || k >= nk-20 || k % 10 == 0 || k == nk)
                #     println("JULIA_DEBUG compheight: k=$k")
                #     println("  ahalf[k]=$(ahalf[k]) hPa, bhalf[k]=$(bhalf[k])")
                #     println("  alevel[k]=$(alevel[k]) hPa, blevel[k]=$(blevel[k])")
                #     println("  ps[1,1]=$(ps[i,j]) hPa")
                #     println("  p_half=$p_half hPa, p_full=$p_full hPa")
                #     println("  T[1,1,k]=$(temperature[i,j,k]) K")
                #     println("  h1=$h1 m, h2=$h2 m, dz=$(h2-h1) m")
                #     println("  hlevel[1,1,k]=$(hlevel[i,j,k]) m, hlayer[1,1,k-1]=$(hlayer[i,j,k-1]) m")
                #     println()
                # end

                # Store the thickness - MATCH FORTRAN: hlayer(i,j,k-1) = h2-h1
                hlayer[i, j, k-1] = h2 - h1

                # Calculate and store the height of the midpoint - MATCH FORTRAN
                denominator = pihl_temp[i, j] - pih
                if abs(denominator) < 1e-9
                    hlevel[i, j, k] = h1 + (h2 - h1) / 2
                else
                    hlevel[i, j, k] = h1 + (h2 - h1) * (pihl_temp[i, j] - pif) / denominator
                end

                # Update for next layer up
                hlev_temp[i, j] = h2
                pihl_temp[i, j] = pih
            end
        end
    end

    # Keep hlevel/hlayer in bottom→top (surface→TOA) order here.
    # The interpolation builder (create_wind_interpolants) will reorder slices to
    # match the ascending sigma grid via `level_perm`, avoiding double reversal.

    return nothing
end

# GFS version (from Oct 29, commit 90dc9a3)
function _compute_model_heights!(
    fields::MeteoFields{T},
    ps::AbstractArray{T,2},
    temperature::AbstractArray{T,3},
    hlevel::AbstractArray{T,3},
    hlayer::AbstractArray{T,3},
    alevel::AbstractVector{T},
    blevel::AbstractVector{T},
    ahalf::AbstractVector{T},
    bhalf::AbstractVector{T},
    ::Type{GFSFormat}
) where T
    nx, ny, nk = fields.nx, fields.ny, fields.nk

    # Physical constants (MUST match Fortran snaptabML.f90:32-33)
    g = T(9.81)         # Gravitational acceleration (m/s²) - Fortran value
    ginv = T(1.0) / g
    p0 = T(1000.0)      # Reference pressure (hPa)
    cp = T(1004.0)      # Specific heat at constant pressure (J/kg·K)
    r = T(287.0)        # Gas constant for dry air (J/kg·K)
    rcp = r / cp        # R/cp = 287/1004 ≈ 0.285856 (exact, not rounded!)
    inv_100 = T(0.01)   # Conversion from Pa → hPa for hybrid A-coefficients

    # Exner function: π = cp * (p/p₀)^(R/cp) (matches Fortran snaptabML.f90:71)
    # NOTE: @fastmath provides significant speedup for the power operation
    @inline @fastmath exner(p::T) = cp * (p / p0)^rcp

    # Temporary 2D arrays to hold the height and Exner pressure at the current lower interface
    # during the upward integration.
    hlev_temp = zeros(T, nx, ny)  # Initialized to 0 (surface height)
    pihl_temp = zeros(T, nx, ny)

    # Initialize with values at the surface (the k=nk+1 interface)
    for j in 1:ny, i in 1:nx
        # Pressure at surface is ps (already in hPa)
        p_surf = ahalf[nk+1] * inv_100 + bhalf[nk+1] * ps[i, j]
        pihl_temp[i, j] = exner(p_surf)
    end

    # Integrate from the bottom-up. Model levels are 1=top, nk=bottom.
    for k in nk:-1:1
        for j in 1:ny
            for i in 1:nx
                # Pressure at the upper interface of the current layer k (hPa)
                p_half = ahalf[k] * inv_100 + bhalf[k] * ps[i, j]
                pih = exner(p_half)

                # Pressure at the midpoint of the current layer k
                p_full = alevel[k] * inv_100 + blevel[k] * ps[i, j]
                pif = exner(p_full)

                # h1 is the height of the lower interface (from the previous iteration, or surface)
                h1 = hlev_temp[i, j]

                # h2 is the height of the upper interface, calculated via the hypsometric equation
                # We use the temperature of the current layer k for the integration
                h2 = h1 + temperature[i, j, k] * (pihl_temp[i, j] - pih) * ginv

                # DEBUG: Log temperature for first grid point, layers near particle
                if i == 1 && j == 1 && k >= 5 && k <= 8
                    p_half_val = ahalf[k] * inv_100 + bhalf[k] * ps[i, j]
                    sigma_approx = (ahalf[k] + bhalf[k] * ps[i, j] * 100) / (ps[i, j] * 100)
                    println("DEBUG compheight: i=$i,j=$j,k=$k, T=$(temperature[i,j,k])K, p=$p_half_val hPa, σ≈$sigma_approx, dz=$(h2-h1)m")
                end

                # Store the thickness of the current layer k
                hlayer[i, j, k] = h2 - h1

                # Calculate and store the height of the midpoint of the current layer k
                denominator = pihl_temp[i, j] - pih
                if abs(denominator) < 1e-9 # Avoid division by zero
                    hlevel[i, j, k] = h1 + (h2 - h1) / 2
                else
                    hlevel[i, j, k] = h1 + (h2 - h1) * (pihl_temp[i, j] - pif) / denominator
                end

                # Update the temporary variables for the next layer up (k-1)
                hlev_temp[i, j] = h2
                pihl_temp[i, j] = pih
            end
        end
    end

    return nothing
end

"""
    compute_model_heights!(::Type{F}, fields::MeteoFields{T}, time_level::Int=2) where {F<:MetFormat, T}

Compute geopotential height at layer midpoints with met-format-specific handling.

- `F = ERA5Format`: ERA5 hybrid coordinate handling (top-to-surface indexing, k=2:nk loop)
- `F = GFSFormat`: GFS hybrid coordinate handling (surface-to-top indexing, k=nk:-1:1 loop, Pa→hPa conversion)
- `time_level = 1`: uses `ps1`, `t1` and writes to `hlevel1`, `hlayer1`
- `time_level = 2`: uses `ps2`, `t2` and writes to `hlevel2`, `hlayer2`
"""
function compute_model_heights!(::Type{F}, fields::MeteoFields{T}, time_level::Int=2) where {F<:MetFormat, T}
    if time_level == 1
        _compute_model_heights!(
            fields,
            fields.ps1,
            fields.t1,
            fields.hlevel1,
            fields.hlayer1,
            fields.alevel,
            fields.blevel,
            fields.ahalf,
            fields.bhalf,
            F
        )
    elseif time_level == 2
        _compute_model_heights!(
            fields,
            fields.ps2,
            fields.t2,
            fields.hlevel2,
            fields.hlayer2,
            fields.alevel,
            fields.blevel,
            fields.ahalf,
            fields.bhalf,
            F
        )
    else
        error("compute_model_heights!: time_level must be 1 or 2 (got $time_level)")
    end

    return nothing
end

"""
    compute_model_heights!(fields::MeteoFields{T}, time_level::Int=2) where T

Default version for backwards compatibility (uses ERA5 behavior).
"""
function compute_model_heights!(fields::MeteoFields{T}, time_level::Int=2) where T
    compute_model_heights!(ERA5Format, fields, time_level)
end

"""
    compute_model_heights_simple!(fields::MeteoFields{T}; R=287.0, cp=1005.0) where T

Alternative simplified height computation using ideal gas law.

This is a simpler version that uses the hypsometric equation directly:
```
Δz = (R*T/g) * ln(p₁/p₂)
```

# Arguments
- `fields::MeteoFields`: Meteorological fields container
- `R`: Gas constant for dry air (J/(kg·K)), default 287.0
- `cp`: Specific heat at constant pressure (J/(kg·K)), default 1005.0
"""
function compute_model_heights_simple!(fields::MeteoFields{T};
                                       R=T(287.0),
                                       cp=T(1005.0)) where T
    nx, ny, nk = fields.nx, fields.ny, fields.nk
    g = T(9.80665)

    # Initialize: surface is at z=0
    fields.hlevel2[:, :, 1] .= zero(T)

    # Compute heights layer by layer
    for k in 2:nk
        for j in 1:ny
            for i in 1:nx
                # Pressure at layer interface and midpoint
                p_lower = fields.ahalf[k] + fields.bhalf[k] * fields.ps2[i, j]
                p_upper = k < nk ? fields.ahalf[k+1] + fields.bhalf[k+1] * fields.ps2[i, j] : T(0.0)

                # Layer mean temperature
                T_mean = fields.t2[i, j, k]

                # Hypsometric equation
                if p_upper > 0
                    dz = (R * T_mean / g) * log(p_lower / p_upper)
                else
                    dz = T(9999.0)  # Top layer
                end

                fields.hlayer2[i, j, k-1] = dz
                fields.hlevel2[i, j, k] = fields.hlevel2[i, j, k-1] + dz
            end
        end
    end

    fields.hlayer2[:, :, nk] .= T(9999.0)

    return nothing
end

export compute_model_heights!, compute_model_heights_simple!
