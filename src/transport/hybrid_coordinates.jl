# SNAP: Severe Nuclear Accident Programme
# Hybrid coordinate utilities for sigma↔height conversions
#
# Provides robust helpers that mirror SNAP's hybrid vertical coordinate handling.
# The helpers build a local vertical column profile from the interpolated
# geopotential heights and offer monotonic conversions between sigma (η) and
# geometric height. NaN values returned by the interpolant are guard-railed by
# propagating the nearest finite neighbour, ensuring we never fall back to a
# crude linear (1-σ)*z_max approximation.

export HybridProfile, hybrid_profile, height_from_sigma, sigma_from_height

const SNAP_DEBUG_HYBRID = false

"""
    HybridProfile

Pre-computed vertical column for a specific (x, y, t) location.

Fields
- `sigma_levels`: Sigma/eta levels (ascending)
- `heights`: Geometric heights (meters) mapped to `sigma_levels`
- `ascending`: `true` if `heights` is monotonically increasing, `false` if decreasing
"""
struct HybridProfile{T<:Real}
    sigma_levels::Vector{T}
    heights::Vector{T}
    ascending::Bool
end

"""
    hybrid_profile(winds::WindFields, x, y, t) -> HybridProfile

Build a vertical column profile for location `(x, y)` at time `t`.
"""
function hybrid_profile(winds::WindFields{T}, x::Real, y::Real, t::Real) where T
    # Clamp horizontal indices to valid grid range to avoid extrapolation artifacts
    xq = clamp(Float64(x), 1.0, Float64(winds.nx))
    yq = clamp(Float64(y), 1.0, Float64(winds.ny))
    σ_levels = Float64.(winds.z_grid)
    n_levels = length(σ_levels)
    heights = Vector{Float64}(undef, n_levels)

    # Sample geopotential heights at each sigma level, guarding against NaNs
    for (idx, σ) in enumerate(winds.z_grid)
        val = winds.h_interp(xq, yq, σ, t)
        if !isfinite(val)
            @warn "Non-finite height value detected in hybrid_profile" xq=xq yq=yq σ=σ t=t val=val
        end
        heights[idx] = Float64(val)
    end

    # DEBUG: Print first few and last few sigma/height pairs
    if SNAP_DEBUG_HYBRID && t < 10.0  # Only at start of simulation
        println("\nDEBUG hybrid_profile at x=$xq, y=$yq, t=$t:")
        println("  First 5 levels:")
        for i in 1:min(5, n_levels)
            println("    σ[$i] = $(σ_levels[i]), height = $(heights[i]) m")
        end
        println("  Last 5 levels:")
        for i in max(1, n_levels-4):n_levels
            println("    σ[$i] = $(σ_levels[i]), height = $(heights[i]) m")
        end
        # Check around σ=0.608
        target_σ = 0.608
        idx = argmin(abs.(σ_levels .- target_σ))
        println("  Nearest to σ=$target_σ:")
        for i in max(1,idx-2):min(n_levels,idx+2)
            println("    σ[$i] = $(σ_levels[i]), height = $(heights[i]) m")
        end
    end

    # Fill NaNs by borrowing the closest finite neighbour
    first_finite = findfirst(isfinite, heights)
    if isnothing(first_finite)
        fill!(heights, 0.0)
    else
        first_idx = first_finite
        # Propagate forward from the first finite value
        last_val = heights[first_idx]
        for idx in first_idx+1:n_levels
            if isfinite(heights[idx])
                last_val = heights[idx]
            else
                heights[idx] = last_val
            end
        end
        # Propagate backwards for leading NaNs
        for idx in first_idx-1:-1:1
            heights[idx] = heights[idx+1]
        end
    end

    # Ensure sigma levels are strictly increasing
    order = sortperm(σ_levels)
    if any(i -> order[i] != i, eachindex(order))
        σ_levels = σ_levels[order]
        heights = heights[order]
    end

    # EXPERIMENTAL: Commenting out aggressive monotonicity enforcement
    # This was creating large plateaus where many sigma values mapped to the same height,
    # causing particles to "teleport" to the surface layer. Testing with raw interpolated data.
    #
    # # Enforce monotonicity so interpolation remains well behaved
    # descending = heights[1] >= heights[end]
    # if descending
    #     for idx in 2:n_levels
    #         if heights[idx] > heights[idx-1]
    #             heights[idx] = heights[idx-1]
    #         end
    #     end
    # else
    #     for idx in 2:n_levels
    #         if heights[idx] < heights[idx-1]
    #             heights[idx] = heights[idx-1]
    #         end
    #     end
    # end

    # Still need to determine if profile is ascending or descending for sigma_from_height
    descending = heights[1] >= heights[end]

    return HybridProfile(σ_levels, heights, !descending)
end

"""
    height_from_sigma(profile::HybridProfile, σ; fallback_height=nothing)

Convert sigma (η) to physical height (meters) using a pre-computed profile.
If interpolation fails, `fallback_height` is returned when provided.
"""
function height_from_sigma(profile::HybridProfile{T},
                           σ::Real;
                           fallback_height::Union{Nothing,Real}=nothing) where T
    σ_levels = profile.sigma_levels
    heights = profile.heights

    σ_min = σ_levels[begin]
    σ_max = σ_levels[end]
    σ_clamped = clamp(Float64(σ), σ_min, σ_max)

    idx_hi = searchsortedfirst(σ_levels, σ_clamped)

    # VERBOSE DEBUG: Track interpolation for extreme sigma values
    verbose_debug = SNAP_DEBUG_HYBRID && (σ < 0.3 || σ > 0.95)

    if idx_hi <= 1
        h = heights[1]
        if verbose_debug
            println("DEBUG height_from_sigma: σ=$σ → idx_hi=$idx_hi (≤1)")
            println("  Returned h=heights[1]=$h m")
        end
    elseif idx_hi > length(σ_levels)
        h = heights[end]
        if verbose_debug
            println("DEBUG height_from_sigma: σ=$σ → idx_hi=$idx_hi (>max)")
            println("  Returned h=heights[end]=$h m")
        end
    else
        idx_lo = idx_hi - 1
        σ_lo = σ_levels[idx_lo]
        σ_hi = σ_levels[idx_hi]
        h_lo = heights[idx_lo]
        h_hi = heights[idx_hi]

        if !isfinite(h_lo) && isfinite(h_hi)
            h_lo = h_hi
        elseif !isfinite(h_hi) && isfinite(h_lo)
            h_hi = h_lo
        end

        if isfinite(h_lo) && isfinite(h_hi) && abs(σ_hi - σ_lo) > eps(Float64)
            ratio = (σ_clamped - σ_lo) / (σ_hi - σ_lo)
            h = h_lo + ratio * (h_hi - h_lo)
            if verbose_debug
                println("DEBUG height_from_sigma: σ=$σ (clamped=$σ_clamped)")
                println("  Interpolating between idx_lo=$idx_lo and idx_hi=$idx_hi")
                println("  σ_lo=$σ_lo → h_lo=$h_lo m")
                println("  σ_hi=$σ_hi → h_hi=$h_hi m")
                println("  ratio=$ratio → h=$h m")
            end
        else
            h = isfinite(h_lo) ? h_lo : h_hi
            if verbose_debug
                println("DEBUG height_from_sigma: σ=$σ → fallback (non-finite)")
                println("  h=$h m")
            end
        end
    end

    if !isfinite(h)
        return fallback_height === nothing ? h : Float64(fallback_height)
    end

    h_min = minimum(heights)
    h_max = maximum(heights)
    h_final = clamp(h, h_min, h_max)

    if verbose_debug && h != h_final
        println("DEBUG height_from_sigma: Clamped $h m → $h_final m (range: $h_min to $h_max)")
    end

    return h_final
end

"""
    sigma_from_height(profile::HybridProfile, height; fallback_sigma=nothing)

Invert the hybrid profile: convert height (meters) to sigma (η).
If interpolation fails, `fallback_sigma` is returned when provided.
"""
function sigma_from_height(profile::HybridProfile{T},
                           height::Real;
                           fallback_sigma::Union{Nothing,Real}=nothing) where T
    σ_levels = profile.sigma_levels
    heights = profile.heights

    if profile.ascending
        h_vec = heights
        σ_vec = σ_levels
    else
        # Use views instead of reverse() to avoid heap allocation —
        # critical for thread safety under heavy GC pressure
        h_vec = @view heights[end:-1:1]
        σ_vec = @view σ_levels[end:-1:1]
    end

    h_min = h_vec[begin]
    h_max = h_vec[end]
    h_target = Float64(height)

    n_levels = length(h_vec)
    if n_levels == 0
        return fallback_sigma === nothing ? NaN : Float64(clamp(fallback_sigma, 0.0, 1.0))
    elseif n_levels == 1
        σ_single = Float64(σ_vec[1])
        return clamp(σ_single, 0.0, 1.0)
    end

    idx_hi::Int = 1
    idx_lo::Int = 1

    if h_target <= h_min
        idx_lo = 1
        idx_hi = 2
    elseif h_target >= h_max
        idx_hi = n_levels
        idx_lo = max(idx_hi - 1, 1)
    else
        idx_hi = searchsortedfirst(h_vec, h_target)
        idx_hi = clamp(idx_hi, 2, n_levels)
        idx_lo = idx_hi - 1
    end

    σ_interp = Float64(σ_vec[idx_lo])
    if idx_hi != idx_lo
        h_lo = Float64(h_vec[idx_lo])
        h_hi = Float64(h_vec[idx_hi])
        σ_lo = Float64(σ_vec[idx_lo])
        σ_hi = Float64(σ_vec[idx_hi])

        if abs(h_hi - h_lo) > eps(Float64)
            ratio = (h_target - h_lo) / (h_hi - h_lo)
            σ_interp = σ_lo + ratio * (σ_hi - σ_lo)
        else
            σ_interp = σ_lo
        end
    end

    if !isfinite(σ_interp)
        σ_interp = fallback_sigma === nothing ? σ_interp : Float64(fallback_sigma)
    end

    return clamp(σ_interp, 0.0, 1.0)
end

"""
    height_from_sigma(winds::WindFields, x, y, σ, t; kwargs...)

Convenience overload that builds the profile on demand.
"""
function height_from_sigma(winds::WindFields, x::Real, y::Real, σ::Real, t::Real;
                           fallback_height::Union{Nothing,Real}=nothing,
                           profile::Union{Nothing,HybridProfile{Float64}}=nothing)
    prof = isnothing(profile) ? hybrid_profile(winds, x, y, t) : profile
    return height_from_sigma(prof, σ; fallback_height=fallback_height)
end

"""
    sigma_from_height(winds::WindFields, x, y, height, t; kwargs...)

Convenience overload that builds the profile on demand.
"""
function sigma_from_height(winds::WindFields, x::Real, y::Real, height::Real, t::Real;
                           fallback_sigma::Union{Nothing,Real}=nothing,
                           profile::Union{Nothing,HybridProfile{Float64}}=nothing)
    prof = isnothing(profile) ? hybrid_profile(winds, x, y, t) : profile
    return sigma_from_height(prof, height; fallback_sigma=fallback_sigma)
end
