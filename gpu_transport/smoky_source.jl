# Smoky 23-parameter release geometry (Phase D).
# ================================================
# Smoky adds three geometric parameters on top of Nancy's 20:
#   stem_top_m, cap_mid_m, cloud_top_m
# These define a layered cylinder release whose vertical structure differs
# from Nancy's NOAA 1984 fixed-altitude bands. Everything else (turbulence,
# size distribution, scoring) is shared with Nancy via GpuTransport.
#
# This file is a stub — wire up against the upstream
# `smoky_cmaes_particle_size.jl` once Nancy validation lands.

using Random

# Smoky test site (Plumbbob/Smoky, 31 August 1957)
const SMOKY_LON = -116.0533f0
const SMOKY_LAT = 37.2589f0

struct SmokyParams
    d_fine::Float32; sg_fine::Float32
    d_coarse::Float32; sg_coarse::Float32
    frac_fine::Float32
    frac_lower::Float32; frac_middle::Float32
    sigma_h_scale::Float32; sigma_w_scale::Float32; tl_scale::Float32
    vd_scale::Float32; vgrav_scale::Float32; omega_scale::Float32
    surface_height_scale::Float32
    activity_Bq::Float32
    # Geometry (Smoky-only)
    stem_top_m::Float32
    cap_mid_m::Float32
    cloud_top_m::Float32
    cap_radius_m::Float32
end

function generate_smoky_particles(rng::AbstractRNG, n::Int, p::SmokyParams)
    bins = generate_bimodal_bins_f32(p.d_fine, p.sg_fine, p.d_coarse, p.sg_coarse)
    weights = compute_bimodal_weights_f32(p.d_fine, p.sg_fine,
                                          p.d_coarse, p.sg_coarse, p.frac_fine, bins)
    cum_w = cumsum(weights)

    n_lower  = max(round(Int, n * p.frac_lower), 1)
    n_middle = max(round(Int, n * p.frac_middle), 1)
    n_upper  = max(n - n_lower - n_middle, 1)
    activity_per = Float32(p.activity_Bq) / Float32(n)

    lons = Vector{Float32}(undef, n)
    lats = Vector{Float32}(undef, n)
    hgts = Vector{Float32}(undef, n)
    grv  = Vector{Float32}(undef, n)
    mass = Vector{Float32}(undef, n)
    bidx = Vector{Int32}(undef, n)

    function draw_layer!(i_start, n_layer, lo, hi, radius)
        for k in 1:n_layer
            i = i_start + k - 1
            ru = rand(rng, Float32); θu = rand(rng, Float32); zu = rand(rng, Float32)
            r = radius * sqrt(ru); θ = 2f0 * Float32(π) * θu
            dxm = r * cos(θ); dym = r * sin(θ)
            lons[i] = SMOKY_LON + dxm / (111_000.0f0 * cosd(SMOKY_LAT))
            lats[i] = SMOKY_LAT + dym / 111_000.0f0
            hgts[i] = lo + (hi - lo) * zu
            mass[i] = activity_per
        end
    end
    draw_layer!(1,                          n_lower,  0.0f0,        p.stem_top_m, 537.0f0)
    draw_layer!(1 + n_lower,                n_middle, p.stem_top_m, p.cap_mid_m,   p.cap_radius_m * 0.6f0)
    draw_layer!(1 + n_lower + n_middle,     n_upper,  p.cap_mid_m,  p.cloud_top_m, p.cap_radius_m)

    for i in 1:n
        r = rand(rng, Float32)
        idx = clamp(searchsortedfirst(cum_w, r), 1, length(bins))
        bidx[i] = Int32(idx)
        grv[i]  = bins[idx].v * 0.01f0 * Float32(p.vgrav_scale)
    end

    return ParticleHost(lons, lats, hgts, grv, mass, bidx)
end
