#!/usr/bin/env julia
# Smoky Mushroom Cloud Release Geometry Visualisation
# =====================================================
# Produces a publication-quality 3D rendering of the CMA-ES-inferred
# mushroom cloud for the Plumbbob Smoky test (44 kT tower shot, 31 Aug 1957).
#
# Unlike the Nancy figure (which uses fixed NOAA three-layer boundaries),
# Smoky's layer heights were optimised by BIPOP-CMA-ES and represent the
# model-inferred cloud geometry.
#
# Best-fit layer heights (AGL):
#   Stem top:   1,719 m   (CMA-ES)   vs DASA-1251: 1,822 m
#   Cap mid:    7,393 m   (CMA-ES)   vs DASA-1251: 5,541 m
#   Cloud top: 10,367 m   (CMA-ES)   vs DASA-1251: 9,259 m
#
# Layer fractions (normalised):
#   Lower  (0–1,719 m):          16.9%
#   Middle (1,719–7,393 m):       6.5%
#   Upper  (7,393–10,367 m):     76.6%
#
# Usage:
#   julia --project=../.. examples/smoky_example/smoky_mushroom_cloud_geometry.jl

using NuclearDetonation
using NuclearDetonation.Transport
using CairoMakie
using Random

# --- CMA-ES optimised cloud geometry (from smoky_cmaes_ou_best.txt) ---
stem_top_m   = 1719.454
cap_mid_m    = 7393.211
cloud_top_m  = 10367.049

# Layer radii (from smoky_cmaes_particle_size.jl rho_core)
stem_r_m     = 0.2  * stem_top_m                    # 343.9 m
mid_r_m      = 0.25 * (cap_mid_m - stem_top_m)      # 1418.4 m
upper_r_m    = 0.25 * (cloud_top_m - cap_mid_m)     # 743.5 m

# Convert to km
stem_top_km  = stem_top_m / 1000.0
cap_mid_km   = cap_mid_m / 1000.0
cloud_top_km = cloud_top_m / 1000.0
stem_r_km    = stem_r_m / 1000.0
mid_r_km     = mid_r_m / 1000.0
upper_r_km   = upper_r_m / 1000.0

# Layer fractions (normalised from CMA-ES frac_lower=0.169, frac_middle=0.065)
frac_lower_raw  = 0.16885
frac_middle_raw = 0.06497
frac_upper_raw  = max(1.0 - frac_lower_raw - frac_middle_raw, 0.05)
ft = frac_lower_raw + frac_middle_raw + frac_upper_raw
frac_lower  = frac_lower_raw / ft
frac_middle = frac_middle_raw / ft
frac_upper  = frac_upper_raw / ft

layer_bounds_km = [(0.0, stem_top_km), (stem_top_km, cap_mid_km), (cap_mid_km, cloud_top_km)]
layer_names = ["Lower (stem)", "Middle (cap)", "Upper (cap)"]
layer_fracs = [frac_lower, frac_middle, frac_upper]

println("Smoky mushroom cloud (44 kT, CMA-ES inferred):")
println("  Stem top:   $(round(stem_top_km, digits=2)) km, r=$(round(stem_r_km, digits=3)) km")
println("  Cap mid:    $(round(cap_mid_km, digits=2)) km, r=$(round(mid_r_km, digits=3)) km")
println("  Cloud top:  $(round(cloud_top_km, digits=2)) km, r=$(round(upper_r_km, digits=3)) km")
println("\nLayer fractions:")
for (name, (lo, hi), frac) in zip(layer_names, layer_bounds_km, layer_fracs)
    println("  $(name) ($(round(lo, digits=1))–$(round(hi, digits=1)) km): $(round(frac*100, digits=1))%")
end

# --- Also compute Glasstone & Dolan empirical cloud for comparison ---
yield_kt = 44.0
hob_m = 213.0  # 700-ft tower
gd_cloud = create_mushroom_cloud_from_yield(yield_kt, hob_m)
gd_cylinders = compute_release_cylinders(gd_cloud)
gd_stem_h_km = gd_cloud.stem_height / 1000.0
gd_cap_h_km  = gd_cloud.cap_height / 1000.0
gd_stem_r_km = gd_cloud.stem_radius / 1000.0
gd_cap_r_km  = gd_cloud.cap_radius / 1000.0

println("\nGlasstone & Dolan empirical (for reference):")
println("  Stem top: $(round(gd_stem_h_km, digits=2)) km, r=$(round(gd_stem_r_km, digits=3)) km")
println("  Cap top:  $(round(gd_cap_h_km, digits=2)) km, r=$(round(gd_cap_r_km, digits=3)) km")

# --- Generate particles distributed by CMA-ES layer fractions ---
# Use G&D cloud shape for radii (realistic mushroom) but CMA-ES layer
# boundaries and mass fractions for the vertical distribution.
rng = Random.default_rng()
Random.seed!(rng, 42)
n_particles = 20_000

particles = NTuple{3,Float64}[]  # (x, y, z) in km

for (i, ((z_lo, z_hi), frac)) in enumerate(zip(layer_bounds_km, layer_fracs))
    n_layer = round(Int, n_particles * frac)
    for _ in 1:n_layer
        z = z_lo + rand(rng) * (z_hi - z_lo)
        # Use G&D cloud shape: stem radius below stem top, cap radius above
        z_m = z * 1000.0
        if z_m <= gd_cloud.stem_height
            r_max = gd_stem_r_km
        else
            r_max = gd_cap_r_km
        end
        r = r_max * sqrt(rand(rng))
        θ = 2π * rand(rng)
        push!(particles, (r * cos(θ), r * sin(θ), z))
    end
end

println("\nGenerated $(length(particles)) particles")

# =============================================================================
# Figure: 3 panels — 3D side view, 3D ¾ view, vertical cross-section
# =============================================================================
fig = Figure(size=(1800, 650), fontsize=13)

# --- Helper: draw wireframe cylinder on Axis3 ---
function draw_cylinder!(ax, r_km, z_lo_km, z_hi_km; color=:gray, linewidth=0.6)
    θ = range(0, 2π, length=36)
    for z in range(z_lo_km, z_hi_km, length=8)
        lines!(ax, r_km .* cos.(θ), r_km .* sin.(θ), fill(z, length(θ)),
            color=color, linewidth=linewidth)
    end
    zz = [z_lo_km, z_hi_km]
    for θi in range(0, 2π, length=12)[1:end-1]
        lines!(ax, fill(r_km * cos(θi), 2), fill(r_km * sin(θi), 2), zz,
            color=color, linewidth=linewidth)
    end
end

# --- Helper: draw layer boundary ring ---
function draw_layer_band!(ax, z_km, r_extent; color=(:steelblue, 0.12))
    θ = range(0, 2π, length=40)
    xs = r_extent .* cos.(θ)
    ys = r_extent .* sin.(θ)
    zs = fill(z_km, length(θ))
    lines!(ax, xs, ys, zs, color=color, linewidth=1.5, linestyle=:dot)
end

# Particle arrays
px = [p[1] for p in particles]
py = [p[2] for p in particles]
pz = [p[3] for p in particles]

max_r = gd_cap_r_km * 1.3

# --- Panel 1: Side view ---
ax1 = Axis3(fig[1, 1],
    xlabel="X (km)", ylabel="Y (km)", zlabel="Altitude (km)",
    title="Side View",
    aspect=(1, 1, 1.2),
    azimuth=0.0, elevation=0.15π,
)

# --- Panel 2: ¾ view ---
ax2 = Axis3(fig[1, 2],
    xlabel="X (km)", ylabel="Y (km)", zlabel="Altitude (km)",
    title="¾ View",
    aspect=(1, 1, 1.2),
    azimuth=0.65π, elevation=0.22π,
)

for ax in [ax1, ax2]
    # CMA-ES layer boundaries
    for (z_lo, z_hi) in layer_bounds_km
        draw_layer_band!(ax, z_lo, max_r, color=(:gray60, 0.3))
    end
    draw_layer_band!(ax, cloud_top_km, max_r, color=(:gray60, 0.3))

    # Wireframe: G&D empirical cloud shape (primary outline)
    draw_cylinder!(ax, gd_stem_r_km, 0.0, gd_stem_h_km, color=(:white, 0.5), linewidth=0.8)
    draw_cylinder!(ax, gd_cap_r_km, gd_stem_h_km, gd_cap_h_km, color=(:cyan, 0.5), linewidth=0.8)

    # Particles coloured by altitude
    scatter!(ax, px, py, pz,
        markersize=1.5,
        color=pz,
        colormap=:hot,
        colorrange=(0.0, cloud_top_km),
    )

    # Ground zero
    scatter!(ax, [0.0], [0.0], [0.0],
        markersize=12, color=:yellow, marker=:star5,
        strokewidth=1.5, strokecolor=:black)

    xlims!(ax, -max_r, max_r)
    ylims!(ax, -max_r, max_r)
    zlims!(ax, 0, cloud_top_km * 1.1)
end

# --- Panel 3: Vertical cross-section with KDE concentration ---
ax3 = Axis(fig[1, 3],
    xlabel="Radial distance (km)",
    ylabel="Altitude (km)",
    title="Vertical Cross-Section",
    aspect=DataAspect(),
)

# Compute 2D KDE concentration field
nx, nz = 140, 120
x_grid = range(-max_r, max_r, length=nx)
z_grid = range(0, cloud_top_km * 1.15, length=nz)
concentration = zeros(Float64, nx, nz)
σ = 0.25  # km

println("Computing KDE concentration field...")

for (ppx, ppy, ppz) in particles
    pr = sqrt(ppx^2 + ppy^2) * sign(ppx)
    ix_lo = max(1, searchsortedfirst(x_grid, pr - 3σ))
    ix_hi = min(nx, searchsortedlast(x_grid, pr + 3σ))
    iz_lo = max(1, searchsortedfirst(z_grid, ppz - 3σ))
    iz_hi = min(nz, searchsortedlast(z_grid, ppz + 3σ))

    for ix in ix_lo:ix_hi
        for iz in iz_lo:iz_hi
            d2 = (x_grid[ix] - pr)^2 + (z_grid[iz] - ppz)^2
            concentration[ix, iz] += exp(-d2 / (2σ^2))
        end
    end
end
concentration ./= maximum(concentration)

# Heatmap
hm = heatmap!(ax3, collect(x_grid), collect(z_grid), concentration,
    colormap=:hot, colorrange=(0.0, 1.0))

Colorbar(fig[1, 4], hm, label="Normalised\nconcentration", width=15)

# G&D cloud outline (primary)
lines!(ax3,
    [-gd_stem_r_km, -gd_stem_r_km, gd_stem_r_km, gd_stem_r_km, -gd_stem_r_km],
    [0.0, gd_stem_h_km, gd_stem_h_km, 0.0, 0.0],
    color=:white, linewidth=2, linestyle=:dash)
lines!(ax3,
    [-gd_cap_r_km, -gd_cap_r_km, gd_cap_r_km, gd_cap_r_km, -gd_cap_r_km],
    [gd_stem_h_km, gd_cap_h_km, gd_cap_h_km, gd_stem_h_km, gd_stem_h_km],
    color=:cyan, linewidth=2, linestyle=:dash)

# Layer boundary lines
layer_colours = [:steelblue, :forestgreen, :firebrick]
for (i, (z_lo, _)) in enumerate(layer_bounds_km)
    hlines!(ax3, [z_lo], color=(layer_colours[i], 0.6), linewidth=1.2, linestyle=:dot)
end
hlines!(ax3, [cloud_top_km], color=(layer_colours[3], 0.6), linewidth=1.2, linestyle=:dot)

# Layer fraction labels
for (i, ((z_lo, z_hi), frac)) in enumerate(zip(layer_bounds_km, layer_fracs))
    mid = (z_lo + min(z_hi, cloud_top_km * 1.15)) / 2
    pct = round(frac * 100, digits=1)
    text!(ax3, -max_r * 0.95, mid,
        text="$(layer_names[i]) $(pct)%",
        color=layer_colours[i], fontsize=11, align=(:left, :center), font=:bold)
end

# Dimension labels — CMA-ES inferred layer heights
label_x = gd_cap_r_km + 0.15
text!(ax3, label_x, cloud_top_km,
    text="Cloud top $(round(cloud_top_km, digits=1)) km",
    color=:cyan, fontsize=11, align=(:left, :center))
text!(ax3, label_x, cap_mid_km,
    text="Cap mid $(round(cap_mid_km, digits=1)) km",
    color=:cyan, fontsize=11, align=(:left, :center))
text!(ax3, label_x, stem_top_km,
    text="Stem top $(round(stem_top_km, digits=1)) km",
    color=:white, fontsize=11, align=(:left, :center))
text!(ax3, label_x, gd_cap_h_km,
    text="G&D cap $(round(gd_cap_h_km, digits=1)) km",
    color=(:gray50, 0.7), fontsize=9, align=(:left, :center))

# Ground zero
scatter!(ax3, [0.0], [0.0],
    markersize=12, color=:yellow, marker=:star5,
    strokewidth=1.5, strokecolor=:black)

xlims!(ax3, -max_r, max_r * 1.6)
ylims!(ax3, 0, cloud_top_km * 1.15)

# Supertitle
Label(fig[0, :], "CMA-ES Inferred Cloud Geometry: Smoky (44 kT, 213 m tower)",
    fontsize=20, font=:bold)

outfile = joinpath(@__DIR__, "smoky_mushroom_cloud.png")
save(outfile, fig, px_per_unit=2)
println("Saved: $(outfile)")

outfile2 = joinpath(@__DIR__, "..", "nancy_fms_plots", "smoky_mushroom_cloud.png")
save(outfile2, fig, px_per_unit=2)
println("Saved: $(outfile2)")
