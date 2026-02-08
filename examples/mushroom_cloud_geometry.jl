# Mushroom Cloud Release Geometry Visualisation
# ===============================================
# Produces a publication-quality 3D rendering of the two-cylinder mushroom
# cloud for the Upshot-Knothole Nancy test (24 kT, 91 m HOB).
#
# Shows:
#   - 3D particle distribution coloured by concentration (fiery colourmap)
#   - Wireframe stem and cap cylinders
#   - NOAA three-layer altitude bands
#   - Vertical cross-section with Gaussian KDE concentration field
#
# Uses the latest BIPOP-CMA-ES OU parameters (76.8% combined score).
#
# Requirements:
#   ] add CairoMakie

using NuclearDetonation
using NuclearDetonation.Transport
using CairoMakie
using Random

# --- Mushroom cloud geometry (Glasstone & Dolan 1977) ---
yield_kt = 24.0
hob_m = 91.0
cloud = create_mushroom_cloud_from_yield(yield_kt, hob_m)
cylinders = compute_release_cylinders(cloud)

stem_cyl = cylinders[1]
cap_cyl = cylinders[2]

stem_h_km = cloud.stem_height / 1000.0
cap_h_km = cloud.cap_height / 1000.0
stem_r_km = cloud.stem_radius / 1000.0
cap_r_km = cloud.cap_radius / 1000.0

println("Nancy mushroom cloud ($(yield_kt) kT, $(hob_m) m HOB):")
println("  Cloud top:   $(round(cap_h_km, digits=2)) km")
println("  Stem height: $(round(stem_h_km, digits=2)) km")
println("  Cap radius:  $(round(cap_r_km, digits=2)) km")
println("  Stem radius: $(round(stem_r_km, digits=2)) km")

# --- NOAA three-layer fractions (from BIPOP-CMA-ES OU 76.8%) ---
params = nancy_optimised_config()
lf = params.layer_fractions

# Layer boundaries in km
layer_bounds = [(0.0, 3.8), (3.8, 6.1), (6.1, 12.5)]
layer_names = ["Lower", "Middle", "Upper"]
layer_fracs = [lf.lower, lf.middle, lf.upper]

println("\nNOAA three-layer activity fractions:")
for (name, (lo, hi), frac) in zip(layer_names, layer_bounds, layer_fracs)
    println("  $(name) ($(lo)–$(hi) km): $(round(frac*100, digits=1))%")
end

# --- Generate particles distributed by NOAA layer fractions ---
rng = Random.default_rng()
Random.seed!(rng, 42)
n_particles = 20_000

particles = NTuple{3,Float64}[]  # (x, y, z) in km

for (i, ((z_lo, z_hi), frac)) in enumerate(zip(layer_bounds, layer_fracs))
    n_layer = round(Int, n_particles * frac)

    for _ in 1:n_layer
        z = z_lo + rand(rng) * (z_hi - z_lo)

        # Determine which cylinder this altitude falls in
        z_m = z * 1000.0
        if z_m <= cloud.stem_height
            # In stem region
            r_max = stem_r_km
        elseif z_m <= cloud.cap_height
            # In cap region
            r_max = cap_r_km
        else
            # Above cloud — distribute across cap radius
            r_max = cap_r_km
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

    # Horizontal rings
    for z in range(z_lo_km, z_hi_km, length=8)
        lines!(ax, r_km .* cos.(θ), r_km .* sin.(θ), fill(z, length(θ)),
            color=color, linewidth=linewidth)
    end

    # Vertical ribs
    zz = [z_lo_km, z_hi_km]
    for θi in range(0, 2π, length=12)[1:end-1]
        lines!(ax, fill(r_km * cos(θi), 2), fill(r_km * sin(θi), 2), zz,
            color=color, linewidth=linewidth)
    end
end

# --- Helper: draw NOAA layer band on Axis3 as transparent horizontal planes ---
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

max_r = cap_r_km * 1.3

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
    # NOAA layer boundaries
    for (z_lo, z_hi) in layer_bounds
        draw_layer_band!(ax, z_lo, max_r, color=(:gray60, 0.3))
    end
    draw_layer_band!(ax, 12.5, max_r, color=(:gray60, 0.3))

    # Wireframe cylinders
    draw_cylinder!(ax, stem_r_km, 0.0, stem_h_km, color=(:white, 0.5), linewidth=0.8)
    draw_cylinder!(ax, cap_r_km, stem_h_km, cap_h_km, color=(:cyan, 0.5), linewidth=0.8)

    # Particles coloured by altitude (fiery)
    scatter!(ax, px, py, pz,
        markersize=1.5,
        color=pz,
        colormap=:hot,
        colorrange=(0.0, cap_h_km),
    )

    # Ground zero
    scatter!(ax, [0.0], [0.0], [0.0],
        markersize=12, color=:yellow, marker=:star5,
        strokewidth=1.5, strokecolor=:black)

    xlims!(ax, -max_r, max_r)
    ylims!(ax, -max_r, max_r)
    zlims!(ax, 0, cap_h_km * 1.1)
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
z_grid = range(0, cap_h_km * 1.15, length=nz)
concentration = zeros(Float64, nx, nz)
σ = 0.22  # km

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

# Stem boundary (white dashed)
lines!(ax3,
    [-stem_r_km, -stem_r_km, stem_r_km, stem_r_km, -stem_r_km],
    [0.0, stem_h_km, stem_h_km, 0.0, 0.0],
    color=:white, linewidth=2, linestyle=:dash)

# Cap boundary (cyan dashed)
lines!(ax3,
    [-cap_r_km, -cap_r_km, cap_r_km, cap_r_km, -cap_r_km],
    [stem_h_km, cap_h_km, cap_h_km, stem_h_km, stem_h_km],
    color=:cyan, linewidth=2, linestyle=:dash)

# NOAA layer boundaries
layer_colours = [:steelblue, :forestgreen, :firebrick]
for (i, (z_lo, z_hi)) in enumerate(layer_bounds)
    hlines!(ax3, [z_lo], color=(layer_colours[i], 0.6), linewidth=1.2, linestyle=:dot)
end
hlines!(ax3, [12.5], color=(layer_colours[3], 0.6), linewidth=1.2, linestyle=:dot)

# Layer fraction labels
for (i, ((z_lo, z_hi), frac)) in enumerate(zip(layer_bounds, layer_fracs))
    mid = (z_lo + min(z_hi, cap_h_km * 1.15)) / 2
    pct = round(frac * 100, digits=1)
    text!(ax3, -max_r * 0.95, mid,
        text="$(layer_names[i]) $(pct)%",
        color=layer_colours[i], fontsize=11, align=(:left, :center), font=:bold)
end

# Dimension labels
text!(ax3, cap_r_km + 0.15, cap_h_km,
    text="Cloud top $(round(cap_h_km, digits=1)) km",
    color=:cyan, fontsize=11, align=(:left, :center))
text!(ax3, cap_r_km + 0.15, stem_h_km,
    text="Stem top $(round(stem_h_km, digits=1)) km",
    color=:white, fontsize=11, align=(:left, :center))

# Ground zero
scatter!(ax3, [0.0], [0.0],
    markersize=12, color=:yellow, marker=:star5,
    strokewidth=1.5, strokecolor=:black)

xlims!(ax3, -max_r, max_r * 1.5)
ylims!(ax3, 0, cap_h_km * 1.15)

# Supertitle
Label(fig[0, :], "Mushroom Cloud Release Geometry — Nancy (24 kT, 91 m HOB)",
    fontsize=20, font=:bold)

outfile = joinpath(@__DIR__, "mushroom_cloud_geometry.png")
save(outfile, fig, px_per_unit=2)
println("Saved: $(outfile)")
