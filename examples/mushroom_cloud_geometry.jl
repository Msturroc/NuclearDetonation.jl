# Mushroom Cloud Release Geometry Visualisation
# ===============================================
# Produces a cross-section diagram of the two-cylinder mushroom cloud
# decomposition for the Upshot-Knothole Nancy test (24 kT, 91 m HOB).
#
# Shows:
#   - Stem and cap cylinders with dimensions
#   - NOAA three-layer altitude bands
#   - Activity fraction per layer
#
# Requirements:
#   ] add PlotlyJS

using NuclearDetonation
using NuclearDetonation.Transport
using PlotlyJS

# --- Mushroom cloud geometry ---
yield_kt = 24.0
hob_m = 91.0
cloud = create_mushroom_cloud_from_yield(yield_kt, hob_m)
cylinders = compute_release_cylinders(cloud)

stem = cylinders[1]
cap = cylinders[2]

println("Nancy mushroom cloud geometry ($(yield_kt) kT, $(hob_m) m HOB):")
println("  Cloud top:   $(cloud.cap_height) m")
println("  Stem height: $(cloud.stem_height) m")
println("  Cap radius:  $(cloud.cap_radius) m")
println("  Stem radius: $(cloud.stem_radius) m")
println("  Stem volume fraction: $(round(stem.volume_fraction, digits=3))")
println("  Cap volume fraction:  $(round(cap.volume_fraction, digits=3))")

# --- NOAA three-layer fractions ---
params = nancy_optimised_config()
lf = params.layer_fractions

layer_lower = (bottom=0.0, top=3800.0, frac=lf.lower, label="Lower")
layer_middle = (bottom=3800.0, top=6100.0, frac=lf.middle, label="Middle")
layer_upper = (bottom=6100.0, top=12500.0, frac=lf.upper, label="Upper")
layers = [layer_lower, layer_middle, layer_upper]

# --- Side-view cross-section ---
# Each cylinder is drawn as a rectangle (symmetrical about x=0)

# Stem rectangle
stem_x = [-cloud.stem_radius, cloud.stem_radius, cloud.stem_radius, -cloud.stem_radius, -cloud.stem_radius]
stem_y = [stem.hlower, stem.hlower, stem.hupper, stem.hupper, stem.hlower]

# Cap rectangle
cap_x = [-cloud.cap_radius, cloud.cap_radius, cloud.cap_radius, -cloud.cap_radius, -cloud.cap_radius]
cap_y = [cap.hlower, cap.hlower, cap.hupper, cap.hupper, cap.hlower]

# NOAA layer band colours (pale fills)
layer_colours = ["rgba(173,216,230,0.15)", "rgba(144,238,144,0.15)", "rgba(255,182,193,0.15)"]
layer_border  = ["rgba(173,216,230,0.5)",  "rgba(144,238,144,0.5)",  "rgba(255,182,193,0.5)"]

# Build traces
traces = GenericTrace[]

# Layer bands (full width background rectangles)
x_extent = cloud.cap_radius * 1.6
for (i, layer) in enumerate(layers)
    push!(traces, scatter(
        x = [-x_extent, x_extent, x_extent, -x_extent, -x_extent],
        y = [layer.bottom, layer.bottom, layer.top, layer.top, layer.bottom],
        fill = "toself",
        fillcolor = layer_colours[i],
        line = attr(color=layer_border[i], width=1, dash="dot"),
        mode = "lines",
        name = "$(layer.label) ($(round(layer.frac*100, digits=1))%)",
        showlegend = true,
    ))
end

# Stem
push!(traces, scatter(
    x = stem_x, y = stem_y,
    fill = "toself",
    fillcolor = "rgba(180,120,60,0.5)",
    line = attr(color="rgb(139,90,43)", width=2),
    mode = "lines",
    name = "Stem (r=$(round(Int, cloud.stem_radius)) m)",
))

# Cap
push!(traces, scatter(
    x = cap_x, y = cap_y,
    fill = "toself",
    fillcolor = "rgba(200,80,60,0.5)",
    line = attr(color="rgb(178,34,34)", width=2),
    mode = "lines",
    name = "Cap (r=$(round(Int, cloud.cap_radius)) m)",
))

# Annotations for dimensions
annotations = [
    # Cloud top
    attr(x=cloud.cap_radius*1.1, y=cloud.cap_height,
         text="Cloud top: $(round(Int, cloud.cap_height)) m",
         showarrow=false, xanchor="left", font=attr(size=11)),
    # Stem/cap boundary
    attr(x=cloud.cap_radius*1.1, y=cloud.stem_height,
         text="Stem top: $(round(Int, cloud.stem_height)) m",
         showarrow=false, xanchor="left", font=attr(size=11)),
    # Layer boundaries
    attr(x=-x_extent*0.95, y=3800.0,
         text="3,800 m", showarrow=false, yanchor="bottom",
         font=attr(size=10, color="gray")),
    attr(x=-x_extent*0.95, y=6100.0,
         text="6,100 m", showarrow=false, yanchor="bottom",
         font=attr(size=10, color="gray")),
    attr(x=-x_extent*0.95, y=12500.0,
         text="12,500 m", showarrow=false, yanchor="bottom",
         font=attr(size=10, color="gray")),
    # Layer fraction labels
    attr(x=x_extent*0.7, y=1900.0,
         text="$(round(lf.lower*100, digits=1))%",
         showarrow=false, font=attr(size=13, color="steelblue")),
    attr(x=x_extent*0.7, y=4950.0,
         text="$(round(lf.middle*100, digits=1))%",
         showarrow=false, font=attr(size=13, color="green")),
    attr(x=x_extent*0.7, y=9300.0,
         text="$(round(lf.upper*100, digits=1))%",
         showarrow=false, font=attr(size=13, color="indianred")),
]

layout = Layout(
    title = attr(text="Mushroom Cloud Release Geometry — Nancy (24 kT, 91 m HOB)"),
    xaxis = attr(
        title = "Horizontal radius (m)",
        zeroline = true,
        range = [-x_extent, x_extent*1.4],
    ),
    yaxis = attr(
        title = "Altitude (m)",
        range = [0, 13500],
    ),
    annotations = annotations,
    legend = attr(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"),
    width = 900,
    height = 700,
    plot_bgcolor = "white",
)

fig = plot(traces, layout)
display(fig)

println("\nDiagram displayed. Close the plot window to exit.")
