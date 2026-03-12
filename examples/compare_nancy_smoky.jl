#!/usr/bin/env julia
# Parameter Comparison: Nancy vs Smoky Best-Fit Analysis
#
# Loads both best-fit parameter sets and produces:
# 1. Console table with side-by-side values, units, and normalised positions
# 2. Grouped bar chart (param_comparison.png) with normalised [0,1] values
#
# Usage:
#   julia --project=. examples/compare_nancy_smoky.jl

using Printf
using CairoMakie

# ============================================================================
# NANCY PARAMETERS — from src/transport/defaults.jl nancy_optimised_config()
# ============================================================================

nancy_params = Dict{String,Float64}(
    "d_median_fine"       => 127.552,
    "sigma_g_fine"        => 2.669,
    "d_median_coarse"     => 141.861,
    "sigma_g_coarse"      => 2.523,
    "frac_fine"           => 0.8652,
    "frac_lower"          => 0.05617,
    "frac_middle"         => 0.35074,
    "sigma_w_scale"       => 4.028,
    "sigma_h_scale"       => 2.220,
    "h_diff_scale"        => 0.2055,
    "tl_scale"            => 4.458,
    "vd_scale"            => 4.397,
    "vgrav_scale"         => 0.5532,
    "omega_scale"         => 2.557,
    "mixing_height_scale" => 4.105,
    "tmix_scale"          => 1.290,
    "surface_height_scale"=> 1.554,
    "roughness_scale"     => 1.174,
    "activity_scale"      => 48.418,
    "smooth_sigma"        => 2.172,
)

nancy_score = 76.8  # % combined

# Nancy bounds (from nancy_cmaes_particle_size.jl)
nancy_lb = Dict{String,Float64}(
    "d_median_fine" => 5.0, "sigma_g_fine" => 1.1, "d_median_coarse" => 50.0,
    "sigma_g_coarse" => 1.1, "frac_fine" => 0.05, "frac_lower" => 0.01,
    "frac_middle" => 0.01, "sigma_w_scale" => 0.01, "sigma_h_scale" => 0.1,
    "h_diff_scale" => 0.05, "tl_scale" => 0.1, "vd_scale" => 0.1,
    "vgrav_scale" => 0.1, "omega_scale" => 0.1, "mixing_height_scale" => 0.1,
    "tmix_scale" => 0.1, "surface_height_scale" => 0.1, "roughness_scale" => 0.1,
    "activity_scale" => 5.0, "smooth_sigma" => 0.5,
)
nancy_ub = Dict{String,Float64}(
    "d_median_fine" => 150.0, "sigma_g_fine" => 3.5, "d_median_coarse" => 300.0,
    "sigma_g_coarse" => 3.5, "frac_fine" => 0.95, "frac_lower" => 0.60,
    "frac_middle" => 0.70, "sigma_w_scale" => 5.0, "sigma_h_scale" => 8.0,
    "h_diff_scale" => 2.0, "tl_scale" => 5.0, "vd_scale" => 10.0,
    "vgrav_scale" => 5.0, "omega_scale" => 3.0, "mixing_height_scale" => 5.0,
    "tmix_scale" => 10.0, "surface_height_scale" => 5.0, "roughness_scale" => 5.0,
    "activity_scale" => 100.0, "smooth_sigma" => 5.0,
)

# ============================================================================
# SMOKY PARAMETERS — from smoky_cmaes_ou_best.txt
# ============================================================================

smoky_file = joinpath(@__DIR__, "smoky_example", "smoky_cmaes_ou_best.txt")
smoky_params = Dict{String,Float64}()
for line in eachline(smoky_file)
    startswith(line, "#") && continue
    parts = split(line, "\t", limit=2)
    length(parts) == 2 || continue
    smoky_params[strip(parts[1])] = parse(Float64, strip(parts[2]))
end
smoky_score = 76.8  # from checkpoint: score_new = 0.7676

# Smoky bounds (from smoky_cmaes_particle_size.jl, first 20 params)
smoky_lb = Dict{String,Float64}(
    "d_median_fine" => 20.0, "sigma_g_fine" => 1.1, "d_median_coarse" => 80.0,
    "sigma_g_coarse" => 1.1, "frac_fine" => 0.05, "frac_lower" => 0.01,
    "frac_middle" => 0.01, "sigma_w_scale" => 0.01, "sigma_h_scale" => 0.1,
    "h_diff_scale" => 0.05, "tl_scale" => 0.1, "vd_scale" => 0.1,
    "vgrav_scale" => 0.1, "omega_scale" => 0.1, "mixing_height_scale" => 0.1,
    "tmix_scale" => 0.1, "surface_height_scale" => 0.1, "roughness_scale" => 0.1,
    "activity_scale" => 10.0, "smooth_sigma" => 0.5,
)
smoky_ub = Dict{String,Float64}(
    "d_median_fine" => 100.0, "sigma_g_fine" => 3.5, "d_median_coarse" => 200.0,
    "sigma_g_coarse" => 5.0, "frac_fine" => 0.70, "frac_lower" => 0.50,
    "frac_middle" => 0.50, "sigma_w_scale" => 10.0, "sigma_h_scale" => 8.0,
    "h_diff_scale" => 2.0, "tl_scale" => 10.0, "vd_scale" => 20.0,
    "vgrav_scale" => 10.0, "omega_scale" => 5.0, "mixing_height_scale" => 5.0,
    "tmix_scale" => 5.0, "surface_height_scale" => 10.0, "roughness_scale" => 5.0,
    "activity_scale" => 500.0, "smooth_sigma" => 5.0,
)

# ============================================================================
# PARAMETER METADATA
# ============================================================================

struct ParamInfo
    name::String
    display::String
    meaning::String
    unit::String
    category::String
end

param_list = [
    # Particle size
    ParamInfo("d_median_fine",    "d_fine",           "Fine mode median diameter",    "\\mu m",    "Particle size"),
    ParamInfo("sigma_g_fine",     "\\sigma_{g,fine}",  "Fine mode geometric std dev",  "--",       "Particle size"),
    ParamInfo("d_median_coarse",  "d_coarse",         "Coarse mode median diameter",  "\\mu m",    "Particle size"),
    ParamInfo("sigma_g_coarse",   "\\sigma_{g,coarse}","Coarse mode geometric std dev","--",       "Particle size"),
    ParamInfo("frac_fine",        "f_fine",           "Fine mode mass fraction",      "--",       "Particle size"),
    # Layer fractions
    ParamInfo("frac_lower",       "f_lower",          "Lower layer mass fraction",    "--",       "Layer fractions"),
    ParamInfo("frac_middle",      "f_middle",         "Middle layer mass fraction",   "--",       "Layer fractions"),
    # Turbulence
    ParamInfo("sigma_w_scale",    "\\sigma_w",        "Vertical diffusivity scale",   "\\times",  "Turbulence"),
    ParamInfo("sigma_h_scale",    "\\sigma_h",        "Horizontal diffusivity scale", "\\times",  "Turbulence"),
    ParamInfo("h_diff_scale",     "K_h",              "Horiz.\\ diffusion in BL",     "\\times",  "Turbulence"),
    ParamInfo("tl_scale",         "\\tau_L",          "Lagrangian timescale",         "\\times",  "Turbulence"),
    # Physics
    ParamInfo("vd_scale",         "v_d",              "Dry deposition velocity",      "\\times",  "Physics"),
    ParamInfo("vgrav_scale",      "v_g",              "Gravitational settling",       "\\times",  "Physics"),
    ParamInfo("omega_scale",      "\\omega",          "ERA5 vertical velocity",       "\\times",  "Physics"),
    ParamInfo("mixing_height_scale","h_{mix}",        "BL mixing height",             "\\times",  "Physics"),
    ParamInfo("tmix_scale",       "\\tau_{mix}",      "Mixing timescale",             "\\times",  "Physics"),
    # Deposition
    ParamInfo("surface_height_scale","h_{sfc}",       "Surface height scale",         "\\times",  "Deposition"),
    ParamInfo("roughness_scale",  "z_0",              "Surface roughness",            "\\times",  "Deposition"),
    # Calibration
    ParamInfo("activity_scale",   "A",                "Total release activity",       "\\times 10^{15}\\,Bq", "Calibration"),
    ParamInfo("smooth_sigma",     "\\sigma_s",        "Gaussian smoothing width",     "cells",    "Calibration"),
]

# ============================================================================
# NORMALISE TO BOUND RANGE [0, 1]
# ============================================================================

function normalise(val, lb, ub)
    range = ub - lb
    range <= 0 && return 0.5
    return clamp((val - lb) / range, 0.0, 1.0)
end

# ============================================================================
# CONSOLE TABLE
# ============================================================================

println("="^100)
println("PARAMETER COMPARISON: Nancy 24 kT vs Smoky 44 kT (Best-fit OU)")
println("Nancy score: $(nancy_score)% | Smoky score: $(smoky_score)%")
println("="^100)

local_cat = ""
@printf("%-24s  %12s  %12s  %8s  %8s  %s\n",
        "Parameter", "Nancy", "Smoky", "N %rng", "S %rng", "Agreement")
println("-"^100)

for p in param_list
    global local_cat
    if p.category != local_cat
        local_cat = p.category
        println("\n  --- $(local_cat) ---")
    end

    n_val = nancy_params[p.name]
    s_val = smoky_params[p.name]

    n_norm = normalise(n_val, nancy_lb[p.name], nancy_ub[p.name])
    s_norm = normalise(s_val, smoky_lb[p.name], smoky_ub[p.name])

    diff = abs(n_norm - s_norm)
    agree = diff < 0.20 ? "AGREE" : "DIVERGE"

    @printf("  %-22s  %12.4f  %12.4f  %6.1f%%  %6.1f%%  %s\n",
            p.name, n_val, s_val, n_norm*100, s_norm*100, agree)
end
println()

# ============================================================================
# GROUPED BAR CHART
# ============================================================================

println("Generating grouped bar chart...")

# Collect data for plotting
names_short = [p.display for p in param_list]
# Use plain text names for plot labels
names_plot = [
    "d_fine", "sig_g_f", "d_coarse", "sig_g_c", "f_fine",
    "f_lower", "f_mid",
    "sig_w", "sig_h", "K_h", "tau_L",
    "v_d", "v_g", "omega", "h_mix", "tau_mix",
    "h_sfc", "z_0",
    "A", "sig_s",
]
categories = [p.category for p in param_list]

n_norms = [normalise(nancy_params[p.name], nancy_lb[p.name], nancy_ub[p.name]) for p in param_list]
s_norms = [normalise(smoky_params[p.name], smoky_lb[p.name], smoky_ub[p.name]) for p in param_list]

n_params = length(param_list)

# Category boundaries for visual grouping
cat_names = unique(categories)
cat_starts = [findfirst(==(c), categories) for c in cat_names]
cat_ends = [findlast(==(c), categories) for c in cat_names]

# Create figure
fig = Figure(size=(1400, 600), fontsize=13)
ax = Axis(fig[1, 1],
    xlabel="Parameter",
    ylabel="Normalised value (fraction of bound range)",
    title="Best-Fit Parameter Comparison: Nancy 24 kT vs Smoky 44 kT",
    xticks=(1:n_params, names_plot),
    xticklabelrotation=pi/4,
    yticks=0:0.2:1.0,
)
ylims!(ax, 0, 1.15)

# Bar width and offset
bw = 0.35
x_nancy = collect(1:n_params) .- bw/2
x_smoky = collect(1:n_params) .+ bw/2

barplot!(ax, x_nancy, n_norms, width=bw, color=:steelblue, label="Nancy 24 kT")
barplot!(ax, x_smoky, s_norms, width=bw, color=:darkorange, label="Smoky 44 kT")

# 20% agreement band — horizontal lines at each bar showing threshold
hlines!(ax, [0.0, 1.0], color=:gray80, linewidth=0.5)

# Category separators
for i in 2:length(cat_starts)
    vlines!(ax, [cat_starts[i] - 0.5], color=:gray60, linewidth=1, linestyle=:dash)
end

# Category labels at top
for (ci, cname) in enumerate(cat_names)
    mid = (cat_starts[ci] + cat_ends[ci]) / 2
    text!(ax, mid, 1.08, text=cname, align=(:center, :bottom), fontsize=10, color=:gray40)
end

# Mark divergent parameters (|diff| >= 0.20) with a star
for i in 1:n_params
    if abs(n_norms[i] - s_norms[i]) >= 0.20
        xmid = Float64(i)
        ymax = max(n_norms[i], s_norms[i]) + 0.03
        text!(ax, xmid, ymax, text="*", align=(:center, :bottom), fontsize=16, color=:red)
    end
end

Legend(fig[2, 1], ax, orientation=:horizontal, tellheight=true, tellwidth=false)

outpath = joinpath(@__DIR__, "nancy_fms_plots", "param_comparison.png")
save(outpath, fig, px_per_unit=2)
println("Saved: $(outpath)")

# ============================================================================
# LATEX TABLE (copy-paste ready)
# ============================================================================

println("\n" * "="^80)
println("LaTeX table (copy into nancy_test.tex):")
println("="^80)

latex_cat = ""
for p in param_list
    global latex_cat
    if p.category != latex_cat
        if latex_cat != ""
            println("\\midrule")
        end
        latex_cat = p.category
        println("\\multicolumn{5}{l}{\\textit{$(latex_cat)}} \\\\")
    end

    n_val = nancy_params[p.name]
    s_val = smoky_params[p.name]

    # Format values
    n_str = if abs(n_val) >= 100; @sprintf("%.1f", n_val)
            elseif abs(n_val) >= 1; @sprintf("%.2f", n_val)
            else @sprintf("%.4f", n_val) end
    s_str = if abs(s_val) >= 100; @sprintf("%.1f", s_val)
            elseif abs(s_val) >= 1; @sprintf("%.2f", s_val)
            else @sprintf("%.4f", s_val) end

    println("\$$(p.display)\$ & $(p.meaning) & $(n_str) & $(s_str) & $(p.unit) \\\\")
end

println("\nDone.")
