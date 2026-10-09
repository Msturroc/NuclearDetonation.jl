# Default optimised parameter configurations
# Nancy: BIPOP-CMA-ES on the GPU forward model (10k particles, 6000 evaluations),
# refitted 2026-10-09 after the met-window fix. Held-out score 82.2 +/- 1.0%
# (8 seeds); 82.4 +/- 0.4% through Transport.run_simulation! itself.

export nancy_optimised_config, etex_optimised_config

"""
    nancy_optimised_config()

Return optimised HannaTurbulenceConfig and associated physics scaling parameters
for the Upshot-Knothole Nancy nuclear test (24 kT, 24 March 1953).

Fitted by BIPOP-CMA-ES against digitised historical fallout observations (dose-rate
contours at H+12 and arrival times; combined FMS / shape / bearing / extent / TOA
score), with the release at 13:00 UTC and the met windows stepped continuously
across ERA5 files. Re-scored on 8 held-out seeds: 82.2 +/- 1.0%.

The data constrain these parameters only jointly: a surrogate-assisted CMA-ES fit
(champion5, `nancy-recalibration` branch) scores the same 82.2% with quite
different values. Treat them as a calibrated set, not as measured physical
quantities. `h_diff_scale` and `tmix_scale` are not read by the transport model.

# Returns
- `NamedTuple` with fields:
  - `hanna_config::HannaTurbulenceConfig` — turbulence configuration
  - `particle_size_config` — bimodal particle size distribution parameters
  - `layer_fractions` — (lower, middle, upper) release altitude fractions
  - `physics_scales` — NamedTuple of scaling factors for physics parameters
  - `activity_Bq` — total release activity (Bq)

# Example
```julia
params = nancy_optimised_config()
config = SimulationConfig(
    hanna_config = params.hanna_config,
)
```
"""
function nancy_optimised_config()
    # Hanna turbulence configuration with optimised scaling
    hanna_config = HannaTurbulenceConfig(
        apply_turbulence = true,
        use_cbl = true,
        use_simple_convection = false,
        use_dynamic_L = false,
    )

    # Bimodal particle size distribution (fine + coarse modes)
    # Fine mode: d_median = 37.7 μm, σ_g = 1.83
    # Coarse mode: d_median = 280.4 μm, σ_g = 2.00
    # Fine fraction: 30.3%
    particle_size_config = (
        d_median_fine_μm = 37.692,
        sigma_g_fine = 1.826,
        d_median_coarse_μm = 280.441,
        sigma_g_coarse = 1.999,
        frac_fine = 0.3035,
    )

    # Release layer fractions (NOAA 1984 three-layer model)
    # Lower (0–3,800 m): 18.2%, Middle (3,800–6,100 m): 23.1%, Upper (6,100–12,500 m): 58.7%
    layer_fractions = (
        lower = 0.18243,
        middle = 0.23104,
        upper = 1.0 - 0.18243 - 0.23104,
    )

    # Physics scaling factors
    physics_scales = (
        sigma_w_scale = 4.233,              # Vertical diffusivity
        sigma_h_scale = 5.137,              # Horizontal diffusivity
        h_diff_scale = 0.1810,              # Horizontal diffusion in BL (unused by the model)
        tl_scale = 1.677,                   # Lagrangian timescale
        vd_scale = 5.060,                   # Dry deposition velocity
        vgrav_scale = 0.6608,               # Gravitational settling
        omega_scale = 2.906,                # Vertical velocity
        mixing_height_scale = 0.4477,       # BL mixing height
        tmix_scale = 0.3645,                # Mixing timescale (unused by the model)
        surface_height_scale = 2.702,       # Surface height
        roughness_scale = 0.1758,           # Surface roughness
        smooth_sigma = 1.071,               # Gaussian smoothing (grid cells)
    )

    return (
        hanna_config = hanna_config,
        particle_size_config = particle_size_config,
        layer_fractions = layer_fractions,
        physics_scales = physics_scales,
        activity_Bq = 59.736e15,
    )
end

"""
    etex_optimised_config()

Return optimised transport/turbulence parameters calibrated against the ETEX-1
European Tracer Experiment (340 kg PMCH gas release, Monterfil France, Oct 1994,
168 stations across Europe).

These parameters are appropriate for point source / continuous releases in
European-scale dispersion modelling. ETEX is an inert gas tracer, so no particle
size distribution or gravitational settling parameters are included.

Parameters were obtained via CMA-ES optimisation (FMS = 0.572, 400 evaluations)
using gridded Figure of Merit in Space scoring against observed PMCH concentrations.
They were fitted before the met-window fix (2026-10-09), when the hour between
consecutive ERA5 files was skipped, and have not been refitted since.

# Returns
- `NamedTuple` with fields:
  - `hanna_config::HannaTurbulenceConfig` — turbulence configuration
  - `physics_scales` — NamedTuple of scaling factors for transport parameters

# Example
```julia
params = etex_optimised_config()
p = params.physics_scales
hanna = HannaTurbulenceConfig(
    sigma_scale = p.sigma_h_scale,
    sigma_scale_vertical = p.sigma_w_scale,
    tl_scale = p.tl_scale,
    use_cbl = true,
)
dep = DepositionConfig(
    simple_deposition_velocity = 0.002,
    mixing_height = 1000.0 * p.mixing_height_scale,
    surface_roughness = 0.1 * p.roughness_scale,
)
sim_cfg = SimulationConfig(omega_scale = p.omega_scale)
```
"""
function etex_optimised_config()
    hanna_config = HannaTurbulenceConfig(
        apply_turbulence = true,
        use_cbl = true,
        use_simple_convection = false,
        use_dynamic_L = false,
    )

    # Transport scaling factors from CMA-ES optimisation against ETEX-1 (FMS = 0.572)
    # Key differences from Nancy/NTS bomb calibration:
    #   - Much lower vertical diffusivity (gas tracer, no mushroom cloud)
    #   - Lower omega_scale (less vertical velocity amplification)
    #   - Much higher horizontal BL diffusion and Lagrangian timescale
    physics_scales = (
        sigma_w_scale = 0.2021,             # Vertical diffusivity (Nancy: 4.028)
        sigma_h_scale = 7.6081,             # Horizontal diffusivity (Nancy: 2.220)
        h_diff_scale = 36.909,              # Horizontal diffusion in BL (Nancy: 0.2055)
        tl_scale = 42.369,                  # Lagrangian timescale (Nancy: 4.458)
        omega_scale = 0.8155,               # Vertical velocity (Nancy: 2.557)
        mixing_height_scale = 8.0,          # BL mixing height (Nancy: 4.105)
        tmix_scale = 6.0215,               # Mixing timescale (Nancy: 1.290)
        roughness_scale = 4.4314,           # Surface roughness (Nancy: 1.174)
    )

    return (
        hanna_config = hanna_config,
        physics_scales = physics_scales,
    )
end
