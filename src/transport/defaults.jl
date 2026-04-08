# Default optimised parameter configurations
# Best-fit parameters from BIPOP-CMA-ES optimisation against Nancy observations
# Current best OU score: 76.8% combined (FMS=0.370, Shape=0.860, Extent=1.0, TOA=1.0)

export nancy_optimised_config, etex_optimised_config

"""
    nancy_optimised_config()

Return optimised HannaTurbulenceConfig and associated physics scaling parameters
for the Upshot-Knothole Nancy nuclear test (24 kT, 24 March 1953).

These parameters were obtained via BIPOP-CMA-ES optimisation against digitised
historical fallout observations. The optimisation is ongoing; these represent
the current best checkpoint values.

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
    # Fine mode: d_median = 127.6 μm, σ_g = 2.67
    # Coarse mode: d_median = 141.9 μm, σ_g = 2.52
    # Fine fraction: 86.5%
    particle_size_config = (
        d_median_fine_μm = 127.552,
        sigma_g_fine = 2.669,
        d_median_coarse_μm = 141.861,
        sigma_g_coarse = 2.523,
        frac_fine = 0.8652,
    )

    # Release layer fractions (NOAA 1984 three-layer model)
    # Lower (0–3,800 m): 5.6%, Middle (3,800–6,100 m): 35.1%, Upper (6,100–12,500 m): 59.3%
    layer_fractions = (
        lower = 0.05617,
        middle = 0.35074,
        upper = 1.0 - 0.05617 - 0.35074,
    )

    # Physics scaling factors (from BIPOP-CMA-ES OU, 76.8%)
    physics_scales = (
        sigma_w_scale = 4.028,              # Vertical diffusivity
        sigma_h_scale = 2.220,              # Horizontal diffusivity
        h_diff_scale = 0.2055,              # Horizontal diffusion in BL
        tl_scale = 4.458,                   # Lagrangian timescale
        vd_scale = 4.397,                   # Dry deposition velocity
        vgrav_scale = 0.5532,               # Gravitational settling
        omega_scale = 2.557,                # Vertical velocity
        mixing_height_scale = 4.105,        # BL mixing height
        tmix_scale = 1.290,                 # Mixing timescale
        surface_height_scale = 1.554,       # Surface height
        roughness_scale = 1.174,            # Surface roughness
        smooth_sigma = 2.172,               # Gaussian smoothing (grid cells)
    )

    return (
        hanna_config = hanna_config,
        particle_size_config = particle_size_config,
        layer_fractions = layer_fractions,
        physics_scales = physics_scales,
        activity_Bq = 48.418e15,
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
