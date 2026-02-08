# Default optimised parameter configurations
# Best-fit parameters from BIPOP-CMA-ES optimisation against Nancy observations
# NOTE: Optimisation still in progress — these are intermediate checkpoint values.
# Current best OU score: 75.1% combined (FMS=0.379, Shape=0.843, Extent=1.0, TOA=1.0)

export nancy_optimised_config

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
    # Fine mode: d_median = 111.5 μm, σ_g = 3.14
    # Coarse mode: d_median = 220.7 μm, σ_g = 3.00
    # Fine fraction: 55.2%
    particle_size_config = (
        d_median_fine_μm = 111.46,
        sigma_g_fine = 3.137,
        d_median_coarse_μm = 220.70,
        sigma_g_coarse = 3.004,
        frac_fine = 0.5517,
    )

    # Release layer fractions (NOAA 1984 three-layer model)
    # Lower (0–3,800 m): 1.0%, Middle (3,800–6,100 m): 18.0%, Upper (6,100–12,500 m): 81.0%
    layer_fractions = (
        lower = 0.01,
        middle = 0.1804,
        upper = 1.0 - 0.01 - 0.1804,
    )

    # Physics scaling factors (from BIPOP-CMA-ES OU checkpoint)
    physics_scales = (
        sigma_w_scale = 4.394,          # Vertical diffusivity
        sigma_h_scale = 1.978,          # Horizontal diffusivity
        h_diff_scale = 0.3331,          # Horizontal diffusion in BL
        tl_scale = 2.298,              # Lagrangian timescale
        vd_scale = 2.479,               # Dry deposition velocity
        vgrav_scale = 0.5797,           # Gravitational settling
        omega_scale = 2.155,            # Vertical velocity
        mixing_height_scale = 1.144,    # BL mixing height
        tmix_scale = 7.396,             # Mixing timescale
        surface_height_scale = 0.7905,  # Surface height
        roughness_scale = 1.038,        # Surface roughness
        smooth_sigma = 2.099,           # Gaussian smoothing (grid cells)
    )

    return (
        hanna_config = hanna_config,
        particle_size_config = particle_size_config,
        layer_fractions = layer_fractions,
        physics_scales = physics_scales,
        activity_Bq = 36.21e15,
    )
end
