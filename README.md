# NuclearDetonation.jl

Nuclear detonation atmospheric dispersion modelling in Julia.

<!-- [![CI](https://github.com/Msturroc/NuclearDetonation.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/Msturroc/NuclearDetonation.jl/actions/workflows/CI.yml) -->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## What it does

**NuclearDetonation.jl** provides:

- **Atmospheric transport** — Lagrangian particle dispersion with advection, turbulent diffusion (Ornstein-Uhlenbeck process), dry and wet deposition, radioactive decay, and gravitational settling
- **Weapon effects** — Glasstone-based blast overpressure, thermal radiation, initial nuclear radiation, and WSEG-10 fallout models

The transport module uses modern Julia ODE solvers ([OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl)) with configurable numerical schemes, and supports both ERA5 reanalysis and GFS forecast meteorological data via multiple dispatch.

## Inspired by

- [GLASSTONE](https://github.com/NukeWorker/glasstone) — Nuclear weapon effects (Glasstone & Dolan, 1977)
- [FLEXPART](https://www.flexpart.eu/) — Lagrangian particle dispersion model (Stohl et al., 2005)

## Nancy showcase

The package is validated against digitised historical fallout observations from the Upshot-Knothole Nancy nuclear test (24 kT, 24 March 1953, Nevada Test Site). The Ornstein-Uhlenbeck turbulence model achieves ~75% combined validation score (FMS + shape + extent + time-of-arrival) against the observed fallout pattern.

<!-- TODO: Add showcase images here -->

## Installation

```julia
using Pkg
Pkg.add("NuclearDetonation")
```

Or from the Julia REPL:

```
] add NuclearDetonation
```

## Quick start

### Bomb release

```julia
using NuclearDetonation
using NuclearDetonation.Transport

# Load ERA5 met data (downloads ~300 MB on first use)
met_files = nancy_era5_files()

# Configure detonation
domain = SimulationDomain(
    t_start = Transport.DateTime(1953, 3, 24, 13),
    duration_hours = 48,
    lat_min = 35.0, lat_max = 42.0,
    lon_min = -120.0, lon_max = -110.0,
)

mushroom = create_mushroom_cloud_from_yield(24.0, 91.0)
source = ReleaseSource(lat=37.0956, lon=-116.1028,
                       geometry=mushroom, profile=BombRelease(0.5))

state = initialize_simulation(domain, [source],
    ["Mixed_fission_products"], [DecayParams(NoDecay())],
    n_particles=2000)

snapshots = run_simulation!(state, met_files)
```

### Point release

```julia
source = ReleaseSource(lat=37.0956, lon=-116.1028,
    geometry=ColumnRelease(bottom_m=90.0, top_m=110.0),
    profile=ConstantRelease())

state = initialize_simulation(domain, [source],
    ["Cs137"], [DecayParams(ExponentialDecay(half_life_hours=2.63e5))],
    n_particles=1000)

snapshots = run_simulation!(state, met_files)
```

## Configuring the solver

```julia
# Forward Euler (default, fast)
config = ERA5NumericalConfig(ode_solver_type=:Euler, fixed_dt=300.0)

# Tsit5 (5th-order Runge-Kutta, more accurate)
config = ERA5NumericalConfig(ode_solver_type=:Tsit5, fixed_dt=300.0)

# Adaptive timestepping
config = ERA5NumericalConfig(ode_solver_type=:AutoTsit5, fixed_dt=nothing,
                             reltol=1e-5, abstol=1e-7)

# Turbulence models
config = ERA5NumericalConfig(turbulence=OrnsteinUhlenbeck)  # default
config = ERA5NumericalConfig(turbulence=RandomWalk)          # simple
config = ERA5NumericalConfig(turbulence=HannaTurbulence)     # Hanna (1982)
```

## Meteorological data

The package supports two met data formats via multiple dispatch:

| Format | Source | Variables | Vertical |
|--------|--------|-----------|----------|
| `ERA5Format` | ECMWF ERA5 reanalysis | `x_wind_ml`, `y_wind_ml`, `air_temperature_ml` | 137 hybrid model levels |
| `GFSFormat` | NCEP GFS forecast | `x_wind_pl`, `y_wind_pl`, `air_temperature_pl` | Pressure levels |

ERA5 data for the Nancy test case is available as a lazy Julia Artifact (~300 MB, downloaded automatically on first use).

## GUI for Windows

A standalone Windows executable with a browser-based GUI is planned (Genie.jl + PackageCompiler). See the `app/` and `build/` directories.

## References

- Glasstone, S. & Dolan, P.J. (1977). *The Effects of Nuclear Weapons*. 3rd ed.
- Hanna, S.R. (1982). *Applications in Air Pollution Modeling*. In: Atmospheric Turbulence and Air Pollution Modelling.
- Stohl, A. et al. (2005). Technical note: The Lagrangian particle dispersion model FLEXPART version 6.2. *Atmos. Chem. Phys.*, 5, 2461-2474.

## Licence

MIT — see [LICENSE](LICENSE).
