# NuclearDetonation.jl

Nuclear detonation effects and atmospheric dispersion modelling in Julia.

<!-- [![CI](https://github.com/Msturroc/NuclearDetonation.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/Msturroc/NuclearDetonation.jl/actions/workflows/CI.yml) -->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

NuclearDetonation.jl is a Julia package for modelling nuclear weapon effects and simulating post-detonation atmospheric transport of radioactive fallout. It couples Glasstone-based weapon effects calculations with a Lagrangian particle dispersion model driven by ERA5 reanalysis meteorological data.

<p align="center">
  <img src="examples/mushroom_cloud_geometry.png" width="100%" alt="Mushroom cloud release geometry — 3D particle views and KDE cross-section for the Nancy 24 kT test"/>
</p>

<p align="center">
  <img src="examples/nancy_bomb_release.png" width="50%" alt="Model-predicted dose rate contours at H+12 for the Nancy nuclear test"/>
</p>

## Features

### Atmospheric transport

The transport module tracks thousands of Lagrangian particles through three-dimensional wind fields on the full 137-level ERA5 hybrid sigma-pressure vertical grid. Vertical coordinates are handled through hypsometric height integration, allowing accurate representation of the atmosphere from the surface to the lower stratosphere.

Turbulent diffusion uses an Ornstein-Uhlenbeck process to generate temporally correlated stochastic velocities, parameterised following Hanna (1982) with stability-dependent scaling and a convective boundary layer scheme for strongly unstable conditions.

Trajectory integration supports forward Euler (for reference parity with FLEXPART) or Tsit5 fifth-order Runge-Kutta via [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl), with optional adaptive timestepping.

Deposition is handled through both resistance-based and simplified schemes for dry removal, together with below-cloud wet scavenging following Bartnicki (2003). Gravitational settling uses Stokes-Cunningham fall speeds with Sutherland viscosity corrections, supporting bimodal log-normal particle size distributions.

Radioactive decay supports both exponential half-life decay and the Glasstone t^(-1.2) bomb decay law.

### Weapon effects (Glasstone & Dolan)

The weapon effects module provides blast overpressure calculations (including Soviet overpressure data), thermal radiation estimates, initial nuclear radiation, the WSEG-10 fallout model, and mushroom cloud geometry from yield scaling laws.

### Release geometries

Particles can be released from several source configurations: a zero-radius vertical column for stack releases, a finite-radius cylindrical volume, a two-cylinder stem-and-cap mushroom cloud model with volume-proportional particle distribution, or the NOAA three-layer scheme with configurable mass fractions across lower, middle and upper altitude bands.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/Msturroc/NuclearDetonation.jl")
```

## Quick start

### Bomb release (Nancy 24 kT)

See [`examples/nancy_bomb_release.jl`](examples/nancy_bomb_release.jl) for the full working example. The key steps:

```julia
using NuclearDetonation
using NuclearDetonation.Transport
using Dates

# ERA5 met data downloads automatically from Zenodo (~96 MB, first run only)
era5_files = nancy_era5_files()

# Set up simulation domain from the met grid
domain = Transport.SimulationDomain(
    lon_min = 240.09, lon_max = 249.93,
    lat_min = 35.15,  lat_max = 41.90,
    z_min = 0.0, z_max = 35000.0,
    nx = 36, ny = 25, nz = 137,
    start_time = DateTime(1953, 3, 24, 13),
    end_time   = DateTime(1953, 3, 25, 1),
)

# Release source at Nevada Test Site
release_x, release_y = Transport.latlon_to_grid(domain, 37.0956, -116.1028)
source = ReleaseSource(
    (release_x, release_y),
    CylinderRelease(0.0, 12500.0, 2500.0),
    BombRelease(0.0),
    [48.4e15],   # total activity (Bq)
    10_000,      # particles
)

# Initialise and run
state = Transport.initialize_simulation(domain, [source],
    ["MixedFP"], [Transport.DecayParams(kdecay=Transport.NoDecay)];
    log_depositions=true)

# ... generate particles, configure physics, then:
Transport.run_simulation!(state, era5_files, ...)
```

### Point release (constant source)

See [`examples/point_release.jl`](examples/point_release.jl) for an industrial stack release example using the Tsit5 solver and a single-bin 5 um aerosol.

## Configuring the solver

```julia
# Forward Euler with O-U turbulence (default)
config = ERA5NumericalConfig(
    ode_solver_type = :Euler,
    fixed_dt = 300.0,
    turbulence = OrnsteinUhlenbeck,
)

# Tsit5 (5th-order Runge-Kutta)
config = ERA5NumericalConfig(
    ode_solver_type = :Tsit5,
    fixed_dt = 300.0,
    turbulence = OrnsteinUhlenbeck,
)

# Adaptive timestepping
config = ERA5NumericalConfig(
    ode_solver_type = :AutoTsit5,
    fixed_dt = nothing,
    reltol = 1e-5,
    abstol = 1e-7,
)
```

## Meteorological data

The package uses ECMWF ERA5 reanalysis data on 137 hybrid model levels, reading `x_wind_ml`, `y_wind_ml`, `air_temperature_ml` and `omega_ml` from NetCDF files.

ERA5 data for the Nancy test case is available as a Julia Artifact (~96 MB, [Zenodo DOI: 10.5281/zenodo.18529331](https://doi.org/10.5281/zenodo.18529331)). It downloads automatically on first use via `nancy_era5_files()`.

## Inspired by

- [FLEXPART](https://github.com/flexpart/flexpart) — Lagrangian particle dispersion model
- [glasstone](https://github.com/NukeWorker/glasstone) — Nuclear weapon effects (Python)
- [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl) — Julia ODE solvers
- [Interpolations.jl](https://github.com/JuliaMath/Interpolations.jl) — Grid interpolation

## Licence

MIT — see [LICENSE](LICENSE).
