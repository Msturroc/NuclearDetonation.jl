# Point Release Example
# =====================
# Simulates a constant point-source release using ERA5 reanalysis data.
# Demonstrates non-weapon use case (e.g. industrial point source in Nevada).
#
# This example shows how to:
#   - Configure a constant release (vs bomb release)
#   - Change the ODE solver (Euler vs Tsit5)
#   - Adjust timestep and interpolation scheme

using NuclearDetonation
using NuclearDetonation.Transport

# --- Met data ---
met_files = nancy_era5_files()

# --- Release parameters ---
lat = 37.0956
lon = -116.1028
release_height_m = 100.0  # 100 m stack height
activity_Bq = 1e12        # 1 TBq total release
release_time = Transport.DateTime(1953, 3, 24, 13)

# --- Numerical configuration ---
# Try Tsit5 solver with linear interpolation for comparison
num_config = ERA5NumericalConfig(
    interpolation_order = LinearInterp,
    ode_solver_type = :Tsit5,     # 5th-order Runge-Kutta
    fixed_dt = 300.0,
    turbulence = OrnsteinUhlenbeck,
    store_turbulent_velocities = true,
    name = "point_release_tsit5",
)

# --- Simulation domain ---
domain = SimulationDomain(
    t_start = release_time,
    duration_hours = 24,
    lat_min = 35.0, lat_max = 42.0,
    lon_min = -120.0, lon_max = -110.0,
)

# --- Source term: constant release from a column ---
source = ReleaseSource(
    lat = lat,
    lon = lon,
    geometry = ColumnRelease(
        bottom_m = release_height_m - 10.0,
        top_m = release_height_m + 10.0,
    ),
    profile = ConstantRelease(),
)

# --- Particle size (single bin, small aerosol) ---
particle = ParticleProperties(diameter_μm=5.0, density_gcm3=2.0)
size_config = ParticleSizeConfig(
    size_bins = [particle],
    vgrav_tables = build_vgrav_tables([particle]),
)

# --- Initialise and run ---
state = initialize_simulation(
    domain, [source],
    ["Cs137"],
    [DecayParams(ExponentialDecay(half_life_hours=30.0 * 365.25 * 24.0))],  # ~30 yr
    n_particles = 1000,
)

println("Running point release simulation (24 hours, Tsit5 solver)...")
snapshots = run_simulation!(
    state, met_files,
    numerical_config = num_config,
    particle_size_config = size_config,
)

println("Simulation complete — $(length(snapshots)) snapshots saved.")
