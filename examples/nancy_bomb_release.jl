# Nancy Bomb Release Example
# ===========================
# Simulates the Upshot-Knothole Nancy nuclear test (24 kT, 24 March 1953)
# using ERA5 reanalysis data and Ornstein-Uhlenbeck turbulence.
#
# This example demonstrates:
#   - Loading ERA5 met data via Julia Artifacts
#   - Configuring a bomb release with mushroom cloud geometry
#   - Running a 48-hour dispersion simulation
#   - Exporting dose rate fields to NetCDF

using NuclearDetonation
using NuclearDetonation.Transport

# --- Met data ---
# On first run, this downloads ~300 MB of ERA5 data from Zenodo.
met_files = nancy_era5_files()

# --- Detonation parameters ---
lat = 37.0956       # Nevada Test Site
lon = -116.1028
yield_kt = 24.0     # 24 kilotons
hob_m = 91.0        # Height of burst (300 ft)
detonation_time = Transport.DateTime(1953, 3, 24, 13)  # 13:10 UTC

# --- Numerical configuration ---
# Ornstein-Uhlenbeck turbulence (default for ERA5)
num_config = ERA5NumericalConfig(
    interpolation_order = LinearInterp,
    ode_solver_type = :Euler,
    fixed_dt = 300.0,            # 5-minute timestep
    turbulence = OrnsteinUhlenbeck,
    store_turbulent_velocities = true,
)

# --- Optimised physics parameters ---
params = nancy_optimised_config()

# --- Simulation domain ---
domain = SimulationDomain(
    t_start = detonation_time,
    duration_hours = 48,
    lat_min = 35.0, lat_max = 42.0,
    lon_min = -120.0, lon_max = -110.0,
)

# --- Source term ---
mushroom = create_mushroom_cloud_from_yield(yield_kt, hob_m)
source = ReleaseSource(
    lat = lat,
    lon = lon,
    geometry = mushroom,
    profile = BombRelease(0.5),  # Release over 30 minutes
)

# --- Particle size ---
size_config = ParticleSizeConfig(
    size_bins = [
        ParticleProperties(diameter_μm=params.particle_size_config.d_median_fine_μm, density_gcm3=2.5),
        ParticleProperties(diameter_μm=params.particle_size_config.d_median_coarse_μm, density_gcm3=2.5),
    ],
    vgrav_tables = build_vgrav_tables([
        ParticleProperties(diameter_μm=params.particle_size_config.d_median_fine_μm, density_gcm3=2.5),
        ParticleProperties(diameter_μm=params.particle_size_config.d_median_coarse_μm, density_gcm3=2.5),
    ]),
)

# --- Initialise and run ---
state = initialize_simulation(
    domain, [source],
    ["Mixed_fission_products"],
    [DecayParams(NoDecay())],
    n_particles = 2000,
)

println("Running Nancy bomb release simulation (48 hours)...")
snapshots = run_simulation!(
    state, met_files,
    numerical_config = num_config,
    particle_size_config = size_config,
    hanna_config = params.hanna_config,
)

println("Simulation complete — $(length(snapshots)) snapshots saved.")
