# Nancy Bomb Release Example
# ===========================
# Simulates the Upshot-Knothole Nancy nuclear test (24 kT, 24 March 1953)
# using ERA5 reanalysis data and Ornstein-Uhlenbeck turbulence.
#
# This example demonstrates:
#   - Loading ERA5 met data via Julia Artifacts
#   - Configuring a bomb release with mushroom cloud geometry
#   - Running a 48-hour dispersion simulation
#   - Visualising model-predicted deposition and particle positions
#
# Requirements:
#   ] add PlotlyJS   (for visualisation at the end)

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

# =============================================================================
# Visualisation — model-predicted deposition and particle positions
# =============================================================================
using PlotlyJS

# Extract the final snapshot
final = snapshots[end]

# --- Build lat/lon grids from the domain ---
dom = state.domain
lons = range(dom.lon_min, dom.lon_max, length=dom.nx)
lats = range(dom.lat_min, dom.lat_max, length=dom.ny)

# --- Total ground deposition (Bq/m²), summed over components ---
deposition = dropdims(sum(final.total_deposition, dims=3), dims=3)  # (nx, ny)

# Convert to mR/hr (approximate: 1 Bq/m² mixed fission products ≈ 3.7e-11 mR/hr)
# This is a rough conversion for visualisation only
dose_rate = deposition .* 3.7e-11

# Log10 for plotting (mask zeros)
log_dose = copy(dose_rate)
log_dose[log_dose .<= 0] .= NaN
log_dose = log10.(log_dose)

# --- Deposition contour map ---
contour_trace = contour(
    x = collect(lons),
    y = collect(lats),
    z = log_dose',   # Transpose: PlotlyJS expects (ny, nx) row-major
    colorscale = "YlOrRd",
    contours = attr(
        start = -2.0,
        size = 0.5,
        coloring = "heatmap",
    ),
    colorbar = attr(
        title = "log₁₀(mR/hr)",
        titleside = "right",
    ),
    name = "Dose rate",
)

# --- Particle positions (projected to lat/lon) ---
# Positions are in grid coordinates — convert to lat/lon
positions = final.particle_positions
if !isempty(positions)
    # Grid indices to geographic coordinates
    px = [p[1] for p in positions]
    py = [p[2] for p in positions]
    pz = [p[3] for p in positions]

    # Linear mapping: grid index 1..nx → lon_min..lon_max
    p_lons = dom.lon_min .+ (px .- 1.0) ./ (dom.nx - 1) .* (dom.lon_max - dom.lon_min)
    p_lats = dom.lat_min .+ (py .- 1.0) ./ (dom.ny - 1) .* (dom.lat_max - dom.lat_min)

    particle_trace = scatter(
        x = p_lons,
        y = p_lats,
        mode = "markers",
        marker = attr(
            size = 3,
            color = pz ./ 1000.0,  # Colour by altitude (km)
            colorscale = "Viridis",
            colorbar = attr(
                title = "Altitude (km)",
                x = 1.12,
            ),
            opacity = 0.6,
        ),
        name = "Particles (n=$(length(positions)))",
    )

    traces = [contour_trace, particle_trace]
else
    traces = [contour_trace]
end

layout = Layout(
    title = attr(text="Nancy 24 kT — Model-Predicted Deposition (48 h)"),
    xaxis = attr(title="Longitude (°E)", scaleanchor="y"),
    yaxis = attr(title="Latitude (°N)"),
    width = 900,
    height = 700,
    plot_bgcolor = "white",
)

fig = plot(traces, layout)
display(fig)

println("Visualisation displayed.")
