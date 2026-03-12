# Web GUI for NuclearDetonation.jl

## Context

EPA Ireland staff need a way to run atmospheric dispersion simulations without installing Julia or writing code. We are building a local web application using Genie.jl that runs in the browser, similar in spirit to NOAA's HYSPLIT web interface. The user clicks a map to set the detonation location, configures yield and duration, runs the simulation, and sees dose rate contours overlaid on a geographic map.

The app ships with the Nancy ERA5 dataset (auto-downloaded from Zenodo, ~96 MB) so it works out of the box for the Nevada Test Site region and timeframe (24-27 March 1953).

---

## Architecture

- **Backend**: Genie.jl + Stipple for reactive UI state over WebSockets
- **Map**: Leaflet.js with OpenStreetMap tiles, GeoJSON contour overlays
- **Contours**: Contour.jl generates isolines from the dose rate grid, serialised as GeoJSON
- **Simulation**: Wraps the existing NuclearDetonation.jl API (same logic as `examples/nancy_bomb_release.jl`)
- **Deployment**: `julia --project web/app.jl` opens browser at `localhost:9000`

---

## File Structure

```
web/
  Project.toml              # Deps: GenieFramework, NuclearDetonation, JSON3, Contour, etc.
  app.jl                    # Entry point — starts Genie server, opens browser
  app/
    handlers.jl             # @app reactive model, @onchange handlers, ui() function
    simulation.jl           # Wrapper around NuclearDetonation API
    contours.jl             # Dose grid → GeoJSON contour generation
  public/
    css/app.css             # Layout and legend styling
    js/leaflet_bridge.js    # Leaflet init, click-to-place, contour overlay, legend
  views/
    layout.jl               # Injects Leaflet CSS/JS into page <head>
```

The `web/` folder has its own `Project.toml` referencing NuclearDetonation via `[sources] NuclearDetonation = {path = ".."}`.

---

## Reactive Model (handlers.jl)

### User inputs (`@in`)
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `det_lat` | Float64 | 37.0956 | Detonation latitude |
| `det_lon` | Float64 | -116.1028 | Detonation longitude |
| `yield_kt` | Float64 | 24.0 | Weapon yield (kT) |
| `start_date` | String | "1953-03-24" | ISO date |
| `start_hour` | Int | 13 | UTC hour |
| `duration_hours` | Int | 12 | Simulation duration |
| `n_particles` | Int | 5000 | Particle count |
| `run_clicked` | Bool | false | Button trigger |
| `export_clicked` | Bool | false | CSV export trigger |
| `map_click_lat/lon` | Float64 | 0.0 | From Leaflet click |
| `map_clicked` | Bool | false | Click trigger |

### Outputs (`@out`)
| Field | Type | Description |
|-------|------|-------------|
| `geojson_contours` | String | GeoJSON FeatureCollection for Leaflet |
| `progress_pct` | Int | 0-100 progress |
| `progress_msg` | String | Status text |
| `sim_running` | Bool | Disables controls |
| `sim_complete` | Bool | Shows results |
| `max_dose_mRh` | Float64 | Peak dose rate |
| `error_msg` | String | Error display |

### Key handlers
- `@onchange map_clicked` — copies lat/lon from Leaflet click to input fields
- `@onchange run_clicked` — runs simulation via `@async`, updates progress, generates GeoJSON contours
- `@onchange export_clicked` — writes deposition CSV to `public/results.csv`

---

## Simulation Wrapper (simulation.jl)

Single function `run_dispersion_simulation(; lat, lon, yield_kt, start_date, start_hour, duration_hours, n_particles, progress_callback)` that:

1. Calls `nancy_era5_files()` and pre-caches met fields
2. Creates `SimulationDomain` from ERA5 grid extents + user time window
3. Uses `nancy_optimised_config()` for physics parameters, scales activity by yield ratio
4. Sets up 3-layer NOAA release geometry with `CylinderRelease`
5. Generates bimodal particle distribution (extracted helpers from nancy example)
6. Calls `Transport.run_simulation!()` with trace output disabled for speed
7. Post-processes deposition log into a dose rate grid with Gaussian smoothing
8. Returns `(deposition_log, domain, dose_grid, lon_grid, lat_grid, max_dose_mRh)`

Key reference: `examples/nancy_bomb_release.jl` lines 110-329.

---

## Contour Generation (contours.jl)

Uses `Contour.jl` to extract isolines from the dose rate grid at levels `[0.4, 1.0, 4.0, 10.0, 40.0, 100.0]` mR/h. Each isoline becomes a GeoJSON LineString Feature with `color` and `label` properties. The full FeatureCollection is serialised with `JSON3.write()` and pushed to the browser via Stipple's reactive `geojson_contours` variable.

Also provides `export_deposition_csv()` for CSV download.

---

## Leaflet Bridge (leaflet_bridge.js)

~120 lines of JavaScript:

1. Initialises Leaflet map centred on Nevada Test Site
2. Places a star marker at the detonation location
3. On map click: sets `map_click_lat`, `map_click_lon`, flips `map_clicked` on Stipple VM
4. Watches `det_lat`/`det_lon` for programmatic updates (user typing in fields)
5. Watches `geojson_contours`: parses JSON, renders as colour-coded `L.geoJSON` layer with hover tooltips
6. Adds a static legend control (bottom-right) matching the 6 contour levels

---

## UI Layout

Two-column layout via Stipple/Quasar:
- **Left panel (col-3)**: Input fields (lat, lon, yield, date, hour, duration, particles), Run button with loading state, progress bar, results summary with CSV export
- **Right panel (col-9)**: Full-height Leaflet map (`#map` div, 80vh)

---

## Implementation Phases

### Phase 1: Skeleton
- Create `web/` directory and `Project.toml` with dependencies
- Create `app.jl` entry point that starts Genie and serves a page
- Create `handlers.jl` with reactive model declarations and `ui()` function
- Create `layout.jl` with Leaflet CDN injection
- Create `leaflet_bridge.js` with map init and click handler
- Create `app.css`
- **Verify**: `julia --project web/app.jl` opens browser with map + control panel; clicking map updates lat/lon

### Phase 2: Simulation integration
- Create `simulation.jl` wrapping the Nancy example logic
- Wire `@onchange run_clicked` to call `run_dispersion_simulation`
- Test with 100 particles for quick iteration
- Add error handling

### Phase 3: Contour display
- Create `contours.jl` with Contour.jl GeoJSON generation
- Wire `geojson_contours` output to Leaflet `$watch`
- Test contour rendering and tooltips

### Phase 4: Polish
- CSV export handler and download link
- Progress bar (time-based estimate before/after `run_simulation!`)
- Input validation (bounds checking, date format)
- Test on Windows
- Brief `web/README.md` with usage instructions

---

## Constraints

- The bundled ERA5 data covers only Nevada (lat 35-42, lon 240-250) on 24-27 March 1953. The UI should display a note about this limitation and pre-populate defaults accordingly.
- `run_simulation!` has no progress callback. For the MVP, progress jumps from 40% to 90% around the simulation call. A future enhancement can add a callback to `orchestration.jl`.
- Single-user only (one simulation at a time, guarded by `task_running` flag).

---

## Key Files Referenced

| File | What we use from it |
|------|-------------------|
| `examples/nancy_bomb_release.jl` | Template for simulation.jl — particle generation, physics config, dose calculation |
| `src/NuclearDetonation.jl` | Exports: `nancy_era5_files()`, `nancy_optimised_config()`, release types |
| `src/transport/simulation.jl` | `SimulationDomain`, `SimulationState`, `latlon_to_grid`, `grid_to_latlon` |
| `src/transport/orchestration.jl` | `run_simulation!`, `SimulationConfig`, `OutputConfig`, `TRACE_DISABLED` |

---

## Verification

```bash
# Start the app
cd web && julia --project app.jl

# In browser at localhost:9000:
# 1. Click map to place marker — verify lat/lon fields update
# 2. Click "Run Simulation" with 100 particles — verify progress bar and completion
# 3. Verify contour lines appear on map with correct colours
# 4. Click "Export CSV" — verify download works
# 5. Test on Windows: same steps
```
