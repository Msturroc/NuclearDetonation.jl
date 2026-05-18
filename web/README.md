# NuclearDetonation.jl Web GUI

A local web application for running atmospheric fallout dispersion simulations
from the browser — no Julia knowledge required. A Julia HTTP backend wraps the
`NuclearDetonation.jl` transport core; the frontend is React + Leaflet.

## Running (development)

```
julia --threads=2 --project=web web/app.jl
```

This loads the packages, pre-loads the ERA5 meteorological data, and opens
`http://localhost:9000`. The `--threads=2` flag keeps the UI responsive while a
simulation runs.

## Simulation history (PostgreSQL)

Completed runs are persisted to PostgreSQL so they survive server restarts, and
re-running identical parameters returns cached results instantly.

- The backend connects to `host=localhost dbname=nucleardetonation user=marc
  password=nuclear_dev` by default. Override with the `NUCDET_DB_URL`
  environment variable.
- If no database is reachable the server logs a warning and continues — run
  history and result caching are simply unavailable.
- Set `NUCDET_DISABLE_DB=1` to skip the database entirely. The standalone
  Windows installer build does this automatically, since it ships without a
  Postgres server.

## Rebuilding the frontend

The React frontend lives in `web/frontend/`. After changing it:

```
cd web/frontend
npm install      # first time only
npm run build
```

`npm run build` emits the production bundle into `web/public_react/`, which the
Julia server serves directly. Commit the regenerated bundle alongside the
source change.

## Building the standalone Windows installer

`web/build_app.jl` compiles a self-contained executable with PackageCompiler, so
the target machine needs no Julia install. Run `npm run build` first so the
bundled frontend is current, then:

```
julia --project=web web/build_app.jl
```

The build bundles the ERA5 artifacts, the observation datasets under `data/`,
and ffmpeg (for animation export), and writes a `NuclearDetonation.bat`
launcher. It bakes `NUCDET_DISABLE_DB=1` so the installed app runs without a
database.

## Layout

| Path | Purpose |
|---|---|
| `web/app.jl` | Dev entry point — loads data and starts the server |
| `web/src/server.jl` | HTTP routes and the JSON API |
| `web/src/simulation.jl` | Wraps the transport core; bomb and point-release paths |
| `web/src/animation.jl` | Plume animation frames and GIF/MP4 export |
| `web/src/database.jl` | PostgreSQL persistence (with no-op stubs when disabled) |
| `web/src/contours.jl` | Dose/deposition contour generation as GeoJSON |
| `web/src/prediction.jl` | XGBoost impact prediction for NPP sites |
| `web/frontend/` | React + Vite source for the frontend |
| `web/public_react/` | Built frontend bundle served by the backend |
| `web/build_app.jl` | PackageCompiler build of the standalone installer |
