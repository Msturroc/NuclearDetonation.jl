# NuclearDetonation.jl GUI

A desktop application for running atmospheric fallout dispersion simulations
without writing code. It is plain Julia: a GLMakie window wrapping the
`NuclearDetonation.jl` transport core.

## Running

```
julia --threads=2 --project=gui gui/app.jl
```

The window opens straight away and loads the built-in ERA5 data in the
background. The `--threads=2` flag runs simulations on a worker thread so the
window stays responsive.

Drag the map to pan, and scroll or use the +/− buttons to zoom; Fit returns to
the weather domain. Click the map to place the release, or click one of the NPP
sites. Exports (CSV, PNG,
GIF, MP4) go to the output folder shown at the bottom of the panel, which
defaults to `Documents/NuclearDetonation`. Errors are also written to
`error.log` there.

## Map

The basemap is OpenStreetMap's standard tiles, fetched as you pan and zoom, so
it needs internet. Requests identify the app with its own User-Agent, as OSM's
[tile usage policy](https://operations.osmfoundation.org/policies/tiles/)
requires; the policy is fine with light use like this but not bulk downloading.
Offline, Natural Earth country outlines bundled in `data/naturalearth/` show
instead. To change provider, edit `gui/src/tiles.jl`.

## Point-release source terms

Enter source terms as `isotope activity_TBq [half-life_h]`, separated by commas,
for example `Cs-137 1.0, I-131 5.0, X-1 2.0 12.5`. Isotopes with a preset
half-life (Cs-137, Cs-134, I-131, I-133, Sr-90, Co-60, Ru-103, Ru-106, Te-132,
Xe-133, Pu-239, Generic) can omit it; any other nuclide needs one, and 0 means no
decay. Particles are released uniformly over the release duration.

## Building the standalone Windows installer

`gui/build_app.jl` compiles a self-contained executable with PackageCompiler, so
the target machine needs no Julia install:

```
julia --project=gui gui/build_app.jl
```

The build bundles the ERA5 artifacts, the basemap and observation datasets under
`data/`, and the prediction models, and writes a `NuclearDetonation.bat`
launcher. GLMakie needs OpenGL 3.3. On virtual machines or remote desktops
without GPU drivers, put Mesa's software `opengl32.dll` next to the executable.

## Layout

| Path | Purpose |
|---|---|
| `gui/app.jl` | Dev entry point |
| `gui/src/NuclearDetonationGUI.jl` | Module and `julia_main` for the compiled app |
| `gui/src/window.jl` | Window layout, map layers, callbacks and exports |
| `gui/src/simulation.jl` | Wraps the transport core: bomb and point-release setups |
| `gui/src/animation.jl` | Plume animation frames per model level |
| `gui/src/tiles.jl` | OpenStreetMap tile provider for the map |
| `gui/src/basemap.jl` | Offline Natural Earth country outlines under the tiles |
| `gui/src/observations.jl` | Nancy and ETEX observation overlays |
| `gui/src/prediction.jl` | XGBoost impact prediction for NPP sites |
| `gui/src/arl_reader.jl`, `arl_converter.jl` | ARL weather files → ERA5-layout NetCDF |
| `gui/build_app.jl` | PackageCompiler build of the standalone installer |
