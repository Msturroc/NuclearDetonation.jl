#!/usr/bin/env julia
# Build standalone Windows executable using PackageCompiler
#
# Prerequisites:
#   1. Julia 1.12.5 installed
#   2. ffmpeg installed and on PATH (winget install ffmpeg)
#
# Usage (from the web/ directory):
#   julia --project build_app.jl
#
# Output: ../build/NuclearDetonationGUI/ containing the .exe

using Pkg
println("Installing dependencies (first time may take a while)...")
Pkg.instantiate()

using PackageCompiler

src_dir = @__DIR__
out_dir = joinpath(dirname(src_dir), "build", "NuclearDetonationGUI")

println("="^60)
println("  Building NuclearDetonation.jl standalone app")
println("  Output: $out_dir")
println("="^60)
println()
println("This will take 20-40 minutes. Go get a coffee.")
println()

# Build the app
create_app(
    src_dir,           # project directory (web/)
    out_dir;           # output directory
    executables = ["NuclearDetonation" => "julia_main"],
    precompile_execution_file = joinpath(src_dir, "precompile_script.jl"),
    include_lazy_artifacts = true,  # bundle ERA5 data
    cpu_target = "generic",         # single target to reduce memory usage
    force = true,
)

# --- Post-build: copy Artifacts.toml for JLL artifact resolution ---
artifacts_toml_src = joinpath(dirname(src_dir), "Artifacts.toml")
if isfile(artifacts_toml_src)
    artifacts_toml_dst = joinpath(out_dir, "share", "julia", "Artifacts.toml")
    mkpath(dirname(artifacts_toml_dst))
    cp(artifacts_toml_src, artifacts_toml_dst; force=true)
    println("Bundled Artifacts.toml")
end

# --- Post-build: ensure ALL artifacts are bundled ---
# PackageCompiler sometimes misses artifacts. Copy everything from the depot.
artifacts_dst = joinpath(out_dir, "share", "julia", "artifacts")
mkpath(artifacts_dst)

depot_artifacts = joinpath(first(DEPOT_PATH), "artifacts")
if isdir(depot_artifacts)
    local n_copied = 0
    for entry in readdir(depot_artifacts)
        src_path = joinpath(depot_artifacts, entry)
        dst_path = joinpath(artifacts_dst, entry)
        if isdir(src_path) && !isdir(dst_path)
            cp(src_path, dst_path; force=true)
            n_copied += 1
        end
    end
    println("Bundled $n_copied additional artifacts from depot")
end

# Copy web assets (public/, public_react/, src/, models/) into the output
web_out = joinpath(out_dir, "web")
mkpath(web_out)

# Copy public/ directory (legacy/static assets)
cp(joinpath(src_dir, "public"), joinpath(web_out, "public"); force=true)

# Copy public_react/ directory (built React frontend; run `npm run build` in
# web/frontend/ before this script if the bundle is stale)
public_react_src = joinpath(src_dir, "public_react")
if isdir(public_react_src)
    cp(public_react_src, joinpath(web_out, "public_react"); force=true)
    println("Bundled public_react/ (React frontend)")
else
    println("WARNING: web/public_react/ missing. Run `npm run build` in web/frontend/ first.")
end

# Copy src/ directory (needed for include() at runtime)
cp(joinpath(src_dir, "src"), joinpath(web_out, "src"); force=true)

# Copy models/ directory (XGBoost impact prediction)
models_src = joinpath(src_dir, "models")
if isdir(models_src)
    cp(models_src, joinpath(web_out, "models"); force=true)
    println("Bundled models/ (XGBoost predictors)")
end

# Copy app.jl
cp(joinpath(src_dir, "app.jl"), joinpath(web_out, "app.jl"); force=true)

# Copy LocalPreferences.toml (suppresses CUDA artifact via XGBoost_GPU_jll)
prefs_src = joinpath(src_dir, "LocalPreferences.toml")
if isfile(prefs_src)
    cp(prefs_src, joinpath(web_out, "LocalPreferences.toml"); force=true)
    println("Bundled LocalPreferences.toml")
end

# Try to bundle ffmpeg alongside the exe (needed for animation export)
ffmpeg_path = Sys.which("ffmpeg")
if ffmpeg_path !== nothing
    bin_dir = joinpath(out_dir, "bin")
    cp(ffmpeg_path, joinpath(bin_dir, basename(ffmpeg_path)); force=true)
    println("Bundled ffmpeg from: $ffmpeg_path")
else
    println("WARNING: ffmpeg not found on PATH.")
    println("Animation export (GIF/MP4) will not work without ffmpeg.")
    println("Install it: winget install ffmpeg")
end

# Write launcher .bat that sets DEPOT_PATH and disables Postgres.
# Users double-click this rather than the .exe directly.
bat_path = joinpath(out_dir, "NuclearDetonation.bat")
open(bat_path, "w") do io
    println(io, "@echo off")
    println(io, "set APP_DIR=%~dp0")
    println(io, "set JULIA_DEPOT_PATH=%APP_DIR%share\\julia")
    println(io, "set NUCDET_DISABLE_DB=1")
    println(io, "\"%APP_DIR%bin\\NuclearDetonation.exe\"")
end
println("Wrote launcher: $bat_path")

println()
println("="^60)
println("  Build complete!")
println("  Launcher: $bat_path")
println("  Executable: $(joinpath(out_dir, "bin", "NuclearDetonation.exe"))")
println("  Distribute the entire '$(basename(out_dir))' folder.")
println("="^60)
