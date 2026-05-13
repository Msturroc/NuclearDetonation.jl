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

# --- Relocatable JLL paths (Windows) ---
# JLLs bake their artifact path at precompile time. If we let Julia use the
# build user's home (C:\Users\<builder>\.julia), the resulting exe only works
# for that exact user. Instead, route Julia's depot through a junction at
# C:\Users\Public\NuclearDetonationDepot — a path writable by any standard
# user on any Windows machine. The launcher .bat creates the same-named
# junction → the bundle's share\julia at runtime, so baked paths resolve.
const FIXED_DEPOT = raw"C:\Users\Public\NuclearDetonationDepot"

if Sys.iswindows()
    real_depot = first(DEPOT_PATH)
    if lowercase(real_depot) != lowercase(FIXED_DEPOT)
        # Remove any stale junction/dir at FIXED_DEPOT and re-create pointing
        # at the build user's real depot so artifacts/packages are reused
        # without re-downloading.
        if ispath(FIXED_DEPOT)
            try
                run(`cmd /c rmdir "$FIXED_DEPOT"`)
            catch e
                error("Could not remove existing $FIXED_DEPOT. Delete it manually and re-run. ($e)")
            end
        end
        run(`cmd /c mklink /J "$FIXED_DEPOT" "$real_depot"`)
        println("Created build-time depot junction: $FIXED_DEPOT -> $real_depot")
        empty!(DEPOT_PATH)
        push!(DEPOT_PATH, FIXED_DEPOT)
        ENV["JULIA_DEPOT_PATH"] = FIXED_DEPOT
        println("DEPOT_PATH rewritten to: $(DEPOT_PATH)")
    end
end

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
    # Don't bake transitive deps into the sysimage. CairoMakie/PlotlyJS are deps
    # of NuclearDetonation (for examples) but unused by the GUI; pulling them in
    # crashes Julia 1.12 during sysimage compile (Colors.jl conversions.jl:616).
    # Only our direct [deps] (HTTP, JSON3, NuclearDetonation, NCDatasets, etc.)
    # land in the sysimage; the rest stays in the bundled depot, available but
    # lazy-loaded if anything reaches for it (nothing in julia_main does).
    include_transitive_dependencies = false,
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

# Write launcher .bat. Creates a junction at the same FIXED_DEPOT path used at
# build time so the JLL-baked artifact paths resolve to the bundled depot.
# C:\Users\Public\ is writable by standard users on all Windows installs, so
# the junction step doesn't need admin.
bat_path = joinpath(out_dir, "NuclearDetonation.bat")
open(bat_path, "w") do io
    println(io, "@echo off")
    println(io, "set APP_DIR=%~dp0")
    println(io, "set FIXED_DEPOT=$FIXED_DEPOT")
    println(io)
    println(io, "rem Re-point the build-time depot junction at this install.")
    println(io, "if exist \"%FIXED_DEPOT%\" rmdir \"%FIXED_DEPOT%\" >nul 2>&1")
    println(io, "mklink /J \"%FIXED_DEPOT%\" \"%APP_DIR%share\\julia\" >nul")
    println(io, "if errorlevel 1 (")
    println(io, "  echo Failed to create depot junction at %FIXED_DEPOT%.")
    println(io, "  echo This usually means another user already has it pointed elsewhere,")
    println(io, "  echo or C:\\Users\\Public is not writable on this machine.")
    println(io, "  pause")
    println(io, "  exit /b 1")
    println(io, ")")
    println(io)
    println(io, "set JULIA_DEPOT_PATH=%FIXED_DEPOT%")
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
