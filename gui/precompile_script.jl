# Precompile script for PackageCompiler.
# Build the window off-screen and render it once, so the GLMakie drawing paths
# are compiled into the sysimage. No simulation runs here (that'd cost a lot of
# build time for marginal sysimage benefit).
using NuclearDetonationGUI
using GLMakie

GLMakie.activate!(visible = false)
try
    g = NuclearDetonationGUI.build_gui()
    GLMakie.Makie.colorbuffer(g.fig)
catch e
    # Build machines without OpenGL can still produce a working app
    @warn "Skipping GLMakie precompile workload" exception = e
end
