# Precompile script for PackageCompiler.
# Loading the module is enough to trace and compile the hot import + include
# graph; we don't need to run a simulation here (that'd cost a lot of build time
# for marginal sysimage benefit).
using NuclearDetonationGUI
