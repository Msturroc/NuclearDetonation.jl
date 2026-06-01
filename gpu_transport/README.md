# gpu_transport — GPU forward simulator for the Nancy/Smoky Lagrangian model

> **Provenance.** This directory was rescued from uncommitted scratch work in
> `/home/marc/julia_snap_explorations/` (files dated April 2026) and committed
> here on 2026-06-02 so it isn't lost. It is **research/prototype code**, not
> part of the shipped `NuclearDetonation` package — it reuses the package only
> as a library boundary (`Transport.MeteoFields`, met readers, defaults, …).

## What it is

A from-scratch **hand-written CUDA.jl kernel** that ports the per-particle inner
loop of the Lagrangian dispersion model to the GPU — **one thread = one
particle**. This is the "speed-of-light" hand-tuned approach (the same thing you
would write in CUDA-C++), implemented in the *same language* as the rest of the
model. It deliberately does **not** use `DiffEqGPU.jl` / `EnsembleGPUKernel`.

The kernel mirrors the CPU reference (`host_shadow_v2.jl`) **Float32-for-Float32**
so the per-cell difference stays within ~1e-3 relative and the FMS gap versus the
reference stays at the chaotic noise floor (~0.03). One kernel launch per met
window (12 launches for a 12-hour Nancy run), with hourly deposition snapshots
taken between launches.

## Why a GPU at all

The payoff is **inverse modelling**: fitting ~20 physics parameters (bimodal
particle-size distribution, turbulence scales, layer fractions, deposition
velocities, …) to the observed Nancy/Smoky fallout patterns with BIPOP-CMA-ES.
That requires *thousands* of forward dispersion runs — the classic "many forward
solves" ensemble workload where the GPU replaces a per-candidate threaded CPU
evaluator with a single kernel launch. Best fit recorded so far
(`artifacts/gpu_nancy_cmaes_v3_best.txt`): FMS 0.34, shape 0.92, extent 1.0, TOA 1.0.

## Layout

| Path | Role |
|---|---|
| `GpuTransport.jl` | Module entry; includes `hanna.jl`. Exports the Hanna turbulence surface. |
| `gpu_kernel_v2.jl` | The CUDA kernel + device helpers (4D wind interp, settling, deposition). Entry: `run_gpu_shadow(params, gen_seed; windows)`. |
| `host_shadow_v2.jl` | Float32 CPU mirror of the kernel — the validation reference. Entry: `run_host_shadow(params, gen_seed)`. |
| `met_upload.jl` | Uploads ERA5 met windows to the GPU. Entry: `load_nancy_gpu_windows()`. |
| `hanna.jl` | Hanna turbulence scheme (host + device `hanna_dev`, OU step). |
| `cpu_native_integrator.jl`, `cpu_reference.jl` | Native CPU integrators used as cross-checks. |
| `validate_against_reference.jl` | Validation harness: GPU vs CPU via FMS over 6 dose-rate thresholds, PASS/FAIL tolerance; prints `reference: Xs | candidate: Xs (Nx)`. |
| `test_*.jl` | Unit/behaviour tests for the kernel, Hanna, shadow, integrators. |
| `runners/` | Standalone drivers: `gpu_nancy_simulation.jl` (simplified CPU==GPU + speedup bench), `gpu_advection_prototype.jl`, and the `*_bipop_cmaes*.jl` parameter-fitting drivers. |
| `artifacts/` | Fitted-parameter records (`*_best.txt`, `*_results.txt`), CPU-vs-GPU comparison plots, and profiling output. |

## Running

Requires a CUDA GPU and the `NuclearDetonation` project environment (which
resolves `CUDA.jl`). The simplest self-contained check:

```bash
julia --threads=auto --project=/home/marc/NuclearDetonation.jl \
      gpu_transport/runners/gpu_nancy_simulation.jl
# → artifacts/gpu_nancy_comparison.png (CPU vs GPU dose contours + diff map)
```

## Caveats / TODO if integrating into the package

- **Absolute paths.** The `runners/` scripts `include(...)` the original scratch
  paths (`/home/marc/julia_snap_explorations/gpu_transport/...`). Repoint these to
  this directory before running from the repo.
- **Coupling.** `host_shadow_v2.jl` / `validate_against_reference.jl` reference
  bindings defined at the top of the CMA-ES drivers (`MET_CACHE`, `DOMAIN`, `NX`,
  `NY`, `NK`, `RELEASE_X/Y`, `LAYER_*`, `generate_bimodal_bins`, …) rather than
  importing them — they are meant to be `include`d into a driver's scope.
- **Not packaged.** To ship this, the natural home is a CUDA-gated package
  extension (`ext/`) with `CUDA` as a weak dependency, an out-of-place SVector RHS,
  and the absolute includes replaced by package-relative ones.
