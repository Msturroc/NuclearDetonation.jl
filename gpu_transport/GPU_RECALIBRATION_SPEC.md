# GPU recalibration spec — 6 US/NTS tests, corrected loss, log10 search space

Implementation brief for the CUDA-equipped machine. The coordinating (CPU-only)
session cannot compile CUDA, so this spec is written to be implemented and tested
where a GPU exists. Branch: `gpu-calibration-logspace` (off `gpu-transport`).

## Goal

Run BIPOP-CMA-ES calibration on the GPU for **6 tests** — Trinity, Harry,
SmallBoy, Nancy, Smoky (the RIVM-5) plus Doppler (exploratory, not in the RIVM
deliverable) — all using **one identical loss function** and **log10 search
space**, at **10,000 particles** and **6000 evaluations** each, seeded from the
existing best-fit vectors.

RIVM deliverable = 5 tests × {pre, post} = 10 files, H+12 dose rate in mR/hr on a
WGS84 grid, with CRS recorded. Doppler is run too but flagged exploratory.

## Why this is not a copy-paste

Three blockers were found in the current GPU code:

1. **`host_shadow_v2.jl` line 455 hardcodes `n_particles = 1000`** — not read from
   env. Must become configurable for the 10,000-particle requirement.
2. **`host_shadow_v2.jl` lines 461–471 use fixed Nancy cylinders**
   (`LAYER_LOWER/MIDDLE/UPPER`, the 20-param Nancy geometry). Trinity/Harry/
   SmallBoy/Doppler are **23-param** with *tunable* cloud heights in
   `params[21..23]`. The GPU release builder must construct cylinders from those
   params (see §3).
3. **The GPU `rho_core_gpu` loss is the OLD formula** (no bearing term, no gate,
   no log10). It must be replaced by the corrected shared loss (§1).

Only Nancy has a real working GPU runner; the Smoky GPU runner is a stub. Trinity,
Harry, SmallBoy, Doppler have **no** GPU runner yet.

---

## §1 — Shared scoring + reparam module

Create `gpu_transport/calibration_shared.jl`. This is the single source of truth
for the loss + the log10 transform, imported by every runner (CPU and GPU) so the
formula can never drift again. It must contain, verbatim from the corrected CPU
version in `examples/calibration_us_tests/cmaes_calibration.jl` (main working tree
of the CPU box; reproduced here):

### Combined loss (replaces old 0.35·FMS + 0.20·shape + 0.15·extent + 0.30·TOA)

```julia
# default tests
combined = 0.25*geo_mean_fms + 0.15*geo_mean_shape + 0.20*bearing_score +
           0.10*extent_score + 0.30*toa_score
# doppler override (degenerate low-dose contours)
combined_doppler = 0.45*geo_mean_fms + 0.10*geo_mean_shape + 0.20*bearing_score +
                   0.05*extent_score + 0.20*toa_score
```

### Hard gate (after computing combined, before returning)

```julia
if bearing_score < 0.5
    return (loss = 2.0, fms = geo_mean_fms, shape = geo_mean_shape,
            bearing = bearing_score, extent = extent_score, toa = toa_score,
            combined_old = combined_old)
end
```

`loss = 2.0` (not Inf) keeps a finite gradient toward fixing bearing. This stops
geometric-cheating solutions (plume aimed >~45° off) from ever ranking best — the
exact bug that let SmallBoy point east instead of NE.

### Bearing score

The GPU runners currently do NOT compute `bearing_score`. Port it from the CPU
`cmaes_calibration.jl`: dose-rate-weighted cos⁴(Δbearing) of model-vs-obs plume
bearing per contour, using `OBS_BEARINGS` (precomputed per test from the obs
contours) and the model mask centroid bearing from source. Copy `centroid_bearing`
+ the `OBS_BEARINGS` precompute block from cmaes_calibration.jl.

### log10 reparameterisation

```julia
const LOG_MASK = let m = falses(N_DIM); for j in 8:19; m[j]=true; end; m end
encode_params(x) = Float64[LOG_MASK[j] ? log10(x[j]) : x[j] for j in 1:N_DIM]
decode_params(s) = Float64[LOG_MASK[j] ? 10.0^s[j] : s[j] for j in 1:N_DIM]
const LB_S = encode_params(LB)
const UB_S = encode_params(UB)
```

Params 8–19 are the multiplicative scales (turbulence, physics, deposition,
activity). CMA-ES runs entirely in encoded space; **decode before every
`run_gpu_shadow` call and before writing any checkpoint** so stored vectors stay
physical. Verified on CPU: `decode∘encode` is identity to 1e-15, and reproduces
Nancy's known artifact behaviour. See cmaes_calibration.jl for the exact wiring
points (CMAES constructed with `encode_params(run_x0)`, `lb=LB_S`, `ub=UB_S`;
`evaluate_generation` calls `rho_core_gpu(decode_params(candidates[i]), …)`;
`global_best_x = decode_params(gen_best_x)`; small-restart random component drawn
in encoded space).

---

## §2 — GPU particle count configurable

In `host_shadow_v2.jl`, replace the hardcoded `n_particles = 1000` (≈ line 455)
with:

```julia
n_particles = parse(Int, get(ENV, "N_PARTICLES", "10000"))
```

Then audit the kernel launch config (`gpu_kernel_v2.jl`): block/grid sizing,
shared-memory, and any fixed-size device arrays sized to 1000 must scale to
N_PARTICLES. **This is the highest-risk change — test Nancy at N_PARTICLES=10000
against the existing 1000-particle Nancy result first and confirm the dose field
is consistent (not just that it runs).**

---

## §3 — GPU release geometry from tunable heights (23-param)

In `host_shadow_v2.jl`, the source block (≈ lines 461–471) currently uses fixed
`LAYER_LOWER/MIDDLE/UPPER`. Replace with cylinders built from `params[21..23]`,
mirroring the CPU 23-param path exactly (from
`examples/smoky_example/smoky_cmaes_particle_size.jl`):

```julia
# 23-param geometry: sort heights, normalise fractions
stem_top_m, cap_mid_m, cloud_top_m = sort([params[21], params[22], params[23]])
frac_lower_raw, frac_middle_raw = params[6], params[7]
frac_upper_raw = max(1.0 - frac_lower_raw - frac_middle_raw, 0.05)
ftot = frac_lower_raw + frac_middle_raw + frac_upper_raw
frac_lower, frac_middle, frac_upper = frac_lower_raw/ftot, frac_middle_raw/ftot, frac_upper_raw/ftot

layer_lower  = Transport.CylinderRelease(0.0, stem_top_m, 0.2 * stem_top_m)
layer_middle = Transport.CylinderRelease(stem_top_m, cap_mid_m, 0.25 * (cap_mid_m - stem_top_m))
layer_upper  = Transport.CylinderRelease(cap_mid_m, cloud_top_m, 0.25 * (cloud_top_m - cap_mid_m))
```

Then `ReleaseSource((RELEASE_X, RELEASE_Y), layer_*, BombRelease(0.0),
[total_activity*frac_*], n_*)` per layer.

**Nancy stays 20-param fixed-geometry** — keep its existing path. Make the geometry
selection switch on whether the runner is Nancy (20-param) or a US test (23-param),
or simpler: give Nancy a 23-param wrapper using its known fixed heights. Decide
based on what keeps the GPU kernel single-codepath; document the choice.

---

## §4 — Per-test config + ERA5 via Zenodo artifacts

The US-test ERA5 is now on Zenodo (added to `Artifacts.toml` on this branch).
DOIs: trinity 20509225, harry 20509233, smallboy 20509237, doppler 20509247.
Each artifact extracts to `<name>_era5_data/*_snap.nc`.

Add `us_test_era5_files(name)` to `src/transport/data_access.jl` mirroring
`nancy_era5_files()` (artifact_hash → ensure_artifact_installed → readdir the
`<name>_era5_data` subdir). Then generalise `met_upload.jl`'s
`load_nancy_gpu_windows()` into `load_gpu_windows(met_cache, file_start, file_end)`
so each runner passes its own cache/range.

Per-test config (yield, source lat/lon, detonation datetime, met anchor file
index, obs loader) is in `examples/calibration_us_tests/cmaes_calibration.jl`
`TEST_CONFIG` block — copy those values. **Critical met-anchor note:** SmallBoy
detonates 18:30 UTC so its met cache must start at the 18:00 file (index 7 in its
snap sequence), NOT noon — this bug was fixed on CPU; replicate it. Trinity 11:29,
Harry 12:05, Doppler 12:30 anchor to their nearest preceding hourly file.

---

## §5 — Six runners

Create `gpu_transport/runners/gpu_<test>_bipop_cmaes.jl` for trinity, harry,
smallboy, doppler; update nancy (real) and smoky (currently a stub). Best: a
single parameterised `gpu_bipop_cmaes.jl <test>` (like the CPU
cmaes_calibration.jl) importing `calibration_shared.jl`, so the loss exists once.

Each runner:
- `N_PARTICLES=10000`, `MAX_EVALS=6000`
- warm-start from the existing best vector (see §6)
- log10 search space (§1), corrected loss (§1)
- writes `<test>_gpu_best.txt` + `<test>_gpu_results.txt` in **physical units**

---

## §6 — Seed vectors (warm starts)

Use the existing best fits as `x0`:
- Trinity: `examples/calibration_us_tests/trinity_cmaes_ou_best.txt` (23-param, ~76% combined under OLD loss — will re-score under new loss)
- Harry: `harry_cmaes_ou_best.txt`
- SmallBoy: best CPU result is ~72% — on the CPU box at
  `smallboy_cmaes_ou_best.txt.4000eval_72pct`. Coordinator will commit this.
- Nancy: `examples/nancy_cmaes_ou_best.txt` (20-param)
- Smoky: `examples/smoky_example/smoky_cmaes_ou_best.txt` (23-param)
- Doppler: `examples/calibration_us_tests/doppler_cmaes_ou_best.txt`

Note: seeds were fitted under the OLD loss, so initial scores will look different
under the corrected loss — that's expected; the gate may reject a seed if its
bearing < 0.5 (then it starts climbing from loss=2.0).

---

## §7 — pre/post outputs for RIVM

Each test emits two NetCDFs:
- **post** = calibrated (the GPU best vector), H+12 dose mR/hr, WGS84, CF-1.8.
- **pre** = uncalibrated baseline (interp B): same code, parameter vector replaced
  by an uncalibrated default (DASA-1251 median particle sizes, Glasstone-Dolan
  yield-scaled cloud heights, all scale factors = 1.0). The CPU box has a working
  `NC_EXPORT` block in cmaes_calibration.jl producing exactly this CF-1.8 layout
  (vars: `dose_rate_mR_hr`, `dose_rate_raw_mR_hr`, `crs`=WGS84 EPSG:4326,
  reference_time = detonation + 12h). Reuse that writer.

Output dir on GPU box: `rivm_intercomparison/` — 10 NetCDFs (5 tests × pre/post)
+ Doppler pre/post flagged exploratory. README listing per-test param vectors,
score breakdown, CRS, units.

---

## §8 — Validation gates (do these before trusting any fit)

1. **N-particle consistency:** Nancy at 10k particles vs 1k — dose field
   qualitatively identical, score within noise. If not, the kernel scaling (§2) is
   wrong.
2. **Geometry correctness:** for one US test (Trinity), confirm a known-good CPU
   23-param vector produces a GPU dose field matching the CPU forward sim for the
   same vector + seed (within Monte-Carlo noise). This catches §3 release bugs.
3. **Loss identity:** feed one vector through both the CPU `rho_core` (corrected)
   and GPU `rho_core_gpu` (new shared loss) — all 6 score components must match to
   ~3 dp (only the forward sim differs, not the scoring).
4. **Gate behaviour:** confirm a deliberately mis-aimed plume (bearing < 0.5)
   returns loss = 2.0.
5. **Bearing health during run:** every generation's best should keep bearing
   ≥ ~0.7; if it collapses, the gate/term wiring is wrong.

Report which gates pass before launching the full 6×6000-eval campaign.

---

## §9 — Disk

Each US-test ERA5 artifact ≈ 384 MB extracted (4 new) + Nancy/Smoky already
cached. ~10 output NetCDFs are small. Run `df -h` on the Julia depot partition
first; the coordinator was told disk may be tight on the GPU box. Free space
before downloading all 4 artifacts if needed.

---

## Open coordination items (CPU box will handle)

- Commit untracked observations (`data/{trinity,harry,smallboy,doppler}_observations`,
  incl. self-digitised Doppler contours) so the GPU box gets them on clone.
- Commit `smallboy_cmaes_ou_best.txt.4000eval_72pct` as the SmallBoy seed.
- The corrected CPU `cmaes_calibration.jl` is currently untracked on the CPU box;
  its loss/log10 logic is the reference — reproduced in §1 above.
