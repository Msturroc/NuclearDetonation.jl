# Nancy recalibration after the met-window fix (2026-10-09)

The transport loop used to skip the hour between consecutive ERA5 files (3 hourly
steps per file, 2 windows simulated), so the weather ran 1.5x ahead of the model
clock, and the Nancy runs started on the 12:00 weather for a 13:10 UTC shot. Both
are fixed in the package and in the GPU path (`met_window_sequence`).

All runs: GPU forward model, 10k particles, corrected 5-term loss, start point
`gpu_nancy_corrected_best.txt`, about 6000 forward runs each. Held-out = mean ± sd
over 8 seeds none of the runs trained on (`runners/eval_nancy_params.jl`).
Package = the same parameters through `Transport.run_simulation!` (cpu_reference.jl,
10k particles).

| Parameters | Own best | Held-out | Package |
|---|---|---|---|
| previous best, old windows | 82.3 | 81.14 ± 0.41 | |
| previous best, corrected windows | | 78.94 ± 1.38 | |
| BIPOP, new seed per generation (`gpu_nancy_bipop_bridged`) | 83.27 | **82.18 ± 1.01** | **82.43 ± 0.41** |
| champion5, 1 fixed seed, opt seed 1 | 82.48 | 77.76 ± 2.48 | |
| champion5, 1 fixed seed, opt seed 2 | 83.35 | 80.62 ± 1.56 | |
| champion5, unscaled stops / c5-mbh (opt seed 1) | 82.48 | (identical to seed 1) | |
| champion5, 4-seed average, 1500 evals (`gpu_nancy_c5_avg4`) | 80.70 | 80.18 ± 0.79 | |
| champion5, 2-seed average, 3000 evals (`gpu_nancy_c5_avg2`) | 82.21 | **82.15 ± 0.16** | **82.30 ± 0.33** |

Findings:
- champion5 (best_global_optimiser) on a single fixed seed fits that seed's particle
  noise: 3-5 points lost on held-out seeds. Averaging two seeds per evaluation fixes
  it at equal compute and gives the most stable fit (sd 0.16).
- Seed 1 stalled after eval 337 with no restart in 6000 evals; the restart levers
  (unscaled stop windows, basin-hopping restarts) don't change that, since no
  restart is ever triggered.
- BIPOP and champion5-avg2 tie on score with very different parameters (sigma_h 5.1
  vs 0.17, omega 2.9 vs 0.5, fine-mode 38 vs 130 um, activity 60 vs 28 PBq): one
  test does not identify 18 parameters. champion5-avg2 has better overlap (FMS 0.43
  vs 0.36) but sits on five search bounds; BIPOP is interior.

Chosen for `nancy_optimised_config()` on main: the BIPOP set.
Fit plots: `examples/calibration_us_tests/nancy_recal_{bipop,champion5}_cmaes_ou_fit.png`.
