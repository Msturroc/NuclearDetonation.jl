# Running the Nancy and Smoky examples

## Nancy (Upshot-Knothole Nancy, 24 kT, 1953)

```bash
cd examples
julia --project=.. nancy_bomb_release.jl
```

Output: `examples/nancy_bomb_release.png` (observed vs model, dose rate + time-of-arrival).

## Smoky (Plumbbob Smoky, 44 kT, 1957)

```bash
cd examples/smoky_example
julia --project=../.. smoky_bomb_release.jl
```

Output: `examples/smoky_example/smoky_bomb_release.png`.

## Notes

- **Don't run them in parallel on a fresh checkout.** Both depend on `NCDatasets`; if it's not yet precompiled, two Julia processes will collide on the precompile pidfile and both fail. Either run sequentially, or warm the cache first with `julia --project=. -e 'using Pkg; Pkg.precompile()'`.
- Reference (working) Smoky figure used in the EPA talk is checked in at `examples/nancy_fms_plots/smoky_bomb_release.png`. The current `smoky_cmaes_ou_best.txt` params were calibrated against the pre-`a4804dc` (`Fix time-blend in ReferenceTrilinearInterpolant`) transport behaviour, so today's run won't match Figure 10 without re-tuning.
