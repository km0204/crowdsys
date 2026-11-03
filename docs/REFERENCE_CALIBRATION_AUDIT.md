# Reference calibration audit

This audit evaluates whether the appendix source and explicitly estimated
network inputs can reproduce the four reference condition waiting time values
reported in the manuscript. It does not treat the estimated spatial inputs as
the original OSM files.

## Protocol

- The reported demand, intervention settings, awareness rates, candidate rate,
  and target participation rate are fixed.
- Only unreported operational and behavioural parameters are screened.
- Calibration targets are limited to the four reference condition aggregates.
- Capacity interaction cells and other robustness scenarios are excluded from
  calibration.
- Waiting time is computed from each simulated spectator's queue entry and
  service times. The appendix summary field `avg_queue_wait_min` is not used
  because it is a time average of station queue length divided by capacity.
- The final audit uses 30 matched seeds for the four intervention conditions.

The executable settings are in `configs/reference_calibration.json`. The five
estimated spatial input files are in `params/reference_network_estimates/`.
Their file and directory names identify them as estimates.

## Results

| Condition | Manuscript value | Audit mean | Relative difference |
| --- | ---: | ---: | ---: |
| No intervention | 8.46 | 8.154 | -3.6% |
| Shuttle only | 4.25 | 3.551 | -16.5% |
| Activities only | 4.62 | 4.674 | 1.2% |
| Joint intervention | 2.48 | 3.564 | 43.7% |

The matched interaction estimate is -3.493 minutes, with standard error 0.025
and a 95% confidence interval from -3.545 to -3.441 minutes. It is classified
as supported substitution under the prespecified rule.

The baseline and activity conditions are close to the manuscript aggregates,
but the shuttle and joint conditions are not. In particular, the joint gap is
too large to attribute to seed variability. Additional screening of route
times and the activity extra walking sensitivity did not resolve this gap
while holding the reported experimental settings fixed.

## Interpretation

These outputs are a reproducibility audit and are not replacement manuscript
results. The audit indicates that the appendix source plus estimated inputs is
insufficient to recover the reported joint intervention result. Adding a
joint-specific correction would fit the target but would introduce an
unsupported mechanism and is therefore excluded.

Before new numerical results are inserted into the manuscript, at least one of
the following is needed:

1. the original scenario and OSM JSON files;
2. the original optimisation and postprocessing scripts that define the
   reported waiting time statistic; or
3. archived seed-level outputs for the four reference conditions.

Run the matched audit with:

```bash
python scripts/run_reference_audit.py \
  --config configs/reference_calibration.json \
  --replications 30 \
  --base-seed 42 \
  --output-dir outputs/reference_audit
```

The command writes `reference_seed_level.csv` and `reference_summary.csv`.
