# Reproducibility guide

## Reference environment

- Python 3.10 or later
- NumPy 1.24 or later
- pandas 2.0 or later
- Matplotlib 3.7 or later
- XGBoost 2.0 or later only for the optional XGBoost surrogate

## Baseline reference command

```bash
python -m pip install -e .
stadium-ingress simulate \
  --config configs/baseline.json \
  --n-agents 15000 \
  --replications 30 \
  --base-seed 42 \
  --output-dir outputs/baseline
```

This evaluates seeds 42 through 71 inclusive. The cleaned implementation gives
the following ensemble-mean reference values:

| Measure | Reference value |
|---|---:|
| Numerical arrival-rate peak | -16 min |
| Arrival-rate peak value | 0.7382% per min |
| Total-queue peak | -11 min |
| Mean queue length at peak | 2,813.1 spectators |

The arrival-rate maximum lies within a broad capacity-limited plateau and is
not a unique structural breakpoint.

## Random seeds and common random numbers

The command line uses consecutive seeds beginning at `--base-seed`. A
replication has a local legacy NumPy stream, preserving the appendix draw
sequence for S0, and a separately seeded decision-order stream. Factorial and
sensitivity analyses reuse the same seed set across alternatives to reduce
Monte Carlo noise in contrasts.

## Output audit

Before using a result in a manuscript, confirm:

1. the configuration, spectator count, seed range, and replications;
2. whether `spatial_paths_loaded` is true or the duration fallback was used;
3. the number not arrived by the finite simulation end;
4. that figures, CSV files, and metadata came from the same output directory;
5. that Pareto solutions were re-evaluated in the simulation rather than
   reported directly from surrogate predictions;
6. that held-out surrogate validation was conducted for any accuracy claim.

## OSM reproducibility

The repository does not ship a validated OSM snapshot. A spatial replication
must archive or identify the OSM extract date, routing profile, coordinate
reference system, generated JSON files, and required attribution. Without
these files, only temporal results are reproducible.
