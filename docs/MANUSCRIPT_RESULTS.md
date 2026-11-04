# Final reproducible manuscript experiments

This document records the final experiment suite generated from
`configs/reference_calibration.json`. It is intended to make the reported
manuscript values traceable to executable scripts and saved seed level output.

## Evidential scope

The behavioural population is synthetic. Network coordinates and route times
are documented model inputs. The reference capacity scalar was fixed using four
aggregate waiting time outcomes at the reference operating condition. The 16
interaction cells, external surrogate tests, frontier verification, fixed
budget comparison, and stress scenarios were excluded from calibration.

## Interaction experiment

Run `scripts/run_interaction_grid.py` with four demand multipliers, four
capacity multipliers, and 30 common seeds. The resulting 1,920 ABS runs support
complementarity in 5 cells and substitution in 11; no interval includes zero.
At the reference condition, the interaction contrast is minus 3.493 minutes
with a 95 percent confidence interval from minus 3.545 to minus 3.441.

Separate descriptive regressions across the 16 cells have the following
coefficients of determination:

| Descriptor | R squared |
| --- | ---: |
| Normalized demand to capacity ratio | 0.078 |
| Peak facility load | 0.103 |
| Peak arrival pressure | 0.073 |
| Arrival concentration | 0.848 |

Saved outputs are in `outputs/manuscript_full/interaction` when the experiment
is run locally.

### Boundary focused interaction grid

The transition region is evaluated on a separate 5 by 5 grid with demand
scales from 0.80 to 1.00, capacity scales from 0.68 to 1.02, and 30 new matched
seeds per cell:

```bash
python scripts/run_interaction_grid.py \
  --config configs/reference_calibration.json \
  --demand-scales 0.80,0.85,0.90,0.95,1.00 \
  --capacity-multipliers 1.00,1.125,1.25,1.375,1.50 \
  --replications 30 \
  --base-seed 70042 \
  --workers 4 \
  --output-dir outputs/manuscript_full/interaction_dense
```

The 3,000 ABS runs support complementarity in 11 cells and substitution in 14;
no interval includes zero. The closest contrast is minus 0.021 minutes at
demand 0.90 and capacity 0.85, with a 95 percent interval from minus 0.041 to
minus 0.001. Linear interpolation across adjacent signs places the local
transition at normalized demand to capacity ratios from 0.687 to 0.706. This is
a local boundary and not a global threshold.

## Portfolio search and verification

Run `scripts/run_portfolio_experiment.py` with 100 training designs and 3 seeds
per design. The external test contains 30 separate designs and 5 seeds per
design. The initial XGBoost model has external R squared 0.964 and RMSE 0.474.
Independent simulation of 20 spaced frontier points retains 75 percent as
nondominated, with 11.59 percent hypervolume degradation.

Run `scripts/refine_portfolio_experiment.py` to add the 20 initially verified
frontier designs to the training data, refit once, and verify the final frontier
with new seeds. The final model has external R squared 0.954 and RMSE 0.530.
The search returns 143 predicted Pareto solutions. Of 20 independently verified
points, 60 percent remain nondominated. Mean congestion degradation is 5.75
percent and hypervolume degradation is 8.97 percent.

The independently simulated balanced compromise has:

| Measure | Value |
| --- | ---: |
| Mean waiting time | 3.480 minutes |
| Peak total queue | 1,110.2 spectators |
| Congestion score | 4.220 |
| Resource index | 7.760 |

Relative to the matched baseline, waiting time falls by 57.0 percent, peak
queue by 71.5 percent, and congestion score by 60.5 percent.

## Fixed budget comparison

Run `scripts/run_fixed_budget_from_training.py` to compare quadratic and
XGBoost surrogates using the same initial 300 ABS training runs and 600 ABS
verification runs per model.

| Model | External R squared | Retained nondominance | Mean congestion degradation | Hypervolume degradation |
| --- | ---: | ---: | ---: | ---: |
| Quadratic | 0.994 | 0.90 | 21.30 percent | 33.62 percent |
| XGBoost | 0.964 | 0.75 | 8.19 percent | 11.59 percent |

The nondominated subset of the shared simulation design is a direct screening
reference, not a global direct simulation optimiser.

### Direct simulation optimisation benchmark

Run the direct ABS NSGA II benchmark with the same 300 search ABS runs used to
train each initial surrogate:

```bash
python scripts/run_direct_simulation_benchmark.py \
  --config configs/reference_calibration.json \
  --fixed-budget-dir outputs/manuscript_full/fixed_budget \
  --population-size 20 \
  --generations 4 \
  --search-replications 3 \
  --verification-replications 30 \
  --base-seed 42 \
  --workers 4 \
  --output-dir outputs/manuscript_full/direct_benchmark
```

The direct method evaluates 100 candidate slots and returns seven frontier
points. Six remain nondominated under independent verification. Under a common
hypervolume reference, the verified direct frontier is 31.46 percent below the
initial XGBoost frontier. The quadratic frontier is 8.88 percent below XGBoost.
The direct estimates have only 0.40 percent mean absolute congestion
degradation, indicating that limited frontier coverage, not direct estimation
bias, is the principal disadvantage at this budget.

## Scenario reoptimisation

Run `scripts/run_scenario_reoptimization.py`. Each of four scenarios uses 60
designs, 3 training seeds, and 30 independent verification seeds. The scenario
specific policy dominates the fixed balanced policy at 130 percent demand.
Under capacity stress it lowers congestion while using more resources, so the
policies remain incomparable in the two objective sense.

## Final budget and seed separation

The reported suite contains 8,760 ABS runs and 351,400 surrogate evaluation
slots. The boundary focused grid adds 3,000 runs, and the direct benchmark adds
300 search runs plus 210 runs that verify all seven returned frontier points.
Training, external testing, frontier verification, direct comparison, and
scenario verification use documented seed ranges. Preliminary calibration and
code audit runs are not included in the reported experiment budget.

Generated output directories are ignored by Git because seed level data are
large. Every table is regenerated by the scripts in `scripts/`, and each
script writes CSV output plus JSON metadata where applicable.

The four manuscript figures that depend on the final specification are
regenerated from the saved CSV results with:

```bash
python scripts/run_temporal_profile.py \
  --config configs/reference_calibration.json \
  --representatives outputs/manuscript_full/portfolio_refined/representative_policies.csv \
  --base-seed 60042 \
  --replications 30 \
  --output-dir outputs/manuscript_full/temporal_profile
```

This matched run records the minute level arrival and total queue profiles for
the baseline and balanced compromise using the same independent seeds as the
representative policy verification.

The figure generation command is:

```bash
python scripts/generate_manuscript_figures.py \
  --results-dir outputs/manuscript_full \
  --output-dir path/to/manuscript
```

This produces the updated computational framework, temporal profiles, matched
interaction classification, and independently verified Pareto frontier
figures.
