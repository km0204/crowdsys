# Analysis methods

## Spatial-temporal interaction

For every demand/capacity regime, the simulation evaluates four portfolios
with common random seeds: none, shuttle only, pre-event only, and joint.

Let `C` be the congestion score, where lower is better. Separate and joint
benefits are `B_S = C_none - C_spatial`, `B_T = C_none - C_temporal`, and
`B_ST = C_none - C_joint`. Interaction is:

```text
I = B_ST - B_S - B_T
  = C_spatial + C_temporal - C_none - C_joint
```

`I > 0` is labelled complementarity, `I < 0` substitution, and values within
the configured numerical tolerance additive. This is a difference-in-
differences definition; changing the KPI or sign convention changes the
interpretation.

## Experimental design and surrogate

The design varies shuttle awareness, event awareness, fleet size, event
candidate rate, and event target join rate. Each dimension is stratified over
its range and independently permuted. Simulation objectives are averaged over
common consecutive seeds.

The `xgboost` backend uses `XGBRegressor`. The dependency-free fallback is a
ridge-stabilised quadratic response surface including main effects, squares,
and pairwise interactions. The exported diagnostics contain training and
five-fold cross-validated errors. An external test set is still recommended
before final accuracy claims.

## NSGA-II and Pareto set

The included NSGA-II performs non-dominated sorting, crowding-distance
selection, tournament selection, blend crossover, and bounded Gaussian
mutation. It minimises surrogate-predicted congestion and intervention cost.
Fleet size is rounded to an integer after decoding.

The full Pareto CSV contains predictions. The command selects spaced solutions
across the front and re-runs them with a separate seed range, writing predicted
and simulated values to `pareto_verification.csv`. Final reporting should use
more verification seeds and uncertainty intervals.

## Sensitivity analysis

One-at-a-time analysis supports capacity scale, demand scale, shuttle
awareness, and event awareness. Every level uses the same supplied seed set.
This diagnoses local response but does not replace a global variance-based
analysis.

## Robustness analysis

Robustness cases draw independent truncated normal multipliers for demand and
capacity and use a different simulation seed per case. Output includes every
case and mean, standard deviation, 5th percentile, median, and 95th percentile
summaries.
