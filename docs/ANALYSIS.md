# Analysis methods

## Spatial-temporal interaction

For every demand/capacity regime, the simulation evaluates four portfolios
with common random seeds: none, shuttle only, pre-event only, and joint.
The command defaults to mean queue waiting time and can also analyse the
composite congestion score, peak queue, or pre-kick-off arrival share.

Let `C` be the congestion score, where lower is better. Separate and joint
benefits are `B_S = C_none - C_spatial`, `B_T = C_none - C_temporal`, and
`B_ST = C_none - C_joint`. Interaction is:

```text
I = B_ST - B_S - B_T
  = C_spatial + C_temporal - C_none - C_joint
```

The contrast is computed for every matched seed. Its mean is reported with a
two-sided 95% Student t interval. Complementarity is statistically supported
when the lower limit is positive, substitution is supported when the upper
limit is negative, and the result is inconclusive when the interval contains
zero. This is a difference-in-differences definition; changing the KPI or sign
convention changes the interpretation.

The interaction output also records three dimensionless mechanism descriptors
for the baseline and joint policies:

- maximum facility queue divided by that facility's per-minute service capacity;
- maximum minute arrivals divided by total per-minute service capacity;
- the Herfindahl index of arrivals across simulation minutes.

Standardised descriptive OLS models compare these variables with the normalized
demand-to-capacity ratio. The regressions contain only the tested cells and are
mechanism diagnostics, not causal estimates.

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
and simulated values to `pareto_verification.csv`. It classifies the predicted
and simulated points by nondominance and computes two-objective hypervolume with
a common reference point. Reported degradation is conditional on the verified
subset. Final reporting should use more verification seeds and a prespecified
stratification rule.

## Scenario-specific reoptimisation

The `scenario-regret` command first selects the reference compromise by minimum
Euclidean distance to the normalized ideal point. It then refits the surrogate
and reruns NSGA-II for every demand and capacity scenario. The fixed reference
policy and scenario-specific compromise are evaluated with common independent
seeds. Congestion difference, cost difference, normalized balanced score, and
the dominance relation are reported. A negative congestion difference can occur
when the fixed policy spends more or when simulation error changes the surrogate
ranking; it is not labelled scalar regret in this multiobjective problem.

## Fixed-budget benchmark

The `fixed-budget-benchmark` command generates one shared simulation design and
fits each requested surrogate to the same replication averages. It compares the
independently evaluated surrogate candidates with the nondominated subset of
the shared ABS design. The output reports ABS calls, surrogate evaluation slots,
retained nondominance, objective degradation, and hypervolume under a common
reference point. The direct comparator is a screening baseline. It does not
establish equivalence to a specialised direct simulation optimisation method.

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
