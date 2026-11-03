# Additional statistical and computational experiments

## Scope

These experiments were run on 3 November 2026 with the public reconstructed
model and its default 15,000 spectator configuration. The public model is not
the final 30,000 spectator, seven decision variable, three objective model
described in the manuscript. The results below validate the new analysis code
and must not be substituted for results from the final research model.

## Matched interaction intervals

Mean queue waiting time was evaluated with 30 matched seeds in each of 16
demand and capacity cells. Capacity values 0.225, 0.300, 0.375, and 0.450 are
75%, 100%, 125%, and 150% of the public model reference capacity scale of 0.300.
Each table entry is the mean interaction followed by its 95% confidence
interval in minutes.

| Demand scale | 75% capacity | 100% capacity | 125% capacity | 150% capacity |
| --- | ---: | ---: | ---: | ---: |
| 0.8 | -4.957 [-5.131, -4.782] | -1.247 [-1.321, -1.173] | -0.624 [-0.660, -0.587] | -1.006 [-1.054, -0.958] |
| 1.0 | -4.826 [-4.980, -4.672] | -3.975 [-4.095, -3.855] | -1.674 [-1.724, -1.623] | -0.689 [-0.710, -0.667] |
| 1.2 | -4.319 [-4.453, -4.185] | -6.062 [-6.201, -5.924] | -3.773 [-3.896, -3.651] | -1.170 [-1.228, -1.112] |
| 1.3 | -3.962 [-4.094, -3.830] | -5.810 [-5.955, -5.666] | -4.374 [-4.509, -4.239] | -2.590 [-2.687, -2.493] |

Every public model cell supports substitution. The positive low pressure
interaction reported in the manuscript is therefore not reproduced by this
reconstructed implementation.

## Mechanism diagnostics

The dependent variable is the cell mean interaction. The models are descriptive
standardized OLS regressions across 16 cells.

| Predictors | R squared | Adjusted R squared |
| --- | ---: | ---: |
| Normalized demand to capacity ratio | 0.568 | 0.537 |
| Joint facility backlog load | 0.325 | 0.277 |
| Joint peak arrival pressure | 0.588 | 0.559 |
| Joint arrival concentration | 0.170 | 0.110 |
| Ratio and facility load | 0.751 | 0.713 |
| Combined four predictor model | 0.794 | 0.719 |

The diagnostics explain variation in the magnitude of substitution. They do
not explain a sign transition because the reconstructed model produced no
positive or inconclusive cells.

## Fixed budget benchmark

Both surrogates used the same 30 design configurations and three simulation
replications per configuration. Ten surrogate solutions per backend were then
evaluated with ten independent replications.

| Backend | Initial ABS runs | Verification runs | Retained nondominated share | Mean absolute congestion degradation | Verified to direct hypervolume ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Polynomial | 90 | 100 | 0.90 | 305.6% | 0.952 |
| XGBoost | 90 | 100 | 1.00 | 51.7% | 0.925 |

The direct comparator is the nondominated subset of the shared Latin hypercube
design and is not a global direct simulation optimiser. The small design is
adequate for testing the workflow but not for a surrogate accuracy claim.

## Scenario specific reoptimisation

The reference policy and each scenario specific compromise were evaluated with
20 common independent seeds. Congestion difference is fixed minus reoptimised,
and cost difference uses the same sign convention.

| Demand scale | Congestion difference | Cost difference | Balanced score difference | Dominance relation |
| --- | ---: | ---: | ---: | --- |
| 1.1 | -2.973 | 1,576.9 | -0.285 | Incomparable |
| 1.2 | -2.848 | 1,099.3 | -0.108 | Incomparable |
| 1.3 | -5.066 | 1,140.9 | -0.217 | Incomparable |

The fixed policy achieved lower congestion at higher cost in every scenario.
Neither policy dominated the other. These results show why baseline robustness
does not establish near optimality and why a multiobjective comparison cannot
be reduced to congestion regret without an explicit preference function.

## Reproducibility note

The commands and output definitions are documented in `README.md` and
`docs/ANALYSIS.md`. Generated result files remain outside version control by
default. Reproduce the experiments with the documented seeds and settings
before using the values in another model version.
