# Stadium Ingress Intervention Portfolio Simulation

Reference implementation and experiment drivers for stadium ingress management
under capacity constraints. The repository includes the agent based simulation,
matched interaction inference, contextual XGBoost ensembles, NSGA II search,
frontier verification, staged interaction aware sampling, fixed budget
comparisons, and scenario specific search used in the reported experiments.

Associated study:

> *Interaction Aware Sequential Surrogate Optimization for Capacity Constrained
> Intervention Portfolios: An Application to Stadium Ingress Management*

## What is included

| Component | Status | Provenance |
|---|---|---|
| Baseline arrivals, six-station choice, FIFO queues, walking time | Implemented | Model specification and appendix code |
| Shuttle mode, queue, fleet and headway capacity | Implemented | Model specification and appendix code |
| Pre-event participation, zone choice, dwell time | Implemented | Model specification and appendix code |
| Shuttle and event awareness with behavioural adoption | Implemented | Model specification and appendix code |
| Combined spatial and temporal portfolios | Implemented | Model specification and appendix code |
| OSM coordinate routes, position logs, pedestrian density figure | Available with route JSON inputs | Model specification and documented fallback |
| Capacity dependent complementarity and substitution | Implemented | Analysis module |
| Experimental design and XGBoost or quadratic surrogate | Implemented | Analysis module |
| NSGA II and Pareto solution verification | Implemented | Optimisation and validation modules |
| Staged interaction and Pareto uncertainty acquisition | Implemented | Adaptive module |
| Sensitivity and stochastic robustness analysis | Implemented | Analysis module |

The reference experiment uses the documented coordinate and route time inputs
in `params/reference_network_estimates`. These files are model inputs and are
not represented as a validated OSM snapshot or as observed individual traces.

## Installation

Python 3.10 or later is required.

```bash
git clone https://github.com/km0204/crowdsys.git
cd crowdsys
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .
```

XGBoost is optional. If it is not installed, the optimisation pipeline can use
the included quadratic surrogate.

```bash
python -m pip install -e ".[analysis]"
```

## Scenario runs

Baseline S0:

```bash
stadium-ingress simulate \
  --config configs/baseline.json \
  --replications 30 \
  --base-seed 42 \
  --output-dir outputs/S0
```

The intervention configurations are:

- `configs/S1_shuttle.json`: shuttle plus shuttle awareness;
- `configs/S2_pre_event.json`: pre-event activity plus event awareness;
- `configs/S3_combined.json`: joint shuttle and pre-event portfolio.

For example:

```bash
stadium-ingress simulate \
  --config configs/S3_combined.json \
  --replications 30 \
  --output-dir outputs/S3
```

Each run writes the arrival/queue time series, run metadata, a representative
arrival log, a mode/station choice log, and a PNG figure. If valid OSM paths are
available and position logging is enabled, it also writes a position CSV and a
pedestrian-density PNG.

## Capacity-dependent policy interaction

The `interaction` command evaluates a 2×2 factorial portfolio—none, shuttle,
pre-event, and joint—under common random seeds for each demand/capacity regime.

```bash
stadium-ingress interaction \
  --config configs/baseline.json \
  --demand-scales 0.7,1.0,1.3 \
  --capacity-scales 0.2,0.3,0.4 \
  --replications 10 \
  --refine-inconclusive 10 \
  --interaction-metric mean_queue_wait_minutes \
  --output-dir outputs/interaction
```

The command writes seed-level matched contrasts, 95% confidence intervals, and
a three-level classification: supported complementarity, supported
substitution, or inconclusive. It also exports descriptive mechanism diagnostics
using facility backlog, peak arrival pressure, and arrival concentration. See
[`docs/ANALYSIS.md`](docs/ANALYSIS.md) for the exact definitions.
The optional `--refine-inconclusive` stage assigns additional matched seeds only
to cells whose initial interval contains zero.

## Surrogate assisted NSGA II and Pareto solutions

This command generates a stratified experimental design, runs the simulation,
fits the congestion surrogate, computes the resource index exactly, runs NSGA
II, and exports the estimated Pareto set:

```bash
stadium-ingress optimise \
  --config configs/baseline.json \
  --design-samples 60 \
  --replications 5 \
  --surrogate-backend xgboost \
  --population-size 100 \
  --generations 150 \
  --verify-solutions 10 \
  --verification-replications 10 \
  --output-dir outputs/optimisation
```

Use `--surrogate-backend polynomial` when XGBoost is unavailable. The complete
Pareto CSV contains surrogate predictions. A spaced subset is automatically
re-evaluated with independent simulation seeds and written to
`pareto_verification.csv`. The command also reports retained nondominance,
verified hypervolume, and objective degradation in
`frontier_verification_metrics.json`; increase the verification settings for
final runs.

## Scenario-specific reoptimisation and regret

The following command compares the reference balanced compromise with a new
balanced compromise optimised for each demand and capacity scenario:

```bash
stadium-ingress scenario-regret \
  --config configs/baseline.json \
  --demand-scales 1.1,1.2,1.3 \
  --capacity-scales 1.0 \
  --design-samples 60 \
  --replications 5 \
  --verification-replications 30 \
  --output-dir outputs/scenario_regret
```

`scenario_regret.csv` reports independently simulated congestion differences,
cost differences, normalized balanced scores, dominance relations, and policy
settings. Congestion difference alone is not labelled regret because the
problem is multiobjective.

## Fixed-budget computational benchmark

```bash
stadium-ingress fixed-budget-benchmark \
  --config configs/baseline.json \
  --design-samples 60 \
  --replications 5 \
  --surrogate-backends polynomial,xgboost \
  --verify-solutions 10 \
  --verification-replications 10 \
  --output-dir outputs/fixed_budget_benchmark
```

Both surrogate models use the same simulation design and replication budget.
The direct comparator is the nondominated subset of those shared ABS design
evaluations. It is a fixed-budget direct screening baseline, not a global direct
simulation optimiser.

The separate direct ABS NSGA II benchmark uses the same 300 run search budget
as each initial surrogate. Its command and verification protocol are given in
[`docs/MANUSCRIPT_RESULTS.md`](docs/MANUSCRIPT_RESULTS.md).

The reported experimental protocol, numerical results, seed allocation, and
output files are summarized in
[`docs/MANUSCRIPT_RESULTS.md`](docs/MANUSCRIPT_RESULTS.md). Audit notes in
`docs/` record differences between preliminary and reported specifications.

## Sensitivity and robustness

```bash
stadium-ingress sensitivity \
  --parameter capacity_scale \
  --values 0.2,0.25,0.3,0.35,0.4 \
  --replications 10 \
  --output-dir outputs/sensitivity

stadium-ingress robustness \
  --config configs/S3_combined.json \
  --cases 100 \
  --demand-sd 0.10 \
  --capacity-sd 0.10 \
  --output-dir outputs/robustness
```

## OSM inputs and density maps

Place the validated production files below in `params/`:

```text
walk_times_osm.json
walk_paths_osm.json
event_zones_osm.json
walk_paths_event_osm.json
walk_times_event_osm.json
```

Accepted schemas and coordinate assumptions are documented in
[`params/README.md`](params/README.md). Missing files do not stop the temporal
simulation. Coordinate outputs are simply disabled when routes are absent.

## Repository layout

```text
configs/                    scenario definitions
params/                     OSM input schemas; no validated OSM data included
src/stadium_ingress/        simulation and analysis package
scripts/                    experiment and figure generation scripts
tests/                      deterministic and policy-analysis tests
docs/                       model, data, analysis, and reproducibility notes
examples/                   reference baseline outputs
```

## Scientific-use notes

- `docs/MODEL.md` defines the simulation mechanisms and fallback behaviour.
- `docs/PROVENANCE.md` distinguishes appendix-derived and newly added code.
- `docs/ANALYSIS.md` defines interaction, surrogate, Pareto, sensitivity, and
  robustness methods.
- `docs/REPRODUCIBILITY.md` records the seed policy and verification steps.
- `docs/MANUSCRIPT_RESULTS.md` records the commands and numerical summaries for
  the reported experiment suite.

## Tests

```bash
python -m pip install -e ".[dev]"
pytest
ruff check .
```

