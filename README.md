# Stadium Ingress Intervention Portfolio Simulation

Reproducible agent-based simulation and analysis code for stadium ingress under
capacity constraints. The repository contains a cleaned implementation of the
appendix model supplied by the study author, plus clearly separated analysis
modules for policy interaction, surrogate modelling, NSGA-II optimisation,
Pareto analysis, sensitivity analysis, and robustness analysis.

Associated study:

> *Optimising Behaviourally Responsive Intervention Portfolios in
> Capacity-Constrained Crowd Systems: A Surrogate-Assisted Simulation
> Optimization Approach*

## What is included

| Component | Status | Provenance |
|---|---|---|
| Baseline arrivals, six-station choice, FIFO queues, walking time | Implemented | Cleaned from the supplied appendix |
| Shuttle mode, queue, fleet/headway capacity | Implemented | Cleaned from the supplied appendix |
| Pre-event participation, zone choice, dwell time | Implemented | Cleaned from the supplied appendix |
| Shuttle/event awareness and behavioural adoption | Implemented | Cleaned from the supplied appendix |
| Combined spatial and temporal portfolios | Implemented | Cleaned and modularised from the appendix |
| OSM coordinate routes, position logs, pedestrian density figure | Implemented when OSM JSON is supplied | Appendix logic, safe fallback added |
| Capacity-dependent complementarity/substitution | Implemented | New release analysis module |
| Design-of-experiments and XGBoost or quadratic surrogate | Implemented | New release analysis module |
| Dependency-free NSGA-II and Pareto solutions | Implemented | New release analysis module |
| One-at-a-time sensitivity and stochastic robustness | Implemented | New release analysis module |

The repository does not contain a validated OSM snapshot. Without the five OSM
JSON inputs, walking is simulated by station-level duration and spatial maps are
not produced. Example coordinate files document the schema only and must not be
treated as research data.

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
  --output-dir outputs/interaction
```

The reported difference-in-differences is positive for complementarity,
negative for substitution, and close to zero for additive effects. See
[`docs/ANALYSIS.md`](docs/ANALYSIS.md) for the exact definition.

## Surrogate-assisted NSGA-II and Pareto solutions

This command generates a stratified experimental design, runs the simulation,
fits two surrogate objectives, runs NSGA-II, and exports the estimated Pareto
set:

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
`pareto_verification.csv`; increase the verification settings for final runs.

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
- `PUBLIC_RELEASE_CHECKLIST.md` lists the author decisions required before the
  repository is made public.

## Tests

```bash
python -m pip install -e ".[dev]"
pytest
ruff check .
```

