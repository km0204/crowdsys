# Code provenance and implementation status

## Core simulation implementation

The core simulation follows the appendix source for the stadium ingress model.
That source describes:

- baseline arrival sampling and six-station capacity-constrained queues;
- heterogeneous time, crowding, food, and stage preferences;
- shuttle capacity, choice, queue, and in-vehicle movement;
- pre-event candidacy, zone utility, calibrated adoption, and dwell time;
- shuttle and event awareness rates;
- direct, station-to-zone, and zone-to-gate OSM path inputs;
- coordinate progress, route congestion, position logs, and CSV/JSON outputs.

The appendix source contained page headers, omitted imports, and incomplete
indentation. The package in `src/stadium_ingress` is therefore a structured
implementation of the specified mechanisms, not a byte for byte copy.

## Repository extensions

The following components were not present in the supplied appendix and were
implemented for this repository:

- common-seed factorial interaction analysis;
- an explicit complementarity/substitution statistic;
- stratified design generation;
- optional XGBoost and built-in quadratic surrogates;
- a dependency-free NSGA-II implementation;
- Pareto CSV and figure generation;
- one-at-a-time sensitivity analysis;
- stochastic demand/capacity robustness analysis;
- contextual ensembles for four matched intervention responses;
- interaction and Pareto uncertainty acquisition;
- held out boundary grid assessment;
- direct ABS NSGA II under a fixed simulation budget;
- package structure, command-line interface, tests, CI, and documentation.

The reported experiment suite uses `configs/reference_calibration.json` and the
scripts listed in `docs/MANUSCRIPT_RESULTS.md`. The provenance record documents
the implementation basis and does not constitute empirical validation of the
synthetic behavioural population or network inputs.
