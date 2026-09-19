# Code provenance and implementation status

## Supplied appendix

The study author supplied a pasted appendix headed “Base model for ingress
(pre-kickoff arrivals) at Japan National Stadium”. It describes:

- baseline arrival sampling and six-station capacity-constrained queues;
- heterogeneous time, crowding, food, and stage preferences;
- shuttle capacity, choice, queue, and in-vehicle movement;
- pre-event candidacy, zone utility, calibrated adoption, and dwell time;
- shuttle and event awareness rates;
- direct, station-to-zone, and zone-to-gate OSM path inputs;
- coordinate progress, route congestion, position logs, and CSV/JSON outputs.

The pasted text had page headers embedded in the code, omitted imports, and lost
Python indentation. The package in `src/stadium_ingress` is therefore a cleaned
reconstruction of those mechanisms, not a byte-for-byte copy.

## Added for the public research workflow

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
- package structure, command-line interface, tests, CI, and documentation.

These additions need comparison with the final manuscript methods and original
research scripts before the repository is described as reproducing every
published numerical result.
