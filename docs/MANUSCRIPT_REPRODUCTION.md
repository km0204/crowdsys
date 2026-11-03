# Manuscript result reproduction check

## Purpose

The pasted appendix code was compared with the numerical values reported in the
manuscript. The appendix was the source for the cleaned public implementation,
but it does not include the configuration overrides, OSM inputs, optimisation
scripts, cost specification, or constraint thresholds used for the reported
experiments.

## Reference cell comparison

Thirty common seeds were used for the baseline, shuttle, activity, and joint
policies at the reference capacity condition. The manuscript table settings
were applied to shuttle headway, fleet size, vehicle capacity, activity
candidacy, target participation, duration, and awareness.

| Model and outcome definition | Baseline | Shuttle | Activity | Joint | Interaction |
| --- | ---: | ---: | ---: | ---: | ---: |
| Manuscript values | 8.46 | 4.25 | 4.62 | 2.48 | -2.07 |
| 30,000 agents, individual queue wait | 38.87 | 26.62 | 41.86 | 16.36 | 13.25 |
| 30,000 agents, appendix queue workload proxy | 62.73 | 24.70 | 59.31 | 13.44 | 7.84 |
| 15,000 agents, appendix queue workload proxy | 6.72 | 1.41 | 2.78 | 1.40 | -3.94 |

The 95% interval for the 30,000 agent appendix proxy interaction was
[7.73, 7.94]. The corresponding interval for 15,000 agents was
[-4.00, -3.88]. The population setting changes both magnitude and sign in the
reconstructed model.

## Outcome definition

The appendix output named `avg_queue_wait_min` is not an average of recorded
individual waits. It is calculated by taking the time average of queue length
divided by service capacity for each station and then averaging across stations.
The public implementation now exposes this quantity separately as
`mean_station_queue_to_capacity`. Recorded individual waiting time remains
`mean_queue_wait_minutes`.

## Missing inputs

Exact manuscript reproduction requires the following original research files:

- the scenario JSON files that override the appendix defaults;
- the five validated OSM route, time, and event zone JSON files;
- the seven variable experimental design and surrogate training scripts;
- the three objective NSGA II implementation;
- operational cost coefficients and normalization;
- numerical feasibility thresholds and the constraint handling rule;
- seed level outputs or execution logs for the reported experiments.

Until these inputs are available, results from the reconstructed public model
must be identified as implementation validation and not as replications of the
manuscript results.
