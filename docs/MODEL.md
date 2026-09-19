# Model specification

## Scope

The model represents spectators approaching a stadium through six railway
access points or an optional shuttle, joining capacity-constrained queues, and
walking directly to the stadium or via an optional pre-event activity.

## Desired arrival time

Baseline desired arrival time is sampled from three segments:

1. triangular early-arrival density from -140 to -60 minutes;
2. a uniform plateau from -60 to 0 minutes;
3. an exponentially decaying late tail from 0 to 30 minutes.

Pre-event participants instead use the configured event window and asymmetric
early/late schedule penalties. Target decision time equals desired arrival time
minus rail access and share-weighted walking time.

## Behavioural heterogeneity and awareness

Time and crowding sensitivities are truncated normal draws. Food and stage
preferences are a two-component Dirichlet draw. When information is enabled,
separate Bernoulli draws determine whether each spectator knows about the
shuttle and pre-event activity. An unknown alternative is not included in that
spectator's choice set.

Event-aware spectators become candidates at `candidate_rate`. Candidate zone
utility combines taste, zone attractiveness, event reward, time sensitivity,
and additional walking time. A calibrated logit centres adoption near
`target_join_rate` among candidates, without forcing the realised rate.

## Mode and station choice

For rail station `j`, the spectator minimises:

```text
U_ij = beta_time_i × (rail access + station bias + expected wait + walking)
     + beta_crowd_i × (queue length / service capacity)
     + prior_alpha × [-log(share_hint_j)]
     + Gumbel noise
```

The shuttle is included only when enabled and known to the spectator. Its
disutility contains expected queueing, in-vehicle time, drop-off walking,
configured penalty, crowding, and Gumbel noise.

## Capacities and one-minute event order

Rail capacity is station capacity multiplied by the common `capacity_scale`.
Shuttle capacity is the lesser of the boarding rate and fleet capacity implied
by vehicle capacity, fleet size, and headway.

Each simulated minute:

1. due spectators make a mode/station choice and join a FIFO queue;
2. rail queues release up to their effective capacities;
3. the shuttle queue boards up to its effective capacity;
4. in-vehicle and walking states advance;
5. event participants transition through walk-to-zone, dwell, and walk-to-gate;
6. arrivals, queues, choices, and optional positions are recorded.

## OSM coordinate movement

If valid route JSON is supplied, progress is interpolated along the coordinate
polyline. Active walkers sharing an origin route slow when their count exceeds
the configured path capacity. Position logging is optional because it can
produce large files.

If OSM files are absent, the same queues and station-level walking durations
remain active, but no coordinates or density map are claimed. Event legs use
documented duration fallbacks when event routing data is absent.

## Outputs and KPIs

- minute-level arrivals and rail/shuttle/total queues;
- individual arrival, mode, station, wait, travel, and event records;
- optional minute-level walking coordinates;
- mean and 90th-percentile queue wait;
- peak total queue;
- pre-kick-off arrival share;
- shuttle and event participation shares;
- accounting-style intervention cost.

The optimisation `congestion_score` is:

```text
mean queue wait in minutes + 20 × peak queue / number of spectators
```

This scalar is an implementation choice for the public analysis pipeline and
should be aligned with the final paper's exact objective before publication.

## Differences from the supplied appendix

- The pasted appendix had lost indentation and imports; it was reconstructed as
  typed modules rather than preserved as one executable script.
- Mesa random activation is represented by an explicit one-minute event loop
  with seeded random ordering of due decisions.
- OSM files are loaded through a validated optional input layer rather than at
  module import time.
- Interaction, surrogate, NSGA-II, sensitivity, and robustness modules are new
  release additions and are not presented as verbatim appendix code.
