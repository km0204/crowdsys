from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from stadium_ingress.analysis import (
    capacity_interaction_replications,
    summarize_capacity_interaction,
)
from stadium_ingress.config import load_config

TARGETS = np.array([8.46, 4.25, 4.62, 2.48], dtype=float)
PARAMETER_NAMES = [
    "capacity_scale",
    "shuttle_boarding_rate",
    "shuttle_extra_penalty",
    "event_join_temperature",
    "event_early_penalty",
    "event_late_penalty",
    "event_extra_walk_weight",
]
LOWER = np.array([0.50, 20.0, 0.0, 0.10, 0.02, 0.30, 0.50], dtype=float)
UPPER = np.array([0.80, 48.0, 15.0, 3.00, 0.20, 1.00, 5.00], dtype=float)
REFERENCE = np.array([0.68, 48.0, 0.0, 0.35, 0.10, 0.60, 1.00], dtype=float)


def _latin_hypercube(samples: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    unit = np.empty((samples, len(PARAMETER_NAMES)))
    for column in range(unit.shape[1]):
        unit[:, column] = (rng.permutation(samples) + rng.random(samples)) / samples
    return LOWER + unit * (UPPER - LOWER)


def _evaluate(
    task: tuple[int, np.ndarray, str, int, tuple[int, ...], str],
) -> dict[str, float | str]:
    candidate_id, values, config_path, n_agents, seeds, metric = task
    base = load_config(Path(config_path))
    boarding_rate = int(round(values[1]))
    config = replace(
        base,
        n_agents=n_agents,
        capacity_scale=float(values[0]),
        shuttle=replace(
            base.shuttle,
            boarding_rate_per_minute=boarding_rate,
            extra_time_penalty_minutes=float(values[2]),
        ),
        event=replace(
            base.event,
            join_temperature=float(values[3]),
            early_penalty=float(values[4]),
            late_penalty=float(values[5]),
            extra_walk_weight=float(values[6]),
        ),
    )
    replications = capacity_interaction_replications(
        config,
        [1.0],
        [config.capacity_scale],
        seeds,
        shuttle_awareness=0.5,
        event_awareness=0.5,
        interaction_metric=metric,
    )
    summary = summarize_capacity_interaction(replications).iloc[0]
    predicted = np.array(
        [
            summary["baseline_score"],
            summary["spatial_score"],
            summary["temporal_score"],
            summary["joint_score"],
        ],
        dtype=float,
    )
    relative_errors = (predicted - TARGETS) / TARGETS
    scaled = (values - REFERENCE) / (UPPER - LOWER)
    regularization = 0.01 * float(np.mean(scaled**2))
    row = {
        "candidate_id": float(candidate_id),
        **{name: float(value) for name, value in zip(PARAMETER_NAMES, values, strict=True)},
        "shuttle_boarding_rate": float(boarding_rate),
        "baseline_wait": float(predicted[0]),
        "shuttle_wait": float(predicted[1]),
        "activity_wait": float(predicted[2]),
        "joint_wait": float(predicted[3]),
        "interaction": float(
            predicted[1] + predicted[2] - predicted[3] - predicted[0]
        ),
        "relative_rmse": float(np.sqrt(np.mean(relative_errors**2))),
        "regularization": regularization,
        "loss": float(np.mean(relative_errors**2) + regularization),
        "replications": float(len(seeds)),
        "metric": metric,
    }
    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate unreported parameters to four reference-condition aggregates."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--n-agents", type=int, default=30000)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument(
        "--design-csv",
        type=Path,
        help="Optional prior result table. Its first --samples rows are reevaluated.",
    )
    parser.add_argument(
        "--center-csv",
        type=Path,
        help="Optional prior result table whose first row centres a local design.",
    )
    parser.add_argument(
        "--local-fraction",
        type=float,
        default=0.15,
        help="Half width of a local design as a fraction of each global range.",
    )
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--replications", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--metric",
        choices=["mean_queue_wait_minutes", "mean_station_queue_to_capacity"],
        default="mean_queue_wait_minutes",
        help="Outcome used for calibration. The station ratio reproduces the appendix proxy.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 1 or args.replications < 1 or args.workers < 1:
        raise SystemExit("samples, replications, and workers must be positive")
    if args.design_csv is not None and args.center_csv is not None:
        raise SystemExit("use only one of --design-csv and --center-csv")
    if not 0.0 < args.local_fraction <= 1.0:
        raise SystemExit("local-fraction must be in (0, 1]")
    if args.center_csv is not None:
        prior = pd.read_csv(args.center_csv)
        missing = set(PARAMETER_NAMES) - set(prior.columns)
        if missing:
            raise SystemExit(f"center CSV is missing columns: {sorted(missing)}")
        center = prior.loc[0, PARAMETER_NAMES].to_numpy(dtype=float)
        local_unit = _latin_hypercube(args.samples, args.base_seed)
        local_unit = (local_unit - LOWER) / (UPPER - LOWER)
        half_width = args.local_fraction * (UPPER - LOWER)
        designs = center + (2.0 * local_unit - 1.0) * half_width
        designs = np.clip(designs, LOWER, UPPER)
        designs = np.vstack([center, designs])
    elif args.design_csv is None:
        designs = _latin_hypercube(args.samples, args.base_seed)
        designs = np.vstack([REFERENCE, designs])
    else:
        prior = pd.read_csv(args.design_csv)
        missing = set(PARAMETER_NAMES) - set(prior.columns)
        if missing:
            raise SystemExit(f"design CSV is missing columns: {sorted(missing)}")
        designs = prior.loc[:, PARAMETER_NAMES].head(args.samples).to_numpy(dtype=float)
    seeds = tuple(range(args.base_seed, args.base_seed + args.replications))
    tasks = [
        (index, values, str(args.config), args.n_agents, seeds, args.metric)
        for index, values in enumerate(designs)
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        rows = list(executor.map(_evaluate, tasks))
    results = pd.DataFrame(rows).sort_values(["loss", "candidate_id"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    print(results.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
