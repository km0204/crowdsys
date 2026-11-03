from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from stadium_ingress.analysis import (
    capacity_interaction_replications,
    interaction_mechanism_diagnostics,
    summarize_capacity_interaction,
)
from stadium_ingress.config import load_config
from stadium_ingress.plotting import plot_interaction_heatmap


def _csv_floats(text: str) -> list[float]:
    return [float(value.strip()) for value in text.split(",") if value.strip()]


def _evaluate_cell(
    task: tuple[str, float, float, tuple[int, ...], float, float]
) -> pd.DataFrame:
    config_path, demand_scale, capacity_scale, seeds, shuttle_awareness, event_awareness = task
    config = load_config(config_path)
    return capacity_interaction_replications(
        config,
        [demand_scale],
        [capacity_scale],
        seeds,
        shuttle_awareness=shuttle_awareness,
        event_awareness=event_awareness,
        interaction_metric="mean_queue_wait_minutes",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full matched demand and capacity interaction grid in parallel."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--demand-scales", type=_csv_floats, default=[0.8, 1.0, 1.2, 1.3])
    parser.add_argument(
        "--capacity-multipliers", type=_csv_floats, default=[0.75, 1.0, 1.25, 1.5]
    )
    parser.add_argument("--replications", type=int, default=30)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--shuttle-awareness", type=float, default=0.5)
    parser.add_argument("--event-awareness", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.replications < 2 or args.workers < 1:
        raise SystemExit("replications must be at least two and workers must be positive")

    config = load_config(args.config)
    capacities = [config.capacity_scale * multiplier for multiplier in args.capacity_multipliers]
    seeds = tuple(range(args.base_seed, args.base_seed + args.replications))
    tasks = [
        (
            str(args.config),
            demand_scale,
            capacity_scale,
            seeds,
            args.shuttle_awareness,
            args.event_awareness,
        )
        for demand_scale in args.demand_scales
        for capacity_scale in capacities
    ]

    completed: list[pd.DataFrame] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_evaluate_cell, task): task for task in tasks}
        for index, future in enumerate(as_completed(futures), start=1):
            frame = future.result()
            completed.append(frame)
            task = futures[future]
            print(
                f"completed {index}/{len(tasks)}: demand={task[1]:.2f}, "
                f"capacity={task[2]:.3f}",
                flush=True,
            )

    replications = pd.concat(completed, ignore_index=True).sort_values(
        ["demand_scale", "capacity_scale", "seed"], ignore_index=True
    )
    summary = summarize_capacity_interaction(replications)
    mechanisms = interaction_mechanism_diagnostics(summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    replications.to_csv(args.output_dir / "interaction_seed_level.csv", index=False)
    summary.to_csv(args.output_dir / "interaction_summary.csv", index=False)
    mechanisms.to_csv(args.output_dir / "interaction_mechanisms.csv", index=False)
    plot_interaction_heatmap(summary, args.output_dir / "interaction_heatmap.png")
    print(summary.to_string(index=False))
    print(mechanisms.to_string(index=False))


if __name__ == "__main__":
    main()
