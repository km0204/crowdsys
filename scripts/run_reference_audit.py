from __future__ import annotations

import argparse
from pathlib import Path

from stadium_ingress.analysis import (
    capacity_interaction_replications,
    summarize_capacity_interaction,
)
from stadium_ingress.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a matched-seed audit of the four reference policy conditions."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--replications", type=int, default=30)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.replications < 2:
        raise SystemExit("replications must be at least two for an interval estimate")

    config = load_config(args.config)
    seeds = range(args.base_seed, args.base_seed + args.replications)
    seed_level = capacity_interaction_replications(
        config,
        demand_scales=[1.0],
        capacity_scales=[config.capacity_scale],
        seeds=seeds,
        shuttle_awareness=0.5,
        event_awareness=0.5,
        interaction_metric="mean_queue_wait_minutes",
    )
    summary = summarize_capacity_interaction(seed_level)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_level.to_csv(args.output_dir / "reference_seed_level.csv", index=False)
    summary.to_csv(args.output_dir / "reference_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
