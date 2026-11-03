from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from stadium_ingress.analysis import (
    interaction_mechanism_diagnostics,
    summarize_capacity_interaction,
)
from stadium_ingress.plotting import plot_interaction_heatmap


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine parallel seed-level interaction outputs and recompute summaries."
    )
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    frames = [pd.read_csv(path) for path in args.inputs]
    replications = pd.concat(frames, ignore_index=True)
    duplicate_keys = ["demand_scale", "capacity_scale", "seed"]
    if replications.duplicated(duplicate_keys).any():
        raise SystemExit("duplicate demand, capacity, and seed rows were found")
    summary = summarize_capacity_interaction(replications)
    mechanisms = interaction_mechanism_diagnostics(summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    replications.to_csv(
        args.output_dir / "capacity_policy_interaction_replications.csv", index=False
    )
    summary.to_csv(args.output_dir / "capacity_policy_interaction.csv", index=False)
    mechanisms.to_csv(
        args.output_dir / "interaction_mechanism_diagnostics.csv", index=False
    )
    plot_interaction_heatmap(
        summary, args.output_dir / "capacity_policy_interaction.png"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
