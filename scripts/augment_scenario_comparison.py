from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add balanced scores and dominance relations to existing scenario results."
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    path = args.output_dir / "scenario_regret.csv"
    results = pd.read_csv(path)
    rows: list[dict[str, float]] = []
    for _, source in results.iterrows():
        scenario = int(source["scenario"])
        pareto = pd.read_csv(
            args.output_dir / f"scenario_{scenario:02d}_pareto_solutions.csv"
        )
        values = pareto[
            ["predicted_congestion_score", "predicted_intervention_cost"]
        ].to_numpy(dtype=float)
        lower = values.min(axis=0)
        span = values.max(axis=0) - lower
        span[span == 0.0] = 1.0
        fixed = np.array(
            [source["fixed_congestion_score"], source["fixed_intervention_cost"]]
        )
        reoptimised = np.array(
            [
                source["reoptimised_congestion_score"],
                source["reoptimised_intervention_cost"],
            ]
        )
        row = source.to_dict()
        row["congestion_difference"] = float(fixed[0] - reoptimised[0])
        row["congestion_difference_pct"] = float(
            (fixed[0] - reoptimised[0]) / reoptimised[0] * 100.0
        )
        row["cost_difference"] = float(fixed[1] - reoptimised[1])
        row["fixed_balanced_score"] = float(np.linalg.norm((fixed - lower) / span))
        row["reoptimised_balanced_score"] = float(
            np.linalg.norm((reoptimised - lower) / span)
        )
        row["balanced_score_difference"] = float(
            row["fixed_balanced_score"] - row["reoptimised_balanced_score"]
        )
        fixed_dominates = bool(np.all(fixed <= reoptimised) and np.any(fixed < reoptimised))
        reoptimised_dominates = bool(
            np.all(reoptimised <= fixed) and np.any(reoptimised < fixed)
        )
        row["fixed_dominates_reoptimised"] = fixed_dominates
        row["reoptimised_dominates_fixed"] = reoptimised_dominates
        row["tradeoff_incomparable"] = not fixed_dominates and not reoptimised_dominates
        row.pop("congestion_regret", None)
        row.pop("congestion_regret_pct", None)
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
