from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from run_portfolio_experiment import _external_diagnostics, _parallel_evaluate

from stadium_ingress.analysis import DESIGN_COLUMNS, config_from_design, evaluate_config
from stadium_ingress.config import load_config
from stadium_ingress.optimisation import nsga2
from stadium_ingress.plotting import plot_pareto_front
from stadium_ingress.surrogate import fit_surrogates, surrogate_diagnostics
from stadium_ingress.validation import (
    frontier_verification_metrics,
    select_balanced_solution,
)


def _refinement_rows(
    training: pd.DataFrame,
    verification: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for offset, verified in verification.reset_index(drop=True).iterrows():
        row: dict[str, float] = {"design_id": float(len(training) + offset)}
        for column in DESIGN_COLUMNS:
            row[column] = float(verified[column])
        for column in training.columns:
            if column == "design_id" or column in DESIGN_COLUMNS:
                continue
            simulated = f"simulated_{column}"
            if simulated not in verification.columns:
                raise ValueError(f"initial verification is missing {simulated}")
            row[column] = float(verified[simulated])
        rows.append(row)
    return pd.DataFrame(rows, columns=training.columns)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add one frontier-focused refinement and independently verify the new front."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--initial-dir", type=Path, required=True)
    parser.add_argument("--population-size", type=int, default=200)
    parser.add_argument("--generations", type=int, default=250)
    parser.add_argument("--verify-solutions", type=int, default=20)
    parser.add_argument("--verification-replications", type=int, default=30)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    training = pd.read_csv(args.initial_dir / "training_designs.csv")
    external_test = pd.read_csv(args.initial_dir / "external_test_designs.csv")
    initial_verification = pd.read_csv(args.initial_dir / "frontier_verification.csv")
    refinement = _refinement_rows(training, initial_verification)
    augmented = pd.concat([training, refinement], ignore_index=True)
    print(
        f"augmented {len(training)} initial designs with {len(refinement)} frontier designs",
        flush=True,
    )

    surrogate = fit_surrogates(augmented, backend="xgboost", seed=args.base_seed + 1)
    cross_validation = surrogate_diagnostics(
        surrogate, augmented, seed=args.base_seed + 1
    )
    external = _external_diagnostics(surrogate, external_test)
    pareto = nsga2(
        surrogate,
        population_size=args.population_size,
        generations=args.generations,
        seed=args.base_seed + 1,
    )

    count = min(args.verify_solutions, len(pareto))
    selected_indices = np.unique(np.linspace(0, len(pareto) - 1, count, dtype=int))
    selected = pareto.iloc[selected_indices].reset_index(drop=True)
    verified_raw = _parallel_evaluate(
        args.config,
        selected[DESIGN_COLUMNS],
        tuple(
            range(
                args.base_seed + 50_000,
                args.base_seed + 50_000 + args.verification_replications,
            )
        ),
        args.workers,
        "independent refined frontier verification",
    )
    verification = selected.copy()
    for column in verified_raw.columns:
        if column not in DESIGN_COLUMNS and column != "design_id":
            verification[f"simulated_{column}"] = verified_raw[column]
    metric_frame = pd.DataFrame(
        {
            "predicted_congestion_score": verification["predicted_congestion_score"],
            "predicted_intervention_cost": verification["predicted_intervention_cost"],
            "simulated_congestion_score": verification["simulated_congestion_score"],
            "simulated_intervention_cost": verification["simulated_intervention_cost"],
        }
    )
    metric_frame, metrics = frontier_verification_metrics(metric_frame)
    for column in metric_frame.columns:
        if column not in verification.columns:
            verification[column] = metric_frame[column]

    representative_ids = {
        "minimum congestion": int(pareto["predicted_congestion_score"].idxmin()),
        "minimum resource": int(pareto["predicted_intervention_cost"].idxmin()),
        "balanced compromise": int(select_balanced_solution(pareto).name),
    }
    representative_rows = []
    for label, index in representative_ids.items():
        solution = pareto.loc[index]
        verified = evaluate_config(
            config_from_design(load_config(args.config), solution),
            range(
                args.base_seed + 60_000,
                args.base_seed + 60_000 + args.verification_replications,
            ),
        )
        representative_rows.append(
            {
                "policy": label,
                **{column: float(solution[column]) for column in DESIGN_COLUMNS},
                "predicted_congestion_score": float(solution["predicted_congestion_score"]),
                "predicted_resource_index": float(solution["predicted_intervention_cost"]),
                **{f"simulated_{key}": value for key, value in verified.items()},
            }
        )
    representatives = pd.DataFrame(representative_rows)
    baseline = evaluate_config(
        load_config(args.config),
        range(
            args.base_seed + 60_000,
            args.base_seed + 60_000 + args.verification_replications,
        ),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    refinement.to_csv(args.output_dir / "frontier_refinement_designs.csv", index=False)
    augmented.to_csv(args.output_dir / "augmented_training_designs.csv", index=False)
    cross_validation.to_csv(args.output_dir / "cross_validation.csv", index=False)
    external.to_csv(args.output_dir / "external_test_metrics.csv", index=False)
    pareto.to_csv(args.output_dir / "pareto_solutions.csv", index=False)
    verification.to_csv(args.output_dir / "frontier_verification.csv", index=False)
    representatives.to_csv(args.output_dir / "representative_policies.csv", index=False)
    plot_pareto_front(pareto, args.output_dir / "pareto_front.png")
    metadata = {
        "baseline": baseline,
        "frontier_verification_metrics": metrics,
        "initial_designs": len(training),
        "refinement_designs": len(refinement),
        "refinement_replications_per_design": float(
            initial_verification["simulated_replications"].iloc[0]
        ),
        "independent_verification_replications": args.verification_replications,
        "surrogate_evaluation_slots": args.population_size * (args.generations + 1),
        "pareto_solutions": len(pareto),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(external.to_string(index=False))
    print(metric_frame.to_string(index=False))
    print(representatives.to_string(index=False))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
