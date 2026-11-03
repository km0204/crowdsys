from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from run_portfolio_experiment import _external_diagnostics, _parallel_evaluate

from stadium_ingress.analysis import DESIGN_COLUMNS
from stadium_ingress.optimisation import nsga2
from stadium_ingress.surrogate import fit_surrogates
from stadium_ingress.validation import (
    frontier_verification_metrics,
    hypervolume_2d,
    nondominated_mask,
)


def _verify_backend(
    backend: str,
    training: pd.DataFrame,
    test: pd.DataFrame,
    config: Path,
    population_size: int,
    generations: int,
    verify_solutions: int,
    verification_replications: int,
    base_seed: int,
    workers: int,
    output_dir: Path,
) -> tuple[dict[str, float | str], np.ndarray]:
    surrogate = fit_surrogates(training, backend=backend, seed=base_seed)
    external = _external_diagnostics(surrogate, test)
    pareto = nsga2(
        surrogate,
        population_size=population_size,
        generations=generations,
        seed=base_seed,
    )
    count = min(verify_solutions, len(pareto))
    indices = np.unique(np.linspace(0, len(pareto) - 1, count, dtype=int))
    selected = pareto.iloc[indices].reset_index(drop=True)
    verified = _parallel_evaluate(
        config,
        selected[DESIGN_COLUMNS],
        tuple(range(base_seed + 20_000, base_seed + 20_000 + verification_replications)),
        workers,
        f"{backend} verification",
    )
    details = selected.copy()
    details["simulated_congestion_score"] = verified["congestion_score"]
    details["simulated_intervention_cost"] = verified["intervention_cost"]
    metric_frame = pd.DataFrame(
        {
            "predicted_congestion_score": details["predicted_congestion_score"],
            "predicted_intervention_cost": details["predicted_intervention_cost"],
            "simulated_congestion_score": details["simulated_congestion_score"],
            "simulated_intervention_cost": details["simulated_intervention_cost"],
        }
    )
    metric_frame, metrics = frontier_verification_metrics(metric_frame)
    for column in metric_frame.columns:
        if column not in details.columns:
            details[column] = metric_frame[column]
    pareto.to_csv(output_dir / f"{backend}_pareto.csv", index=False)
    details.to_csv(output_dir / f"{backend}_verification.csv", index=False)
    external.to_csv(output_dir / f"{backend}_external_test.csv", index=False)
    simulated = details[
        ["simulated_congestion_score", "simulated_intervention_cost"]
    ].to_numpy(dtype=float)
    row: dict[str, float | str] = {
        "backend": backend,
        "external_r_squared": float(external["r_squared_external"].iloc[0]),
        "external_rmse": float(external["rmse_external"].iloc[0]),
        "retained_nondominated_share": metrics["retained_nondominated_share"],
        "mean_absolute_congestion_degradation_pct": metrics[
            "mean_absolute_congestion_degradation_pct"
        ],
        "max_absolute_congestion_degradation_pct": metrics[
            "max_absolute_congestion_degradation_pct"
        ],
        "verification_hypervolume_degradation_pct": metrics[
            "hypervolume_degradation_pct"
        ],
        "pareto_solutions": float(len(pareto)),
    }
    return row, simulated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare polynomial and XGBoost surrogates under one simulation budget."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--portfolio-dir", type=Path, required=True)
    parser.add_argument("--population-size", type=int, default=200)
    parser.add_argument("--generations", type=int, default=250)
    parser.add_argument("--verify-solutions", type=int, default=20)
    parser.add_argument("--verification-replications", type=int, default=30)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    training = pd.read_csv(args.portfolio_dir / "training_designs.csv")
    test = pd.read_csv(args.portfolio_dir / "external_test_designs.csv")
    direct_values = training[["congestion_score", "intervention_cost"]].to_numpy(
        dtype=float
    )
    direct_front = direct_values[nondominated_mask(direct_values)]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    polynomial_row, polynomial_simulated = _verify_backend(
        "polynomial",
        training,
        test,
        args.config,
        args.population_size,
        args.generations,
        args.verify_solutions,
        args.verification_replications,
        args.base_seed,
        args.workers,
        args.output_dir,
    )
    xgboost_details = pd.read_csv(args.portfolio_dir / "frontier_verification.csv")
    xgboost_external = pd.read_csv(args.portfolio_dir / "external_test_metrics.csv")
    xgboost_pareto = pd.read_csv(args.portfolio_dir / "pareto_solutions.csv")
    xgboost_metric_frame = xgboost_details[
        [
            "predicted_congestion_score",
            "predicted_intervention_cost",
            "simulated_congestion_score",
            "simulated_intervention_cost",
        ]
    ]
    _, xgboost_metrics = frontier_verification_metrics(xgboost_metric_frame)
    xgboost_row: dict[str, float | str] = {
        "backend": "xgboost",
        "external_r_squared": float(xgboost_external["r_squared_external"].iloc[0]),
        "external_rmse": float(xgboost_external["rmse_external"].iloc[0]),
        "retained_nondominated_share": xgboost_metrics["retained_nondominated_share"],
        "mean_absolute_congestion_degradation_pct": xgboost_metrics[
            "mean_absolute_congestion_degradation_pct"
        ],
        "max_absolute_congestion_degradation_pct": xgboost_metrics[
            "max_absolute_congestion_degradation_pct"
        ],
        "verification_hypervolume_degradation_pct": xgboost_metrics[
            "hypervolume_degradation_pct"
        ],
        "pareto_solutions": float(len(xgboost_pareto)),
    }
    xgboost_simulated = xgboost_details[
        ["simulated_congestion_score", "simulated_intervention_cost"]
    ].to_numpy(dtype=float)
    xgboost_details.to_csv(args.output_dir / "xgboost_verification.csv", index=False)
    xgboost_external.to_csv(args.output_dir / "xgboost_external_test.csv", index=False)
    xgboost_pareto.to_csv(args.output_dir / "xgboost_pareto.csv", index=False)
    rows = [polynomial_row, xgboost_row]
    simulated_by_backend = {
        "polynomial": polynomial_simulated,
        "xgboost": xgboost_simulated,
    }

    combined = np.vstack([direct_front, *simulated_by_backend.values()])
    maximum = combined.max(axis=0)
    minimum = combined.min(axis=0)
    reference = maximum + np.maximum(0.05 * (maximum - minimum), 1e-9)
    direct_hypervolume = hypervolume_2d(direct_front, reference)
    for row in rows:
        backend = str(row["backend"])
        verified_hypervolume = hypervolume_2d(simulated_by_backend[backend], reference)
        row["direct_screening_hypervolume"] = direct_hypervolume
        row["verified_hypervolume"] = verified_hypervolume
        row["verified_to_direct_hypervolume_ratio"] = (
            verified_hypervolume / direct_hypervolume
            if direct_hypervolume > 0.0
            else float("nan")
        )
        row["initial_abs_runs"] = float(len(training) * int(training["replications"].iloc[0]))
        row["verification_abs_runs"] = float(
            args.verify_solutions * args.verification_replications
        )
        row["surrogate_evaluation_slots"] = float(
            args.population_size * (args.generations + 1)
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(args.output_dir / "fixed_budget_summary.csv", index=False)
    pd.DataFrame(direct_front, columns=["congestion_score", "resource_index"]).to_csv(
        args.output_dir / "direct_screening_front.csv", index=False
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
