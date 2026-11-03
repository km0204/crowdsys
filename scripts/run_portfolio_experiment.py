from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from stadium_ingress.analysis import (
    DESIGN_COLUMNS,
    config_from_design,
    evaluate_config,
    sample_designs,
)
from stadium_ingress.config import load_config
from stadium_ingress.optimisation import nsga2
from stadium_ingress.plotting import plot_pareto_front
from stadium_ingress.surrogate import fit_surrogates, surrogate_diagnostics
from stadium_ingress.validation import (
    frontier_verification_metrics,
    select_balanced_solution,
)


def _evaluate_design(
    task: tuple[str, int, dict[str, float], tuple[int, ...]]
) -> dict[str, float]:
    config_path, design_id, design, seeds = task
    config = config_from_design(load_config(config_path), design)
    return {
        "design_id": float(design_id),
        **{column: float(design[column]) for column in DESIGN_COLUMNS},
        **evaluate_config(config, seeds),
    }


def _parallel_evaluate(
    config_path: Path,
    designs: pd.DataFrame,
    seeds: tuple[int, ...],
    workers: int,
    label: str,
) -> pd.DataFrame:
    tasks = [
        (str(config_path), int(index), row.to_dict(), seeds)
        for index, row in designs.reset_index(drop=True).iterrows()
    ]
    rows: list[dict[str, float]] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_evaluate_design, task): task for task in tasks}
        for completed, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if completed == len(tasks) or completed % max(1, len(tasks) // 10) == 0:
                print(f"{label}: completed {completed}/{len(tasks)}", flush=True)
    return pd.DataFrame(rows).sort_values("design_id", ignore_index=True)


def _external_diagnostics(
    surrogate,
    test: pd.DataFrame,
) -> pd.DataFrame:
    predictions = surrogate.predict(test)
    rows = []
    for target in surrogate.models:
        observed = test[target].to_numpy(dtype=float)
        predicted = predictions[target].to_numpy(dtype=float)
        residual = observed - predicted
        denominator = float(np.sum((observed - observed.mean()) ** 2))
        rows.append(
            {
                "target": target,
                "backend": surrogate.backend,
                "rmse_external": float(np.sqrt(np.mean(residual**2))),
                "mae_external": float(np.mean(np.abs(residual))),
                "r_squared_external": (
                    1.0 - float(np.sum(residual**2)) / denominator
                    if denominator > 0.0
                    else float("nan")
                ),
                "n_external": float(len(test)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the seven-variable surrogate portfolio experiment."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--training-designs", type=int, default=100)
    parser.add_argument("--training-replications", type=int, default=3)
    parser.add_argument("--test-designs", type=int, default=30)
    parser.add_argument("--test-replications", type=int, default=5)
    parser.add_argument("--population-size", type=int, default=200)
    parser.add_argument("--generations", type=int, default=250)
    parser.add_argument("--verify-solutions", type=int, default=20)
    parser.add_argument("--verification-replications", type=int, default=30)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--reuse-evaluations",
        action="store_true",
        help="Reuse training_designs.csv and external_test_designs.csv in the output directory.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    positive = [
        args.training_designs,
        args.training_replications,
        args.test_designs,
        args.test_replications,
        args.verify_solutions,
        args.verification_replications,
        args.workers,
    ]
    if any(value < 1 for value in positive):
        raise SystemExit("design, replication, verification, and worker counts must be positive")

    training_path = args.output_dir / "training_designs.csv"
    test_path = args.output_dir / "external_test_designs.csv"
    if args.reuse_evaluations:
        if not training_path.exists() or not test_path.exists():
            raise SystemExit("reuse requested but saved training or test evaluations are missing")
        training = pd.read_csv(training_path)
        test = pd.read_csv(test_path)
        print("reused saved training and external test evaluations", flush=True)
    else:
        training_designs = sample_designs(args.training_designs, seed=args.base_seed)
        test_designs = sample_designs(args.test_designs, seed=args.base_seed + 1_000)
        training = _parallel_evaluate(
            args.config,
            training_designs,
            tuple(range(args.base_seed, args.base_seed + args.training_replications)),
            args.workers,
            "training",
        )
        test = _parallel_evaluate(
            args.config,
            test_designs,
            tuple(
                range(
                    args.base_seed + 10_000,
                    args.base_seed + 10_000 + args.test_replications,
                )
            ),
            args.workers,
            "external test",
        )

    surrogate = fit_surrogates(training, backend="xgboost", seed=args.base_seed)
    cross_validation = surrogate_diagnostics(surrogate, training, seed=args.base_seed)
    external = _external_diagnostics(surrogate, test)
    pareto = nsga2(
        surrogate,
        population_size=args.population_size,
        generations=args.generations,
        seed=args.base_seed,
    )

    count = min(args.verify_solutions, len(pareto))
    selected_indices = np.unique(np.linspace(0, len(pareto) - 1, count, dtype=int))
    selected = pareto.iloc[selected_indices].reset_index(drop=True)
    verified_raw = _parallel_evaluate(
        args.config,
        selected[DESIGN_COLUMNS],
        tuple(
            range(
                args.base_seed + 20_000,
                args.base_seed + 20_000 + args.verification_replications,
            )
        ),
        args.workers,
        "frontier verification",
    )
    verification = selected.copy()
    for column in verified_raw.columns:
        if column not in DESIGN_COLUMNS and column != "design_id":
            verification[f"simulated_{column}"] = verified_raw[column]
    verification_for_metrics = pd.DataFrame(
        {
            "predicted_congestion_score": verification["predicted_congestion_score"],
            "predicted_intervention_cost": verification["predicted_intervention_cost"],
            "simulated_congestion_score": verification["simulated_congestion_score"],
            "simulated_intervention_cost": verification["simulated_intervention_cost"],
        }
    )
    verification_for_metrics, metrics = frontier_verification_metrics(
        verification_for_metrics
    )
    for column in verification_for_metrics.columns:
        if column not in verification.columns:
            verification[column] = verification_for_metrics[column]

    baseline = evaluate_config(
        load_config(args.config),
        range(
            args.base_seed + 20_000,
            args.base_seed + 20_000 + args.verification_replications,
        ),
    )
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
                args.base_seed + 30_000,
                args.base_seed + 30_000 + args.verification_replications,
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

    args.output_dir.mkdir(parents=True, exist_ok=True)
    training.to_csv(args.output_dir / "training_designs.csv", index=False)
    test.to_csv(args.output_dir / "external_test_designs.csv", index=False)
    cross_validation.to_csv(args.output_dir / "cross_validation.csv", index=False)
    external.to_csv(args.output_dir / "external_test_metrics.csv", index=False)
    pareto.to_csv(args.output_dir / "pareto_solutions.csv", index=False)
    verification.to_csv(args.output_dir / "frontier_verification.csv", index=False)
    representatives.to_csv(args.output_dir / "representative_policies.csv", index=False)
    plot_pareto_front(pareto, args.output_dir / "pareto_front.png")
    metadata = {
        "baseline": baseline,
        "frontier_verification_metrics": metrics,
        "abs_runs": {
            "training": args.training_designs * args.training_replications,
            "external_test": args.test_designs * args.test_replications,
            "frontier_verification": len(selected) * args.verification_replications,
            "representative_verification": len(representative_rows)
            * args.verification_replications,
            "baseline": args.verification_replications,
        },
        "surrogate_evaluation_slots": args.population_size * (args.generations + 1),
        "pareto_solutions": len(pareto),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(external.to_string(index=False))
    print(verification_for_metrics.to_string(index=False))
    print(representatives.to_string(index=False))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
