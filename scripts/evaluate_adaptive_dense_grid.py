from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from stadium_ingress.adaptive import CONTEXT_COLUMNS, fit_contextual_ensemble

METHODS = (
    "offline",
    "pareto_adaptive",
    "interaction_adaptive",
    "staged_adaptive",
)


def _csv_strings(text: str) -> list[str]:
    return [value.strip() for value in text.split(",") if value.strip()]


def _cross_validated_gamma_rmse(
    data: pd.DataFrame,
    members: int,
    folds: int,
    seed: int,
) -> float:
    splitter = KFold(n_splits=min(folds, len(data)), shuffle=True, random_state=seed)
    errors: list[np.ndarray] = []
    for fold, (train_indices, test_indices) in enumerate(splitter.split(data)):
        ensemble = fit_contextual_ensemble(
            data.iloc[train_indices],
            members=members,
            seed=seed + 101 * fold,
        )
        predicted = ensemble.gamma_members(data.iloc[test_indices]).mean(axis=0)
        observed = data.iloc[test_indices]["gamma"].to_numpy(dtype=float)
        errors.append(predicted - observed)
    combined = np.concatenate(errors)
    return float(np.sqrt(np.mean(combined**2)))


def _reference_policy_grid(dense: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "shuttle_headway_minutes": 10.0,
            "fleet_size": 6.0,
            "shuttle_vehicle_capacity": 80.0,
            "event_target_join_rate": 0.80,
            "event_dwell_mean_minutes": 20.0,
            "shuttle_awareness": 0.50,
            "event_awareness": 0.50,
            "demand_scale": dense["demand_scale"].to_numpy(dtype=float),
            "capacity_scale": dense["capacity_scale"].to_numpy(dtype=float),
        }
    )
    return frame.loc[:, CONTEXT_COLUMNS]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate contextual surrogate learning on the held out dense grid."
    )
    parser.add_argument("--adaptive-dir", type=Path, required=True)
    parser.add_argument("--dense-summary", type=Path, required=True)
    parser.add_argument("--methods", type=_csv_strings, default=list(METHODS))
    parser.add_argument("--ensemble-members", type=int, default=5)
    parser.add_argument("--cv-members", type=int, default=3)
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--boundary-threshold", type=float, default=0.20)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-output", type=Path)
    args = parser.parse_args()

    dense = pd.read_csv(args.dense_summary).sort_values(
        ["demand_scale", "capacity_scale"], ignore_index=True
    )
    test = _reference_policy_grid(dense)
    truth = dense["interaction"].to_numpy(dtype=float)
    boundary = np.abs(truth) <= args.boundary_threshold
    if not boundary.any():
        raise SystemExit("the boundary threshold selects no dense grid cells")

    unknown_methods = sorted(set(args.methods) - set(METHODS))
    if unknown_methods:
        raise SystemExit(f"unknown methods: {unknown_methods}")
    rows: list[dict[str, float | str]] = []
    for method in args.methods:
        paths = sorted(args.adaptive_dir.glob(f"training_{method}_seed_*.csv"))
        if not paths:
            raise SystemExit(f"no training files found for {method}")
        for path in paths:
            training = pd.read_csv(path)
            algorithm_seed = int(path.stem.rsplit("_", maxsplit=1)[-1])
            for iteration in sorted(training["iteration"].astype(int).unique()):
                selected = training[training["iteration"] <= iteration].copy()
                model_data = (
                    training[training["iteration"] <= min(iteration, 1)].copy()
                    if method == "staged_adaptive"
                    else selected
                )
                model_iteration = min(iteration, 1) if method == "staged_adaptive" else iteration
                ensemble = fit_contextual_ensemble(
                    model_data,
                    members=args.ensemble_members,
                    seed=algorithm_seed + 37 * model_iteration,
                )
                predictions = ensemble.gamma_members(test)
                mean = predictions.mean(axis=0)
                sd = predictions.std(axis=0, ddof=1)
                lower = mean - 1.96 * sd
                upper = mean + 1.96 * sd
                rows.append(
                    {
                        "method": method,
                        "algorithm_seed": float(algorithm_seed),
                        "iteration": float(iteration),
                        "training_points": float(len(selected)),
                        "search_abs_runs": float(len(selected) * 4 * 2),
                        "dense_gamma_rmse": float(np.sqrt(np.mean((mean - truth) ** 2))),
                        "dense_sign_accuracy": float(np.mean(np.sign(mean) == np.sign(truth))),
                        "dense_boundary_rmse": float(
                            np.sqrt(np.mean((mean[boundary] - truth[boundary]) ** 2))
                        ),
                        "dense_boundary_sign_accuracy": float(
                            np.mean(np.sign(mean[boundary]) == np.sign(truth[boundary]))
                        ),
                        "dense_interval_coverage": float(
                            np.mean((truth >= lower) & (truth <= upper))
                        ),
                        "dense_boundary_cells": float(boundary.sum()),
                        "cv_gamma_rmse": _cross_validated_gamma_rmse(
                            model_data,
                            members=args.cv_members,
                            folds=args.cv_folds,
                            seed=algorithm_seed + 503 * model_iteration,
                        ),
                    }
                )

    result = pd.DataFrame(rows).sort_values(
        ["method", "algorithm_seed", "iteration"], ignore_index=True
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    if args.checkpoint_output is not None:
        selected_indices: list[int] = []
        for (method, _), frame in result.groupby(["method", "algorithm_seed"]):
            if method == "staged_adaptive":
                checkpoint = frame[frame["iteration"] == 1]
                if checkpoint.empty:
                    raise SystemExit("staged adaptive results do not contain iteration 1")
                selected_indices.append(int(checkpoint.index[0]))
            else:
                selected_indices.append(int(frame["cv_gamma_rmse"].idxmin()))
        checkpoints = result.loc[selected_indices].sort_values(
            ["method", "algorithm_seed"], ignore_index=True
        )
        args.checkpoint_output.parent.mkdir(parents=True, exist_ok=True)
        checkpoints.to_csv(args.checkpoint_output, index=False)
        print("\nInteraction checkpoint diagnostics")
        print(
            checkpoints.groupby("method")
            .agg(
                selected_abs_runs=("search_abs_runs", "mean"),
                dense_gamma_rmse=("dense_gamma_rmse", "mean"),
                dense_boundary_rmse=("dense_boundary_rmse", "mean"),
                dense_sign_accuracy=("dense_sign_accuracy", "mean"),
            )
            .round(4)
            .to_string()
        )
    summary = result.groupby(["method", "search_abs_runs"], sort=True).agg(
        dense_gamma_rmse=("dense_gamma_rmse", "mean"),
        dense_sign_accuracy=("dense_sign_accuracy", "mean"),
        dense_boundary_rmse=("dense_boundary_rmse", "mean"),
        dense_boundary_sign_accuracy=("dense_boundary_sign_accuracy", "mean"),
        dense_interval_coverage=("dense_interval_coverage", "mean"),
    )
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
