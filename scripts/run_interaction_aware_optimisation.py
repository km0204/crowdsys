from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from stadium_ingress.adaptive import (
    CONTEXT_COLUMNS,
    FACTORIAL_TARGETS,
    OPTIMISATION_TARGET,
    ContextualEnsemble,
    acquisition_pool,
    acquisition_scores,
    evaluate_policy_state,
    fit_contextual_ensemble,
    sample_policy_states,
    select_adaptive_batch,
)
from stadium_ingress.analysis import DESIGN_COLUMNS, config_from_design, evaluate_config
from stadium_ingress.config import load_config
from stadium_ingress.optimisation import nsga2
from stadium_ingress.validation import hypervolume_2d, nondominated_mask

ACQUISITION_STATES = (
    (0.80, 0.51),
    (0.80, 0.765),
    (0.80, 1.02),
    (1.05, 0.51),
    (1.05, 0.765),
    (1.05, 1.02),
    (1.30, 0.51),
    (1.30, 0.765),
    (1.30, 1.02),
)
VERIFICATION_STATES = (
    (0.80, 0.85),
    (1.00, 0.68),
    (1.20, 0.68),
)
METHODS = ("offline", "pareto_adaptive", "interaction_adaptive", "staged_adaptive")


def _csv_strings(text: str) -> list[str]:
    return [value.strip() for value in text.split(",") if value.strip()]


def _evaluate_factorial_task(
    task: tuple[str, int, dict[str, float], tuple[int, ...]],
) -> tuple[dict[str, float], pd.DataFrame]:
    config_path, design_id, design, seeds = task
    summary, seed_level = evaluate_policy_state(load_config(config_path), design, seeds)
    aggregate = {
        "design_id": float(design_id),
        **{column: float(design[column]) for column in CONTEXT_COLUMNS},
        **summary,
    }
    seed_level.insert(0, "design_id", design_id)
    return aggregate, seed_level


def _parallel_factorial(
    config_path: Path,
    designs: pd.DataFrame,
    seeds: tuple[int, ...],
    workers: int,
    label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tasks = [
        (str(config_path), int(index), row.to_dict(), seeds)
        for index, row in designs.reset_index(drop=True).iterrows()
    ]
    aggregate_rows: list[dict[str, float]] = []
    seed_frames: list[pd.DataFrame] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_evaluate_factorial_task, task): task for task in tasks}
        for completed, future in enumerate(as_completed(futures), start=1):
            aggregate, seed_level = future.result()
            aggregate_rows.append(aggregate)
            seed_frames.append(seed_level)
            if completed == len(tasks) or completed % max(1, len(tasks) // 5) == 0:
                print(f"{label}: completed {completed}/{len(tasks)}", flush=True)
    aggregate = pd.DataFrame(aggregate_rows).sort_values("design_id", ignore_index=True)
    seed_level = pd.concat(seed_frames, ignore_index=True).sort_values(
        ["design_id", "seed"], ignore_index=True
    )
    return aggregate, seed_level


def _evaluate_joint_task(
    task: tuple[str, int, int, dict[str, float], float, float, tuple[int, ...]],
) -> dict[str, float]:
    config_path, policy_id, state_id, design, demand, capacity, seeds = task
    base = load_config(config_path)
    state = replace(
        base,
        n_agents=max(1, int(round(base.n_agents * demand))),
        capacity_scale=capacity,
    )
    summary = evaluate_config(config_from_design(state, design), seeds)
    return {
        "policy_id": float(policy_id),
        "state_id": float(state_id),
        "demand_scale": demand,
        "capacity_scale": capacity,
        "verified_congestion_score": summary["congestion_score"],
        "verified_congestion_score_sd": summary["congestion_score_sd"],
        "verification_replications": summary["replications"],
    }


def _parallel_joint_verification(
    config_path: Path,
    policies: pd.DataFrame,
    states: tuple[tuple[float, float], ...],
    seeds: tuple[int, ...],
    workers: int,
    label: str,
) -> pd.DataFrame:
    tasks = []
    for policy_id, row in policies.reset_index(drop=True).iterrows():
        design = {column: float(row[column]) for column in DESIGN_COLUMNS}
        for state_id, (demand, capacity) in enumerate(states):
            tasks.append(
                (
                    str(config_path),
                    policy_id,
                    state_id,
                    design,
                    demand,
                    capacity,
                    seeds,
                )
            )
    rows: list[dict[str, float]] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_evaluate_joint_task, task): task for task in tasks}
        for completed, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if completed == len(tasks) or completed % max(1, len(tasks) // 5) == 0:
                print(f"{label}: completed {completed}/{len(tasks)}", flush=True)
    return pd.DataFrame(rows).sort_values(["policy_id", "state_id"], ignore_index=True)


def _diagnostics(ensemble, external: pd.DataFrame) -> dict[str, float]:
    gamma = ensemble.gamma_members(external)
    gamma_mean = gamma.mean(axis=0)
    gamma_sd = gamma.std(axis=0, ddof=1)
    observed_gamma = external["gamma"].to_numpy(dtype=float)
    joint = ensemble.predict_members(external, OPTIMISATION_TARGET)
    joint_mean = joint.mean(axis=0)
    observed_joint = external[OPTIMISATION_TARGET].to_numpy(dtype=float)
    boundary = np.abs(observed_gamma) <= np.quantile(np.abs(observed_gamma), 0.25)
    return {
        "gamma_rmse": float(np.sqrt(np.mean((observed_gamma - gamma_mean) ** 2))),
        "gamma_sign_accuracy": float(
            np.mean(np.sign(observed_gamma) == np.sign(gamma_mean))
        ),
        "boundary_sign_accuracy": float(
            np.mean(np.sign(observed_gamma[boundary]) == np.sign(gamma_mean[boundary]))
        ),
        "gamma_interval_coverage": float(
            np.mean(
                (observed_gamma >= gamma_mean - 1.96 * gamma_sd)
                & (observed_gamma <= gamma_mean + 1.96 * gamma_sd)
            )
        ),
        "joint_rmse": float(np.sqrt(np.mean((observed_joint - joint_mean) ** 2))),
    }


class RobustContextualSurrogate:
    def __init__(self, ensemble, states: tuple[tuple[float, float], ...]):
        self.ensemble = ensemble
        self.states = states

    def predict(self, designs: pd.DataFrame) -> pd.DataFrame:
        state_predictions = []
        for demand, capacity in self.states:
            contextual = designs.copy()
            contextual["demand_scale"] = demand
            contextual["capacity_scale"] = capacity
            state_predictions.append(
                self.ensemble.predict_members(contextual, OPTIMISATION_TARGET).mean(axis=0)
            )
        return pd.DataFrame(
            {"congestion_score": np.mean(state_predictions, axis=0)},
            index=designs.index,
        )


def _fit_and_record(
    data: pd.DataFrame,
    external: pd.DataFrame,
    method: str,
    algorithm_seed: int,
    iteration: int,
    ensemble_members: int,
) -> tuple[object, dict[str, float | str]]:
    ensemble = fit_contextual_ensemble(
        data,
        members=ensemble_members,
        seed=algorithm_seed * 10_000 + iteration,
    )
    row: dict[str, float | str] = {
        "method": method,
        "algorithm_seed": float(algorithm_seed),
        "iteration": float(iteration),
        "training_points": float(len(data)),
        "search_abs_runs": float(len(data) * 4 * int(data["replications"].iloc[0])),
        **_diagnostics(ensemble, external),
    }
    return ensemble, row


def _run_method(
    method: str,
    algorithm_seed: int,
    initial: pd.DataFrame,
    config_path: Path,
    external: pd.DataFrame,
    training_seeds: tuple[int, ...],
    initial_points: int,
    total_points: int,
    batch_size: int,
    ensemble_members: int,
    workers: int,
    output_dir: Path,
) -> tuple[object, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = initial.copy()
    learning_rows: list[dict[str, float | str]] = []
    seed_frames: list[pd.DataFrame] = []
    offline_design = sample_policy_states(
        total_points,
        seed=algorithm_seed + 40_000,
    ).iloc[initial_points:].reset_index(drop=True)
    iteration = 0
    interaction_checkpoint = None
    checkpoint_diagnostics: dict[str, float | str] | None = None
    ensemble, row = _fit_and_record(
        data,
        external,
        method,
        algorithm_seed,
        iteration,
        ensemble_members,
    )
    learning_rows.append(row)
    while len(data) < total_points:
        count = min(batch_size, total_points - len(data))
        if method == "offline":
            selected = offline_design.iloc[
                len(data) - initial_points : len(data) - initial_points + count
            ].reset_index(drop=True)
        else:
            pool = acquisition_pool(
                policy_count=160,
                states=ACQUISITION_STATES,
                seed=algorithm_seed * 1_000 + iteration,
            )
            scores = acquisition_scores(ensemble, pool)
            selected = select_adaptive_batch(
                scores,
                data,
                count,
                (
                    "pareto"
                    if method == "pareto_adaptive"
                    or (method == "staged_adaptive" and iteration > 0)
                    else "interaction"
                ),
            )
        new_data, seed_level = _parallel_factorial(
            config_path,
            selected,
            training_seeds,
            workers,
            f"{method} seed {algorithm_seed} iteration {iteration + 1}",
        )
        seed_level["method"] = method
        seed_level["algorithm_seed"] = algorithm_seed
        seed_level["iteration"] = iteration + 1
        seed_frames.append(seed_level)
        new_data["iteration"] = iteration + 1
        data = pd.concat([data, new_data], ignore_index=True)
        iteration += 1
        ensemble, row = _fit_and_record(
            data,
            external,
            method,
            algorithm_seed,
            iteration,
            ensemble_members,
        )
        if method == "staged_adaptive" and iteration == 1:
            interaction_checkpoint = ensemble
            checkpoint_diagnostics = {
                key: row[key]
                for key in (
                    "gamma_rmse",
                    "gamma_sign_accuracy",
                    "boundary_sign_accuracy",
                    "gamma_interval_coverage",
                )
            }
        elif method == "staged_adaptive" and checkpoint_diagnostics is not None:
            row.update(checkpoint_diagnostics)
        learning_rows.append(row)
        output_dir.mkdir(parents=True, exist_ok=True)
        data.to_csv(
            output_dir / f"training_{method}_seed_{algorithm_seed}.csv",
            index=False,
        )
    if method == "staged_adaptive" and interaction_checkpoint is not None:
        models = {
            target: interaction_checkpoint.models[target] for target in FACTORIAL_TARGETS
        }
        models[OPTIMISATION_TARGET] = ensemble.models[OPTIMISATION_TARGET]
        ensemble = ContextualEnsemble(models)
    seed_level = pd.concat(seed_frames, ignore_index=True) if seed_frames else pd.DataFrame()
    return ensemble, data, pd.DataFrame(learning_rows), seed_level


def _select_frontier(ensemble, algorithm_seed: int, points: int) -> pd.DataFrame:
    surrogate = RobustContextualSurrogate(ensemble, VERIFICATION_STATES)
    pareto = nsga2(
        surrogate,
        population_size=120,
        generations=120,
        seed=algorithm_seed + 70_000,
    ).sort_values("predicted_intervention_cost", ignore_index=True)
    count = min(points, len(pareto))
    indices = np.unique(np.linspace(0, len(pareto) - 1, count, dtype=int))
    return pareto.iloc[indices].reset_index(drop=True)


def _summarize_verified_frontiers(verification: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    grouped = (
        verification.groupby(["method", "algorithm_seed", "policy_id"], sort=True)
        .agg(
            verified_congestion_score=("verified_congestion_score", "mean"),
            predicted_congestion_score=("predicted_congestion_score", "first"),
            resource_index=("predicted_intervention_cost", "first"),
        )
        .reset_index()
    )
    objectives = grouped[["verified_congestion_score", "resource_index"]].to_numpy()
    maximum = objectives.max(axis=0)
    minimum = objectives.min(axis=0)
    reference = maximum + np.maximum(0.05 * (maximum - minimum), 1e-9)
    rows: list[dict[str, float | str]] = []
    for (method, seed), frame in grouped.groupby(["method", "algorithm_seed"]):
        values = frame[["verified_congestion_score", "resource_index"]].to_numpy()
        mask = nondominated_mask(values)
        degradation = (
            frame["verified_congestion_score"] - frame["predicted_congestion_score"]
        ) / frame["predicted_congestion_score"].abs().clip(lower=1e-9)
        rows.append(
            {
                "method": method,
                "algorithm_seed": seed,
                "verified_points": float(len(frame)),
                "verified_nondominated_share": float(mask.mean()),
                "mean_absolute_degradation_pct": float(degradation.abs().mean() * 100),
                "verified_hypervolume": hypervolume_2d(values, reference),
                "reference_congestion": float(reference[0]),
                "reference_resource": float(reference[1]),
            }
        )
    run_summary = pd.DataFrame(rows)
    method_summary = (
        run_summary.groupby("method", sort=True)
        .agg(
            hypervolume_mean=("verified_hypervolume", "mean"),
            hypervolume_sd=("verified_hypervolume", "std"),
            retained_mean=("verified_nondominated_share", "mean"),
            degradation_mean=("mean_absolute_degradation_pct", "mean"),
        )
        .reset_index()
    )
    best = float(method_summary["hypervolume_mean"].max())
    method_summary["hypervolume_gap_pct"] = (
        (best - method_summary["hypervolume_mean"]) / best * 100.0
    )
    return run_summary, method_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a closed loop interaction aware surrogate experiment."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--initial-points", type=int, default=30)
    parser.add_argument("--total-points", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--search-replications", type=int, default=2)
    parser.add_argument("--external-points", type=int, default=25)
    parser.add_argument("--external-replications", type=int, default=10)
    parser.add_argument("--frontier-points", type=int, default=10)
    parser.add_argument("--verification-replications", type=int, default=10)
    parser.add_argument("--ensemble-members", type=int, default=6)
    parser.add_argument("--algorithm-replications", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--methods",
        type=_csv_strings,
        default=["offline", "pareto_adaptive", "staged_adaptive"],
    )
    parser.add_argument("--reuse-external-dir", type=Path)
    parser.add_argument("--reuse-initial-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.total_points <= args.initial_points:
        raise SystemExit("total points must exceed initial points")
    if (args.total_points - args.initial_points) % args.batch_size:
        raise SystemExit("additional points must be divisible by batch size")

    unknown_methods = sorted(set(args.methods) - set(METHODS))
    if unknown_methods:
        raise SystemExit(f"unknown methods: {unknown_methods}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.reuse_external_dir is not None:
        external = pd.read_csv(args.reuse_external_dir / "external_factorial_test.csv")
        external_seed_level = pd.read_csv(
            args.reuse_external_dir / "external_factorial_seed_level.csv"
        )
    else:
        external_design = sample_policy_states(args.external_points, args.base_seed + 80_000)
        external, external_seed_level = _parallel_factorial(
            args.config,
            external_design,
            tuple(
                range(
                    args.base_seed + 90_000,
                    args.base_seed + 90_000 + args.external_replications,
                )
            ),
            args.workers,
            "external factorial test",
        )
    external.to_csv(args.output_dir / "external_factorial_test.csv", index=False)
    external_seed_level.to_csv(
        args.output_dir / "external_factorial_seed_level.csv", index=False
    )

    learning_frames: list[pd.DataFrame] = []
    verification_frames: list[pd.DataFrame] = []
    acquisition_seed_frames: list[pd.DataFrame] = []
    for replication in range(args.algorithm_replications):
        algorithm_seed = args.base_seed + replication * 1_000
        training_seeds = tuple(
            range(algorithm_seed, algorithm_seed + args.search_replications)
        )
        if args.reuse_initial_dir is not None:
            initial = pd.read_csv(
                args.reuse_initial_dir / f"training_offline_seed_{algorithm_seed}.csv"
            )
            initial = initial.loc[initial["iteration"] == 0].copy()
        else:
            initial_design = sample_policy_states(args.initial_points, algorithm_seed)
            initial, initial_seed_level = _parallel_factorial(
                args.config,
                initial_design,
                training_seeds,
                args.workers,
                f"initial seed {algorithm_seed}",
            )
            initial["iteration"] = 0
            initial_seed_level["method"] = "shared_initial"
            initial_seed_level["algorithm_seed"] = algorithm_seed
            initial_seed_level["iteration"] = 0
            acquisition_seed_frames.append(initial_seed_level)
        for method in args.methods:
            ensemble, training, learning, seed_level = _run_method(
                method,
                algorithm_seed,
                initial,
                args.config,
                external,
                training_seeds,
                args.initial_points,
                args.total_points,
                args.batch_size,
                args.ensemble_members,
                args.workers,
                args.output_dir,
            )
            training.to_csv(
                args.output_dir / f"training_{method}_seed_{algorithm_seed}.csv",
                index=False,
            )
            learning_frames.append(learning)
            if not seed_level.empty:
                acquisition_seed_frames.append(seed_level)
            frontier = _select_frontier(
                ensemble,
                algorithm_seed + METHODS.index(method) * 100,
                args.frontier_points,
            )
            verified = _parallel_joint_verification(
                args.config,
                frontier,
                VERIFICATION_STATES,
                tuple(
                    range(
                        args.base_seed + 100_000,
                        args.base_seed
                        + 100_000
                        + args.verification_replications,
                    )
                ),
                args.workers,
                f"verify {method} seed {algorithm_seed}",
            )
            verified["method"] = method
            verified["algorithm_seed"] = algorithm_seed
            verified = verified.merge(
                frontier[
                    [
                        "predicted_congestion_score",
                        "predicted_intervention_cost",
                        *DESIGN_COLUMNS,
                    ]
                ].reset_index(names="policy_id"),
                on="policy_id",
                how="left",
            )
            verification_frames.append(verified)
            pd.concat(learning_frames, ignore_index=True).to_csv(
                args.output_dir / "learning_curve.csv", index=False
            )
            pd.concat(verification_frames, ignore_index=True).to_csv(
                args.output_dir / "frontier_verification.csv", index=False
            )

    learning_curve = pd.concat(learning_frames, ignore_index=True)
    verification = pd.concat(verification_frames, ignore_index=True)
    seed_level = pd.concat(acquisition_seed_frames, ignore_index=True)
    run_summary, method_summary = _summarize_verified_frontiers(verification)
    learning_curve.to_csv(args.output_dir / "learning_curve.csv", index=False)
    verification.to_csv(args.output_dir / "frontier_verification.csv", index=False)
    seed_level.to_csv(args.output_dir / "adaptive_seed_level.csv", index=False)
    run_summary.to_csv(args.output_dir / "frontier_run_summary.csv", index=False)
    method_summary.to_csv(args.output_dir / "method_summary.csv", index=False)
    metadata = {
        "methods": list(args.methods),
        "algorithm_replications": args.algorithm_replications,
        "initial_points": args.initial_points,
        "total_points": args.total_points,
        "batch_size": args.batch_size,
        "search_replications": args.search_replications,
        "search_abs_budget_per_method": args.total_points
        * 4
        * args.search_replications,
        "external_points": args.external_points,
        "external_replications": args.external_replications,
        "verification_states": VERIFICATION_STATES,
        "verification_replications": args.verification_replications,
        "frontier_points": args.frontier_points,
        "ensemble_members": args.ensemble_members,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(method_summary.to_string(index=False))


if __name__ == "__main__":
    main()
