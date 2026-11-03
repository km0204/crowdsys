from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
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
from stadium_ingress.surrogate import fit_surrogates
from stadium_ingress.validation import select_balanced_solution

SCENARIOS = [
    ("demand_120", 1.20, 1.00),
    ("demand_130", 1.30, 1.00),
    ("capacity_075", 1.00, 0.75),
    ("demand_120_capacity_075", 1.20, 0.75),
]


def _scenario_config(config_path: str, demand_scale: float, capacity_multiplier: float):
    base = load_config(config_path)
    return replace(
        base,
        n_agents=max(1, int(round(base.n_agents * demand_scale))),
        capacity_scale=base.capacity_scale * capacity_multiplier,
    )


def _evaluate_training(
    task: tuple[str, str, float, float, int, dict[str, float], tuple[int, ...]]
) -> dict[str, float | str]:
    config_path, scenario, demand, capacity, design_id, design, seeds = task
    config = config_from_design(_scenario_config(config_path, demand, capacity), design)
    return {
        "scenario": scenario,
        "demand_scale": demand,
        "capacity_multiplier": capacity,
        "design_id": float(design_id),
        **{column: float(design[column]) for column in DESIGN_COLUMNS},
        **evaluate_config(config, seeds),
    }


def _fit_scenario(
    task: tuple[str, pd.DataFrame, int, int, int]
) -> tuple[str, pd.DataFrame, dict[str, float]]:
    scenario, training, population_size, generations, seed = task
    surrogate = fit_surrogates(training, backend="xgboost", seed=seed)
    pareto = nsga2(
        surrogate,
        population_size=population_size,
        generations=generations,
        seed=seed,
    )
    balanced = select_balanced_solution(pareto)
    return scenario, pareto, {column: float(balanced[column]) for column in DESIGN_COLUMNS}


def _verify_policy(
    task: tuple[
        str,
        str,
        float,
        float,
        str,
        dict[str, float],
        tuple[int, ...],
    ]
) -> dict[str, float | str]:
    config_path, scenario, demand, capacity, policy, design, seeds = task
    config = config_from_design(_scenario_config(config_path, demand, capacity), design)
    return {
        "scenario": scenario,
        "demand_scale": demand,
        "capacity_multiplier": capacity,
        "policy": policy,
        **{column: float(design[column]) for column in DESIGN_COLUMNS},
        **evaluate_config(config, seeds),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare a fixed balanced policy with scenario specific reoptimization."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reference-policies", type=Path, required=True)
    parser.add_argument("--designs", type=int, default=60)
    parser.add_argument("--replications", type=int, default=3)
    parser.add_argument("--population-size", type=int, default=200)
    parser.add_argument("--generations", type=int, default=250)
    parser.add_argument("--verification-replications", type=int, default=30)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    policies = pd.read_csv(args.reference_policies)
    fixed_row = policies.loc[policies["policy"] == "balanced compromise"].iloc[0]
    fixed_design = {column: float(fixed_row[column]) for column in DESIGN_COLUMNS}
    designs = sample_designs(args.designs, seed=args.base_seed + 2_000)
    training_tasks = []
    for scenario_index, (name, demand, capacity) in enumerate(SCENARIOS, start=1):
        seeds = tuple(
            range(
                args.base_seed + scenario_index * 1_000,
                args.base_seed + scenario_index * 1_000 + args.replications,
            )
        )
        for design_id, design in designs.iterrows():
            training_tasks.append(
                (
                    str(args.config),
                    name,
                    demand,
                    capacity,
                    int(design_id),
                    design.to_dict(),
                    seeds,
                )
            )

    training_rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_evaluate_training, task) for task in training_tasks]
        for completed, future in enumerate(as_completed(futures), start=1):
            training_rows.append(future.result())
            if completed % args.designs == 0:
                print(
                    f"scenario training: completed {completed}/{len(training_tasks)} designs",
                    flush=True,
                )
    training = pd.DataFrame(training_rows).sort_values(
        ["scenario", "design_id"], ignore_index=True
    )

    fit_tasks = []
    for scenario_index, (name, _, _) in enumerate(SCENARIOS, start=1):
        fit_tasks.append(
            (
                name,
                training.loc[training["scenario"] == name].copy(),
                args.population_size,
                args.generations,
                args.base_seed + scenario_index * 10_000,
            )
        )
    pareto_by_scenario: dict[str, pd.DataFrame] = {}
    design_by_scenario: dict[str, dict[str, float]] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_fit_scenario, task) for task in fit_tasks]
        for future in as_completed(futures):
            scenario, pareto, balanced = future.result()
            pareto_by_scenario[scenario] = pareto
            design_by_scenario[scenario] = balanced
            print(f"scenario optimization completed: {scenario}", flush=True)

    verification_tasks = []
    for scenario_index, (name, demand, capacity) in enumerate(SCENARIOS, start=1):
        seeds = tuple(
            range(
                args.base_seed + 70_000 + scenario_index * 1_000,
                args.base_seed
                + 70_000
                + scenario_index * 1_000
                + args.verification_replications,
            )
        )
        verification_tasks.extend(
            [
                (
                    str(args.config),
                    name,
                    demand,
                    capacity,
                    "fixed reference",
                    fixed_design,
                    seeds,
                ),
                (
                    str(args.config),
                    name,
                    demand,
                    capacity,
                    "scenario specific",
                    design_by_scenario[name],
                    seeds,
                ),
            ]
        )
    verification_rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_verify_policy, task) for task in verification_tasks]
        for completed, future in enumerate(as_completed(futures), start=1):
            verification_rows.append(future.result())
            print(f"scenario verification: completed {completed}/8", flush=True)
    verification = pd.DataFrame(verification_rows).sort_values(
        ["scenario", "policy"], ignore_index=True
    )

    comparison_rows = []
    for name, demand, capacity in SCENARIOS:
        subset = verification.loc[verification["scenario"] == name].set_index("policy")
        fixed = subset.loc["fixed reference"]
        reoptimized = subset.loc["scenario specific"]
        pareto = pareto_by_scenario[name]
        objective_columns = ["predicted_congestion_score", "predicted_intervention_cost"]
        bounds = pareto[objective_columns].to_numpy(dtype=float)
        lower = bounds.min(axis=0)
        span = bounds.max(axis=0) - lower
        span[span == 0.0] = 1.0
        fixed_vector = np.array([fixed["congestion_score"], fixed["intervention_cost"]])
        reoptimized_vector = np.array(
            [reoptimized["congestion_score"], reoptimized["intervention_cost"]]
        )
        fixed_score = float(np.linalg.norm((fixed_vector - lower) / span))
        reoptimized_score = float(np.linalg.norm((reoptimized_vector - lower) / span))
        fixed_dominates = bool(
            np.all(fixed_vector <= reoptimized_vector)
            and np.any(fixed_vector < reoptimized_vector)
        )
        reoptimized_dominates = bool(
            np.all(reoptimized_vector <= fixed_vector)
            and np.any(reoptimized_vector < fixed_vector)
        )
        comparison_rows.append(
            {
                "scenario": name,
                "demand_scale": demand,
                "capacity_multiplier": capacity,
                "fixed_congestion_score": fixed["congestion_score"],
                "reoptimized_congestion_score": reoptimized["congestion_score"],
                "congestion_difference": (
                    fixed["congestion_score"] - reoptimized["congestion_score"]
                ),
                "fixed_resource_index": fixed["intervention_cost"],
                "reoptimized_resource_index": reoptimized["intervention_cost"],
                "resource_difference": (
                    fixed["intervention_cost"] - reoptimized["intervention_cost"]
                ),
                "fixed_balanced_score": fixed_score,
                "reoptimized_balanced_score": reoptimized_score,
                "balanced_score_difference": fixed_score - reoptimized_score,
                "dominance": (
                    "fixed dominates"
                    if fixed_dominates
                    else "scenario specific dominates"
                    if reoptimized_dominates
                    else "incomparable"
                ),
            }
        )
    comparison = pd.DataFrame(comparison_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    training.to_csv(args.output_dir / "scenario_training.csv", index=False)
    verification.to_csv(args.output_dir / "scenario_verification.csv", index=False)
    comparison.to_csv(args.output_dir / "scenario_comparison.csv", index=False)
    for name, pareto in pareto_by_scenario.items():
        pareto.to_csv(args.output_dir / f"{name}_pareto.csv", index=False)
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
