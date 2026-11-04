from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from run_portfolio_experiment import _parallel_evaluate

from stadium_ingress.analysis import DESIGN_COLUMNS
from stadium_ingress.optimisation import (
    _decode,
    _fronts,
    _rank_and_crowding,
    _select,
    _tournament_winner,
    resource_index_from_designs,
)
from stadium_ingress.validation import (
    frontier_verification_metrics,
    hypervolume_2d,
    nondominated_mask,
)


def _offspring(
    population: np.ndarray,
    objectives: np.ndarray,
    rng: np.random.RandomState,
) -> np.ndarray:
    ranks, crowding = _rank_and_crowding(objectives)
    children: list[np.ndarray] = []
    while len(children) < len(population):
        contestants = rng.randint(0, len(population), size=4)
        parent_a = population[
            _tournament_winner(contestants[0], contestants[1], ranks, crowding)
        ]
        parent_b = population[
            _tournament_winner(contestants[2], contestants[3], ranks, crowding)
        ]
        mix = rng.random(len(DESIGN_COLUMNS))
        child_a = mix * parent_a + (1.0 - mix) * parent_b
        child_b = mix * parent_b + (1.0 - mix) * parent_a
        for child in (child_a, child_b):
            mutation = rng.random(len(DESIGN_COLUMNS)) < (1.0 / len(DESIGN_COLUMNS))
            child[mutation] += rng.normal(0.0, 0.08, size=int(mutation.sum()))
            children.append(np.clip(child, 0.0, 1.0))
            if len(children) == len(population):
                break
    return np.asarray(children)


def _direct_search(
    config: Path,
    population_size: int,
    generations: int,
    replications: int,
    base_seed: int,
    workers: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.RandomState(base_seed)
    population = rng.random((population_size, len(DESIGN_COLUMNS)))
    seeds = tuple(range(base_seed, base_seed + replications))
    history: list[pd.DataFrame] = []

    def evaluate(values: np.ndarray, generation: int) -> np.ndarray:
        designs = _decode(values)
        results = _parallel_evaluate(
            config,
            designs,
            seeds,
            workers,
            f"direct generation {generation}",
        )
        record = designs.copy()
        record.insert(0, "candidate", np.arange(len(record)))
        record.insert(0, "generation", generation)
        record["estimated_congestion_score"] = results["congestion_score"]
        record["resource_index"] = resource_index_from_designs(designs)
        history.append(record)
        return record[["estimated_congestion_score", "resource_index"]].to_numpy(
            dtype=float
        )

    objectives = evaluate(population, 0)
    for generation in range(1, generations + 1):
        children = _offspring(population, objectives, rng)
        child_objectives = evaluate(children, generation)
        population, objectives = _select(
            np.vstack([population, children]),
            np.vstack([objectives, child_objectives]),
            population_size,
        )

    front, _ = _fronts(objectives)
    indices = front[0]
    pareto = _decode(population[indices])
    pareto["predicted_congestion_score"] = objectives[indices, 0]
    pareto["predicted_intervention_cost"] = objectives[indices, 1]
    pareto = (
        pareto.drop_duplicates()
        .sort_values("predicted_intervention_cost")
        .reset_index(drop=True)
    )
    pareto.insert(0, "pareto_id", np.arange(1, len(pareto) + 1))
    return pareto, pd.concat(history, ignore_index=True)


def _common_hypervolume_summary(
    direct: pd.DataFrame,
    xgboost: pd.DataFrame,
    polynomial: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    frames = {
        "Direct ABS NSGA II": direct,
        "XGBoost NSGA II": xgboost,
        "Quadratic NSGA II": polynomial,
    }
    arrays = {
        name: frame[
            ["simulated_congestion_score", "simulated_intervention_cost"]
        ].to_numpy(dtype=float)
        for name, frame in frames.items()
    }
    combined = np.vstack(list(arrays.values()))
    maximum = combined.max(axis=0)
    minimum = combined.min(axis=0)
    reference = maximum + np.maximum(0.05 * (maximum - minimum), 1e-9)
    hypervolumes = {
        name: hypervolume_2d(values, reference) for name, values in arrays.items()
    }
    best = max(hypervolumes.values())
    rows: list[dict[str, float | str]] = []
    for name, frame in frames.items():
        _, metrics = frontier_verification_metrics(frame)
        values = arrays[name]
        rows.append(
            {
                "method": name,
                "search_abs_runs": 300.0,
                "search_candidate_slots": (
                    100.0 if name == "Direct ABS NSGA II" else 50_200.0
                ),
                "verification_abs_runs": float(30 * len(frame)),
                "verified_points": float(len(frame)),
                "verified_nondominated_points": float(
                    nondominated_mask(values).sum()
                ),
                "retained_nondominated_share": metrics[
                    "retained_nondominated_share"
                ],
                "mean_absolute_congestion_degradation_pct": metrics[
                    "mean_absolute_congestion_degradation_pct"
                ],
                "verified_hypervolume": hypervolumes[name],
                "verified_hypervolume_gap_pct": (
                    (best - hypervolumes[name]) / best * 100.0 if best > 0.0 else 0.0
                ),
            }
        )
    return pd.DataFrame(rows), reference


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare direct ABS and surrogate searches under equal ABS budgets."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fixed-budget-dir", type=Path, required=True)
    parser.add_argument("--population-size", type=int, default=20)
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--search-replications", type=int, default=3)
    parser.add_argument("--verify-solutions", type=int, default=20)
    parser.add_argument("--verification-replications", type=int, default=30)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    search_runs = (
        args.population_size
        * (args.generations + 1)
        * args.search_replications
    )
    if search_runs != 300:
        raise SystemExit("the manuscript benchmark requires exactly 300 search ABS runs")
    if args.verification_replications != 30:
        raise SystemExit("the manuscript benchmark requires 30 verification replications")

    pareto, history = _direct_search(
        args.config,
        args.population_size,
        args.generations,
        args.search_replications,
        args.base_seed,
        args.workers,
    )
    count = min(args.verify_solutions, len(pareto))
    indices = np.unique(np.linspace(0, len(pareto) - 1, count, dtype=int))
    selected = pareto.iloc[indices].reset_index(drop=True)
    verified = _parallel_evaluate(
        args.config,
        selected[DESIGN_COLUMNS],
        tuple(
            range(
                args.base_seed + 20_000,
                args.base_seed + 20_000 + args.verification_replications,
            )
        ),
        args.workers,
        "direct frontier verification",
    )
    direct = selected.copy()
    direct["simulated_congestion_score"] = verified["congestion_score"]
    direct["simulated_intervention_cost"] = verified["intervention_cost"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    history.to_csv(args.output_dir / "direct_search_history.csv", index=False)
    pareto.to_csv(args.output_dir / "direct_pareto.csv", index=False)
    direct.to_csv(args.output_dir / "direct_verification.csv", index=False)

    xgboost = pd.read_csv(args.fixed_budget_dir / "xgboost_verification.csv")
    xgboost = xgboost.iloc[
        np.unique(np.linspace(0, len(xgboost) - 1, min(20, len(xgboost)), dtype=int))
    ].reset_index(drop=True)
    polynomial = pd.read_csv(args.fixed_budget_dir / "polynomial_verification.csv")
    polynomial = polynomial.iloc[
        np.unique(
            np.linspace(0, len(polynomial) - 1, min(20, len(polynomial)), dtype=int)
        )
    ].reset_index(drop=True)
    summary, reference = _common_hypervolume_summary(
        direct,
        xgboost,
        polynomial,
    )

    xgboost.to_csv(args.output_dir / "xgboost_verification.csv", index=False)
    polynomial.to_csv(args.output_dir / "polynomial_verification.csv", index=False)
    summary.to_csv(args.output_dir / "direct_vs_surrogates_summary.csv", index=False)
    metadata = {
        "search_abs_budget_per_method": search_runs,
        "verification_replications_per_policy": args.verification_replications,
        "direct_candidate_slots": args.population_size * (args.generations + 1),
        "direct_common_search_seeds": list(
            range(args.base_seed, args.base_seed + args.search_replications)
        ),
        "common_verification_seeds": list(
            range(
                args.base_seed + 20_000,
                args.base_seed + 20_000 + args.verification_replications,
            )
        ),
        "hypervolume_reference": reference.tolist(),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
