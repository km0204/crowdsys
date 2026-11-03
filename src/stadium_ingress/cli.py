from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import (
    capacity_interaction_replications,
    config_from_design,
    evaluate_config,
    interaction_mechanism_diagnostics,
    robustness_analysis,
    run_design,
    sample_designs,
    sensitivity_analysis,
    summarize_capacity_interaction,
)
from .config import ModelConfig, load_config, load_walk_time_overrides
from .experiment import aggregate_replications, run_replications, summarize_peaks
from .optimisation import nsga2
from .surrogate import fit_surrogates, surrogate_diagnostics
from .validation import (
    frontier_verification_metrics,
    hypervolume_2d,
    nondominated_mask,
    select_balanced_solution,
)


def _csv_floats(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def _csv_strings(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=None, help="JSON configuration file")
    parser.add_argument("--n-agents", type=int, default=None)
    parser.add_argument("--base-seed", type=int, default=42)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stadium ingress simulation and analysis")
    commands = parser.add_subparsers(dest="command", required=True)

    simulate = commands.add_parser("simulate", help="run one scenario")
    _common(simulate)
    simulate.add_argument("--output-dir", type=Path, default=Path("outputs/simulation"))
    simulate.add_argument("--replications", type=int, default=30)
    simulate.add_argument("--walk-times-json", type=Path, default=None)
    simulate.add_argument("--no-plot", action="store_true")

    interaction = commands.add_parser("interaction", help="capacity-dependent policy interaction")
    _common(interaction)
    interaction.add_argument("--output-dir", type=Path, default=Path("outputs/interaction"))
    interaction.add_argument("--demand-scales", type=_csv_floats, default=[0.7, 1.0, 1.3])
    interaction.add_argument("--capacity-scales", type=_csv_floats, default=[0.2, 0.3, 0.4])
    interaction.add_argument("--replications", type=int, default=5)
    interaction.add_argument(
        "--refine-inconclusive",
        type=int,
        default=0,
        help="additional matched seeds for cells whose initial 95%% interval includes zero",
    )
    interaction.add_argument("--shuttle-awareness", type=float, default=0.6)
    interaction.add_argument("--event-awareness", type=float, default=0.6)
    interaction.add_argument(
        "--interaction-metric",
        choices=[
            "mean_queue_wait_minutes",
            "mean_station_queue_to_capacity",
            "congestion_score",
            "peak_total_queue",
            "pre_kickoff_arrival_share",
        ],
        default="mean_queue_wait_minutes",
    )
    interaction.add_argument("--no-plot", action="store_true")

    optimise = commands.add_parser("optimise", aliases=["optimize"], help="surrogate-assisted NSGA-II")
    _common(optimise)
    optimise.add_argument("--output-dir", type=Path, default=Path("outputs/optimisation"))
    optimise.add_argument("--design-samples", type=int, default=40)
    optimise.add_argument("--replications", type=int, default=3)
    optimise.add_argument("--surrogate-backend", choices=["auto", "xgboost", "polynomial"], default="auto")
    optimise.add_argument("--population-size", type=int, default=80)
    optimise.add_argument("--generations", type=int, default=100)
    optimise.add_argument("--verify-solutions", type=int, default=5)
    optimise.add_argument("--verification-replications", type=int, default=5)
    optimise.add_argument("--no-plot", action="store_true")

    sensitivity = commands.add_parser("sensitivity", help="one-at-a-time sensitivity analysis")
    _common(sensitivity)
    sensitivity.add_argument("--output-dir", type=Path, default=Path("outputs/sensitivity"))
    sensitivity.add_argument(
        "--parameter",
        choices=["capacity_scale", "demand_scale", "shuttle_awareness", "event_awareness"],
        default="capacity_scale",
    )
    sensitivity.add_argument("--values", type=_csv_floats, default=[0.2, 0.25, 0.3, 0.35, 0.4])
    sensitivity.add_argument("--replications", type=int, default=5)
    sensitivity.add_argument("--no-plot", action="store_true")

    robustness = commands.add_parser("robustness", help="stochastic demand/capacity robustness")
    _common(robustness)
    robustness.add_argument("--output-dir", type=Path, default=Path("outputs/robustness"))
    robustness.add_argument("--cases", type=int, default=30)
    robustness.add_argument("--demand-sd", type=float, default=0.10)
    robustness.add_argument("--capacity-sd", type=float, default=0.10)

    regret = commands.add_parser(
        "scenario-regret",
        help="compare a fixed reference compromise with scenario-specific reoptimisation",
    )
    _common(regret)
    regret.add_argument("--output-dir", type=Path, default=Path("outputs/scenario_regret"))
    regret.add_argument("--demand-scales", type=_csv_floats, default=[1.1, 1.2, 1.3])
    regret.add_argument("--capacity-scales", type=_csv_floats, default=[1.0])
    regret.add_argument("--design-samples", type=int, default=40)
    regret.add_argument("--replications", type=int, default=3)
    regret.add_argument(
        "--surrogate-backend",
        choices=["auto", "xgboost", "polynomial"],
        default="auto",
    )
    regret.add_argument("--population-size", type=int, default=80)
    regret.add_argument("--generations", type=int, default=100)
    regret.add_argument("--verification-replications", type=int, default=20)

    benchmark = commands.add_parser(
        "fixed-budget-benchmark",
        help="compare surrogate search with direct screening under a documented ABS budget",
    )
    _common(benchmark)
    benchmark.add_argument(
        "--output-dir", type=Path, default=Path("outputs/fixed_budget_benchmark")
    )
    benchmark.add_argument("--design-samples", type=int, default=60)
    benchmark.add_argument("--replications", type=int, default=5)
    benchmark.add_argument(
        "--surrogate-backends", type=_csv_strings, default=["polynomial", "xgboost"]
    )
    benchmark.add_argument("--population-size", type=int, default=100)
    benchmark.add_argument("--generations", type=int, default=150)
    benchmark.add_argument("--verify-solutions", type=int, default=10)
    benchmark.add_argument("--verification-replications", type=int, default=10)
    return parser


def _configured(args: argparse.Namespace) -> ModelConfig:
    config = load_config(args.config)
    if args.n_agents is not None:
        config = replace(config, n_agents=args.n_agents)
    config.validate()
    return config


def _write_json(path: Path, values: dict) -> None:
    path.write_text(json.dumps(values, ensure_ascii=False, indent=2), encoding="utf-8")


def _simulate(args: argparse.Namespace) -> None:
    if args.replications <= 0:
        raise SystemExit("--replications must be positive")
    config = _configured(args)
    overrides = load_walk_time_overrides(args.walk_times_json)
    if args.walk_times_json is not None and not args.walk_times_json.exists():
        print(f"[warning] {args.walk_times_json} was not found; using default walking times", file=sys.stderr)
    if overrides:
        config = config.with_walk_time_overrides(overrides)
    seeds = range(args.base_seed, args.base_seed + args.replications)
    results = run_replications(config, seeds)
    aggregated = aggregate_replications(results)
    peaks = summarize_peaks(aggregated)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    aggregated.to_csv(args.output_dir / "arrival_queue_timeseries.csv", index=False, float_format="%.6f")
    representative = results[0]
    representative.arrivals.to_csv(args.output_dir / "representative_arrivals.csv", index=False)
    representative.choices.to_csv(args.output_dir / "representative_choices.csv", index=False)
    if not representative.positions.empty:
        representative.positions.to_csv(args.output_dir / "representative_walk_positions.csv", index=False)
    metadata = {
        "replications": args.replications,
        "base_seed": args.base_seed,
        "n_agents_per_replication": config.n_agents,
        "walk_time_overrides_applied": bool(overrides),
        "peaks": peaks,
        "run_diagnostics": [result.metadata for result in results],
    }
    _write_json(args.output_dir / "run_metadata.json", metadata)
    if not args.no_plot:
        from .plotting import plot_arrivals_and_queue, plot_density_map

        plot_arrivals_and_queue(aggregated, args.output_dir / "arrival_queue.png")
        if not representative.positions.empty:
            minute = int(representative.positions["minute"].value_counts().idxmax())
            plot_density_map(
                representative.positions,
                args.output_dir / "pedestrian_density.png",
                minute=minute,
            )
    print(json.dumps(peaks, ensure_ascii=False, indent=2))


def _interaction(args: argparse.Namespace) -> None:
    if args.replications <= 0:
        raise SystemExit("--replications must be positive")
    config = _configured(args)
    seeds = range(args.base_seed, args.base_seed + args.replications)
    replications = capacity_interaction_replications(
        config,
        args.demand_scales,
        args.capacity_scales,
        seeds,
        args.shuttle_awareness,
        args.event_awareness,
        args.interaction_metric,
    )
    frame = summarize_capacity_interaction(replications)
    if args.refine_inconclusive > 0:
        refined: list[pd.DataFrame] = [replications]
        unresolved = frame[frame["classification"] == "inconclusive"]
        extra_seeds = range(
            args.base_seed + args.replications,
            args.base_seed + args.replications + args.refine_inconclusive,
        )
        for _, cell in unresolved.iterrows():
            refined.append(
                capacity_interaction_replications(
                    config,
                    [float(cell["demand_scale"])],
                    [float(cell["capacity_scale"])],
                    extra_seeds,
                    args.shuttle_awareness,
                    args.event_awareness,
                    args.interaction_metric,
                )
            )
        replications = pd.concat(refined, ignore_index=True)
        frame = summarize_capacity_interaction(replications)
    mechanisms = interaction_mechanism_diagnostics(frame)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    replications.to_csv(args.output_dir / "capacity_policy_interaction_replications.csv", index=False)
    frame.to_csv(args.output_dir / "capacity_policy_interaction.csv", index=False)
    mechanisms.to_csv(args.output_dir / "interaction_mechanism_diagnostics.csv", index=False)
    if not args.no_plot:
        from .plotting import plot_interaction_heatmap

        plot_interaction_heatmap(frame, args.output_dir / "capacity_policy_interaction.png")
    print(
        frame[
            [
                "demand_scale",
                "capacity_scale",
                "interaction",
                "ci95_lower",
                "ci95_upper",
                "classification",
            ]
        ].to_string(index=False)
    )


def _optimise(args: argparse.Namespace) -> None:
    config = _configured(args)
    designs = sample_designs(args.design_samples, seed=args.base_seed)
    seeds = range(args.base_seed, args.base_seed + args.replications)
    training = run_design(config, designs, seeds)
    surrogate = fit_surrogates(training, backend=args.surrogate_backend, seed=args.base_seed)
    diagnostics = surrogate_diagnostics(surrogate, training)
    pareto = nsga2(
        surrogate,
        population_size=args.population_size,
        generations=args.generations,
        seed=args.base_seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    training.to_csv(args.output_dir / "design_evaluations.csv", index=False)
    diagnostics.to_csv(args.output_dir / "surrogate_diagnostics.csv", index=False)
    pareto.to_csv(args.output_dir / "pareto_solutions.csv", index=False)
    verification_rows: list[dict[str, float]] = []
    if args.verify_solutions > 0 and not pareto.empty:
        count = min(args.verify_solutions, len(pareto))
        selected = np.unique(np.linspace(0, len(pareto) - 1, count, dtype=int))
        verification_seeds = range(
            args.base_seed + 10_000,
            args.base_seed + 10_000 + args.verification_replications,
        )
        for index in selected:
            solution = pareto.iloc[int(index)]
            verified = evaluate_config(
                config_from_design(config, solution),
                verification_seeds,
            )
            verification_rows.append(
                {
                    "pareto_id": float(solution["pareto_id"]),
                    "predicted_congestion_score": float(
                        solution["predicted_congestion_score"]
                    ),
                    "simulated_congestion_score": verified["congestion_score"],
                    "predicted_intervention_cost": float(
                        solution["predicted_intervention_cost"]
                    ),
                    "simulated_intervention_cost": verified["intervention_cost"],
                    "simulation_replications": float(args.verification_replications),
                }
            )
    verification = pd.DataFrame(verification_rows)
    verification_metrics: dict[str, float] = {}
    if not verification.empty:
        verification, verification_metrics = frontier_verification_metrics(verification)
    verification.to_csv(args.output_dir / "pareto_verification.csv", index=False)
    _write_json(args.output_dir / "frontier_verification_metrics.json", verification_metrics)
    if not args.no_plot:
        from .plotting import plot_pareto_front

        plot_pareto_front(pareto, args.output_dir / "pareto_front.png")
    _write_json(
        args.output_dir / "optimisation_metadata.json",
        {
            "surrogate_backend": surrogate.backend,
            "design_samples": args.design_samples,
            "replications_per_design": args.replications,
            "population_size": args.population_size,
            "generations": args.generations,
            "verified_solutions": len(verification_rows),
            "verification_replications": args.verification_replications,
            "frontier_verification_metrics": verification_metrics,
            "important_note": (
                "The full Pareto set contains surrogate predictions; selected solutions are "
                "reported separately in pareto_verification.csv."
            ),
        },
    )
    print(f"Created {len(pareto)} Pareto solutions using {surrogate.backend}.")


def _scenario_regret(args: argparse.Namespace) -> None:
    if args.design_samples < 2 or args.replications <= 0:
        raise SystemExit("scenario-regret requires at least two designs and positive replications")
    if args.verification_replications <= 0:
        raise SystemExit("--verification-replications must be positive")
    base = _configured(args)
    designs = sample_designs(args.design_samples, seed=args.base_seed)

    def fit_front(config: ModelConfig, seed_offset: int) -> tuple[pd.DataFrame, pd.DataFrame, str]:
        seeds = range(
            args.base_seed + seed_offset,
            args.base_seed + seed_offset + args.replications,
        )
        training = run_design(config, designs, seeds)
        surrogate = fit_surrogates(
            training,
            backend=args.surrogate_backend,
            seed=args.base_seed + seed_offset,
        )
        pareto = nsga2(
            surrogate,
            population_size=args.population_size,
            generations=args.generations,
            seed=args.base_seed + seed_offset,
        )
        return training, pareto, surrogate.backend

    reference_training, reference_pareto, backend = fit_front(base, 0)
    fixed_solution = select_balanced_solution(reference_pareto)
    fixed_design = config_from_design(base, fixed_solution)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference_training.to_csv(args.output_dir / "reference_design_evaluations.csv", index=False)
    reference_pareto.to_csv(args.output_dir / "reference_pareto_solutions.csv", index=False)
    fixed_solution.to_frame().T.to_csv(args.output_dir / "fixed_reference_compromise.csv", index=False)

    rows: list[dict[str, float]] = []
    scenario_number = 0
    for demand_scale in args.demand_scales:
        for capacity_scale in args.capacity_scales:
            scenario_number += 1
            scenario = replace(
                base,
                n_agents=max(1, int(round(base.n_agents * demand_scale))),
                capacity_scale=base.capacity_scale * capacity_scale,
            )
            _, scenario_pareto, _ = fit_front(scenario, scenario_number * 1_000)
            scenario_solution = select_balanced_solution(scenario_pareto)
            scenario_design = config_from_design(scenario, scenario_solution)
            fixed_in_scenario = replace(
                fixed_design,
                n_agents=scenario.n_agents,
                capacity_scale=scenario.capacity_scale,
            )
            verification_seeds = range(
                args.base_seed + 100_000 + scenario_number * 1_000,
                args.base_seed
                + 100_000
                + scenario_number * 1_000
                + args.verification_replications,
            )
            fixed_result = evaluate_config(fixed_in_scenario, verification_seeds)
            reoptimised_result = evaluate_config(scenario_design, verification_seeds)
            congestion_difference = (
                fixed_result["congestion_score"] - reoptimised_result["congestion_score"]
            )
            relative_difference = (
                congestion_difference / reoptimised_result["congestion_score"] * 100.0
                if reoptimised_result["congestion_score"] != 0.0
                else float("nan")
            )
            objective_columns = [
                "predicted_congestion_score",
                "predicted_intervention_cost",
            ]
            bounds = scenario_pareto[objective_columns].to_numpy(dtype=float)
            lower = bounds.min(axis=0)
            span = bounds.max(axis=0) - lower
            span[span == 0.0] = 1.0
            fixed_vector = np.array(
                [fixed_result["congestion_score"], fixed_result["intervention_cost"]]
            )
            reoptimised_vector = np.array(
                [
                    reoptimised_result["congestion_score"],
                    reoptimised_result["intervention_cost"],
                ]
            )
            fixed_balanced_score = float(np.linalg.norm((fixed_vector - lower) / span))
            reoptimised_balanced_score = float(
                np.linalg.norm((reoptimised_vector - lower) / span)
            )
            fixed_dominates = bool(
                np.all(fixed_vector <= reoptimised_vector)
                and np.any(fixed_vector < reoptimised_vector)
            )
            reoptimised_dominates = bool(
                np.all(reoptimised_vector <= fixed_vector)
                and np.any(reoptimised_vector < fixed_vector)
            )
            row: dict[str, float] = {
                "scenario": float(scenario_number),
                "demand_scale": float(demand_scale),
                "capacity_scale": float(capacity_scale),
                "fixed_congestion_score": fixed_result["congestion_score"],
                "reoptimised_congestion_score": reoptimised_result["congestion_score"],
                "congestion_difference": congestion_difference,
                "congestion_difference_pct": relative_difference,
                "fixed_intervention_cost": fixed_result["intervention_cost"],
                "reoptimised_intervention_cost": reoptimised_result["intervention_cost"],
                "cost_difference": (
                    fixed_result["intervention_cost"]
                    - reoptimised_result["intervention_cost"]
                ),
                "fixed_balanced_score": fixed_balanced_score,
                "reoptimised_balanced_score": reoptimised_balanced_score,
                "balanced_score_difference": (
                    fixed_balanced_score - reoptimised_balanced_score
                ),
                "fixed_dominates_reoptimised": float(fixed_dominates),
                "reoptimised_dominates_fixed": float(reoptimised_dominates),
                "tradeoff_incomparable": float(
                    not fixed_dominates and not reoptimised_dominates
                ),
                "verification_replications": float(args.verification_replications),
            }
            for column in (
                "shuttle_headway_minutes",
                "fleet_size",
                "shuttle_vehicle_capacity",
                "event_target_join_rate",
                "event_dwell_mean_minutes",
                "shuttle_awareness",
                "event_awareness",
            ):
                row[f"fixed_{column}"] = float(fixed_solution[column])
                row[f"reoptimised_{column}"] = float(scenario_solution[column])
            rows.append(row)
            scenario_pareto.to_csv(
                args.output_dir / f"scenario_{scenario_number:02d}_pareto_solutions.csv",
                index=False,
            )
    results = pd.DataFrame(rows)
    results.to_csv(args.output_dir / "scenario_regret.csv", index=False)
    _write_json(
        args.output_dir / "scenario_regret_metadata.json",
        {
            "surrogate_backend": backend,
            "design_samples_per_scenario": args.design_samples,
            "replications_per_design": args.replications,
            "population_size": args.population_size,
            "generations": args.generations,
            "verification_replications": args.verification_replications,
            "interpretation": (
                "The independently simulated policies are compared by congestion, cost, "
                "normalized balanced score, and dominance. Congestion difference alone is "
                "not labelled regret because the problem is multiobjective."
            ),
        },
    )
    print(results.to_string(index=False))


def _fixed_budget_benchmark(args: argparse.Namespace) -> None:
    if args.design_samples < 2 or args.replications <= 0:
        raise SystemExit("benchmark requires at least two designs and positive replications")
    config = _configured(args)
    designs = sample_designs(args.design_samples, seed=args.base_seed)
    training = run_design(
        config,
        designs,
        range(args.base_seed, args.base_seed + args.replications),
    )
    direct_values = training[["congestion_score", "intervention_cost"]].to_numpy(
        dtype=float
    )
    direct_front = training.loc[nondominated_mask(direct_values)].copy()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    training.to_csv(args.output_dir / "shared_design_evaluations.csv", index=False)
    direct_front.to_csv(args.output_dir / "direct_abs_screening_front.csv", index=False)

    summary_rows: list[dict[str, float | str]] = []
    for backend_index, backend in enumerate(args.surrogate_backends):
        if backend not in {"polynomial", "xgboost"}:
            raise SystemExit(f"unsupported surrogate backend: {backend}")
        try:
            surrogate = fit_surrogates(training, backend=backend, seed=args.base_seed)
        except ImportError:
            summary_rows.append(
                {
                    "backend": backend,
                    "status": "unavailable",
                    "note": "optional dependency is not installed",
                }
            )
            continue
        pareto = nsga2(
            surrogate,
            population_size=args.population_size,
            generations=args.generations,
            seed=args.base_seed + backend_index,
        )
        count = min(args.verify_solutions, len(pareto))
        selected = np.unique(np.linspace(0, len(pareto) - 1, count, dtype=int))
        verification_rows: list[dict[str, float]] = []
        verification_seeds = range(
            args.base_seed + 10_000 + backend_index * 1_000,
            args.base_seed
            + 10_000
            + backend_index * 1_000
            + args.verification_replications,
        )
        for index in selected:
            solution = pareto.iloc[int(index)]
            verified = evaluate_config(
                config_from_design(config, solution), verification_seeds
            )
            verification_rows.append(
                {
                    "pareto_id": float(solution["pareto_id"]),
                    "predicted_congestion_score": float(
                        solution["predicted_congestion_score"]
                    ),
                    "simulated_congestion_score": verified["congestion_score"],
                    "predicted_intervention_cost": float(
                        solution["predicted_intervention_cost"]
                    ),
                    "simulated_intervention_cost": verified["intervention_cost"],
                }
            )
        verification = pd.DataFrame(verification_rows)
        verification, metrics = frontier_verification_metrics(verification)
        simulated = verification[
            ["simulated_congestion_score", "simulated_intervention_cost"]
        ].to_numpy(dtype=float)
        direct = direct_front[["congestion_score", "intervention_cost"]].to_numpy(
            dtype=float
        )
        combined = np.vstack([direct, simulated])
        maxima = combined.max(axis=0)
        minima = combined.min(axis=0)
        reference = maxima + np.maximum(0.05 * (maxima - minima), 1e-9)
        direct_hypervolume = hypervolume_2d(direct, reference)
        surrogate_hypervolume = hypervolume_2d(simulated, reference)
        pareto.to_csv(args.output_dir / f"{backend}_pareto_solutions.csv", index=False)
        verification.to_csv(
            args.output_dir / f"{backend}_pareto_verification.csv", index=False
        )
        summary_rows.append(
            {
                "backend": backend,
                "status": "completed",
                "initial_abs_runs": float(args.design_samples * args.replications),
                "verification_abs_runs": float(
                    len(verification) * args.verification_replications
                ),
                "surrogate_evaluation_slots": float(
                    args.population_size * (args.generations + 1)
                ),
                "direct_screening_candidates": float(len(training)),
                "surrogate_pareto_candidates": float(len(pareto)),
                "verified_candidates": float(len(verification)),
                "retained_nondominated_share": metrics[
                    "retained_nondominated_share"
                ],
                "mean_absolute_congestion_degradation_pct": metrics[
                    "mean_absolute_congestion_degradation_pct"
                ],
                "direct_abs_screening_hypervolume": direct_hypervolume,
                "verified_surrogate_hypervolume": surrogate_hypervolume,
                "verified_to_direct_hypervolume_ratio": (
                    surrogate_hypervolume / direct_hypervolume
                    if direct_hypervolume > 0.0
                    else float("nan")
                ),
                "note": (
                    "Direct comparator is the nondominated subset of the shared LHS "
                    "evaluations, not a global direct simulation optimiser."
                ),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output_dir / "fixed_budget_benchmark.csv", index=False)
    print(summary.to_string(index=False))


def _sensitivity(args: argparse.Namespace) -> None:
    config = _configured(args)
    frame = sensitivity_analysis(
        config,
        args.parameter,
        args.values,
        range(args.base_seed, args.base_seed + args.replications),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "sensitivity_results.csv", index=False)
    if not args.no_plot:
        from .plotting import plot_sensitivity

        plot_sensitivity(frame, args.output_dir / "sensitivity_results.png")
    print(frame[["parameter", "value", "congestion_score"]].to_string(index=False))


def _robustness(args: argparse.Namespace) -> None:
    config = _configured(args)
    cases, summary = robustness_analysis(
        config,
        args.cases,
        args.base_seed,
        args.demand_sd,
        args.capacity_sd,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases.to_csv(args.output_dir / "robustness_cases.csv", index=False)
    summary.to_csv(args.output_dir / "robustness_summary.csv", index=False)
    print(summary.to_string(index=False))


def main(argv: list[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0].startswith("-"):
        arguments.insert(0, "simulate")
    args = build_parser().parse_args(arguments)
    handlers = {
        "simulate": _simulate,
        "interaction": _interaction,
        "optimise": _optimise,
        "optimize": _optimise,
        "sensitivity": _sensitivity,
        "robustness": _robustness,
        "scenario-regret": _scenario_regret,
        "fixed-budget-benchmark": _fixed_budget_benchmark,
    }
    handlers[args.command](args)
