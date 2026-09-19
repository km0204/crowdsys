from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import (
    capacity_interaction_analysis,
    config_from_design,
    evaluate_config,
    robustness_analysis,
    run_design,
    sample_designs,
    sensitivity_analysis,
)
from .config import ModelConfig, load_config, load_walk_time_overrides
from .experiment import aggregate_replications, run_replications, summarize_peaks
from .optimisation import nsga2
from .surrogate import fit_surrogates, surrogate_diagnostics


def _csv_floats(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


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
    interaction.add_argument("--shuttle-awareness", type=float, default=0.6)
    interaction.add_argument("--event-awareness", type=float, default=0.6)
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
    frame = capacity_interaction_analysis(
        config,
        args.demand_scales,
        args.capacity_scales,
        seeds,
        args.shuttle_awareness,
        args.event_awareness,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "capacity_policy_interaction.csv", index=False)
    if not args.no_plot:
        from .plotting import plot_interaction_heatmap

        plot_interaction_heatmap(frame, args.output_dir / "capacity_policy_interaction.png")
    print(frame[["demand_scale", "capacity_scale", "interaction", "classification"]].to_string(index=False))


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
    pd.DataFrame(verification_rows).to_csv(
        args.output_dir / "pareto_verification.csv", index=False
    )
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
            "important_note": (
                "The full Pareto set contains surrogate predictions; selected solutions are "
                "reported separately in pareto_verification.csv."
            ),
        },
    )
    print(f"Created {len(pareto)} Pareto solutions using {surrogate.backend}.")


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
    }
    handlers[args.command](args)
