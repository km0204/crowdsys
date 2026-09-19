from __future__ import annotations

from dataclasses import replace
from typing import Iterable

import numpy as np
import pandas as pd

from .config import ModelConfig
from .model import SimulationResult, simulate


def result_kpis(result: SimulationResult) -> dict[str, float]:
    metadata = result.metadata
    n_agents = float(metadata["n_agents"])
    peak_queue = float(metadata["peak_total_queue"])
    mean_wait = float(metadata["mean_queue_wait_minutes"])
    return {
        "peak_total_queue": peak_queue,
        "peak_queue_share": peak_queue / n_agents,
        "mean_queue_wait_minutes": mean_wait,
        "p90_queue_wait_minutes": float(metadata["p90_queue_wait_minutes"]),
        "pre_kickoff_arrival_share": float(metadata["pre_kickoff_arrival_share"]),
        "shuttle_share": float(metadata["shuttle_share"]),
        "event_participation_rate": float(metadata["event_participation_rate"]),
        "intervention_cost": float(metadata["intervention_cost"]),
        "congestion_score": mean_wait + 20.0 * peak_queue / n_agents,
    }


def evaluate_config(config: ModelConfig, seeds: Iterable[int]) -> dict[str, float]:
    rows = [result_kpis(simulate(config, int(seed))) for seed in seeds]
    frame = pd.DataFrame(rows)
    summary = {column: float(frame[column].mean()) for column in frame.columns}
    summary.update(
        {
            f"{column}_sd": float(
                frame[column].std(ddof=1) if len(frame) > 1 else 0.0
            )
            for column in frame.columns
        }
    )
    summary["replications"] = float(len(rows))
    return summary


def capacity_interaction_analysis(
    base: ModelConfig,
    demand_scales: Iterable[float],
    capacity_scales: Iterable[float],
    seeds: Iterable[int],
    shuttle_awareness: float = 0.6,
    event_awareness: float = 0.6,
    tolerance: float = 1e-4,
) -> pd.DataFrame:
    """Evaluate spatial-temporal interaction using a difference-in-differences metric.

    Positive interaction means the joint reduction in congestion exceeds the sum
    of separate reductions (complementarity); negative values mean overlap
    (substitution). Common seeds are used for all four portfolios in each regime.
    """
    seeds = tuple(int(seed) for seed in seeds)
    rows: list[dict[str, float | str]] = []
    portfolios = {
        "none": (False, False, 0.0, 0.0),
        "spatial": (True, False, shuttle_awareness, 0.0),
        "temporal": (False, True, 0.0, event_awareness),
        "joint": (True, True, shuttle_awareness, event_awareness),
    }
    for demand_scale in demand_scales:
        for capacity_scale in capacity_scales:
            regime = replace(
                base,
                n_agents=max(1, int(round(base.n_agents * float(demand_scale)))),
                capacity_scale=float(capacity_scale),
            )
            scores: dict[str, float] = {}
            for name, policy in portfolios.items():
                configured = regime.with_policy(
                    shuttle_enabled=policy[0],
                    event_enabled=policy[1],
                    shuttle_awareness=policy[2],
                    event_awareness=policy[3],
                )
                summary = evaluate_config(configured, seeds)
                scores[name] = summary["congestion_score"]
            spatial_benefit = scores["none"] - scores["spatial"]
            temporal_benefit = scores["none"] - scores["temporal"]
            joint_benefit = scores["none"] - scores["joint"]
            interaction = joint_benefit - spatial_benefit - temporal_benefit
            classification = (
                "complementarity"
                if interaction > tolerance
                else "substitution"
                if interaction < -tolerance
                else "additive"
            )
            rows.append(
                {
                    "demand_scale": float(demand_scale),
                    "capacity_scale": float(capacity_scale),
                    "baseline_score": scores["none"],
                    "spatial_score": scores["spatial"],
                    "temporal_score": scores["temporal"],
                    "joint_score": scores["joint"],
                    "spatial_benefit": spatial_benefit,
                    "temporal_benefit": temporal_benefit,
                    "joint_benefit": joint_benefit,
                    "interaction": interaction,
                    "classification": classification,
                }
            )
    return pd.DataFrame(rows)


DESIGN_COLUMNS = [
    "shuttle_awareness",
    "event_awareness",
    "fleet_size",
    "event_candidate_rate",
    "event_target_join_rate",
]


def sample_designs(n_samples: int, seed: int = 123) -> pd.DataFrame:
    """Create a reproducible stratified design without a SciPy dependency."""
    if n_samples < 2:
        raise ValueError("n_samples must be at least two")
    rng = np.random.RandomState(seed)
    unit = np.empty((n_samples, len(DESIGN_COLUMNS)))
    for column in range(unit.shape[1]):
        unit[:, column] = (rng.permutation(n_samples) + rng.random(n_samples)) / n_samples
    return pd.DataFrame(
        {
            "shuttle_awareness": unit[:, 0],
            "event_awareness": unit[:, 1],
            "fleet_size": np.rint(1 + unit[:, 2] * 19).astype(int),
            "event_candidate_rate": unit[:, 3] * 0.70,
            "event_target_join_rate": unit[:, 4],
        }
    )


def config_from_design(base: ModelConfig, row: pd.Series | dict[str, float]) -> ModelConfig:
    shuttle = replace(base.shuttle, enabled=True, fleet_size=int(round(row["fleet_size"])))
    event = replace(
        base.event,
        enabled=True,
        candidate_rate=float(row["event_candidate_rate"]),
        target_join_rate=float(row["event_target_join_rate"]),
    )
    information = replace(
        base.information,
        enabled=True,
        shuttle_awareness_rate=float(row["shuttle_awareness"]),
        event_awareness_rate=float(row["event_awareness"]),
    )
    return replace(base, shuttle=shuttle, event=event, information=information)


def run_design(
    base: ModelConfig,
    designs: pd.DataFrame,
    seeds: Iterable[int],
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    seeds = tuple(int(seed) for seed in seeds)
    for design_id, design in designs.reset_index(drop=True).iterrows():
        summary = evaluate_config(config_from_design(base, design), seeds)
        rows.append(
            {
                "design_id": float(design_id),
                **{key: float(design[key]) for key in DESIGN_COLUMNS},
                **summary,
            }
        )
    return pd.DataFrame(rows)


def sensitivity_analysis(
    base: ModelConfig,
    parameter: str,
    values: Iterable[float],
    seeds: Iterable[int],
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for value in values:
        if parameter == "capacity_scale":
            config = replace(base, capacity_scale=float(value))
        elif parameter == "demand_scale":
            config = replace(base, n_agents=max(1, int(round(base.n_agents * float(value)))))
        elif parameter == "shuttle_awareness":
            config = base.with_policy(shuttle_enabled=True, shuttle_awareness=float(value))
        elif parameter == "event_awareness":
            config = base.with_policy(event_enabled=True, event_awareness=float(value))
        else:
            raise ValueError(f"unsupported sensitivity parameter: {parameter}")
        rows.append({"parameter": parameter, "value": float(value), **evaluate_config(config, seeds)})
    return pd.DataFrame(rows)


def robustness_analysis(
    base: ModelConfig,
    n_cases: int,
    base_seed: int = 1000,
    demand_sd: float = 0.10,
    capacity_sd: float = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.RandomState(base_seed)
    rows: list[dict[str, float]] = []
    for case in range(n_cases):
        demand_multiplier = max(0.25, float(rng.normal(1.0, demand_sd)))
        capacity_multiplier = max(0.25, float(rng.normal(1.0, capacity_sd)))
        config = replace(
            base,
            n_agents=max(1, int(round(base.n_agents * demand_multiplier))),
            capacity_scale=base.capacity_scale * capacity_multiplier,
        )
        kpis = result_kpis(simulate(config, base_seed + case))
        rows.append(
            {
                "case": float(case),
                "seed": float(base_seed + case),
                "demand_multiplier": demand_multiplier,
                "capacity_multiplier": capacity_multiplier,
                **kpis,
            }
        )
    cases = pd.DataFrame(rows)
    metrics = ["congestion_score", "peak_total_queue", "mean_queue_wait_minutes", "pre_kickoff_arrival_share"]
    summary_rows = []
    for metric in metrics:
        series = cases[metric]
        summary_rows.append(
            {
                "metric": metric,
                "mean": float(series.mean()),
                "sd": float(series.std(ddof=1) if len(series) > 1 else 0.0),
                "q05": float(series.quantile(0.05)),
                "median": float(series.median()),
                "q95": float(series.quantile(0.95)),
            }
        )
    return cases, pd.DataFrame(summary_rows)
