from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from statistics import NormalDist

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
        "peak_facility_queue_to_capacity": float(
            metadata["peak_facility_queue_to_capacity"]
        ),
        "arrival_concentration_hhi": float(metadata["arrival_concentration_hhi"]),
        "peak_arrivals_to_capacity": float(metadata["peak_arrivals_to_capacity"]),
        "mean_station_queue_to_capacity": float(
            metadata["mean_station_queue_to_capacity"]
        ),
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


def _student_t_critical_975(degrees_of_freedom: int) -> float:
    """Approximate the two-sided 95 percent Student t critical value.

    The expansion is accurate to better than 1e-3 at 29 degrees of freedom,
    which is the principal use case for the matched 30-replication experiment.
    """
    if degrees_of_freedom < 1:
        raise ValueError("degrees_of_freedom must be positive")
    z = NormalDist().inv_cdf(0.975)
    df = float(degrees_of_freedom)
    return float(
        z
        + (z**3 + z) / (4.0 * df)
        + (5.0 * z**5 + 16.0 * z**3 + 3.0 * z) / (96.0 * df**2)
        + (3.0 * z**7 + 19.0 * z**5 + 17.0 * z**3 - 15.0 * z)
        / (384.0 * df**3)
    )


def capacity_interaction_replications(
    base: ModelConfig,
    demand_scales: Iterable[float],
    capacity_scales: Iterable[float],
    seeds: Iterable[int],
    shuttle_awareness: float = 0.6,
    event_awareness: float = 0.6,
    interaction_metric: str = "congestion_score",
) -> pd.DataFrame:
    """Return seed-level factorial contrasts and mechanism diagnostics."""
    seeds = tuple(int(seed) for seed in seeds)
    if not seeds:
        raise ValueError("at least one seed is required")
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
            normalized_capacity = float(capacity_scale) / base.capacity_scale
            rho = float(demand_scale) / normalized_capacity
            for seed in seeds:
                outcomes: dict[str, dict[str, float]] = {}
                for name, policy in portfolios.items():
                    configured = regime.with_policy(
                        shuttle_enabled=policy[0],
                        event_enabled=policy[1],
                        shuttle_awareness=policy[2],
                        event_awareness=policy[3],
                    )
                    outcomes[name] = result_kpis(simulate(configured, seed))
                if interaction_metric not in outcomes["none"]:
                    raise ValueError(f"unsupported interaction metric: {interaction_metric}")
                scores = {
                    name: values[interaction_metric] for name, values in outcomes.items()
                }
                interaction = (
                    scores["spatial"]
                    + scores["temporal"]
                    - scores["joint"]
                    - scores["none"]
                )
                row: dict[str, float | str] = {
                    "demand_scale": float(demand_scale),
                    "capacity_scale": float(capacity_scale),
                    "normalized_demand_capacity": rho,
                    "seed": float(seed),
                    "interaction_metric": interaction_metric,
                    "baseline_score": scores["none"],
                    "spatial_score": scores["spatial"],
                    "temporal_score": scores["temporal"],
                    "joint_score": scores["joint"],
                    "interaction": interaction,
                }
                for policy_name in ("none", "joint"):
                    for metric in (
                        "peak_facility_queue_to_capacity",
                        "arrival_concentration_hhi",
                        "peak_arrivals_to_capacity",
                    ):
                        row[f"{policy_name}_{metric}"] = outcomes[policy_name][metric]
                rows.append(row)
    return pd.DataFrame(rows)


def summarize_capacity_interaction(replications: pd.DataFrame) -> pd.DataFrame:
    """Summarize matched contrasts with standard errors and 95 percent intervals."""
    required = {"demand_scale", "capacity_scale", "interaction"}
    missing = required - set(replications.columns)
    if missing:
        raise ValueError(f"interaction replications are missing columns: {sorted(missing)}")
    groups = replications.groupby(["demand_scale", "capacity_scale"], sort=True)
    rows: list[dict[str, float | str]] = []
    mechanism_columns = [
        column
        for column in replications.columns
        if column.startswith("none_") or column.startswith("joint_")
    ]
    for (demand_scale, capacity_scale), group in groups:
        interactions = group["interaction"].astype(float)
        replications_count = len(group)
        mean = float(interactions.mean())
        sd = float(interactions.std(ddof=1)) if replications_count > 1 else 0.0
        se = sd / np.sqrt(replications_count) if replications_count > 0 else float("nan")
        critical = (
            _student_t_critical_975(replications_count - 1)
            if replications_count > 1
            else float("nan")
        )
        lower = mean - critical * se if replications_count > 1 else float("nan")
        upper = mean + critical * se if replications_count > 1 else float("nan")
        classification = (
            "supported complementarity"
            if lower > 0.0
            else "supported substitution"
            if upper < 0.0
            else "inconclusive"
        )
        row: dict[str, float | str] = {
            "demand_scale": float(demand_scale),
            "capacity_scale": float(capacity_scale),
            "interaction_metric": str(group["interaction_metric"].iloc[0]),
            "normalized_demand_capacity": float(
                group["normalized_demand_capacity"].iloc[0]
            ),
            "replications": float(replications_count),
            "interaction": mean,
            "interaction_sd": sd,
            "interaction_se": se,
            "ci95_lower": lower,
            "ci95_upper": upper,
            "classification": classification,
        }
        for column in ("baseline_score", "spatial_score", "temporal_score", "joint_score"):
            row[column] = float(group[column].mean())
        for column in mechanism_columns:
            row[column] = float(group[column].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def capacity_interaction_analysis(
    base: ModelConfig,
    demand_scales: Iterable[float],
    capacity_scales: Iterable[float],
    seeds: Iterable[int],
    shuttle_awareness: float = 0.6,
    event_awareness: float = 0.6,
    interaction_metric: str = "congestion_score",
) -> pd.DataFrame:
    """Evaluate matched spatial and temporal interaction with confidence intervals."""
    replications = capacity_interaction_replications(
        base,
        demand_scales,
        capacity_scales,
        seeds,
        shuttle_awareness,
        event_awareness,
        interaction_metric,
    )
    return summarize_capacity_interaction(replications)


def interaction_mechanism_diagnostics(summary: pd.DataFrame) -> pd.DataFrame:
    """Compare descriptive standardized OLS models for the interaction contrast."""
    candidate_models = {
        "state_ratio": ["normalized_demand_capacity"],
        "facility_load": ["joint_peak_facility_queue_to_capacity"],
        "arrival_pressure": ["joint_peak_arrivals_to_capacity"],
        "arrival_concentration": ["joint_arrival_concentration_hhi"],
        "state_plus_facility": [
            "normalized_demand_capacity",
            "joint_peak_facility_queue_to_capacity",
        ],
        "state_plus_arrivals": [
            "normalized_demand_capacity",
            "joint_peak_arrivals_to_capacity",
            "joint_arrival_concentration_hhi",
        ],
        "combined": [
            "normalized_demand_capacity",
            "joint_peak_facility_queue_to_capacity",
            "joint_peak_arrivals_to_capacity",
            "joint_arrival_concentration_hhi",
        ],
    }
    target = summary["interaction"].to_numpy(dtype=float)
    target_sd = float(target.std(ddof=0))
    standardized_target = (target - target.mean()) / target_sd if target_sd else target * 0.0
    rows: list[dict[str, float | str]] = []
    for model_name, predictors in candidate_models.items():
        values = summary[predictors].to_numpy(dtype=float)
        means = values.mean(axis=0)
        sds = values.std(axis=0, ddof=0)
        usable = sds > 0.0
        values = (values[:, usable] - means[usable]) / sds[usable]
        used_predictors = [
            name for name, keep in zip(predictors, usable, strict=True) if keep
        ]
        design = np.column_stack([np.ones(len(values)), values])
        coefficients, *_ = np.linalg.lstsq(design, standardized_target, rcond=None)
        fitted = design @ coefficients
        residual_sum = float(np.sum((standardized_target - fitted) ** 2))
        total_sum = float(np.sum((standardized_target - standardized_target.mean()) ** 2))
        r_squared = 1.0 - residual_sum / total_sum if total_sum else 1.0
        predictor_count = len(used_predictors)
        adjusted = (
            1.0
            - (1.0 - r_squared) * (len(values) - 1) / (len(values) - predictor_count - 1)
            if len(values) > predictor_count + 1
            else float("nan")
        )
        row: dict[str, float | str] = {
            "model": model_name,
            "predictors": ";".join(used_predictors),
            "n_cells": float(len(values)),
            "r_squared": r_squared,
            "adjusted_r_squared": adjusted,
        }
        for predictor, coefficient in zip(
            used_predictors, coefficients[1:], strict=True
        ):
            row[f"beta_{predictor}"] = float(coefficient)
        rows.append(row)
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
