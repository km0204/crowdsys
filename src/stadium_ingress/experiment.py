from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .config import ModelConfig
from .model import SimulationResult, simulate


def run_replications(config: ModelConfig, seeds: Iterable[int]) -> list[SimulationResult]:
    return [simulate(config, seed=int(seed)) for seed in seeds]


def aggregate_replications(
    results: list[SimulationResult],
    display_end_minute: int = 0,
) -> pd.DataFrame:
    if not results:
        raise ValueError("at least one simulation result is required")

    frames: list[pd.DataFrame] = []
    for replication, result in enumerate(results, start=1):
        frame = result.minute_data.copy()
        frame["replication"] = replication
        frame["seed"] = int(result.metadata["seed"])
        frames.append(frame)

    all_runs = pd.concat(frames, ignore_index=True)
    grouped = all_runs.groupby("minute_before_kickoff", sort=True)
    aggregated = grouped.agg(
        arrivals_count_mean=("arrivals_count", "mean"),
        arrivals_count_sd=("arrivals_count", "std"),
        arrival_rate_pct_mean=("arrival_rate_pct", "mean"),
        arrival_rate_pct_sd=("arrival_rate_pct", "std"),
        total_queue_length_mean=("total_queue_length", "mean"),
        total_queue_length_sd=("total_queue_length", "std"),
    ).reset_index()
    sd_columns = [
        "arrivals_count_sd",
        "arrival_rate_pct_sd",
        "total_queue_length_sd",
    ]
    aggregated[sd_columns] = aggregated[sd_columns].fillna(0.0)

    display_mask = aggregated["minute_before_kickoff"] <= display_end_minute
    maximum_queue = float(aggregated.loc[display_mask, "total_queue_length_mean"].max())
    aggregated["queue_pct_of_pre_kickoff_max"] = np.where(
        maximum_queue > 0,
        aggregated["total_queue_length_mean"] / maximum_queue * 100.0,
        0.0,
    )
    return aggregated


def summarize_peaks(
    aggregated: pd.DataFrame,
    display_end_minute: int = 0,
) -> dict[str, float | int]:
    shown = aggregated[aggregated["minute_before_kickoff"] <= display_end_minute]
    arrival_row = shown.loc[shown["arrival_rate_pct_mean"].idxmax()]
    queue_row = shown.loc[shown["total_queue_length_mean"].idxmax()]
    return {
        "arrival_peak_minute": int(arrival_row["minute_before_kickoff"]),
        "arrival_peak_rate_pct_per_minute": float(arrival_row["arrival_rate_pct_mean"]),
        "queue_peak_minute": int(queue_row["minute_before_kickoff"]),
        "queue_peak_length": float(queue_row["total_queue_length_mean"]),
    }
