from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from stadium_ingress.analysis import config_from_design
from stadium_ingress.config import load_config
from stadium_ingress.model import simulate


def _run_policy(
    policy: str,
    config_path: Path,
    design: pd.Series | None,
    seeds: range,
) -> pd.DataFrame:
    base = load_config(config_path)
    config = base if design is None else config_from_design(base, design)
    frames: list[pd.DataFrame] = []
    for seed in seeds:
        frame = simulate(config, seed).minute_data[
            ["minute_before_kickoff", "arrivals_count", "total_queue_length"]
        ].copy()
        frame.insert(0, "seed", seed)
        frame.insert(0, "policy", policy)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate matched baseline and balanced temporal profiles."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--representatives", type=Path, required=True)
    parser.add_argument("--base-seed", type=int, default=60042)
    parser.add_argument("--replications", type=int, default=30)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    representatives = pd.read_csv(args.representatives)
    balanced = representatives.loc[
        representatives["policy"] == "balanced compromise"
    ].iloc[0]
    seeds = range(args.base_seed, args.base_seed + args.replications)
    seed_level = pd.concat(
        [
            _run_policy("Baseline", args.config, None, seeds),
            _run_policy("Balanced compromise", args.config, balanced, seeds),
        ],
        ignore_index=True,
    )
    summary = (
        seed_level.groupby(["policy", "minute_before_kickoff"], sort=True)
        .agg(
            arrivals_count_mean=("arrivals_count", "mean"),
            arrivals_count_sd=("arrivals_count", "std"),
            total_queue_length_mean=("total_queue_length", "mean"),
            total_queue_length_sd=("total_queue_length", "std"),
            replications=("seed", "nunique"),
        )
        .reset_index()
    )

    peaks: list[dict[str, float | int | str]] = []
    for policy, frame in summary.groupby("policy", sort=False):
        arrival_peak = frame.loc[frame["arrivals_count_mean"].idxmax()]
        queue_peak = frame.loc[frame["total_queue_length_mean"].idxmax()]
        peaks.append(
            {
                "policy": policy,
                "arrival_peak_minute": int(arrival_peak["minute_before_kickoff"]),
                "arrival_peak_count": float(arrival_peak["arrivals_count_mean"]),
                "queue_peak_minute": int(queue_peak["minute_before_kickoff"]),
                "queue_peak_length": float(queue_peak["total_queue_length_mean"]),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_level.to_csv(args.output_dir / "temporal_profile_seed_level.csv", index=False)
    summary.to_csv(args.output_dir / "temporal_profile_summary.csv", index=False)
    pd.DataFrame(peaks).to_csv(args.output_dir / "temporal_profile_peaks.csv", index=False)


if __name__ == "__main__":
    main()
