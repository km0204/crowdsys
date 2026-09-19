from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from .spatial import density_grid


def plot_arrivals_and_queue(
    aggregated: pd.DataFrame,
    output_path: str | Path,
    display_start_minute: int = -180,
    display_end_minute: int = 0,
) -> None:
    shown = aggregated[
        aggregated["minute_before_kickoff"].between(
            display_start_minute, display_end_minute
        )
    ]
    blue = "#1746B6"
    red = "#E51C23"
    figure, left_axis = plt.subplots(figsize=(12, 7))
    right_axis = left_axis.twinx()

    left_axis.plot(
        shown["minute_before_kickoff"],
        shown["arrival_rate_pct_mean"],
        color=blue,
        linewidth=2.4,
        label="Arrival rate (% of spectators per minute)",
    )
    right_axis.plot(
        shown["minute_before_kickoff"],
        shown["queue_pct_of_pre_kickoff_max"],
        color=red,
        linewidth=2.4,
        label="Total queue length (% of pre-kick-off maximum)",
    )

    left_axis.set_xlabel("Time before kick-off (minutes)")
    left_axis.set_ylabel("Arrival rate (% of total spectators per minute)", color=blue)
    right_axis.set_ylabel("Total queue length (% of pre-kick-off maximum)", color=red)
    left_axis.tick_params(axis="y", colors=blue)
    right_axis.tick_params(axis="y", colors=red)
    left_axis.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    right_axis.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    right_axis.set_ylim(0, 102)
    left_axis.grid(True, linestyle="--", alpha=0.45)

    handles_left, labels_left = left_axis.get_legend_handles_labels()
    handles_right, labels_right = right_axis.get_legend_handles_labels()
    figure.legend(
        handles_left + handles_right,
        labels_left + labels_right,
        loc="upper center",
        ncol=2,
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_interaction_heatmap(data: pd.DataFrame, output_path: str | Path) -> None:
    table = data.pivot(index="demand_scale", columns="capacity_scale", values="interaction")
    figure, axis = plt.subplots(figsize=(8, 6))
    maximum = float(np.nanmax(np.abs(table.to_numpy()))) or 1.0
    image = axis.imshow(
        table.to_numpy(),
        origin="lower",
        aspect="auto",
        cmap="coolwarm_r",
        vmin=-maximum,
        vmax=maximum,
    )
    axis.set_xticks(range(len(table.columns)), [f"{value:.2f}" for value in table.columns])
    axis.set_yticks(range(len(table.index)), [f"{value:.2f}" for value in table.index])
    axis.set_xlabel("Capacity scale")
    axis.set_ylabel("Demand scale")
    axis.set_title("Policy interaction (>0 complementarity, <0 substitution)")
    figure.colorbar(image, ax=axis, label="Difference-in-differences")
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_density_map(
    positions: pd.DataFrame,
    output_path: str | Path,
    minute: int | None = None,
    bins: int = 60,
) -> None:
    counts, x_edges, y_edges = density_grid(positions, minute=minute, bins=bins)
    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.pcolormesh(x_edges, y_edges, counts, cmap="magma", shading="auto")
    axis.set_xlabel("Longitude / projected x")
    axis.set_ylabel("Latitude / projected y")
    title = "Pedestrian density" if minute is None else f"Pedestrian density at minute {minute}"
    axis.set_title(title)
    figure.colorbar(image, ax=axis, label="Recorded agents per grid cell")
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_pareto_front(pareto: pd.DataFrame, output_path: str | Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 6))
    axis.scatter(
        pareto["predicted_intervention_cost"],
        pareto["predicted_congestion_score"],
        c=pareto["shuttle_awareness"],
        cmap="viridis",
        edgecolor="black",
        linewidth=0.3,
    )
    axis.set_xlabel("Predicted intervention cost")
    axis.set_ylabel("Predicted congestion score")
    axis.set_title("Surrogate-assisted NSGA-II Pareto front")
    axis.grid(True, linestyle="--", alpha=0.35)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_sensitivity(data: pd.DataFrame, output_path: str | Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 6))
    axis.plot(data["value"], data["congestion_score"], marker="o")
    axis.set_xlabel(str(data["parameter"].iloc[0]))
    axis.set_ylabel("Mean congestion score")
    axis.set_title("One-at-a-time sensitivity analysis")
    axis.grid(True, linestyle="--", alpha=0.35)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
