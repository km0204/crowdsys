from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def framework_figure(output: Path) -> None:
    fig, ax = plt.subplots(figsize=(14.5, 4.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    stages = [
        (
            "1. Policy\ndesign",
            "Seven decision variables\nLatin hypercube design\nDemand and capacity states",
        ),
        (
            "2. Agent based\nsimulation",
            "Synthetic population\nMode and activity choice\n"
            "Capacity constrained queues\nWaiting and arrival outputs",
        ),
        (
            "3. Interaction and\nsurrogate learning",
            "Matched interaction intervals\nXGBoost congestion model\n"
            "Exact resource index\nIndependent external test",
        ),
        (
            "4. Portfolio\nsearch",
            "Two objective NSGA II\nFrontier focused refinement\nFixed budget benchmark",
        ),
        (
            "5. Independent\nverification",
            "Twenty policy verification\nRepresentative portfolios\nScenario reoptimisation",
        ),
    ]
    x_positions = np.array([0.018, 0.214, 0.410, 0.606, 0.802])
    width = 0.18
    colors = ["#e8f1f8", "#eaf5ec", "#f4f0e8", "#f3eaf5", "#eef0f3"]
    for idx, ((title, body), x, color) in enumerate(
        zip(stages, x_positions, colors, strict=True)
    ):
        box = FancyBboxPatch(
            (x, 0.27),
            width,
            0.62,
            boxstyle="round,pad=0.012,rounding_size=0.012",
            linewidth=1.4,
            edgecolor="#30343b",
            facecolor=color,
        )
        ax.add_patch(box)
        ax.text(
            x + width / 2,
            0.79,
            title,
            ha="center",
            va="center",
            fontsize=11.0,
            fontweight="bold",
            color="#17202a",
        )
        ax.plot(
            [x + 0.015, x + width - 0.015],
            [0.705, 0.705],
            color="#7a828a",
            lw=0.8,
        )
        ax.text(
            x + 0.018,
            0.655,
            body,
            ha="left",
            va="top",
            fontsize=7.8,
            linespacing=1.5,
            color="#20262e",
        )
        if idx < len(stages) - 1:
            arrow = FancyArrowPatch(
                (x + width + 0.002, 0.57),
                (x_positions[idx + 1] - 0.002, 0.57),
                arrowstyle="-|>",
                mutation_scale=14,
                linewidth=1.5,
                color="#4f5964",
            )
            ax.add_patch(arrow)

    band = FancyBboxPatch(
        (0.12, 0.06),
        0.76,
        0.11,
        boxstyle="round,pad=0.012,rounding_size=0.012",
        linewidth=1.2,
        edgecolor="#36566f",
        facecolor="#f5f9fc",
    )
    ax.add_patch(band)
    ax.text(
        0.50,
        0.115,
        "Evidence boundaries: predictive accuracy versus optimisation quality"
        "   |   robustness versus optimality",
        ha="center",
        va="center",
        fontsize=9.4,
        fontweight="bold",
        color="#24465f",
    )
    _save(fig, output)


def _interaction_panel(
    ax: plt.Axes,
    summary_path: Path,
    title: str,
) -> matplotlib.image.AxesImage:
    data = pd.read_csv(summary_path)
    demands = sorted(data["demand_scale"].unique())
    capacities = sorted(data["capacity_scale"].unique())
    values = data.pivot(
        index="demand_scale", columns="capacity_scale", values="interaction"
    ).loc[demands, capacities]
    classes = data.pivot(
        index="demand_scale", columns="capacity_scale", values="classification"
    ).loc[demands, capacities]

    limit = float(np.abs(values.to_numpy()).max())
    image = ax.imshow(
        values.to_numpy(),
        cmap="RdBu",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        origin="lower",
        aspect="auto",
    )
    ax.set_xticks(range(len(capacities)), [f"{value:.2f}" for value in capacities])
    ax.set_yticks(range(len(demands)), [f"{value:.2f}" for value in demands])
    ax.set_xlabel("Capacity scale", fontsize=10.5)
    ax.set_ylabel("Demand scale", fontsize=10.5)
    ax.set_title(title, fontsize=12)
    ax.tick_params(labelsize=9.5)

    for row in range(len(demands)):
        for column in range(len(capacities)):
            value = values.iloc[row, column]
            classification = classes.iloc[row, column]
            label = (
                "C"
                if classification == "supported complementarity"
                else "S"
                if classification == "supported substitution"
                else "I"
            )
            text_color = "white" if abs(value) > limit * 0.48 else "black"
            ax.text(
                column,
                row,
                f"{value:.2f}\n{label}",
                ha="center",
                va="center",
                fontsize=8.8,
                fontweight="bold",
                color=text_color,
            )
    return image


def interaction_figure(
    summary_path: Path,
    dense_summary_path: Path,
    output: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.2))
    broad_image = _interaction_panel(axes[0], summary_path, "(a) Broad operating grid")
    dense_image = _interaction_panel(
        axes[1], dense_summary_path, "(b) Boundary focused grid"
    )
    for ax, image in zip(axes, (broad_image, dense_image), strict=True):
        colorbar = fig.colorbar(image, ax=ax, fraction=0.047, pad=0.035)
        colorbar.ax.tick_params(labelsize=8.5)
    fig.supxlabel(
        "C: interval above zero     S: interval below zero     I: interval includes zero",
        fontsize=9.5,
        color="#3f4851",
        y=0.01,
    )
    fig.subplots_adjust(bottom=0.16, wspace=0.28)
    _save(fig, output)


def frontier_figure(
    pareto_path: Path,
    verification_path: Path,
    representatives_path: Path,
    output: Path,
) -> None:
    pareto = pd.read_csv(pareto_path).sort_values("predicted_intervention_cost")
    verification = pd.read_csv(verification_path)
    representatives = pd.read_csv(representatives_path)

    fig, ax = plt.subplots(figsize=(9.1, 5.9))
    ax.plot(
        pareto["predicted_intervention_cost"],
        pareto["predicted_congestion_score"],
        color="#395f8c",
        linewidth=1.4,
        alpha=0.80,
        label="Predicted Pareto frontier",
        zorder=1,
    )
    ax.scatter(
        pareto["predicted_intervention_cost"],
        pareto["predicted_congestion_score"],
        s=18,
        color="#6e91b8",
        alpha=0.65,
        edgecolor="none",
        zorder=2,
    )

    retained = verification["simulated_nondominated"].astype(bool)
    ax.scatter(
        verification.loc[retained, "simulated_intervention_cost"],
        verification.loc[retained, "simulated_congestion_score"],
        s=62,
        marker="o",
        facecolor="#2a9d70",
        edgecolor="white",
        linewidth=0.8,
        label="Verified and nondominated",
        zorder=4,
    )
    ax.scatter(
        verification.loc[~retained, "simulated_intervention_cost"],
        verification.loc[~retained, "simulated_congestion_score"],
        s=62,
        marker="o",
        facecolor="#d9745b",
        edgecolor="white",
        linewidth=0.8,
        label="Verified and dominated",
        zorder=4,
    )
    for _, row in verification.iterrows():
        ax.plot(
            [row["predicted_intervention_cost"], row["simulated_intervention_cost"]],
            [row["predicted_congestion_score"], row["simulated_congestion_score"]],
            color="#9aa0a6",
            linewidth=0.65,
            alpha=0.55,
            zorder=0,
        )

    labels = {
        "minimum congestion": ("Minimum congestion", (-116, 14)),
        "minimum resource": ("Minimum resource", (12, -25)),
        "balanced compromise": ("Balanced compromise", (12, 12)),
    }
    for representative_index, (_, row) in enumerate(representatives.iterrows()):
        text, offset = labels[row["policy"]]
        ax.scatter(
            row["predicted_resource_index"],
            row["simulated_congestion_score"],
            s=170,
            marker="*",
            facecolor="#f2b134",
            edgecolor="#4b3b16",
            linewidth=0.9,
            label="Representative policy" if representative_index == 0 else None,
            zorder=6,
        )
        ax.annotate(
            text,
            (row["predicted_resource_index"], row["simulated_congestion_score"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=9.5,
            arrowprops={"arrowstyle": "-", "color": "#6c7074", "lw": 0.8},
        )

    ax.set_xlabel("Operational resource index", fontsize=12)
    ax.set_ylabel("Congestion score", fontsize=12)
    ax.set_title(
        "Predicted frontier and independent ABS verification", fontsize=14, pad=12
    )
    ax.grid(True, color="#d8dadd", linewidth=0.7, alpha=0.75)
    ax.tick_params(labelsize=10.5)
    ax.legend(frameon=True, fontsize=8.8, loc="upper right")
    _save(fig, output)


def temporal_profile_figure(summary_path: Path, output: Path) -> None:
    data = pd.read_csv(summary_path)
    colors = {"Baseline": "#3b6f9c", "Balanced compromise": "#d97932"}
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), sharex=True)
    panels = [
        ("arrivals_count", "Arrivals per minute", "(a) Arrival profile"),
        ("total_queue_length", "Spectators", "(b) Total queue length"),
    ]
    for ax, (metric, ylabel, title) in zip(axes, panels, strict=True):
        for policy in ("Baseline", "Balanced compromise"):
            frame = data.loc[data["policy"] == policy].sort_values(
                "minute_before_kickoff"
            )
            mean = frame[f"{metric}_mean"].to_numpy()
            sd = frame[f"{metric}_sd"].to_numpy()
            replications = frame["replications"].to_numpy()
            half_width = 1.96 * sd / np.sqrt(replications)
            minutes = frame["minute_before_kickoff"].to_numpy()
            ax.plot(
                minutes,
                mean,
                color=colors[policy],
                linewidth=1.8,
                label=policy,
            )
            ax.fill_between(
                minutes,
                mean - half_width,
                mean + half_width,
                color=colors[policy],
                alpha=0.16,
                linewidth=0,
            )
        ax.axvline(0, color="#4f5358", linewidth=0.9, linestyle=":")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Minutes relative to kickoff", fontsize=10.5)
        ax.set_ylabel(ylabel, fontsize=10.5)
        ax.grid(True, color="#d8dadd", linewidth=0.7, alpha=0.72)
        ax.tick_params(labelsize=9.5)
        ax.set_xlim(data["minute_before_kickoff"].min(), 30)
    axes[0].legend(frameon=True, fontsize=9, loc="upper left")
    _save(fig, output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate final manuscript figures.")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    interaction = args.results_dir / "interaction"
    portfolio = args.results_dir / "portfolio_refined"
    framework_figure(args.output_dir / "fig1_reproducible.png")
    interaction_figure(
        interaction / "interaction_summary.csv",
        args.results_dir / "interaction_dense" / "interaction_summary.csv",
        args.output_dir / "fig7_reproducible.png",
    )
    frontier_figure(
        portfolio / "pareto_solutions.csv",
        portfolio / "frontier_verification.csv",
        portfolio / "representative_policies.csv",
        args.output_dir / "fig10_reproducible.png",
    )
    temporal_profile_figure(
        args.results_dir / "temporal_profile" / "temporal_profile_summary.csv",
        args.output_dir / "fig4_reproducible.png",
    )


if __name__ == "__main__":
    main()
