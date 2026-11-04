from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch, Rectangle


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def framework_figure(output: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.4, 5.1))
    ax.set_xlim(0, 12.4)
    ax.set_ylim(0, 5.1)
    ax.axis("off")

    navy = "#244b6b"
    ink = "#20262d"
    mid = "#6c7780"
    search_fill = "#f4f7f9"
    verify_fill = "#f6f4ef"

    ax.text(
        0.35,
        4.82,
        "Sequential design and surrogate search",
        fontsize=11.5,
        fontweight="bold",
        color=ink,
        va="center",
    )
    ax.plot([0.35, 8.85], [4.62, 4.62], color=navy, lw=1.5)
    ax.text(
        9.25,
        4.82,
        "Independent evaluation",
        fontsize=11.5,
        fontweight="bold",
        color=ink,
        va="center",
    )
    ax.plot([9.25, 12.05], [4.62, 4.62], color=navy, lw=1.5)

    def box(
        x: float,
        y: float,
        width: float,
        height: float,
        number: str,
        title: str,
        body: str,
        fill: str = search_fill,
    ) -> None:
        ax.add_patch(
            Rectangle(
                (x, y),
                width,
                height,
                facecolor=fill,
                edgecolor=ink,
                linewidth=1.05,
            )
        )
        ax.text(
            x + 0.14,
            y + height - 0.18,
            number,
            fontsize=9.0,
            fontweight="bold",
            color=navy,
            va="top",
        )
        ax.text(
            x + 0.42,
            y + height - 0.18,
            title,
            fontsize=8.8,
            fontweight="bold",
            color=ink,
            va="top",
        )
        ax.plot(
            [x + 0.14, x + width - 0.14],
            [y + height - 0.55, y + height - 0.55],
            color="#a4abb1",
            lw=0.65,
        )
        ax.text(
            x + 0.16,
            y + height - 0.72,
            body,
            fontsize=7.7,
            color=ink,
            va="top",
            linespacing=1.38,
        )

    box(
        0.35,
        2.62,
        1.92,
        1.66,
        "1",
        "Initial design",
        "Policy  $\\mathbf{x}$: seven controls\nState  $\\mathbf{z}=(D,K)$\nLatin hypercube sample",
    )
    box(
        2.58,
        2.62,
        1.92,
        1.66,
        "2",
        "Matched ABS",
        "Common seeds\n$Y_{00},Y_{10},Y_{01},Y_{11}$\nJoint congestion  $G_{11}$",
    )
    box(
        4.81,
        2.62,
        1.92,
        1.66,
        "3",
        "Surrogate models",
        "$\\widehat{Y}_{ab}(\\mathbf{x},\\mathbf{z})$ and $\\widehat{G}_{11}$\n"
        "$\\overline{\\Gamma}_Y$ and $s_{\\Gamma}$\n"
        "Nondominance probability $p_{ND}$",
    )
    box(
        7.04,
        2.62,
        1.92,
        1.66,
        "4",
        "Sequential sampling",
        "First batch:  $2A_I+3A_P$\nLater batches:  $A_P$\nRefit until budget is reached",
    )

    for start, end in ((2.27, 2.58), (4.50, 4.81), (6.73, 7.04)):
        ax.add_patch(
            FancyArrowPatch(
                (start, 3.45),
                (end, 3.45),
                arrowstyle="-|>",
                mutation_scale=11,
                linewidth=1.15,
                color=navy,
            )
        )

    ax.add_patch(
        FancyArrowPatch(
            (8.00, 2.59),
            (3.54, 2.59),
            connectionstyle="arc3,rad=0.31",
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.15,
            color=navy,
        )
    )
    ax.text(
        5.78,
        1.98,
        "new policy and state evaluations",
        fontsize=7.5,
        color=navy,
        ha="center",
    )

    box(
        4.81,
        0.35,
        4.15,
        1.12,
        "5",
        "Multiobjective portfolio search",
        "Mean predicted congestion across three states\n"
        "Exact resource index $C(\\mathbf{x})$; NSGA II search",
    )
    ax.add_patch(
        FancyArrowPatch(
            (8.00, 2.61),
            (7.78, 1.47),
            connectionstyle="arc3,rad=-0.18",
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.15,
            color=navy,
        )
    )
    ax.text(8.22, 1.82, "fixed budget", fontsize=7.3, color=mid, ha="center")

    box(
        9.25,
        2.62,
        2.80,
        1.66,
        "A",
        "Interaction assessment",
        "Held out demand and capacity grid\nMatched confidence intervals\nRMSE and sign classification",
        fill=verify_fill,
    )
    box(
        9.25,
        0.35,
        2.80,
        1.66,
        "B",
        "Portfolio assessment",
        "Independent ABS replications\n"
        "Retained nondominance and hypervolume\n"
        "Scenario specific reoptimisation",
        fill=verify_fill,
    )
    for start, end in (((6.73, 3.14), (9.25, 3.14)), ((8.96, 0.91), (9.25, 0.91))):
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=11,
                linewidth=1.0,
                linestyle=(0, (4, 3)),
                color=mid,
            )
        )
    ax.text(
        10.62,
        2.29,
        "Evaluation data are excluded from fitting and acquisition",
        fontsize=7.3,
        color=mid,
        ha="center",
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


def adaptive_benchmark_figure(
    learning_path: Path,
    summary_path: Path,
    output: Path,
) -> None:
    learning = pd.read_csv(learning_path)
    summary = pd.read_csv(summary_path).set_index("method")
    labels = {
        "offline": "Offline design",
        "pareto_adaptive": "Pareto adaptive",
        "staged_adaptive": "Staged interaction aware",
    }
    colors = {
        "offline": "#68737d",
        "pareto_adaptive": "#d07a2d",
        "staged_adaptive": "#2d7c72",
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3))
    for method in labels:
        frame = learning.loc[learning["method"] == method]
        grouped = frame.groupby("search_abs_runs")["dense_gamma_rmse"]
        mean = grouped.mean()
        sd = grouped.std()
        axes[0].plot(
            mean.index,
            mean.to_numpy(),
            marker="o",
            linewidth=1.8,
            markersize=4.8,
            color=colors[method],
            label=labels[method],
        )
        axes[0].fill_between(
            mean.index,
            mean.to_numpy() - sd.to_numpy(),
            mean.to_numpy() + sd.to_numpy(),
            color=colors[method],
            alpha=0.14,
            linewidth=0,
        )
    axes[0].set_xlabel("Search ABS runs", fontsize=10.5)
    axes[0].set_ylabel(r"Held out $\Gamma_Y$ RMSE (min)", fontsize=10.5)
    axes[0].set_title("(a) Interaction learning", fontsize=12)
    axes[0].grid(True, color="#d8dadd", linewidth=0.7, alpha=0.72)
    axes[0].legend(frameon=True, fontsize=8.5)

    order = ["offline", "staged_adaptive", "pareto_adaptive"]
    x = np.arange(len(order))
    values = summary.loc[order, "hypervolume_mean"].to_numpy()
    errors = summary.loc[order, "hypervolume_sd"].to_numpy()
    axes[1].bar(
        x,
        values,
        yerr=errors,
        capsize=4,
        color=[colors[method] for method in order],
        edgecolor="white",
        linewidth=0.8,
    )
    axes[1].set_xticks(x, [labels[method] for method in order], rotation=15, ha="right")
    axes[1].set_ylabel("Verified hypervolume", fontsize=10.5)
    axes[1].set_title("(b) Portfolio quality", fontsize=12)
    axes[1].grid(True, axis="y", color="#d8dadd", linewidth=0.7, alpha=0.72)
    axes[1].set_ylim(0, max(values + errors) * 1.18)
    for position, value in zip(x, values, strict=True):
        axes[1].text(position, value + 1.2, f"{value:.1f}", ha="center", fontsize=9)
    for ax in axes:
        ax.tick_params(labelsize=9.2)
    fig.subplots_adjust(wspace=0.30, bottom=0.20)
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
    adaptive_benchmark_figure(
        _first_existing(
            args.results_dir
            / "staged_adaptive"
            / "combined_dense_grid_learning_curve.csv",
            args.results_dir / "staged_adaptive" / "dense_grid_learning_curve.csv",
        ),
        _first_existing(
            args.results_dir / "staged_adaptive" / "combined_method_summary.csv",
            args.results_dir / "staged_adaptive" / "method_summary.csv",
        ),
        args.output_dir / "fig12_interaction_aware.png",
    )


if __name__ == "__main__":
    main()
