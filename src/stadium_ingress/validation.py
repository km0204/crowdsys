from __future__ import annotations

import numpy as np
import pandas as pd


def nondominated_mask(objectives: np.ndarray) -> np.ndarray:
    """Return the nondominated rows for a minimisation problem."""
    values = np.asarray(objectives, dtype=float)
    if values.ndim != 2:
        raise ValueError("objectives must be a two-dimensional array")
    mask = np.ones(len(values), dtype=bool)
    for index, point in enumerate(values):
        if not mask[index]:
            continue
        dominated_by = np.all(values <= point, axis=1) & np.any(values < point, axis=1)
        mask[index] = not bool(np.any(dominated_by))
    return mask


def hypervolume_2d(objectives: np.ndarray, reference: np.ndarray) -> float:
    """Compute dominated hypervolume for two minimised objectives."""
    values = np.asarray(objectives, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or reference.shape != (2,):
        raise ValueError("hypervolume_2d requires n by 2 objectives and a length-two reference")
    values = values[np.all(values <= reference, axis=1)]
    if len(values) == 0:
        return 0.0
    front = values[nondominated_mask(values)]
    front = front[np.argsort(front[:, 0])]
    area = 0.0
    previous_y = float(reference[1])
    for x_value, y_value in front:
        if y_value < previous_y:
            area += float(reference[0] - x_value) * float(previous_y - y_value)
            previous_y = float(y_value)
    return area


def select_balanced_solution(pareto: pd.DataFrame) -> pd.Series:
    """Select the point nearest the ideal after min-max objective normalisation."""
    objective_columns = ["predicted_congestion_score", "predicted_intervention_cost"]
    if pareto.empty:
        raise ValueError("pareto set must not be empty")
    values = pareto[objective_columns].to_numpy(dtype=float)
    lower = values.min(axis=0)
    span = values.max(axis=0) - lower
    span[span == 0.0] = 1.0
    distances = np.linalg.norm((values - lower) / span, axis=1)
    return pareto.iloc[int(np.argmin(distances))]


def frontier_verification_metrics(verification: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Classify verified points and summarize frontier degradation."""
    required = {
        "predicted_congestion_score",
        "simulated_congestion_score",
        "predicted_intervention_cost",
        "simulated_intervention_cost",
    }
    missing = required - set(verification.columns)
    if missing:
        raise ValueError(f"verification data are missing columns: {sorted(missing)}")
    result = verification.copy()
    predicted = result[
        ["predicted_congestion_score", "predicted_intervention_cost"]
    ].to_numpy(dtype=float)
    simulated = result[
        ["simulated_congestion_score", "simulated_intervention_cost"]
    ].to_numpy(dtype=float)
    predicted_mask = nondominated_mask(predicted)
    simulated_mask = nondominated_mask(simulated)
    result["predicted_nondominated"] = predicted_mask
    result["simulated_nondominated"] = simulated_mask
    result["congestion_degradation"] = (
        result["simulated_congestion_score"] - result["predicted_congestion_score"]
    )
    denominator = result["predicted_congestion_score"].abs().clip(lower=1e-12)
    result["congestion_degradation_pct"] = result["congestion_degradation"] / denominator * 100.0

    combined = np.vstack([predicted, simulated])
    maxima = combined.max(axis=0)
    minima = combined.min(axis=0)
    reference = maxima + np.maximum(0.05 * (maxima - minima), 1e-9)
    predicted_hv = hypervolume_2d(predicted, reference)
    simulated_hv = hypervolume_2d(simulated, reference)
    predicted_count = int(predicted_mask.sum())
    retained_count = int(np.sum(predicted_mask & simulated_mask))
    metrics = {
        "verified_points": float(len(result)),
        "predicted_nondominated_points": float(predicted_count),
        "retained_nondominated_points": float(retained_count),
        "retained_nondominated_share": (
            float(retained_count / predicted_count) if predicted_count else float("nan")
        ),
        "predicted_hypervolume": predicted_hv,
        "simulated_hypervolume": simulated_hv,
        "hypervolume_degradation_pct": (
            float((predicted_hv - simulated_hv) / predicted_hv * 100.0)
            if predicted_hv > 0.0
            else float("nan")
        ),
        "mean_congestion_degradation_pct": float(
            result["congestion_degradation_pct"].mean()
        ),
        "mean_absolute_congestion_degradation_pct": float(
            result["congestion_degradation_pct"].abs().mean()
        ),
        "max_absolute_congestion_degradation_pct": float(
            result["congestion_degradation_pct"].abs().max()
        ),
        "reference_congestion": float(reference[0]),
        "reference_cost": float(reference[1]),
    }
    return result, metrics
