from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .analysis import DESIGN_COLUMNS, config_from_design, result_kpis
from .config import ModelConfig
from .model import simulate
from .optimisation import resource_index_from_designs

STATE_COLUMNS = ["demand_scale", "capacity_scale"]
CONTEXT_COLUMNS = DESIGN_COLUMNS + STATE_COLUMNS
FACTORIAL_TARGETS = ("y00", "y10", "y01", "y11")
OPTIMISATION_TARGET = "g11"
MODEL_TARGETS = (*FACTORIAL_TARGETS, OPTIMISATION_TARGET)
TARGET_FEATURES = {
    "y00": STATE_COLUMNS,
    "y10": [
        "shuttle_headway_minutes",
        "fleet_size",
        "shuttle_vehicle_capacity",
        "shuttle_awareness",
        *STATE_COLUMNS,
    ],
    "y01": [
        "event_target_join_rate",
        "event_dwell_mean_minutes",
        "event_awareness",
        *STATE_COLUMNS,
    ],
    "y11": CONTEXT_COLUMNS,
    OPTIMISATION_TARGET: CONTEXT_COLUMNS,
}


def sample_policy_states(
    n_samples: int,
    seed: int,
    demand_bounds: tuple[float, float] = (0.80, 1.30),
    capacity_bounds: tuple[float, float] = (0.51, 1.02),
) -> pd.DataFrame:
    """Generate a reproducible nine dimensional Latin hypercube design."""
    if n_samples < 2:
        raise ValueError("n_samples must be at least two")
    rng = np.random.RandomState(seed)
    unit = np.empty((n_samples, len(CONTEXT_COLUMNS)))
    for column in range(unit.shape[1]):
        unit[:, column] = (rng.permutation(n_samples) + rng.random(n_samples)) / n_samples
    frame = pd.DataFrame(
        {
            "shuttle_headway_minutes": np.rint(6 + unit[:, 0] * 9).astype(int),
            "fleet_size": np.rint(3 + unit[:, 1] * 6).astype(int),
            "shuttle_vehicle_capacity": np.rint(60 + unit[:, 2] * 40).astype(int),
            "event_target_join_rate": 0.30 + unit[:, 3] * 0.50,
            "event_dwell_mean_minutes": np.rint(10 + unit[:, 4] * 20).astype(int),
            "shuttle_awareness": 0.25 + unit[:, 5] * 0.75,
            "event_awareness": 0.25 + unit[:, 6] * 0.75,
            "demand_scale": demand_bounds[0]
            + unit[:, 7] * (demand_bounds[1] - demand_bounds[0]),
            "capacity_scale": capacity_bounds[0]
            + unit[:, 8] * (capacity_bounds[1] - capacity_bounds[0]),
        }
    )
    return frame.loc[:, CONTEXT_COLUMNS]


def policy_state_configs(
    base: ModelConfig,
    design: pd.Series | dict[str, float],
) -> dict[str, ModelConfig]:
    """Construct four matched counterfactual configurations for one policy state."""
    state = replace(
        base,
        n_agents=max(1, int(round(base.n_agents * float(design["demand_scale"])))),
        capacity_scale=float(design["capacity_scale"]),
    )
    joint = config_from_design(state, design)
    return {
        "y00": joint.with_policy(
            shuttle_enabled=False,
            event_enabled=False,
            shuttle_awareness=0.0,
            event_awareness=0.0,
        ),
        "y10": joint.with_policy(
            shuttle_enabled=True,
            event_enabled=False,
            shuttle_awareness=float(design["shuttle_awareness"]),
            event_awareness=0.0,
        ),
        "y01": joint.with_policy(
            shuttle_enabled=False,
            event_enabled=True,
            shuttle_awareness=0.0,
            event_awareness=float(design["event_awareness"]),
        ),
        "y11": joint,
    }


def evaluate_policy_state(
    base: ModelConfig,
    design: pd.Series | dict[str, float],
    seeds: tuple[int, ...],
    interaction_metric: str = "mean_queue_wait_minutes",
    optimisation_metric: str = "congestion_score",
) -> tuple[dict[str, float], pd.DataFrame]:
    """Evaluate matched counterfactuals and return means plus seed level values."""
    if not seeds:
        raise ValueError("at least one seed is required")
    configurations = policy_state_configs(base, design)
    rows: list[dict[str, float]] = []
    for seed in seeds:
        row: dict[str, float] = {"seed": float(seed)}
        for target, config in configurations.items():
            values = result_kpis(simulate(config, seed))
            row[target] = float(values[interaction_metric])
            if target == "y11":
                row[OPTIMISATION_TARGET] = float(values[optimisation_metric])
        row["gamma"] = row["y10"] + row["y01"] - row["y11"] - row["y00"]
        rows.append(row)
    seed_level = pd.DataFrame(rows)
    summary = {target: float(seed_level[target].mean()) for target in FACTORIAL_TARGETS}
    summary[OPTIMISATION_TARGET] = float(seed_level[OPTIMISATION_TARGET].mean())
    summary["gamma"] = float(seed_level["gamma"].mean())
    summary["gamma_sd"] = float(
        seed_level["gamma"].std(ddof=1) if len(seed_level) > 1 else 0.0
    )
    summary["gamma_se"] = summary["gamma_sd"] / np.sqrt(len(seed_level))
    summary["replications"] = float(len(seed_level))
    return summary, seed_level


def _xgboost(seed: int):
    from xgboost import XGBRegressor

    return XGBRegressor(
        n_estimators=120,
        max_depth=3,
        learning_rate=0.05,
        min_child_weight=1,
        subsample=0.85,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=seed,
        n_jobs=1,
    )


@dataclass(slots=True)
class ContextualEnsemble:
    models: dict[str, list]

    @property
    def members(self) -> int:
        return len(next(iter(self.models.values())))

    def predict_members(self, frame: pd.DataFrame, target: str) -> np.ndarray:
        features = TARGET_FEATURES[target]
        values = frame.loc[:, features].to_numpy(dtype=float)
        return np.vstack([model.predict(values) for model in self.models[target]])

    def factorial_predictions(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        return {
            target: self.predict_members(frame, target) for target in FACTORIAL_TARGETS
        }

    def gamma_members(self, frame: pd.DataFrame) -> np.ndarray:
        predictions = self.factorial_predictions(frame)
        return (
            predictions["y10"]
            + predictions["y01"]
            - predictions["y11"]
            - predictions["y00"]
        )


def fit_contextual_ensemble(
    data: pd.DataFrame,
    members: int = 6,
    seed: int = 123,
) -> ContextualEnsemble:
    """Fit paired bootstrap ensembles for interaction and optimisation responses."""
    missing = sorted(set(CONTEXT_COLUMNS + list(MODEL_TARGETS)) - set(data.columns))
    if missing:
        raise ValueError(f"training data is missing columns: {missing}")
    if members < 2:
        raise ValueError("members must be at least two")
    rng = np.random.RandomState(seed)
    indices = [rng.randint(0, len(data), size=len(data)) for _ in range(members)]
    models: dict[str, list] = {target: [] for target in MODEL_TARGETS}
    for member, sampled in enumerate(indices):
        for target in MODEL_TARGETS:
            features = TARGET_FEATURES[target]
            model = _xgboost(seed + 100 * member + MODEL_TARGETS.index(target))
            model.fit(
                data.iloc[sampled][features].to_numpy(dtype=float),
                data.iloc[sampled][target].to_numpy(dtype=float),
            )
            models[target].append(model)
    return ContextualEnsemble(models)


def acquisition_pool(
    policy_count: int,
    states: tuple[tuple[float, float], ...],
    seed: int,
) -> pd.DataFrame:
    policies = sample_policy_states(policy_count, seed).drop(columns=STATE_COLUMNS)
    rows: list[pd.DataFrame] = []
    for state_id, (demand, capacity) in enumerate(states):
        frame = policies.copy()
        frame["demand_scale"] = demand
        frame["capacity_scale"] = capacity
        frame["state_id"] = state_id
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def acquisition_scores(
    ensemble: ContextualEnsemble,
    pool: pd.DataFrame,
) -> pd.DataFrame:
    """Compute interaction margin and ensemble Pareto ranking uncertainty."""
    result = pool.copy()
    gamma = ensemble.gamma_members(pool)
    gamma_mean = gamma.mean(axis=0)
    gamma_sd = gamma.std(axis=0, ddof=1)
    result["gamma_mean"] = gamma_mean
    result["gamma_sd"] = gamma_sd
    result["interaction_margin"] = np.abs(gamma_mean) / (gamma_sd + 1e-6)

    joint = ensemble.predict_members(pool, OPTIMISATION_TARGET)
    result["joint_mean"] = joint.mean(axis=0)
    result["joint_sd"] = joint.std(axis=0, ddof=1)
    resources = resource_index_from_designs(pool)
    probability = np.zeros(len(pool), dtype=float)
    for _, state_indices in result.groupby("state_id", sort=True).groups.items():
        indices = np.asarray(list(state_indices), dtype=int)
        member_masks = []
        for member in range(ensemble.members):
            objectives = np.column_stack([joint[member, indices], resources[indices]])
            mask = np.ones(len(indices), dtype=bool)
            for position, point in enumerate(objectives):
                dominated = np.all(objectives <= point, axis=1) & np.any(
                    objectives < point, axis=1
                )
                mask[position] = not bool(np.any(dominated))
            member_masks.append(mask)
        probability[indices] = np.mean(member_masks, axis=0)
    result["nondominated_probability"] = probability
    result["pareto_uncertainty"] = probability * (1.0 - probability)
    result["relative_joint_uncertainty"] = result["joint_sd"] / (
        result["joint_mean"].abs() + 1e-6
    )
    return result


def _normalized_distance(left: pd.Series, right: pd.Series) -> float:
    spans = np.array([9, 6, 40, 0.5, 20, 0.75, 0.75, 0.5, 0.51], dtype=float)
    delta = (
        left.loc[CONTEXT_COLUMNS].to_numpy(dtype=float)
        - right.loc[CONTEXT_COLUMNS].to_numpy(dtype=float)
    ) / spans
    return float(np.linalg.norm(delta))


def select_adaptive_batch(
    scored: pd.DataFrame,
    existing: pd.DataFrame,
    batch_size: int,
    mode: str,
) -> pd.DataFrame:
    """Select a diverse batch using Pareto uncertainty and interaction margin."""
    if mode not in {"pareto", "interaction"}:
        raise ValueError("mode must be pareto or interaction")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    selected: list[pd.Series] = []

    def add_from(order: pd.DataFrame, count: int) -> None:
        for _, candidate in order.iterrows():
            comparisons = [*selected]
            if not existing.empty:
                nearest_existing = existing.iloc[
                    np.argmin(
                        [
                            _normalized_distance(candidate, row)
                            for _, row in existing.iterrows()
                        ]
                    )
                ]
                comparisons.append(nearest_existing)
            if comparisons and min(
                _normalized_distance(candidate, row) for row in comparisons
            ) < 0.035:
                continue
            selected.append(candidate)
            if len(selected) >= count:
                return

    pareto_order = scored.sort_values(
        ["pareto_uncertainty", "relative_joint_uncertainty"],
        ascending=[False, False],
    )
    if mode == "pareto":
        add_from(pareto_order, batch_size)
    else:
        interaction_count = batch_size // 2
        interaction_order = scored.sort_values(
            ["interaction_margin", "gamma_sd"],
            ascending=[True, False],
        )
        add_from(interaction_order, interaction_count)
        add_from(pareto_order, batch_size)
    if len(selected) < batch_size:
        fallback = scored.sort_values("relative_joint_uncertainty", ascending=False)
        add_from(fallback, batch_size)
    return pd.DataFrame(selected).loc[:, CONTEXT_COLUMNS].reset_index(drop=True)

