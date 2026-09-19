from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from .analysis import DESIGN_COLUMNS


class Predictor(Protocol):
    def predict(self, values: np.ndarray) -> np.ndarray: ...


class PolynomialRegressor:
    """Small deterministic quadratic surrogate used when XGBoost is unavailable."""

    def __init__(self, ridge: float = 1e-6):
        self.ridge = ridge
        self.coefficients: np.ndarray | None = None

    @staticmethod
    def expand(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        terms = [np.ones(len(values))]
        terms.extend(values[:, index] for index in range(values.shape[1]))
        terms.extend(values[:, index] ** 2 for index in range(values.shape[1]))
        for first in range(values.shape[1]):
            for second in range(first + 1, values.shape[1]):
                terms.append(values[:, first] * values[:, second])
        return np.column_stack(terms)

    def fit(self, values: np.ndarray, target: np.ndarray) -> "PolynomialRegressor":
        expanded = self.expand(values)
        penalty = self.ridge * np.eye(expanded.shape[1])
        penalty[0, 0] = 0.0
        self.coefficients = np.linalg.solve(
            expanded.T @ expanded + penalty,
            expanded.T @ np.asarray(target, dtype=float),
        )
        return self

    def predict(self, values: np.ndarray) -> np.ndarray:
        if self.coefficients is None:
            raise RuntimeError("surrogate has not been fitted")
        return self.expand(values) @ self.coefficients


@dataclass(slots=True)
class SurrogateSet:
    models: dict[str, Predictor]
    backend: str
    feature_columns: tuple[str, ...]

    def predict(self, designs: pd.DataFrame) -> pd.DataFrame:
        values = designs.loc[:, self.feature_columns].to_numpy(dtype=float)
        return pd.DataFrame(
            {target: model.predict(values) for target, model in self.models.items()},
            index=designs.index,
        )


def _xgboost_regressor(seed: int):
    from xgboost import XGBRegressor

    return XGBRegressor(
        n_estimators=250,
        max_depth=3,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=seed,
        n_jobs=1,
    )


def fit_surrogates(
    data: pd.DataFrame,
    targets: tuple[str, ...] = ("congestion_score", "intervention_cost"),
    backend: str = "auto",
    seed: int = 123,
) -> SurrogateSet:
    missing = sorted(set(DESIGN_COLUMNS + list(targets)) - set(data.columns))
    if missing:
        raise ValueError(f"training data is missing columns: {missing}")
    chosen = backend
    if backend == "auto":
        try:
            _xgboost_regressor(seed)
            chosen = "xgboost"
        except ImportError:
            chosen = "polynomial"
    if chosen not in {"xgboost", "polynomial"}:
        raise ValueError("backend must be auto, xgboost, or polynomial")

    values = data[DESIGN_COLUMNS].to_numpy(dtype=float)
    models: dict[str, Predictor] = {}
    for target in targets:
        model = _xgboost_regressor(seed) if chosen == "xgboost" else PolynomialRegressor()
        model.fit(values, data[target].to_numpy(dtype=float))
        models[target] = model
    return SurrogateSet(models, chosen, tuple(DESIGN_COLUMNS))


def surrogate_diagnostics(
    surrogate: SurrogateSet,
    data: pd.DataFrame,
    folds: int = 5,
    seed: int = 123,
) -> pd.DataFrame:
    predictions = surrogate.predict(data)
    fold_count = min(max(2, folds), len(data))
    rng = np.random.RandomState(seed)
    fold_ids = np.empty(len(data), dtype=int)
    fold_ids[rng.permutation(len(data))] = np.arange(len(data)) % fold_count
    cross_validated = {
        target: np.empty(len(data), dtype=float) for target in surrogate.models
    }
    for fold in range(fold_count):
        training = data.iloc[np.flatnonzero(fold_ids != fold)]
        held_out = data.iloc[np.flatnonzero(fold_ids == fold)]
        fitted = fit_surrogates(
            training,
            targets=tuple(surrogate.models),
            backend=surrogate.backend,
            seed=seed + fold,
        )
        held_predictions = fitted.predict(held_out)
        for target in surrogate.models:
            cross_validated[target][fold_ids == fold] = held_predictions[target].to_numpy()
    rows = []
    for target in surrogate.models:
        observed = data[target].to_numpy(dtype=float)
        predicted = predictions[target].to_numpy(dtype=float)
        residual = observed - predicted
        cv_residual = observed - cross_validated[target]
        denominator = float(np.sum((observed - observed.mean()) ** 2))
        r_squared = 1.0 - float(np.sum(residual**2)) / denominator if denominator else 1.0
        cv_r_squared = (
            1.0 - float(np.sum(cv_residual**2)) / denominator if denominator else 1.0
        )
        rows.append(
            {
                "target": target,
                "backend": surrogate.backend,
                "rmse_training": float(np.sqrt(np.mean(residual**2))),
                "mae_training": float(np.mean(np.abs(residual))),
                "r_squared_training": r_squared,
                "rmse_cross_validated": float(np.sqrt(np.mean(cv_residual**2))),
                "mae_cross_validated": float(np.mean(np.abs(cv_residual))),
                "r_squared_cross_validated": cv_r_squared,
                "folds": fold_count,
                "n_training": len(data),
                "validation_note": (
                    "random-fold cross-validation; retain an external test set for final claims"
                ),
            }
        )
    return pd.DataFrame(rows)
