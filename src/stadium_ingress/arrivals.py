from __future__ import annotations

import math

import numpy as np

from .config import ArrivalConfig


def sample_desired_arrival(config: ArrivalConfig, rng: np.random.RandomState) -> float:
    """Sample a desired stadium-arrival time from the published S0 distribution."""
    triangular_area = 0.5 * (config.left_end - config.left_start)
    plateau_area = config.plateau_end - config.plateau_start
    late_area = (1.0 - math.exp(-config.decay_rate * config.late_max)) / config.decay_rate
    probabilities = np.asarray([triangular_area, plateau_area, late_area], dtype=float)
    probabilities /= probabilities.sum()

    segment = int(rng.choice(3, p=probabilities))
    uniform = float(rng.random())
    if segment == 0:
        return float(
            config.left_start
            + (config.left_end - config.left_start) * math.sqrt(uniform)
        )
    if segment == 1:
        return float(
            config.plateau_start
            + (config.plateau_end - config.plateau_start) * uniform
        )

    late = -math.log(
        1.0 - uniform * (1.0 - math.exp(-config.decay_rate * config.late_max))
    ) / config.decay_rate
    return float(late)
