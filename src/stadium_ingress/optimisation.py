from __future__ import annotations

import numpy as np
import pandas as pd

from .analysis import DESIGN_COLUMNS
from .surrogate import SurrogateSet

LOWER = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
UPPER = np.array([1.0, 1.0, 20.0, 0.70, 1.0])


def _decode(population: np.ndarray) -> pd.DataFrame:
    decoded = LOWER + np.clip(population, 0.0, 1.0) * (UPPER - LOWER)
    decoded[:, 2] = np.rint(decoded[:, 2])
    return pd.DataFrame(decoded, columns=DESIGN_COLUMNS)


def _dominates(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(np.all(left <= right) and np.any(left < right))


def _fronts(objectives: np.ndarray) -> tuple[list[list[int]], np.ndarray]:
    dominates: list[list[int]] = [[] for _ in range(len(objectives))]
    domination_count = np.zeros(len(objectives), dtype=int)
    ranks = np.zeros(len(objectives), dtype=int)
    first: list[int] = []
    for p in range(len(objectives)):
        for q in range(len(objectives)):
            if p == q:
                continue
            if _dominates(objectives[p], objectives[q]):
                dominates[p].append(q)
            elif _dominates(objectives[q], objectives[p]):
                domination_count[p] += 1
        if domination_count[p] == 0:
            first.append(p)
    fronts = [first]
    rank = 0
    while fronts[rank]:
        following: list[int] = []
        for p in fronts[rank]:
            ranks[p] = rank
            for q in dominates[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    following.append(q)
        rank += 1
        fronts.append(following)
    return fronts[:-1], ranks


def _crowding(objectives: np.ndarray, front: list[int]) -> dict[int, float]:
    distance = {index: 0.0 for index in front}
    if len(front) <= 2:
        return {index: float("inf") for index in front}
    for objective in range(objectives.shape[1]):
        ordered = sorted(front, key=lambda index: objectives[index, objective])
        distance[ordered[0]] = distance[ordered[-1]] = float("inf")
        low = objectives[ordered[0], objective]
        high = objectives[ordered[-1], objective]
        if high <= low:
            continue
        for position in range(1, len(ordered) - 1):
            distance[ordered[position]] += (
                objectives[ordered[position + 1], objective]
                - objectives[ordered[position - 1], objective]
            ) / (high - low)
    return distance


def _rank_and_crowding(objectives: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fronts, ranks = _fronts(objectives)
    crowding = np.zeros(len(objectives))
    for front in fronts:
        for index, value in _crowding(objectives, front).items():
            crowding[index] = value
    return ranks, crowding


def _tournament_winner(
    first: int,
    second: int,
    ranks: np.ndarray,
    crowding: np.ndarray,
) -> int:
    if ranks[first] != ranks[second]:
        return first if ranks[first] < ranks[second] else second
    return first if crowding[first] >= crowding[second] else second


def _select(
    population: np.ndarray,
    objectives: np.ndarray,
    size: int,
) -> tuple[np.ndarray, np.ndarray]:
    fronts, _ = _fronts(objectives)
    selected: list[int] = []
    for front in fronts:
        if len(selected) + len(front) <= size:
            selected.extend(front)
            continue
        crowding = _crowding(objectives, front)
        ordered = sorted(front, key=lambda index: crowding[index], reverse=True)
        selected.extend(ordered[: size - len(selected)])
        break
    return population[selected], objectives[selected]


def nsga2(
    surrogate: SurrogateSet,
    population_size: int = 80,
    generations: int = 100,
    seed: int = 321,
) -> pd.DataFrame:
    """Run a compact, dependency-free NSGA-II on the fitted surrogate models."""
    if population_size < 4 or generations < 1:
        raise ValueError("population_size must be at least 4 and generations positive")
    rng = np.random.RandomState(seed)
    population = rng.random((population_size, len(DESIGN_COLUMNS)))

    def evaluate(values: np.ndarray) -> np.ndarray:
        predicted = surrogate.predict(_decode(values))
        return predicted[["congestion_score", "intervention_cost"]].to_numpy(dtype=float)

    objectives = evaluate(population)
    for _ in range(generations):
        ranks, crowding = _rank_and_crowding(objectives)
        offspring: list[np.ndarray] = []
        while len(offspring) < population_size:
            contestants = rng.randint(0, population_size, size=4)
            parent_a = population[
                _tournament_winner(contestants[0], contestants[1], ranks, crowding)
            ]
            parent_b = population[
                _tournament_winner(contestants[2], contestants[3], ranks, crowding)
            ]
            mix = rng.random(len(DESIGN_COLUMNS))
            child_a = mix * parent_a + (1.0 - mix) * parent_b
            child_b = mix * parent_b + (1.0 - mix) * parent_a
            for child in (child_a, child_b):
                mutation = rng.random(len(DESIGN_COLUMNS)) < (1.0 / len(DESIGN_COLUMNS))
                child[mutation] += rng.normal(0.0, 0.08, size=int(mutation.sum()))
                offspring.append(np.clip(child, 0.0, 1.0))
                if len(offspring) == population_size:
                    break
        offspring_array = np.asarray(offspring)
        combined = np.vstack([population, offspring_array])
        combined_objectives = np.vstack([objectives, evaluate(offspring_array)])
        population, objectives = _select(combined, combined_objectives, population_size)

    fronts, _ = _fronts(objectives)
    pareto_indices = fronts[0]
    pareto = _decode(population[pareto_indices])
    pareto["predicted_congestion_score"] = objectives[pareto_indices, 0]
    pareto["predicted_intervention_cost"] = objectives[pareto_indices, 1]
    pareto = pareto.drop_duplicates().sort_values(
        ["predicted_congestion_score", "predicted_intervention_cost"]
    )
    pareto.insert(0, "pareto_id", np.arange(1, len(pareto) + 1))
    return pareto.reset_index(drop=True)
