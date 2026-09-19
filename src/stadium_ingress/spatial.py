from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class Route:
    coordinates: tuple[tuple[float, float], ...]
    cumulative_metres: tuple[float, ...]
    length_metres: float

    def interpolate(self, fraction: float) -> tuple[float, float]:
        fraction = min(1.0, max(0.0, fraction))
        target = fraction * self.length_metres
        index = int(np.searchsorted(self.cumulative_metres, target, side="right") - 1)
        index = min(max(index, 0), len(self.coordinates) - 2)
        d0, d1 = self.cumulative_metres[index : index + 2]
        weight = 0.0 if d1 <= d0 else (target - d0) / (d1 - d0)
        x0, y0 = self.coordinates[index]
        x1, y1 = self.coordinates[index + 1]
        return x0 + (x1 - x0) * weight, y0 + (y1 - y0) * weight


@dataclass(frozen=True, slots=True)
class SpatialInputs:
    walk_times: dict[str, float] = field(default_factory=dict)
    direct: dict[str, Route] = field(default_factory=dict)
    station_to_zone: dict[str, dict[str, Route]] = field(default_factory=dict)
    zone_to_gate: dict[str, dict[str, Route]] = field(default_factory=dict)
    zones: dict[str, tuple[float, float]] = field(default_factory=dict)
    station_to_zone_minutes: dict[str, dict[str, float]] = field(default_factory=dict)
    zone_to_gate_minutes: dict[str, dict[str, float]] = field(default_factory=dict)
    source_directory: str | None = None

    @property
    def paths_available(self) -> bool:
        return bool(self.direct)


def _segment_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    x0, y0 = a
    x1, y1 = b
    if all(abs(v) <= 180 for v in (x0, x1)) and all(abs(v) <= 90 for v in (y0, y1)):
        mean_lat = math.radians((y0 + y1) / 2.0)
        dx = math.radians(x1 - x0) * math.cos(mean_lat) * 6_371_000
        dy = math.radians(y1 - y0) * 6_371_000
        return math.hypot(dx, dy)
    return math.hypot(x1 - x0, y1 - y0)


def build_route(raw: Any) -> Route | None:
    if isinstance(raw, dict):
        raw = raw.get("coords", raw.get("path"))
    if not isinstance(raw, list) or len(raw) < 2:
        return None
    coordinates = tuple((float(point[0]), float(point[1])) for point in raw)
    cumulative = [0.0]
    for start, end in zip(coordinates[:-1], coordinates[1:], strict=True):
        cumulative.append(cumulative[-1] + _segment_distance(start, end))
    if cumulative[-1] <= 0:
        return None
    return Route(coordinates, tuple(cumulative), cumulative[-1])


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _nested_routes(raw: dict[str, Any]) -> dict[str, dict[str, Route]]:
    result: dict[str, dict[str, Route]] = {}
    for first, children in raw.items():
        if not isinstance(children, dict):
            continue
        for second, route_raw in children.items():
            route = build_route(route_raw)
            if route is not None:
                result.setdefault(str(first), {})[str(second)] = route
    return result


def _nested_times(raw: Any) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    if not isinstance(raw, dict):
        return result
    for first, children in raw.items():
        if not isinstance(children, dict):
            continue
        result[str(first)] = {
            str(second): float(value)
            for second, value in children.items()
            if isinstance(value, (int, float))
        }
    return result


def load_spatial_inputs(directory: str | Path | None) -> SpatialInputs:
    if directory is None:
        return SpatialInputs()
    root = Path(directory)
    walk_times_raw = _read_json(root / "walk_times_osm.json")
    walk_times = {
        str(name): float(value)
        for name, value in walk_times_raw.items()
        if isinstance(value, (int, float))
    }
    paths_raw = _read_json(root / "walk_paths_osm.json")
    direct = {
        str(name): route
        for name, raw in paths_raw.items()
        if (route := build_route(raw)) is not None
    }
    event_paths = _read_json(root / "walk_paths_event_osm.json")
    station_to_zone = _nested_routes(event_paths.get("station_to_zone", {}))
    zone_to_gate = _nested_routes(event_paths.get("zone_to_gate", {}))

    zone_file = _read_json(root / "event_zones_osm.json")
    zones: dict[str, tuple[float, float]] = {}
    for name, raw in zone_file.get("zones", {}).items():
        if not isinstance(raw, dict):
            continue
        x = raw.get("snapped_lon", raw.get("lon", raw.get("x")))
        y = raw.get("snapped_lat", raw.get("lat", raw.get("y")))
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            zones[str(name)] = (float(x), float(y))

    event_times = _read_json(root / "walk_times_event_osm.json")
    return SpatialInputs(
        walk_times=walk_times,
        direct=direct,
        station_to_zone=station_to_zone,
        zone_to_gate=zone_to_gate,
        zones=zones,
        station_to_zone_minutes=_nested_times(event_times.get("station_to_zone", {})),
        zone_to_gate_minutes=_nested_times(event_times.get("zone_to_gate", {})),
        source_directory=str(root),
    )


def density_grid(
    positions: pd.DataFrame,
    minute: int | None = None,
    bins: int = 60,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if positions.empty:
        raise ValueError("position log is empty; provide OSM paths and enable position logging")
    shown = positions if minute is None else positions[positions["minute"] == minute]
    if shown.empty:
        raise ValueError(f"no positions were recorded at minute {minute}")
    counts, x_edges, y_edges = np.histogram2d(shown["x"], shown["y"], bins=bins)
    return counts.T, x_edges, y_edges
