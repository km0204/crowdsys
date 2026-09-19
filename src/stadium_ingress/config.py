from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class StationConfig:
    capacity_per_minute: int
    walk_minutes: float
    share_hint: float
    time_bias_minutes: float = 0.0

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "StationConfig":
        return cls(
            capacity_per_minute=int(values.get("capacity_per_minute", values.get("cap"))),
            walk_minutes=float(values.get("walk_minutes", values.get("walk"))),
            share_hint=float(values.get("share_hint", values.get("share"))),
            time_bias_minutes=float(values.get("time_bias_minutes", 0.0)),
        )


@dataclass(frozen=True, slots=True)
class ArrivalConfig:
    left_start: int = -140
    left_end: int = -60
    plateau_start: int = -60
    plateau_end: int = 0
    late_max: int = 30
    decay_rate: float = 0.18


@dataclass(frozen=True, slots=True)
class StochasticConfig:
    walk_multiplier_mean: float = 1.0
    walk_multiplier_sd: float = 0.15
    walk_multiplier_min: float = 0.7
    walk_multiplier_max: float = 1.4
    queue_wait_jitter_min: float = 0.0
    queue_wait_jitter_max: float = 1.0


@dataclass(frozen=True, slots=True)
class ShuttleConfig:
    enabled: bool = False
    headway_minutes: int = 6
    vehicle_capacity: int = 60
    fleet_size: int = 15
    boarding_rate_per_minute: int = 120
    in_vehicle_minutes: float = 15.0
    dropoff_walk_minutes: float = 0.0
    extra_time_penalty_minutes: float = 0.0
    cost_per_vehicle_hour: float = 120.0

    @property
    def effective_capacity_per_minute(self) -> int:
        if not self.enabled:
            return 0
        fleet_capacity = self.vehicle_capacity / max(self.headway_minutes, 1) * self.fleet_size
        return max(1, int(min(self.boarding_rate_per_minute, fleet_capacity)))


@dataclass(frozen=True, slots=True)
class EventConfig:
    enabled: bool = False
    candidate_rate: float = 0.35
    window_start: int = -150
    window_end: int = -60
    early_penalty: float = 0.1
    late_penalty: float = 0.6
    reward: float = 0.6
    join_temperature: float = 0.35
    target_join_rate: float = 0.5
    dwell_mean_minutes: float = 20.0
    dwell_sd_minutes: float = 7.0
    dwell_min_minutes: float = 5.0
    dwell_max_minutes: float = 60.0
    taste_weight: float = 1.0
    extra_walk_weight: float = 1.0
    operating_cost: float = 3000.0
    zone_features: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            "Zone_CE": {"food": 1.0, "stage": 0.0},
            "Zone_EG": {"food": 0.0, "stage": 1.0},
        }
    )
    zone_attractiveness: dict[str, float] = field(
        default_factory=lambda: {"Zone_CE": 1.0, "Zone_EG": 1.0}
    )


@dataclass(frozen=True, slots=True)
class InformationConfig:
    enabled: bool = False
    shuttle_awareness_rate: float = 0.0
    event_awareness_rate: float = 0.0
    cost_per_person_reached: float = 0.05


@dataclass(frozen=True, slots=True)
class SpatialConfig:
    input_directory: str | None = None
    log_positions: bool = False
    walk_congestion_alpha: float = 1.0
    default_path_capacity: int = 200


@dataclass(frozen=True, slots=True)
class ModelConfig:
    sim_start_minute: int
    sim_end_minute: int
    n_agents: int
    capacity_scale: float
    rail_access_minutes: float
    prior_alpha: float
    epsilon_noise: float
    default_time_sensitivity: float
    default_crowding_sensitivity: float
    arrivals: ArrivalConfig
    stochastic: StochasticConfig
    stations: dict[str, StationConfig]
    shuttle: ShuttleConfig = field(default_factory=ShuttleConfig)
    event: EventConfig = field(default_factory=EventConfig)
    information: InformationConfig = field(default_factory=InformationConfig)
    spatial: SpatialConfig = field(default_factory=SpatialConfig)

    def validate(self) -> None:
        if self.sim_start_minute >= self.sim_end_minute:
            raise ValueError("sim_start_minute must be earlier than sim_end_minute")
        if self.n_agents <= 0:
            raise ValueError("n_agents must be positive")
        if self.capacity_scale <= 0:
            raise ValueError("capacity_scale must be positive")
        if not self.stations:
            raise ValueError("at least one station is required")
        if self.arrivals.left_start >= self.arrivals.left_end:
            raise ValueError("arrival left interval is invalid")
        if self.arrivals.plateau_start >= self.arrivals.plateau_end:
            raise ValueError("arrival plateau interval is invalid")
        if self.arrivals.plateau_start > self.arrivals.left_end:
            raise ValueError("arrival distribution contains a gap")
        if self.sim_start_minute > self.arrivals.left_start:
            raise ValueError("simulation starts after the arrival distribution")
        if self.sim_end_minute < self.arrivals.late_max:
            raise ValueError("simulation ends before the late-arrival tail")
        rates = {
            "shuttle_awareness_rate": self.information.shuttle_awareness_rate,
            "event_awareness_rate": self.information.event_awareness_rate,
            "event candidate_rate": self.event.candidate_rate,
            "event target_join_rate": self.event.target_join_rate,
        }
        for label, value in rates.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{label} must be between zero and one")
        for name, station in self.stations.items():
            if station.capacity_per_minute <= 0:
                raise ValueError(f"station {name!r} has non-positive capacity")
            if station.walk_minutes <= 0:
                raise ValueError(f"station {name!r} has non-positive walking time")
            if station.share_hint <= 0:
                raise ValueError(f"station {name!r} has non-positive share_hint")

    def with_walk_time_overrides(self, overrides: dict[str, float]) -> "ModelConfig":
        unknown = sorted(set(overrides) - set(self.stations))
        if unknown:
            raise ValueError(f"unknown station names in walk-time overrides: {unknown}")
        stations = {
            name: replace(station, walk_minutes=float(overrides.get(name, station.walk_minutes)))
            for name, station in self.stations.items()
        }
        updated = replace(self, stations=stations)
        updated.validate()
        return updated

    def with_policy(
        self,
        *,
        shuttle_enabled: bool | None = None,
        event_enabled: bool | None = None,
        shuttle_awareness: float | None = None,
        event_awareness: float | None = None,
    ) -> "ModelConfig":
        shuttle = replace(
            self.shuttle,
            enabled=self.shuttle.enabled if shuttle_enabled is None else shuttle_enabled,
        )
        event = replace(
            self.event,
            enabled=self.event.enabled if event_enabled is None else event_enabled,
        )
        information = replace(
            self.information,
            enabled=(shuttle.enabled or event.enabled),
            shuttle_awareness_rate=(
                self.information.shuttle_awareness_rate
                if shuttle_awareness is None
                else shuttle_awareness
            ),
            event_awareness_rate=(
                self.information.event_awareness_rate
                if event_awareness is None
                else event_awareness
            ),
        )
        updated = replace(self, shuttle=shuttle, event=event, information=information)
        updated.validate()
        return updated

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ModelConfig":
        arrivals_values = dict(values.get("arrivals", {}))
        if "plat_start" in arrivals_values:
            arrivals_values["plateau_start"] = arrivals_values.pop("plat_start")
            arrivals_values["plateau_end"] = arrivals_values.pop("plat_end")
        arrivals = ArrivalConfig(**arrivals_values)

        stochastic_values = dict(values.get("stochastic", {}))
        if "walk_time_mult_mean" in stochastic_values:
            stochastic_values["walk_multiplier_mean"] = stochastic_values.pop("walk_time_mult_mean")
            stochastic_values["walk_multiplier_sd"] = stochastic_values.pop("walk_time_mult_sd")
        if "walk_time_mult_clip" in stochastic_values:
            clip = stochastic_values.pop("walk_time_mult_clip")
            stochastic_values["walk_multiplier_min"] = clip[0]
            stochastic_values["walk_multiplier_max"] = clip[1]
        if "queue_wait_jitter_min" in stochastic_values and isinstance(
            stochastic_values["queue_wait_jitter_min"], list
        ):
            jitter = stochastic_values["queue_wait_jitter_min"]
            stochastic_values["queue_wait_jitter_min"] = jitter[0]
            stochastic_values["queue_wait_jitter_max"] = jitter[1]
        stochastic = StochasticConfig(**stochastic_values)
        stations = {name: StationConfig.from_dict(station) for name, station in values["stations"].items()}

        shuttle_values = dict(values.get("shuttle", {}))
        aliases = {
            "headway_min": "headway_minutes",
            "in_vehicle_time_min": "in_vehicle_minutes",
            "dropoff_walk_min": "dropoff_walk_minutes",
            "extra_time_penalty_min": "extra_time_penalty_minutes",
        }
        routes = shuttle_values.pop("routes", None)
        if routes and "in_vehicle_minutes" not in shuttle_values:
            times = [float(r.get("in_vehicle_time_min", 15.0)) for r in routes]
            shuttle_values["in_vehicle_minutes"] = sum(times) / len(times)
        for old, new in aliases.items():
            if old in shuttle_values:
                shuttle_values[new] = shuttle_values.pop(old)
        shuttle = ShuttleConfig(**shuttle_values)

        event_values = dict(values.get("event", {}))
        event_aliases = {
            "p_event": "candidate_rate",
            "theta_E": "early_penalty",
            "theta_L": "late_penalty",
            "reward_event": "reward",
            "join_tau": "join_temperature",
            "join_target_rate": "target_join_rate",
            "dwell_min_mean": "dwell_mean_minutes",
            "dwell_min_sd": "dwell_sd_minutes",
        }
        window = event_values.pop("window", None)
        if window:
            event_values["window_start"] = int(window.get("start_min", -150))
            event_values["window_end"] = int(window.get("end_min", -60))
        dwell_clip = event_values.pop("dwell_min_clip", None)
        if dwell_clip:
            event_values["dwell_min_minutes"] = float(dwell_clip[0])
            event_values["dwell_max_minutes"] = float(dwell_clip[1])
        unused = (
            "join_model",
            "join_threshold",
            "join_calibrate",
            "choose_gate_by_shortest",
            "zone_choice_walk_weight",
        )
        for key in unused:
            event_values.pop(key, None)
        for old, new in event_aliases.items():
            if old in event_values:
                event_values[new] = event_values.pop(old)
        event = EventConfig(**event_values)

        info_values = dict(values.get("information", values.get("info", {})))
        information = InformationConfig(**info_values)
        spatial_values = dict(values.get("spatial", {}))
        if "log_walk_positions" in values:
            spatial_values.setdefault("log_positions", values["log_walk_positions"])
        if "walk_congestion_alpha" in values:
            spatial_values.setdefault("walk_congestion_alpha", values["walk_congestion_alpha"])
        spatial = SpatialConfig(**spatial_values)
        sim = values.get("sim", {})
        config = cls(
            sim_start_minute=int(values.get("sim_start_minute", sim.get("start_min", -180))),
            sim_end_minute=int(values.get("sim_end_minute", sim.get("end_min", 60))),
            n_agents=int(values.get("n_agents", 15_000)),
            capacity_scale=float(values.get("capacity_scale", 0.3)),
            rail_access_minutes=float(values.get("rail_access_minutes", values.get("rail_access_min", 2.0))),
            prior_alpha=float(values.get("prior_alpha", 1.0)),
            epsilon_noise=float(values.get("epsilon_noise", values.get("eps_noise", 0.5))),
            default_time_sensitivity=float(values.get("default_time_sensitivity", 1.0)),
            default_crowding_sensitivity=float(values.get("default_crowding_sensitivity", 0.6)),
            arrivals=arrivals,
            stochastic=stochastic,
            stations=stations,
            shuttle=shuttle,
            event=event,
            information=information,
            spatial=spatial,
        )
        config.validate()
        return config


def default_config() -> ModelConfig:
    return ModelConfig.from_dict(
        {
            "sim_start_minute": -180,
            "sim_end_minute": 60,
            "n_agents": 15_000,
            "capacity_scale": 0.3,
            "rail_access_minutes": 2.0,
            "prior_alpha": 1.0,
            "epsilon_noise": 0.5,
            "default_time_sensitivity": 1.0,
            "default_crowding_sensitivity": 0.6,
            "stations": {
                "KokuritsuKyogijo_Oedo": {
                    "capacity_per_minute": 60,
                    "walk_minutes": 8.0,
                    "share_hint": 0.40,
                    "time_bias_minutes": 1.0,
                },
                "Sendagaya_JR": {
                    "capacity_per_minute": 60,
                    "walk_minutes": 13.0,
                    "share_hint": 0.30,
                    "time_bias_minutes": -1.0,
                },
                "Shinanomachi_JR": {
                    "capacity_per_minute": 60,
                    "walk_minutes": 11.0,
                    "share_hint": 0.15,
                    "time_bias_minutes": 0.0,
                },
                "Gaienmae_Ginza": {
                    "capacity_per_minute": 60,
                    "walk_minutes": 11.0,
                    "share_hint": 0.10,
                    "time_bias_minutes": -1.0,
                },
                "AoyamaItchome_OedoHanzomonGinza": {
                    "capacity_per_minute": 60,
                    "walk_minutes": 20.0,
                    "share_hint": 0.05,
                    "time_bias_minutes": -1.0,
                },
                "Kitasando_Fukutoshin": {
                    "capacity_per_minute": 60,
                    "walk_minutes": 16.0,
                    "share_hint": 0.02,
                    "time_bias_minutes": -1.0,
                },
            },
        }
    )


def load_config(path: str | Path | None = None) -> ModelConfig:
    if path is None:
        return default_config()
    user_values = json.loads(Path(path).read_text(encoding="utf-8"))
    values = asdict(default_config())

    def merge(target: dict[str, Any], updates: dict[str, Any]) -> None:
        for key, value in updates.items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                merge(target[key], value)
            else:
                target[key] = value

    merge(values, user_values)
    return ModelConfig.from_dict(values)


def load_walk_time_overrides(path: str | Path | None) -> dict[str, float]:
    if path is None or not Path(path).exists():
        return {}
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        str(name): float(minutes)
        for name, minutes in values.items()
        if isinstance(minutes, (int, float))
    }
