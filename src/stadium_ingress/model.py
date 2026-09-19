from __future__ import annotations

import math
import random
from collections import Counter, deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .arrivals import sample_desired_arrival
from .config import ModelConfig
from .spatial import Route, SpatialInputs, load_spatial_inputs


@dataclass(slots=True)
class Fan:
    agent_id: int
    target_minute: float
    desired_arrival_minute: float
    time_sensitivity: float
    crowding_sensitivity: float
    preference_food: float
    preference_stage: float
    knows_shuttle: bool = False
    knows_event: bool = False
    plans_event: bool = False
    event_zone: str | None = None
    event_dwell_minutes: int = 0
    station: str | None = None
    mode: str | None = None
    queue_enter_minute: int | None = None


@dataclass(slots=True)
class Walker:
    fan: Fan
    station: str
    queue_wait_minutes: float
    phase: str
    planned_minutes: float
    remaining_minutes: float
    route: Route | None
    zone: str | None = None
    moving_minutes: float = 0.0


@dataclass(slots=True)
class ShuttlePassenger:
    fan: Fan
    queue_wait_minutes: float
    remaining_minutes: float
    travel_minutes: float


@dataclass(frozen=True, slots=True)
class SimulationResult:
    minute_data: pd.DataFrame
    arrivals: pd.DataFrame
    metadata: dict[str, int | float | str | bool]
    choices: pd.DataFrame
    positions: pd.DataFrame


def _average_baseline_travel_minutes(config: ModelConfig) -> float:
    shares = np.asarray([station.share_hint for station in config.stations.values()], dtype=float)
    shares /= shares.sum()
    walks = np.asarray([station.walk_minutes for station in config.stations.values()], dtype=float)
    return config.rail_access_minutes + float(np.dot(walks, shares))


def _event_desired_arrival(config: ModelConfig, rng: np.random.RandomState) -> float:
    event = config.event
    candidates = np.arange(event.window_start, event.window_end + 1, dtype=int)
    if candidates.size == 0:
        return float(event.window_start)
    target = min(max(event.window_start, event.window_start + 15), event.window_end)
    early = np.maximum(0, target - candidates)
    late = np.maximum(0, candidates - target)
    weights = np.exp(-(event.early_penalty * early + event.late_penalty * late))
    weights /= weights.sum()
    return float(rng.choice(candidates, p=weights))


def _expected_zone_extra_minutes(
    config: ModelConfig,
    spatial: SpatialInputs,
    zone: str,
) -> float:
    gate_times = spatial.zone_to_gate_minutes.get(zone, {})
    gate_time = min(gate_times.values()) if gate_times else 0.0
    numerator = 0.0
    denominator = 0.0
    for station, station_config in config.stations.items():
        first_leg = spatial.station_to_zone_minutes.get(station, {}).get(zone)
        if first_leg is None:
            continue
        extra = max(0.0, float(first_leg) + gate_time - station_config.walk_minutes)
        numerator += station_config.share_hint * extra
        denominator += station_config.share_hint
    return numerator / denominator if denominator else 0.0


def _best_event_zone(
    config: ModelConfig,
    spatial: SpatialInputs,
    preference_food: float,
    preference_stage: float,
    time_sensitivity: float,
) -> tuple[str | None, float]:
    best_zone: str | None = None
    best_utility = -math.inf
    for zone, features in config.event.zone_features.items():
        taste = (
            preference_food * float(features.get("food", 0.0))
            + preference_stage * float(features.get("stage", 0.0))
        )
        attractiveness = config.event.zone_attractiveness.get(zone, 1.0)
        extra = _expected_zone_extra_minutes(config, spatial, zone)
        utility = (
            config.event.taste_weight * taste * config.event.reward * attractiveness
            - time_sensitivity * config.event.extra_walk_weight * extra
        )
        if utility > best_utility:
            best_zone, best_utility = zone, utility
    return best_zone, best_utility


def _build_agents(
    config: ModelConfig,
    spatial: SpatialInputs,
    rng: np.random.RandomState,
) -> tuple[list[Fan], int, int]:
    preferences: list[tuple[float, float, float, float]] = []
    awareness: list[tuple[bool, bool]] = []
    candidates: list[bool] = []
    zones: list[str | None] = []
    utilities: list[float] = []

    if config.information.enabled:
        awareness = [
            (
                bool(rng.random() < config.information.shuttle_awareness_rate),
                bool(rng.random() < config.information.event_awareness_rate),
            )
            for _ in range(config.n_agents)
        ]
    else:
        awareness = [(True, True)] * config.n_agents

    for index in range(config.n_agents):
        tastes = rng.dirichlet([2.0, 2.0])
        time_sensitivity = max(float(rng.normal(config.default_time_sensitivity, 0.1)), 0.1)
        crowding_sensitivity = max(
            float(rng.normal(config.default_crowding_sensitivity, 0.1)), 0.0
        )
        preferences.append(
            (time_sensitivity, crowding_sensitivity, float(tastes[0]), float(tastes[1]))
        )
        event_candidate = bool(
            config.event.enabled
            and awareness[index][1]
            and rng.random() < config.event.candidate_rate
        )
        candidates.append(event_candidate)
        if event_candidate:
            zone, utility = _best_event_zone(
                config,
                spatial,
                float(tastes[0]),
                float(tastes[1]),
                time_sensitivity,
            )
        else:
            zone, utility = None, -math.inf
        zones.append(zone)
        utilities.append(utility)

    finite_utilities = np.asarray(
        [
            value
            for value, candidate in zip(utilities, candidates, strict=True)
            if candidate and math.isfinite(value)
        ]
    )
    threshold = 0.0
    if finite_utilities.size:
        threshold = float(
            np.quantile(finite_utilities, 1.0 - config.event.target_join_rate)
        )

    base_travel = _average_baseline_travel_minutes(config)
    fans: list[Fan] = []
    participant_count = 0
    for agent_id, preference in enumerate(preferences):
        plan_event = False
        if candidates[agent_id] and zones[agent_id] is not None:
            scaled = (utilities[agent_id] - threshold) / max(config.event.join_temperature, 1e-6)
            join_probability = 1.0 / (1.0 + math.exp(-float(np.clip(scaled, -40, 40))))
            plan_event = bool(rng.random() < join_probability)
        participant_count += int(plan_event)
        desired = (
            _event_desired_arrival(config, rng)
            if plan_event
            else sample_desired_arrival(config.arrivals, rng)
        )
        target = max(float(config.sim_start_minute), desired - base_travel)
        dwell = 0
        if plan_event:
            dwell = int(
                round(
                    np.clip(
                        rng.normal(
                            config.event.dwell_mean_minutes,
                            config.event.dwell_sd_minutes,
                        ),
                        config.event.dwell_min_minutes,
                        config.event.dwell_max_minutes,
                    )
                )
            )
        fans.append(
            Fan(
                agent_id=agent_id,
                target_minute=target,
                desired_arrival_minute=desired,
                time_sensitivity=preference[0],
                crowding_sensitivity=preference[1],
                preference_food=preference[2],
                preference_stage=preference[3],
                knows_shuttle=awareness[agent_id][0],
                knows_event=awareness[agent_id][1],
                plans_event=plan_event,
                event_zone=zones[agent_id] if plan_event else None,
                event_dwell_minutes=dwell,
            )
        )
    return fans, int(sum(candidates)), participant_count


def _effective_capacity(config: ModelConfig, station_name: str) -> int:
    base_capacity = config.stations[station_name].capacity_per_minute
    return max(1, int(base_capacity * config.capacity_scale))


def _planned_extra_walk(
    config: ModelConfig,
    spatial: SpatialInputs,
    station: str,
    zone: str | None,
) -> float:
    if zone is None:
        return 0.0
    first = spatial.station_to_zone_minutes.get(station, {}).get(zone)
    gates = spatial.zone_to_gate_minutes.get(zone, {})
    if first is None or not gates:
        return 0.0
    return max(0.0, float(first) + min(gates.values()) - config.stations[station].walk_minutes)


def _choose_mode(
    fan: Fan,
    config: ModelConfig,
    spatial: SpatialInputs,
    rail_queues: dict[str, deque[Fan]],
    shuttle_queue: deque[Fan],
    rng: np.random.RandomState,
) -> tuple[str, str | None]:
    alternatives: list[tuple[float, str, str | None]] = []
    for name, station in config.stations.items():
        capacity = _effective_capacity(config, name)
        queue_length = len(rail_queues[name])
        wait = queue_length / capacity
        extra_walk = _planned_extra_walk(config, spatial, name, fan.event_zone)
        time_term = fan.time_sensitivity * (
            config.rail_access_minutes
            + station.time_bias_minutes
            + wait
            + station.walk_minutes
            + config.event.extra_walk_weight * extra_walk
        )
        crowding_term = fan.crowding_sensitivity * queue_length / capacity
        prior_term = config.prior_alpha * (-math.log(max(station.share_hint, 1e-6)))
        noise = float(rng.gumbel(0.0, config.epsilon_noise))
        alternatives.append((time_term + crowding_term + prior_term + noise, "rail", name))

    shuttle_capacity = config.shuttle.effective_capacity_per_minute
    if config.shuttle.enabled and fan.knows_shuttle and shuttle_capacity > 0:
        queue_length = len(shuttle_queue)
        wait = queue_length / shuttle_capacity
        time_term = fan.time_sensitivity * (
            wait
            + config.shuttle.in_vehicle_minutes
            + config.shuttle.dropoff_walk_minutes
            + config.shuttle.extra_time_penalty_minutes
        )
        crowding_term = fan.crowding_sensitivity * queue_length / shuttle_capacity
        alternatives.append(
            (
                time_term + crowding_term + float(rng.gumbel(0.0, config.epsilon_noise)),
                "shuttle",
                None,
            )
        )
    _, mode, station = min(alternatives, key=lambda item: item[0])
    return mode, station


def _walk_multiplier(config: ModelConfig, rng: np.random.RandomState) -> float:
    stochastic = config.stochastic
    return float(
        np.clip(
            rng.normal(stochastic.walk_multiplier_mean, stochastic.walk_multiplier_sd),
            stochastic.walk_multiplier_min,
            stochastic.walk_multiplier_max,
        )
    )


def _route_and_duration(
    fan: Fan,
    station: str,
    phase: str,
    config: ModelConfig,
    spatial: SpatialInputs,
    rng: np.random.RandomState,
) -> tuple[Route | None, float]:
    zone = fan.event_zone
    if phase == "direct":
        duration = config.stations[station].walk_minutes * _walk_multiplier(config, rng)
        return spatial.direct.get(station), duration
    if phase == "to_zone" and zone:
        duration = spatial.station_to_zone_minutes.get(station, {}).get(
            zone, config.stations[station].walk_minutes
        )
        return spatial.station_to_zone.get(station, {}).get(zone), duration * _walk_multiplier(config, rng)
    if phase == "to_gate" and zone:
        times = spatial.zone_to_gate_minutes.get(zone, {})
        routes = spatial.zone_to_gate.get(zone, {})
        gate = min(times, key=times.get) if times else (next(iter(routes), "Gate_A"))
        duration = times.get(gate, 5.0)
        return routes.get(gate), duration * _walk_multiplier(config, rng)
    return None, 0.0


def simulate(config: ModelConfig, seed: int = 42) -> SimulationResult:
    """Run one ingress replication including optional shuttle, event, and OSM paths."""
    config.validate()
    rng = np.random.RandomState(seed)
    activation_rng = random.Random(seed)
    spatial = load_spatial_inputs(config.spatial.input_directory)
    if spatial.walk_times:
        known_overrides = {
            name: value for name, value in spatial.walk_times.items() if name in config.stations
        }
        config = config.with_walk_time_overrides(known_overrides)
    fans, candidate_count, participant_count = _build_agents(config, spatial, rng)
    fans.sort(key=lambda fan: fan.target_minute)

    rail_queues = {name: deque() for name in config.stations}
    shuttle_queue: deque[Fan] = deque()
    walkers: list[Walker] = []
    passengers: list[ShuttlePassenger] = []
    pending_index = 0
    queue_rows: list[tuple[int, int, int]] = []
    arrival_rows: list[tuple[int, float, str, str | None, float, float, int, str, float]] = []
    choice_rows: list[tuple[int, int, str, str | None]] = []
    position_rows: list[tuple[int, int, float, float, str]] = []
    arrivals_by_minute: Counter[int] = Counter()
    shuttle_boarded = 0

    def record_arrival(
        fan: Fan,
        minute: float,
        mode: str,
        station: str | None,
        wait: float,
        travel: float,
    ) -> None:
        arrival_rows.append(
            (
                fan.agent_id,
                minute,
                mode,
                station,
                wait,
                travel,
                int(fan.plans_event),
                fan.event_zone or "",
                fan.desired_arrival_minute,
            )
        )
        arrivals_by_minute[math.floor(minute)] += 1

    for minute in range(config.sim_start_minute, config.sim_end_minute + 1):
        due: list[Fan] = []
        while pending_index < len(fans) and fans[pending_index].target_minute <= minute:
            due.append(fans[pending_index])
            pending_index += 1
        activation_rng.shuffle(due)
        for fan in due:
            mode, station = _choose_mode(
                fan, config, spatial, rail_queues, shuttle_queue, rng
            )
            fan.mode, fan.station, fan.queue_enter_minute = mode, station, minute
            choice_rows.append((minute, fan.agent_id, mode, station))
            if mode == "shuttle":
                shuttle_queue.append(fan)
            else:
                rail_queues[station].append(fan)  # type: ignore[index]

        for station_name in config.stations:
            queue = rail_queues[station_name]
            for _ in range(min(_effective_capacity(config, station_name), len(queue))):
                fan = queue.popleft()
                entered = fan.queue_enter_minute if fan.queue_enter_minute is not None else minute
                wait = float(minute - entered) + float(
                    rng.uniform(
                        config.stochastic.queue_wait_jitter_min,
                        config.stochastic.queue_wait_jitter_max,
                    )
                )
                phase = "to_zone" if fan.plans_event and fan.event_zone else "direct"
                route, duration = _route_and_duration(
                    fan, station_name, phase, config, spatial, rng
                )
                walkers.append(
                    Walker(
                        fan=fan,
                        station=station_name,
                        queue_wait_minutes=wait,
                        phase=phase,
                        planned_minutes=max(duration, 0.01),
                        remaining_minutes=max(duration, 0.01),
                        route=route,
                        zone=fan.event_zone,
                    )
                )

        shuttle_capacity = config.shuttle.effective_capacity_per_minute
        for _ in range(min(shuttle_capacity, len(shuttle_queue))):
            fan = shuttle_queue.popleft()
            entered = fan.queue_enter_minute if fan.queue_enter_minute is not None else minute
            wait = float(minute - entered) + float(
                rng.uniform(
                    config.stochastic.queue_wait_jitter_min,
                    config.stochastic.queue_wait_jitter_max,
                )
            )
            travel = (
                config.shuttle.in_vehicle_minutes
                + config.shuttle.dropoff_walk_minutes
                + float(rng.uniform(-0.5, 0.5))
            )
            if fan.plans_event:
                travel += fan.event_dwell_minutes + 5.0
            passengers.append(ShuttlePassenger(fan, wait, travel, travel))
            shuttle_boarded += 1

        remaining_passengers: list[ShuttlePassenger] = []
        for passenger in passengers:
            passenger.remaining_minutes -= 1.0
            if passenger.remaining_minutes <= 0:
                fraction = max(0.0, passenger.remaining_minutes + 1.0)
                arrival_minute = minute + fraction
                record_arrival(
                    passenger.fan,
                    arrival_minute,
                    "shuttle",
                    None,
                    passenger.queue_wait_minutes,
                    passenger.travel_minutes,
                )
            else:
                remaining_passengers.append(passenger)
        passengers = remaining_passengers

        route_counts = Counter(walker.station for walker in walkers if walker.phase != "dwell")
        remaining_walkers: list[Walker] = []
        for walker in walkers:
            if walker.phase == "dwell":
                walker.remaining_minutes -= 1.0
                if config.spatial.log_positions and walker.zone in spatial.zones:
                    x, y = spatial.zones[walker.zone]
                    position_rows.append((minute, walker.fan.agent_id, x, y, f"dwell:{walker.zone}"))
                if walker.remaining_minutes <= 0:
                    route, duration = _route_and_duration(
                        walker.fan, walker.station, "to_gate", config, spatial, rng
                    )
                    walker.phase = "to_gate"
                    walker.route = route
                    walker.planned_minutes = max(duration, 0.01)
                    walker.remaining_minutes = max(duration, 0.01)
                remaining_walkers.append(walker)
                continue

            # Route congestion is meaningful only when a coordinate route is
            # available. Duration fallback must preserve the appendix's
            # station-level walking time rather than inventing path crowding.
            progress = 1.0
            if walker.route is not None:
                path_capacity = max(config.spatial.default_path_capacity, 1)
                ratio = route_counts[walker.station] / path_capacity
                progress = 1.0 if ratio <= 1.0 else 1.0 / (
                    1.0 + config.spatial.walk_congestion_alpha * (ratio - 1.0)
                )
            before = walker.remaining_minutes
            walker.remaining_minutes -= progress
            walker.moving_minutes += progress
            if config.spatial.log_positions and walker.route is not None:
                fraction = 1.0 - max(walker.remaining_minutes, 0.0) / walker.planned_minutes
                x, y = walker.route.interpolate(fraction)
                position_rows.append((minute, walker.fan.agent_id, x, y, walker.phase))
            if walker.remaining_minutes > 0:
                remaining_walkers.append(walker)
                continue

            fraction = min(1.0, before / max(progress, 1e-9))
            transition_minute = minute + fraction
            if walker.phase == "to_zone":
                walker.phase = "dwell"
                walker.route = None
                walker.planned_minutes = max(float(walker.fan.event_dwell_minutes), 0.01)
                walker.remaining_minutes = walker.planned_minutes
                remaining_walkers.append(walker)
            else:
                travel = config.rail_access_minutes + walker.queue_wait_minutes + walker.moving_minutes
                record_arrival(
                    walker.fan,
                    transition_minute,
                    "rail",
                    walker.station,
                    walker.queue_wait_minutes,
                    travel,
                )
        walkers = remaining_walkers
        rail_queue_length = sum(len(queue) for queue in rail_queues.values())
        queue_rows.append((minute, rail_queue_length, len(shuttle_queue)))

    minutes = np.arange(config.sim_start_minute, config.sim_end_minute + 1, dtype=int)
    queue_lookup = {row[0]: row[1:] for row in queue_rows}
    arrival_counts = np.asarray([arrivals_by_minute[int(minute)] for minute in minutes])
    minute_data = pd.DataFrame(
        {
            "minute_before_kickoff": minutes,
            "arrivals_count": arrival_counts,
            "arrival_rate_pct": arrival_counts / config.n_agents * 100.0,
            "rail_queue_length": [queue_lookup[int(minute)][0] for minute in minutes],
            "shuttle_queue_length": [queue_lookup[int(minute)][1] for minute in minutes],
        }
    )
    minute_data["total_queue_length"] = (
        minute_data["rail_queue_length"] + minute_data["shuttle_queue_length"]
    )
    columns = [
        "agent_id",
        "arrival_minute",
        "mode",
        "station",
        "queue_wait_minutes",
        "travel_minutes",
        "plan_event",
        "event_zone",
        "desired_arrival_minute",
    ]
    arrivals = pd.DataFrame(arrival_rows, columns=columns)
    if not arrivals.empty:
        arrivals = arrivals.sort_values(["arrival_minute", "agent_id"], ignore_index=True)
    choices = pd.DataFrame(choice_rows, columns=["minute", "agent_id", "mode", "station"])
    positions = pd.DataFrame(position_rows, columns=["minute", "agent_id", "x", "y", "phase"])

    waits = arrivals["queue_wait_minutes"] if not arrivals.empty else pd.Series([0.0])
    pre_kickoff = int((arrivals["arrival_minute"] <= 0).sum()) if not arrivals.empty else 0
    simulation_hours = (config.sim_end_minute - config.sim_start_minute + 1) / 60.0
    intervention_cost = 0.0
    if config.shuttle.enabled:
        intervention_cost += (
            config.shuttle.fleet_size
            * config.shuttle.cost_per_vehicle_hour
            * simulation_hours
        )
    if config.event.enabled:
        intervention_cost += config.event.operating_cost
    if config.information.enabled:
        reach = max(
            config.information.shuttle_awareness_rate,
            config.information.event_awareness_rate,
        )
        intervention_cost += config.n_agents * reach * config.information.cost_per_person_reached

    metadata: dict[str, int | float | str | bool] = {
        "seed": seed,
        "n_agents": config.n_agents,
        "arrivals_recorded": len(arrivals),
        "arrivals_by_simulation_end": int(len(arrivals)),
        "arrivals_pre_kickoff": pre_kickoff,
        "not_arrived_by_simulation_end": (
            config.n_agents - len(arrivals)
        ),
        "waiting_in_queues_at_simulation_end": (
            sum(len(queue) for queue in rail_queues.values()) + len(shuttle_queue)
        ),
        "total_effective_rail_capacity_per_minute": sum(
            _effective_capacity(config, name) for name in config.stations
        ),
        "shuttle_capacity_per_minute": shuttle_capacity,
        "shuttle_boarded": shuttle_boarded,
        "shuttle_share": float((arrivals["mode"] == "shuttle").mean()) if not arrivals.empty else 0.0,
        "event_candidates": candidate_count,
        "event_participants": participant_count,
        "event_participation_rate": participant_count / config.n_agents,
        "mean_queue_wait_minutes": float(waits.mean()),
        "p90_queue_wait_minutes": float(waits.quantile(0.9)),
        "peak_total_queue": int(minute_data["total_queue_length"].max()),
        "pre_kickoff_arrival_share": pre_kickoff / config.n_agents,
        "intervention_cost": float(intervention_cost),
        "spatial_paths_loaded": spatial.paths_available,
        "spatial_route_mode": "osm-coordinate" if spatial.paths_available else "duration-fallback",
    }
    return SimulationResult(minute_data, arrivals, metadata, choices, positions)
