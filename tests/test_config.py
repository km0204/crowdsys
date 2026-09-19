from __future__ import annotations

import unittest

from stadium_ingress.config import ModelConfig, default_config, load_walk_time_overrides


class ConfigTests(unittest.TestCase):
    def test_default_config_is_valid(self) -> None:
        config = default_config()
        config.validate()
        self.assertEqual(config.n_agents, 15_000)
        self.assertEqual(len(config.stations), 6)

    def test_walk_time_overrides_are_applied(self) -> None:
        config = default_config().with_walk_time_overrides({"Sendagaya_JR": 12.5})
        self.assertEqual(config.stations["Sendagaya_JR"].walk_minutes, 12.5)

    def test_unknown_walk_time_station_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown station"):
            default_config().with_walk_time_overrides({"Unknown": 10.0})

    def test_missing_walk_time_file_returns_empty_mapping(self) -> None:
        self.assertEqual(load_walk_time_overrides("missing.json"), {})

    def test_config_rejects_arrival_gap(self) -> None:
        values = {
            "arrivals": {"left_end": -80, "plateau_start": -60},
            "stations": {
                "A": {
                    "capacity_per_minute": 1,
                    "walk_minutes": 1,
                    "share_hint": 1,
                }
            },
        }
        with self.assertRaisesRegex(ValueError, "gap"):
            ModelConfig.from_dict(values)


if __name__ == "__main__":
    unittest.main()
