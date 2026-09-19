from __future__ import annotations

import unittest
from dataclasses import replace

import pandas as pd

from stadium_ingress import aggregate_replications, default_config, simulate


def small_config():
    return replace(default_config(), n_agents=1_000, capacity_scale=0.03)


class ModelTests(unittest.TestCase):
    def test_simulation_is_deterministic_for_a_fixed_seed(self) -> None:
        first = simulate(small_config(), seed=7)
        second = simulate(small_config(), seed=7)
        pd.testing.assert_frame_equal(first.minute_data, second.minute_data)
        pd.testing.assert_frame_equal(first.arrivals, second.arrivals)

    def test_minute_outputs_are_complete_and_nonnegative(self) -> None:
        result = simulate(small_config(), seed=11)
        expected_rows = (
            result.minute_data["minute_before_kickoff"].iloc[-1]
            - result.minute_data["minute_before_kickoff"].iloc[0]
            + 1
        )
        self.assertEqual(len(result.minute_data), expected_rows)
        self.assertTrue((result.minute_data["arrivals_count"] >= 0).all())
        self.assertTrue((result.minute_data["total_queue_length"] >= 0).all())
        self.assertLessEqual(result.metadata["arrivals_recorded"], 1_000)

    def test_aggregation_normalizes_pre_kickoff_queue_to_100_percent(self) -> None:
        config = small_config()
        aggregated = aggregate_replications(
            [simulate(config, seed=1), simulate(config, seed=2)]
        )
        shown = aggregated[aggregated["minute_before_kickoff"] <= 0]
        self.assertLess(
            abs(shown["queue_pct_of_pre_kickoff_max"].max() - 100.0), 1e-9
        )

    def test_combined_policy_activates_shuttle_and_event(self) -> None:
        base = replace(default_config(), n_agents=500)
        config = base.with_policy(
            shuttle_enabled=True,
            event_enabled=True,
            shuttle_awareness=0.8,
            event_awareness=0.8,
        )
        result = simulate(config, seed=19)
        self.assertGreater(result.metadata["shuttle_boarded"], 0)
        self.assertGreater(result.metadata["event_participants"], 0)
        self.assertIn("mode", result.arrivals.columns)


if __name__ == "__main__":
    unittest.main()
