from __future__ import annotations

import unittest
from dataclasses import replace

from stadium_ingress.adaptive import (
    CONTEXT_COLUMNS,
    OPTIMISATION_TARGET,
    evaluate_policy_state,
    sample_policy_states,
)
from stadium_ingress.config import default_config


class AdaptiveTests(unittest.TestCase):
    def test_policy_state_design_includes_policy_and_state_variables(self) -> None:
        design = sample_policy_states(8, seed=12)
        self.assertEqual(list(design.columns), CONTEXT_COLUMNS)
        self.assertTrue(design["demand_scale"].between(0.80, 1.30).all())
        self.assertTrue(design["capacity_scale"].between(0.51, 1.02).all())

    def test_interaction_and_optimisation_outputs_are_separate(self) -> None:
        config = replace(default_config(), n_agents=1_000, capacity_scale=0.03)
        design = sample_policy_states(2, seed=7).iloc[0].copy()
        design["demand_scale"] = 1.3
        design["capacity_scale"] = 0.03
        summary, seed_level = evaluate_policy_state(config, design, (3,))
        expected = (
            seed_level.loc[0, "y10"]
            + seed_level.loc[0, "y01"]
            - seed_level.loc[0, "y11"]
            - seed_level.loc[0, "y00"]
        )
        self.assertAlmostEqual(seed_level.loc[0, "gamma"], expected)
        self.assertIn(OPTIMISATION_TARGET, summary)
        self.assertIn(OPTIMISATION_TARGET, seed_level.columns)
        self.assertNotEqual(seed_level.loc[0, "y11"], seed_level.loc[0, OPTIMISATION_TARGET])


if __name__ == "__main__":
    unittest.main()
