from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np
import pandas as pd

from stadium_ingress.analysis import (
    DESIGN_COLUMNS,
    capacity_interaction_analysis,
    run_design,
    sample_designs,
)
from stadium_ingress.config import default_config
from stadium_ingress.optimisation import nsga2
from stadium_ingress.surrogate import fit_surrogates
from stadium_ingress.validation import (
    frontier_verification_metrics,
    hypervolume_2d,
    nondominated_mask,
)


class AnalysisTests(unittest.TestCase):
    def test_design_uses_seven_reported_policy_variables(self) -> None:
        designs = sample_designs(20, seed=8)
        self.assertEqual(list(designs.columns), DESIGN_COLUMNS)
        self.assertEqual(len(DESIGN_COLUMNS), 7)
        self.assertTrue(designs["shuttle_headway_minutes"].between(6, 15).all())
        self.assertTrue(designs["fleet_size"].between(3, 9).all())
        self.assertTrue(designs["shuttle_vehicle_capacity"].between(60, 100).all())
        self.assertTrue(designs["event_target_join_rate"].between(0.30, 0.80).all())
        self.assertTrue(designs["event_dwell_mean_minutes"].between(10, 30).all())
        self.assertTrue(designs["shuttle_awareness"].between(0.25, 1.00).all())
        self.assertTrue(designs["event_awareness"].between(0.25, 1.00).all())

    def test_factorial_interaction_has_all_four_scores(self) -> None:
        config = replace(default_config(), n_agents=100)
        frame = capacity_interaction_analysis(config, [1.0], [0.3], [4])
        self.assertEqual(len(frame), 1)
        for column in ("baseline_score", "spatial_score", "temporal_score", "joint_score", "interaction"):
            self.assertIn(column, frame.columns)
        for column in ("interaction_se", "ci95_lower", "ci95_upper", "classification"):
            self.assertIn(column, frame.columns)

    def test_surrogate_and_nsga2_produce_pareto_solutions(self) -> None:
        config = replace(default_config(), n_agents=80)
        training = run_design(config, sample_designs(4, seed=2), [3])
        surrogate = fit_surrogates(training, backend="polynomial")
        pareto = nsga2(surrogate, population_size=8, generations=2, seed=5)
        self.assertFalse(pareto.empty)
        self.assertIn("predicted_congestion_score", pareto.columns)

    def test_frontier_metrics_report_retained_nondominance_and_hypervolume(self) -> None:
        objectives = np.array([[1.0, 4.0], [2.0, 2.0], [3.0, 3.0]])
        np.testing.assert_array_equal(nondominated_mask(objectives), [True, True, False])
        self.assertAlmostEqual(hypervolume_2d(objectives, np.array([5.0, 5.0])), 10.0)
        verification = pd.DataFrame(
            {
                "predicted_congestion_score": [1.0, 2.0, 3.0],
                "predicted_intervention_cost": [4.0, 2.0, 3.0],
                "simulated_congestion_score": [1.1, 2.1, 3.2],
                "simulated_intervention_cost": [4.0, 2.0, 3.0],
            }
        )
        detailed, metrics = frontier_verification_metrics(verification)
        self.assertIn("simulated_nondominated", detailed.columns)
        self.assertEqual(metrics["retained_nondominated_points"], 2.0)
        self.assertGreater(metrics["predicted_hypervolume"], 0.0)


if __name__ == "__main__":
    unittest.main()
