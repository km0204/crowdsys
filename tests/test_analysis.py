from __future__ import annotations

import unittest
from dataclasses import replace

from stadium_ingress.analysis import capacity_interaction_analysis, run_design, sample_designs
from stadium_ingress.config import default_config
from stadium_ingress.optimisation import nsga2
from stadium_ingress.surrogate import fit_surrogates


class AnalysisTests(unittest.TestCase):
    def test_factorial_interaction_has_all_four_scores(self) -> None:
        config = replace(default_config(), n_agents=100)
        frame = capacity_interaction_analysis(config, [1.0], [0.3], [4])
        self.assertEqual(len(frame), 1)
        for column in ("baseline_score", "spatial_score", "temporal_score", "joint_score", "interaction"):
            self.assertIn(column, frame.columns)

    def test_surrogate_and_nsga2_produce_pareto_solutions(self) -> None:
        config = replace(default_config(), n_agents=80)
        training = run_design(config, sample_designs(4, seed=2), [3])
        surrogate = fit_surrogates(training, backend="polynomial")
        pareto = nsga2(surrogate, population_size=8, generations=2, seed=5)
        self.assertFalse(pareto.empty)
        self.assertIn("predicted_congestion_score", pareto.columns)


if __name__ == "__main__":
    unittest.main()
