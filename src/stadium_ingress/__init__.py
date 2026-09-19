"""Agent-based stadium ingress simulation."""

from .analysis import capacity_interaction_analysis, robustness_analysis, sensitivity_analysis
from .config import ModelConfig, default_config, load_config
from .experiment import aggregate_replications, run_replications
from .model import SimulationResult, simulate

__all__ = [
    "ModelConfig",
    "SimulationResult",
    "aggregate_replications",
    "capacity_interaction_analysis",
    "default_config",
    "load_config",
    "run_replications",
    "robustness_analysis",
    "sensitivity_analysis",
    "simulate",
]

__version__ = "0.2.0"
