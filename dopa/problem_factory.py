"""Shared problem-instance factory for DOPA and baseline comparisons."""
from __future__ import annotations

from typing import Dict

import numpy as np

from .problem import DynamicUAVTargetAssignmentProblem, UAVTargetAssignmentProblem
from .utils import set_seed


def make_problem(cfg: Dict, seed: int):
    """
    Create a (static or dynamic) problem instance with deterministic RNG.
    This is shared so DOPA and baselines see the exact same instance per seed.
    """
    set_seed(seed)

    num_uavs = int(cfg.get("num_uavs", 50))
    num_targets = int(cfg.get("num_targets", 20))
    max_distance = cfg.get("max_distance", 400)
    # Default to num_uavs so the per-UAV sum==1 constraint is feasible.
    max_total_missions = cfg.get("max_total_missions", num_uavs)
    penalty_scale = cfg.get("penalty_scale", 0.1)
    device = cfg.get("device", None)

    v_j = np.random.uniform(0.8, 1.2, size=num_targets)
    d_ij = np.random.uniform(50, 500, size=(num_uavs, num_targets))
    T_detect = np.random.uniform(5, 20, size=(num_uavs, num_targets))
    T_mission = np.random.uniform(30, 60, size=num_uavs)

    env_type = cfg.get("env_type", "static")
    dynamic_mode = cfg.get("dynamic_mode", "dynamic_targets")

    base_kwargs = dict(
        N=num_uavs,
        M=num_targets,
        d_ij=d_ij,
        v_j=v_j,
        T_detect_ij=T_detect,
        T_mission_i=T_mission,
        D_max=max_distance,
        C_total=max_total_missions,
        penalty_scale=penalty_scale,
        device=device,
    )

    if env_type == "dynamic":
        problem = DynamicUAVTargetAssignmentProblem(
            **base_kwargs, env_type=env_type, dynamic_mode=dynamic_mode, dynamic_seed=seed
        )
    else:
        problem = UAVTargetAssignmentProblem(**base_kwargs)
        problem.env_type = env_type
        problem.dynamic_mode = dynamic_mode

    # Pass scheme/device through for compatibility with run_scenario.
    problem.switching_scheme = cfg.get("switching_scheme", "dopa_entropy")
    problem.device = device
    return problem
