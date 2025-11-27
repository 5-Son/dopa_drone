"""YAML-based configuration loader with sensible defaults and sweep helpers."""
from __future__ import annotations

import copy
import itertools
import os
from typing import Dict, Iterable, List

import yaml

DEFAULT_CONFIG = {
    "switching_scheme": "dopa_entropy",
    "num_uavs": 50,
    "num_targets": 20,
    "generations": 400,
    "pop_size": 200,
    "seeds": [0, 1, 2, 3, 4],
    "scenarios": ["S1", "S2", "S3", "S4"],
    "env_type": "static",
    "dynamic_mode": "dynamic_targets",
    "enable_plot": False,
    "pareto_plot_type": "2d",
    "pareto_axes": [0, 1],
    "plot_metrics": ["entropy_trace", "wasserstein_trace", "cxpb_trace", "mutpb_trace"],
    "mode": "normal",
    "time_complexity": {
        "num_uavs": [50, 100, 150],
        "num_targets": [20, 40, 60],
        "generations": [200],
        "pop_size": [100],
        "switching_schemes": ["dopa_entropy", "sa_temp"],
        "seeds": [0],
        "scenarios": ["S4"],
    },
}


def load_config(path: str | None = None) -> Dict:
    """Load YAML config; fall back to defaults if file is missing."""
    if path is None:
        path = "config.yaml"
    if not os.path.exists(path):
        return copy.deepcopy(DEFAULT_CONFIG)
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg.update({k: v for k, v in data.items() if k != "time_complexity"})
    if "time_complexity" in data:
        tc = copy.deepcopy(DEFAULT_CONFIG["time_complexity"])
        tc.update(data["time_complexity"] or {})
        cfg["time_complexity"] = tc
    return cfg


def to_list(value) -> List:
    if isinstance(value, list):
        return value
    return [value]


def expand_sweeps(base_cfg: Dict) -> List[Dict]:
    """
    Expand combinations for normal mode. Keys that are lists will be combined via cartesian product.
    """
    sweep_keys = ["switching_scheme", "num_uavs", "num_targets", "generations", "pop_size", "env_type", "dynamic_mode"]
    sweep_values = []
    for k in sweep_keys:
        v = base_cfg.get(k)
        sweep_values.append(to_list(v))
    products = itertools.product(*sweep_values)
    configs = []
    for combo in products:
        cfg = copy.deepcopy(base_cfg)
        for key, val in zip(sweep_keys, combo):
            cfg[key] = val
        configs.append(cfg)
    return configs
