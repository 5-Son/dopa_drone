"""
Compare DOPA against pymoo baselines (NSGA-III, NSGA-II w/ CDP).

Outputs:
  - Per-seed JSON with raw populations + CV for each algorithm.
  - Aggregated metrics + three comparison plots (HV, IGD+, feasibility/CV/runtime).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from datetime import datetime
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
try:
    from pymoo.core.callback import Callback
except Exception:  # pragma: no cover - fallback for older/newer pymoo
    class Callback:
        def __call__(self, algorithm):
            self.notify(algorithm)

        def notify(self, algorithm):
            pass
from pymoo.core.problem import Problem
from pymoo.indicators.hv import HV
from pymoo.indicators.igd_plus import IGDPlus
# Pymoo import paths differ slightly across versions; use small fallbacks.
try:
    from pymoo.operators.crossover.pntx import TwoPointCrossover
except Exception:  # pragma: no cover - fallback for older/newer pymoo
    from pymoo.operators.crossover.two_point_crossover import TwoPointCrossover
try:
    from pymoo.operators.mutation.bitflip import BitflipMutation
except Exception:  # pragma: no cover - fallback for older/newer pymoo
    from pymoo.operators.mutation.bitflip_mutation import BitflipMutation
try:
    from pymoo.operators.sampling.rnd import BinaryRandomSampling
except Exception:  # pragma: no cover - fallback for older/newer pymoo
    from pymoo.operators.sampling.binary import BinaryRandomSampling
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.ref_dirs import get_reference_directions

from dopa.config_loader import load_config, to_list
from dopa.metrics import compute_population_entropy, compute_wasserstein_distance
from dopa.problem_factory import make_problem
from dopa.scenarios import run_scenario


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%dT%H%M%S")


def _to_minimization(raw: np.ndarray) -> np.ndarray:
    """Convert to minimization space (pymoo assumes minimization)."""
    return np.column_stack([-raw[:, 0], raw[:, 1], raw[:, 2]])


def _feasible_min(raw: np.ndarray, cv: np.ndarray) -> np.ndarray:
    """Return feasible-only set in minimization space."""
    if raw.size == 0:
        return np.empty((0, 3))
    feasible_mask = np.asarray(cv) == 0
    if not np.any(feasible_mask):
        return np.empty((0, 3))
    return _to_minimization(raw[feasible_mask])


def _feasible_nd_min(raw: np.ndarray, cv: np.ndarray) -> np.ndarray:
    """Return feasible-only nondominated set in minimization space."""
    if raw.size == 0:
        return np.empty((0, 3))
    feasible_mask = np.asarray(cv) == 0
    if not np.any(feasible_mask):
        return np.empty((0, 3))
    f_min = _to_minimization(raw[feasible_mask])
    nd_idx = NonDominatedSorting().do(f_min, only_non_dominated_front=True)
    return f_min[nd_idx]


def _feasible_nd_raw(raw: np.ndarray, cv: np.ndarray) -> np.ndarray:
    """Return feasible-only nondominated set in raw (F1 max, F2/F3 min) space."""
    if raw.size == 0:
        return np.empty((0, 3))
    feasible_mask = np.asarray(cv) == 0
    if not np.any(feasible_mask):
        return np.empty((0, 3))
    raw_feasible = raw[feasible_mask]
    f_min = _to_minimization(raw_feasible)
    nd_idx = NonDominatedSorting().do(f_min, only_non_dominated_front=True)
    return raw_feasible[nd_idx]


class BaselineTraceCallback(Callback):
    """Collect entropy and Wasserstein traces on penalized objectives."""

    def __init__(self, eval_problem):
        super().__init__()
        self.eval_problem = eval_problem
        self.entropy_trace: List[float] = []
        self.wasserstein_trace: List[float] = []
        self._prev_fit = None

    def notify(self, algorithm):
        X = algorithm.pop.get("X")
        if X is None:
            return
        arr = np.asarray(X, dtype=int)
        fits = self.eval_problem.evaluate_with_penalty_batch(arr)
        self.entropy_trace.append(compute_population_entropy(fits))
        if self._prev_fit is None:
            w = 0.0
        else:
            w = compute_wasserstein_distance(self._prev_fit, fits)
        self.wasserstein_trace.append(w)
        self._prev_fit = fits


class UAVAssignmentPymooProblem(Problem):
    """Binary multi-objective UAV assignment with explicit constraints for CDP."""

    def __init__(self, *, d_ij, v_j, T_detect_ij, T_mission_i, D_max, C_total):
        self.N, self.M = d_ij.shape
        self.d_ij = d_ij
        self.v_j = v_j
        self.T_detect_ij = T_detect_ij
        self.T_mission_i = T_mission_i
        self.D_max = D_max
        self.C_total = C_total
        self._p_ij = np.exp(-0.005 * d_ij)
        self._invalid_mask = (d_ij > D_max).reshape(-1)
        # Constraints: 2N (sum==1 as <=0), 1 total, and invalid-distance bits
        n_constr = 2 * self.N + 1 + int(self._invalid_mask.sum())
        super().__init__(n_var=self.N * self.M, n_obj=3, n_constr=n_constr, xl=0, xu=1, vtype=bool)

    def _evaluate(self, X, out, *args, **kwargs):
        X = X.astype(int)
        I = X.reshape(-1, self.N, self.M)

        # Objectives (raw), then convert to minimization for pymoo
        F1 = np.sum(self.v_j * self._p_ij * I, axis=(1, 2))
        F2 = np.sum(self.T_detect_ij * I, axis=(1, 2))
        T_uav = np.sum(self.T_detect_ij * I, axis=2)
        F3 = np.max(T_uav + self.T_mission_i, axis=1)
        out["F"] = np.column_stack([-F1, F2, F3])

        # Constraints (<= 0)
        row_sum = np.sum(I, axis=2)
        g1 = row_sum - 1
        g2 = 1 - row_sum
        total_sum = np.sum(I, axis=(1, 2))
        g3 = (total_sum - self.C_total).reshape(-1, 1)
        if self._invalid_mask.any():
            flat = I.reshape(-1, self.N * self.M)
            g4 = flat[:, self._invalid_mask]
            out["G"] = np.hstack([g1, g2, g3, g4])
        else:
            out["G"] = np.hstack([g1, g2, g3])


def _run_pymoo(problem, alg_name: str, ngen: int, pop_size: int, seed: int, eval_problem=None, track_trace: bool = False):
    """Run a pymoo algorithm and return runtime + final population (+ traces if requested)."""
    random.seed(seed)
    np.random.seed(seed)

    if alg_name == "NSGA3":
        # pymoo API differs by version (n_obj vs n_dim). Try both for compatibility.
        try:
            ref_dirs = get_reference_directions("das-dennis", n_obj=3, n_partitions=19)
        except TypeError:
            ref_dirs = get_reference_directions("das-dennis", n_dim=3, n_partitions=19)
        algorithm = NSGA3(
            pop_size=pop_size,
            ref_dirs=ref_dirs,
            sampling=BinaryRandomSampling(),
            crossover=TwoPointCrossover(),
            mutation=BitflipMutation(),
            eliminate_duplicates=True,
        )
    elif alg_name == "NSGA2_CDP":
        algorithm = NSGA2(
            pop_size=pop_size,
            sampling=BinaryRandomSampling(),
            crossover=TwoPointCrossover(),
            mutation=BitflipMutation(),
            eliminate_duplicates=True,
        )
    else:
        raise ValueError(f"Unknown algorithm: {alg_name}")

    start = time.time()
    callback = None
    if track_trace and eval_problem is not None:
        callback = BaselineTraceCallback(eval_problem)
    res = minimize(problem, algorithm, ("n_gen", ngen), seed=seed, verbose=False, callback=callback)
    runtime = time.time() - start
    X = res.pop.get("X")
    traces = {}
    if callback is not None:
        traces = {
            "entropy_trace": callback.entropy_trace,
            "wasserstein_trace": callback.wasserstein_trace,
        }
    return runtime, np.asarray(X, dtype=int), traces


def _compute_metrics(per_alg: Dict[str, Dict[str, np.ndarray]], margin: float = 0.05):
    """Compute HV/IGD+ using pooled feasible set for a stable reference."""
    feasible_sets = []
    for data in per_alg.values():
        feasible_sets.append(_feasible_min(data["raw"], data["cv"]))
    combined = np.vstack(feasible_sets) if any(s.size > 0 for s in feasible_sets) else np.empty((0, 3))

    if combined.size == 0:
        ref_set = np.empty((0, 3))
        ref_point = None
    else:
        ref_set = combined
        # Pooled worst + margin; use range-based margin to handle negative values.
        maxs = ref_set.max(axis=0)
        mins = ref_set.min(axis=0)
        ref_point = maxs + margin * (maxs - mins + 1e-12)

    metrics = {}
    for name, data in per_alg.items():
        raw = data["raw"]
        cv = data["cv"]
        pop_size = len(cv)
        feasible_count = int(np.sum(cv == 0))
        feasible_rate = feasible_count / pop_size if pop_size else 0.0
        mean_cv = float(np.mean(cv)) if pop_size else 0.0

        S = _feasible_nd_min(raw, cv)
        if ref_set.size == 0 or S.size == 0:
            hv = 0.0
            igd = float("inf")
        else:
            hv = float(HV(ref_point=ref_point).do(S))
            igd = float(IGDPlus(ref_set).do(S))

        metrics[name] = {
            "hv": hv,
            "igd_plus": igd,
            "feasible_rate": feasible_rate,
            "num_feasible": feasible_count,
            "mean_cv": mean_cv,
        }
    return metrics


def _plot_pareto_3d(points_by_alg: Dict[str, np.ndarray], path: str):
    colors = {"DOPA": "tab:blue", "NSGA3": "tab:orange", "NSGA2_CDP": "tab:green"}
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    for name, points in points_by_alg.items():
        if points.size == 0:
            continue
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=15, alpha=0.6, label=name, color=colors.get(name))
    ax.set_xlabel("F1 (maximize)")
    ax.set_ylabel("F2 (minimize)")
    ax.set_zlabel("F3 (minimize)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_trace(traces_by_alg: Dict[str, List[List[float]]], ylabel: str, path: str):
    fig = plt.figure(figsize=(7, 4))
    for name, traces in traces_by_alg.items():
        traces = [t for t in traces if t]
        if not traces:
            continue
        min_len = min(len(t) for t in traces)
        if min_len == 0:
            continue
        arr = np.array([t[:min_len] for t in traces], dtype=float)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(mean)
        x = np.arange(min_len)
        plt.plot(x, mean, label=name)
        plt.fill_between(x, mean - std, mean + std, alpha=0.2)
    plt.xlabel("Generation")
    plt.ylabel(ylabel)
    plt.grid(True, axis="y", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close(fig)


def _boxplot(metrics_by_alg: Dict[str, List[float]], ylabel: str, path: str):
    labels = list(metrics_by_alg.keys())
    data = [metrics_by_alg[k] for k in labels]
    plt.figure(figsize=(7, 4))
    plt.boxplot(data, labels=labels, showmeans=True)
    plt.ylabel(ylabel)
    plt.grid(True, axis="y", alpha=0.4)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _barplot_summary(stats_by_alg: Dict[str, Dict[str, float]], path: str):
    labels = list(stats_by_alg.keys())
    x = np.arange(len(labels))

    feas = [stats_by_alg[k]["feasible_rate_mean"] for k in labels]
    feas_std = [stats_by_alg[k]["feasible_rate_std"] for k in labels]
    cv = [stats_by_alg[k]["mean_cv_mean"] for k in labels]
    cv_std = [stats_by_alg[k]["mean_cv_std"] for k in labels]
    runtime = [stats_by_alg[k]["runtime_mean"] for k in labels]
    runtime_std = [stats_by_alg[k]["runtime_std"] for k in labels]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].bar(x, feas, yerr=feas_std, capsize=4)
    axes[0].set_title("Feasible Rate")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=15)
    axes[0].grid(True, axis="y", alpha=0.4)

    axes[1].bar(x, cv, yerr=cv_std, capsize=4)
    axes[1].set_title("Mean Constraint Violation")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=15)
    axes[1].grid(True, axis="y", alpha=0.4)

    axes[2].bar(x, runtime, yerr=runtime_std, capsize=4)
    axes[2].set_title("Runtime [s]")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=15)
    axes[2].grid(True, axis="y", alpha=0.4)

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compare DOPA vs pymoo baselines.")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config.")
    parser.add_argument("--scenario", default="S4", help="DOPA scenario key (default: S4).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["scenarios"] = [args.scenario]

    # Baselines assume static environments; force static for a fair comparison.
    if cfg.get("env_type", "static") != "static":
        print("[!] Forcing env_type=static for baseline comparability.")
        cfg["env_type"] = "static"
    # Keep C_total aligned with num_uavs so sum==1 feasibility is possible.
    cfg["max_total_missions"] = cfg.get("num_uavs", 50)

    seeds = to_list(cfg.get("seeds", [0]))
    pop_size = int(cfg.get("pop_size", 210))
    ngen = int(cfg.get("generations", 381))

    out_root = os.path.join("results", f"compare_{_timestamp()}")
    json_dir = os.path.join(out_root, "json")
    plot_dir = os.path.join(out_root, "plot")
    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    per_seed_results = []
    hv_by_alg = {"DOPA": [], "NSGA3": [], "NSGA2_CDP": []}
    igd_by_alg = {"DOPA": [], "NSGA3": [], "NSGA2_CDP": []}
    feas_by_alg = {"DOPA": [], "NSGA3": [], "NSGA2_CDP": []}
    cv_by_alg = {"DOPA": [], "NSGA3": [], "NSGA2_CDP": []}
    runtime_by_alg = {"DOPA": [], "NSGA3": [], "NSGA2_CDP": []}
    pareto_points_by_alg = {"DOPA": [], "NSGA3": [], "NSGA2_CDP": []}
    entropy_traces_by_alg = {"DOPA": [], "NSGA3": [], "NSGA2_CDP": []}
    wasserstein_traces_by_alg = {"DOPA": [], "NSGA3": [], "NSGA2_CDP": []}

    for seed in seeds:
        problem = make_problem(cfg, seed)

        # DOPA run
        dopa_res = run_scenario(
            scenario_key=args.scenario,
            problem=problem,
            seed=seed,
            ngen=ngen,
            pop_size=pop_size,
            track_wasserstein=True,
        )

        # pymoo baselines use the same instance (arrays copied from problem)
        pymoo_problem = UAVAssignmentPymooProblem(
            d_ij=problem.d_ij,
            v_j=problem.v_j,
            T_detect_ij=problem.T_detect_ij,
            T_mission_i=problem.T_mission_i,
            D_max=problem.D_max,
            C_total=problem.C_total,
        )

        nsga3_time, nsga3_X, nsga3_traces = _run_pymoo(
            pymoo_problem, "NSGA3", ngen, pop_size, seed, eval_problem=problem, track_trace=True
        )
        nsga2_time, nsga2_X, nsga2_traces = _run_pymoo(
            pymoo_problem, "NSGA2_CDP", ngen, pop_size, seed, eval_problem=problem, track_trace=True
        )

        # Compute raw objectives + CV for baselines using the same logic as DOPA.
        nsga3_raw = problem.evaluate_objectives_batch(nsga3_X)
        nsga3_cv = problem.constraint_violation_score_batch(nsga3_X)
        nsga2_raw = problem.evaluate_objectives_batch(nsga2_X)
        nsga2_cv = problem.constraint_violation_score_batch(nsga2_X)

        per_alg = {
            "DOPA": {
                "raw": np.asarray(dopa_res.get("final_population_raw", []), dtype=float),
                "cv": np.asarray(dopa_res.get("final_population_cv", []), dtype=float),
                "runtime": float(dopa_res.get("execution_time", 0.0)),
            },
            "NSGA3": {"raw": nsga3_raw, "cv": nsga3_cv, "runtime": nsga3_time},
            "NSGA2_CDP": {"raw": nsga2_raw, "cv": nsga2_cv, "runtime": nsga2_time},
        }

        metrics = _compute_metrics(per_alg, margin=0.05)

        for name in ["DOPA", "NSGA3", "NSGA2_CDP"]:
            pareto_points_by_alg[name].append(_feasible_nd_raw(per_alg[name]["raw"], per_alg[name]["cv"]))

        if dopa_res.get("entropy_trace"):
            entropy_traces_by_alg["DOPA"].append(dopa_res["entropy_trace"])
        if dopa_res.get("wasserstein_trace"):
            wasserstein_traces_by_alg["DOPA"].append(dopa_res["wasserstein_trace"])
        if nsga3_traces.get("entropy_trace"):
            entropy_traces_by_alg["NSGA3"].append(nsga3_traces["entropy_trace"])
        if nsga3_traces.get("wasserstein_trace"):
            wasserstein_traces_by_alg["NSGA3"].append(nsga3_traces["wasserstein_trace"])
        if nsga2_traces.get("entropy_trace"):
            entropy_traces_by_alg["NSGA2_CDP"].append(nsga2_traces["entropy_trace"])
        if nsga2_traces.get("wasserstein_trace"):
            wasserstein_traces_by_alg["NSGA2_CDP"].append(nsga2_traces["wasserstein_trace"])

        # Aggregate metrics for plots.
        for name in ["DOPA", "NSGA3", "NSGA2_CDP"]:
            hv_by_alg[name].append(metrics[name]["hv"])
            igd_by_alg[name].append(metrics[name]["igd_plus"])
            feas_by_alg[name].append(metrics[name]["feasible_rate"])
            cv_by_alg[name].append(metrics[name]["mean_cv"])
            runtime_by_alg[name].append(per_alg[name]["runtime"])

        # Save per-seed JSON (includes raw populations + CV).
        seed_payload = {
            "seed": seed,
            "pop_size": pop_size,
            "generations": ngen,
            "algorithms": {
                "DOPA": {
                    "final_population_raw": per_alg["DOPA"]["raw"].tolist(),
                    "final_population_cv": per_alg["DOPA"]["cv"].tolist(),
                    "metrics": metrics["DOPA"],
                    "runtime": per_alg["DOPA"]["runtime"],
                    "entropy_trace": dopa_res.get("entropy_trace"),
                    "wasserstein_trace": dopa_res.get("wasserstein_trace"),
                },
                "NSGA3": {
                    "final_population_raw": per_alg["NSGA3"]["raw"].tolist(),
                    "final_population_cv": per_alg["NSGA3"]["cv"].tolist(),
                    "metrics": metrics["NSGA3"],
                    "runtime": per_alg["NSGA3"]["runtime"],
                    "entropy_trace": nsga3_traces.get("entropy_trace"),
                    "wasserstein_trace": nsga3_traces.get("wasserstein_trace"),
                },
                "NSGA2_CDP": {
                    "final_population_raw": per_alg["NSGA2_CDP"]["raw"].tolist(),
                    "final_population_cv": per_alg["NSGA2_CDP"]["cv"].tolist(),
                    "metrics": metrics["NSGA2_CDP"],
                    "runtime": per_alg["NSGA2_CDP"]["runtime"],
                    "entropy_trace": nsga2_traces.get("entropy_trace"),
                    "wasserstein_trace": nsga2_traces.get("wasserstein_trace"),
                },
            },
        }
        per_seed_results.append(seed_payload)
        with open(os.path.join(json_dir, f"compare_seed{seed}.json"), "w") as f:
            json.dump(seed_payload, f, indent=2)

    # Aggregate summary stats for bars.
    summary = {}
    for name in ["DOPA", "NSGA3", "NSGA2_CDP"]:
        summary[name] = {
            "hv_mean": float(np.mean(hv_by_alg[name])),
            "hv_std": float(np.std(hv_by_alg[name], ddof=1)) if len(hv_by_alg[name]) > 1 else 0.0,
            "igd_plus_mean": float(np.mean(igd_by_alg[name])),
            "igd_plus_std": float(np.std(igd_by_alg[name], ddof=1)) if len(igd_by_alg[name]) > 1 else 0.0,
            "feasible_rate_mean": float(np.mean(feas_by_alg[name])),
            "feasible_rate_std": float(np.std(feas_by_alg[name], ddof=1)) if len(feas_by_alg[name]) > 1 else 0.0,
            "mean_cv_mean": float(np.mean(cv_by_alg[name])),
            "mean_cv_std": float(np.std(cv_by_alg[name], ddof=1)) if len(cv_by_alg[name]) > 1 else 0.0,
            "runtime_mean": float(np.mean(runtime_by_alg[name])),
            "runtime_std": float(np.std(runtime_by_alg[name], ddof=1)) if len(runtime_by_alg[name]) > 1 else 0.0,
        }

    with open(os.path.join(json_dir, "summary.json"), "w") as f:
        json.dump({"summary": summary, "seeds": seeds}, f, indent=2)

    pareto_points = {}
    for name, points in pareto_points_by_alg.items():
        if points:
            pareto_points[name] = np.vstack(points) if any(p.size > 0 for p in points) else np.empty((0, 3))
        else:
            pareto_points[name] = np.empty((0, 3))

    # Plot 1: HV distribution
    _boxplot(hv_by_alg, "Hypervolume (HV)", os.path.join(plot_dir, "hv_boxplot.png"))
    # Plot 2: IGD+ distribution
    _boxplot(igd_by_alg, "IGD+", os.path.join(plot_dir, "igd_plus_boxplot.png"))
    # Plot 3: Feasibility/CV/Runtime bars
    _barplot_summary(summary, os.path.join(plot_dir, "feasibility_cv_runtime.png"))
    # Plot 4: Combined Pareto front (3D)
    _plot_pareto_3d(pareto_points, os.path.join(plot_dir, "pareto_3d.png"))
    # Plot 5: Wasserstein-1 trace
    _plot_trace(wasserstein_traces_by_alg, "Wasserstein-1 Distance", os.path.join(plot_dir, "wasserstein_trace.png"))
    # Plot 6: Population entropy trace
    _plot_trace(entropy_traces_by_alg, "Population Entropy", os.path.join(plot_dir, "entropy_trace.png"))

    print(f"Comparison outputs saved under: {out_root}")


if __name__ == "__main__":
    main()
