#!/usr/bin/env python3
"""
Standalone plotting utility for compare_baselines.py outputs.

Reads JSON files under <timestamp>/json/ and writes plots to <timestamp>/plot/.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import numpy as np


def find_json_files(json_dir: str) -> List[str]:
    if not os.path.isdir(json_dir):
        raise FileNotFoundError(f"JSON directory not found: {json_dir}")
    matches: List[str] = []
    for root, _, files in os.walk(json_dir):
        for name in files:
            if name.lower().endswith(".json"):
                matches.append(os.path.join(root, name))
    return sorted(matches)


def load_runs(file_paths: List[str]) -> Tuple[List[Dict], Dict, List[str]]:
    runs: List[Dict] = []
    summaries: Dict[str, Dict] = {}
    warnings: List[str] = []
    for path in file_paths:
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except Exception as exc:
            warnings.append(f"failed to load {path}: {exc}")
            continue
        if not isinstance(data, dict):
            warnings.append(f"skipping non-object JSON: {path}")
            continue
        if "algorithms" in data and isinstance(data["algorithms"], dict):
            scenario = (
                data.get("scenario")
                or data.get("scenario_key")
                or data.get("scenario_name")
                or data.get("scenarios")
            )
            if isinstance(scenario, list):
                scenario = scenario[0] if scenario else None
            if scenario is None:
                scenario = "unknown"
            run = {
                "file": path,
                "seed": data.get("seed"),
                "scenario": scenario,
                "pop_size": data.get("pop_size"),
                "generations": data.get("generations"),
                "algorithms": data["algorithms"],
            }
            runs.append(run)
        elif "summary" in data:
            summaries[path] = data
        else:
            warnings.append(f"skipping JSON without algorithms: {path}")
    return runs, summaries, warnings


def group_runs(runs: List[Dict]) -> Tuple[Dict[str, List[Dict]], List[str], List[str], List[int]]:
    grouped: Dict[str, List[Dict]] = {}
    algorithms: set[str] = set()
    scenarios: set[str] = set()
    seeds: set[int] = set()
    for run in runs:
        scenario = str(run.get("scenario", "unknown"))
        grouped.setdefault(scenario, []).append(run)
        scenarios.add(scenario)
        seed = run.get("seed")
        if isinstance(seed, int):
            seeds.add(seed)
        algs = run.get("algorithms", {})
        if isinstance(algs, dict):
            algorithms.update(algs.keys())
    return grouped, sorted(algorithms), sorted(scenarios), sorted(seeds)


def aggregate_traces(traces: List[List[float]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    # Align by truncating to the shortest length to match compare_baselines.py behavior.
    traces = [t for t in traces if t]
    if not traces:
        return None
    min_len = min(len(t) for t in traces)
    if min_len <= 0:
        return None
    arr = np.array([t[:min_len] for t in traces], dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(mean)
    x = np.arange(min_len)
    return x, mean, std


def plot_metric(
    ax: plt.Axes,
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    label: str,
):
    ax.plot(x, mean, label=label)
    ax.fill_between(x, mean - std, mean + std, alpha=0.2)


def save_fig(fig: plt.Figure, path: str, pdf: bool):
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    if pdf:
        pdf_path = os.path.splitext(path)[0] + ".pdf"
        fig.savefig(pdf_path)
    plt.close(fig)


def _safe_mean(values: List[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def _safe_std(values: List[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return 0.0
    return float(np.std(arr, ddof=1))


def _finite_only(values: List[float]) -> List[float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return [float("nan")]
    return arr.tolist()


def _to_minimization(raw: np.ndarray) -> np.ndarray:
    return np.column_stack([-raw[:, 0], raw[:, 1], raw[:, 2]])


def _nondominated_indices(f_min: np.ndarray) -> np.ndarray:
    n = f_min.shape[0]
    if n == 0:
        return np.array([], dtype=int)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        if dominated[i]:
            continue
        for j in range(n):
            if i == j or dominated[i]:
                continue
            if np.all(f_min[j] <= f_min[i]) and np.any(f_min[j] < f_min[i]):
                dominated[i] = True
    return np.where(~dominated)[0]


def _feasible_nd_raw(raw: np.ndarray, cv: np.ndarray) -> np.ndarray:
    if raw.size == 0:
        return np.empty((0, 3))
    cv = np.asarray(cv, dtype=float)
    feasible_mask = cv == 0
    if not np.any(feasible_mask):
        return np.empty((0, 3))
    raw_feasible = raw[feasible_mask]
    f_min = _to_minimization(raw_feasible)
    nd_idx = _nondominated_indices(f_min)
    return raw_feasible[nd_idx]


def _plot_trace(
    traces_by_alg: Dict[str, List[List[float]]],
    ylabel: str,
    title: str,
    out_path: str,
    pdf: bool,
) -> bool:
    fig, ax = plt.subplots(figsize=(7, 4))
    plotted = False
    for name, traces in traces_by_alg.items():
        agg = aggregate_traces(traces)
        if agg is None:
            continue
        x, mean, std = agg
        plot_metric(ax, x, mean, std, label=name)
        plotted = True
    if not plotted:
        plt.close(fig)
        return False
    ax.set_xlabel("Generation")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.4)
    ax.legend()
    save_fig(fig, out_path, pdf)
    return True


def _plot_boxplot(
    data_by_alg: Dict[str, List[float]],
    ylabel: str,
    title: str,
    out_path: str,
    pdf: bool,
) -> bool:
    labels = list(data_by_alg.keys())
    data = [_finite_only(data_by_alg[k]) for k in labels]
    fig = plt.figure(figsize=(7, 4))
    try:
        plt.boxplot(data, tick_labels=labels, showmeans=True)
    except TypeError:  # pragma: no cover - older Matplotlib
        plt.boxplot(data, labels=labels, showmeans=True)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.4)
    save_fig(fig, out_path, pdf)
    return True


def _plot_bar_summary(
    stats_by_alg: Dict[str, Dict[str, float]],
    title: str,
    out_path: str,
    pdf: bool,
) -> bool:
    labels = list(stats_by_alg.keys())
    if not labels:
        return False
    x = np.arange(len(labels))

    feas = [stats_by_alg[k]["feasible_rate_mean"] for k in labels]
    feas_std = [stats_by_alg[k]["feasible_rate_std"] for k in labels]
    cv = [stats_by_alg[k]["mean_cv_mean"] for k in labels]
    cv_std = [stats_by_alg[k]["mean_cv_std"] for k in labels]
    runtime = [stats_by_alg[k]["runtime_mean"] for k in labels]
    runtime_std = [stats_by_alg[k]["runtime_std"] for k in labels]
    evals = [stats_by_alg[k]["eval_mean"] for k in labels]
    evals_std = [stats_by_alg[k]["eval_std"] for k in labels]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

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

    axes[3].bar(x, evals, yerr=evals_std, capsize=4)
    axes[3].set_title("Evaluation Budget")
    axes[3].set_xticks(x)
    axes[3].set_xticklabels(labels, rotation=15)
    axes[3].grid(True, axis="y", alpha=0.4)

    fig.suptitle(title)
    save_fig(fig, out_path, pdf)
    return True


def _plot_pareto_3d(
    points_by_alg: Dict[str, np.ndarray],
    title: str,
    out_path: str,
    pdf: bool,
) -> bool:
    colors = {"DOPA": "tab:blue", "NSGA3": "tab:orange", "NSGA2_CDP": "tab:green"}
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    plotted = False
    for name, points in points_by_alg.items():
        if points.size == 0:
            continue
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=15, alpha=0.6, label=name, color=colors.get(name))
        plotted = True
    if not plotted:
        plt.close(fig)
        return False
    ax.set_xlabel("F1 (maximize)")
    ax.set_ylabel("F2 (minimize)")
    ax.set_zlabel("F3 (minimize)")
    ax.set_title(title)
    ax.legend()
    save_fig(fig, out_path, pdf)
    return True


def _sanitize_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)


def _plot_entropy_per_algorithm(
    grouped: Dict[str, List[Dict]],
    algorithms: List[str],
    timestamp: str,
    out_dir: str,
    pdf: bool,
) -> List[str]:
    generated: List[str] = []
    scenarios = sorted(grouped.keys())
    for algo in algorithms:
        fig, axes = plt.subplots(len(scenarios), 1, figsize=(7, 4 * len(scenarios)), squeeze=False)
        plotted_any = False
        for idx, scenario in enumerate(scenarios):
            traces: List[List[float]] = []
            for run in grouped[scenario]:
                alg = run["algorithms"].get(algo, {})
                trace = alg.get("entropy_trace")
                if trace:
                    traces.append(trace)
            agg = aggregate_traces(traces)
            ax = axes[idx][0]
            if agg is None:
                ax.set_title(f"{algo} - {scenario} (no data)")
                ax.axis("off")
                continue
            x, mean, std = agg
            plot_metric(ax, x, mean, std, label=scenario)
            ax.set_xlabel("Generation")
            ax.set_ylabel("Population Entropy")
            ax.set_title(f"{algo} - {scenario}")
            ax.grid(True, axis="y", alpha=0.4)
            plotted_any = True
        if not plotted_any:
            plt.close(fig)
            continue
        fig.suptitle(f"Population Entropy by Scenario ({algo}) - {timestamp}")
        out_path = os.path.join(out_dir, f"entropy_algo_{_sanitize_name(algo)}.png")
        save_fig(fig, out_path, pdf)
        generated.append(out_path)
    return generated


def _write_summary(
    out_dir: str,
    timestamp: str,
    file_count: int,
    algorithms: List[str],
    scenarios: List[str],
    seeds: List[int],
    plots: List[str],
    warnings: List[str],
):
    lines = [
        f"timestamp: {timestamp}",
        f"json_files_loaded: {file_count}",
        f"algorithms: {', '.join(algorithms) if algorithms else 'none'}",
        f"scenarios: {', '.join(scenarios) if scenarios else 'none'}",
        f"seeds: {', '.join(str(s) for s in seeds) if seeds else 'none'}",
        "plots:",
    ]
    for plot in plots:
        lines.append(f"  - {os.path.basename(plot)}")
    if warnings:
        lines.append("warnings:")
        for warning in warnings:
            lines.append(f"  - {warning}")
    summary_path = os.path.join(out_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot compare_baselines results from JSON outputs.")
    parser.add_argument("--timestamp", required=True, help="Timestamp folder name.")
    parser.add_argument("--base_dir", default="results", help="Base directory containing timestamp folders.")
    parser.add_argument("--pdf", action="store_true", help="Also write PDF outputs.")
    parser.add_argument("--no_entropy", action="store_true", help="Skip entropy plots.")
    args = parser.parse_args()

    run_dir = os.path.join(args.base_dir, args.timestamp)
    if not os.path.isdir(run_dir) and args.base_dir == "results":
        alt_dir = os.path.join("result", args.timestamp)
        if os.path.isdir(alt_dir):
            run_dir = alt_dir
    json_dir = os.path.join(run_dir, "json")
    plot_dir = os.path.join(run_dir, "plot")
    os.makedirs(plot_dir, exist_ok=True)

    json_files = find_json_files(json_dir)
    if not json_files:
        print(f"No JSON files found under: {json_dir}", file=sys.stderr)
        return 1

    runs, summaries, warnings = load_runs(json_files)
    if not runs:
        print("No per-seed JSON files with 'algorithms' found.", file=sys.stderr)
        return 1

    grouped, algorithms, scenarios, seeds = group_runs(runs)

    missing_entropy: List[str] = []
    if not args.no_entropy:
        for run in runs:
            for algo, payload in run["algorithms"].items():
                if "entropy_trace" not in payload or payload.get("entropy_trace") is None:
                    missing_entropy.append(f"{run['file']} (algo={algo})")
        if missing_entropy:
            print(
                "Missing entropy_trace in JSON files:\n  "
                + "\n  ".join(missing_entropy)
                + "\nUse --no_entropy to skip entropy plots.",
                file=sys.stderr,
            )
            return 1

    hv_by_alg: Dict[str, List[float]] = {a: [] for a in algorithms}
    igd_by_alg: Dict[str, List[float]] = {a: [] for a in algorithms}
    feas_by_alg: Dict[str, List[float]] = {a: [] for a in algorithms}
    cv_by_alg: Dict[str, List[float]] = {a: [] for a in algorithms}
    runtime_by_alg: Dict[str, List[float]] = {a: [] for a in algorithms}
    evals_by_alg: Dict[str, List[float]] = {a: [] for a in algorithms}
    pareto_points_by_alg: Dict[str, List[np.ndarray]] = {a: [] for a in algorithms}
    entropy_traces_by_alg: Dict[str, List[List[float]]] = {a: [] for a in algorithms}
    wasserstein_traces_by_alg: Dict[str, List[List[float]]] = {a: [] for a in algorithms}

    for run in runs:
        for algo, payload in run["algorithms"].items():
            metrics = payload.get("metrics", {})
            if "hv" in metrics:
                hv_by_alg[algo].append(metrics["hv"])
            else:
                warnings.append(f"missing metrics.hv in {run['file']} (algo={algo})")
            if "igd_plus" in metrics:
                igd_by_alg[algo].append(metrics["igd_plus"])
            else:
                warnings.append(f"missing metrics.igd_plus in {run['file']} (algo={algo})")
            if "feasible_rate" in metrics:
                feas_by_alg[algo].append(metrics["feasible_rate"])
            else:
                warnings.append(f"missing metrics.feasible_rate in {run['file']} (algo={algo})")
            if "mean_cv" in metrics:
                cv_by_alg[algo].append(metrics["mean_cv"])
            else:
                warnings.append(f"missing metrics.mean_cv in {run['file']} (algo={algo})")

            runtime = payload.get("runtime")
            if runtime is not None:
                runtime_by_alg[algo].append(runtime)
            else:
                warnings.append(f"missing runtime in {run['file']} (algo={algo})")

            evals = payload.get("num_evaluations")
            if evals is not None:
                evals_by_alg[algo].append(evals)
            else:
                warnings.append(f"missing num_evaluations in {run['file']} (algo={algo})")

            raw = payload.get("final_population_raw")
            cv = payload.get("final_population_cv")
            if raw is not None and cv is not None:
                raw_arr = np.asarray(raw, dtype=float)
                cv_arr = np.asarray(cv, dtype=float)
                pareto_points_by_alg[algo].append(_feasible_nd_raw(raw_arr, cv_arr))
            else:
                warnings.append(f"missing population raw/cv in {run['file']} (algo={algo})")

            entropy = payload.get("entropy_trace")
            if entropy:
                entropy_traces_by_alg[algo].append(entropy)
            wasser = payload.get("wasserstein_trace")
            if wasser:
                wasserstein_traces_by_alg[algo].append(wasser)

    stats_by_alg: Dict[str, Dict[str, float]] = {}
    for algo in algorithms:
        stats_by_alg[algo] = {
            "feasible_rate_mean": _safe_mean(feas_by_alg[algo]),
            "feasible_rate_std": _safe_std(feas_by_alg[algo]),
            "mean_cv_mean": _safe_mean(cv_by_alg[algo]),
            "mean_cv_std": _safe_std(cv_by_alg[algo]),
            "runtime_mean": _safe_mean(runtime_by_alg[algo]),
            "runtime_std": _safe_std(runtime_by_alg[algo]),
            "eval_mean": _safe_mean(evals_by_alg[algo]),
            "eval_std": _safe_std(evals_by_alg[algo]),
        }

    plots: List[str] = []
    timestamp = args.timestamp

    if _plot_boxplot(
        hv_by_alg,
        "Hypervolume (HV)",
        f"HV Distribution - {timestamp}",
        os.path.join(plot_dir, "hv_boxplot.png"),
        args.pdf,
    ):
        plots.append(os.path.join(plot_dir, "hv_boxplot.png"))

    if _plot_boxplot(
        igd_by_alg,
        "IGD+",
        f"IGD+ Distribution - {timestamp}",
        os.path.join(plot_dir, "igd_plus_boxplot.png"),
        args.pdf,
    ):
        plots.append(os.path.join(plot_dir, "igd_plus_boxplot.png"))

    if _plot_bar_summary(
        stats_by_alg,
        f"Feasibility/CV/Runtime/Evals - {timestamp}",
        os.path.join(plot_dir, "feasibility_cv_runtime.png"),
        args.pdf,
    ):
        plots.append(os.path.join(plot_dir, "feasibility_cv_runtime.png"))

    pareto_points = {}
    for name, points in pareto_points_by_alg.items():
        if points:
            pareto_points[name] = np.vstack(points) if any(p.size > 0 for p in points) else np.empty((0, 3))
        else:
            pareto_points[name] = np.empty((0, 3))
    if _plot_pareto_3d(
        pareto_points,
        f"Combined Pareto Front - {timestamp}",
        os.path.join(plot_dir, "pareto_3d.png"),
        args.pdf,
    ):
        plots.append(os.path.join(plot_dir, "pareto_3d.png"))

    if _plot_trace(
        wasserstein_traces_by_alg,
        "Wasserstein-1 Distance",
        f"Wasserstein Trace - {timestamp}",
        os.path.join(plot_dir, "wasserstein_trace.png"),
        args.pdf,
    ):
        plots.append(os.path.join(plot_dir, "wasserstein_trace.png"))
    else:
        warnings.append("skipped wasserstein_trace plot (no traces found)")

    if _plot_trace(
        entropy_traces_by_alg,
        "Population Entropy",
        f"Entropy Trace - {timestamp}",
        os.path.join(plot_dir, "entropy_trace.png"),
        args.pdf,
    ):
        plots.append(os.path.join(plot_dir, "entropy_trace.png"))
    else:
        warnings.append("skipped entropy_trace plot (no traces found)")

    if not args.no_entropy:
        entropy_plots = _plot_entropy_per_algorithm(
            grouped, algorithms, timestamp=timestamp, out_dir=plot_dir, pdf=args.pdf
        )
        plots.extend(entropy_plots)
        if not entropy_plots:
            warnings.append("skipped per-algorithm entropy plots (no traces found)")

    _write_summary(
        plot_dir,
        timestamp=timestamp,
        file_count=len(json_files),
        algorithms=algorithms,
        scenarios=scenarios,
        seeds=seeds,
        plots=plots,
        warnings=warnings,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
