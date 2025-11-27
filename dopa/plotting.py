"""Plotting utilities for traces, Pareto fronts, and time-complexity curves."""
from __future__ import annotations

import os
from datetime import datetime
from typing import Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import numpy as np


def _timestamp():
    return datetime.now().strftime("%Y%m%dT%H%M%S")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_trace_plot(
    trace: Sequence[float],
    metric_name: str,
    scenario: str,
    seed: int,
    switching_scheme: str,
    output_dir: str,
):
    if trace is None or len(trace) == 0:
        return None
    ensure_dir(output_dir)
    ts = _timestamp()
    fname = f"{scenario}_seed{seed}_{metric_name}_{switching_scheme}_{ts}.png"
    path = os.path.join(output_dir, fname)
    plt.figure(figsize=(7, 4))
    plt.plot(trace)
    plt.xlabel("Generation")
    plt.ylabel(metric_name)
    plt.title(f"{scenario} {metric_name}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def save_pareto_plot(
    pareto: Iterable[Sequence[float]],
    scenario: str,
    seed: int,
    switching_scheme: str,
    output_dir: str,
    plot_type: str = "2d",
    axes: Sequence[int] = (0, 1),
):
    arr = np.asarray(list(pareto), dtype=float)
    if arr.size == 0:
        return None
    ensure_dir(output_dir)
    ts = _timestamp()
    if plot_type == "3d":
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(arr[:, 0], arr[:, 1], arr[:, 2], s=20)
        ax.set_xlabel("F1")
        ax.set_ylabel("F2")
        ax.set_zlabel("F3")
        fname = f"{scenario}_seed{seed}_pareto_3d_{switching_scheme}_{ts}.png"
    else:
        a, b = axes if len(axes) >= 2 else (0, 1)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(arr[:, a], arr[:, b], s=20)
        ax.set_xlabel(f"F{a+1}")
        ax.set_ylabel(f"F{b+1}")
        fname = f"{scenario}_seed{seed}_pareto_2d_F{a+1}_F{b+1}_{switching_scheme}_{ts}.png"
    path = os.path.join(output_dir, fname)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def save_time_complexity_plot(
    logs: List[dict],
    output_dir: str,
    x_key: str = "num_uavs",
    title: str = "Runtime vs Problem Size",
):
    if not logs:
        return None
    ensure_dir(output_dir)
    ts = _timestamp()
    fname = f"time_complexity_{x_key}_{ts}.png"
    path = os.path.join(output_dir, fname)

    # Group by switching scheme
    schemes = sorted({log.get("switching_scheme", "unknown") for log in logs})
    plt.figure(figsize=(7, 5))
    for scheme in schemes:
        xs = []
        ys = []
        for log in logs:
            if log.get("switching_scheme", "unknown") != scheme:
                continue
            if x_key in log and "execution_time" in log:
                xs.append(log[x_key])
                ys.append(log["execution_time"])
        if xs and ys:
            idx = np.argsort(xs)
            xs_sorted = np.array(xs)[idx]
            ys_sorted = np.array(ys)[idx]
            plt.plot(xs_sorted, ys_sorted, marker="o", label=scheme)
    plt.xlabel(x_key)
    plt.ylabel("Runtime [s]")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    return path
