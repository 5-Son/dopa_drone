# run.py
"""
Enhanced DOPA runner with YAML config, sweep support, dynamic environments,
and optional plotting/time-complexity experiments.

Usage examples:
    python run.py                          # default behavior (same as original)
    python run.py --config config.yaml     # custom YAML
    python run.py --mode time_complexity   # run scaling experiments
    python run.py --plot                   # enable plotting from CLI
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from datetime import datetime
from typing import Dict, List

import numpy as np
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None

from dopa.problem_factory import make_problem
from dopa.scenarios import run_scenario
from dopa.config_loader import expand_sweeps, load_config, to_list
from dopa.plotting import save_pareto_plot, save_time_complexity_plot, save_trace_plot

RESULT_ROOT = "results"
_TIMEFRAME = datetime.now().strftime("%Y%m%dT%H%M%S")
RUN_DIR = os.path.join(RESULT_ROOT, _TIMEFRAME)
JSON_DIR = os.path.join(RUN_DIR, "json")
PLOT_DIR = os.path.join(RUN_DIR, "plot")
os.makedirs(JSON_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

_PROGRESS_ENABLED = False


class _NoOpProgress:
    def __init__(self, total=None):
        self.total = total
        self.n = 0

    def update(self, n=1):
        self.n += n

    def set_postfix(self, *args, **kwargs):
        pass

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def _set_progress_enabled(enabled: bool):
    global _PROGRESS_ENABLED
    _PROGRESS_ENABLED = enabled


def _progress_write(msg: str):
    if tqdm is not None and _PROGRESS_ENABLED:
        tqdm.write(msg)
    else:
        print(msg)


def _get_progress(total, desc, enabled: bool, leave: bool = True, position: int | None = None):
    if tqdm is None or not enabled:
        return _NoOpProgress(total=total)
    return tqdm(total=total, desc=desc, leave=leave, position=position, dynamic_ncols=True)


def _safe_update(bar, delta: int):
    if delta <= 0:
        return
    total = getattr(bar, "total", None)
    if total is None:
        bar.update(delta)
        return
    remaining = total - getattr(bar, "n", 0)
    if remaining <= 0:
        return
    bar.update(min(delta, remaining))


def _resolve_progress_enabled(args) -> bool:
    if args.no_progress or os.getenv("DOPA_NO_PROGRESS") == "1":
        return False
    if args.progress or os.getenv("DOPA_PROGRESS") == "1":
        return True
    return sys.stdout.isatty()


def _make_gen_progress(
    *,
    ngen: int,
    pop_size: int,
    seed: int,
    scenario: str,
    env_type: str,
    algo: str,
    enabled: bool,
):
    bar = _get_progress(ngen, desc=f"{algo} gen", enabled=enabled, leave=False, position=1)
    total_evals = pop_size * ngen if pop_size and ngen else None

    def progress_cb(gen=None, eval_count=None, feasible_rate=None, feasible_count=None):
        if gen is None:
            return
        delta = gen - getattr(bar, "n", 0)
        _safe_update(bar, delta)
        postfix = {
            "seed": seed,
            "scenario": scenario,
            "env": env_type,
            "algo": algo,
            "gen": gen,
        }
        if eval_count is not None:
            if total_evals:
                postfix["eval"] = f"{int(eval_count)}/{int(total_evals)}"
            else:
                postfix["eval"] = int(eval_count)
        if feasible_count is not None:
            postfix["feas#"] = int(feasible_count)
        if feasible_rate is not None:
            postfix["feas"] = f"{feasible_rate:.3f}"
        bar.set_postfix(postfix)

    return bar, progress_cb


def timestamp():
    return datetime.now().strftime("%Y%m%dT%H%M%S")


# make_problem is now shared in dopa.problem_factory to keep baseline comparisons aligned.


def result_filename(scenario: str, seed: int, switching_scheme: str):
    base = f"result_{scenario}_seed{seed}"
    if switching_scheme and switching_scheme != "dopa_entropy":
        base += f"_{switching_scheme}"
    return os.path.join(JSON_DIR, f"{base}.json")


def save_result(result: Dict, cfg: Dict):
    scenario = result["scenario"]
    seed = result["seed"]
    switching_scheme = result.get("switching_scheme", cfg.get("switching_scheme", "dopa_entropy"))
    fname = result_filename(scenario, seed, switching_scheme)
    with open(fname, "w") as f:
        json.dump(result, f, indent=2)
    _progress_write(f"✔ 저장 완료 → {fname}")


def attach_metadata(result: Dict, cfg: Dict, seed: int):
    result.update(
        {
            "num_uavs": cfg.get("num_uavs"),
            "num_targets": cfg.get("num_targets"),
            "generations": cfg.get("generations"),
            "pop_size": cfg.get("pop_size"),
            "env_type": cfg.get("env_type", "static"),
            "dynamic_mode": cfg.get("dynamic_mode"),
            "switching_scheme": cfg.get("switching_scheme", "dopa_entropy"),
            "seed": seed,
        }
    )
    return result


def run_single_configuration(
    cfg: Dict,
    *,
    progress_enabled: bool = False,
    outer_progress=None,
) -> List[Dict]:
    seeds = to_list(cfg.get("seeds", [0]))
    scenarios = to_list(cfg.get("scenarios", ["S1", "S2", "S3", "S4"]))
    total_runs = len(seeds) * len(scenarios)
    own_outer = False
    if outer_progress is None:
        outer_progress = _get_progress(total_runs, desc="runs", enabled=progress_enabled, leave=True, position=0)
        own_outer = True
    results = []
    ngen = int(cfg.get("generations", 400))
    pop_size = int(cfg.get("pop_size", 200))
    for sd in seeds:
        _progress_write(f"\n=== Seed {sd} : 문제 생성 ===")
        for scenario in scenarios:
            _progress_write(f"=== Seed {sd} : {scenario} 실행 (scheme={cfg.get('switching_scheme')}) ===")
            problem = make_problem(cfg, sd)
            gen_bar, progress_cb = _make_gen_progress(
                ngen=ngen,
                pop_size=pop_size,
                seed=sd,
                scenario=scenario,
                env_type=cfg.get("env_type", "static"),
                algo="DOPA",
                enabled=progress_enabled,
            )
            res = run_scenario(
                scenario_key=scenario,
                problem=problem,
                seed=sd,
                ngen=ngen,
                pop_size=pop_size,
                track_wasserstein=True,
                progress_cb=progress_cb,
            )
            gen_bar.close()
            res = attach_metadata(res, cfg, sd)
            save_result(res, cfg)
            results.append(res)
            outer_progress.set_postfix(
                {
                    "seed": sd,
                    "scenario": scenario,
                    "env": cfg.get("env_type", "static"),
                    "algo": "DOPA",
                }
            )
            _safe_update(outer_progress, 1)

            if cfg.get("enable_plot", False):
                _maybe_plot(res, cfg)
    if own_outer:
        outer_progress.close()
    return results


def _maybe_plot(res: Dict, cfg: Dict):
    scheme = res.get("switching_scheme", cfg.get("switching_scheme", "dopa_entropy"))
    scenario = res.get("scenario", "S?")
    seed = res.get("seed", 0)
    plot_type = cfg.get("pareto_plot_type", "2d")
    axes = cfg.get("pareto_axes", [0, 1])

    # traces
    metrics = cfg.get("plot_metrics", [])
    for metric in metrics:
        trace = res.get(metric)
        save_trace_plot(trace, metric, scenario, seed, scheme, PLOT_DIR)

    # Pareto
    pareto = res.get("final_pareto", [])
    save_pareto_plot(pareto, scenario, seed, scheme, PLOT_DIR, plot_type=plot_type, axes=axes)


def run_normal_mode(cfg: Dict, *, progress_enabled: bool = False):
    all_cfgs = expand_sweeps(cfg)
    total_runs = 0
    for local_cfg in all_cfgs:
        seeds = to_list(local_cfg.get("seeds", [0]))
        scenarios = to_list(local_cfg.get("scenarios", ["S1", "S2", "S3", "S4"]))
        total_runs += len(seeds) * len(scenarios)
    outer_progress = _get_progress(total_runs, desc="runs", enabled=progress_enabled, leave=True, position=0)
    all_results = []
    for local_cfg in all_cfgs:
        all_results.extend(
            run_single_configuration(local_cfg, progress_enabled=progress_enabled, outer_progress=outer_progress)
        )
    outer_progress.close()
    _progress_write("\n🎉 모든 실험 종료 완료!")
    _progress_write(f"저장 위치: {RUN_DIR}/ (json under json/, plots under plot/)")
    return all_results


def run_time_complexity(cfg: Dict, *, progress_enabled: bool = False):
    tc = cfg.get("time_complexity", {})
    num_uavs_list = to_list(tc.get("num_uavs", []))
    num_targets_list = to_list(tc.get("num_targets", []))
    generations_list = to_list(tc.get("generations", []))
    pop_size_list = to_list(tc.get("pop_size", []))
    switching_schemes = to_list(tc.get("switching_schemes", []))
    seeds = to_list(tc.get("seeds", [0]))
    scenarios = to_list(tc.get("scenarios", ["S4"]))

    logs = []
    total_runs = (
        len(num_uavs_list)
        * len(num_targets_list)
        * len(generations_list)
        * len(pop_size_list)
        * len(switching_schemes)
        * len(seeds)
        * len(scenarios)
    )
    outer_progress = _get_progress(total_runs, desc="runs", enabled=progress_enabled, leave=True, position=0)
    combos = itertools.product(num_uavs_list, num_targets_list, generations_list, pop_size_list, switching_schemes)
    for (n_uav, n_tgt, gen, pop, scheme) in combos:
        local_cfg = dict(cfg)
        local_cfg.update(
            {
                "num_uavs": n_uav,
                "num_targets": n_tgt,
                "generations": gen,
                "pop_size": pop,
                "switching_scheme": scheme,
                "seeds": seeds,
                "scenarios": scenarios,
                "env_type": tc.get("env_type", cfg.get("env_type", "static")),
                "dynamic_mode": tc.get("dynamic_mode", cfg.get("dynamic_mode")),
            }
        )
        _progress_write(f"\n[time_complexity] UAV={n_uav}, TGT={n_tgt}, gen={gen}, pop={pop}, scheme={scheme}")
        results = run_single_configuration(local_cfg, progress_enabled=progress_enabled, outer_progress=outer_progress)
        for res in results:
            entry = {
                "num_uavs": n_uav,
                "num_targets": n_tgt,
                "generations": gen,
                "pop_size": pop,
                "switching_scheme": scheme,
                "env_type": cfg.get("env_type", "static"),
                "dynamic_mode": cfg.get("dynamic_mode"),
                "scenario": res.get("scenario"),
                "seed": res.get("seed"),
                "execution_time": res.get("execution_time"),
            }
            logs.append(entry)

    outer_progress.close()
    log_path = os.path.join(JSON_DIR, "time_complexity_log.json")
    with open(log_path, "w") as f:
        json.dump(logs, f, indent=2)
    _progress_write(f"\n⏱  시간 복잡도 로그 저장 → {log_path}")

    # plot runtime vs. UAV count
    save_time_complexity_plot(logs, PLOT_DIR, x_key="num_uavs", title="Runtime vs Number of UAVs")
    save_time_complexity_plot(logs, PLOT_DIR, x_key="num_targets", title="Runtime vs Number of Targets")


def parse_args():
    parser = argparse.ArgumentParser(description="Run DOPA experiments with YAML configuration.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config file.")
    parser.add_argument(
        "--mode", type=str, choices=["normal", "time_complexity"], default=None, help="Override mode from config."
    )
    parser.add_argument("--plot", action="store_true", help="Enable plotting regardless of config.")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bars.")
    parser.add_argument("--progress", action="store_true", help="Force progress bars even without a TTY.")
    return parser.parse_args()


def main():
    args = parse_args()
    progress_enabled = _resolve_progress_enabled(args)
    _set_progress_enabled(progress_enabled)
    cfg = load_config(args.config)
    if args.mode:
        cfg["mode"] = args.mode
    if args.plot:
        cfg["enable_plot"] = True

    _progress_write(f"🚀 Running DOPA experiments (mode={cfg.get('mode', 'normal')})")
    if cfg.get("mode", "normal") == "time_complexity":
        run_time_complexity(cfg, progress_enabled=progress_enabled)
    else:
        run_normal_mode(cfg, progress_enabled=progress_enabled)


if __name__ == "__main__":
    main()
