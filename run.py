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
from datetime import datetime
from typing import Dict, List

import numpy as np

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
    print(f"✔ 저장 완료 → {fname}")


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


def run_single_configuration(cfg: Dict) -> List[Dict]:
    seeds = to_list(cfg.get("seeds", [0]))
    scenarios = cfg.get("scenarios", ["S1", "S2", "S3", "S4"])
    results = []
    for sd in seeds:
        print(f"\n=== Seed {sd} : 문제 생성 ===")
        for scenario in scenarios:
            print(f"=== Seed {sd} : {scenario} 실행 (scheme={cfg.get('switching_scheme')}) ===")
            problem = make_problem(cfg, sd)
            res = run_scenario(
                scenario_key=scenario,
                problem=problem,
                seed=sd,
                ngen=int(cfg.get("generations", 400)),
                pop_size=int(cfg.get("pop_size", 200)),
                track_wasserstein=True,
            )
            res = attach_metadata(res, cfg, sd)
            save_result(res, cfg)
            results.append(res)

            if cfg.get("enable_plot", False):
                _maybe_plot(res, cfg)
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


def run_normal_mode(cfg: Dict):
    all_cfgs = expand_sweeps(cfg)
    all_results = []
    for local_cfg in all_cfgs:
        all_results.extend(run_single_configuration(local_cfg))
    print("\n🎉 모든 실험 종료 완료!")
    print(f"저장 위치: {RUN_DIR}/ (json under json/, plots under plot/)")
    return all_results


def run_time_complexity(cfg: Dict):
    tc = cfg.get("time_complexity", {})
    num_uavs_list = to_list(tc.get("num_uavs", []))
    num_targets_list = to_list(tc.get("num_targets", []))
    generations_list = to_list(tc.get("generations", []))
    pop_size_list = to_list(tc.get("pop_size", []))
    switching_schemes = to_list(tc.get("switching_schemes", []))
    seeds = to_list(tc.get("seeds", [0]))
    scenarios = to_list(tc.get("scenarios", ["S4"]))

    logs = []
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
        print(f"\n[time_complexity] UAV={n_uav}, TGT={n_tgt}, gen={gen}, pop={pop}, scheme={scheme}")
        results = run_single_configuration(local_cfg)
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

    log_path = os.path.join(JSON_DIR, "time_complexity_log.json")
    with open(log_path, "w") as f:
        json.dump(logs, f, indent=2)
    print(f"\n⏱  시간 복잡도 로그 저장 → {log_path}")

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
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.mode:
        cfg["mode"] = args.mode
    if args.plot:
        cfg["enable_plot"] = True

    print(f"🚀 Running DOPA experiments (mode={cfg.get('mode', 'normal')})")
    if cfg.get("mode", "normal") == "time_complexity":
        run_time_complexity(cfg)
    else:
        run_normal_mode(cfg)


if __name__ == "__main__":
    main()
