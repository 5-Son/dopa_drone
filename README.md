DOPA Project
===========

Multi-objective UAV target-assignment experiments using the DOPA algorithm (adaptive crossover/mutation probabilities) built on DEAP.

Project layout
--------------
- `run.py`: generates synthetic problem instances, runs scenarios S1~S4 across seeds, and saves JSON outputs under `results/`.
- `analyze_results.py`: reads all JSON results and prints per-file and per-scenario summaries.
- `dopa/`: core library (problem definition, DEAP-based algorithm, operators, metrics, utilities, and scenario helpers).
- `results/`: sample outputs (`result_S*_seed*.json`) produced by `run.py`.
- `Untitled.ipynb`: exploratory notebook (optional).

Setup
-----
1) Create/activate a Python environment (Python 3.10+ recommended).  
2) Install dependencies:
```
pip install -r requirements.txt
```

Run experiments
---------------
Execute the full experiment suite (seeds defined in `config.yaml`, scenarios S1~S4 by default) and store results:
```
python run.py
```
Outputs: one file per scenario/seed, e.g. `results/<timestamp>/json/result_S4_seed0.json`, with plots under `results/<timestamp>/plot/`. Key fields include `scenario`, `scenario_name`, `seed`, `adaptive_mut`, `adaptive_cx`, `execution_time`, `final_pareto` (list of [F1, F2, F3]), `wasserstein_trace`, `entropy_trace`, `cxpb_trace`, and `mutpb_trace`.

Configuration (YAML, sweeps, and schemes)
-----------------------------------------
- Edit `config.yaml` (auto-loaded by `run.py`) to change problem size, seeds, schemes, plotting, etc. Provide lists to sweep combinations automatically.
- `switching_scheme` options: `dopa_entropy` (default), `sa_temp`, `mab_ucb`, `drl_dqn`, `moead_adaptive`, `strategy1`, `strategy2`, `strategy3`.
- CLI overrides:
```
python run.py --config config.yaml          # custom config path
python run.py --plot                        # force-enable plotting
python run.py --mode time_complexity        # run scaling experiments
```

Switching schemes (quick guide)
-------------------------------
- `dopa_entropy`: original entropy-driven adaptive cx/mut schedule.
- `sa_temp`: simulated-annealing temperature; higher exploration early, cools to exploit.
- `mab_ucb`: UCB1 bandit picks among fixed cx/mut arms using improvement/diversity reward.
- `drl_dqn`: DQN (RLlib) policy selects arms from state features (entropy/Wasserstein/improvement/gen), with heuristic fallback if RL is unavailable.
- `moead_adaptive`: MOEA/D-inspired scalarization signal adjusts cx/mut based on subproblem improvement.
- `strategy1`: entropy hysteresis switching between explore/exploit modes.
- `strategy2`: Wasserstein change drives diversification vs stabilization.
- `strategy3`: composite indicator combining entropy, Wasserstein, and improvement into one controller.

Dynamic/static environments
---------------------------
- `env_type: static | dynamic` and `dynamic_mode: dynamic_targets | dynamic_uavs | dynamic_both | dynamic_noise` control whether distances/values/mission times drift during evolution. `dynamic_noise` applies only stochastic noise (no appear/disappear).
- Dynamic runs stay reproducible given the same seed; environment changes are logged under `environment_events` in each result JSON.

Time-complexity mode
--------------------
- Set `mode: time_complexity` in `config.yaml` (or `--mode time_complexity`) to sweep grid sizes defined under `time_complexity` (UAVs/targets/pop/gen/scheme).
- Logs are written to `results/time_complexity_log.json`, and runtime plots to `results/plots/`.

Plotting (optional)
-------------------
- Enable via `enable_plot: true` in YAML or `--plot` CLI flag.
- Saves traces for entropy/wasserstein/cxpb/mutpb and Pareto (2D or 3D) to `results/plots/` with informative filenames.

Analyze saved results
---------------------
Summarize all JSON files in `results/`:
```
python analyze_results.py
```
The script prints per-file stats (runtime, Pareto size) and scenario-level aggregates (mean/std for objectives and runtime). Run `run.py` first if the folder is empty.

Using the library directly
--------------------------
- Import and run an individual scenario with custom settings:
```
from dopa.problem import UAVTargetAssignmentProblem
from dopa.scenarios import run_scenario

problem = UAVTargetAssignmentProblem(...)  # provide d_ij, v_j, T_detect_ij, T_mission_i, etc.
result = run_scenario("S4", problem, seed=0, ngen=400, pop_size=200)
```
- `dopa.scenarios.run_all_scenarios` iterates all four scenario configs; `dopa.algorithm.run_dopa` exposes the DEAP loop if you want to tweak operators or traces.

Notes
-----
- Randomness is controlled via `dopa.utils.set_seed`; PyTorch/CUDA are seeded for reproducibility.
- GPU acceleration: the problem tensors default to CUDA when available; override via `device` in the YAML if needed.
- Public APIs remain the same (`UAVTargetAssignmentProblem`, `run_scenario`, `run_all_scenarios`); new config is opt-in.
