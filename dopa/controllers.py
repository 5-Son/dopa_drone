"""
Exploration–Exploitation switching schemes used by DOPA.
All controllers expose a uniform interface:
    controller.update(gen=..., metrics=..., population=..., fitnesses=...)
and return (cxpb, mutpb, extra_info).
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

try:
    import gym
    _GYM_AVAILABLE = True
except Exception:
    _GYM_AVAILABLE = False

try:
    import ray
    from ray.rllib.algorithms.dqn import DQNConfig
    from ray.rllib.policy.sample_batch import SampleBatch

    _RAY_AVAILABLE = True
except Exception:
    _RAY_AVAILABLE = False


class SwitchingController:
    def __init__(self, scheme: str, ngen: int, adaptive_mut: bool, adaptive_cx: bool):
        self.scheme = scheme
        self.ngen = ngen
        self.adaptive_mut = adaptive_mut
        self.adaptive_cx = adaptive_cx

    def update(self, gen: int, metrics: Dict, population=None, fitnesses=None):
        raise NotImplementedError


class DOPAEntropyController(SwitchingController):
    """Original entropy-based schedule."""

    def update(self, gen: int, metrics: Dict, population=None, fitnesses=None):
        norm_entropy = metrics.get("norm_entropy", 0.0)
        if self.adaptive_cx:
            cxpb = 0.3 + 0.6 * norm_entropy
        else:
            cxpb = 0.9

        if self.adaptive_mut:
            mutpb = 0.05 + 0.25 * norm_entropy
        else:
            mutpb = 0.1
        return cxpb, mutpb, {"norm_entropy": norm_entropy}


class SATempController(SwitchingController):
    """Simulated-annealing style: high exploration early, cools down to exploit."""

    def __init__(self, scheme: str, ngen: int, adaptive_mut: bool, adaptive_cx: bool, t_min=0.1, t_max=1.0):
        super().__init__(scheme, ngen, adaptive_mut, adaptive_cx)
        self.t_min = t_min
        self.t_max = t_max

    def _temperature(self, gen: int):
        frac = gen / max(1, self.ngen - 1)
        return self.t_max * (self.t_min / self.t_max) ** frac

    def update(self, gen: int, metrics: Dict, population=None, fitnesses=None):
        temp = self._temperature(gen)
        cxpb = 0.5 + 0.4 * temp if self.adaptive_cx else 0.9
        mutpb = 0.05 + 0.35 * temp if self.adaptive_mut else 0.1
        return cxpb, mutpb, {"temperature": temp}


class MABUCBController(SwitchingController):
    """UCB1 bandit over a discrete set of (cxpb, mutpb) arms."""

    def __init__(self, scheme: str, ngen: int, adaptive_mut: bool, adaptive_cx: bool):
        super().__init__(scheme, ngen, adaptive_mut, adaptive_cx)
        self.arms: List[Tuple[float, float]] = [
            (0.9, 0.1),
            (0.7, 0.2),
            (0.5, 0.25),
            (0.3, 0.3),
        ]
        self.counts = np.zeros(len(self.arms), dtype=int)
        self.values = np.zeros(len(self.arms), dtype=float)
        self.last_action = 0
        self.last_reward = 0.0

    def _ucb(self, total):
        if total == 0:
            return np.arange(len(self.arms))
        exploration = np.sqrt(2 * np.log(total + 1) / (self.counts + 1e-6))
        return self.values + exploration

    def update(self, gen: int, metrics: Dict, population=None, fitnesses=None):
        # reward from improvement + diversity
        improvement = metrics.get("improvement", 0.0)
        norm_entropy = metrics.get("norm_entropy", 0.0)
        wasserstein = metrics.get("wasserstein", 0.0)
        reward = improvement + 0.2 * norm_entropy - 0.05 * wasserstein

        # update previous action value
        self.values[self.last_action] = (
            (self.values[self.last_action] * self.counts[self.last_action] + reward)
            / (self.counts[self.last_action] + 1)
        )
        self.counts[self.last_action] += 1

        total = int(np.sum(self.counts))
        scores = self._ucb(total)
        action = int(np.argmax(scores))
        self.last_action = action
        cxpb, mutpb = self.arms[action]

        # respect adaptive flags by blending with baseline
        if not self.adaptive_cx:
            cxpb = 0.9
        if not self.adaptive_mut:
            mutpb = 0.1

        return cxpb, mutpb, {"arm": action, "reward": reward}


class DQNSwitchingController(SwitchingController):
    """DQN via RLlib; falls back to heuristic when ray is unavailable."""

    def __init__(self, scheme: str, ngen: int, adaptive_mut: bool, adaptive_cx: bool):
        super().__init__(scheme, ngen, adaptive_mut, adaptive_cx)
        self.arms = [
            (0.9, 0.1),
            (0.7, 0.2),
            (0.5, 0.25),
            (0.4, 0.3),
        ]
        self.last_obs = None
        self.last_action = 0
        self._use_ray = _RAY_AVAILABLE
        self._algo = None
        if self._use_ray and _GYM_AVAILABLE:
            try:
                num_actions = len(self.arms)

                class _SwitchingEnv(gym.Env):
                    metadata = {"render.modes": []}

                    def __init__(self):
                        super().__init__()
                        self.observation_space = gym.spaces.Box(
                            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
                        )
                        self.action_space = gym.spaces.Discrete(num_actions)
                        self._state = np.zeros(4, dtype=np.float32)

                    def reset(self, *, seed=None, options=None):
                        self._state = np.zeros(4, dtype=np.float32)
                        return self._state, {}

                    def step(self, action):
                        # Dummy transition; reward is zero because the MOEA loop supplies rewards separately.
                        return self._state, 0.0, True, False, {}

                env_creator = lambda cfg=None: _SwitchingEnv()
                ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)
                cfg = (
                    DQNConfig()
                    .environment(env=env_creator)
                    .framework("torch")
                    .rollouts(num_rollout_workers=0)
                    .training(model={"fcnet_hiddens": [32, 32]}, lr=1e-3)
                )
                # dummy env spaces: obs size 4, discrete actions
                cfg = cfg.rl_module(_enable_rl_module_api=False)
                cfg.env_runners(
                    sample_timeout_s=0,
                    num_env_runners=0,
                )
                cfg.num_gpus = 0
                cfg.num_cpus_per_worker = 0.25
                cfg.num_envs_per_worker = 1
                cfg.exploration_config["epsilon_timesteps"] = self.ngen
                self._algo = cfg.build(env=env_creator)
                self._policy = self._algo.get_policy()
            except Exception:
                self._use_ray = False
                self._algo = None

    def update(self, gen: int, metrics: Dict, population=None, fitnesses=None):
        obs = np.array(
            [
                metrics.get("norm_entropy", 0.0),
                metrics.get("wasserstein", 0.0),
                metrics.get("improvement", 0.0),
                gen / max(1, self.ngen),
            ],
            dtype=np.float32,
        )
        if self._use_ray and self._algo is not None:
            action = int(self._algo.compute_single_action(obs))
        else:
            # heuristic fallback: favor diversity early, exploitation later
            if obs[0] > 0.6:
                action = 2
            elif obs[1] < 0.1:
                action = 1
            else:
                action = 0
        cxpb, mutpb = self.arms[action]

        # update policy with one-step batch if ray available
        if self._use_ray and self._algo is not None and self.last_obs is not None:
            reward = metrics.get("improvement", 0.0)
            batch = SampleBatch(
                {
                    "obs": np.stack([self.last_obs]),
                    "actions": np.array([self.last_action]),
                    "rewards": np.array([reward], dtype=np.float32),
                    "dones": np.array([False]),
                    "new_obs": np.stack([obs]),
                }
            )
            try:
                self._policy.learn_on_batch(batch)
            except Exception:
                pass

        self.last_obs = obs
        self.last_action = action

        if not self.adaptive_cx:
            cxpb = 0.9
        if not self.adaptive_mut:
            mutpb = 0.1
        return cxpb, mutpb, {"arm": action, "ray": self._use_ray}


class MOEADAdaptiveController(SwitchingController):
    """Lightweight MOEA/D-inspired controller adjusting probabilities via subproblem signal."""

    def __init__(self, scheme: str, ngen: int, adaptive_mut: bool, adaptive_cx: bool, population_size: int = 50):
        super().__init__(scheme, ngen, adaptive_mut, adaptive_cx)
        self.population_size = population_size
        rng = np.random.default_rng(42)
        self.weights = rng.dirichlet(np.ones(3), size=max(2, population_size))
        self.prev_scalar = None

    def update(self, gen: int, metrics: Dict, population=None, fitnesses=None):
        scalar = None
        if fitnesses is not None:
            fit_arr = np.asarray(fitnesses)
            w = self.weights[: fit_arr.shape[0]]
            scalar = np.min(np.sum(w * fit_arr, axis=1))
        improvement = 0.0
        if self.prev_scalar is not None and scalar is not None:
            improvement = self.prev_scalar - scalar
        self.prev_scalar = scalar

        base_mut = 0.1
        base_cx = 0.9
        if self.adaptive_cx:
            cxpb = base_cx - 0.2 * np.tanh(improvement)
        else:
            cxpb = base_cx
        if self.adaptive_mut:
            mutpb = base_mut + 0.2 * (1.0 / (1.0 + np.exp(-5 * improvement)))
        else:
            mutpb = base_mut
        return float(cxpb), float(mutpb), {"scalar_improvement": improvement}


class Strategy1Controller(SwitchingController):
    """Entropy + hysteresis mode switching."""

    def __init__(self, scheme: str, ngen: int, adaptive_mut: bool, adaptive_cx: bool, low=0.35, high=0.7):
        super().__init__(scheme, ngen, adaptive_mut, adaptive_cx)
        self.low = low
        self.high = high
        self.mode = "explore"

    def update(self, gen: int, metrics: Dict, population=None, fitnesses=None):
        ent = metrics.get("norm_entropy", 0.0)
        if self.mode == "explore" and ent > self.high:
            self.mode = "exploit"
        elif self.mode == "exploit" and ent < self.low:
            self.mode = "explore"

        if self.mode == "explore":
            cxpb = 0.6 if self.adaptive_cx else 0.9
            mutpb = 0.25 if self.adaptive_mut else 0.1
        else:
            cxpb = 0.85 if self.adaptive_cx else 0.9
            mutpb = 0.08 if self.adaptive_mut else 0.1
        return cxpb, mutpb, {"mode": self.mode}


class Strategy2Controller(SwitchingController):
    """Wasserstein-based diversification/stabilization."""

    def __init__(self, scheme: str, ngen: int, adaptive_mut: bool, adaptive_cx: bool, ema=0.3):
        super().__init__(scheme, ngen, adaptive_mut, adaptive_cx)
        self.ema = ema
        self.prev_w = None
        self.ema_w = None

    def update(self, gen: int, metrics: Dict, population=None, fitnesses=None):
        w = metrics.get("wasserstein", 0.0)
        if self.ema_w is None:
            self.ema_w = w
        else:
            self.ema_w = self.ema * w + (1 - self.ema) * self.ema_w
        delta = 0.0 if self.prev_w is None else w - self.prev_w
        self.prev_w = w

        diversify = delta < 0.0 and w < self.ema_w * 0.8
        if diversify:
            cxpb = 0.55 if self.adaptive_cx else 0.9
            mutpb = 0.3 if self.adaptive_mut else 0.1
        else:
            cxpb = 0.85 if self.adaptive_cx else 0.9
            mutpb = 0.12 if self.adaptive_mut else 0.1
        return cxpb, mutpb, {"wasserstein_delta": delta, "ema_w": self.ema_w}


class Strategy3Controller(SwitchingController):
    """Composite indicator controller mixing entropy, Wasserstein, and improvement."""

    def update(self, gen: int, metrics: Dict, population=None, fitnesses=None):
        ent = metrics.get("norm_entropy", 0.0)
        w = metrics.get("wasserstein", 0.0)
        imp = metrics.get("improvement", 0.0)
        score = 0.5 * ent - 0.3 * w + 0.2 * imp
        cxpb = 0.4 + 0.5 * np.clip(score, 0, 1) if self.adaptive_cx else 0.9
        mutpb = 0.05 + 0.3 * np.clip(1 - score, 0, 1) if self.adaptive_mut else 0.1
        return cxpb, mutpb, {"composite_score": score}


def build_controller(scheme: str, ngen: int, adaptive_mut: bool, adaptive_cx: bool, population_size: int | None = None) -> SwitchingController:
    scheme = (scheme or "dopa_entropy").lower()
    if scheme == "sa_temp":
        return SATempController(scheme, ngen, adaptive_mut, adaptive_cx)
    if scheme == "mab_ucb":
        return MABUCBController(scheme, ngen, adaptive_mut, adaptive_cx)
    if scheme == "drl_dqn":
        return DQNSwitchingController(scheme, ngen, adaptive_mut, adaptive_cx)
    if scheme == "moead_adaptive":
        return MOEADAdaptiveController(scheme, ngen, adaptive_mut, adaptive_cx, population_size=population_size or 50)
    if scheme == "strategy1":
        return Strategy1Controller(scheme, ngen, adaptive_mut, adaptive_cx)
    if scheme == "strategy2":
        return Strategy2Controller(scheme, ngen, adaptive_mut, adaptive_cx)
    if scheme == "strategy3":
        return Strategy3Controller(scheme, ngen, adaptive_mut, adaptive_cx)
    # default
    return DOPAEntropyController(scheme, ngen, adaptive_mut, adaptive_cx)
