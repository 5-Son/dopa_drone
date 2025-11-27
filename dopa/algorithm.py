# dopa/algorithm.py
import random
import time
import numpy as np
from dataclasses import dataclass
from typing import List

from deap import base, creator, tools

from .problem import UAVTargetAssignmentProblem
from .operators import hybrid_select
from .metrics import compute_population_entropy, compute_wasserstein_distance, K_BINS
from .controllers import build_controller
from .torch_utils import describe_device
from .utils import set_seed


@dataclass
class DOPAConfig:
    adaptive_mut: bool = True
    adaptive_cx: bool = True
    seed: int = 0
    ngen: int = 50
    pop_size: int = 50
    track_wasserstein: bool = True
    switching_scheme: str = "dopa_entropy"
    device: str | None = None
    env_type: str = "static"
    dynamic_mode: str | None = None


def _setup_deap_types():
    """
    DEAP creator에 FitnessMulti, Individual이 이미 등록되어 있는지 확인하고,
    없으면 생성한다. (중복 생성 방지)
    """
    if "FitnessMulti" not in creator.__dict__:
        creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0, -1.0))
    if "Individual" not in creator.__dict__:
        creator.create("Individual", list, fitness=creator.FitnessMulti)


def run_dopa(problem: UAVTargetAssignmentProblem, config: DOPAConfig):
    """
    원래의 run_dopa(adaptive_mut, adaptive_cx, seed, ...)를
    클래스/모듈 기반으로 옮긴 버전.

    - problem: UAVTargetAssignmentProblem 인스턴스
    - config: DOPAConfig (adaptive_mut, adaptive_cx, seed, ngen, pop_size, track_wasserstein)

    반환값은 기존 run_dopa와 동일한 dict 구조:
        {
            "seed": ...,
            "adaptive_mut": ...,
            "adaptive_cx": ...,
            "execution_time": ...,
            "final_pareto": ...,
            "entropy": ...,
            "wasserstein_trace": ...,
            "entropy_trace": ...,
            "cxpb_trace": ...,
            "mutpb_trace": ...
        }
    """
    _setup_deap_types()

    # 랜덤 시드 설정 (torch 포함)
    set_seed(config.seed)
    rng = np.random.default_rng(config.seed)

    N = problem.N
    M = problem.M

    # -----------------------------
    # 평가 함수 (apply_constraints 로직 포함)
    # -----------------------------
    def evaluate(ind):
        arr = np.array(ind, dtype=int)
        return problem.evaluate_with_penalty(arr)

    def batch_evaluate(individuals: List):
        if not individuals:
            return
        values = problem.evaluate_with_penalty_batch([np.array(ind, dtype=int) for ind in individuals])
        for ind, val in zip(individuals, values):
            ind.fitness.values = tuple(float(v) for v in val)

    # -----------------------------
    # DEAP Toolbox 설정
    # -----------------------------
    toolbox = base.Toolbox()

    # 개체 생성: problem.encode_random_individual 사용
    def _make_individual():
        genes = problem.encode_random_individual(rng)  # 길이 N*M 벡터
        return creator.Individual(genes.tolist())

    toolbox.register("individual", _make_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", hybrid_select)  # Riemann + NSGA-II 조합 선택

    # -----------------------------
    # 초기 Population 생성 및 평가
    # -----------------------------
    pop = toolbox.population(n=config.pop_size)
    batch_evaluate(pop)
    pop = toolbox.select(pop, len(pop))

    wasserstein_trace = []
    entropy_trace = []
    cxpb_trace = []
    mutpb_trace = []
    controller_info_trace = []

    controller = build_controller(
        config.switching_scheme,
        config.ngen,
        adaptive_mut=config.adaptive_mut,
        adaptive_cx=config.adaptive_cx,
        population_size=config.pop_size,
    )
    last_improvement = 0.0
    last_wasserstein = 0.0

    start_time = time.time()

    # -----------------------------
    # 진화 루프
    # -----------------------------
    for gen in range(config.ngen):
        if hasattr(problem, "step_dynamics"):
            problem.step_dynamics(gen)
            # environment changed; refresh fitness to keep consistency
            batch_evaluate(pop)

        fitnesses = np.array([ind.fitness.values for ind in pop])

        # Population Entropy 계산
        entropy = compute_population_entropy(fitnesses)

        # 정규화 범위 (원 코드의 경험적 범위: [0.3, 1.6])
        min_entropy = 0.3
        max_entropy = 1.6
        norm_entropy = (entropy - min_entropy) / (max_entropy - min_entropy + 1e-8)
        norm_entropy = np.clip(norm_entropy, 0.0, 1.0)

        metrics = {
            "norm_entropy": float(norm_entropy),
            "entropy": float(entropy),
            "wasserstein": float(last_wasserstein),
            "improvement": float(last_improvement),
            "gen": gen,
        }

        cxpb, mutpb, extra_info = controller.update(gen, metrics, population=pop, fitnesses=fitnesses)
        if isinstance(extra_info, dict):
            clean_info = {}
            for k, v in extra_info.items():
                if isinstance(v, np.generic):
                    clean_info[k] = float(v)
                else:
                    clean_info[k] = v
            extra_info = clean_info

        entropy_trace.append(entropy)
        cxpb_trace.append(cxpb)
        mutpb_trace.append(mutpb)
        controller_info_trace.append(extra_info)

        prev_fits = np.array([ind.fitness.values for ind in pop])

        # -----------------------------
        # Selection → Offspring 생성
        # -----------------------------
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # Crossover
        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(ind1, ind2)
                if "fitness" in ind1.__dict__:
                    del ind1.fitness.values
                if "fitness" in ind2.__dict__:
                    del ind2.fitness.values

        # Mutation
        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mutant)
                if "fitness" in mutant.__dict__:
                    del mutant.fitness.values

        # 새로 생성된 개체 평가
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        batch_evaluate(invalid)

        # 부모 + 자식 합쳐서 NSGA-II 선택(hybrid_select)
        pop = toolbox.select(pop + offspring, config.pop_size)

        # Wasserstein-1 Distance 추적 및 개선량 계산
        new_fits = np.array([ind.fitness.values for ind in pop])
        if config.track_wasserstein:
            wd = compute_wasserstein_distance(prev_fits, new_fits)
            wasserstein_trace.append(wd)
            last_wasserstein = wd

        mean_prev = np.mean(prev_fits, axis=0)
        mean_new = np.mean(new_fits, axis=0)
        last_improvement = (mean_new[0] - mean_prev[0]) - (mean_new[1] - mean_prev[1]) - (mean_new[2] - mean_prev[2])

    end_time = time.time()

    # 최종 Pareto front (fitness.values만 모음)
    pareto = [ind.fitness.values for ind in pop if ind.fitness.valid]

    result = {
        "seed": config.seed,
        "adaptive_mut": config.adaptive_mut,
        "adaptive_cx": config.adaptive_cx,
        "execution_time": end_time - start_time,
        "final_pareto": pareto,
        "entropy": entropy_trace[-1] if len(entropy_trace) > 0 else None,
        "wasserstein_trace": wasserstein_trace if config.track_wasserstein else None,
        "entropy_trace": entropy_trace,
        "cxpb_trace": cxpb_trace,
        "mutpb_trace": mutpb_trace,
        "controller_trace": controller_info_trace,
        "switching_scheme": config.switching_scheme,
        "device": describe_device(problem._device) if hasattr(problem, "_device") else None,
        "env_type": getattr(problem, "env_type", "static"),
        "dynamic_mode": getattr(problem, "dynamic_mode", None),
        "environment_events": getattr(problem, "environment_events", None)
        if hasattr(problem, "environment_events")
        else (problem.get_environment_events() if hasattr(problem, "get_environment_events") else None),
    }

    return result
