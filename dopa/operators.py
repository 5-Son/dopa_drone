# dopa/operators.py
import numpy as np
from deap import tools


# -----------------------------
# Variance-normalized Riemannian Crowding Distance
# -----------------------------
def riemannian_crowding_distance(population):
    """
    DOPA.ipynb에서 사용하던 Riemannian crowding distance 그대로 분리한 함수.

    - population: DEAP individual 리스트
    - 각 individual.fitness.values 를 가져와 (pop_size, n_obj) 행렬로 만든 뒤
      분산으로 정규화된 objective 공간에서 crowding distance 계산.
    """
    nobj = len(population[0].fitness.values)
    distances = np.zeros(len(population))
    fits = np.array([ind.fitness.values for ind in population])
    variances = np.var(fits, axis=0)
    norm_fits = fits / (variances + 1e-10)

    for m in range(nobj):
        sorted_idx = np.argsort(norm_fits[:, m])
        distances[sorted_idx[0]] = distances[sorted_idx[-1]] = float("inf")
        for i in range(1, len(population) - 1):
            distances[sorted_idx[i]] += (
                norm_fits[sorted_idx[i + 1], m] - norm_fits[sorted_idx[i - 1], m]
            )
    return distances


def hybrid_select(pop, k):
    """
    DOPA.ipynb의 hybrid_select(pop, k)를 그대로 모듈로 분리한 함수.

    - 먼저 NSGA-II non-dominated sorting 수행
    - front 단위로 채우다가 마지막 front에서는
      Riemannian crowding distance가 큰 순서대로 선택
    """
    # Pareto 정렬 (dominance rank 기준)
    fronts = tools.sortNondominated(pop, k, first_front_only=False)
    selected = []

    for front in fronts:
        if len(selected) + len(front) <= k:
            selected.extend(front)
        else:
            # Riemannian 거리 기반으로 front 내에서 선택
            distances = riemannian_crowding_distance(front)
            distances = np.nan_to_num(distances, nan=0.0, posinf=0.0, neginf=0.0)
            sorted_idx = np.argsort(-distances)  # 큰 거리 순
            selected.extend([front[i] for i in sorted_idx[:k - len(selected)]])
            break

    return selected


def _select_by_fronts(pop, k):
    """Select with NSGA-II fronts + Riemannian crowding distance."""
    if k <= 0:
        return []
    fronts = tools.sortNondominated(pop, k, first_front_only=False)
    selected = []
    for front in fronts:
        if len(selected) + len(front) <= k:
            selected.extend(front)
        else:
            distances = riemannian_crowding_distance(front)
            distances = np.nan_to_num(distances, nan=0.0, posinf=0.0, neginf=0.0)
            sorted_idx = np.argsort(-distances)
            selected.extend([front[i] for i in sorted_idx[: k - len(selected)]])
            break
    return selected


def constraint_dominance_select(pop, k):
    """
    Deb's constraint-domination: feasible dominates infeasible; among infeasible,
    lower constraint violation is preferred; nondominated sorting applies only
    within the feasible set.
    """
    if k <= 0:
        return []

    cvs = np.array([float(getattr(ind, "cv", 0.0)) for ind in pop])
    feasible = [ind for ind, cv in zip(pop, cvs) if cv <= 0.0]
    infeasible = [ind for ind, cv in zip(pop, cvs) if cv > 0.0]
    infeasible_cvs = cvs[cvs > 0.0]

    selected = []
    if feasible:
        selected = _select_by_fronts(feasible, min(k, len(feasible)))

    if len(selected) < k and infeasible:
        remaining = k - len(selected)
        order = []
        unique_cvs = np.unique(infeasible_cvs)
        for cv in unique_cvs:
            group = [ind for ind, cv_i in zip(infeasible, infeasible_cvs) if cv_i == cv]
            if len(group) > 1:
                distances = riemannian_crowding_distance(group)
                distances = np.nan_to_num(distances, nan=0.0, posinf=0.0, neginf=0.0)
                sorted_idx = np.argsort(-distances)
                order.extend([group[i] for i in sorted_idx])
            else:
                order.extend(group)
        selected.extend(order[:remaining])

    return selected


def select_with_riemannian(pop, k):
    """
    DOPA.ipynb에 있던 select_with_riemannian(pop, k) 그대로.

    - 전체 population에서 Riemannian crowding distance를 계산한 뒤
      거리가 큰 개체부터 k개 선택.
    - baseline 실험이나 비교용으로 사용 가능.
    """
    distances = riemannian_crowding_distance(pop)
    sorted_idx = np.argsort(-distances)  # 거리 큰 순서대로
    return [pop[i] for i in sorted_idx[:k]]
