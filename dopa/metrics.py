# dopa/metrics.py
import numpy as np
from scipy.spatial.distance import cdist

# 기본 bin 수 (원 코드의 K_BINS)
K_BINS = 10


# -----------------------------
# Population Entropy 계산
# -----------------------------
def compute_population_entropy(fitnesses, k_bins: int = K_BINS):
    """
    DOPA.ipynb의 compute_population_entropy(fitnesses, k_bins).

    - fitnesses: [(F1, F2, F3), ...] 형태의 리스트 또는 (pop_size, n_obj) 배열
    - 각 objective에 대해 히스토그램을 만들고 엔트로피 계산 후 평균
    """
    obj_array = np.array(fitnesses)
    entropies = []

    for i in range(obj_array.shape[1]):
        hist, _ = np.histogram(obj_array[:, i], bins=k_bins, density=True)
        hist = hist[hist > 0]  # 0인 bin 제거
        if len(hist) == 0:
            ent = 0.0
        else:
            ent = -np.sum(hist * np.log(hist))
        entropies.append(ent)

    return float(np.mean(entropies))


# -----------------------------
# Wasserstein-1 Distance 계산 (Sinkhorn 근사)
# -----------------------------
def compute_wasserstein_distance(pop_fits_prev, pop_fits_next, epsilon: float = 0.1) -> float:
    """
    DOPA.ipynb의 Sinkhorn 근사 기반 Wasserstein-1 거리 계산 함수.

    - pop_fits_prev: 이전 세대의 fitness 배열 (N x m)
    - pop_fits_next: 현재 세대의 fitness 배열 (N x m)
    - epsilon: Sinkhorn 정규화 파라미터
    """
    pop_fits_prev = np.asarray(pop_fits_prev, dtype=float)
    pop_fits_next = np.asarray(pop_fits_next, dtype=float)

    # Normalize both generations together to [0, 1] per objective
    all_f = np.vstack([pop_fits_prev, pop_fits_next])
    mins = all_f.min(axis=0)
    maxs = all_f.max(axis=0)
    scale = maxs - mins
    scale[scale == 0.0] = 1.0

    pop_fits_prev = (pop_fits_prev - mins) / scale
    pop_fits_next = (pop_fits_next - mins) / scale

    a = np.ones(len(pop_fits_prev)) / len(pop_fits_prev)
    b = np.ones(len(pop_fits_next)) / len(pop_fits_next)
    C = cdist(pop_fits_prev, pop_fits_next)

    # ✅ 방어 처리: 거리 행렬이 유한한지 확인
    if not np.all(np.isfinite(C)):
        return 0.0

    K = np.exp(-C / epsilon)
    u = np.ones_like(a)

    for _ in range(50):
        # 방어 처리: divide by zero 방지
        Ku = K @ (b / (K.T @ u + 1e-8))
        u = a / (Ku + 1e-8)

    v = b / (K.T @ u + 1e-8)
    transport = np.outer(u, v) * K
    W = np.sum(transport * C)

    return float(W) if np.isfinite(W) else 0.0
