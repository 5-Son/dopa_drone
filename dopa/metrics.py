# dopa/metrics.py
import numpy as np
from scipy.spatial.distance import cdist

# Default bin count for entropy histograms.
K_BINS = 10


# -----------------------------
# Population entropy.
# -----------------------------
def compute_population_entropy(fitnesses, k_bins: int = K_BINS):
    """
    Compute population entropy over objective values.

    - fitnesses: array-like shape (pop_size, n_obj) with objective values.
    - k_bins: histogram bin count per objective.
    """
    obj_array = np.array(fitnesses)
    entropies = []

    for i in range(obj_array.shape[1]):
        hist, bin_edges = np.histogram(obj_array[:, i], bins=k_bins, density=True)
        dx = np.diff(bin_edges)
        mask = (hist > 0) & (dx > 0)
        if not np.any(mask):
            ent = 0.0
        else:
            ent = -np.sum(hist[mask] * np.log(hist[mask]) * dx[mask])
        entropies.append(ent)

    return float(np.mean(entropies))


# -----------------------------
# Wasserstein-1 distance (Sinkhorn approximation).
# -----------------------------
def compute_wasserstein_distance(pop_fits_prev, pop_fits_next, epsilon: float = 0.1) -> float:
    """
    Compute Wasserstein-1 distance between two populations (Sinkhorn approx).

    - pop_fits_prev: previous population fitness array (N x m).
    - pop_fits_next: next population fitness array (N x m).
    - epsilon: Sinkhorn regularization parameter.
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

    # Fail-safe: return 0 if distance matrix has non-finite values.
    if not np.all(np.isfinite(C)):
        return 0.0

    K = np.exp(-C / epsilon)
    u = np.ones_like(a)

    for _ in range(50):
        # Sinkhorn update with small epsilon to avoid divide-by-zero.
        Ku = K @ (b / (K.T @ u + 1e-8))
        u = a / (Ku + 1e-8)

    v = b / (K.T @ u + 1e-8)
    transport = np.outer(u, v) * K
    W = np.sum(transport * C)

    return float(W) if np.isfinite(W) else 0.0
