# dopa/utils.py
import time
import random
import numpy as np
import matplotlib.pyplot as plt

try:
    import torch
    _TORCH_AVAILABLE = True
except Exception:
    _TORCH_AVAILABLE = False


# ============================================================
# 1. Seed & RNG utilities
# ============================================================
def set_seed(seed: int = 0):
    """
    Python, NumPy 난수를 고정하여 재현성을 확보한다.
    """
    random.seed(seed)
    np.random.seed(seed)
    if _TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Deterministic flags keep reproducibility across CUDA kernels
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_rng(seed: int | None = None):
    """
    numpy random generator를 일관적으로 생성하기 위한 함수.
    """
    if seed is None:
        return np.random.default_rng()
    else:
        return np.random.default_rng(seed)


# ============================================================
# 2. Timer utility (for runtime measurement)
# ============================================================
class Timer:
    """코드 블록 실행 시간을 측정하기 위한 타이머 (with context manager 지원)"""

    def __init__(self):
        self.start_time: float | None = None
        self.end_time: float | None = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = time.time()

    @property
    def elapsed(self):
        if self.start_time is None or self.end_time is None:
            return None
        return self.end_time - self.start_time


# ============================================================
# 3. 단순 시각화 helper (Pareto front, trace)
# ============================================================
def plot_pareto(F1, F2, xlabel="F1", ylabel="F2", title="Pareto Front"):
    """
    F1 vs F2 Pareto front를 간단히 그리는 helper.
    """
    plt.figure(figsize=(6, 5))
    plt.scatter(F1, F2, s=20)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_trace(trace, ylabel="Value", xlabel="Generation", title="Trace Plot"):
    """
    엔트로피, wasserstein distance 등의 세대별 변화를 plot.
    """
    plt.figure(figsize=(7, 4))
    plt.plot(trace)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
