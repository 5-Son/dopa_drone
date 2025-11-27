# dopa/problem.py
from dataclasses import dataclass, field
from typing import Iterable, List, Optional

import numpy as np
import torch

from .torch_utils import get_default_device, to_tensor, to_numpy


@dataclass
class UAVTargetAssignmentProblem:
    """
    UAV-표적 다중 할당 문제 정의 클래스.

    - N: UAV 수
    - M: 표적 수
    - d_ij: (N, M) 거리 행렬
    - v_j: (M,) 표적 가치 (DOPA 코드의 v_i와 동일 역할)
    - T_detect_ij: (N, M) 탐지 시간
    - T_mission_i: (N,) UAV별 고정 임무 시간
    - D_max: 최대 타격 거리 (MAX_DISTANCE와 동일)
    - C_total: 전체 임무(할당) 수 상한 (MAX_TOTAL_MISSIONS와 동일)
    - penalty_scale: apply_constraints 에서 사용하던 penalty 스케일
    """

    N: int
    M: int
    d_ij: np.ndarray
    v_j: np.ndarray
    T_detect_ij: np.ndarray
    T_mission_i: np.ndarray
    D_max: float
    C_total: int
    penalty_scale: float = 0.1  # 기존 코드의 penalty_scale
    device: Optional[str] = None
    use_torch: bool = True
    enable_batch_eval: bool = True
    env_type: str = "static"

    # internal caches (lazy init)
    _device: torch.device = field(init=False)
    _torch_ready: bool = field(init=False, default=False)
    _p_ij_t: Optional[torch.Tensor] = field(init=False, default=None)
    _d_ij_t: Optional[torch.Tensor] = field(init=False, default=None)
    _v_j_t: Optional[torch.Tensor] = field(init=False, default=None)
    _T_detect_t: Optional[torch.Tensor] = field(init=False, default=None)
    _T_mission_t: Optional[torch.Tensor] = field(init=False, default=None)

    def __post_init__(self):
        self._device = get_default_device(self.device)
        self._torch_ready = bool(self.use_torch)
        self.device = str(self._device)
        self._prepare_tensors()

    # ------------------------------------------------------------------
    # tensor preparation helpers
    # ------------------------------------------------------------------
    def _prepare_tensors(self):
        """Prepare torch tensors for hot paths (objective and constraints)."""
        self._p_ij = np.exp(-0.005 * self.d_ij)
        try:
            if self._torch_ready:
                self._p_ij_t = to_tensor(self._p_ij, self._device)
                self._d_ij_t = to_tensor(self.d_ij, self._device)
                self._v_j_t = to_tensor(self.v_j, self._device)
                self._T_detect_t = to_tensor(self.T_detect_ij, self._device)
                self._T_mission_t = to_tensor(self.T_mission_i, self._device)
        except Exception:
            # Fallback to NumPy-only execution if the selected device fails
            self._torch_ready = False
            self._p_ij_t = None
            self._d_ij_t = None
            self._v_j_t = None
            self._T_detect_t = None
            self._T_mission_t = None

    # ------------------------------------------------------------------
    # 1. 개체(해) 인코딩: generate_diverse_suicide_individual 기반
    # ------------------------------------------------------------------
    def encode_random_individual(self, rng: np.random.Generator, max_assign: int = 3) -> np.ndarray:
        """
        DOPA.ipynb의 generate_diverse_suicide_individual(d_ij, max_assign=3)를
        클래스 내부 버전으로 옮긴 것.

        - 각 UAV i 에 대해:
          * 거리 기반 sigmoid 확률로 표적 선택
          * assign_count ~ Uniform{1, ..., max_assign}
          * 선택된 표적에 1을 주고 나머지는 0
        - 결과: 길이 N*M인 0/1 벡터 (flat)
        """
        individual = []

        for i in range(self.N):
            gene = np.zeros(self.M, dtype=int)
            dist = self.d_ij[i]  # (M,)

            # 거리 기반 선택 확률 (원 코드: prob = 1 / (1 + exp((dist - MAX_DISTANCE)/50)))
            prob = 1.0 / (1.0 + np.exp((dist - self.D_max) / 50.0))

            # 모든 확률이 0이 되는 경우 방어 처리
            prob_sum = prob.sum()
            if prob_sum <= 0:
                prob = np.ones_like(prob) / len(prob)
            else:
                prob = prob / prob_sum  # 정규화

            # UAV당 1~max_assign 개 표적 할당 (원 코드: random.randint(1, max_assign))
            assign_count = int(rng.integers(1, max_assign + 1))

            # 선택된 표적에 1 할당 (원 코드: np.random.choice)
            selected_targets = rng.choice(self.M, size=assign_count, replace=False, p=prob)
            gene[selected_targets] = 1

            individual.extend(gene.tolist())

        return np.array(individual, dtype=int)

    # ------------------------------------------------------------------
    # 2. 개체 → x_ij 변환 (I 행렬)
    # ------------------------------------------------------------------
    def decode_to_xij(self, individual: np.ndarray) -> np.ndarray:
        """
        flat 벡터 individual (길이 N*M)를 (N, M) 행렬로 변환.
        DOPA.ipynb에서:
            I = np.array(ind).reshape(NUM_UAV, NUM_TARGET)
        와 동일한 역할.
        """
        arr = np.array(individual, dtype=int)
        return arr.reshape(self.N, self.M)

    # ------------------------------------------------------------------
    # 3. 목적함수 F1, F2, F3 계산 (원 evaluate(ind)에서 패널티 적용 전 부분)
    # ------------------------------------------------------------------
    def evaluate_objectives_batch(self, individuals: Iterable[np.ndarray]):
        """
        Vectorized objective computation for a batch of individuals.
        Returns an array shaped (batch, 3) on CPU.
        """
        arr = np.asarray(list(individuals), dtype=int)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        batch_size = arr.shape[0]
        mats = arr.reshape(batch_size, self.N, self.M)

        if self._torch_ready and self.enable_batch_eval:
            I = to_tensor(mats, self._device, dtype=torch.float32)
            v = self._v_j_t.view(1, 1, -1)
            F1 = torch.sum(v * self._p_ij_t * I, dim=(1, 2))
            F2 = torch.sum(self._T_detect_t * I, dim=(1, 2))
            T_uav = torch.sum(self._T_detect_t * I, dim=2)
            F3 = torch.max(T_uav + self._T_mission_t, dim=1).values
            stacked = torch.stack([F1, F2, F3], dim=1)
            return to_numpy(stacked)

        # NumPy fallback
        F1_list: List[float] = []
        F2_list: List[float] = []
        F3_list: List[float] = []

        for I in mats:
            F1_list.append(float(np.sum(self.v_j * self._p_ij * I)))
            F2_list.append(float(np.sum(self.T_detect_ij * I)))
            T_uav = np.sum(self.T_detect_ij * I, axis=1)
            F3_list.append(float(np.max(T_uav + self.T_mission_i)))

        return np.column_stack([F1_list, F2_list, F3_list])

    def evaluate_objectives(self, individual: np.ndarray):
        """Single-individual objective wrapper for backward compatibility."""
        res = self.evaluate_objectives_batch([individual])
        return tuple(float(v) for v in res[0])

    # ------------------------------------------------------------------
    # 4. 제약조건 패널티 계산 (apply_constraints에서 penalty 부분만 분리)
    # ------------------------------------------------------------------
    def constraint_penalty_batch(self, individuals: Iterable[np.ndarray]) -> np.ndarray:
        """
        Batched constraint penalty calculation.
        """
        arr = np.asarray(list(individuals), dtype=int)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        batch_size = arr.shape[0]
        mats = arr.reshape(batch_size, self.N, self.M)

        if self._torch_ready and self.enable_batch_eval:
            I = to_tensor(mats, self._device, dtype=torch.float32)
            suicide_violation = torch.sum(torch.sum(I, dim=2) != 1, dim=1)
            mission_violation = (torch.sum(I, dim=(1, 2)) > self.C_total).int()
            distance_violation = torch.sum((self._d_ij_t > self.D_max) * (I > 0.5), dim=(1, 2))
            violation_score = suicide_violation + mission_violation + distance_violation
            penalty = self.penalty_scale * violation_score.float() / (self.N + self.M + 1)
            return to_numpy(penalty)

        penalties: List[float] = []
        for I in mats:
            suicide_violation = int(np.sum(np.sum(I, axis=1) != 1))
            mission_violation = int(np.sum(I) > self.C_total)
            distance_violation = int(np.sum((self.d_ij > self.D_max) * I))
            violation_score = suicide_violation + mission_violation + distance_violation
            penalty = self.penalty_scale * violation_score / (self.N + self.M + 1)
            penalties.append(float(penalty))

        return np.array(penalties, dtype=float)

    def constraint_penalty(self, individual: np.ndarray) -> float:
        res = self.constraint_penalty_batch([individual])
        return float(res[0])

    # ------------------------------------------------------------------
    # 5. Combined evaluation with penalties (batched)
    # ------------------------------------------------------------------
    def evaluate_with_penalty_batch(self, individuals: Iterable[np.ndarray]) -> np.ndarray:
        objs = self.evaluate_objectives_batch(individuals)
        penalties = self.constraint_penalty_batch(individuals).reshape(-1, 1)
        F1_adj = objs[:, 0:1] * (1.0 - penalties)
        F2_adj = objs[:, 1:2] * (1.0 + penalties)
        F3_adj = objs[:, 2:3] * (1.0 + penalties)
        return np.concatenate([F1_adj, F2_adj, F3_adj], axis=1)

    def evaluate_with_penalty(self, individual: np.ndarray):
        res = self.evaluate_with_penalty_batch([individual])
        row = res[0]
        return float(row[0]), float(row[1]), float(row[2])

    # ------------------------------------------------------------------
    # 6. Environment update helper (for dynamic subclass reuse)
    # ------------------------------------------------------------------
    def refresh_parameters(self, *, d_ij=None, v_j=None, T_detect_ij=None, T_mission_i=None):
        """Update internal arrays and rebuild tensors."""
        if d_ij is not None:
            self.d_ij = np.asarray(d_ij)
        if v_j is not None:
            self.v_j = np.asarray(v_j)
        if T_detect_ij is not None:
            self.T_detect_ij = np.asarray(T_detect_ij)
        if T_mission_i is not None:
            self.T_mission_i = np.asarray(T_mission_i)
        self._prepare_tensors()


# ----------------------------------------------------------------------
# Dynamic environment variant
# ----------------------------------------------------------------------
@dataclass
class DynamicUAVTargetAssignmentProblem(UAVTargetAssignmentProblem):
    env_type: str = "dynamic"
    dynamic_mode: str = "dynamic_targets"
    dynamic_interval: int = 5
    dynamic_noise_scale: float = 0.05
    dynamic_seed: Optional[int] = None
    max_target_shift: float = 0.15
    environment_events: List[dict] = field(init=False, default_factory=list)

    _base_d_ij: np.ndarray = field(init=False)
    _base_T_detect: np.ndarray = field(init=False)
    _base_v_j: np.ndarray = field(init=False)
    _base_T_mission: np.ndarray = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self.dynamic_seed = 0 if self.dynamic_seed is None else self.dynamic_seed
        self._base_d_ij = np.array(self.d_ij, copy=True)
        self._base_T_detect = np.array(self.T_detect_ij, copy=True)
        self._base_v_j = np.array(self.v_j, copy=True)
        self._base_T_mission = np.array(self.T_mission_i, copy=True)
        self.environment_events = []

    def step_dynamics(self, generation: int):
        """Apply deterministic environment changes every `dynamic_interval` generations."""
        if self.env_type != "dynamic":
            return
        if generation % max(1, self.dynamic_interval) != 0:
            return

        rng = np.random.default_rng(self.dynamic_seed + generation)
        events: dict = {"generation": generation, "mode": self.dynamic_mode}

        # Target dynamics
        if self.dynamic_mode in ("dynamic_targets", "dynamic_both", "dynamic_noise"):
            noise = rng.normal(scale=self.dynamic_noise_scale, size=self._base_d_ij.shape)
            new_d = np.clip(self._base_d_ij * (1.0 + noise), 1.0, None)
            new_v = np.array(self._base_v_j)
            appear_mask = disappear_mask = None
            if self.dynamic_mode != "dynamic_noise":
                appear_mask = rng.random(self.M) < 0.05
                disappear_mask = rng.random(self.M) < 0.05
                if np.any(disappear_mask):
                    new_d[:, disappear_mask] = self.D_max * 1.5
                    new_v[disappear_mask] = 0.0
                if np.any(appear_mask):
                    new_d[:, appear_mask] *= 0.8
                    new_v[appear_mask] = new_v[appear_mask] * (1.1 + 0.1 * rng.random(np.sum(appear_mask)))
            events["targets"] = {
                "mode": self.dynamic_mode,
                "appear": appear_mask.tolist() if appear_mask is not None else None,
                "disappear": disappear_mask.tolist() if disappear_mask is not None else None,
                "noise_scale": self.dynamic_noise_scale,
            }
            self.d_ij = new_d
            self.v_j = new_v

        # UAV dynamics
        if self.dynamic_mode in ("dynamic_uavs", "dynamic_both", "dynamic_noise"):
            uav_shift = rng.normal(scale=self.max_target_shift, size=self._base_T_mission.shape)
            new_T_mission = np.clip(self._base_T_mission * (1.0 + uav_shift), 1.0, None)
            detect_noise = rng.normal(scale=self.dynamic_noise_scale, size=self._base_T_detect.shape)
            new_T_detect = np.clip(self._base_T_detect * (1.0 + detect_noise), 1.0, None)
            events["uavs"] = {
                "mode": self.dynamic_mode,
                "mission_shift_mean": float(np.mean(uav_shift)),
                "detect_noise_scale": self.dynamic_noise_scale,
            }
            self.T_mission_i = new_T_mission
            self.T_detect_ij = new_T_detect

        if len(events) > 2:  # generation/mode always present
            self.environment_events.append(events)
        self._prepare_tensors()

    def get_environment_events(self) -> List[dict]:
        return list(self.environment_events)
