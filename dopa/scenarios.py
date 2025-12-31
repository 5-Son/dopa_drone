# dopa/scenarios.py
from dataclasses import dataclass
from typing import Dict, List

from .problem import UAVTargetAssignmentProblem
from .algorithm import run_dopa, DOPAConfig


@dataclass
class ScenarioConfig:
    """각 시나리오별 설정"""
    name: str
    adaptive_mut: bool
    adaptive_cx: bool


# 논문 기준 S1~S4 설정
SCENARIOS: Dict[str, ScenarioConfig] = {
    "S1": ScenarioConfig(name="Baseline",      adaptive_mut=False, adaptive_cx=False),
    "S2": ScenarioConfig(name="AdaptiveMut",   adaptive_mut=True,  adaptive_cx=False),
    "S3": ScenarioConfig(name="AdaptiveCx",    adaptive_mut=False, adaptive_cx=True),
    "S4": ScenarioConfig(name="DOPAFull",      adaptive_mut=True,  adaptive_cx=True),
}


def run_scenario(
    scenario_key: str,
    problem: UAVTargetAssignmentProblem,
    seed: int = 0,
    ngen: int = 50,
    pop_size: int = 50,
    track_wasserstein: bool = True,
    progress_cb=None,
):
    """
    단일 시나리오(S1~S4)를 실행하는 helper 함수.

    사용 예:
        from dopa.scenarios import run_scenario
        result = run_scenario("S4", problem, seed=0, ngen=400, pop_size=200)
    """
    cfg = SCENARIOS[scenario_key]

    switching_scheme = getattr(problem, "switching_scheme", "dopa_entropy")
    env_type = getattr(problem, "env_type", "static")
    dynamic_mode = getattr(problem, "dynamic_mode", None)
    device = getattr(problem, "device", None)

    dopa_cfg = DOPAConfig(
        adaptive_mut=cfg.adaptive_mut,
        adaptive_cx=cfg.adaptive_cx,
        seed=seed,
        ngen=ngen,
        pop_size=pop_size,
        track_wasserstein=track_wasserstein,
        switching_scheme=switching_scheme,
        env_type=env_type,
        dynamic_mode=dynamic_mode,
        device=device,
    )

    result = run_dopa(problem, dopa_cfg, progress_cb=progress_cb)
    result["scenario"] = scenario_key
    result["scenario_name"] = cfg.name
    return result


def run_all_scenarios(
    problem: UAVTargetAssignmentProblem,
    seeds: List[int],
    ngen: int = 50,
    pop_size: int = 50,
    track_wasserstein: bool = True,
):
    """
    모든 시나리오(S1~S4)에 대해 여러 seed로 반복 실험을 실행하는 helper.

    반환: 결과 dict들의 리스트
    """
    results = []
    for key in ["S1", "S2", "S3", "S4"]:
        for sd in seeds:
            res = run_scenario(
                scenario_key=key,
                problem=problem,
                seed=sd,
                ngen=ngen,
                pop_size=pop_size,
                track_wasserstein=track_wasserstein,
            )
            results.append(res)
    return results
