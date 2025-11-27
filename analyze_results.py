# analyze_results.py
"""
results 폴더에 저장된 DOPA 실험 결과(JSON)들을 한 번에 읽어서
요약 통계를 출력하는 스크립트.

실행:
    python analyze_results.py
"""

import os
import json
import math
from collections import defaultdict

RESULT_DIR = "results"


def load_all_results(result_dir=RESULT_DIR):
    """results/ 폴더(하위 디렉토리 포함)의 result_*.json 파일을 모두 읽어 리스트로 반환"""
    if not os.path.isdir(result_dir):
        print(f"[!] 결과 폴더가 없습니다: {result_dir}")
        return []

    results = []
    for root, _, files in os.walk(result_dir):
        for fname in sorted(files):
            if not fname.endswith(".json"):
                continue
            path = os.path.join(root, fname)
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                if not isinstance(data, dict):
                    # 시간 복잡도 로그 등 리스트 기반 파일은 요약 대상에서 제외
                    continue
                data["_filename"] = os.path.relpath(path, result_dir)
                results.append(data)
            except Exception as e:
                print(f"[!] 파일 읽기 실패: {path} → {e}")
    return results


def mean_std(values):
    """평균과 표준편차 계산 (데이터가 1개면 std=0 처리)"""
    if not values:
        return 0.0, 0.0
    n = len(values)
    m = sum(values) / n
    if n == 1:
        return m, 0.0
    var = sum((v - m) ** 2 for v in values) / (n - 1)
    return m, math.sqrt(var)


def summarize_by_scenario(results):
    """
    결과 리스트를 시나리오(S1~S4) 기준으로 묶어서 통계 계산.
    - 실행시간 평균/표준편차
    - 파레토 크기 평균/표준편차
    - Pareto 점들을 모두 모아서 F1, F2, F3 평균/표준편차
    """
    per_scenario = defaultdict(list)
    for r in results:
        key = r.get("scenario", "UNKNOWN")
        per_scenario[key].append(r)

    summary = {}

    for scenario, lst in per_scenario.items():
        exec_times = [r.get("execution_time", 0.0) for r in lst]
        pareto_sizes = [len(r.get("final_pareto", [])) for r in lst]

        # 모든 Pareto 점들을 풀어서 F1, F2, F3 통계 계산
        F1_all, F2_all, F3_all = [], [], []
        for r in lst:
            for f in r.get("final_pareto", []):
                if len(f) >= 3:
                    F1_all.append(f[0])
                    F2_all.append(f[1])
                    F3_all.append(f[2])

        exec_mean, exec_std = mean_std(exec_times)
        size_mean, size_std = mean_std(pareto_sizes)
        F1_mean, F1_std = mean_std(F1_all)
        F2_mean, F2_std = mean_std(F2_all)
        F3_mean, F3_std = mean_std(F3_all)

        summary[scenario] = {
            "num_runs": len(lst),
            "exec_mean": exec_mean,
            "exec_std": exec_std,
            "size_mean": size_mean,
            "size_std": size_std,
            "F1_mean": F1_mean,
            "F1_std": F1_std,
            "F2_mean": F2_mean,
            "F2_std": F2_std,
            "F3_mean": F3_mean,
            "F3_std": F3_std,
            "names": list({r.get("scenario_name", "") for r in lst}),
        }

    return summary


def print_file_level_summary(results):
    """각 result_*.json 파일별 간단 요약 출력"""
    if not results:
        print("[!] 읽어온 결과가 없습니다.")
        return

    print("\n================ 파일별 결과 요약 ================\n")
    print(f"{'파일명':30s}  {'Scen':4s}  {'Seed':4s}  {'Time[s]':8s}  {'ParetoSize':10s}")
    print("-" * 70)

    for r in results:
        fname = r.get("_filename", "unknown")
        scen = r.get("scenario", "NA")
        seed = r.get("seed", "NA")
        t = r.get("execution_time", 0.0)
        ps = len(r.get("final_pareto", []))
        print(f"{fname:30s}  {scen:4s}  {str(seed):4s}  {t:8.2f}  {ps:10d}")


def print_scenario_summary(summary):
    """시나리오(S1~S4)별 통계 요약 출력"""
    if not summary:
        print("[!] 시나리오 요약이 없습니다.")
        return

    print("\n================ 시나리오별 통계 요약 ================\n")
    for scen in sorted(summary.keys()):
        s = summary[scen]
        names = ", ".join(s["names"]) if s["names"] else ""
        print(f"[{scen}] {names}")
        print(f"  - 실험 횟수: {s['num_runs']}")
        print(f"  - 실행시간 평균 / 표준편차: {s['exec_mean']:.2f} s / {s['exec_std']:.2f} s")
        print(f"  - Pareto 크기 평균 / 표준편차: {s['size_mean']:.2f} / {s['size_std']:.2f}")
        print(f"  - F1 평균 / 표준편차: {s['F1_mean']:.2f} / {s['F1_std']:.2f}")
        print(f"  - F2 평균 / 표준편차: {s['F2_mean']:.2f} / {s['F2_std']:.2f}")
        print(f"  - F3 평균 / 표준편차: {s['F3_mean']:.2f} / {s['F3_std']:.2f}")
        print()


def main():
    print("📂 results 폴더에서 실험 결과(JSON)를 읽는 중...")
    results = load_all_results(RESULT_DIR)

    if not results:
        print("[!] 읽어올 결과 파일이 없습니다. run.py를 먼저 실행했는지 확인하세요.")
        return

    # 1) 파일별 요약 출력
    print_file_level_summary(results)

    # 2) 시나리오별 요약 출력
    summary = summarize_by_scenario(results)
    print_scenario_summary(summary)


if __name__ == "__main__":
    main()
