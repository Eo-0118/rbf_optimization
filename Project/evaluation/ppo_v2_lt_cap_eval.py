"""Day 12 (Phase D-4): PPO v2/v2b를 (L*, cap*) 환경에서 평가

배경:
  PPO v2: state 확장 + multi-condition + 비례 페널티 → 기본 환경에서 completion 0% (실패)
  PPO v2b: state 확장 + multi-condition + 기본 페널티 → 기본 환경에서 completion 87.7%

진짜 시험: R_p10 적합 81명 + R_mean 적합 160명에서 (L*, cap*) 환경 평가
목표: PPO v2/v2b가 CVaR을 능가하는가?

산출:
  Data/ppo_v2_lt_cap_compare.csv
  Data/ppo_v2_lt_cap_summary.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.rbf_env import RBFEnv
from envs.baselines import FixedRatePolicy, CVaRPolicy

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

ROOT = PROJECT_ROOT
DATA = ROOT / "Data"
MODELS = ROOT / "models"
SEED = 42


class PPOPolicy:
    def __init__(self, model_path: Path, name: str):
        from stable_baselines3 import PPO
        self.model = PPO.load(str(model_path))
        self.name = name

    def predict(self, state):
        action, _ = self.model.predict(state, deterministic=True)
        return action


def run_episode(env, policy, sid, L_override=None, cap_override=None):
    options = {"seller_id": sid}
    if L_override is not None:
        options["L_override"] = L_override
    if cap_override is not None:
        options["cap_override"] = cap_override
    obs, info_init = env.reset(options=options)
    burden_sum, burden_count = 0.0, 0
    hh_v, hh_amt, hh_ratio_max = 0, 0.0, 0.0
    while True:
        action = policy.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        if info["burden"] > 0:
            burden_sum += info["burden"]; burden_count += 1
        if info.get("household_violated", False):
            hh_v += 1
        hh_amt += info.get("household_violation_amount", 0.0)
        hh_ratio_max = max(hh_ratio_max, info.get("household_violation_ratio", 0.0))
        if terminated:
            break
    log = env.get_episode_log()
    return dict(
        seller_id=info_init["seller_id"], type=info_init["type"],
        final_recovery=float(log[-1]["recovery_progress"]),
        completed=bool(log[-1]["recovery_progress"] >= 1.0),
        burden_mean=burden_sum / max(burden_count, 1),
        household_violation_count=hh_v,
        household_violation_amount_total=hh_amt,
        household_violation_ratio_max=hh_ratio_max,
    )


def summarize(df, name, n_eligible_total):
    return dict(
        policy=name, n=len(df),
        completion_rate=float(df["completed"].mean() * 100),
        mean_burden=float(df["burden_mean"].mean()),
        mean_household_violation_months=float(df["household_violation_count"].mean()),
        household_safe_rate=float((df["household_violation_count"] == 0).mean() * 100),
        mean_violation_amount=float(df["household_violation_amount_total"].mean()),
        mean_violation_ratio_max=float(df["household_violation_ratio_max"].mean()),
        p95_violation_ratio_max=float(df["household_violation_ratio_max"].quantile(0.95)),
        max_violation_ratio_max=float(df["household_violation_ratio_max"].max()),
    )


def main():
    # cohort_kr_v2의 셀러 사용 (학습 셀러)
    # R_p10 적합 셀러 추출 (optimal_lt_cap_variance_classification.csv 활용)
    print("[1/4] R_p10 / R_mean 적합 셀러 로드")
    var_class = pd.read_csv(DATA / "optimal_lt_cap_variance_classification.csv")
    mean_class = pd.read_csv(DATA / "optimal_lt_cap_classification.csv")

    eligible_p10 = var_class[var_class["R_p10_eligible"]].copy()
    eligible_mean = mean_class[mean_class["eligible"]].copy()
    print(f"  R_p10 적합: {len(eligible_p10)}명 (R_p10 보수)")
    print(f"  R_mean 적합: {len(eligible_mean)}명 (R_mean 낙관)")

    # 정책 환경 — use_lct_state=True (v2/v2b PPO용)
    env_lct = RBFEnv(seed=SEED, use_lct_state=True)
    # 정책 환경 — use_lct_state=False (base PPO/CVaR/Fixed용)
    env_base = RBFEnv(seed=SEED)

    print(f"\n[2/4] 정책 로드")
    policies_lct = {}
    policies_base = {}

    # PPO v2/v2b (state 확장 필요)
    for name, path in [("PPO_v2", MODELS / "ppo_v2" / "ppo_final.zip"),
                        ("PPO_v2b", MODELS / "ppo_v2b" / "ppo_final.zip")]:
        if path.exists():
            policies_lct[name] = PPOPolicy(path, name)

    # PPO base / x1_only (state 13/16차원, 기본 env)
    for name, path in [("PPO_base", MODELS / "ppo" / "ppo_final.zip"),
                        ("PPO_x1_only", MODELS / "ppo_x1_only" / "ppo_final.zip")]:
        if path.exists():
            policies_base[name] = PPOPolicy(path, name)

    # 정적 정책 (기본 env)
    policies_base["Fixed-0.15"] = FixedRatePolicy(env_base, rate=0.15)
    policies_base["CVaR"] = CVaRPolicy(env_base)

    print(f"  use_lct_state PPO: {list(policies_lct.keys())}")
    print(f"  기본 env 정책: {list(policies_base.keys())}")

    # 평가
    print(f"\n[3/4] R_p10 적합 {len(eligible_p10)}명 평가")
    summaries_p10 = []
    for name, pol in {**policies_base, **policies_lct}.items():
        # state 차원 다른 env 선택
        use_lct = name in policies_lct
        env_use = env_lct if use_lct else env_base
        # L_star, cap_star 컬럼명 다름
        if "R_p10_L_star" in eligible_p10.columns:
            L_col, cap_col = "R_p10_L_star", "R_p10_cap_star"
        else:
            L_col, cap_col = "L_star", "cap_star"

        rows = []
        for _, row in eligible_p10.iterrows():
            r = run_episode(env_use, pol, row["seller_id"],
                            L_override=row[L_col], cap_override=row[cap_col])
            rows.append(r)
        df = pd.DataFrame(rows)
        summaries_p10.append(summarize(df, name, len(eligible_p10)))

    p10_df = pd.DataFrame(summaries_p10)
    p10_df.to_csv(DATA / "ppo_v2_lt_cap_p10_compare.csv", index=False)

    print(f"\n=== R_p10 적합 {len(eligible_p10)}명 정책 비교 ===")
    cols = ["policy", "completion_rate", "mean_burden",
            "household_safe_rate", "mean_violation_amount",
            "mean_violation_ratio_max", "p95_violation_ratio_max",
            "max_violation_ratio_max"]
    pd.set_option('display.width', 200)
    print(p10_df[cols].to_string(index=False))

    # R_mean 평가 (160명)
    print(f"\n[4/4] R_mean 적합 {len(eligible_mean)}명 평가")
    summaries_mean = []
    for name, pol in {**policies_base, **policies_lct}.items():
        use_lct = name in policies_lct
        env_use = env_lct if use_lct else env_base
        rows = []
        for _, row in eligible_mean.iterrows():
            r = run_episode(env_use, pol, row["seller_id"],
                            L_override=row["L_star"], cap_override=row["cap_star"])
            rows.append(r)
        df = pd.DataFrame(rows)
        summaries_mean.append(summarize(df, name, len(eligible_mean)))

    mean_df = pd.DataFrame(summaries_mean)
    mean_df.to_csv(DATA / "ppo_v2_lt_cap_mean_compare.csv", index=False)

    print(f"\n=== R_mean 적합 {len(eligible_mean)}명 정책 비교 ===")
    print(mean_df[cols].to_string(index=False))

    # 종합 저장
    summary_json = {
        "config": {
            "R_p10_n": int(len(eligible_p10)),
            "R_mean_n": int(len(eligible_mean)),
        },
        "p10_results": summaries_p10,
        "mean_results": summaries_mean,
    }
    (DATA / "ppo_v2_lt_cap_summary.json").write_text(
        json.dumps(summary_json, indent=2, ensure_ascii=False, default=str))

    # 핵심 비교 출력
    print(f"\n=== 핵심 비교 (R_p10 81명, CVaR vs PPO 변형) ===")
    cvar_p10 = next(s for s in summaries_p10 if s["policy"] == "CVaR")
    print(f"  {'정책':18s} {'Completion':>12s} {'최대비':>10s} {'평가':>14s}")
    for s in summaries_p10:
        verdict = "⭐ CVaR 능가" if s["completion_rate"] > cvar_p10["completion_rate"] and s["mean_violation_ratio_max"] < cvar_p10["mean_violation_ratio_max"] else \
                  "✅ 균형" if abs(s["completion_rate"] - cvar_p10["completion_rate"]) < 10 else \
                  "회수 ↑" if s["completion_rate"] > cvar_p10["completion_rate"] else \
                  "안전 ↑" if s["mean_violation_ratio_max"] < cvar_p10["mean_violation_ratio_max"] else \
                  "⚠️"
        print(f"  {s['policy']:18s} {s['completion_rate']:11.1f}% {s['mean_violation_ratio_max']:10.2f} {verdict:>14s}")

    print("\n=== Phase D-4 완료 ===")


if __name__ == "__main__":
    main()
