"""Phase E-1 평가: CVaR-PPO Hybrid 를 R_p10 81명 + R_mean 160명에 평가

학습된 모델: Project/models/ppo_cvar_hybrid/ppo_final.zip
환경: use_lct_state=True + CVaRHybridWrapper

산출: Data/ppo_hybrid_lt_cap_compare.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.rbf_env import RBFEnv
from agents.train_ppo_v2 import RBFEnvMultiCondition
from agents.train_ppo_cvar_hybrid import CVaRHybridWrapper, load_cvar_lookup

ROOT = PROJECT_ROOT
DATA = ROOT / "Data"
MODELS = ROOT / "models"
SEED = 42


def main():
    from stable_baselines3 import PPO

    print("[1/3] Hybrid 모델 + 적합 셀러 로드")
    model = PPO.load(str(MODELS / "ppo_cvar_hybrid" / "ppo_final.zip"))
    cvar_lookup = load_cvar_lookup()
    print(f"  Hybrid 모델 로드 OK")
    print(f"  CVaR r* lookup: {len(cvar_lookup)} 셀러")

    var_class = pd.read_csv(DATA / "optimal_lt_cap_variance_classification.csv")
    eligible_p10 = var_class[var_class["R_p10_eligible"]].copy()
    print(f"  R_p10 적합: {len(eligible_p10)}명")

    mean_class = pd.read_csv(DATA / "optimal_lt_cap_classification.csv")
    eligible_mean = mean_class[mean_class["eligible"]].copy()
    print(f"  R_mean 적합: {len(eligible_mean)}명")

    print(f"\n[2/3] R_p10 적합 평가 (Hybrid 정책)")
    rows_p10 = []
    for _, row in eligible_p10.iterrows():
        env = RBFEnvMultiCondition(
            seller_ids=[row["seller_id"]], seed=SEED + 100,
            eta=1.0, eta_proportional=False, use_lct_state=True,
        )
        wrapper = CVaRHybridWrapper(env, cvar_lookup)
        obs, info_init = wrapper.reset(options={
            "seller_id": row["seller_id"],
            "L_override": row["R_p10_L_star"],
            "cap_override": row["R_p10_cap_star"],
        })
        burden_sum, burden_count = 0.0, 0
        hh_v, hh_amt, hh_ratio_max = 0, 0.0, 0.0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = wrapper.step(action)
            if info["burden"] > 0:
                burden_sum += info["burden"]; burden_count += 1
            if info.get("household_violated", False):
                hh_v += 1
            hh_amt += info.get("household_violation_amount", 0.0)
            hh_ratio_max = max(hh_ratio_max, info.get("household_violation_ratio", 0.0))
            if terminated:
                break
        log = env.get_episode_log()
        rows_p10.append(dict(
            seller_id=row["seller_id"], type=info_init["type"],
            final_recovery=float(log[-1]["recovery_progress"]),
            completed=bool(log[-1]["recovery_progress"] >= 1.0),
            burden_mean=burden_sum / max(burden_count, 1),
            household_violation_count=hh_v,
            household_violation_amount_total=hh_amt,
            household_violation_ratio_max=hh_ratio_max,
        ))
    p10_df = pd.DataFrame(rows_p10)

    summary_p10 = dict(
        policy="PPO_cvar_hybrid", subset="R_p10", n=len(p10_df),
        completion_rate=float(p10_df["completed"].mean() * 100),
        mean_burden=float(p10_df["burden_mean"].mean()),
        household_safe_rate=float((p10_df["household_violation_count"] == 0).mean() * 100),
        mean_violation_amount=float(p10_df["household_violation_amount_total"].mean()),
        mean_violation_ratio_max=float(p10_df["household_violation_ratio_max"].mean()),
        p95_violation_ratio_max=float(p10_df["household_violation_ratio_max"].quantile(0.95)),
        max_violation_ratio_max=float(p10_df["household_violation_ratio_max"].max()),
    )

    print(f"\n  Hybrid R_p10 81명:")
    for k, v in summary_p10.items():
        if isinstance(v, float):
            print(f"    {k:30s}: {v:.3f}")

    print(f"\n[3/3] R_mean 160명 평가")
    rows_mean = []
    for _, row in eligible_mean.iterrows():
        env = RBFEnvMultiCondition(
            seller_ids=[row["seller_id"]], seed=SEED + 100,
            eta=1.0, eta_proportional=False, use_lct_state=True,
        )
        wrapper = CVaRHybridWrapper(env, cvar_lookup)
        obs, info_init = wrapper.reset(options={
            "seller_id": row["seller_id"],
            "L_override": row["L_star"],
            "cap_override": row["cap_star"],
        })
        burden_sum, burden_count = 0.0, 0
        hh_v, hh_amt, hh_ratio_max = 0, 0.0, 0.0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = wrapper.step(action)
            if info["burden"] > 0:
                burden_sum += info["burden"]; burden_count += 1
            if info.get("household_violated", False):
                hh_v += 1
            hh_amt += info.get("household_violation_amount", 0.0)
            hh_ratio_max = max(hh_ratio_max, info.get("household_violation_ratio", 0.0))
            if terminated:
                break
        log = env.get_episode_log()
        rows_mean.append(dict(
            seller_id=row["seller_id"], type=info_init["type"],
            final_recovery=float(log[-1]["recovery_progress"]),
            completed=bool(log[-1]["recovery_progress"] >= 1.0),
            burden_mean=burden_sum / max(burden_count, 1),
            household_violation_count=hh_v,
            household_violation_amount_total=hh_amt,
            household_violation_ratio_max=hh_ratio_max,
        ))
    mean_df = pd.DataFrame(rows_mean)

    summary_mean = dict(
        policy="PPO_cvar_hybrid", subset="R_mean", n=len(mean_df),
        completion_rate=float(mean_df["completed"].mean() * 100),
        mean_burden=float(mean_df["burden_mean"].mean()),
        household_safe_rate=float((mean_df["household_violation_count"] == 0).mean() * 100),
        mean_violation_amount=float(mean_df["household_violation_amount_total"].mean()),
        mean_violation_ratio_max=float(mean_df["household_violation_ratio_max"].mean()),
        p95_violation_ratio_max=float(mean_df["household_violation_ratio_max"].quantile(0.95)),
        max_violation_ratio_max=float(mean_df["household_violation_ratio_max"].max()),
    )

    print(f"\n  Hybrid R_mean 160명:")
    for k, v in summary_mean.items():
        if isinstance(v, float):
            print(f"    {k:30s}: {v:.3f}")

    # 저장
    pd.DataFrame([summary_p10, summary_mean]).to_csv(
        DATA / "ppo_hybrid_lt_cap_compare.csv", index=False)
    (DATA / "ppo_hybrid_lt_cap_summary.json").write_text(
        json.dumps({"p10": summary_p10, "mean": summary_mean},
                    indent=2, ensure_ascii=False, default=str))

    # 핵심 비교 — CVaR 단독 vs Hybrid
    print(f"\n=== 핵심 비교: CVaR 단독 vs CVaR-PPO Hybrid (R_p10 81명) ===")
    print(f"  {'':22s} {'CVaR 단독':>12s} {'Hybrid':>12s}")
    print(f"  {'Completion %':22s} {'100.0':>11s}% {summary_p10['completion_rate']:>11.1f}%")
    print(f"  {'단일월 최대비':22s} {'0.99':>12s} {summary_p10['mean_violation_ratio_max']:>12.3f}")
    print(f"  {'P95 최대비':22s} {'1.02':>12s} {summary_p10['p95_violation_ratio_max']:>12.3f}")
    print(f"  {'Max 최대비':22s} {'1.06':>12s} {summary_p10['max_violation_ratio_max']:>12.3f}")

    verdict = "⭐ CVaR 능가" if (summary_p10["completion_rate"] >= 100 and
                                summary_p10["mean_violation_ratio_max"] < 0.99) else \
              "✅ 동등 수준" if abs(summary_p10["completion_rate"] - 100) < 5 else \
              "⚠️ CVaR 못 따라감"
    print(f"\n  결론: {verdict}")
    print(f"\n=== 평가 완료 ===")


if __name__ == "__main__":
    main()
