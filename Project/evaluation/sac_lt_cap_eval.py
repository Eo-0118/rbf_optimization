"""SAC를 R_p10 81명 + R_mean 160명에 평가 — 마지막 시험"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.train_ppo_v2 import RBFEnvMultiCondition

ROOT = PROJECT_ROOT
DATA = ROOT / "Data"
MODELS = ROOT / "models"
SEED = 42


def main():
    from stable_baselines3 import SAC

    print("[1/2] SAC 모델 + 적합 셀러 로드")
    model = SAC.load(str(MODELS / "sac" / "sac_final.zip"))

    var_class = pd.read_csv(DATA / "optimal_lt_cap_variance_classification.csv")
    eligible_p10 = var_class[var_class["R_p10_eligible"]].copy()
    mean_class = pd.read_csv(DATA / "optimal_lt_cap_classification.csv")
    eligible_mean = mean_class[mean_class["eligible"]].copy()
    print(f"  R_p10 {len(eligible_p10)} / R_mean {len(eligible_mean)}")

    def evaluate(subset_df, L_col, cap_col, label):
        env = RBFEnvMultiCondition(
            seller_ids=subset_df["seller_id"].tolist(), seed=SEED + 100,
            eta=1.0, eta_proportional=False, use_lct_state=True,
        )
        rows = []
        for _, row in subset_df.iterrows():
            obs, info_init = env.reset(options={
                "seller_id": row["seller_id"],
                "L_override": row[L_col],
                "cap_override": row[cap_col],
            })
            burden_sum, burden_count = 0.0, 0
            hh_v, hh_amt, hh_ratio_max = 0, 0.0, 0.0
            while True:
                action, _ = model.predict(obs, deterministic=True)
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
            rows.append(dict(
                seller_id=row["seller_id"], type=info_init["type"],
                final_recovery=float(log[-1]["recovery_progress"]),
                completed=bool(log[-1]["recovery_progress"] >= 1.0),
                burden_mean=burden_sum / max(burden_count, 1),
                household_violation_count=hh_v,
                household_violation_amount_total=hh_amt,
                household_violation_ratio_max=hh_ratio_max,
            ))
        df = pd.DataFrame(rows)
        return dict(
            policy="SAC", subset=label, n=len(df),
            completion_rate=float(df["completed"].mean() * 100),
            mean_burden=float(df["burden_mean"].mean()),
            household_safe_rate=float((df["household_violation_count"] == 0).mean() * 100),
            mean_violation_amount=float(df["household_violation_amount_total"].mean()),
            mean_violation_ratio_max=float(df["household_violation_ratio_max"].mean()),
            p95_violation_ratio_max=float(df["household_violation_ratio_max"].quantile(0.95)),
            max_violation_ratio_max=float(df["household_violation_ratio_max"].max()),
        )

    print(f"\n[2/2] 평가")
    s_p10 = evaluate(eligible_p10, "R_p10_L_star", "R_p10_cap_star", "R_p10")
    s_mean = evaluate(eligible_mean, "L_star", "cap_star", "R_mean")

    print(f"\n=== R_p10 적합 {s_p10['n']}명 ===")
    for k, v in s_p10.items():
        if isinstance(v, float):
            print(f"  {k:30s}: {v:.3f}")

    print(f"\n=== R_mean 적합 {s_mean['n']}명 ===")
    for k, v in s_mean.items():
        if isinstance(v, float):
            print(f"  {k:30s}: {v:.3f}")

    # CVaR 비교
    print(f"\n=== R_p10 핵심 비교: SAC vs CVaR vs PPO base ===")
    print(f"  {'':22s} {'SAC':>10s} {'CVaR':>10s} {'PPO base':>10s}")
    print(f"  {'Completion %':22s} {s_p10['completion_rate']:>9.1f}% "
          f"{'100.0':>9s}% {'50.6':>9s}%")
    print(f"  {'단일월 최대비':22s} {s_p10['mean_violation_ratio_max']:>10.3f} "
          f"{'0.99':>10s} {'1.61':>10s}")
    print(f"  {'P95 최대비':22s} {s_p10['p95_violation_ratio_max']:>10.3f} "
          f"{'1.02':>10s} {'4.15':>10s}")

    if s_p10["completion_rate"] >= 90 and s_p10["mean_violation_ratio_max"] < 1.5:
        verdict = "⭐⭐ SAC가 CVaR 수준 달성"
    elif s_p10["completion_rate"] >= 80:
        verdict = "✅ 균형 정책"
    elif s_p10["mean_violation_ratio_max"] < 0.5:
        verdict = "안전 절대 (회수 약함)"
    else:
        verdict = "⚠️ CVaR 못 따라감"
    print(f"\n  결론: {verdict}")

    pd.DataFrame([s_p10, s_mean]).to_csv(DATA / "sac_lt_cap_compare.csv", index=False)
    (DATA / "sac_lt_cap_summary.json").write_text(
        json.dumps({"p10": s_p10, "mean": s_mean}, indent=2, ensure_ascii=False, default=str))
    print(f"\n  [save] sac_lt_cap_compare.csv + sac_lt_cap_summary.json")


if __name__ == "__main__":
    main()
