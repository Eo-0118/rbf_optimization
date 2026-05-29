"""Day 11: 사전 적합도 필터 정식 통합 평가

배경 (Day 8 발견):
  "Reward shaping (X1, X2, η 강화)으로 회수율+침해 빈도+침해 강도 동시 최적화 불가능"
  원인: 합성 영세 셀러 대다수가 영업이익 < 가계비 구조

해결:
  Reward 조정이 아닌 **사전 적합도 평가 (대출 거절)** 로 부적합 셀러 제외 후 PPO 적용

3-Tier 분류:
  Tier A (적합): coverage_ratio = E[월영업이익] / L_personal ≥ 2.0  → RBF 허용
  Tier B (경계): 1.0 ≤ coverage_ratio < 2.0                          → 조심
  Tier C (부적합): coverage_ratio < 1.0                                → RBF 거절

평가 흐름:
  1. cohort_kr_v2의 1,302명을 3-Tier 분류
  2. Tier별 분포 + 셀러 특성 (type, 평균매출) 분석
  3. Tier A 셀러만 5-정책 (Fixed/CVaR/PPO 변형) 평가
  4. Tier 종합: 거절율 + 적합 셀러에서 PPO 성능 측정

산출:
  Data/eligibility_filter_classification.csv (1,302명 × tier)
  Data/eligibility_filter_summary.json (tier별 통계)
  Data/eligibility_tier_a_policy_compare.csv (Tier A 셀러 5-정책 비교)
  Data/eligibility_filter_analysis.png
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

# 본 연구 적합도 임계값
COVERAGE_TIER_A = 2.0   # ≥ 2.0: 적합
COVERAGE_TIER_B = 1.0   # 1.0~2.0: 경계
                         # < 1.0: 부적합

DEFAULT_M_I = 0.10
DEFAULT_L_PERSONAL = 128.21


class PPOPolicy:
    def __init__(self, model_path: Path, name: str):
        from stable_baselines3 import PPO
        self.model = PPO.load(str(model_path))
        self.name = name

    def predict(self, state):
        action, _ = self.model.predict(state, deterministic=True)
        return action


def classify_tier(coverage_ratio: float) -> str:
    if coverage_ratio >= COVERAGE_TIER_A:
        return "A_적합"
    elif coverage_ratio >= COVERAGE_TIER_B:
        return "B_경계"
    else:
        return "C_부적합"


def run_episode(env, policy, sid):
    obs, info_init = env.reset(options={"seller_id": sid})
    total_reward = 0.0
    burden_sum, burden_count = 0.0, 0
    hh_v = 0
    hh_amt = 0.0
    hh_ratio_max = 0.0
    while True:
        action = policy.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if info["burden"] > 0:
            burden_sum += info["burden"]
            burden_count += 1
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
        burden_months=burden_count,
        household_violation_count=hh_v,
        household_violation_amount_total=hh_amt,
        household_violation_ratio_max=hh_ratio_max,
        total_reward=total_reward,
    )


def main():
    print("[1/5] Cohort 로드 + 셀러별 coverage_ratio 계산")
    env = RBFEnv(seed=SEED)
    sellers_meta = []
    for sid, info in env.sellers.items():
        avg_rev = info["mean_rev"]
        avg_operating_profit = avg_rev * DEFAULT_M_I
        coverage = avg_operating_profit / DEFAULT_L_PERSONAL
        sellers_meta.append({
            "seller_id": sid,
            "type": info["type"],
            "mean_rev": avg_rev,
            "operating_profit_mean": avg_operating_profit,
            "coverage_ratio": coverage,
            "tier": classify_tier(coverage),
        })
    meta_df = pd.DataFrame(sellers_meta)
    meta_df.to_csv(DATA / "eligibility_filter_classification.csv", index=False)
    print(f"  전체: {len(meta_df)} 셀러")

    print(f"\n[2/5] Tier 분포 분석")
    tier_dist = meta_df["tier"].value_counts().sort_index()
    print(tier_dist)
    print()
    print(f"  Tier별 평균 매출/영업이익/coverage:")
    for tier in sorted(meta_df["tier"].unique()):
        sub = meta_df[meta_df["tier"] == tier]
        print(f"  {tier:12s} n={len(sub):4d} ({len(sub)/len(meta_df)*100:5.1f}%) "
              f"avg_rev {sub['mean_rev'].mean():7.0f}만 "
              f"avg_OP {sub['operating_profit_mean'].mean():6.0f}만 "
              f"coverage {sub['coverage_ratio'].mean():.2f}")

    print(f"\n  Tier × Type 분포:")
    print(pd.crosstab(meta_df["tier"], meta_df["type"]))

    # Tier A 셀러 추출
    tier_a_ids = meta_df[meta_df["tier"] == "A_적합"]["seller_id"].tolist()
    print(f"\n  Tier A 적합 셀러: {len(tier_a_ids)} ({len(tier_a_ids)/len(meta_df)*100:.1f}%)")

    if len(tier_a_ids) < 5:
        print("\n⚠ Tier A 셀러 너무 적음. 임계값 검토 필요.")
        return

    print(f"\n[3/5] Tier A 셀러에 5-정책 평가")
    policies_std = {
        "Fixed-0.15": FixedRatePolicy(env, rate=0.15),
        "CVaR": CVaRPolicy(env),
        "PPO_base": PPOPolicy(MODELS / "ppo" / "ppo_final.zip", "PPO_base"),
        "PPO_eta_strong": PPOPolicy(MODELS / "ppo_eta_strong" / "ppo_final.zip", "PPO_eta_strong"),
        "PPO_x1_only": PPOPolicy(MODELS / "ppo_x1_only" / "ppo_final.zip", "PPO_x1_only"),
    }

    summaries_tier_a = []
    all_tier_a_results = {}
    for name, pol in policies_std.items():
        print(f"  {name} Tier A 평가 중...")
        df_a = pd.DataFrame([run_episode(env, pol, sid) for sid in tier_a_ids])
        all_tier_a_results[name] = df_a
        summaries_tier_a.append(dict(
            policy=name, subset="Tier_A_only", n=len(df_a),
            completion_rate=float(df_a["completed"].mean() * 100),
            mean_burden=float(df_a["burden_mean"].mean()),
            mean_household_violation_months=float(df_a["household_violation_count"].mean()),
            mean_violation_amount_total=float(df_a["household_violation_amount_total"].mean()),
            mean_violation_ratio_max=float(df_a["household_violation_ratio_max"].mean()),
            p95_violation_ratio_max=float(df_a["household_violation_ratio_max"].quantile(0.95)),
        ))

    tier_a_compare = pd.DataFrame(summaries_tier_a)
    tier_a_compare.to_csv(DATA / "eligibility_tier_a_policy_compare.csv", index=False)
    print(f"  [save] eligibility_tier_a_policy_compare.csv")

    print(f"\n=== Tier A 적합 셀러 {len(tier_a_ids)}명 정책 비교 ===")
    cols = ["policy", "completion_rate", "mean_burden",
            "mean_household_violation_months", "mean_violation_amount_total",
            "mean_violation_ratio_max", "p95_violation_ratio_max"]
    pd.set_option('display.width', 200)
    print(tier_a_compare[cols].to_string(index=False))

    print(f"\n[4/5] Tier 전체 평가 결과 (필터 + PPO 결합 vs 단순 PPO)")
    # Tier A의 PPO_base 결과
    ta_ppo = next(s for s in summaries_tier_a if s["policy"] == "PPO_base")
    # 전체 1,302명의 PPO_base (기존 결과)
    full_compare = pd.read_csv(DATA / "policy_full_comparison.csv")
    full_ppo = full_compare[full_compare["policy"] == "PPO"].iloc[0]

    n_tier_a = len(tier_a_ids)
    n_total = len(meta_df)
    rejection_rate = (n_total - n_tier_a) / n_total * 100

    print(f"\n  단순 PPO (전체 {n_total}명):")
    print(f"    Completion: {full_ppo['completion_rate']:.1f}%")
    print(f"    누적 침해액: {full_ppo['mean_violation_amount_total']:.0f}만")
    print(f"    단일월 최대비: {full_ppo['mean_violation_ratio_max']:.2f}")

    print(f"\n  필터+PPO (Tier A {n_tier_a}명, 거절 {rejection_rate:.1f}%):")
    print(f"    Completion: {ta_ppo['completion_rate']:.1f}% ({ta_ppo['completion_rate'] - full_ppo['completion_rate']:+.1f}%p)")
    print(f"    누적 침해액: {ta_ppo['mean_violation_amount_total']:.0f}만 ({ta_ppo['mean_violation_amount_total'] - full_ppo['mean_violation_amount_total']:+.0f})")
    print(f"    단일월 최대비: {ta_ppo['mean_violation_ratio_max']:.2f} ({ta_ppo['mean_violation_ratio_max'] - full_ppo['mean_violation_ratio_max']:+.2f})")

    summary = {
        "config": {
            "m_i": DEFAULT_M_I, "L_personal": DEFAULT_L_PERSONAL,
            "tier_a_threshold": COVERAGE_TIER_A,
            "tier_b_threshold": COVERAGE_TIER_B,
        },
        "tier_distribution": {
            tier: {
                "n": int(len(meta_df[meta_df["tier"] == tier])),
                "pct": float(len(meta_df[meta_df["tier"] == tier]) / len(meta_df) * 100),
                "avg_revenue": float(meta_df[meta_df["tier"] == tier]["mean_rev"].mean()),
                "avg_coverage": float(meta_df[meta_df["tier"] == tier]["coverage_ratio"].mean()),
            }
            for tier in sorted(meta_df["tier"].unique())
        },
        "rejection_rate": float(rejection_rate),
        "tier_a_n": int(n_tier_a),
        "filter_plus_ppo_vs_simple_ppo": {
            "completion": {
                "simple_ppo": float(full_ppo["completion_rate"]),
                "filter_ppo": ta_ppo["completion_rate"],
                "delta": ta_ppo["completion_rate"] - float(full_ppo["completion_rate"]),
            },
            "violation_amount": {
                "simple_ppo": float(full_ppo["mean_violation_amount_total"]),
                "filter_ppo": ta_ppo["mean_violation_amount_total"],
                "delta": ta_ppo["mean_violation_amount_total"] - float(full_ppo["mean_violation_amount_total"]),
            },
            "violation_ratio_max": {
                "simple_ppo": float(full_ppo["mean_violation_ratio_max"]),
                "filter_ppo": ta_ppo["mean_violation_ratio_max"],
                "delta": ta_ppo["mean_violation_ratio_max"] - float(full_ppo["mean_violation_ratio_max"]),
            },
        },
        "tier_a_policy_compare": summaries_tier_a,
    }
    (DATA / "eligibility_filter_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n  [save] eligibility_filter_summary.json")

    print(f"\n[5/5] 시각화")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. Coverage ratio 분포 + tier 임계값
    ax = axes[0, 0]
    ax.hist(meta_df["coverage_ratio"].clip(upper=5), bins=50,
            color="steelblue", alpha=0.7, edgecolor="white")
    ax.axvline(COVERAGE_TIER_A, color="green", linestyle="--", linewidth=2,
               label=f"Tier A (≥{COVERAGE_TIER_A})")
    ax.axvline(COVERAGE_TIER_B, color="orange", linestyle="--", linewidth=2,
               label=f"Tier B (≥{COVERAGE_TIER_B})")
    ax.set_xlabel("Coverage ratio = E[영업이익] / L_personal")
    ax.set_ylabel("셀러 수")
    ax.set_title("셀러별 적합도 분포 (1,302명)")
    ax.legend(); ax.grid(alpha=0.3)

    # 2. Tier 분포 pie
    ax = axes[0, 1]
    tier_counts = meta_df["tier"].value_counts().sort_index()
    colors_t = ["lightgreen", "orange", "salmon"]
    wedges, texts, autotexts = ax.pie(tier_counts.values, labels=tier_counts.index,
                                       colors=colors_t, autopct="%1.1f%%",
                                       startangle=90)
    ax.set_title(f"3-Tier 분포 (n={len(meta_df)})")

    # 3. Tier × Type cross
    ax = axes[0, 2]
    cross = pd.crosstab(meta_df["tier"], meta_df["type"])
    cross.plot(kind="bar", stacked=True, ax=ax,
               color=["steelblue", "mediumseagreen", "crimson",
                      "darkorange", "gray", "lightgray"])
    ax.set_title("Tier × Type 분포")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3, axis="y")

    # 4. Tier A vs 전체 PPO 비교 (completion + 침해)
    ax = axes[1, 0]
    metrics = ["Completion %", "침해 월수", "단일월 최대비 × 10"]
    simple_vals = [
        float(full_ppo["completion_rate"]),
        float(full_ppo["mean_household_violation_months"]),
        float(full_ppo["mean_violation_ratio_max"]) * 10,
    ]
    filter_vals = [
        ta_ppo["completion_rate"],
        ta_ppo["mean_household_violation_months"],
        ta_ppo["mean_violation_ratio_max"] * 10,
    ]
    x_pos = np.arange(len(metrics))
    width = 0.35
    ax.bar(x_pos - width/2, simple_vals, width, label="단순 PPO (전체 1,302)", color="crimson", alpha=0.8)
    ax.bar(x_pos + width/2, filter_vals, width, label=f"필터+PPO (Tier A {n_tier_a})", color="mediumseagreen", alpha=0.8)
    for i, (s, f) in enumerate(zip(simple_vals, filter_vals)):
        ax.text(i - width/2, s + 1, f"{s:.1f}", ha="center", fontsize=8)
        ax.text(i + width/2, f + 1, f"{f:.1f}", ha="center", fontsize=8)
    ax.set_xticks(x_pos); ax.set_xticklabels(metrics, fontsize=9)
    ax.set_title("필터+PPO vs 단순 PPO")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

    # 5. Tier A 5-정책 비교 (completion)
    ax = axes[1, 1]
    pol_names = [s["policy"] for s in summaries_tier_a]
    completions = [s["completion_rate"] for s in summaries_tier_a]
    colors_p = ["steelblue", "darkorange", "crimson", "darkgreen", "purple"]
    bars = ax.bar(pol_names, completions, color=colors_p, alpha=0.8)
    for b, v in zip(bars, completions):
        ax.text(b.get_x() + b.get_width()/2, v + 1, f"{v:.0f}%",
                ha="center", fontweight="bold")
    ax.set_title(f"Tier A {n_tier_a}명 정책별 Completion")
    ax.tick_params(axis="x", rotation=15, labelsize=9)
    ax.set_ylabel("Completion %"); ax.grid(alpha=0.3, axis="y")

    # 6. Tier A 5-정책 비교 (침해 강도)
    ax = axes[1, 2]
    ratios = [s["mean_violation_ratio_max"] for s in summaries_tier_a]
    bars = ax.bar(pol_names, ratios, color=colors_p, alpha=0.8)
    for b, v in zip(bars, ratios):
        ax.text(b.get_x() + b.get_width()/2, v + max(ratios)*0.02,
                f"{v:.2f}", ha="center", fontweight="bold")
    ax.set_title(f"Tier A {n_tier_a}명 정책별 단일월 최대 침해비")
    ax.tick_params(axis="x", rotation=15, labelsize=9)
    ax.set_ylabel("최대 침해비 (가계비 1배 기준)")
    ax.grid(alpha=0.3, axis="y")

    plt.suptitle(f"Day 11: 사전 적합도 필터 통합 — Tier A {n_tier_a}명 (거절 {rejection_rate:.0f}%) 정책 비교",
                 fontsize=13, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(DATA / "eligibility_filter_analysis.png", dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  [save] eligibility_filter_analysis.png")

    print("\n=== Day 11 완료 ===")


if __name__ == "__main__":
    main()
