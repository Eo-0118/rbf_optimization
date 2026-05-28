"""Day 5: 3-PPO 변형 + 베이스라인 종합 비교 (전체 1,302명 동일 조건)

대상:
  1. Fixed-0.15 (정적 baseline)
  2. CVaR (정적)
  3. PPO 기존 (eta=1.0, forecast X)
  4. PPO 분포 통합 (forecast state, eta=1.0)
  5. PPO eta 강화 (eta=5.0, forecast X)

핵심 질문:
  - 분포 통합이 어떤 지표에서 도움이 되는가?
  - eta 강화가 침해 액도를 진짜 줄이는가?
  - 두 변형 중 어느 게 더 가치 있는가?

산출:
  - Data/ppo_3variants_comparison.csv (5정책 × 1,302명 통계)
  - Data/ppo_3variants_comparison.png
  - Data/ppo_3variants_by_type.csv
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


def run_episode(env, policy, seller_id):
    obs, info_init = env.reset(options={"seller_id": seller_id})
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


def evaluate(env, policy, sids):
    return pd.DataFrame([run_episode(env, policy, sid) for sid in sids])


def summarize(df, name):
    return {
        "policy": name,
        "n": int(len(df)),
        "completion_rate": float(df["completed"].mean() * 100),
        "mean_recovery": float(df["final_recovery"].mean()),
        "mean_burden": float(df["burden_mean"].mean()),
        "mean_household_violation_months": float(df["household_violation_count"].mean()),
        "household_violation_zero_rate": float((df["household_violation_count"] == 0).mean() * 100),
        "mean_violation_amount_total": float(df["household_violation_amount_total"].mean()),
        "p95_violation_amount_total": float(df["household_violation_amount_total"].quantile(0.95)),
        "mean_violation_ratio_max": float(df["household_violation_ratio_max"].mean()),
        "p95_violation_ratio_max": float(df["household_violation_ratio_max"].quantile(0.95)),
        "mean_reward": float(df["total_reward"].mean()),
    }


def main():
    print("[1/5] Env 초기화")
    # 표준 env (forecast 없음 — Fixed/CVaR/기존PPO/eta강화PPO 용)
    env_std = RBFEnv(seed=SEED)
    # forecast env (분포 통합 PPO 용)
    env_fc = RBFEnv(seed=SEED, use_forecast_state=True, forecast_lag=6)
    all_ids = list(env_std.sellers.keys())
    print(f"  전체 셀러: {len(all_ids)}")

    print(f"\n[2/5] 정책 로드")
    policies_std = {
        "Fixed-0.15": FixedRatePolicy(env_std, rate=0.15),
        "CVaR": CVaRPolicy(env_std),
        "PPO_base": PPOPolicy(MODELS / "ppo" / "ppo_final.zip", "PPO_base"),
        "PPO_eta_strong": PPOPolicy(MODELS / "ppo_eta_strong" / "ppo_final.zip", "PPO_eta_strong"),
    }
    policies_fc = {
        "PPO_forecast": PPOPolicy(MODELS / "ppo_forecast" / "ppo_final.zip", "PPO_forecast"),
    }
    print(f"  표준 env 정책: {list(policies_std.keys())}")
    print(f"  forecast env 정책: {list(policies_fc.keys())}")

    print(f"\n[3/5] 전체 1,302명 평가")
    all_results = {}
    summaries = []
    by_type_rows = []

    # 표준 env 평가
    for name, pol in policies_std.items():
        print(f"  {name} (표준 env) ...")
        df = evaluate(env_std, pol, all_ids)
        all_results[name] = df
        summaries.append(summarize(df, name))
        for typ, g in df.groupby("type"):
            by_type_rows.append(dict(policy=name, type=typ, n=len(g),
                                     completion_rate=g["completed"].mean()*100,
                                     mean_burden=g["burden_mean"].mean(),
                                     mean_violation_amount_total=g["household_violation_amount_total"].mean(),
                                     mean_violation_ratio_max=g["household_violation_ratio_max"].mean()))

    # forecast env 평가
    for name, pol in policies_fc.items():
        print(f"  {name} (forecast env) ...")
        df = evaluate(env_fc, pol, all_ids)
        all_results[name] = df
        summaries.append(summarize(df, name))
        for typ, g in df.groupby("type"):
            by_type_rows.append(dict(policy=name, type=typ, n=len(g),
                                     completion_rate=g["completed"].mean()*100,
                                     mean_burden=g["burden_mean"].mean(),
                                     mean_violation_amount_total=g["household_violation_amount_total"].mean(),
                                     mean_violation_ratio_max=g["household_violation_ratio_max"].mean()))

    print(f"\n[4/5] 저장")
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(DATA / "ppo_3variants_comparison.csv", index=False)
    pd.DataFrame(by_type_rows).to_csv(DATA / "ppo_3variants_by_type.csv", index=False)
    print(f"  [save] ppo_3variants_comparison.csv")
    print(f"  [save] ppo_3variants_by_type.csv")

    # 콘솔 출력
    print(f"\n=== 5-정책 종합 비교 (전체 1,302명) ===\n")
    cols = ["policy", "completion_rate", "mean_burden",
            "mean_household_violation_months",
            "mean_violation_amount_total", "p95_violation_amount_total",
            "mean_violation_ratio_max", "p95_violation_ratio_max"]
    pd.set_option('display.width', 200)
    pd.set_option('display.max_columns', 15)
    print(summary_df[cols].to_string(index=False))

    print(f"\n=== 핵심 분석 ===")
    base = next(s for s in summaries if s["policy"] == "PPO_base")
    fc = next(s for s in summaries if s["policy"] == "PPO_forecast")
    eta = next(s for s in summaries if s["policy"] == "PPO_eta_strong")
    fix = next(s for s in summaries if s["policy"] == "Fixed-0.15")

    print(f"\n  Q1: 분포 통합이 도움됐나? (PPO_forecast vs PPO_base)")
    print(f"    Completion: {base['completion_rate']:.1f}% → {fc['completion_rate']:.1f}% ({fc['completion_rate']-base['completion_rate']:+.1f}%p)")
    print(f"    누적 침해액: {base['mean_violation_amount_total']:.0f} → {fc['mean_violation_amount_total']:.0f} ({fc['mean_violation_amount_total']-base['mean_violation_amount_total']:+.0f}만)")
    print(f"    단일월 최대비: {base['mean_violation_ratio_max']:.2f} → {fc['mean_violation_ratio_max']:.2f}")

    print(f"\n  Q2: eta 강화가 침해 정도 줄였나? (PPO_eta_strong vs PPO_base)")
    print(f"    Completion: {base['completion_rate']:.1f}% → {eta['completion_rate']:.1f}% ({eta['completion_rate']-base['completion_rate']:+.1f}%p)")
    print(f"    누적 침해액: {base['mean_violation_amount_total']:.0f} → {eta['mean_violation_amount_total']:.0f} ({eta['mean_violation_amount_total']-base['mean_violation_amount_total']:+.0f}만)")
    print(f"    단일월 최대비: {base['mean_violation_ratio_max']:.2f} → {eta['mean_violation_ratio_max']:.2f}")

    print(f"\n  Q3: 최고 변형은? (vs Fixed baseline 기준 침해액 감소)")
    for name in ["PPO_base", "PPO_forecast", "PPO_eta_strong"]:
        s = next(x for x in summaries if x["policy"] == name)
        amt_red = (1 - s["mean_violation_amount_total"] / fix["mean_violation_amount_total"]) * 100
        comp_red = s["completion_rate"] - fix["completion_rate"]
        print(f"    {name:18s}: completion {comp_red:+5.1f}%p, 침해액 {amt_red:+5.1f}% 변화")

    print(f"\n[5/5] 시각화")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    pol_names = ["Fixed-0.15", "CVaR", "PPO_base", "PPO_forecast", "PPO_eta_strong"]
    colors = {"Fixed-0.15": "steelblue", "CVaR": "darkorange",
              "PPO_base": "crimson", "PPO_forecast": "purple", "PPO_eta_strong": "darkgreen"}
    x = np.arange(len(pol_names))
    s_map = {s["policy"]: s for s in summaries}

    def bar_plot(ax, key, title, fmt=".1f", suffix=""):
        vals = [s_map[p][key] for p in pol_names]
        bars = ax.bar(x, vals, color=[colors[p] for p in pol_names])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, v + max(vals)*0.02,
                    f"{v:{fmt}}{suffix}", ha="center", fontweight="bold", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(pol_names, rotation=20, fontsize=8)
        ax.set_title(title); ax.grid(alpha=0.3, axis="y")

    bar_plot(axes[0, 0], "completion_rate", "Completion %", ".1f", "%")
    bar_plot(axes[0, 1], "mean_burden", "Mean burden (사업)", ".3f")
    bar_plot(axes[0, 2], "mean_household_violation_months", "HH 침해 월수 (기존)", ".1f", "mo")
    bar_plot(axes[1, 0], "mean_violation_amount_total", "[A-2] 누적 침해액 (만원)", ".0f")
    bar_plot(axes[1, 1], "mean_violation_ratio_max", "[A-2] 단일월 최대 침해비", ".2f")
    bar_plot(axes[1, 2], "p95_violation_ratio_max", "[A-2] 상위 5% 셀러 최대비", ".2f")

    plt.suptitle("Day 5: 5-정책 종합 비교 (전체 1,302명)\n"
                 "PPO 변형: base / forecast 통합 / eta 강화",
                 fontsize=13, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(DATA / "ppo_3variants_comparison.png", dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  [save] ppo_3variants_comparison.png")
    print("\n=== 완료 ===")


if __name__ == "__main__":
    main()
