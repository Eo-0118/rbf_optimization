"""A-1 + A-2: PPO를 전체 1,302명 셀러에 재평가 + 침해 액도 지표 포함

배경:
- 기존 PPO 평가는 260명 평가 셀러만 (학습 1,041 / 평가 261)
- CVaR/Fixed-0.15는 1,302명 전체 평가
- 비교가 불공평. PPO를 전체 1,302명에 재평가 필요

또한:
- 기존 가계 침해 지표 = "침해 월수" (binary 합산)
- 한계: 한 달 10% 침해와 90% 침해를 동일하게 카운트
- A-2: 침해 액도 (violation_amount, violation_ratio) 추가 수집 → 정도 측정

산출:
- Data/ppo_eval_full_results.csv (1,302명 PPO + 신규 지표)
- Data/policy_full_comparison.csv (Fixed-0.15, CVaR, PPO 전체 비교)
- Data/policy_full_comparison.png (시각화)
- Data/ppo_train_vs_test_gap.json (overfitting 진단)
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

ROOT = PROJECT_ROOT
DATA = ROOT / "Data"
MODELS = ROOT / "models" / "ppo"

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

SEED = 42
PPO_MODEL_PATH = MODELS / "ppo_final.zip"


class PPOPolicy:
    """학습된 PPO 모델을 베이스라인 정책 인터페이스로 wrap."""
    name = "PPO"

    def __init__(self, model_path: Path):
        from stable_baselines3 import PPO
        self.model = PPO.load(str(model_path))

    def predict(self, state):
        action, _ = self.model.predict(state, deterministic=True)
        return action


def split_sellers_same_as_train(env: RBFEnv, test_ratio: float = 0.2, seed: int = SEED):
    """train_ppo.py의 split_sellers와 동일한 분할 재현."""
    df = pd.read_parquet(DATA / "cohort_kr_v2.parquet")
    sids = df["seller_id"].unique().tolist()
    rng = np.random.default_rng(seed)
    rng.shuffle(sids)
    n_test = int(len(sids) * test_ratio)
    test_ids = set(sids[:n_test])
    train_ids = set(sids[n_test:])
    return train_ids, test_ids


def run_episode_full(env: RBFEnv, policy, seller_id: str) -> dict:
    """A-2: 침해 액도 지표 포함한 episode 실행."""
    state, info_init = env.reset(options={"seller_id": seller_id})
    total_reward = 0.0
    burden_sum = 0.0
    burden_count = 0
    hh_violation_count = 0
    hh_violation_amount_sum = 0.0
    hh_violation_ratio_max = 0.0      # 단일 월 최대 침해 비율
    while True:
        action = policy.predict(state)
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if info["burden"] > 0:
            burden_sum += info["burden"]
            burden_count += 1
        if info.get("household_violated", False):
            hh_violation_count += 1
        hh_violation_amount_sum += info.get("household_violation_amount", 0.0)
        hh_violation_ratio_max = max(hh_violation_ratio_max,
                                      info.get("household_violation_ratio", 0.0))
        if terminated:
            break
    log = env.get_episode_log()
    final_recovery = log[-1]["recovery_progress"]
    return dict(
        seller_id=info_init["seller_id"],
        type=info_init["type"],
        final_recovery=float(final_recovery),
        completed=bool(final_recovery >= 1.0),
        burden_mean=float(burden_sum / max(burden_count, 1)),
        burden_months=int(burden_count),
        household_violation_count=int(hh_violation_count),
        # 신규 (A-2):
        household_violation_amount_total=float(hh_violation_amount_sum),  # 24개월 누적 침해액 (만원)
        household_violation_ratio_max=float(hh_violation_ratio_max),       # 단일 월 최대 (가계비 대비)
        total_reward=float(total_reward),
    )


def evaluate_all(env: RBFEnv, policy, seller_ids: list[str]) -> pd.DataFrame:
    rows = []
    for sid in seller_ids:
        rows.append(run_episode_full(env, policy, sid))
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, policy_name: str, subset_label: str = "all") -> dict:
    """기존 + 신규(A-2) 지표 모두 요약."""
    return {
        "policy": policy_name,
        "subset": subset_label,
        "n": int(len(df)),
        "mean_recovery": float(df["final_recovery"].mean()),
        "completion_rate": float(df["completed"].mean() * 100),
        "mean_burden": float(df["burden_mean"].mean()),
        "mean_household_violation_months": float(df["household_violation_count"].mean()),
        "household_violation_zero_rate": float((df["household_violation_count"] == 0).mean() * 100),
        # A-2 신규:
        "mean_violation_amount_total": float(df["household_violation_amount_total"].mean()),
        "median_violation_amount_total": float(df["household_violation_amount_total"].median()),
        "mean_violation_ratio_max": float(df["household_violation_ratio_max"].mean()),
        "p95_violation_ratio_max": float(df["household_violation_ratio_max"].quantile(0.95)),
        "mean_reward": float(df["total_reward"].mean()),
    }


def main():
    print("[1/5] Env 초기화 + 학습/평가 셀러 분할 확인")
    env = RBFEnv(seed=SEED)
    train_ids, test_ids = split_sellers_same_as_train(env)
    all_ids = list(env.sellers.keys())
    print(f"  전체: {len(all_ids)} / 학습: {len(train_ids)} / 평가: {len(test_ids)}")
    assert len(train_ids) + len(test_ids) == len(all_ids), "분할 불일치"

    print(f"\n[2/5] 정책 정의")
    if not PPO_MODEL_PATH.exists():
        raise FileNotFoundError(f"PPO 모델 없음: {PPO_MODEL_PATH}")
    policies = {
        "Fixed-0.15": FixedRatePolicy(env, rate=0.15),
        "CVaR": CVaRPolicy(env),
        "PPO": PPOPolicy(PPO_MODEL_PATH),
    }
    print(f"  정책: {list(policies.keys())}")

    print(f"\n[3/5] 전체 1,302명 평가 (각 정책)")
    all_results = {}
    summaries = []
    for name, pol in policies.items():
        print(f"  {name} 평가 중...")
        df_full = evaluate_all(env, pol, all_ids)
        all_results[name] = df_full
        summaries.append(summarize(df_full, name, "all_1302"))

    print(f"\n[4/5] 학습 vs 평가 셀러 분리 (overfitting 진단)")
    train_test_summaries = []
    for name, df_full in all_results.items():
        df_train = df_full[df_full["seller_id"].isin(train_ids)]
        df_test = df_full[df_full["seller_id"].isin(test_ids)]
        s_train = summarize(df_train, name, "train_1041")
        s_test = summarize(df_test, name, "test_261")
        train_test_summaries.extend([s_train, s_test])
        print(f"  {name}: train completion {s_train['completion_rate']:.1f}% / test {s_test['completion_rate']:.1f}% "
              f"(격차 {s_train['completion_rate'] - s_test['completion_rate']:+.1f}%p)")

    print(f"\n[5/5] 저장 + 시각화")
    # 전체 비교
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(DATA / "policy_full_comparison.csv", index=False)
    print(f"  [save] policy_full_comparison.csv")

    # train/test 분리 비교
    tt_df = pd.DataFrame(train_test_summaries)
    tt_df.to_csv(DATA / "ppo_train_vs_test_gap.csv", index=False)

    # PPO 전체 셀러 raw 결과 (신규 지표 포함)
    all_results["PPO"].to_csv(DATA / "ppo_eval_full_results.csv", index=False)
    print(f"  [save] ppo_eval_full_results.csv")

    # JSON overfit 진단
    ppo_train_s = next(s for s in train_test_summaries if s["policy"] == "PPO" and s["subset"] == "train_1041")
    ppo_test_s = next(s for s in train_test_summaries if s["policy"] == "PPO" and s["subset"] == "test_261")
    overfit_diag = {
        "ppo_train_completion": ppo_train_s["completion_rate"],
        "ppo_test_completion": ppo_test_s["completion_rate"],
        "completion_gap_pp": ppo_train_s["completion_rate"] - ppo_test_s["completion_rate"],
        "ppo_train_violation_amount": ppo_train_s["mean_violation_amount_total"],
        "ppo_test_violation_amount": ppo_test_s["mean_violation_amount_total"],
        "overfit_assessment": (
            "robust" if abs(ppo_train_s["completion_rate"] - ppo_test_s["completion_rate"]) < 5
            else "mild_overfit" if abs(ppo_train_s["completion_rate"] - ppo_test_s["completion_rate"]) < 10
            else "overfit"
        ),
    }
    (DATA / "ppo_train_vs_test_gap.json").write_text(
        json.dumps(overfit_diag, indent=2, ensure_ascii=False))
    print(f"  [save] ppo_train_vs_test_gap.json")

    # 시각화 — 4-panel (기존 4 + A-2 침해 액도 4)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    pol_names = ["Fixed-0.15", "CVaR", "PPO"]
    colors = {"Fixed-0.15": "steelblue", "CVaR": "darkorange", "PPO": "crimson"}
    x = np.arange(len(pol_names))
    s_map = {s["policy"]: s for s in summaries}

    # 1. Completion rate
    ax = axes[0, 0]
    vals = [s_map[p]["completion_rate"] for p in pol_names]
    ax.bar(x, vals, color=[colors[p] for p in pol_names])
    for i, v in enumerate(vals):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(pol_names)
    ax.set_title("Completion rate (1,302명)"); ax.grid(alpha=0.3, axis="y")

    # 2. Mean burden
    ax = axes[0, 1]
    vals = [s_map[p]["mean_burden"] for p in pol_names]
    ax.bar(x, vals, color=[colors[p] for p in pol_names])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(pol_names)
    ax.set_title("Mean burden (낮을수록 좋음)"); ax.grid(alpha=0.3, axis="y")

    # 3. HH 침해 월수 (기존 지표)
    ax = axes[0, 2]
    vals = [s_map[p]["mean_household_violation_months"] for p in pol_names]
    ax.bar(x, vals, color=[colors[p] for p in pol_names])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.3, f"{v:.1f}mo", ha="center", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(pol_names)
    ax.set_title("HH 침해 월수 (기존, /24)"); ax.grid(alpha=0.3, axis="y")

    # 4. HH 침해 누적액 (A-2 신규)
    ax = axes[1, 0]
    vals = [s_map[p]["mean_violation_amount_total"] for p in pol_names]
    ax.bar(x, vals, color=[colors[p] for p in pol_names])
    for i, v in enumerate(vals):
        ax.text(i, v + max(vals) * 0.02, f"{v:.0f}", ha="center", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(pol_names)
    ax.set_title("[A-2 신규] 24개월 누적 침해액 (만원)"); ax.grid(alpha=0.3, axis="y")

    # 5. HH 침해 비율 max (A-2)
    ax = axes[1, 1]
    vals = [s_map[p]["mean_violation_ratio_max"] for p in pol_names]
    ax.bar(x, vals, color=[colors[p] for p in pol_names])
    for i, v in enumerate(vals):
        ax.text(i, v + max(vals) * 0.02, f"{v:.2f}", ha="center", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(pol_names)
    ax.set_title("[A-2 신규] 단일 월 최대 침해비\n(가계비 1.0 대비)")
    ax.grid(alpha=0.3, axis="y")

    # 6. PPO train vs test gap
    ax = axes[1, 2]
    sub_names = ["train_1041", "test_261"]
    train_vals = [s["completion_rate"] for s in train_test_summaries if s["subset"] == "train_1041"]
    test_vals = [s["completion_rate"] for s in train_test_summaries if s["subset"] == "test_261"]
    width = 0.35
    ax.bar(x - width/2, train_vals, width, label="train 1041", color="lightsteelblue")
    ax.bar(x + width/2, test_vals, width, label="test 261", color="cornflowerblue")
    for i, (t, e) in enumerate(zip(train_vals, test_vals)):
        ax.text(i - width/2, t + 1, f"{t:.0f}", ha="center", fontsize=8)
        ax.text(i + width/2, e + 1, f"{e:.0f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(pol_names)
    ax.set_title("train vs test completion (PPO overfit 진단)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

    plt.suptitle(f"A-1 + A-2: 전체 1,302명 평가 + 침해 액도 지표 추가\n"
                 f"PPO overfit: {overfit_diag['overfit_assessment']} "
                 f"(train-test 격차 {overfit_diag['completion_gap_pp']:+.1f}%p)",
                 fontsize=13, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(DATA / "policy_full_comparison.png", dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  [save] policy_full_comparison.png")

    # 콘솔 출력
    print(f"\n=== 전체 1,302명 평가 결과 ===")
    show_cols = ["policy", "completion_rate", "mean_burden",
                 "mean_household_violation_months", "mean_violation_amount_total",
                 "mean_violation_ratio_max"]
    print(summary_df[show_cols].to_string(index=False))

    print(f"\n=== PPO train vs test 격차 ===")
    for s in train_test_summaries:
        if s["policy"] == "PPO":
            print(f"  PPO {s['subset']:12s}: completion {s['completion_rate']:5.1f}%, "
                  f"violation_amt {s['mean_violation_amount_total']:6.1f}만, "
                  f"violation_ratio_max {s['mean_violation_ratio_max']:.2f}")
    print(f"  → {overfit_diag['overfit_assessment']} (격차 {overfit_diag['completion_gap_pp']:+.1f}%p)")

    print(f"\n=== 침해 액도 핵심 비교 (A-2 신규) ===")
    for p in pol_names:
        s = s_map[p]
        print(f"  {p:12s}: 월수 {s['mean_household_violation_months']:5.1f}, "
              f"누적액 {s['mean_violation_amount_total']:6.1f}만, "
              f"최대비 {s['mean_violation_ratio_max']:.2f}")

    print("\n=== 완료 ===")


if __name__ == "__main__":
    main()
