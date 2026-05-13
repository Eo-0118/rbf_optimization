"""RBF 정책 베이스라인 — RL 학습 전 sanity check 및 비교 기준.

3가지 정책:
1. RandomPolicy: r_t를 균등 분포로 샘플링
2. FixedRatePolicy: r_t = 고정값 (예: 0.10)
3. RevenueProportionalPolicy: 최근 매출 추세에 비례 (간단한 휴리스틱)

평가 지표:
- 평균 회수율 (cumulative_recovery / target)
- 평균 침해율 (burden 누적)
- 디폴트율 (회수율 < 1.0인 셀러 비율)
- 평균 만기 (회수 완료 시점)
- 평균 reward (env 정의)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from envs.rbf_env import RBFEnv

ROOT = Path("/Users/eoseungyun/Desktop/project/SW_Capstone/Project")
DATA = ROOT / "Data"

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False


# === 정책들 ===
class RandomPolicy:
    name = "Random"

    def __init__(self, env: RBFEnv, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.r_min = env.r_min
        self.r_max = env.r_max

    def predict(self, state):
        return np.array([self.rng.uniform(self.r_min, self.r_max)], dtype=np.float32)


class FixedRatePolicy:
    def __init__(self, env: RBFEnv, rate: float = 0.10):
        self.rate = float(np.clip(rate, env.r_min, env.r_max))
        self.name = f"Fixed-{self.rate:.2f}"

    def predict(self, state):
        return np.array([self.rate], dtype=np.float32)


class RevenueProportionalPolicy:
    """최근 3개월 매출 추세에 비례.
    매출 평균 대비 클수록 r_t 키움 (회수 가속), 작을수록 r_t 줄임 (셀러 보호).
    state index: [0]=t/T, [1]=recovery_progress, [2-4]=recent_revs (normalized by mean_rev)
    """
    name = "RevProportional"

    def __init__(self, env: RBFEnv, base_rate: float = 0.10, sensitivity: float = 0.05):
        self.r_min = env.r_min
        self.r_max = env.r_max
        self.base = base_rate
        self.sensitivity = sensitivity

    def predict(self, state):
        recent_norm = state[2:5]  # 정규화된 최근 3개월
        recent_avg = float(np.mean(recent_norm))
        # recent_avg > 1 → 평균 이상 매출 → r 키움. < 1 → r 줄임.
        adjustment = self.sensitivity * (recent_avg - 1.0)
        r_t = float(np.clip(self.base + adjustment, self.r_min, self.r_max))
        return np.array([r_t], dtype=np.float32)


class CVaRPolicy:
    """CVaR 정적 최적화로 셀러별 r* 사전 산출 → 모든 월 동일 r* 적용.

    Phase 3의 핵심 비교 정책. 셀러 위험 프로파일에 따라 차별화된 r 사용.
    """
    name = "CVaR"

    def __init__(self, env: RBFEnv, results_path: str | None = None):
        self.r_min = env.r_min
        self.r_max = env.r_max
        path = Path(results_path) if results_path else (DATA / "cvar_optimizer_results.csv")
        if not path.exists():
            raise FileNotFoundError(f"CVaR results not found: {path}. Run optim/cvar_optimizer.py first.")
        df = pd.read_csv(path)
        self.r_lookup = dict(zip(df["seller_id"], df["r_star"]))
        self._current_seller = None
        self._current_r = None
        self.env = env  # episode 시작 시 셀러 ID 추적용

    def predict(self, state):
        # 현재 셀러 변경 감지 (env가 reset하면 current_seller_id 변경됨)
        sid = self.env.current_seller_id
        if sid != self._current_seller:
            r_star = self.r_lookup.get(sid, np.nan)
            if np.isnan(r_star):
                r_star = 0.10  # fallback
            self._current_r = float(np.clip(r_star, self.r_min, self.r_max))
            self._current_seller = sid
        return np.array([self._current_r], dtype=np.float32)


# === 평가 ===
def run_episode(env: RBFEnv, policy, seller_id: str | None = None) -> dict:
    """단일 셀러 에피소드 실행 → 결과 dict 반환."""
    options = {"seller_id": seller_id} if seller_id else None
    state, info_init = env.reset(options=options)
    total_reward = 0.0
    burden_sum = 0.0
    burden_count = 0
    household_violation_count = 0
    household_penalty_sum = 0.0
    while True:
        action = policy.predict(state)
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if info["burden"] > 0:
            burden_sum += info["burden"]
            burden_count += 1
        if info.get("household_violated", False):
            household_violation_count += 1
            household_penalty_sum += abs(info.get("household_penalty", 0.0))
        if terminated:
            break
    log = env.get_episode_log()
    final_recovery = log[-1]["recovery_progress"]
    completed = final_recovery >= 1.0
    completion_t = next((i + 1 for i, e in enumerate(log) if e["completed"]), env.T)
    return dict(
        seller_id=info_init["seller_id"],
        type=info_init["type"],
        L=info_init["L"],
        target=info_init["target"],
        final_recovery=final_recovery,
        completed=completed,
        completion_t=completion_t if completed else env.T + 1,
        burden_mean=burden_sum / max(burden_count, 1),
        burden_months=burden_count,
        household_violation_count=household_violation_count,
        household_penalty_sum=household_penalty_sum,
        total_reward=total_reward,
        n_months=len(log),
    )


def evaluate_policy(env: RBFEnv, policy, n_episodes: int | None = None,
                     seller_subset: list[str] | None = None) -> pd.DataFrame:
    """정책 평가 (모든 셀러 또는 subset).
    n_episodes=None이면 모든 셀러 1번씩 (deterministic 평가).
    """
    if seller_subset is not None:
        sids = seller_subset
    else:
        sids = list(env.sellers.keys())
    if n_episodes is not None:
        sids = sids[:n_episodes]

    rows = []
    for sid in sids:
        rows.append(run_episode(env, policy, seller_id=sid))
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, policy_name: str) -> dict:
    return {
        "policy": policy_name,
        "n": int(len(df)),
        "mean_recovery": float(df["final_recovery"].mean()),
        "median_recovery": float(df["final_recovery"].median()),
        "completion_rate": float(df["completed"].mean() * 100),
        "default_rate": float((1 - df["completed"]).mean() * 100),
        "mean_completion_t": float(df[df["completed"]]["completion_t"].mean()) if df["completed"].any() else float("nan"),
        "mean_burden": float(df["burden_mean"].mean()),
        "mean_burden_months": float(df["burden_months"].mean()),
        "mean_household_violation_months": float(df["household_violation_count"].mean()),
        "household_violation_zero_rate": float((df["household_violation_count"] == 0).mean() * 100),  # 가계 침범 0건 셀러 비율
        "mean_reward": float(df["total_reward"].mean()),
    }


def main():
    print("[1/3] Env 초기화 + 정책 정의")
    env = RBFEnv(seed=42)
    print(f"  Sellers: {len(env.sellers)}")

    policies = {
        "Random": RandomPolicy(env, seed=42),
        "Fixed-0.05": FixedRatePolicy(env, rate=0.05),
        "Fixed-0.10": FixedRatePolicy(env, rate=0.10),
        "Fixed-0.15": FixedRatePolicy(env, rate=0.15),
        "Fixed-0.20": FixedRatePolicy(env, rate=0.20),
        "RevProportional-0.10": RevenueProportionalPolicy(env, base_rate=0.10),
    }
    # CVaR 정책은 사전 최적화 결과 필요 — 파일 있을 때만 추가
    cvar_path = DATA / "cvar_optimizer_results.csv"
    if cvar_path.exists():
        policies["CVaR"] = CVaRPolicy(env, results_path=cvar_path)
        print(f"  + CVaR 정책 로드 ({len(policies['CVaR'].r_lookup)} 셀러)")

    print(f"\n[2/3] 정책 평가 (각 정책 × 1302 셀러)")
    summaries = []
    by_type_records = []
    detailed_dfs = {}
    for name, pol in policies.items():
        df = evaluate_policy(env, pol)
        s = summarize(df, name)
        summaries.append(s)
        detailed_dfs[name] = df
        # 유형별
        for typ, g in df.groupby("type"):
            by_type_records.append(dict(
                policy=name, type=typ, n=len(g),
                mean_recovery=float(g["final_recovery"].mean()),
                completion_rate=float(g["completed"].mean() * 100),
                mean_burden=float(g["burden_mean"].mean()),
                mean_reward=float(g["total_reward"].mean()),
            ))
        print(f"  {name:25s}: completion={s['completion_rate']:5.1f}%  "
              f"recovery={s['mean_recovery']:.3f}  burden={s['mean_burden']:.4f}  "
              f"hh_viol={s['mean_household_violation_months']:5.1f}mo  "
              f"hh_safe={s['household_violation_zero_rate']:5.1f}%  "
              f"reward={s['mean_reward']:+.3f}")

    summary_df = pd.DataFrame(summaries)
    by_type_df = pd.DataFrame(by_type_records)

    print(f"\n[3/3] 결과 저장 + 시각화")
    # 결과 저장
    summary_df.to_csv(DATA / "baselines_summary.csv", index=False)
    by_type_df.to_csv(DATA / "baselines_by_type.csv", index=False)
    # 상세 (sample)
    detailed_dfs["Fixed-0.10"].to_csv(DATA / "baselines_fixed010_detail.csv", index=False)
    print(f"  [save] baselines_summary.csv, baselines_by_type.csv")

    # 시각화
    color_map = {"stable": "steelblue", "growth": "mediumseagreen",
                 "volatile": "crimson", "seasonal": "darkorange",
                 "decline": "gray", "other": "lightgray"}
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (1) 정책별 completion rate + mean recovery
    ax = axes[0, 0]
    pol_names = summary_df["policy"].tolist()
    x = np.arange(len(pol_names))
    width = 0.35
    ax.bar(x - width/2, summary_df["completion_rate"], width, label="completion %",
           color="steelblue", alpha=0.8)
    ax.set_ylabel("completion rate (%)")
    ax2 = ax.twinx()
    ax2.bar(x + width/2, summary_df["mean_recovery"] * 100, width, label="mean recovery × 100",
            color="darkorange", alpha=0.8)
    ax2.set_ylabel("mean recovery × 100", color="darkorange")
    ax.set_xticks(x)
    ax.set_xticklabels(pol_names, rotation=20, ha="right")
    ax.set_title("정책별 회수 성과")
    ax.legend(loc="upper left"); ax2.legend(loc="upper right")
    ax.grid(alpha=0.3)

    # (2) 정책별 burden
    ax = axes[0, 1]
    ax.bar(pol_names, summary_df["mean_burden"], color="crimson", alpha=0.7)
    ax.set_ylabel("mean burden")
    ax.set_title("정책별 평균 셀러 침해율")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(alpha=0.3)

    # (3) 정책별 mean reward
    ax = axes[1, 0]
    ax.bar(pol_names, summary_df["mean_reward"], color="mediumseagreen", alpha=0.7)
    ax.set_ylabel("mean reward")
    ax.set_title("정책별 평균 누적 보상 (env reward 기준)")
    ax.tick_params(axis="x", rotation=20)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(alpha=0.3)

    # (4) 유형별 completion (Fixed-0.10 기준)
    ax = axes[1, 1]
    sub = by_type_df[by_type_df["policy"] == "Fixed-0.10"]
    sub = sub.sort_values("type")
    colors = [color_map.get(t, "gray") for t in sub["type"]]
    ax.bar(sub["type"], sub["completion_rate"], color=colors, alpha=0.8)
    for i, v in enumerate(sub["completion_rate"].values):
        ax.text(i, v + 1, f"{v:.0f}%", ha="center", fontweight="bold")
    ax.set_ylabel("completion rate (%)")
    ax.set_title("유형별 완납률 (Fixed-0.10 기준)")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(alpha=0.3)

    plt.suptitle("RBF Baseline 정책 비교", fontsize=13, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(DATA / "baselines_diagnostics.png", dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  [save] baselines_diagnostics.png")

    # JSON 요약
    out = {"policies": summaries, "by_type_sample": by_type_df.to_dict(orient="records")}
    (DATA / "baselines_summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print("\n=== 베이스라인 요약 ===")
    print(summary_df.to_string(index=False))
    print("\n=== 완료 ===")


if __name__ == "__main__":
    main()
