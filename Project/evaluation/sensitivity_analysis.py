"""Phase 4 민감도 분석 — L_personal, m_i 변동에 대한 정책 robustness

목적:
- L_personal_min 추정 불확실성, m_i 가정 부정확성에도 정책이 일관되게 작동하는가?
- 정책 자체는 학습 시점(L=172.0612, m_i=0.25) 그대로 사용
- 평가 환경의 가정만 변경 → "환경 변화에 대한 일반화 능력" 측정

방법:
- L_personal_min ∈ [86, 130, 150, 172, 200, 230, 260] (만원, 기준값 ±50%)
- m_i ∈ [0.15, 0.20, 0.25, 0.30, 0.35]
- 평가 셀러: PPO 평가 split 260명 (재현 가능)

산출:
- Data/sensitivity_results.csv
- Data/sensitivity_heatmap.png (L × m_i 히트맵, 정책별)
- Data/sensitivity_tornado.png (변수별 영향력)
- Data/sensitivity_report.md
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.rbf_env import RBFEnv
from envs.baselines import FixedRatePolicy, CVaRPolicy

warnings.filterwarnings("ignore")

DATA = PROJECT_ROOT / "Data"
MODELS = PROJECT_ROOT / "models" / "ppo"

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

SEED = 42
L_PERSONAL_GRID = [86.03, 129.05, 150.0, 172.0612, 200.0, 230.0, 258.09]   # 기준값 ±50%
M_I_GRID = [0.15, 0.20, 0.25, 0.30, 0.35]


def get_eval_sellers() -> list[str]:
    """train_ppo.py와 동일 분할: 80/20, seed=42 → 마지막 20%."""
    df = pd.read_parquet(DATA / "cohort_kr_v2.parquet")
    sids = df["seller_id"].unique().tolist()
    rng = np.random.default_rng(SEED)
    rng.shuffle(sids)
    n_test = int(len(sids) * 0.2)
    return sids[:n_test]


def run_episode(env: RBFEnv, policy, sid: str) -> dict:
    state, info_init = env.reset(options={"seller_id": sid})
    burden_sum = 0.0
    burden_count = 0
    hh_violations = 0
    while True:
        action = policy.predict(state)
        # PPO model.predict returns (action, _states), normalize
        if isinstance(action, tuple):
            action = action[0]
        if hasattr(action, 'shape') and action.shape == ():
            action = np.array([float(action)], dtype=np.float32)
        elif isinstance(action, np.ndarray) and action.ndim == 0:
            action = action.reshape(1)
        state, reward, terminated, truncated, info = env.step(np.asarray(action, dtype=np.float32))
        if info["burden"] > 0:
            burden_sum += info["burden"]
            burden_count += 1
        if info.get("household_violated", False):
            hh_violations += 1
        if terminated:
            break
    log = env.get_episode_log()
    final_recovery = log[-1]["recovery_progress"]
    return dict(
        seller_id=sid,
        type=info_init["type"],
        final_recovery=final_recovery,
        completed=bool(final_recovery >= 1.0),
        burden_mean=burden_sum / max(burden_count, 1),
        burden_months=burden_count,
        household_violation_count=hh_violations,
    )


class PPOPolicyAdapter:
    """PPO model을 baselines policy 인터페이스에 맞춤."""
    name = "PPO"

    def __init__(self, model):
        self.model = model

    def predict(self, state):
        action, _ = self.model.predict(state, deterministic=True)
        return action.flatten().astype(np.float32)


def evaluate_at_grid(L_personal: float, m_i: float, eval_sids: list[str],
                      ppo_model, cvar_results_path: Path) -> dict:
    """주어진 (L, m_i)에서 3-정책 평가."""
    env = RBFEnv(
        seller_ids=eval_sids,
        m_i=m_i,
        L_personal_min=L_personal,
        seed=SEED + 100,
    )
    policies = {
        "Fixed-0.15": FixedRatePolicy(env, rate=0.15),
        "CVaR": CVaRPolicy(env, results_path=cvar_results_path),
        "PPO": PPOPolicyAdapter(ppo_model),
    }
    results = {}
    for name, pol in policies.items():
        rows = [run_episode(env, pol, sid) for sid in eval_sids]
        df = pd.DataFrame(rows)
        results[name] = dict(
            completion_rate=float(df["completed"].mean() * 100),
            mean_recovery=float(df["final_recovery"].mean()),
            mean_burden=float(df["burden_mean"].mean()),
            mean_household_violation_months=float(df["household_violation_count"].mean()),
            household_violation_zero_rate=float((df["household_violation_count"] == 0).mean() * 100),
        )
    return results


def main():
    print("[1/4] 환경 + 정책 + 평가 셀러 준비")
    eval_sids = get_eval_sellers()
    print(f"  평가 셀러: {len(eval_sids)} (PPO 평가 split)")

    # PPO 모델 로드
    from stable_baselines3 import PPO
    ppo_path = MODELS / "ppo_final.zip"
    if not ppo_path.exists():
        raise FileNotFoundError(f"{ppo_path} 없음. agents/train_ppo.py 먼저 실행.")
    ppo_model = PPO.load(str(ppo_path))
    print(f"  PPO 모델 로드 OK: {ppo_path.name}")

    cvar_path = DATA / "cvar_optimizer_results.csv"
    print(f"  CVaR 결과: {cvar_path.name}")

    print(f"\n[2/4] 그리드 평가 ({len(L_PERSONAL_GRID)} × {len(M_I_GRID)} = {len(L_PERSONAL_GRID) * len(M_I_GRID)} 조합 × 3 정책)")
    rows = []
    total = len(L_PERSONAL_GRID) * len(M_I_GRID)
    for i, L in enumerate(L_PERSONAL_GRID):
        for j, m_i in enumerate(M_I_GRID):
            print(f"  [{i*len(M_I_GRID) + j + 1}/{total}] L_personal={L:.1f}, m_i={m_i}")
            results = evaluate_at_grid(L, m_i, eval_sids, ppo_model, cvar_path)
            for policy_name, metrics in results.items():
                rows.append(dict(
                    L_personal=L,
                    m_i=m_i,
                    policy=policy_name,
                    **metrics,
                ))

    res_df = pd.DataFrame(rows)
    res_df.to_csv(DATA / "sensitivity_results.csv", index=False)
    print(f"  [save] sensitivity_results.csv ({len(res_df)} 행)")

    print("\n[3/4] 시각화 생성")
    visualize_heatmap(res_df, DATA / "sensitivity_heatmap.png")
    visualize_tornado(res_df, DATA / "sensitivity_tornado.png")
    print(f"  [save] sensitivity_heatmap.png, sensitivity_tornado.png")

    print("\n[4/4] 보고서 작성")
    write_report(res_df, DATA / "sensitivity_report.md")
    print(f"  [save] sensitivity_report.md")

    print("\n=== 민감도 분석 결과 ===")
    pivot = res_df.pivot_table(
        index=["policy"],
        values=["completion_rate", "mean_burden", "household_violation_zero_rate", "mean_household_violation_months"],
        aggfunc=["mean", "std"],
    ).round(2)
    print(pivot.to_string())
    print("\n=== 완료 ===")


def visualize_heatmap(df: pd.DataFrame, output_path: Path):
    """4-metric × 3-policy heatmap (L × m_i)."""
    metrics = [
        ("completion_rate", "Completion %", "Greens"),
        ("mean_burden", "Mean Burden (낮을수록 좋음)", "Reds"),
        ("household_violation_zero_rate", "HH 안전 셀러 % (높을수록 좋음)", "Blues"),
        ("mean_household_violation_months", "평균 HH 침범 개월 (낮을수록 좋음)", "Oranges"),
    ]
    policies = ["Fixed-0.15", "CVaR", "PPO"]

    fig, axes = plt.subplots(len(metrics), len(policies), figsize=(15, 16))
    for i, (metric, title, cmap) in enumerate(metrics):
        for j, pol in enumerate(policies):
            ax = axes[i, j]
            sub = df[df["policy"] == pol]
            pivot = sub.pivot(index="L_personal", columns="m_i", values=metric)
            im = ax.imshow(pivot.values, aspect="auto", cmap=cmap, origin="lower")
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels([f"{x:.2f}" for x in pivot.columns])
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels([f"{y:.0f}" for y in pivot.index])
            ax.set_xlabel("m_i")
            ax.set_ylabel("L_personal (만원)")
            ax.set_title(f"{pol}: {title.split('(')[0].strip()}")
            for ii in range(len(pivot.index)):
                for jj in range(len(pivot.columns)):
                    val = pivot.values[ii, jj]
                    ax.text(jj, ii, f"{val:.1f}", ha="center", va="center",
                            fontsize=7, color="black")
            plt.colorbar(im, ax=ax, fraction=0.046)
    plt.suptitle("민감도 분석 — L_personal × m_i (정책별)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()


def visualize_tornado(df: pd.DataFrame, output_path: Path):
    """각 정책에서 변수(L_personal, m_i) 변동에 따른 metric 변동 (range)."""
    metrics = [
        ("completion_rate", "Completion %"),
        ("mean_burden", "Mean Burden"),
        ("household_violation_zero_rate", "HH 안전 셀러 %"),
        ("mean_household_violation_months", "평균 HH 침범 개월"),
    ]
    policies = ["Fixed-0.15", "CVaR", "PPO"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (metric, title) in zip(axes.flat, metrics):
        x = np.arange(len(policies))
        width = 0.35
        # L_personal range (m_i=0.25 fixed)
        l_ranges = []
        m_ranges = []
        for pol in policies:
            sub_l = df[(df["policy"] == pol) & (df["m_i"] == 0.25)]
            l_ranges.append(sub_l[metric].max() - sub_l[metric].min())
            sub_m = df[(df["policy"] == pol) & (np.isclose(df["L_personal"], 172.0612))]
            m_ranges.append(sub_m[metric].max() - sub_m[metric].min())
        ax.bar(x - width/2, l_ranges, width, label="L_personal 변동 영향", color="steelblue", alpha=0.8)
        ax.bar(x + width/2, m_ranges, width, label="m_i 변동 영향", color="darkorange", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(policies)
        ax.set_title(f"{title} 변동 폭")
        ax.set_ylabel("range (max - min)")
        ax.legend()
        ax.grid(alpha=0.3)
    plt.suptitle("Tornado Plot — 변수별 정책 metric 변동 폭", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close()


def write_report(df: pd.DataFrame, output_path: Path):
    lines = [
        "# Phase 4 민감도 분석 결과\n",
        f"**평가 그리드**: L_personal {len(L_PERSONAL_GRID)}개 × m_i {len(M_I_GRID)}개 = {len(L_PERSONAL_GRID)*len(M_I_GRID)} 조합 × 3 정책\n",
        "## 정책별 robustness 요약 (모든 조합 평균 ± 표준편차)\n",
        "| 정책 | Completion % | Mean Burden | HH 안전 % | HH 침범 개월 |",
        "|---|---|---|---|---|",
    ]
    for pol in ["Fixed-0.15", "CVaR", "PPO"]:
        sub = df[df["policy"] == pol]
        lines.append(
            f"| **{pol}** | "
            f"{sub['completion_rate'].mean():.1f} ± {sub['completion_rate'].std():.1f} | "
            f"{sub['mean_burden'].mean():.3f} ± {sub['mean_burden'].std():.3f} | "
            f"{sub['household_violation_zero_rate'].mean():.1f} ± {sub['household_violation_zero_rate'].std():.1f} | "
            f"{sub['mean_household_violation_months'].mean():.1f} ± {sub['mean_household_violation_months'].std():.1f} |"
        )

    lines += [
        "\n## 해석\n",
        "- **표준편차가 작을수록 robust** (가정 변동에 영향 적음)",
        "- 모든 가정 조합에서 정책 우열이 일관되면 본 연구 결과 일반화 가능\n",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
