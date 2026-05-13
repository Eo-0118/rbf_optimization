"""3-정책 통합 비교 평가 (Phase 3 v1)

비교 정책:
- Baseline: Fixed-0.15 (회수 우선 정적 정책)
- CVaR: 셀러별 r* (가계 보호 우선 정적 정책)
- CVaR+RL (PPO): 월별 동적 r_t (학습된 RL)

핵심 가설:
- RL이 정적 정책 trade-off를 동시 개선해야 가치 입증
  → 회수 ↑ + burden ↓ + 가계 침범 ↓

입력:
- Data/baselines_summary.csv (Fixed-0.15, CVaR 등)
- Data/baselines_by_type.csv
- Data/ppo_eval_results.csv (PPO 학습 후 평가 결과, 사용자가 train_ppo 실행 후 생성)

산출:
- Data/policy_comparison_table.csv (3-정책 통합 비교)
- Data/policy_comparison_by_type.csv (유형별)
- Data/policy_comparison.png (4-panel 시각화)
- Data/policy_comparison_report.md (보고서 형식)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("/Users/eoseungyun/Desktop/project/SW_Capstone/Project")
DATA = ROOT / "Data"

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

POLICIES_TO_COMPARE = ["Fixed-0.15", "CVaR", "PPO"]


def load_baseline_results() -> pd.DataFrame:
    """baselines_summary.csv에서 Fixed-0.15, CVaR 등 로드."""
    path = DATA / "baselines_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} 없음. envs/baselines.py 먼저 실행.")
    return pd.read_csv(path)


def load_ppo_results() -> dict | None:
    """ppo_eval_results.csv에서 PPO 결과 로드 (없으면 None)."""
    path = DATA / "ppo_eval_results.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return dict(
        policy="PPO",
        n=len(df),
        mean_recovery=float(df["final_recovery"].mean()),
        median_recovery=float(df["final_recovery"].median()),
        completion_rate=float(df["completed"].mean() * 100),
        default_rate=float((1 - df["completed"]).mean() * 100),
        mean_completion_t=float("nan"),
        mean_burden=float(df["burden_mean"].mean()),
        mean_burden_months=float(df["burden_months"].mean()),
        mean_household_violation_months=float(df["household_violation_count"].mean()),
        household_violation_zero_rate=float((df["household_violation_count"] == 0).mean() * 100),
        mean_reward=float(df["total_reward"].mean()),
    )


def build_comparison_table(baselines_df: pd.DataFrame, ppo_row: dict | None) -> pd.DataFrame:
    """비교 테이블 구성. 핵심 정책만 추출."""
    rows = []
    for pol in ["Fixed-0.15", "CVaR"]:
        sub = baselines_df[baselines_df["policy"] == pol]
        if len(sub) == 0:
            continue
        rows.append(sub.iloc[0].to_dict())
    if ppo_row is not None:
        rows.append(ppo_row)
    return pd.DataFrame(rows)


def visualize(comp_df: pd.DataFrame, output_path: Path):
    """4-panel 시각화."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    policies = comp_df["policy"].tolist()
    colors = {"Fixed-0.15": "steelblue", "CVaR": "darkorange", "PPO": "crimson"}
    bar_colors = [colors.get(p, "gray") for p in policies]
    x = np.arange(len(policies))

    # (1) 회수율 (completion rate + mean recovery)
    ax = axes[0, 0]
    width = 0.35
    ax.bar(x - width/2, comp_df["completion_rate"], width, label="completion %", color=bar_colors, alpha=0.8)
    ax.set_ylabel("completion rate (%)")
    for i, v in enumerate(comp_df["completion_rate"].values):
        ax.text(i - width/2, v + 1, f"{v:.0f}%", ha="center", fontweight="bold")
    ax2 = ax.twinx()
    ax2.bar(x + width/2, comp_df["mean_recovery"] * 100, width, label="mean recovery × 100",
            color=bar_colors, alpha=0.5)
    ax2.set_ylabel("mean recovery × 100", color="darkgray")
    for i, v in enumerate(comp_df["mean_recovery"].values):
        ax2.text(i + width/2, v * 100 + 2, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(policies)
    ax.set_title("(1) 회수 성과")
    ax.legend(loc="upper left", fontsize=8); ax2.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    # (2) 셀러 보호 — burden
    ax = axes[0, 1]
    ax.bar(x, comp_df["mean_burden"], color=bar_colors, alpha=0.8)
    for i, v in enumerate(comp_df["mean_burden"].values):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(policies)
    ax.set_ylabel("mean burden")
    ax.set_title("(2) 평균 셀러 침해율 (낮을수록 좋음)")
    ax.grid(alpha=0.3)

    # (3) 가계 보호 — household safe rate
    ax = axes[1, 0]
    bar_h = ax.bar(x, comp_df["household_violation_zero_rate"], color=bar_colors, alpha=0.8)
    for i, v in enumerate(comp_df["household_violation_zero_rate"].values):
        ax.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(policies)
    ax.set_ylabel("household violation 0건 셀러 비율 (%)")
    ax.set_title("(3) 가계 안전 셀러 비율 (높을수록 좋음)")
    ax.grid(alpha=0.3)

    # (4) 가계 침범 평균 개월 (낮을수록)
    ax = axes[1, 1]
    ax.bar(x, comp_df["mean_household_violation_months"], color=bar_colors, alpha=0.8)
    for i, v in enumerate(comp_df["mean_household_violation_months"].values):
        ax.text(i, v + 0.3, f"{v:.1f}mo", ha="center", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(policies)
    ax.set_ylabel("평균 가계 침범 개월 수")
    ax.set_title("(4) 평균 가계 침범 개월 (낮을수록 좋음, /24)")
    ax.grid(alpha=0.3)

    plt.suptitle("Phase 3 v1 — 3-정책 비교 (Baseline / CVaR / RL)",
                 fontsize=14, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close()


def write_report(comp_df: pd.DataFrame, output_path: Path):
    """마크다운 보고서 생성."""
    lines = [
        "# Phase 3 v1 — 3-정책 비교 결과\n",
        "## 통합 비교 표\n",
        "| 정책 | Recovery | Completion | Burden | HH safe | HH violation (mo/24) | Reward |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, row in comp_df.iterrows():
        lines.append(
            f"| **{row['policy']}** | {row['mean_recovery']:.3f} | "
            f"{row['completion_rate']:.1f}% | {row['mean_burden']:.4f} | "
            f"{row['household_violation_zero_rate']:.1f}% | "
            f"{row['mean_household_violation_months']:.1f} | "
            f"{row['mean_reward']:+.2f} |"
        )

    lines += [
        "\n## 해석\n",
        "- **Fixed-0.15** (정적, 회수 우선): 단순 정책. 회수율 높지만 가계 침범 많음.",
        "- **CVaR** (정적, 가계 보호 우선): 셀러별 r* 차별화. 가계 침범 적지만 회수율 손해.",
        "- **PPO** (동적 RL): 매월 r_t 조정. 두 trade-off 동시 개선 목표.\n",
        "## 본 연구의 기여 가설\n",
        "RL이 단순 평균값으로 회귀하지 않고 매출 변동에 반응하여:",
        "  - 매출 큰 달에 회수 가속 (recovery ≥ Fixed-0.15)",
        "  - 매출 작은 달에 r 감소 (가계 보호 ≥ CVaR)",
        "→ 두 정적 정책의 강점을 동시 달성\n",
    ]
    if "PPO" in comp_df["policy"].values:
        ppo_row = comp_df[comp_df["policy"] == "PPO"].iloc[0]
        fixed_row = comp_df[comp_df["policy"] == "Fixed-0.15"].iloc[0]
        cvar_row = comp_df[comp_df["policy"] == "CVaR"].iloc[0]
        lines += [
            "## RL 효과 정량 분석\n",
            f"- vs Fixed-0.15: completion {ppo_row['completion_rate'] - fixed_row['completion_rate']:+.1f}%p, "
            f"burden {ppo_row['mean_burden'] - fixed_row['mean_burden']:+.4f}",
            f"- vs CVaR: completion {ppo_row['completion_rate'] - cvar_row['completion_rate']:+.1f}%p, "
            f"burden {ppo_row['mean_burden'] - cvar_row['mean_burden']:+.4f}",
            f"- 가계 안전 개선: vs Fixed-0.15 {ppo_row['household_violation_zero_rate'] - fixed_row['household_violation_zero_rate']:+.1f}%p, "
            f"vs CVaR {ppo_row['household_violation_zero_rate'] - cvar_row['household_violation_zero_rate']:+.1f}%p\n",
        ]

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    print("[1/4] Baseline + CVaR 결과 로드")
    baselines_df = load_baseline_results()
    print(f"  baseline 정책: {baselines_df['policy'].tolist()}")

    print("\n[2/4] PPO 결과 로드")
    ppo_row = load_ppo_results()
    if ppo_row is None:
        print("  ⚠️ ppo_eval_results.csv 없음. PPO 학습 미완료 — 2-정책만 비교")
    else:
        print(f"  PPO 결과 로드 OK (n={ppo_row['n']})")

    print("\n[3/4] 비교 테이블 + 시각화")
    comp_df = build_comparison_table(baselines_df, ppo_row)
    comp_df.to_csv(DATA / "policy_comparison_table.csv", index=False)
    print(f"  [save] policy_comparison_table.csv")

    visualize(comp_df, DATA / "policy_comparison.png")
    print(f"  [save] policy_comparison.png")

    print("\n[4/4] 보고서 작성")
    write_report(comp_df, DATA / "policy_comparison_report.md")
    print(f"  [save] policy_comparison_report.md")

    print("\n=== 비교 결과 ===")
    pd.set_option('display.width', 200)
    pd.set_option('display.max_columns', 15)
    cols_show = ["policy", "mean_recovery", "completion_rate", "mean_burden",
                 "household_violation_zero_rate", "mean_household_violation_months", "mean_reward"]
    print(comp_df[cols_show].to_string(index=False))
    print("\n=== 완료 ===")


if __name__ == "__main__":
    main()
