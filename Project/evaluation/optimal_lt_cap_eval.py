"""Day 11+: 셀러별 (L*, T*, cap*) 동적 최적화 + 침해 0% 엄격 거절 + 5-정책 평가

본 연구의 운영 알고리즘 완성:
  Step 1: 셀러 정보 (R, type) 받음
  Step 2: 침해 0% 보장 조건에서 (L*, T*, cap*) 산출
  Step 3: 거절/허용 판정 (수학적 부적합 + 비즈니스 불가)
  Step 4: 적합 셀러에 5-정책 평가 (각 셀러 (L*, cap*) 적용)
  Step 5: 결과 정량화 — 거절율 + 적합 셀러 안전성

학술 위치:
  본 연구의 차별점 2가지:
    1. r 동적 조정 (PPO, 기존 RBF에 없음)
    2. 가계비 보호 강제 (2-tier burden)
  → "가계 안전 강제 + 동적 r RBF" 세계 최초 학술 보고

산출:
  Data/optimal_lt_cap_classification.csv (1,302명 × (L*, T*, cap*, 적합))
  Data/optimal_lt_cap_summary.json
  Data/optimal_lt_cap_policy_compare.csv (적합 셀러 5-정책 비교)
  Data/optimal_lt_cap_analysis.png
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
from envs.baselines import FixedRatePolicy

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

ROOT = PROJECT_ROOT
DATA = ROOT / "Data"
MODELS = ROOT / "models"
SEED = 42

# 환경 파라미터 (본 연구)
DEFAULT_M_I = 0.10
DEFAULT_L_PERSONAL = 128.21
DEFAULT_CAP_BASE = 1.10        # 기본 이자 (10%)
DEFAULT_T_MAX = 36              # 실제 사업자 대출 표준 (12-60개월)
L_MIN_BUSINESS = 100.0          # 비즈니스 가치 최소 L (100만)
T_SIM = 24                      # 실제 시뮬레이션 episode 길이 (합성 데이터 제약)


class PPOPolicy:
    def __init__(self, model_path: Path, name: str):
        from stable_baselines3 import PPO
        self.model = PPO.load(str(model_path))
        self.name = name

    def predict(self, state):
        action, _ = self.model.predict(state, deterministic=True)
        return action


def compute_optimal_lt_cap(R: float, seller_type: str, cv: float = None,
                            m_i: float = DEFAULT_M_I,
                            L_personal: float = DEFAULT_L_PERSONAL,
                            T_max: int = DEFAULT_T_MAX,
                            cap_base: float = DEFAULT_CAP_BASE) -> dict:
    """셀러별 (L*, T*, cap*) 산출.

    수식:
      L_max = T_max × (R × m_i - L_personal) / cap_base   (침해 0% 보장)
      T_required = ceil(L × cap / (R × m_i - L_personal))
      cap* = cap_base + risk_premium(seller_type, cv)
    """
    # 위험 프리미엄 (셀러 유형 기반)
    risk_premium = {
        "stable": -0.05,
        "growth": 0.0,
        "other": 0.05,
        "seasonal": 0.10,
        "decline": 0.15,
        "volatile": 0.20,
    }.get(seller_type, 0.05)

    # cv 기반 추가 프리미엄
    if cv is not None and cv > 1.0:
        risk_premium += 0.05

    cap_star = cap_base + risk_premium

    # 1. 침해 0% 절대 조건 (R × m_i > L_personal)
    monthly_safe_revenue = R * m_i - L_personal
    if monthly_safe_revenue <= 0:
        return dict(
            R=R, type=seller_type,
            monthly_safe_revenue=monthly_safe_revenue,
            cap_star=cap_star, L_star=0.0, T_star=0,
            eligible=False, reject_reason="가계비 충당 불가 (R × m_i ≤ L_personal)",
        )

    # 2. 최대 적정 L* (침해 없이 T_max 안에 회수 가능)
    L_max = T_max * monthly_safe_revenue / cap_star

    # 3. 비즈니스 가치 검증
    if L_max < L_MIN_BUSINESS:
        return dict(
            R=R, type=seller_type,
            monthly_safe_revenue=monthly_safe_revenue,
            cap_star=cap_star, L_star=L_max, T_star=T_max,
            eligible=False, reject_reason=f"L_max < {L_MIN_BUSINESS} (비즈니스 가치 부족)",
        )

    # 4. 자연스러운 T* (실제 회수 시점)
    # T* = ceil(L_max × cap_star / monthly_safe_revenue) = T_max (정의상)
    # 실제로는 T_max로 통일 (시뮬 가능 T_SIM=24까지)
    L_star = L_max
    T_star = min(T_max, int(np.ceil(L_star * cap_star / monthly_safe_revenue)))

    # T* > 24면 시뮬 불가 (warning), 사용자에게 정보로 제공
    sim_feasible = T_star <= T_SIM

    return dict(
        R=R, type=seller_type,
        monthly_safe_revenue=monthly_safe_revenue,
        cap_star=cap_star, L_star=L_star, T_star=T_star,
        eligible=True, reject_reason=None,
        sim_feasible=sim_feasible,
    )


def run_episode(env, policy, sid, L_override=None, cap_override=None):
    """env에 L/cap override 적용해서 episode 실행."""
    options = {"seller_id": sid}
    if L_override is not None:
        options["L_override"] = L_override
    if cap_override is not None:
        options["cap_override"] = cap_override
    obs, info_init = env.reset(options=options)
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
        L_applied=float(info_init["L"]), cap_applied=float(info_init["cap"]),
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
    print("[1/5] Cohort 로드 + 셀러별 (L*, T*, cap*) 산출")
    env = RBFEnv(seed=SEED)
    all_sids = list(env.sellers.keys())
    print(f"  전체 {len(all_sids)} 셀러")

    sellers_meta = []
    for sid in all_sids:
        info = env.sellers[sid]
        R = info["mean_rev"]
        seller_type = info["type"]
        # cv 추정 (revenues 분산)
        revs = info["revenues"]
        cv = float(np.std(revs) / max(np.mean(revs), 1e-6)) if np.mean(revs) > 0 else 0.0

        result = compute_optimal_lt_cap(R, seller_type, cv=cv)
        result["seller_id"] = sid
        result["cv"] = cv
        sellers_meta.append(result)

    meta_df = pd.DataFrame(sellers_meta)
    meta_df.to_csv(DATA / "optimal_lt_cap_classification.csv", index=False)
    print(f"  [save] optimal_lt_cap_classification.csv")

    print(f"\n[2/5] 거절/허용 분포 분석")
    eligible = meta_df[meta_df["eligible"]]
    rejected = meta_df[~meta_df["eligible"]]
    n_total = len(meta_df)
    n_eligible = len(eligible)
    n_rejected = len(rejected)
    rejection_rate = n_rejected / n_total * 100

    print(f"\n  전체 {n_total}: 적합 {n_eligible} ({n_eligible/n_total*100:.1f}%) "
          f"/ 거절 {n_rejected} ({rejection_rate:.1f}%)")

    # 거절 이유별
    if len(rejected) > 0:
        print(f"\n  거절 이유:")
        reject_dist = rejected["reject_reason"].value_counts()
        for reason, n in reject_dist.items():
            print(f"    {reason:50s}: {n}명 ({n/n_total*100:.1f}%)")

    # 적합 셀러 type 분포
    if n_eligible > 0:
        print(f"\n  적합 셀러 유형 분포:")
        print(eligible["type"].value_counts().to_string())
        print(f"\n  적합 셀러 (L*, T*, cap*) 통계:")
        print(eligible[["L_star", "T_star", "cap_star", "R", "monthly_safe_revenue"]].describe().round(2))

    if n_eligible == 0:
        print("\n⚠ 적합 셀러 없음. 가정 검토 필요.")
        return

    print(f"\n[3/5] 적합 {n_eligible}명에 5-정책 평가 (각 셀러 (L*, cap*) 적용)")
    policies = {
        "Fixed-0.15": FixedRatePolicy(env, rate=0.15),
        "PPO_base": PPOPolicy(MODELS / "ppo" / "ppo_final.zip", "PPO_base"),
        "PPO_eta_strong": PPOPolicy(MODELS / "ppo_eta_strong" / "ppo_final.zip", "PPO_eta_strong"),
        "PPO_x1_only": PPOPolicy(MODELS / "ppo_x1_only" / "ppo_final.zip", "PPO_x1_only"),
    }
    # CVaR은 사전 r* 산출 시 L=3R 가정 → (L*, cap*) override와 불일치. 제외 또는 별도 처리
    try:
        from envs.baselines import CVaRPolicy
        policies["CVaR"] = CVaRPolicy(env)
    except Exception:
        pass

    summaries = []
    raw_results = {}
    for name, pol in policies.items():
        print(f"  {name} 평가 중...")
        rows = []
        for _, row in eligible.iterrows():
            r = run_episode(env, pol, row["seller_id"],
                            L_override=row["L_star"], cap_override=row["cap_star"])
            r["R"] = row["R"]
            r["cv"] = row["cv"]
            rows.append(r)
        df = pd.DataFrame(rows)
        raw_results[name] = df
        summaries.append(dict(
            policy=name, n=len(df),
            completion_rate=float(df["completed"].mean() * 100),
            mean_recovery=float(df["final_recovery"].mean()),
            mean_burden=float(df["burden_mean"].mean()),
            mean_household_violation_months=float(df["household_violation_count"].mean()),
            household_safe_rate=float((df["household_violation_count"] == 0).mean() * 100),
            mean_violation_amount_total=float(df["household_violation_amount_total"].mean()),
            mean_violation_ratio_max=float(df["household_violation_ratio_max"].mean()),
            p95_violation_ratio_max=float(df["household_violation_ratio_max"].quantile(0.95)),
            mean_L_applied=float(df["L_applied"].mean()),
            mean_cap_applied=float(df["cap_applied"].mean()),
        ))

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(DATA / "optimal_lt_cap_policy_compare.csv", index=False)
    print(f"  [save] optimal_lt_cap_policy_compare.csv")

    print(f"\n=== 적합 셀러 {n_eligible}명 5-정책 비교 ((L*, cap*) 적용) ===")
    cols = ["policy", "completion_rate", "mean_burden", "household_safe_rate",
            "mean_violation_amount_total", "mean_violation_ratio_max",
            "p95_violation_ratio_max", "mean_L_applied", "mean_cap_applied"]
    pd.set_option('display.width', 200)
    print(summary_df[cols].to_string(index=False))

    print(f"\n[4/5] 종합 통계 저장")
    summary = {
        "config": {
            "m_i": DEFAULT_M_I, "L_personal": DEFAULT_L_PERSONAL,
            "T_max": DEFAULT_T_MAX, "cap_base": DEFAULT_CAP_BASE,
            "L_min_business": L_MIN_BUSINESS,
        },
        "rejection": {
            "n_total": int(n_total),
            "n_eligible": int(n_eligible),
            "n_rejected": int(n_rejected),
            "rejection_rate": float(rejection_rate),
            "by_reason": rejected["reject_reason"].value_counts().to_dict() if len(rejected) > 0 else {},
        },
        "eligible_stats": {
            "type_dist": eligible["type"].value_counts().to_dict(),
            "L_star_mean": float(eligible["L_star"].mean()),
            "L_star_median": float(eligible["L_star"].median()),
            "cap_star_mean": float(eligible["cap_star"].mean()),
            "T_star_mean": float(eligible["T_star"].mean()),
            "R_mean": float(eligible["R"].mean()),
            "R_median": float(eligible["R"].median()),
        },
        "policy_compare": summaries,
    }
    (DATA / "optimal_lt_cap_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    print(f"  [save] optimal_lt_cap_summary.json")

    print(f"\n[5/5] 시각화")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. 거절 vs 적합 pie + 거절 이유
    ax = axes[0, 0]
    if len(rejected) > 0:
        reject_labels = list(rejected["reject_reason"].value_counts().index)
        reject_counts = list(rejected["reject_reason"].value_counts().values)
        labels = ["적합"] + [f"거절: {l[:30]}" for l in reject_labels]
        sizes = [n_eligible] + reject_counts
        colors_p = ["mediumseagreen"] + ["salmon", "orange", "lightcoral"][:len(reject_labels)]
        ax.pie(sizes, labels=labels, colors=colors_p, autopct="%1.1f%%",
                startangle=90, textprops={"fontsize": 8})
    ax.set_title(f"적합 vs 거절 (n={n_total})")

    # 2. 적합 셀러 R 분포
    ax = axes[0, 1]
    ax.hist([eligible["R"], rejected["R"]], bins=30,
            label=["적합", "거절"], color=["mediumseagreen", "salmon"], alpha=0.7, stacked=False)
    ax.axvline(DEFAULT_L_PERSONAL / DEFAULT_M_I, color="red", linestyle="--",
               label=f"임계 R={DEFAULT_L_PERSONAL/DEFAULT_M_I:.0f}만")
    ax.set_xlabel("매출 R (만원)")
    ax.set_ylabel("셀러 수")
    ax.set_title("매출 R 분포 (적합/거절)")
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_xlim(0, eligible["R"].max() * 1.1)

    # 3. 적합 셀러 (L*, cap*) 분포
    ax = axes[0, 2]
    sc = ax.scatter(eligible["L_star"], eligible["cap_star"],
                     c=eligible["R"], cmap="viridis", s=50, alpha=0.7)
    plt.colorbar(sc, ax=ax, label="매출 R")
    ax.set_xlabel("L* (만원)")
    ax.set_ylabel("cap* (이자율)")
    ax.set_title("적합 셀러 (L*, cap*) 분포")
    ax.grid(alpha=0.3)

    # 4. 정책별 completion + household_safe_rate
    ax = axes[1, 0]
    pol_names = [s["policy"] for s in summaries]
    completions = [s["completion_rate"] for s in summaries]
    safe_rates = [s["household_safe_rate"] for s in summaries]
    x_pos = np.arange(len(pol_names))
    width = 0.35
    ax.bar(x_pos - width/2, completions, width, label="Completion %", color="steelblue", alpha=0.8)
    ax.bar(x_pos + width/2, safe_rates, width, label="HH 안전 %", color="mediumseagreen", alpha=0.8)
    for i, (c, s) in enumerate(zip(completions, safe_rates)):
        ax.text(i - width/2, c + 1, f"{c:.0f}", ha="center", fontsize=8)
        ax.text(i + width/2, s + 1, f"{s:.0f}", ha="center", fontsize=8)
    ax.set_xticks(x_pos); ax.set_xticklabels(pol_names, rotation=15, fontsize=8)
    ax.set_title(f"적합 {n_eligible}명 정책별 완납률 + 가계 안전률")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

    # 5. 정책별 침해 강도 (max + p95)
    ax = axes[1, 1]
    maxes = [s["mean_violation_ratio_max"] for s in summaries]
    p95s = [s["p95_violation_ratio_max"] for s in summaries]
    ax.bar(x_pos - width/2, maxes, width, label="mean 단일월 최대비",
           color="crimson", alpha=0.7)
    ax.bar(x_pos + width/2, p95s, width, label="P95 단일월 최대비",
           color="darkred", alpha=0.7)
    for i, (m, p) in enumerate(zip(maxes, p95s)):
        ax.text(i - width/2, m + 0.02, f"{m:.2f}", ha="center", fontsize=8)
        ax.text(i + width/2, p + 0.02, f"{p:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x_pos); ax.set_xticklabels(pol_names, rotation=15, fontsize=8)
    ax.set_title("침해 강도 (가계비 1배 기준)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

    # 6. 적합 셀러 type × cap*
    ax = axes[1, 2]
    type_groups = sorted(eligible["type"].unique())
    cap_by_type = [eligible[eligible["type"] == t]["cap_star"].values for t in type_groups]
    bp = ax.boxplot(cap_by_type, labels=type_groups, patch_artist=True)
    for patch, t in zip(bp["boxes"], type_groups):
        patch.set_facecolor("lightblue"); patch.set_alpha(0.7)
    ax.set_ylabel("cap* (적용 이자율)")
    ax.set_title("적합 셀러 유형별 cap* (위험 프리미엄)")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(alpha=0.3, axis="y")

    plt.suptitle(
        f"Day 11+: 셀러별 (L*, T*, cap*) 최적화 + 침해 0% 엄격 거절\n"
        f"거절 {n_rejected}명 ({rejection_rate:.1f}%) / 적합 {n_eligible}명 ({100-rejection_rate:.1f}%)",
        fontsize=13, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(DATA / "optimal_lt_cap_analysis.png", dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  [save] optimal_lt_cap_analysis.png")
    print("\n=== Day 11+ 완료 ===")


if __name__ == "__main__":
    main()
