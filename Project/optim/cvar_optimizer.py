"""CVaR 기반 정적 r* 최적화 (Phase 3 v1)

각 셀러에 대해:
  1. 셀러의 합성 매출 path → Monte Carlo N개 시나리오 생성
  2. CVXPY로 단일 r* 최적화:
       maximize E[recovery_capped] - λ · CVaR_α(burden)
       s.t. r ∈ [r_min, r_max]
  3. Output: per-seller r* + CVaR 통계

논리 (본 연구의 핵심 설계):
- recovery는 cap 적용 (target 도달 시 더 회수해도 의미 없음)
- CVaR는 **가계 침범 위험** 측정 (5% 최악 시나리오의 가계 침범 합)
- 즉 "회수 최대 + 가계 침범 위험 통제" trade-off
- 단일 r 기반 (월별 동적 조정은 RL 단계)

산출:
- Data/cvar_optimizer_results.csv
- Data/cvar_optimizer_summary.json
- Data/cvar_optimizer_diagnostics.png
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path("/Users/eoseungyun/Desktop/project/SW_Capstone/Project")
DATA = ROOT / "Data"

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

# === Config (env와 일관) ===
SEED = 42
T = 24
CAP = 1.2
LOAN_MULTIPLIER = 3.0
R_MIN = 0.03
R_MAX = 0.25
M_I = 0.10                              # v2: env와 일관 (스마트스토어 gross 20% - 운영비 10%, 부분 추정)
L_PERSONAL_MIN = 128.21                 # v2: KB 1인가구 / 보건복지부 기준 중위소득 50% (검증)

# CVaR 파라미터
N_SCENARIOS = 200          # 셀러당 Monte Carlo 시나리오 수
ALPHA_CVAR = 0.05          # 5% 꼬리
LAMBDA_CVAR = 1.0          # E[recovery] vs CVaR(burden) trade-off (가계 보호 가중)
NOISE_SCALE = 0.3          # 시나리오 생성 시 노이즈 강도


def generate_scenarios(seller_revenues: np.ndarray, cv: float, n: int = N_SCENARIOS,
                        rng: np.random.Generator | None = None) -> np.ndarray:
    """셀러의 매출 path에 곱셈 Gaussian 노이즈 → N×T 시나리오 행렬.
    노이즈 strength = cv × NOISE_SCALE (셀러 변동성 비례).
    """
    rng = rng or np.random.default_rng(SEED)
    Tt = len(seller_revenues)
    if Tt != T:
        # truncate or pad
        Tt = min(Tt, T)
    base = seller_revenues[:T] if len(seller_revenues) >= T else np.pad(seller_revenues, (0, T - Tt), mode="edge")
    sigma = max(cv * NOISE_SCALE, 1e-6)
    noise = rng.normal(loc=1.0, scale=sigma, size=(n, T))
    noise = np.clip(noise, 0.0, None)   # 음수 매출 방지
    scenarios = base[None, :] * noise
    return scenarios


def optimize_seller_r(scenarios: np.ndarray, L: float, cap: float = CAP,
                       m_i: float = M_I, L_personal_min: float = L_PERSONAL_MIN,
                       r_min: float = R_MIN, r_max: float = R_MAX,
                       alpha: float = ALPHA_CVAR, lambda_cvar: float = LAMBDA_CVAR) -> dict:
    """단일 셀러 CVaR 최적화 (CVXPY).

    수식:
      target = L × cap
      payment[k, t] = r × scenarios[k, t]
      recovery_capped[k] = min(target, sum_t payment[k, t])  ← cap 적용 (over-recovery 무효)
      safe_cap[k, t] = max(0, m_i × scenarios[k, t] - L_personal_min)
      burden[k, t] = max(0, payment[k, t] - safe_cap[k, t])
      burden_total[k] = sum_t burden[k, t]            ← 시나리오별 가계 침범 총합
      CVaR_α(burden) = z + (1/(1-α)) × E[max(burden_total - z, 0)]
      objective: maximize E[recovery_capped] - λ × CVaR(burden)

    핵심 trade-off:
    - r 큼 → recovery 높음 but burden 누적 → CVaR 큼
    - r 작음 → recovery 부족 (target 미달) but burden 작음 → CVaR 작음
    - 최적 r* = trade-off 균형점

    Returns dict: r_star, cvar, expected_recovery, status
    """
    N, T_local = scenarios.shape
    target = L * cap

    # 시나리오별 안전 RBF 한도 (월별, 사전 계산 — 상수)
    safe_cap_matrix = np.maximum(m_i * scenarios - L_personal_min, 0.0)  # (N, T)

    # CVXPY 변수
    r = cp.Variable()
    z = cp.Variable()
    xi = cp.Variable(N, nonneg=True)

    # 시나리오별 총 매출 (24개월 합)
    total_rev = scenarios.sum(axis=1)   # (N,)

    # Recovery (cap 적용)
    raw_recovery = r * total_rev   # (N,) affine in r
    recovery_capped = cp.minimum(raw_recovery, target)   # element-wise min, concave
    expected_recovery = cp.sum(recovery_capped) / N

    # Burden total per scenario: sum_t max(0, r * scenarios[k,t] - safe_cap[k,t])
    payment_matrix = r * scenarios   # (N, T) affine in r
    burden_matrix = cp.pos(payment_matrix - safe_cap_matrix)   # (N, T) ≥ 0
    burden_per_scenario = cp.sum(burden_matrix, axis=1)   # (N,)

    # CVaR(burden)
    cvar = z + (1.0 / (1 - alpha)) * cp.sum(xi) / N

    constraints = [
        r >= r_min,
        r <= r_max,
        xi >= burden_per_scenario - z,
    ]

    objective = cp.Maximize(expected_recovery - lambda_cvar * cvar)
    prob = cp.Problem(objective, constraints)

    try:
        prob.solve(solver=cp.CLARABEL, verbose=False)
        status = prob.status
        if status not in ("optimal", "optimal_inaccurate") or r.value is None:
            return dict(r_star=np.nan, cvar=np.nan, expected_recovery=np.nan, status=status)
        return dict(
            r_star=float(np.clip(r.value, r_min, r_max)),
            cvar=float(cvar.value),
            expected_recovery=float(expected_recovery.value),
            status=status,
        )
    except Exception as e:
        return dict(r_star=np.nan, cvar=np.nan, expected_recovery=np.nan, status=f"error:{e}")


def compute_seller_cv(revenues: np.ndarray) -> float:
    """활성 구간의 CV. 값이 너무 작으면 0.5 default."""
    nz = revenues[revenues > 0]
    if len(nz) < 2 or nz.mean() <= 0:
        return 0.5
    return float(nz.std() / nz.mean())


def main(n_sample: int | None = None):
    print("[1/4] Cohort 로드")
    df = pd.read_parquet(DATA / "cohort_kr_v2.parquet")
    df["date"] = pd.to_datetime(df["date"])
    sellers = list(df["seller_id"].unique())
    if n_sample is not None:
        rng = np.random.default_rng(SEED)
        sellers = list(rng.choice(sellers, size=min(n_sample, len(sellers)), replace=False))
    print(f"  대상 셀러: {len(sellers)}")

    print(f"\n[2/4] 셀러별 CVaR 최적화 (N={N_SCENARIOS} 시나리오, λ={LAMBDA_CVAR})")
    rng = np.random.default_rng(SEED)
    rows = []
    failed = 0
    for i, sid in enumerate(sellers):
        if i % 100 == 0 and i > 0:
            print(f"  [{i}/{len(sellers)}]  failed so far: {failed}")
        sdf = df[df["seller_id"] == sid].sort_values("month_idx")
        revs = sdf["monthly_revenue"].values.astype(float)
        cv = compute_seller_cv(revs)
        mean_rev = float(revs.mean())
        L = LOAN_MULTIPLIER * mean_rev
        seller_type = sdf["type"].iloc[0]

        scenarios = generate_scenarios(revs, cv, n=N_SCENARIOS, rng=rng)
        result = optimize_seller_r(scenarios, L=L)

        if np.isnan(result["r_star"]):
            failed += 1

        rows.append(dict(
            seller_id=sid, type=seller_type, cv=cv, mean_rev=mean_rev, L=L,
            target=L * CAP,
            r_star=result["r_star"],
            cvar=result["cvar"],
            expected_recovery=result["expected_recovery"],
            status=result["status"],
        ))

    res_df = pd.DataFrame(rows)
    print(f"  완료. 성공={len(res_df) - failed}, 실패={failed}")

    print("\n[3/4] 결과 저장 + 통계")
    res_df.to_csv(DATA / "cvar_optimizer_results.csv", index=False)
    valid = res_df[~res_df["r_star"].isna()]

    summary = {
        "config": {
            "n_scenarios": N_SCENARIOS, "alpha": ALPHA_CVAR, "lambda": LAMBDA_CVAR,
            "r_range": [R_MIN, R_MAX], "cap": CAP, "loan_multiplier": LOAN_MULTIPLIER,
            "noise_scale": NOISE_SCALE,
        },
        "n_total": int(len(res_df)),
        "n_success": int(len(valid)),
        "n_failed": int(failed),
        "r_star_stats": {
            "mean": float(valid["r_star"].mean()),
            "median": float(valid["r_star"].median()),
            "std": float(valid["r_star"].std()),
            "min": float(valid["r_star"].min()),
            "max": float(valid["r_star"].max()),
            "p25": float(valid["r_star"].quantile(0.25)),
            "p75": float(valid["r_star"].quantile(0.75)),
        },
        "by_type": {},
    }
    for typ, g in valid.groupby("type"):
        summary["by_type"][typ] = dict(
            n=int(len(g)),
            r_star_mean=float(g["r_star"].mean()),
            r_star_median=float(g["r_star"].median()),
            cvar_mean=float(g["cvar"].mean()),
            expected_recovery_mean=float(g["expected_recovery"].mean()),
        )

    (DATA / "cvar_optimizer_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"\n=== CVaR Optimizer 결과 ===")
    print(f"  성공률: {len(valid)}/{len(res_df)} ({len(valid)/len(res_df)*100:.1f}%)")
    print(f"\n  [r* 통계]")
    print(f"   mean={summary['r_star_stats']['mean']:.4f}  median={summary['r_star_stats']['median']:.4f}  "
          f"std={summary['r_star_stats']['std']:.4f}")
    print(f"   range=[{summary['r_star_stats']['min']:.4f}, {summary['r_star_stats']['max']:.4f}]")
    print(f"   IQR=[{summary['r_star_stats']['p25']:.4f}, {summary['r_star_stats']['p75']:.4f}]")
    print(f"\n  [유형별 r* 평균]")
    for typ, s in summary["by_type"].items():
        print(f"    {typ:10s}: r*={s['r_star_mean']:.4f}  CVaR={s['cvar_mean']:+.2f}  "
              f"E[recovery]={s['expected_recovery_mean']:.2f}  (n={s['n']})")

    print("\n[4/4] 시각화")
    color_map = {"stable": "steelblue", "growth": "mediumseagreen",
                 "volatile": "crimson", "seasonal": "darkorange",
                 "decline": "gray", "other": "lightgray"}

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # (1) r* 분포 히스토그램
    ax = axes[0, 0]
    ax.hist(valid["r_star"], bins=30, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(valid["r_star"].median(), color="red", linestyle="--",
               label=f"median {valid['r_star'].median():.3f}")
    ax.axvline(0.15, color="green", linestyle=":", alpha=0.5, label="Fixed-0.15 baseline")
    ax.set_xlabel("r* (수수료율)"); ax.set_ylabel("셀러 수")
    ax.set_title("CVaR 최적화 r* 분포")
    ax.legend(); ax.grid(alpha=0.3)

    # (2) 유형별 r* boxplot
    ax = axes[0, 1]
    types_ordered = sorted(valid["type"].unique())
    data_lst = [valid[valid["type"] == t]["r_star"].values for t in types_ordered]
    bp = ax.boxplot(data_lst, labels=types_ordered, patch_artist=True)
    for patch, t in zip(bp["boxes"], types_ordered):
        patch.set_facecolor(color_map.get(t, "gray"))
        patch.set_alpha(0.7)
    ax.axhline(0.15, color="green", linestyle=":", alpha=0.5, label="Fixed-0.15")
    ax.set_ylabel("r*"); ax.set_title("유형별 r* 분포")
    ax.tick_params(axis="x", rotation=15); ax.legend(); ax.grid(alpha=0.3)

    # (3) r* vs CV (셀러 변동성과 r*의 관계)
    ax = axes[1, 0]
    for typ in types_ordered:
        sub = valid[valid["type"] == typ]
        ax.scatter(sub["cv"], sub["r_star"], c=color_map.get(typ, "gray"),
                   alpha=0.6, s=15, label=typ)
    ax.set_xlabel("CV (셀러 변동성)"); ax.set_ylabel("r*")
    ax.set_title("r* vs 셀러 CV (높은 CV → 높은 r*?)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (4) E[recovery] vs CVaR
    ax = axes[1, 1]
    for typ in types_ordered:
        sub = valid[valid["type"] == typ]
        ax.scatter(sub["cvar"], sub["expected_recovery"], c=color_map.get(typ, "gray"),
                   alpha=0.6, s=15, label=typ)
    ax.set_xlabel("CVaR (5% 꼬리 손실)"); ax.set_ylabel("E[recovery]")
    ax.set_title("CVaR vs 기대 회수 (trade-off)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.suptitle(f"CVaR Optimizer 결과 (n={len(valid)}, "
                 f"r* mean={valid['r_star'].mean():.3f})",
                 fontsize=13, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(DATA / "cvar_optimizer_diagnostics.png", dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  [save] cvar_optimizer_diagnostics.png")

    print("\n=== 완료 ===")


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(n_sample=n)
