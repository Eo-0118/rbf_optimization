"""B-2: Prophet 통합 평가 — Horizon별 정확도 + 분위수(분포) 품질

배경:
  B-1: 단기/장기 horizon 점 예측 정확도 측정 완료
  B-2: 분포 예측 품질 측정 (coverage, sharpness, pinball loss)
  → 점 예측이 부족해도 분포 예측이 잘 calibrated되면 RBF state 통합 가치 ↑

산출:
  Data/prophet_v2_full_eval_results.csv (셀러 × horizon × 점/분위수)
  Data/prophet_v2_distribution_eval.json (분포 품질 요약)
  Data/prophet_v2_full_eval.png (시각화)

Prophet 단일 학습 (5-10분, 273명) → horizon + distribution 동시 측정
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path("/Users/eoseungyun/Desktop/project/SW_Capstone/Project")
DATA = ROOT / "Data"

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

SEED = 42
TRAIN_MONTHS = 18
VAL_MONTHS = 3
TEST_MONTHS = 3
SAMPLES_PER_TYPE = 50
INTERVAL_WIDTH = 0.8   # 80% confidence interval → [P10, P90]


def mape_pt(actual, pred):
    a = float(actual); p = float(pred)
    if a <= 0:
        return float("nan")
    return abs(a - p) / a * 100


def smape_pt(actual, pred):
    a, p = float(actual), float(pred)
    d = (abs(a) + abs(p)) / 2
    if d <= 0:
        return float("nan")
    return abs(a - p) / d * 100


def wape_arr(actual, pred):
    a, p = np.asarray(actual, dtype=float), np.asarray(pred, dtype=float)
    s = np.abs(a).sum()
    if s == 0:
        return float("nan")
    return float(np.abs(a - p).sum() / s * 100)


def pinball_loss(actual, pred_q, q):
    """Quantile (pinball) loss for a single quantile prediction.
    q: 분위수 (예: 0.1, 0.5, 0.9)
    """
    a, p = float(actual), float(pred_q)
    diff = a - p
    return q * diff if diff > 0 else (q - 1) * diff


def load_cohort():
    df = pd.read_parquet(DATA / "cohort_kr_v2.parquet")
    df["date"] = pd.to_datetime(df["date"])
    return df


def sample_sellers(df, n_per_type):
    rng = np.random.default_rng(SEED)
    seller_types = df.groupby("seller_id")["type"].first()
    sids = []
    for typ, group in seller_types.groupby(seller_types):
        ids = group.index.tolist()
        k = min(n_per_type, len(ids))
        sids.extend(rng.choice(ids, size=k, replace=False).tolist())
    return sids


def fit_predict_full(seller_df):
    """Prophet 학습 + horizon별 점 + 분위수 (P10, P50, P90) 동시 반환."""
    from prophet import Prophet
    s = seller_df.sort_values("date").reset_index(drop=True)
    if len(s) != 24:
        return None
    train = s.iloc[:TRAIN_MONTHS].copy()
    val = s.iloc[TRAIN_MONTHS:TRAIN_MONTHS + VAL_MONTHS].copy()
    test = s.iloc[TRAIN_MONTHS + VAL_MONTHS:].copy()
    if (train["monthly_revenue"] > 0).sum() < 6:
        return None

    train_df = pd.DataFrame({"ds": train["date"].values, "y": train["monthly_revenue"].values})
    try:
        m = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False,
                    seasonality_mode="additive", interval_width=INTERVAL_WIDTH)
        m.add_seasonality(name="monthly", period=30.5, fourier_order=3)
        m.fit(train_df)
    except Exception as e:
        return {"error": str(e)}

    full_df = pd.DataFrame({"ds": s["date"].values})
    fcst = m.predict(full_df)
    fcst = fcst[["ds", "yhat", "yhat_lower", "yhat_upper"]].set_index("ds")
    for c in ["yhat", "yhat_lower", "yhat_upper"]:
        fcst[c] = fcst[c].clip(lower=0)

    # h+1, ..., h+6 (val + test)
    horizons_dates = list(val["date"].values) + list(test["date"].values)
    actuals = list(val["monthly_revenue"].values) + list(test["monthly_revenue"].values)

    horizon_data = []
    for h_i, (d, a) in enumerate(zip(horizons_dates, actuals)):
        row = fcst.loc[d]
        yhat = float(row["yhat"])
        p10 = float(row["yhat_lower"])
        p90 = float(row["yhat_upper"])
        actual = float(a)
        # 분위수 평가 지표
        covered = (p10 <= actual <= p90)
        sharp = p90 - p10
        pin_10 = pinball_loss(actual, p10, 0.1)
        pin_50 = pinball_loss(actual, yhat, 0.5)
        pin_90 = pinball_loss(actual, p90, 0.9)
        pin_avg = (pin_10 + pin_50 + pin_90) / 3
        horizon_data.append(dict(
            horizon=f"h+{h_i+1}",
            actual=actual, yhat=yhat, p10=p10, p90=p90,
            ape=mape_pt(actual, yhat),
            smape=smape_pt(actual, yhat),
            covered=bool(covered),
            sharp=sharp,
            sharp_rel=sharp / max(actual, 1.0),   # 실제값 대비 구간 폭
            pin_10=pin_10, pin_50=pin_50, pin_90=pin_90, pin_avg=pin_avg,
        ))
    return horizon_data


def main():
    print("[1/4] Cohort + 샘플링")
    df = load_cohort()
    sids = sample_sellers(df, SAMPLES_PER_TYPE)
    print(f"  샘플 셀러: {len(sids)}")

    print(f"\n[2/4] Prophet 학습 + horizon + 분위수 동시 측정")
    long_rows = []
    for i, sid in enumerate(sids):
        if i % 25 == 0 and i > 0:
            print(f"  [{i}/{len(sids)}]")
        seller_df = df[df["seller_id"] == sid]
        typ = seller_df["type"].iloc[0]
        out = fit_predict_full(seller_df)
        if out is None or isinstance(out, dict):
            continue
        for hd in out:
            long_rows.append(dict(seller_id=sid, type=typ, **hd))

    long_df = pd.DataFrame(long_rows)
    long_df.to_csv(DATA / "prophet_v2_full_eval_results.csv", index=False)
    print(f"  [save] prophet_v2_full_eval_results.csv ({len(long_df)} rows)")

    print(f"\n[3/4] 평가 지표 요약")

    summary = {"by_horizon": {}, "by_horizon_group": {}, "overall": {}}

    # Horizon별 (h+1, ..., h+6)
    for h in sorted(long_df["horizon"].unique(), key=lambda x: int(x.split("+")[1])):
        sub = long_df[long_df["horizon"] == h]
        coverage = sub["covered"].mean() * 100
        sharp_med = sub["sharp"].median()
        sharp_rel_med = sub["sharp_rel"].median()
        pin_avg = sub["pin_avg"].mean()
        ape_med = sub["ape"].median()
        ape_under_20 = (sub["ape"] < 20).mean() * 100
        summary["by_horizon"][h] = {
            "n": int(len(sub)),
            "ape_median": float(ape_med),
            "ape_under_20_pct": float(ape_under_20),
            "coverage_pct": float(coverage),
            "sharpness_median": float(sharp_med),
            "sharpness_rel_median": float(sharp_rel_med),
            "pinball_avg_mean": float(pin_avg),
        }

    # 그룹별
    for gname, hs in [("short_t1_t3", ["h+1", "h+2", "h+3"]),
                      ("long_t4_t6", ["h+4", "h+5", "h+6"]),
                      ("all_t1_t6", ["h+1", "h+2", "h+3", "h+4", "h+5", "h+6"])]:
        sub = long_df[long_df["horizon"].isin(hs)]
        summary["by_horizon_group"][gname] = {
            "n": int(len(sub)),
            "ape_median": float(sub["ape"].median()),
            "ape_under_20_pct": float((sub["ape"] < 20).mean() * 100),
            "coverage_pct": float(sub["covered"].mean() * 100),
            "sharpness_median": float(sub["sharp"].median()),
            "sharpness_rel_median": float(sub["sharp_rel"].median()),
            "pinball_avg_mean": float(sub["pin_avg"].mean()),
        }

    # 전체
    summary["overall"] = {
        "n_total": int(len(long_df)),
        "interval_width_target": INTERVAL_WIDTH,
        "actual_coverage": float(long_df["covered"].mean() * 100),
        "coverage_gap_pct": float(long_df["covered"].mean() * 100 - INTERVAL_WIDTH * 100),
        "sharpness_median": float(long_df["sharp"].median()),
        "sharpness_rel_median": float(long_df["sharp_rel"].median()),
    }

    (DATA / "prophet_v2_distribution_eval.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"  [save] prophet_v2_distribution_eval.json")

    print(f"\n=== 분포 예측 품질 요약 ===")
    print(f"\n  [Horizon별]")
    print(f"  {'horizon':8s} {'APE<20%':>8s} {'Cov.':>8s} {'Sharp(med)':>12s} {'Sharp_rel':>10s} {'Pinball':>10s}")
    for h, s in summary["by_horizon"].items():
        print(f"  {h:8s} {s['ape_under_20_pct']:7.1f}% {s['coverage_pct']:7.1f}% "
              f"{s['sharpness_median']:11.1f} {s['sharpness_rel_median']:9.2f}  {s['pinball_avg_mean']:9.1f}")

    print(f"\n  [그룹별]")
    for g, s in summary["by_horizon_group"].items():
        print(f"  {g:14s}: APE<20% {s['ape_under_20_pct']:5.1f}%, Coverage {s['coverage_pct']:5.1f}%, "
              f"Sharp_rel {s['sharpness_rel_median']:.2f}, Pinball {s['pinball_avg_mean']:.1f}")

    print(f"\n  [전체]")
    print(f"  목표 Coverage: {INTERVAL_WIDTH*100:.0f}% / 실제: {summary['overall']['actual_coverage']:.1f}% "
          f"(격차 {summary['overall']['coverage_gap_pct']:+.1f}%p)")
    print(f"  → Coverage {summary['overall']['actual_coverage']:.0f}%면 ", end="")
    if abs(summary['overall']['coverage_gap_pct']) < 10:
        print("calibrated ✓ (분포 신뢰 가능)")
    else:
        print("under/over-confident ⚠️ (재조정 필요)")

    print(f"\n[4/4] 시각화")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    hs_ord = sorted(long_df["horizon"].unique(), key=lambda x: int(x.split("+")[1]))

    # 1. Coverage by horizon
    ax = axes[0, 0]
    covs = [summary["by_horizon"][h]["coverage_pct"] for h in hs_ord]
    colors_cov = ["green" if abs(c - 80) < 10 else "orange" if abs(c - 80) < 20 else "red" for c in covs]
    bars = ax.bar(hs_ord, covs, color=colors_cov, alpha=0.7)
    for b, v in zip(bars, covs):
        ax.text(b.get_x() + b.get_width()/2, v + 1, f"{v:.0f}%", ha="center", fontweight="bold")
    ax.axhline(80, color="red", linestyle="--", alpha=0.7, label="목표 80%")
    ax.set_ylabel("Coverage % (실제값 in [P10, P90])")
    ax.set_title("Horizon별 Coverage (목표 80%)")
    ax.set_ylim(0, 105); ax.legend(); ax.grid(alpha=0.3, axis="y")

    # 2. Sharpness (relative) by horizon
    ax = axes[0, 1]
    sharps = [summary["by_horizon"][h]["sharpness_rel_median"] for h in hs_ord]
    ax.bar(hs_ord, sharps, color="steelblue", alpha=0.7)
    for i, v in enumerate(sharps):
        ax.text(i, v + 0.05, f"{v:.2f}", ha="center", fontweight="bold")
    ax.set_ylabel("Sharpness rel (구간폭 / 실제값)")
    ax.set_title("Horizon별 Sharpness (낮을수록 정보 가치 ↑)")
    ax.grid(alpha=0.3, axis="y")

    # 3. APE <20% (점 예측)
    ax = axes[0, 2]
    apes = [summary["by_horizon"][h]["ape_under_20_pct"] for h in hs_ord]
    ax.bar(hs_ord, apes, color="mediumseagreen", alpha=0.7)
    for i, v in enumerate(apes):
        ax.text(i, v + 1, f"{v:.0f}%", ha="center", fontweight="bold")
    ax.axhline(30, color="red", linestyle="--", alpha=0.5, label="기준 30%")
    ax.set_ylabel("APE < 20% 시점 비율 (%)")
    ax.set_title("점 예측 정확도 (참고)")
    ax.legend(); ax.grid(alpha=0.3, axis="y")

    # 4. Coverage 산점도 (셀러별 분포)
    ax = axes[1, 0]
    seller_cov = long_df.groupby("seller_id")["covered"].mean() * 100
    ax.hist(seller_cov, bins=20, color="steelblue", alpha=0.7, edgecolor="white")
    ax.axvline(80, color="red", linestyle="--", label="목표 80%")
    ax.axvline(seller_cov.mean(), color="darkgreen", linestyle="--",
               label=f"평균 {seller_cov.mean():.1f}%")
    ax.set_xlabel("셀러별 평균 Coverage (%)")
    ax.set_ylabel("셀러 수")
    ax.set_title("Coverage 분포 (셀러별)")
    ax.legend(); ax.grid(alpha=0.3)

    # 5. Pinball loss by horizon
    ax = axes[1, 1]
    pins = [summary["by_horizon"][h]["pinball_avg_mean"] for h in hs_ord]
    ax.bar(hs_ord, pins, color="crimson", alpha=0.7)
    for i, v in enumerate(pins):
        ax.text(i, v + max(pins) * 0.02, f"{v:.0f}", ha="center", fontweight="bold", fontsize=9)
    ax.set_ylabel("Pinball loss avg (P10/P50/P90)")
    ax.set_title("Horizon별 Pinball Loss (낮을수록 좋음)")
    ax.grid(alpha=0.3, axis="y")

    # 6. 분위수 예측 예시 (랜덤 셀러 1개)
    ax = axes[1, 2]
    sample_sid = long_df["seller_id"].iloc[0]
    s_data = long_df[long_df["seller_id"] == sample_sid].sort_values("horizon", key=lambda c: c.str.split("+").str[1].astype(int))
    x_h = np.arange(len(s_data))
    ax.plot(x_h, s_data["actual"], "o-", color="black", label="실제")
    ax.plot(x_h, s_data["yhat"], "x--", color="crimson", label="P50 (예측)")
    ax.fill_between(x_h, s_data["p10"], s_data["p90"], color="crimson", alpha=0.2, label="[P10, P90]")
    ax.set_xticks(x_h); ax.set_xticklabels(s_data["horizon"].tolist())
    ax.set_xlabel("Horizon"); ax.set_ylabel("매출 (만원)")
    ax.set_title(f"예시 셀러: {sample_sid[:18]}...")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.suptitle(f"B-2 Prophet 통합 평가 — 점 + 분위수 예측 (n={len(long_df)//6}, 목표 Coverage 80%)",
                 fontsize=13, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(DATA / "prophet_v2_full_eval.png", dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  [save] prophet_v2_full_eval.png")

    print("\n=== 완료 ===")
    print(f"  → 다음: 분포 통합 결정 (Day 2-3)")


if __name__ == "__main__":
    main()
