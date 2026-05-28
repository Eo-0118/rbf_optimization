"""B-1: Prophet 단기 vs 장기 horizon 정확도 분리 측정

배경:
  현재 평가는 test 6개월 평균 WAPE만 측정 (WAPE<20% 17.8%)
  RBF 활용 관점:
    - PPO는 매월 step에서 미래 1-3개월만 알면 충분 (단기)
    - CVaR은 전체 24개월 시나리오 필요 (장기)
  단기 정확도가 6개월 평균보다 좋으면 PPO 통합 가능성 확보

산출:
  Data/prophet_horizon_results.csv (셀러 × horizon × 지표)
  Data/forecast_horizon_summary.json (단기/장기 분리 요약)

기존 prophet_baseline_v2와 동일 학습 (SAMPLES_PER_TYPE=50, SEED=42)
→ test 6개월 예측을 t+1, t+2, t+3, t+4, t+5, t+6 각각 분리 측정
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


def mape(actual, pred):
    a, p = np.asarray(actual, dtype=float), np.asarray(pred, dtype=float)
    mask = a > 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((a[mask] - p[mask]) / a[mask])) * 100)


def smape(actual, pred):
    a, p = np.asarray(actual, dtype=float), np.asarray(pred, dtype=float)
    denom = (np.abs(a) + np.abs(p)) / 2.0
    safe = denom > 0
    if safe.sum() == 0:
        return float("nan")
    err = np.zeros_like(a)
    err[safe] = np.abs(a[safe] - p[safe]) / denom[safe]
    return float(np.mean(err) * 100)


def wape(actual, pred):
    a, p = np.asarray(actual, dtype=float), np.asarray(pred, dtype=float)
    s = np.abs(a).sum()
    if s == 0:
        return float("nan")
    return float(np.abs(a - p).sum() / s * 100)


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


def fit_predict_seller_horizons(seller_df):
    """v2 방식 학습 + horizon별 (t+1 ~ t+6) 개별 예측/실제 반환."""
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
                    seasonality_mode="additive", interval_width=0.8)
        m.add_seasonality(name="monthly", period=30.5, fourier_order=3)
        m.fit(train_df)
    except Exception as e:
        return {"error": str(e)}

    full_df = pd.DataFrame({"ds": s["date"].values})
    fcst = m.predict(full_df)
    fcst = fcst[["ds", "yhat"]].set_index("ds")
    fcst["yhat"] = fcst["yhat"].clip(lower=0)

    val_pred = fcst.loc[val["date"].values, "yhat"].values
    test_pred = fcst.loc[test["date"].values, "yhat"].values
    val_actual = val["monthly_revenue"].values
    test_actual = test["monthly_revenue"].values

    # Horizon = val + test = 6개월. h+1 부터 h+6.
    full_actual = np.concatenate([val_actual, test_actual])  # h+1, h+2, ..., h+6
    full_pred = np.concatenate([val_pred, test_pred])

    horizon_metrics = {}
    for h in range(len(full_actual)):
        a = np.array([full_actual[h]])
        p = np.array([full_pred[h]])
        horizon_metrics[f"h+{h+1}"] = {
            "actual": float(a[0]),
            "pred": float(p[0]),
            # 단일 시점 mape/smape — 절대값
            "mape": mape(a, p),
            "smape": smape(a, p),
            "ape": float(abs(a[0] - p[0]) / a[0] * 100) if a[0] > 0 else float("nan"),  # absolute pct error
        }

    return {
        "horizon_metrics": horizon_metrics,
        "test_wape_avg": wape(test_actual, test_pred),     # 기존 호환 (h+4~h+6)
        "val_wape_avg": wape(val_actual, val_pred),         # 기존 호환 (h+1~h+3)
        "all6_wape": wape(full_actual, full_pred),
    }


def main():
    print("[1/4] Cohort + 샘플링")
    df = load_cohort()
    sids = sample_sellers(df, SAMPLES_PER_TYPE)
    print(f"  샘플 셀러: {len(sids)}")

    print(f"\n[2/4] Prophet 학습 + horizon별 예측 ({len(sids)}개 모델)")
    long_rows = []   # 셀러 × horizon long-format
    seller_rows = []  # 셀러별 요약
    for i, sid in enumerate(sids):
        if i % 25 == 0 and i > 0:
            print(f"  [{i}/{len(sids)}]")
        seller_df = df[df["seller_id"] == sid]
        typ = seller_df["type"].iloc[0]
        out = fit_predict_seller_horizons(seller_df)
        if out is None or "error" in out:
            continue
        for h_name, hm in out["horizon_metrics"].items():
            long_rows.append(dict(
                seller_id=sid, type=typ, horizon=h_name,
                actual=hm["actual"], pred=hm["pred"],
                ape=hm["ape"], mape=hm["mape"], smape=hm["smape"],
            ))
        seller_rows.append(dict(
            seller_id=sid, type=typ,
            val_wape=out["val_wape_avg"],   # h+1~h+3 평균
            test_wape=out["test_wape_avg"], # h+4~h+6 평균
            all6_wape=out["all6_wape"],
        ))

    long_df = pd.DataFrame(long_rows)
    seller_df_out = pd.DataFrame(seller_rows)
    long_df.to_csv(DATA / "prophet_horizon_results.csv", index=False)
    print(f"  [save] prophet_horizon_results.csv")

    print(f"\n[3/4] Horizon별 분리 요약")
    summary = {"by_horizon": {}, "by_horizon_group": {}}

    # h+1, ..., h+6 개별
    for h in sorted(long_df["horizon"].unique(), key=lambda x: int(x.split("+")[1])):
        sub = long_df[long_df["horizon"] == h]
        valid_ape = sub["ape"].dropna()
        summary["by_horizon"][h] = {
            "n": int(len(sub)),
            "n_valid_ape": int(len(valid_ape)),
            "ape_mean": float(valid_ape.mean()),
            "ape_median": float(valid_ape.median()),
            "ape_under_20_pct": float((valid_ape < 20).mean() * 100),
            "ape_under_50_pct": float((valid_ape < 50).mean() * 100),
        }
        print(f"  {h}: APE mean={summary['by_horizon'][h]['ape_mean']:6.1f}% "
              f"median={summary['by_horizon'][h]['ape_median']:6.1f}%  "
              f"<20%: {summary['by_horizon'][h]['ape_under_20_pct']:5.1f}%  "
              f"<50%: {summary['by_horizon'][h]['ape_under_50_pct']:5.1f}%")

    # 그룹별: 단기(h+1~h+3) vs 장기(h+4~h+6)
    for group_name, hs in [("short_t1_t3", ["h+1", "h+2", "h+3"]),
                           ("long_t4_t6", ["h+4", "h+5", "h+6"]),
                           ("all_t1_t6", ["h+1", "h+2", "h+3", "h+4", "h+5", "h+6"])]:
        sub = long_df[long_df["horizon"].isin(hs)]
        valid_ape = sub["ape"].dropna()
        summary["by_horizon_group"][group_name] = {
            "n": int(len(sub)),
            "ape_mean": float(valid_ape.mean()),
            "ape_median": float(valid_ape.median()),
            "ape_under_20_pct": float((valid_ape < 20).mean() * 100),
            "ape_under_50_pct": float((valid_ape < 50).mean() * 100),
        }

    # 셀러 단위 WAPE
    summary["seller_level"] = {
        "n": int(len(seller_df_out)),
        "val_wape_under_20_pct": float((seller_df_out["val_wape"] < 20).mean() * 100),
        "test_wape_under_20_pct": float((seller_df_out["test_wape"] < 20).mean() * 100),
        "all6_wape_under_20_pct": float((seller_df_out["all6_wape"] < 20).mean() * 100),
        "val_wape_median": float(seller_df_out["val_wape"].median()),
        "test_wape_median": float(seller_df_out["test_wape"].median()),
    }

    (DATA / "forecast_horizon_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"  [save] forecast_horizon_summary.json")

    print(f"\n=== 단기 vs 장기 핵심 비교 ===")
    print(f"  [APE 절대 시점별 — 단일 월 예측 정확도]")
    s_short = summary["by_horizon_group"]["short_t1_t3"]
    s_long = summary["by_horizon_group"]["long_t4_t6"]
    print(f"    단기 (h+1~h+3): APE mean {s_short['ape_mean']:5.1f}%, <20%: {s_short['ape_under_20_pct']:.1f}%")
    print(f"    장기 (h+4~h+6): APE mean {s_long['ape_mean']:5.1f}%, <20%: {s_long['ape_under_20_pct']:.1f}%")
    print(f"\n  [셀러별 WAPE — 시계열 통합 정확도]")
    sl = summary["seller_level"]
    print(f"    단기 6개월 (val, h+1~h+3): WAPE<20% {sl['val_wape_under_20_pct']:.1f}% (median {sl['val_wape_median']:.1f}%)")
    print(f"    장기 6개월 (test, h+4~h+6): WAPE<20% {sl['test_wape_under_20_pct']:.1f}% (median {sl['test_wape_median']:.1f}%)")
    print(f"    전체 6개월: WAPE<20% {sl['all6_wape_under_20_pct']:.1f}%")

    print(f"\n[4/4] 시각화")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Horizon별 APE 분포 (boxplot)
    ax = axes[0, 0]
    hs_ord = sorted(long_df["horizon"].unique(), key=lambda x: int(x.split("+")[1]))
    data = [long_df[long_df["horizon"] == h]["ape"].dropna().clip(upper=200).values for h in hs_ord]
    bp = ax.boxplot(data, labels=hs_ord, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("steelblue"); patch.set_alpha(0.6)
    ax.axhline(20, color="red", linestyle="--", alpha=0.5, label="목표 20%")
    ax.set_ylabel("APE (단일 월 예측 오차 %)")
    ax.set_title("Horizon별 APE 분포 (clipped at 200)")
    ax.legend(); ax.grid(alpha=0.3)

    # 2. Horizon별 <20% / <50% 비율
    ax = axes[0, 1]
    pcts_20 = [summary["by_horizon"][h]["ape_under_20_pct"] for h in hs_ord]
    pcts_50 = [summary["by_horizon"][h]["ape_under_50_pct"] for h in hs_ord]
    x = np.arange(len(hs_ord))
    width = 0.35
    ax.bar(x - width/2, pcts_20, width, label="APE < 20%", color="mediumseagreen")
    ax.bar(x + width/2, pcts_50, width, label="APE < 50%", color="lightsteelblue")
    for i, (v1, v2) in enumerate(zip(pcts_20, pcts_50)):
        ax.text(i - width/2, v1 + 1, f"{v1:.0f}%", ha="center", fontsize=8)
        ax.text(i + width/2, v2 + 1, f"{v2:.0f}%", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(hs_ord)
    ax.set_ylabel("셀러 비율 (%)")
    ax.set_title("Horizon별 정확도 비율")
    ax.legend(); ax.grid(alpha=0.3, axis="y")

    # 3. APE 시점별 평균 (line)
    ax = axes[1, 0]
    apes_mean = [summary["by_horizon"][h]["ape_mean"] for h in hs_ord]
    apes_med = [summary["by_horizon"][h]["ape_median"] for h in hs_ord]
    ax.plot(hs_ord, apes_mean, "o-", color="crimson", label="mean APE")
    ax.plot(hs_ord, apes_med, "s-", color="steelblue", label="median APE")
    ax.axhline(20, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("Horizon")
    ax.set_ylabel("APE %")
    ax.set_title("Horizon이 길수록 정확도 떨어지는가?")
    ax.legend(); ax.grid(alpha=0.3)

    # 4. 단기 vs 장기 그룹 비교
    ax = axes[1, 1]
    groups = ["단기\n(h+1~h+3)", "장기\n(h+4~h+6)", "전체\n(h+1~h+6)"]
    pcts_20_g = [
        summary["by_horizon_group"]["short_t1_t3"]["ape_under_20_pct"],
        summary["by_horizon_group"]["long_t4_t6"]["ape_under_20_pct"],
        summary["by_horizon_group"]["all_t1_t6"]["ape_under_20_pct"],
    ]
    bars = ax.bar(groups, pcts_20_g, color=["mediumseagreen", "orange", "gray"])
    for b, v in zip(bars, pcts_20_g):
        ax.text(b.get_x() + b.get_width()/2, v + 1, f"{v:.1f}%", ha="center", fontweight="bold")
    ax.axhline(20, color="red", linestyle="--", alpha=0.5, label="목표 20%")
    ax.set_ylabel("APE < 20% 시점 비율 (%)")
    ax.set_title("단기 vs 장기 정확도 비교")
    ax.legend(); ax.grid(alpha=0.3, axis="y")

    plt.suptitle(f"B-1: Prophet 단기/장기 horizon 정확도 분리 (n={len(sids)} 셀러)",
                 fontsize=13, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(DATA / "forecast_horizon_analysis.png", dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  [save] forecast_horizon_analysis.png")

    print("\n=== 완료 ===")


if __name__ == "__main__":
    main()
