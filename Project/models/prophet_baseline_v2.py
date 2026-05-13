"""Phase 2: Prophet 베이스라인 v2 — Robust 지표 + 음수 예측 차단

v1 대비 변경:
1. 예측값 post-process: max(yhat, 0) — 음수 매출 비현실
2. MAPE에 더해 SMAPE, WAPE 추가 측정
   - MAPE: 작은 actual 값에서 폭증 (분모 민감)
   - SMAPE: 대칭, 0~200% 유한 범위
   - WAPE: sum 기반, 단일 셀러 시계열 전체 절대오차/전체매출
3. 시각화: 3가지 지표 비교 + 유형별 분포

분할: 24개월 → 학습 18 / 검증 3 / 테스트 3
스코프: 유형별 50명 (v1과 동일하게 비교 가능)
외생 변수: 없음 (다음 v3에서 추가)

산출:
  Data/prophet_baseline_v2_results.csv
  Data/prophet_baseline_v2_summary.json
  Data/prophet_baseline_v2_diagnostics.png
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


# === Robust 지표 정의 ===
def mape(actual: np.ndarray, pred: np.ndarray) -> float:
    """MAPE — actual=0 제외 (분모 0 방지). 작은 값에서 민감."""
    a, p = np.asarray(actual, dtype=float), np.asarray(pred, dtype=float)
    mask = a > 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((a[mask] - p[mask]) / a[mask])) * 100)


def smape(actual: np.ndarray, pred: np.ndarray) -> float:
    """SMAPE (대칭 MAPE) — 0~200% 유한 범위.
    분모: (|actual| + |pred|) / 2.  분모 0 처리: 그 시점 0 기여.
    """
    a, p = np.asarray(actual, dtype=float), np.asarray(pred, dtype=float)
    denom = (np.abs(a) + np.abs(p)) / 2.0
    safe = denom > 0
    if safe.sum() == 0:
        return float("nan")
    err = np.zeros_like(a)
    err[safe] = np.abs(a[safe] - p[safe]) / denom[safe]
    return float(np.mean(err) * 100)


def wape(actual: np.ndarray, pred: np.ndarray) -> float:
    """WAPE — sum |error| / sum actual. 셀러 시계열 전체 매출 대비 절대오차 비율.
    분모: sum(actual). actual 합 0이면 NaN.
    """
    a, p = np.asarray(actual, dtype=float), np.asarray(pred, dtype=float)
    s = np.abs(a).sum()
    if s == 0:
        return float("nan")
    return float(np.abs(a - p).sum() / s * 100)


# === 데이터 로드 + 샘플링 ===
def load_cohort() -> pd.DataFrame:
    df = pd.read_parquet(DATA / "cohort_kr_v2.parquet")
    df["date"] = pd.to_datetime(df["date"])
    return df


def sample_sellers(df: pd.DataFrame, n_per_type: int) -> list[str]:
    rng = np.random.default_rng(SEED)
    seller_types = df.groupby("seller_id")["type"].first()
    sids = []
    for typ, group in seller_types.groupby(seller_types):
        ids = group.index.tolist()
        k = min(n_per_type, len(ids))
        sids.extend(rng.choice(ids, size=k, replace=False).tolist())
    return sids


# === Prophet fit/predict (음수 차단 추가) ===
def fit_predict_seller(seller_df: pd.DataFrame) -> dict | None:
    from prophet import Prophet

    s = seller_df.sort_values("date").reset_index(drop=True)
    if len(s) != 24:
        return None
    train = s.iloc[:TRAIN_MONTHS].copy()
    val = s.iloc[TRAIN_MONTHS:TRAIN_MONTHS + VAL_MONTHS].copy()
    test = s.iloc[TRAIN_MONTHS + VAL_MONTHS:].copy()

    if (train["monthly_revenue"] > 0).sum() < 6:
        return None

    train_df = pd.DataFrame({"ds": train["date"], "y": train["monthly_revenue"]})

    try:
        m = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode="additive",
            interval_width=0.8,
        )
        m.add_seasonality(name="monthly", period=30.5, fourier_order=3)
        m.fit(train_df)
    except Exception as e:
        return {"error": str(e)}

    future = m.make_future_dataframe(periods=VAL_MONTHS + TEST_MONTHS, freq="MS")
    fcst = m.predict(future)
    fcst = fcst[["ds", "yhat", "yhat_lower", "yhat_upper"]].set_index("ds")

    # ★ 음수 예측 차단 (매출은 ≥ 0)
    fcst["yhat"] = fcst["yhat"].clip(lower=0)
    fcst["yhat_lower"] = fcst["yhat_lower"].clip(lower=0)
    fcst["yhat_upper"] = fcst["yhat_upper"].clip(lower=0)

    val_pred = fcst.loc[val["date"].values, "yhat"].values
    test_pred = fcst.loc[test["date"].values, "yhat"].values
    val_actual = val["monthly_revenue"].values
    test_actual = test["monthly_revenue"].values

    return {
        # MAPE
        "mape_val": mape(val_actual, val_pred),
        "mape_test": mape(test_actual, test_pred),
        # SMAPE
        "smape_val": smape(val_actual, val_pred),
        "smape_test": smape(test_actual, test_pred),
        # WAPE
        "wape_val": wape(val_actual, val_pred),
        "wape_test": wape(test_actual, test_pred),
        # 시각화용
        "train_dates": train["date"].tolist(),
        "train_actual": train["monthly_revenue"].tolist(),
        "val_dates": val["date"].tolist(),
        "val_actual": val_actual.tolist(),
        "val_pred": val_pred.tolist(),
        "test_dates": test["date"].tolist(),
        "test_actual": test_actual.tolist(),
        "test_pred": test_pred.tolist(),
        "yhat_lower_test": fcst.loc[test["date"].values, "yhat_lower"].values.tolist(),
        "yhat_upper_test": fcst.loc[test["date"].values, "yhat_upper"].values.tolist(),
    }


def main():
    print("[1/4] Cohort v2 로드")
    df = load_cohort()
    print(f"  {df['seller_id'].nunique()} sellers, {len(df)} rows")

    print(f"\n[2/4] 셀러 샘플링 (유형별 {SAMPLES_PER_TYPE}명)")
    sids = sample_sellers(df, SAMPLES_PER_TYPE)
    print(f"  샘플 셀러: {len(sids)}")

    print(f"\n[3/4] Prophet 학습 + 음수 차단 + Robust 지표 ({len(sids)}개)")
    results = []
    detailed = {}
    for i, sid in enumerate(sids):
        if i % 25 == 0 and i > 0:
            print(f"  [{i}/{len(sids)}]")
        seller_df = df[df["seller_id"] == sid]
        typ = seller_df["type"].iloc[0]
        out = fit_predict_seller(seller_df)
        if out is None or "error" in out:
            results.append(dict(seller_id=sid, type=typ, status="skipped",
                                mape_val=np.nan, mape_test=np.nan,
                                smape_val=np.nan, smape_test=np.nan,
                                wape_val=np.nan, wape_test=np.nan))
            continue
        results.append(dict(seller_id=sid, type=typ, status="ok",
                            mape_val=out["mape_val"], mape_test=out["mape_test"],
                            smape_val=out["smape_val"], smape_test=out["smape_test"],
                            wape_val=out["wape_val"], wape_test=out["wape_test"]))
        if i < 24:
            detailed[sid] = {**out, "type": typ}

    res_df = pd.DataFrame(results)
    n_ok = (res_df["status"] == "ok").sum()
    print(f"  완료. ok={n_ok}, skipped={(res_df['status']=='skipped').sum()}")

    print("\n[4/4] 결과 분석 + 저장")
    res_df.to_csv(DATA / "prophet_baseline_v2_results.csv", index=False)
    ok = res_df[res_df["status"] == "ok"]

    summary = {
        "config": {
            "n_samples_per_type": SAMPLES_PER_TYPE,
            "train_months": TRAIN_MONTHS,
            "val_months": VAL_MONTHS,
            "test_months": TEST_MONTHS,
            "exogenous_vars": [],
            "post_process_clip_negative": True,
        },
        "overall": {
            "n_total": int(len(res_df)),
            "n_ok": int(n_ok),
            # Test set 기준
            "mape_test_mean": float(ok["mape_test"].mean()),
            "mape_test_median": float(ok["mape_test"].median()),
            "smape_test_mean": float(ok["smape_test"].mean()),
            "smape_test_median": float(ok["smape_test"].median()),
            "wape_test_mean": float(ok["wape_test"].mean()),
            "wape_test_median": float(ok["wape_test"].median()),
            # 목표 달성률
            "mape_test_pct_under_20": float((ok["mape_test"] < 20).mean() * 100),
            "smape_test_pct_under_20": float((ok["smape_test"] < 20).mean() * 100),
            "wape_test_pct_under_20": float((ok["wape_test"] < 20).mean() * 100),
        },
        "by_type": {},
    }
    for typ, g in ok.groupby("type"):
        summary["by_type"][typ] = {
            "n": int(len(g)),
            "mape_test_median": float(g["mape_test"].median()),
            "smape_test_median": float(g["smape_test"].median()),
            "wape_test_median": float(g["wape_test"].median()),
            "wape_test_pct_under_20": float((g["wape_test"] < 20).mean() * 100),
        }

    (DATA / "prophet_baseline_v2_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"\n=== Prophet Baseline v2 결과 ===")
    print(f"  학습 성공: {n_ok}/{len(res_df)}")
    print(f"\n  [test set 평균]")
    print(f"   MAPE:  mean={summary['overall']['mape_test_mean']:.1f}%  median={summary['overall']['mape_test_median']:.1f}%")
    print(f"   SMAPE: mean={summary['overall']['smape_test_mean']:.1f}%  median={summary['overall']['smape_test_median']:.1f}%")
    print(f"   WAPE:  mean={summary['overall']['wape_test_mean']:.1f}%  median={summary['overall']['wape_test_median']:.1f}%")
    print(f"\n  [< 20% 셀러 비율]")
    print(f"   MAPE:  {summary['overall']['mape_test_pct_under_20']:.1f}%")
    print(f"   SMAPE: {summary['overall']['smape_test_pct_under_20']:.1f}%")
    print(f"   WAPE:  {summary['overall']['wape_test_pct_under_20']:.1f}%")
    print(f"\n  [유형별 WAPE test median]")
    for typ, s in summary["by_type"].items():
        print(f"    {typ:10s}: WAPE={s['wape_test_median']:6.1f}%  SMAPE={s['smape_test_median']:5.1f}%  MAPE={s['mape_test_median']:5.1f}%  (n={s['n']})")

    # === 시각화 ===
    color_map = {"stable": "steelblue", "growth": "mediumseagreen",
                 "volatile": "crimson", "seasonal": "darkorange",
                 "decline": "gray", "other": "lightgray"}

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # (1) 3 지표 분포 비교 (test, clip 200)
    ax = axes[0, 0]
    bins = np.linspace(0, 200, 41)
    ax.hist(ok["mape_test"].clip(upper=200), bins=bins, alpha=0.5, label="MAPE", color="crimson")
    ax.hist(ok["smape_test"].clip(upper=200), bins=bins, alpha=0.5, label="SMAPE", color="steelblue")
    ax.hist(ok["wape_test"].clip(upper=200), bins=bins, alpha=0.5, label="WAPE", color="mediumseagreen")
    ax.axvline(20, color="black", linestyle="--", label="목표 20%")
    ax.set_xlabel("error % (clipped at 200)")
    ax.set_ylabel("셀러 수")
    ax.set_title("3가지 지표 분포 비교 (test)")
    ax.legend()
    ax.grid(alpha=0.3)

    # (2) 유형별 WAPE (가장 robust)
    ax = axes[0, 1]
    types_ordered = sorted(ok["type"].unique())
    data_lst = [ok[ok["type"] == t]["wape_test"].clip(upper=200).values for t in types_ordered]
    bp = ax.boxplot(data_lst, labels=types_ordered, patch_artist=True)
    for patch, t in zip(bp["boxes"], types_ordered):
        patch.set_facecolor(color_map.get(t, "gray"))
        patch.set_alpha(0.7)
    ax.axhline(20, color="red", linestyle="--", alpha=0.5, label="목표 20%")
    ax.set_ylabel("WAPE (test) %")
    ax.set_title("유형별 WAPE 분포 (robust 지표)")
    ax.tick_params(axis="x", rotation=15)
    ax.legend()
    ax.grid(alpha=0.3)

    # (3) 유형별 SMAPE
    ax = axes[0, 2]
    data_lst = [ok[ok["type"] == t]["smape_test"].clip(upper=200).values for t in types_ordered]
    bp = ax.boxplot(data_lst, labels=types_ordered, patch_artist=True)
    for patch, t in zip(bp["boxes"], types_ordered):
        patch.set_facecolor(color_map.get(t, "gray"))
        patch.set_alpha(0.7)
    ax.axhline(20, color="red", linestyle="--", alpha=0.5, label="목표 20%")
    ax.set_ylabel("SMAPE (test) %")
    ax.set_title("유형별 SMAPE 분포 (대칭, 0~200%)")
    ax.tick_params(axis="x", rotation=15)
    ax.legend()
    ax.grid(alpha=0.3)

    # (4-6) 샘플 셀러 시계열 3개
    sample_ids = list(detailed.keys())[:3]
    for i, sid in enumerate(sample_ids):
        d = detailed[sid]
        ax = axes[1, i]
        ax.plot(d["train_dates"], d["train_actual"], "o-", color="steelblue", label="train (실제)")
        ax.plot(d["val_dates"], d["val_actual"], "o-", color="mediumseagreen", label="val (실제)")
        ax.plot(d["test_dates"], d["test_actual"], "o-", color="darkorange", label="test (실제)")
        ax.plot(d["val_dates"], d["val_pred"], "x--", color="mediumseagreen", alpha=0.7, label="val (예측)")
        ax.plot(d["test_dates"], d["test_pred"], "x--", color="darkorange", alpha=0.7, label="test (예측)")
        ax.fill_between(d["test_dates"], d["yhat_lower_test"], d["yhat_upper_test"],
                         color="darkorange", alpha=0.15)
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.set_title(f"{sid[:20]}... [{d['type']}]\n"
                     f"WAPE={d['wape_test']:.1f}%  SMAPE={d['smape_test']:.1f}%  MAPE={d['mape_test']:.1f}%",
                     fontsize=9)
        ax.tick_params(axis="x", rotation=30)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    plt.suptitle(f"Prophet Baseline v2 (n={n_ok}, WAPE 평균 {summary['overall']['wape_test_mean']:.1f}%)",
                 fontsize=13, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(DATA / "prophet_baseline_v2_diagnostics.png", dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\n[save] prophet_baseline_v2_diagnostics.png")
    print("\n=== 완료 ===")


if __name__ == "__main__":
    main()
