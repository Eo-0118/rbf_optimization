"""Phase 2: Prophet 베이스라인 시계열 예측

목적: 합성 코호트 v2 셀러별 매출 시계열 예측 베이스라인 구축
방법: 셀러별 Prophet 모델 학습 → 검증/테스트 MAPE 측정

분할: 24개월 → 학습 18 / 검증 3 / 테스트 3
스코프 (1차): 유형별 sample (각 50명, stable·growth·volatile·seasonal·decline·other)
외생 변수: 없음 (1차 베이스라인, 매출만)

산출:
  Data/prophet_baseline_v1_results.csv  (셀러별 MAPE, 유형, n_active 등)
  Data/prophet_baseline_v1_summary.json (유형별 평균 MAPE 요약)
  Data/prophet_baseline_v1_samples.png  (예측 vs 실제 시각화)
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

# === Config ===
SEED = 42
TRAIN_MONTHS = 18
VAL_MONTHS = 3
TEST_MONTHS = 3
SAMPLES_PER_TYPE = 50  # 1차 베이스라인 — 유형별 50명


def load_cohort() -> pd.DataFrame:
    df = pd.read_parquet(DATA / "cohort_kr_v2.parquet")
    df["date"] = pd.to_datetime(df["date"])
    return df


def sample_sellers(df: pd.DataFrame, n_per_type: int) -> list[str]:
    """유형별 n명 균등 샘플 (rng로 재현 가능)."""
    rng = np.random.default_rng(SEED)
    seller_types = df.groupby("seller_id")["type"].first()
    sids = []
    for typ, group in seller_types.groupby(seller_types):
        ids = group.index.tolist()
        k = min(n_per_type, len(ids))
        sids.extend(rng.choice(ids, size=k, replace=False).tolist())
    return sids


def fit_predict_seller(seller_df: pd.DataFrame) -> dict | None:
    """단일 셀러 Prophet fit/predict.
    Prophet은 ds(datetime), y(value) 컬럼 필요.
    분할: 18(train) / 3(val) / 3(test)
    Return: dict with mape_val, mape_test, predictions
    """
    from prophet import Prophet

    s = seller_df.sort_values("date").reset_index(drop=True)
    if len(s) != 24:
        return None

    train = s.iloc[:TRAIN_MONTHS].copy()
    val = s.iloc[TRAIN_MONTHS:TRAIN_MONTHS + VAL_MONTHS].copy()
    test = s.iloc[TRAIN_MONTHS + VAL_MONTHS:].copy()

    if (train["monthly_revenue"] > 0).sum() < 6:
        return None  # 학습 데이터 활성 월 6 미만은 skip

    train_df = pd.DataFrame({"ds": train["date"], "y": train["monthly_revenue"]})

    try:
        m = Prophet(
            yearly_seasonality=False,  # 18개월 데이터로 yearly 못 잡음
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

    # MAPE (값 0인 월 제외 — MAPE 정의상 분모가 0이면 정의 불가)
    def mape(actual, pred):
        a, p = np.array(actual), np.array(pred)
        mask = a > 0
        if mask.sum() == 0:
            return np.nan
        return float(np.mean(np.abs((a[mask] - p[mask]) / a[mask])) * 100)

    val_pred = fcst.loc[val["date"].values, "yhat"].values
    test_pred = fcst.loc[test["date"].values, "yhat"].values
    mape_val = mape(val["monthly_revenue"].values, val_pred)
    mape_test = mape(test["monthly_revenue"].values, test_pred)

    return {
        "mape_val": mape_val,
        "mape_test": mape_test,
        "train_dates": train["date"].tolist(),
        "train_actual": train["monthly_revenue"].tolist(),
        "val_dates": val["date"].tolist(),
        "val_actual": val["monthly_revenue"].tolist(),
        "val_pred": val_pred.tolist(),
        "test_dates": test["date"].tolist(),
        "test_actual": test["monthly_revenue"].tolist(),
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

    print(f"\n[3/4] Prophet 학습 ({len(sids)}개 모델)")
    results = []
    detailed = {}
    for i, sid in enumerate(sids):
        if i % 25 == 0 and i > 0:
            print(f"  [{i}/{len(sids)}]")
        seller_df = df[df["seller_id"] == sid]
        typ = seller_df["type"].iloc[0]
        out = fit_predict_seller(seller_df)
        if out is None or "error" in out:
            results.append(dict(seller_id=sid, type=typ,
                                mape_val=np.nan, mape_test=np.nan, status="skipped"))
            continue
        results.append(dict(seller_id=sid, type=typ,
                            mape_val=out["mape_val"], mape_test=out["mape_test"],
                            status="ok"))
        if i < 24:  # 처음 24개만 상세 저장 (시각화용)
            detailed[sid] = {**out, "type": typ}

    res_df = pd.DataFrame(results)
    print(f"  완료. ok={(res_df['status']=='ok').sum()}, skipped={(res_df['status']=='skipped').sum()}")

    print("\n[4/4] 결과 분석 + 저장")
    res_df.to_csv(DATA / "prophet_baseline_v1_results.csv", index=False)

    ok = res_df[res_df["status"] == "ok"]
    summary = {
        "config": {
            "n_samples_per_type": SAMPLES_PER_TYPE,
            "train_months": TRAIN_MONTHS,
            "val_months": VAL_MONTHS,
            "test_months": TEST_MONTHS,
            "exogenous_vars": [],
        },
        "overall": {
            "n_total": int(len(res_df)),
            "n_ok": int(len(ok)),
            "n_skipped": int((res_df["status"] == "skipped").sum()),
            "mape_val_mean": float(ok["mape_val"].mean()),
            "mape_val_median": float(ok["mape_val"].median()),
            "mape_test_mean": float(ok["mape_test"].mean()),
            "mape_test_median": float(ok["mape_test"].median()),
            "mape_test_pct_under_20": float((ok["mape_test"] < 20).mean() * 100),
        },
        "by_type": {},
    }
    for typ, g in ok.groupby("type"):
        summary["by_type"][typ] = {
            "n": int(len(g)),
            "mape_val_mean": float(g["mape_val"].mean()),
            "mape_val_median": float(g["mape_val"].median()),
            "mape_test_mean": float(g["mape_test"].mean()),
            "mape_test_median": float(g["mape_test"].median()),
            "mape_test_pct_under_20": float((g["mape_test"] < 20).mean() * 100),
        }

    (DATA / "prophet_baseline_v1_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n=== 베이스라인 결과 ===")
    print(f"  학습 성공: {summary['overall']['n_ok']}/{summary['overall']['n_total']}")
    print(f"  MAPE val 평균: {summary['overall']['mape_val_mean']:.1f}%")
    print(f"  MAPE test 평균: {summary['overall']['mape_test_mean']:.1f}%")
    print(f"  MAPE test 중앙값: {summary['overall']['mape_test_median']:.1f}%")
    print(f"  MAPE test < 20% 셀러 비율: {summary['overall']['mape_test_pct_under_20']:.1f}%")
    print(f"\n  유형별 MAPE test 평균:")
    for typ, s in summary["by_type"].items():
        print(f"    {typ:10s}: {s['mape_test_mean']:6.1f}%  (median {s['mape_test_median']:.1f}%, n={s['n']})")

    # === 시각화 ===
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))

    # (1) MAPE distribution histogram
    ax = axes[0, 0]
    ax.hist(ok["mape_test"].clip(upper=200), bins=40, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(20, color="red", linestyle="--", label="목표 20%")
    ax.axvline(ok["mape_test"].median(), color="green", linestyle="--",
               label=f"중앙값 {ok['mape_test'].median():.1f}%")
    ax.set_xlabel("MAPE (test) %")
    ax.set_ylabel("셀러 수")
    ax.set_title("Prophet 베이스라인 — MAPE 분포 (test)")
    ax.legend()
    ax.grid(alpha=0.3)

    # (2) MAPE by type (boxplot)
    ax = axes[0, 1]
    types_ordered = sorted(ok["type"].unique())
    data_lst = [ok[ok["type"] == t]["mape_test"].clip(upper=200).values for t in types_ordered]
    bp = ax.boxplot(data_lst, labels=types_ordered, patch_artist=True)
    color_map = {"stable": "steelblue", "growth": "mediumseagreen",
                 "volatile": "crimson", "seasonal": "darkorange",
                 "decline": "gray", "other": "lightgray"}
    for patch, t in zip(bp["boxes"], types_ordered):
        patch.set_facecolor(color_map.get(t, "gray"))
        patch.set_alpha(0.7)
    ax.axhline(20, color="red", linestyle="--", alpha=0.5, label="목표 20%")
    ax.set_ylabel("MAPE (test) %")
    ax.set_title("유형별 MAPE 분포")
    ax.tick_params(axis="x", rotation=15)
    ax.legend()
    ax.grid(alpha=0.3)

    # (3-4) Sample seller forecasts (4개)
    sample_ids = list(detailed.keys())[:4]
    for i, sid in enumerate(sample_ids[:4]):
        ax = axes[1, i // 2] if i < 2 else axes[1, 1]
        if i >= 2:
            continue  # axes는 4개만, but we draw 2 panels with 2 sellers each
    # 다시 — 명확히 2개 셀러만 plot
    for i, sid in enumerate(sample_ids[:2]):
        d = detailed[sid]
        ax = axes[1, i]
        ax.plot(d["train_dates"], d["train_actual"], "o-", color="steelblue", label="train (실제)")
        ax.plot(d["val_dates"], d["val_actual"], "o-", color="mediumseagreen", label="val (실제)")
        ax.plot(d["test_dates"], d["test_actual"], "o-", color="darkorange", label="test (실제)")
        ax.plot(d["val_dates"], d["val_pred"], "x--", color="mediumseagreen", alpha=0.7, label="val (예측)")
        ax.plot(d["test_dates"], d["test_pred"], "x--", color="darkorange", alpha=0.7, label="test (예측)")
        ax.fill_between(d["test_dates"], d["yhat_lower_test"], d["yhat_upper_test"],
                         color="darkorange", alpha=0.15, label="80% interval")
        ax.set_title(f"{sid[:24]}... [{d['type']}]\nMAPE val={d['mape_val']:.1f}%, test={d['mape_test']:.1f}%",
                     fontsize=10)
        ax.tick_params(axis="x", rotation=30)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    plt.suptitle(f"Prophet Baseline v1 (n={summary['overall']['n_ok']}, "
                 f"test MAPE 평균 {summary['overall']['mape_test_mean']:.1f}%)",
                 fontsize=13, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(DATA / "prophet_baseline_v1_diagnostics.png", dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\n[save] prophet_baseline_v1_diagnostics.png")
    print("\n=== 완료 ===")


if __name__ == "__main__":
    main()
