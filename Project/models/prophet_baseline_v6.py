"""Phase 2: Prophet 베이스라인 v6 — KOSIS 카테고리별 + Naver 카테고리별

v3/v4/v5 진단 결과:
  - v4 (단순 카테고리 Naver 매칭): WAPE<20% 12.9% (v3 14.1%보다 악화)
  - 원인: 합성 매출-Naver 상관 0.002 (합성 process가 외생변수를 약하게 반영)
  - KOSIS-Naver 실측 |r|=0.226 → 합성도 이 수준 맞춤 (v5: 0.227)
  - v6: KOSIS도 카테고리별 → 두 외생변수 모두 강하게 매출에 반영
       (셀러별 r: KOSIS 0.27, Naver 0.23)

v6 변경:
  cohort_kr_v6 사용 (KOSIS 카테고리별 + Naver 카테고리별)

비교 가능성:
  - SAMPLES_PER_TYPE=50, train/val/test 동일, SEED 동일
  - 외생변수 컬럼명 동일 (naver_index, promo)
  - → 차이는 합성 process 자체 (외생변수가 실제로 매출에 의미 있게 들어감)

산출:
  Data/prophet_baseline_v6_results.csv
  Data/prophet_baseline_v6_summary.json
  Data/prophet_baseline_v6_diagnostics.png
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

EXOG_VARS = ["naver_index", "promo"]


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
    df = pd.read_parquet(DATA / "cohort_kr_v6.parquet")
    df["date"] = pd.to_datetime(df["date"])
    # v6는 추가로 kosis_trend 컬럼 있음 (참고용, Prophet에는 안 씀 — leakage 우려)
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


def fit_predict_seller_with_exog(seller_df):
    from prophet import Prophet

    s = seller_df.sort_values("date").reset_index(drop=True)
    if len(s) != 24:
        return None
    train = s.iloc[:TRAIN_MONTHS].copy()
    val = s.iloc[TRAIN_MONTHS:TRAIN_MONTHS + VAL_MONTHS].copy()
    test = s.iloc[TRAIN_MONTHS + VAL_MONTHS:].copy()

    if (train["monthly_revenue"] > 0).sum() < 6:
        return None

    train_df = pd.DataFrame({
        "ds": train["date"].values,
        "y": train["monthly_revenue"].values,
        **{v: train[v].values for v in EXOG_VARS},
    })

    try:
        m = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode="additive",
            interval_width=0.8,
        )
        m.add_seasonality(name="monthly", period=30.5, fourier_order=3)
        for v in EXOG_VARS:
            m.add_regressor(v)
        m.fit(train_df)
    except Exception as e:
        return {"error": str(e)}

    full_df = pd.DataFrame({
        "ds": s["date"].values,
        **{v: s[v].values for v in EXOG_VARS},
    })

    fcst = m.predict(full_df)
    fcst = fcst[["ds", "yhat", "yhat_lower", "yhat_upper"]].set_index("ds")
    fcst["yhat"] = fcst["yhat"].clip(lower=0)
    fcst["yhat_lower"] = fcst["yhat_lower"].clip(lower=0)
    fcst["yhat_upper"] = fcst["yhat_upper"].clip(lower=0)

    val_pred = fcst.loc[val["date"].values, "yhat"].values
    test_pred = fcst.loc[test["date"].values, "yhat"].values
    val_actual = val["monthly_revenue"].values
    test_actual = test["monthly_revenue"].values

    return {
        "mape_val": mape(val_actual, val_pred),
        "mape_test": mape(test_actual, test_pred),
        "smape_val": smape(val_actual, val_pred),
        "smape_test": smape(test_actual, test_pred),
        "wape_val": wape(val_actual, val_pred),
        "wape_test": wape(test_actual, test_pred),
        "kosis_category": seller_df["kosis_category"].iloc[0],
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
    print("[1/5] Cohort v6 로드 (카테고리별 Naver 매칭)")
    df = load_cohort()
    print(f"  {df['seller_id'].nunique()} sellers, {len(df)} rows")

    same_month = df[df["month_idx"] == 0]
    print(f"  같은 month naver_index 셀러간 std: {same_month['naver_index'].std():.4f}")

    print(f"\n[2/5] 셀러 샘플링 (유형별 {SAMPLES_PER_TYPE}명, SEED={SEED})")
    sids = sample_sellers(df, SAMPLES_PER_TYPE)
    print(f"  샘플 셀러: {len(sids)}")

    print(f"\n[3/5] Prophet 학습 ({len(sids)}개 모델)")
    results = []
    detailed = {}
    for i, sid in enumerate(sids):
        if i % 25 == 0 and i > 0:
            print(f"  [{i}/{len(sids)}]")
        seller_df = df[df["seller_id"] == sid]
        typ = seller_df["type"].iloc[0]
        kosis_cat = seller_df["kosis_category"].iloc[0]
        out = fit_predict_seller_with_exog(seller_df)
        if out is None or "error" in out:
            results.append(dict(seller_id=sid, type=typ, kosis_category=kosis_cat,
                                status="skipped",
                                mape_val=np.nan, mape_test=np.nan,
                                smape_val=np.nan, smape_test=np.nan,
                                wape_val=np.nan, wape_test=np.nan))
            continue
        results.append(dict(seller_id=sid, type=typ, kosis_category=kosis_cat,
                            status="ok",
                            mape_val=out["mape_val"], mape_test=out["mape_test"],
                            smape_val=out["smape_val"], smape_test=out["smape_test"],
                            wape_val=out["wape_val"], wape_test=out["wape_test"]))
        if i < 24:
            detailed[sid] = {**out, "type": typ}

    res_df = pd.DataFrame(results)
    n_ok = (res_df["status"] == "ok").sum()
    print(f"  완료. ok={n_ok}, skipped={(res_df['status']=='skipped').sum()}")

    print("\n[4/5] 결과 분석 + 저장")
    res_df.to_csv(DATA / "prophet_baseline_v6_results.csv", index=False)
    ok = res_df[res_df["status"] == "ok"]

    summary = {
        "config": {
            "cohort": "cohort_kr_v6 (KOSIS 카테고리별 + Naver 카테고리별, β=0.4)",
            "n_samples_per_type": SAMPLES_PER_TYPE,
            "train_months": TRAIN_MONTHS,
            "val_months": VAL_MONTHS,
            "test_months": TEST_MONTHS,
            "exogenous_vars": EXOG_VARS,
            "post_process_clip_negative": True,
        },
        "overall": {
            "n_total": int(len(res_df)),
            "n_ok": int(n_ok),
            "mape_test_mean": float(ok["mape_test"].mean()),
            "mape_test_median": float(ok["mape_test"].median()),
            "smape_test_mean": float(ok["smape_test"].mean()),
            "smape_test_median": float(ok["smape_test"].median()),
            "wape_test_mean": float(ok["wape_test"].mean()),
            "wape_test_median": float(ok["wape_test"].median()),
            "mape_test_pct_under_20": float((ok["mape_test"] < 20).mean() * 100),
            "smape_test_pct_under_20": float((ok["smape_test"] < 20).mean() * 100),
            "wape_test_pct_under_20": float((ok["wape_test"] < 20).mean() * 100),
        },
        "by_type": {},
        "by_kosis_category": {},
    }
    for typ, g in ok.groupby("type"):
        summary["by_type"][typ] = {
            "n": int(len(g)),
            "mape_test_median": float(g["mape_test"].median()),
            "smape_test_median": float(g["smape_test"].median()),
            "wape_test_median": float(g["wape_test"].median()),
            "wape_test_pct_under_20": float((g["wape_test"] < 20).mean() * 100),
        }
    for cat, g in ok.groupby("kosis_category"):
        if len(g) < 3:
            continue
        summary["by_kosis_category"][cat] = {
            "n": int(len(g)),
            "wape_test_median": float(g["wape_test"].median()),
            "wape_test_pct_under_20": float((g["wape_test"] < 20).mean() * 100),
        }

    (DATA / "prophet_baseline_v6_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"\n=== Prophet Baseline v6 결과 (카테고리별 Naver) ===")
    print(f"  학습 성공: {n_ok}/{len(res_df)}")
    print(f"\n  [test set 평균]")
    print(f"   MAPE:  mean={summary['overall']['mape_test_mean']:.1f}%  median={summary['overall']['mape_test_median']:.1f}%")
    print(f"   SMAPE: mean={summary['overall']['smape_test_mean']:.1f}%  median={summary['overall']['smape_test_median']:.1f}%")
    print(f"   WAPE:  mean={summary['overall']['wape_test_mean']:.1f}%  median={summary['overall']['wape_test_median']:.1f}%")
    print(f"\n  [< 20% 셀러 비율]")
    print(f"   MAPE:  {summary['overall']['mape_test_pct_under_20']:.1f}%")
    print(f"   SMAPE: {summary['overall']['smape_test_pct_under_20']:.1f}%")
    print(f"   WAPE:  {summary['overall']['wape_test_pct_under_20']:.1f}%")

    # v2/v3/v4 vs v6 비교
    print(f"\n  [Prophet 버전별 비교 — WAPE<20% 셀러 비율]")
    print(f"   v2 (외생 X):           17.8%")
    print(f"   v3 (글로벌 Naver):     14.1%")
    print(f"   v4 (카테고리 Naver, v3코호트): 12.9%")
    print(f"   v6 (KOSIS+Naver 카테고리 generative): {summary['overall']['wape_test_pct_under_20']:.1f}%")
    v3_path = DATA / "prophet_baseline_v3_summary.json"
    if v3_path.exists():
        v3 = json.loads(v3_path.read_text())
        print(f"\n  [v3 → v6 변화]")
        print(f"   WAPE mean:    {v3['overall']['wape_test_mean']:.1f}% → {summary['overall']['wape_test_mean']:.1f}%  ({summary['overall']['wape_test_mean']-v3['overall']['wape_test_mean']:+.1f}%p)")
        print(f"   WAPE < 20%:   {v3['overall']['wape_test_pct_under_20']:.1f}% → {summary['overall']['wape_test_pct_under_20']:.1f}%  ({summary['overall']['wape_test_pct_under_20']-v3['overall']['wape_test_pct_under_20']:+.1f}%p)")

    print(f"\n  [유형별 WAPE test median]")
    for typ, s in summary["by_type"].items():
        print(f"    {typ:10s}: WAPE={s['wape_test_median']:6.1f}%  (<20%: {s['wape_test_pct_under_20']:5.1f}%, n={s['n']})")

    # === [5/5] 시각화 ===
    color_map = {"stable": "steelblue", "growth": "mediumseagreen",
                 "volatile": "crimson", "seasonal": "darkorange",
                 "decline": "gray", "other": "lightgray"}

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    ax = axes[0, 0]
    bins = np.linspace(0, 200, 41)
    ax.hist(ok["mape_test"].clip(upper=200), bins=bins, alpha=0.5, label="MAPE", color="crimson")
    ax.hist(ok["smape_test"].clip(upper=200), bins=bins, alpha=0.5, label="SMAPE", color="steelblue")
    ax.hist(ok["wape_test"].clip(upper=200), bins=bins, alpha=0.5, label="WAPE", color="mediumseagreen")
    ax.axvline(20, color="black", linestyle="--", label="목표 20%")
    ax.set_xlabel("error %"); ax.set_ylabel("셀러 수")
    ax.set_title("v6 지표 분포 (test)")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    types_ordered = sorted(ok["type"].unique())
    data_lst = [ok[ok["type"] == t]["wape_test"].clip(upper=200).values for t in types_ordered]
    bp = ax.boxplot(data_lst, labels=types_ordered, patch_artist=True)
    for patch, t in zip(bp["boxes"], types_ordered):
        patch.set_facecolor(color_map.get(t, "gray"))
        patch.set_alpha(0.7)
    ax.axhline(20, color="red", linestyle="--", alpha=0.5)
    ax.set_ylabel("WAPE (test) %"); ax.set_title("유형별 WAPE")
    ax.tick_params(axis="x", rotation=15); ax.grid(alpha=0.3)

    # 버전별 WAPE<20% 비교 (bar chart)
    ax = axes[0, 2]
    versions = ["v2\n(외생X)", "v3\n(글로벌)", "v4\n(cat match)", "v6\n(generative)"]
    v2_pct = 17.8
    v3_pct = 14.1
    v4_pct = 12.9
    v6_pct = summary["overall"]["wape_test_pct_under_20"]
    vals = [v2_pct, v3_pct, v4_pct, v6_pct]
    colors_b = ["gray", "lightblue", "orange", "mediumseagreen"]
    bars = ax.bar(versions, vals, color=colors_b, edgecolor="white")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.3, f"{v:.1f}%", ha="center", fontweight="bold")
    ax.axhline(20, color="red", linestyle="--", alpha=0.5, label="목표 20%")
    ax.set_ylabel("WAPE < 20% 셀러 비율 (%)")
    ax.set_title("Prophet 버전별 비교")
    ax.legend(); ax.grid(alpha=0.3, axis="y")

    # 샘플 시계열
    sample_ids = list(detailed.keys())[:3]
    for i, sid in enumerate(sample_ids):
        d = detailed[sid]
        ax = axes[1, i]
        ax.plot(d["train_dates"], d["train_actual"], "o-", color="steelblue", label="train")
        ax.plot(d["val_dates"], d["val_actual"], "o-", color="mediumseagreen", label="val")
        ax.plot(d["test_dates"], d["test_actual"], "o-", color="darkorange", label="test")
        ax.plot(d["val_dates"], d["val_pred"], "x--", color="mediumseagreen", alpha=0.7, label="val 예측")
        ax.plot(d["test_dates"], d["test_pred"], "x--", color="darkorange", alpha=0.7, label="test 예측")
        ax.fill_between(d["test_dates"], d["yhat_lower_test"], d["yhat_upper_test"],
                         color="darkorange", alpha=0.15)
        ax.set_title(f"{sid[:20]}... [{d['type']}/{d['kosis_category']}]\n"
                     f"WAPE={d['wape_test']:.1f}%", fontsize=9)
        ax.tick_params(axis="x", rotation=30); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    plt.suptitle(f"Prophet v6 (cohort_kr_v6, KOSIS+Naver 카테고리 generative, n={n_ok}, "
                 f"WAPE median {summary['overall']['wape_test_median']:.1f}%)",
                 fontsize=13, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(DATA / "prophet_baseline_v6_diagnostics.png", dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\n[save] prophet_baseline_v6_diagnostics.png")

    print("\n=== 완료 ===")


if __name__ == "__main__":
    main()
