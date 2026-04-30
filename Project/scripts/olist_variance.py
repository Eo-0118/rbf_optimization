"""Olist 분산 구조 추출 — 유형별 σ, AR(1), 노이즈 분포, 이상치 빈도.

v4 D4: Olist에서 판매자 단위 분산·노이즈 구조를 학습해
       합성 코호트 생성기(D5)의 파라미터로 사용한다.
       트렌드·계절성은 제거 (한국 KOSIS에서 별도 주입).

입력:
    Data/Olist_Data/monthly_seller_with_type.csv
    Data/Olist_Data/seller_features_with_type.csv

출력:
    data/variance_params.json     유형별 통계 파라미터
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import numpy as np
from scipy import stats as sp_stats

ROOT = Path(__file__).resolve().parents[1]
MONTHLY = ROOT / "Data" / "Olist_Data" / "monthly_seller_with_type.csv"
FEATURES = ROOT / "Data" / "Olist_Data" / "seller_features_with_type.csv"
OUT = ROOT / "data"
OUT.mkdir(exist_ok=True)

TYPE_MAP = {
    0: "volatile",
    1: "seasonal",
    2: "stable",
    3: "growth",
}

BETA_PRIOR = {
    "stable":   {"alpha": 5, "beta": 15, "mean": 0.25},
    "seasonal": {"alpha": 4, "beta": 16, "mean": 0.20},
    "growth":   {"alpha": 3, "beta": 22, "mean": 0.12},
    "volatile": {"alpha": 2, "beta": 18, "mean": 0.10},
}


def load():
    m = pd.read_csv(MONTHLY)
    m["date"] = pd.to_datetime(m["year_month_dt"])
    m = m.sort_values(["seller_id", "date"])
    m["type"] = m["cluster"].map(TYPE_MAP)
    f = pd.read_csv(FEATURES)
    f["type"] = f["cluster"].map(TYPE_MAP)
    return m, f


def seller_stats(grp: pd.DataFrame) -> dict:
    rev = grp["monthly_revenue"].values
    n = len(rev)
    mean = rev.mean()
    std = rev.std(ddof=1) if n > 1 else 0.0
    cv = std / mean if mean > 0 else 0.0

    # AR(1): autocorrelation at lag 1 (on normalized series)
    if n >= 4 and std > 0:
        normed = (rev - mean) / std
        ar1 = np.corrcoef(normed[:-1], normed[1:])[0, 1]
    else:
        ar1 = 0.0

    # spike: months > mean + 3*std
    if std > 0:
        spike_count = int(np.sum(np.abs(rev - mean) > 3 * std))
        spike_ratio = spike_count / n
    else:
        spike_ratio = 0.0

    # zero months ratio (from full timeline, not just active)
    zero_ratio = float(grp.get("active_days", pd.Series()).eq(0).sum()) / max(n, 1)

    return {
        "n_months": int(n),
        "mean_rev": float(mean),
        "std_rev": float(std),
        "cv": float(cv),
        "ar1": float(ar1) if np.isfinite(ar1) else 0.0,
        "spike_ratio": float(spike_ratio),
        "zero_ratio": float(zero_ratio),
    }


def fit_noise_dist(residuals: np.ndarray) -> dict:
    """Fit normal and Student-t to residuals, return comparison."""
    if len(residuals) < 10:
        return {"best": "normal", "t_df": 30.0, "t_loc": 0.0, "t_scale": 1.0}
    res = residuals[np.isfinite(residuals)]
    if len(res) < 10:
        return {"best": "normal", "t_df": 30.0, "t_loc": 0.0, "t_scale": 1.0}
    t_params = sp_stats.t.fit(res)
    df, loc, scale = t_params
    _, p_normal = sp_stats.shapiro(res[:5000]) if len(res) <= 5000 else (0, 0)
    best = "student_t" if (df < 10 or p_normal < 0.05) else "normal"
    return {"best": best, "t_df": float(df), "t_loc": float(loc), "t_scale": float(scale)}


def main():
    m, f = load()
    print(f"[load] monthly: {len(m)} rows, {m['seller_id'].nunique()} sellers")
    print(f"[load] features: {len(f)} rows")

    results = {}

    for typ in ["stable", "seasonal", "growth", "volatile"]:
        sellers = m[m["type"] == typ]["seller_id"].unique()
        print(f"\n=== {typ} (n={len(sellers)}) ===")

        per_seller = []
        all_residuals = []

        for sid in sellers:
            grp = m[m["seller_id"] == sid]
            ss = seller_stats(grp)
            per_seller.append(ss)

            # Detrend: subtract rolling mean to get residuals for noise dist
            rev = grp["monthly_revenue"].values
            if len(rev) >= 3:
                rm = pd.Series(rev).rolling(3, min_periods=1, center=True).mean().values
                resid = (rev - rm)
                if rm.mean() > 0:
                    resid_normed = resid / rm.mean()
                    all_residuals.extend(resid_normed.tolist())
                else:
                    all_residuals.extend(resid.tolist())

        pdf = pd.DataFrame(per_seller)

        # Aggregate stats
        agg = {
            "n_sellers": int(len(sellers)),
            "mean_cv": float(pdf["cv"].mean()),
            "median_cv": float(pdf["cv"].median()),
            "std_cv": float(pdf["cv"].std()),
            "mean_ar1": float(pdf["ar1"].mean()),
            "median_ar1": float(pdf["ar1"].median()),
            "mean_spike_ratio": float(pdf["spike_ratio"].mean()),
            "mean_zero_ratio": float(pdf["zero_ratio"].mean()),
            "mean_rev_mean": float(pdf["mean_rev"].mean()),
            "mean_rev_std": float(pdf["std_rev"].mean()),
            "cv_percentiles": {
                "p10": float(pdf["cv"].quantile(0.1)),
                "p25": float(pdf["cv"].quantile(0.25)),
                "p50": float(pdf["cv"].quantile(0.5)),
                "p75": float(pdf["cv"].quantile(0.75)),
                "p90": float(pdf["cv"].quantile(0.9)),
            },
            "ar1_percentiles": {
                "p10": float(pdf["ar1"].quantile(0.1)),
                "p50": float(pdf["ar1"].quantile(0.5)),
                "p90": float(pdf["ar1"].quantile(0.9)),
            },
        }

        # Noise distribution
        res_arr = np.array(all_residuals)
        noise = fit_noise_dist(res_arr)
        agg["noise_dist"] = noise

        # Beta prior (from v4 plan)
        agg["m_i_beta"] = BETA_PRIOR[typ]

        results[typ] = agg
        print(f"  cv: mean={agg['mean_cv']:.3f} median={agg['median_cv']:.3f}")
        print(f"  ar1: mean={agg['mean_ar1']:.3f} median={agg['median_ar1']:.3f}")
        print(f"  spike_ratio: {agg['mean_spike_ratio']:.4f}")
        print(f"  noise: best={noise['best']}, t_df={noise['t_df']:.1f}")

    # Seasonality strength from features
    for typ in results:
        feat_sub = f[f["type"] == typ]
        if "seasonality_strength" in feat_sub.columns:
            ss = feat_sub["seasonality_strength"]
            results[typ]["seasonality_strength"] = {
                "mean": float(ss.mean()),
                "std": float(ss.std()),
                "p25": float(ss.quantile(0.25)),
                "p75": float(ss.quantile(0.75)),
            }

    out_path = OUT / "variance_params.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n[save] {out_path}")

    # Summary table
    print("\n=== SUMMARY ===")
    rows = []
    for typ, d in results.items():
        rows.append({
            "type": typ,
            "n": d["n_sellers"],
            "mean_cv": d["mean_cv"],
            "mean_ar1": d["mean_ar1"],
            "noise": d["noise_dist"]["best"],
            "t_df": d["noise_dist"]["t_df"],
            "spike%": d["mean_spike_ratio"] * 100,
            "m_i_mean": d["m_i_beta"]["mean"],
        })
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
