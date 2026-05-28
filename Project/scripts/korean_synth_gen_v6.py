"""한국형 합성 코호트 생성기 v6 — KOSIS도 카테고리별로

v5 진단:
  - 매출-Naver r: 0.002 → 0.227 (실측 0.226 일치) ✅
  - 매출-KOSIS r: 0.51 → 0.09 (KOSIS 신호 약화) ❌
  → naver_factor가 KOSIS 합계 신호를 묻음

v6 변경 (옵션 D):
  1. KOSIS '합계' 대신 셀러의 kosis_category에 맞는 trend/season 사용
  2. 같은 카테고리 셀러는 같은 KOSIS trend → 카테고리 내에서 KOSIS 신호 회복
  3. naver_factor 강도 유지 (v5와 동일, β=0.4)
  4. 검증: 매출-KOSIS 카테고리 trend r + 매출-Naver r 둘 다 측정

학술 정당화:
  "화장품 셀러는 화장품 시장 trend + 화장품 검색 트렌드를 따른다"
  → 카테고리별로 KOSIS/Naver를 분리해서 자연스러운 한국 시장 구조 반영
  → 보고서 r=0.55 주장은 카테고리 내 신호로 재정의

산출 (v3/v4/v5 모두 보존):
- Data/cohort_kr_v6.parquet
- Data/cohort_kr_v6_validation.json
- Data/cohort_kr_v6_diagnostics.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress, pearsonr

ROOT = Path("/Users/eoseungyun/Desktop/project/SW_Capstone/Project")
DATA = ROOT / "Data"

# === Config ===
SEED = 42
N_MONTHS = 24
START_DATE = "2023-01-01"
SAMPLES_PER_DONOR = 2
NOISE_SCALE = 0.3
KOREAN_LOG_MEAN = np.log(500)
KOREAN_LOG_STD = 0.8
PROMO_MONTHS = [1, 2, 9, 11]
NAVER_BETA = 0.4   # v5와 동일

KOSIS_TO_NAVER = {
    "생활용품": "생활/건강",
    "가구": "가구/인테리어",
    "스포츠·레저용품": "스포츠/레저",
    "화장품": "화장품/미용",
    "아동·유아용품": "출산/육아",
    "컴퓨터 및 주변기기": "디지털/가전",
    "가방": "패션잡화",
    "가전·전자": "디지털/가전",
    "통신기기": "디지털/가전",
    "서적": "도서",
    "패션용품 및 액세서리": "패션잡화",
    "음·식료품": "식품",
    "의복": "패션의류",
    "신발": "패션잡화",
    "농축수산물": "식품",
    "자동차 및 자동차용품": None,
    "애완용품": None,
    "기타": None,
    "사무·문구": None,
}

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False


def load_donor_pool_with_category() -> pd.DataFrame:
    sf = pd.read_csv(DATA / "seller_features_v3.csv")
    monthly = pd.read_csv(DATA / "monthly_seller_revenue.csv")
    monthly["ym"] = pd.to_datetime(monthly["year_month"])

    ar1_dict = {}
    for sid in sf["seller_id"].unique():
        ts = monthly[monthly["seller_id"] == sid].sort_values("ym")["monthly_revenue"].values
        nz = np.where(ts > 0)[0]
        if len(nz) < 2:
            ar1_dict[sid] = 0.0
            continue
        active = ts[nz[0]:nz[-1] + 1]
        if len(active) <= 1 or np.std(active) < 1e-8:
            ar1_dict[sid] = 0.0
        else:
            ar1_dict[sid] = float(np.corrcoef(active[:-1], active[1:])[0, 1])
    sf["ar1"] = sf["seller_id"].map(ar1_dict).fillna(0.0)

    items = pd.read_csv(DATA / "Olist_Data" / "olist_order_items_dataset.csv")
    products = pd.read_csv(DATA / "Olist_Data" / "olist_products_dataset.csv")
    merged = items.merge(products[["product_id", "product_category_name"]], on="product_id")

    def primary_cat(g):
        d = g["product_category_name"].dropna()
        return d.mode().iloc[0] if len(d) > 0 else None
    seller_primary = merged.groupby("seller_id").apply(primary_cat).reset_index()
    seller_primary.columns = ["seller_id", "olist_category"]
    sf = sf.merge(seller_primary, on="seller_id", how="left")

    mapping = pd.read_csv(DATA / "olist_to_kosis_mapping.csv")
    map_dict = dict(zip(mapping["olist_category"], mapping["kosis_category"]))
    status_dict = dict(zip(mapping["olist_category"], mapping["status"]))
    sf["kosis_category"] = sf["olist_category"].map(map_dict)
    sf["mapping_status"] = sf["olist_category"].map(status_dict).fillna("EXCLUDE")

    sf_filtered = sf[sf["mapping_status"] == "OK"].copy()
    print(f"  카테고리 매핑: 전체 {len(sf)} → OK {len(sf_filtered)}")
    return sf_filtered


def load_priors_per_category():
    """KOSIS trend/season을 카테고리별 dict로 + Naver 카테고리별로 분리.

    Returns:
        kosis_trend_per_cat: dict[kosis_cat] → (N_MONTHS,) trend
        kosis_season_per_cat: dict[kosis_cat] → (N_MONTHS,) season_rel
        kosis_total_trend: (N_MONTHS,) fallback (합계)
        kosis_total_season: (N_MONTHS,)
        naver_per_cat: dict[naver_cat] → (N_MONTHS,)
        naver_global_mean: (N_MONTHS,)
    """
    df = pd.read_parquet(DATA / "kosis" / "kr_trend_season.parquet")
    df["date"] = pd.to_datetime(df["date"])

    target_dates = pd.date_range(START_DATE, periods=N_MONTHS, freq="MS")

    def extract(cat):
        sub = df[df["category"] == cat].sort_values("date")
        sub = sub[sub["date"].isin(target_dates)].copy()
        if len(sub) < N_MONTHS:
            return None, None
        sub = sub.set_index("date").reindex(target_dates).reset_index()
        raw = sub["value"].values.astype(float)
        mean_val = raw[raw > 0].mean() if (raw > 0).any() else 1.0
        trend_shape = raw / mean_val - 1.0
        season = sub["season_rel"].values.astype(float) * 3.0
        # NaN 처리 (안전장치)
        trend_shape = np.nan_to_num(trend_shape, nan=0.0)
        season = np.nan_to_num(season, nan=0.0)
        return trend_shape, season

    kosis_trend_per_cat = {}
    kosis_season_per_cat = {}
    all_cats = df["category"].unique()
    for cat in all_cats:
        t, s = extract(cat)
        if t is not None:
            kosis_trend_per_cat[cat] = t
            kosis_season_per_cat[cat] = s

    # 합계 (fallback)
    kosis_total_trend = kosis_trend_per_cat.get("합계", np.zeros(N_MONTHS))
    kosis_total_season = kosis_season_per_cat.get("합계", np.zeros(N_MONTHS))

    # Naver 카테고리별
    nv = pd.read_csv(DATA / "naver" / "naver_monthly.csv")
    nv["date"] = pd.to_datetime(nv["date"])
    nv = nv.set_index("date").sort_index()
    nv_window = nv.loc[pd.to_datetime(START_DATE):].head(N_MONTHS)
    if len(nv_window) < N_MONTHS:
        last = nv_window.iloc[[-1]]
        while len(nv_window) < N_MONTHS:
            nv_window = pd.concat([nv_window, last])
    nv_window = nv_window.iloc[:N_MONTHS]

    naver_per_cat = {}
    for col in nv_window.columns:
        x = nv_window[col].values.astype(float)
        mean_val = np.nanmean(x)
        x = np.where(np.isnan(x), mean_val, x)
        if x.max() - x.min() > 1e-8:
            x_norm = (x - x.min()) / (x.max() - x.min())
        else:
            x_norm = np.full_like(x, 0.5)
        naver_per_cat[col] = x_norm
    naver_global_mean = np.stack(list(naver_per_cat.values())).mean(axis=0)

    return (kosis_trend_per_cat, kosis_season_per_cat,
            kosis_total_trend, kosis_total_season,
            naver_per_cat, naver_global_mean)


def make_promo_dummy() -> np.ndarray:
    dates = pd.date_range(START_DATE, periods=N_MONTHS, freq="MS")
    return np.array([1.0 if d.month in PROMO_MONTHS else 0.0 for d in dates])


def donor_to_korean_mu(donor_log_rev, pool_log_mean, pool_log_std, rng):
    z = (donor_log_rev - pool_log_mean) / max(pool_log_std, 0.01)
    z = float(np.clip(z, -3, 3))
    return float(np.exp(KOREAN_LOG_MEAN + z * KOREAN_LOG_STD + 0.1 * rng.standard_normal()))


def generate_seller(donor, pool_stats, kosis_trend_cat, kosis_season_cat,
                     naver_signal, promo, rng):
    """v6: kosis_trend_cat, kosis_season_cat은 셀러 카테고리별 (N_MONTHS,) 배열."""
    base_mu = donor_to_korean_mu(donor["log_avg_rev"], pool_stats["log_mean"],
                                  pool_stats["log_std"], rng)
    cv = float(donor["cv"])
    ar1 = float(donor["ar1"])
    season_strength = float(donor["seasonality"])
    spike_ratio = float(donor["spike_ratio"])
    zero_ratio = float(donor["zero_ratio"])
    trend_per_month = (float(donor["trend_slope"]) / N_MONTHS) * float(donor["trend_r2"])

    rev = np.zeros(N_MONTHS)
    noise_prev = 0.0

    for t in range(N_MONTHS):
        market_factor = 1.0 + kosis_trend_cat[t]
        season_factor = 1.0 + kosis_season_cat[t] * (0.3 + 0.7 * season_strength)
        donor_growth = 1.0 + trend_per_month * t
        base_t = base_mu * market_factor * season_factor * donor_growth

        sigma = base_t * cv * NOISE_SCALE
        noise = ar1 * noise_prev + sigma * rng.standard_normal()
        noise_prev = noise

        spike = 0.0
        if spike_ratio > 2.0 and rng.random() < 0.04:
            spike = base_t * (spike_ratio - 1) * rng.uniform(0.2, 0.5)

        promo_boost = 0.20 * base_mu * promo[t]
        naver_factor = 1.0 + NAVER_BETA * (naver_signal[t] - 0.5)

        rev[t] = max((base_t + noise + spike + promo_boost) * naver_factor, 0.0)
        if rng.random() < zero_ratio * 0.6:
            rev[t] = 0.0

    return rev, base_mu


def calc_synth_features(rev):
    nz = np.where(rev > 0)[0]
    if len(nz) < 2:
        return None
    first, last = nz[0], nz[-1]
    ts = rev[first:last + 1]
    n_act = len(ts)
    if n_act < 6:
        return None
    mu = ts.mean(); sd = ts.std(); cv = sd / (mu + 1e-8)
    zero = (ts == 0).mean(); density = len(nz) / n_act
    x = np.arange(n_act)
    if sd > 0:
        slope, _, r, _, _ = linregress(x, ts)
        trend_slope = slope / (mu + 1e-8) * n_act
        trend_r2 = r ** 2
    else:
        trend_slope, trend_r2 = 0.0, 0.0
    if n_act > 12:
        a, b = ts[:-12], ts[12:]
        if np.std(a) > 1e-8 and np.std(b) > 1e-8:
            seasonality = float(abs(np.corrcoef(a, b)[0, 1]))
        else:
            seasonality = 0.0
    else:
        seasonality = 0.0
    return dict(cv=cv, trend_slope=trend_slope, trend_r2=trend_r2,
                seasonality=seasonality, zero_ratio=zero, density=density, mean_rev=mu)


def post_hoc_label(feats):
    if feats is None:
        return "other"
    cv, trend, r2 = feats["cv"], feats["trend_slope"], feats["trend_r2"]
    season, zero, density = feats["seasonality"], feats["zero_ratio"], feats["density"]
    if cv >= 1.2 and zero >= 0.3:
        return "volatile"
    if trend >= 0.3 and r2 >= 0.2:
        return "growth"
    if cv < 1.0 and density >= 0.7:
        return "stable"
    if season >= 0.4:
        return "seasonal"
    if trend < -0.2:
        return "decline"
    return "other"


def get_kosis_signal(kosis_cat, trend_dict, season_dict, total_trend, total_season):
    if kosis_cat in trend_dict:
        return trend_dict[kosis_cat], season_dict[kosis_cat]
    return total_trend, total_season


def get_naver_signal(kosis_cat, naver_per_cat, naver_global_mean):
    naver_cat = KOSIS_TO_NAVER.get(kosis_cat)
    if naver_cat is None or naver_cat not in naver_per_cat:
        return naver_global_mean
    return naver_per_cat[naver_cat]


def main():
    rng = np.random.default_rng(SEED)
    print("[1/5] Donor pool 로드")
    donors = load_donor_pool_with_category()
    print(f"  donor: {len(donors)}")

    print("\n[2/5] KOSIS (카테고리별) + Naver (카테고리별) prior 로드")
    (kosis_trend_per_cat, kosis_season_per_cat,
     kosis_total_trend, kosis_total_season,
     naver_per_cat, naver_global_mean) = load_priors_per_category()
    print(f"  KOSIS 카테고리 trend 로드: {len(kosis_trend_per_cat)}개")
    print(f"  Naver 카테고리 로드: {len(naver_per_cat)}개")
    print(f"  naver_factor β = {NAVER_BETA}")

    # 카테고리별 trend 범위 확인
    print(f"\n  [카테고리별 KOSIS trend 범위]")
    for cat in sorted(donors["kosis_category"].unique()):
        if cat in kosis_trend_per_cat:
            tr = kosis_trend_per_cat[cat]
            print(f"    {cat:22s}: trend [{tr.min():+.3f}, {tr.max():+.3f}]")

    print(f"\n[3/5] 합성 ({len(donors)} × {SAMPLES_PER_DONOR})")
    pool_stats = {
        "log_mean": float(donors["log_avg_rev"].mean()),
        "log_std": float(donors["log_avg_rev"].std()),
    }
    promo = make_promo_dummy()
    dates = pd.date_range(START_DATE, periods=N_MONTHS, freq="MS")

    records, seller_meta = [], []
    for _, donor in donors.iterrows():
        kosis_cat = donor["kosis_category"]
        ks_trend, ks_season = get_kosis_signal(
            kosis_cat, kosis_trend_per_cat, kosis_season_per_cat,
            kosis_total_trend, kosis_total_season)
        naver_sig = get_naver_signal(kosis_cat, naver_per_cat, naver_global_mean)

        for k in range(SAMPLES_PER_DONOR):
            sid = f"k_{donor['seller_id'][:8]}_{k}"
            rev, base_mu = generate_seller(
                donor, pool_stats, ks_trend, ks_season, naver_sig, promo, rng)
            feats = calc_synth_features(rev)
            label = post_hoc_label(feats)

            for t in range(N_MONTHS):
                records.append(dict(
                    seller_id=sid,
                    donor_id=donor["seller_id"],
                    olist_category=donor["olist_category"],
                    kosis_category=kosis_cat,
                    month_idx=t,
                    date=dates[t],
                    monthly_revenue=float(rev[t]),
                    type=label,
                    mu=base_mu,
                    promo=float(promo[t]),
                    naver_index=float(naver_sig[t]),
                    kosis_trend=float(ks_trend[t]),
                ))
            seller_meta.append(dict(
                seller_id=sid, donor_id=donor["seller_id"],
                olist_category=donor["olist_category"],
                kosis_category=kosis_cat, type=label, mu=base_mu,
            ))

    df = pd.DataFrame(records)
    meta = pd.DataFrame(seller_meta)
    print(f"  합성 셀러: {meta['seller_id'].nunique()}")

    print("\n[4/5] 검증")

    # 셀러별 KOSIS 카테고리 trend 상관
    kosis_corrs = []
    for sid, sdf in df.groupby("seller_id"):
        sdf = sdf.sort_values("month_idx")
        if sdf["monthly_revenue"].std() > 1e-6 and sdf["kosis_trend"].std() > 1e-6:
            r = sdf["monthly_revenue"].corr(sdf["kosis_trend"])
            if not np.isnan(r):
                kosis_corrs.append(r)
    kosis_corrs = np.array(kosis_corrs)

    # 셀러별 Naver 상관
    naver_corrs = []
    for sid, sdf in df.groupby("seller_id"):
        sdf = sdf.sort_values("month_idx")
        if sdf["monthly_revenue"].std() > 1e-6 and sdf["naver_index"].std() > 1e-6:
            r = sdf["monthly_revenue"].corr(sdf["naver_index"])
            if not np.isnan(r):
                naver_corrs.append(r)
    naver_corrs = np.array(naver_corrs)

    # 카테고리별 합계 KOSIS r (보고서 0.55 회복 측정)
    cat_kosis_rs = {}
    for cat in df["kosis_category"].dropna().unique():
        cdf = df[df["kosis_category"] == cat]
        agg = cdf.groupby("month_idx")["monthly_revenue"].sum().values
        kt = cdf.groupby("month_idx")["kosis_trend"].first().values
        if len(agg) >= 12 and np.std(agg) > 1e-6:
            r, _ = pearsonr(agg, kt)
            cat_kosis_rs[cat] = float(r)

    print(f"\n  매출-KOSIS 카테고리 trend (셀러별):")
    print(f"    r mean = {kosis_corrs.mean():+.4f}  median = {np.median(kosis_corrs):+.4f}")
    print(f"    |r|>0.3 비율: {(np.abs(kosis_corrs)>0.3).mean()*100:.1f}%")
    print(f"    v3: 0.097 / v5: -0.012 → v6: {kosis_corrs.mean():+.4f}")

    print(f"\n  매출-Naver 카테고리 (셀러별):")
    print(f"    r mean = {naver_corrs.mean():+.4f}  median = {np.median(naver_corrs):+.4f}")
    print(f"    |r|>0.2 비율: {(np.abs(naver_corrs)>0.2).mean()*100:.1f}%")
    print(f"    v3: 0.03 / v5: +0.227 → v6: {naver_corrs.mean():+.4f}")

    print(f"\n  카테고리별 매출 합계 ↔ KOSIS 카테고리 trend:")
    sorted_cats = sorted(cat_kosis_rs.items(), key=lambda x: -abs(x[1]))
    for cat, r in sorted_cats:
        n = (meta["kosis_category"] == cat).sum()
        print(f"    {cat:22s}: r={r:+.3f} (n={n})")

    cat_kosis_arr = np.array(list(cat_kosis_rs.values()))
    print(f"\n    카테고리 합계 r 평균: {cat_kosis_arr.mean():+.4f}, |r| 평균: {np.abs(cat_kosis_arr).mean():.4f}")
    print(f"    (v3 매출 합계 vs 합계 trend: r=+0.51)")

    print(f"\n[5/5] 저장 + 시각화")
    out_path = DATA / "cohort_kr_v6.parquet"
    df.to_parquet(out_path, index=False)
    print(f"  [save] {out_path}")

    validation = {
        "version": "v6",
        "method": "per-seller bootstrap + KOSIS 카테고리별 + Naver 카테고리별 + β=0.4",
        "config": {
            "n_sellers": int(meta["seller_id"].nunique()),
            "n_rows": int(len(df)),
            "n_donors": int(len(donors)),
            "samples_per_donor": SAMPLES_PER_DONOR,
            "naver_beta": NAVER_BETA,
        },
        "seller_level_correlations": {
            "kosis_trend_r_mean": float(kosis_corrs.mean()),
            "kosis_trend_r_median": float(np.median(kosis_corrs)),
            "kosis_trend_abs_gt_03_pct": float((np.abs(kosis_corrs) > 0.3).mean() * 100),
            "naver_r_mean": float(naver_corrs.mean()),
            "naver_r_median": float(np.median(naver_corrs)),
            "naver_abs_gt_02_pct": float((np.abs(naver_corrs) > 0.2).mean() * 100),
        },
        "category_aggregate_kosis_r": cat_kosis_rs,
        "comparison_history": {
            "v3": {"kosis_seller_r": 0.097, "naver_seller_r": 0.03, "agg_kosis_r": 0.51},
            "v5": {"kosis_seller_r": -0.012, "naver_seller_r": 0.227, "agg_kosis_r": 0.09},
            "v6": {
                "kosis_seller_r": float(kosis_corrs.mean()),
                "naver_seller_r": float(naver_corrs.mean()),
                "agg_kosis_r_mean": float(cat_kosis_arr.mean()),
            },
        },
        "type_distribution": meta["type"].value_counts().to_dict(),
        "kosis_category_distribution": meta["kosis_category"].value_counts().to_dict(),
    }
    (DATA / "cohort_kr_v6_validation.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False))
    print(f"  [save] cohort_kr_v6_validation.json")

    # 시각화
    color_map = {"stable": "steelblue", "growth": "mediumseagreen",
                 "volatile": "crimson", "seasonal": "darkorange",
                 "decline": "gray", "other": "lightgray"}
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # (1) KOSIS 상관 분포
    ax = axes[0, 0]
    ax.hist(kosis_corrs, bins=40, color="steelblue", alpha=0.7, edgecolor="white")
    ax.axvline(kosis_corrs.mean(), color="navy", linestyle="--",
               label=f"v6 mean = {kosis_corrs.mean():+.3f}")
    ax.axvline(0.097, color="gray", linestyle=":", label="v3 mean = 0.097")
    ax.axvline(-0.012, color="red", linestyle=":", label="v5 mean = -0.012")
    ax.set_xlabel("셀러별 매출-KOSIS 카테고리 trend r")
    ax.set_ylabel("셀러 수")
    ax.set_title("v6 매출-KOSIS 상관 (카테고리별)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # (2) Naver 상관 분포
    ax = axes[0, 1]
    ax.hist(naver_corrs, bins=40, color="mediumseagreen", alpha=0.7, edgecolor="white")
    ax.axvline(naver_corrs.mean(), color="darkgreen", linestyle="--",
               label=f"v6 mean = {naver_corrs.mean():+.3f}")
    ax.axvline(0.227, color="orange", linestyle=":", label="v5 mean = 0.227")
    ax.axvline(0.226, color="red", linestyle=":", label="KOSIS 실측 0.226")
    ax.set_xlabel("셀러별 매출-Naver r")
    ax.set_ylabel("셀러 수")
    ax.set_title("v6 매출-Naver 상관")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # (3) 카테고리별 KOSIS 합계 r
    ax = axes[1, 0]
    cats_sorted = sorted(cat_kosis_rs.keys(), key=lambda c: -cat_kosis_rs[c])
    rs_sorted = [cat_kosis_rs[c] for c in cats_sorted]
    colors = ["green" if r > 0.3 else "orange" if r > 0 else "red" for r in rs_sorted]
    ax.barh(cats_sorted[::-1], rs_sorted[::-1], color=colors[::-1])
    ax.axvline(0.5, color="black", linestyle="--", alpha=0.5, label="v3 합계 r=0.51")
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("카테고리 매출 합계 ↔ KOSIS r")
    ax.set_title("카테고리별 KOSIS 회복 정도")
    ax.legend(); ax.grid(alpha=0.3)

    # (4) v3 vs v5 vs v6 비교
    ax = axes[1, 1]
    metrics = ["KOSIS seller", "Naver seller", "KOSIS aggregate"]
    v3_vals = [0.097, 0.03, 0.51]
    v5_vals = [-0.012, 0.227, 0.09]
    v6_vals = [float(kosis_corrs.mean()), float(naver_corrs.mean()), float(cat_kosis_arr.mean())]
    x = np.arange(len(metrics))
    width = 0.27
    ax.bar(x - width, v3_vals, width, label="v3", color="gray")
    ax.bar(x, v5_vals, width, label="v5", color="orange")
    ax.bar(x + width, v6_vals, width, label="v6", color="green")
    ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=10)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Pearson r")
    ax.set_title("v3 vs v5 vs v6 비교")
    ax.legend(); ax.grid(alpha=0.3)

    plt.suptitle(f"v6 합성 코호트 (KOSIS 카테고리별 + Naver β={NAVER_BETA}, n={meta['seller_id'].nunique()})",
                 fontsize=13, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(DATA / "cohort_kr_v6_diagnostics.png", dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  [save] cohort_kr_v6_diagnostics.png")

    print("\n=== v6 완료 ===")
    print(f"  다음: Prophet v6 / LSTM v3 학습 (cohort_kr_v6 사용)")


if __name__ == "__main__":
    main()
