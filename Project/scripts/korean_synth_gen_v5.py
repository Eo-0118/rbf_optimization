"""한국형 합성 코호트 생성기 v5 — 외생변수 generative 강화

v3/v4 진단 결과:
  - v3/v4의 매출-Naver 상관: 0.002 (사실상 무관)
  - KOSIS-Naver 실측 상관: |r| 평균 0.226 (raw), 0.305 (1차 차분)
  → 합성 process가 실제 한국 시장보다 100배 약하게 외생변수 반영

v5 변경:
  1. naver_cov를 카테고리별 시그널로 분리 (글로벌 평균 → 셀러의 kosis_category에 맞춤)
  2. naver_factor 강도 강화: ±7.5% → 일단 ±20% (β=0.4)로 시작
  3. 합성 후 매출-Naver |r| 검증 → 목표 0.2~0.3 (KOSIS 실측 일치)
  4. KOSIS-Naver 매핑 표 (validate_exog_correlation.py와 동일 정책)
  5. Promo는 그대로 (별도 검증 필요 시 v6에서)

학술 정당화:
  본 합성은 "한국 KOSIS-Naver 측정 상관 |r|≈0.25"을 합성 가정으로 반영.
  실제 한국 셀러 데이터 부재로 인한 generative assumption임을 보고서 한계로 명시.

이전 결과 (v3/v4)는 보존하고 v5를 신규 산출:
- Data/cohort_kr_v5.parquet
- Data/cohort_kr_v5_validation.json
- Data/cohort_kr_v5_diagnostics.png
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

# === Config (v3와 동일) ===
SEED = 42
N_MONTHS = 24
START_DATE = "2023-01-01"
SAMPLES_PER_DONOR = 2
NOISE_SCALE = 0.3
KOREAN_LOG_MEAN = np.log(500)
KOREAN_LOG_STD = 0.8
PROMO_MONTHS = [1, 2, 9, 11]

# === v5 신규 ===
NAVER_BETA = 0.4   # ±20% 강도 (이전 0.15 → 0.4). 실측 |r|≈0.25를 목표로 합성 후 조정 가능

# KOSIS → Naver 매핑 (validate_exog_correlation.py와 동일)
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

    # AR(1)
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

    # Olist 카테고리
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


def load_priors():
    """KOSIS trend/season + Naver 카테고리별 시그널 로드.

    Returns:
        kosis_trend, kosis_season: (N_MONTHS,) 전체 시장 시그널
        naver_per_cat: dict[naver_cat] → (N_MONTHS,) 정규화된 카테고리별 시그널
        naver_global_mean: (N_MONTHS,) — 카테고리 없는 셀러 fallback
    """
    df = pd.read_parquet(DATA / "kosis" / "kr_trend_season.parquet")
    total = df[df["category"] == "합계"].sort_values("date").tail(N_MONTHS)
    if len(total) < N_MONTHS:
        pad = N_MONTHS - len(total)
        total = pd.concat([total.head(1)] * pad + [total]).reset_index(drop=True)
    raw = total["value"].values[:N_MONTHS].astype(float)
    trend_shape = raw / raw.mean() - 1.0
    season = total["season_rel"].values[:N_MONTHS] * 3.0

    # Naver 카테고리별 (v5 신규)
    nv = pd.read_csv(DATA / "naver" / "naver_monthly.csv")
    nv["date"] = pd.to_datetime(nv["date"])
    nv = nv.set_index("date").sort_index()
    # START_DATE부터 N_MONTHS 추출
    nv_window = nv.loc[pd.to_datetime(START_DATE):].head(N_MONTHS)
    if len(nv_window) < N_MONTHS:
        # 끝부분이 부족하면 마지막 값으로 pad
        last = nv_window.iloc[[-1]]
        while len(nv_window) < N_MONTHS:
            nv_window = pd.concat([nv_window, last])
    nv_window = nv_window.iloc[:N_MONTHS]

    # 카테고리별 정규화 (0 중심으로 — naver_factor가 deviation 기반)
    naver_per_cat = {}
    for col in nv_window.columns:
        x = nv_window[col].values.astype(float)
        # NaN 처리: 전체 평균으로 fill
        mean_val = np.nanmean(x)
        x = np.where(np.isnan(x), mean_val, x)
        # min-max 정규화 (0~1)
        if x.max() - x.min() > 1e-8:
            x_norm = (x - x.min()) / (x.max() - x.min())
        else:
            x_norm = np.full_like(x, 0.5)
        naver_per_cat[col] = x_norm

    # 전체 평균 (fallback)
    all_signals = np.stack(list(naver_per_cat.values()))
    naver_global_mean = all_signals.mean(axis=0)

    return trend_shape, season, naver_per_cat, naver_global_mean


def make_promo_dummy() -> np.ndarray:
    dates = pd.date_range(START_DATE, periods=N_MONTHS, freq="MS")
    return np.array([1.0 if d.month in PROMO_MONTHS else 0.0 for d in dates])


def donor_to_korean_mu(donor_log_rev, pool_log_mean, pool_log_std, rng):
    z = (donor_log_rev - pool_log_mean) / max(pool_log_std, 0.01)
    z = float(np.clip(z, -3, 3))
    return float(np.exp(KOREAN_LOG_MEAN + z * KOREAN_LOG_STD + 0.1 * rng.standard_normal()))


def generate_seller(donor, pool_stats, kosis_trend, kosis_season,
                     naver_signal, promo, rng):
    """v5: naver_signal은 셀러의 kosis_category에 매칭된 (N_MONTHS,) 배열."""
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
        market_factor = 1.0 + kosis_trend[t]
        season_factor = 1.0 + kosis_season[t] * (0.3 + 0.7 * season_strength)
        donor_growth = 1.0 + trend_per_month * t
        base_t = base_mu * market_factor * season_factor * donor_growth

        sigma = base_t * cv * NOISE_SCALE
        noise = ar1 * noise_prev + sigma * rng.standard_normal()
        noise_prev = noise

        spike = 0.0
        if spike_ratio > 2.0 and rng.random() < 0.04:
            spike = base_t * (spike_ratio - 1) * rng.uniform(0.2, 0.5)

        promo_boost = 0.20 * base_mu * promo[t]
        # v5: naver_factor 강도 ↑ (β=0.4 = ±20% 변동) + 카테고리별 signal
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
    n_nonzero = len(nz)
    mu = ts.mean()
    sd = ts.std()
    cv = sd / (mu + 1e-8)
    zero = (ts == 0).mean()
    density = n_nonzero / n_act
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


def get_seller_naver_signal(kosis_category, naver_per_cat, naver_global_mean):
    """셀러의 kosis_category에 해당하는 Naver 시그널 반환."""
    naver_cat = KOSIS_TO_NAVER.get(kosis_category)
    if naver_cat is None or naver_cat not in naver_per_cat:
        return naver_global_mean
    return naver_per_cat[naver_cat]


def main():
    rng = np.random.default_rng(SEED)
    print("[1/5] Donor pool + 카테고리 로드 + 모호 카테고리 제거")
    donors = load_donor_pool_with_category()
    print(f"  최종 donor 수: {len(donors)}")

    print("\n[2/5] KOSIS / Naver prior 로드 (카테고리별)")
    kosis_trend, kosis_season, naver_per_cat, naver_global_mean = load_priors()
    print(f"  KOSIS trend: {len(kosis_trend)} months, range=[{kosis_trend.min():+.3f}, {kosis_trend.max():+.3f}]")
    print(f"  Naver 카테고리: {len(naver_per_cat)}개 (각각 0~1 정규화)")
    print(f"  naver_factor β = {NAVER_BETA} (±{NAVER_BETA/2*100:.0f}% 변동)")

    print(f"\n[3/5] 합성 셀러 생성 ({len(donors)} × {SAMPLES_PER_DONOR})")
    pool_stats = {
        "log_mean": float(donors["log_avg_rev"].mean()),
        "log_std": float(donors["log_avg_rev"].std()),
    }
    promo = make_promo_dummy()
    dates = pd.date_range(START_DATE, periods=N_MONTHS, freq="MS")

    records, seller_meta = [], []
    for _, donor in donors.iterrows():
        seller_naver = get_seller_naver_signal(
            donor["kosis_category"], naver_per_cat, naver_global_mean)
        for k in range(SAMPLES_PER_DONOR):
            sid = f"k_{donor['seller_id'][:8]}_{k}"
            rev, base_mu = generate_seller(donor, pool_stats, kosis_trend, kosis_season,
                                            seller_naver, promo, rng)
            feats = calc_synth_features(rev)
            label = post_hoc_label(feats)

            for t in range(N_MONTHS):
                records.append(dict(
                    seller_id=sid,
                    donor_id=donor["seller_id"],
                    olist_category=donor["olist_category"],
                    kosis_category=donor["kosis_category"],
                    month_idx=t,
                    date=dates[t],
                    monthly_revenue=float(rev[t]),
                    type=label,
                    mu=base_mu,
                    promo=float(promo[t]),
                    naver_index=float(seller_naver[t]),
                ))
            seller_meta.append(dict(
                seller_id=sid, donor_id=donor["seller_id"],
                olist_category=donor["olist_category"],
                kosis_category=donor["kosis_category"],
                type=label, mu=base_mu,
            ))

    df = pd.DataFrame(records)
    meta = pd.DataFrame(seller_meta)
    print(f"  합성 셀러: {meta['seller_id'].nunique()} (총 {len(df)} 행)")

    print("\n[4/5] 검증")
    # KOSIS 상관
    monthly_total = df.groupby("month_idx")["monthly_revenue"].sum().values
    monthly_norm = monthly_total / monthly_total.mean() - 1.0
    r_kosis, p_kosis = pearsonr(monthly_norm, kosis_trend)
    print(f"  KOSIS trend ↔ 합성 매출 합계: r={r_kosis:+.4f} (p={p_kosis:.4f})")

    # 매출-Naver 상관 (셀러별, 그리고 카테고리별)
    print(f"\n  매출-Naver 상관 분포 (목표 0.2~0.3, 실측 KOSIS 0.226):")
    seller_corrs = []
    for sid, sdf in df.groupby("seller_id"):
        if sdf["naver_index"].std() > 1e-6 and sdf["monthly_revenue"].std() > 1e-6:
            r = sdf["monthly_revenue"].corr(sdf["naver_index"])
            if not np.isnan(r):
                seller_corrs.append(r)
    seller_corrs = np.array(seller_corrs)
    print(f"    셀러별 r mean: {seller_corrs.mean():+.4f}, median: {np.median(seller_corrs):+.4f}")
    print(f"    |r| > 0.3 비율: {(np.abs(seller_corrs) > 0.3).mean()*100:.1f}%")
    print(f"    |r| > 0.2 비율: {(np.abs(seller_corrs) > 0.2).mean()*100:.1f}%")
    print(f"  v3/v4: 셀러별 r mean ≈ 0.03 → v5: {seller_corrs.mean():+.4f}")

    # 카테고리별 상관
    print(f"\n  카테고리별 평균 |r|:")
    for cat in sorted(df["kosis_category"].dropna().unique()):
        cdf = df[df["kosis_category"] == cat]
        cat_corrs = []
        for sid, sdf in cdf.groupby("seller_id"):
            if sdf["naver_index"].std() > 1e-6 and sdf["monthly_revenue"].std() > 1e-6:
                r = sdf["monthly_revenue"].corr(sdf["naver_index"])
                if not np.isnan(r):
                    cat_corrs.append(r)
        if len(cat_corrs) >= 3:
            c = np.array(cat_corrs)
            naver_cat = KOSIS_TO_NAVER.get(cat) or "(평균)"
            print(f"    {cat:22s} → {naver_cat:14s} n={len(c):3d}, "
                  f"mean r={c.mean():+.3f}, |r|>0.2: {(np.abs(c)>0.2).mean()*100:4.1f}%")

    print(f"\n[5/5] 저장 + 시각화")
    out_path = DATA / "cohort_kr_v5.parquet"
    df.to_parquet(out_path, index=False)
    print(f"  [save] {out_path}")

    validation = {
        "version": "v5",
        "method": "per-seller bootstrap + 카테고리 통합 + Naver generative 강화",
        "config": {
            "n_sellers": int(meta["seller_id"].nunique()),
            "n_rows": int(len(df)),
            "n_donors": int(len(donors)),
            "samples_per_donor": SAMPLES_PER_DONOR,
            "naver_beta": NAVER_BETA,
            "naver_beta_range_pct": NAVER_BETA / 2 * 100,
        },
        "kosis_validation": {
            "pearson_r": float(r_kosis),
            "p_value": float(p_kosis),
        },
        "naver_validation": {
            "seller_corr_mean": float(seller_corrs.mean()),
            "seller_corr_median": float(np.median(seller_corrs)),
            "abs_r_gt_02_pct": float((np.abs(seller_corrs) > 0.2).mean() * 100),
            "abs_r_gt_03_pct": float((np.abs(seller_corrs) > 0.3).mean() * 100),
            "target_kosis_measured": 0.226,
            "v3_v4_baseline": 0.03,
        },
        "type_distribution": meta["type"].value_counts().to_dict(),
        "kosis_category_distribution": meta["kosis_category"].value_counts().to_dict(),
    }
    (DATA / "cohort_kr_v5_validation.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False))
    print(f"  [save] cohort_kr_v5_validation.json")

    # 시각화
    color_map = {"stable": "steelblue", "growth": "mediumseagreen",
                 "volatile": "crimson", "seasonal": "darkorange",
                 "decline": "gray", "other": "lightgray"}
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    ax = axes[0, 0]
    cat_cnt = meta["kosis_category"].value_counts()
    ax.barh(cat_cnt.index[::-1], cat_cnt.values[::-1], color="steelblue", alpha=0.8)
    ax.set_xlabel("셀러 수")
    ax.set_title(f"카테고리 분포")
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    type_cnt = meta["type"].value_counts()
    colors_t = [color_map.get(t, "gray") for t in type_cnt.index]
    ax.bar(type_cnt.index, type_cnt.values, color=colors_t)
    ax.set_title("유형 분포"); ax.tick_params(axis="x", rotation=15); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(dates, kosis_trend, "o-", color="red", label="KOSIS trend", linewidth=2)
    ax.plot(dates, monthly_norm, "s-", color="steelblue", label="v5 합성 합계", linewidth=2)
    ax.set_title(f"KOSIS vs v5 (r={r_kosis:+.3f})")
    ax.legend(); ax.tick_params(axis="x", rotation=30); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.hist(seller_corrs, bins=40, color="mediumseagreen", alpha=0.7, edgecolor="white")
    ax.axvline(seller_corrs.mean(), color="darkgreen", linestyle="--",
               label=f"v5 mean = {seller_corrs.mean():+.3f}")
    ax.axvline(0.226, color="red", linestyle="--", label="KOSIS 실측 |r|=0.226")
    ax.axvline(0.03, color="gray", linestyle=":", label="v3/v4 mean ≈ 0.03")
    ax.set_xlabel("셀러별 매출-Naver Pearson r")
    ax.set_ylabel("셀러 수")
    ax.set_title("v5 매출-Naver 상관 분포")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    plt.suptitle(f"v5 합성 코호트 (β={NAVER_BETA}, n={meta['seller_id'].nunique()}, "
                 f"매출-Naver r mean = {seller_corrs.mean():+.3f})",
                 fontsize=13, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(DATA / "cohort_kr_v5_diagnostics.png", dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  [save] cohort_kr_v5_diagnostics.png")
    print("\n=== v5 완료 ===")
    print(f"\n다음: prophet_baseline_v5.py / lstm_baseline_v3.py 작성 + 학습")


if __name__ == "__main__":
    main()
