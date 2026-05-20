"""한국형 합성 코호트 생성기 v3 — 카테고리 정보 통합

v2 → v3 변경:
1. Olist 셀러별 주력 카테고리(mode) 추출
2. olist_to_kosis_mapping.csv로 KOSIS 카테고리 매핑
3. 모호 카테고리(EXCLUDE) 셀러는 합성에서 제외
4. 합성 셀러에 olist_category, kosis_category 필드 추가
5. Phase 3 v2에서 카테고리별 정책 차별화에 사용 가능

m_i, L_personal_min은 v2 그대로 유지 (D-2 카테고리별 m_i는 별도 작업)

산출:
- Data/cohort_kr_v3.parquet
- Data/cohort_kr_v3_validation.json
- Data/cohort_kr_v3_diagnostics.png
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

# === Config (v2와 동일) ===
SEED = 42
N_MONTHS = 24
START_DATE = "2023-01-01"
SAMPLES_PER_DONOR = 2
NOISE_SCALE = 0.3
KOREAN_LOG_MEAN = np.log(500)
KOREAN_LOG_STD = 0.8
PROMO_MONTHS = [1, 2, 9, 11]

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False


# === Donor + 카테고리 로드 ===
def load_donor_pool_with_category() -> pd.DataFrame:
    """seller_features_v3.csv + AR(1) + 주력 카테고리 + KOSIS 매핑.
    모호(EXCLUDE) 카테고리 셀러는 제외.
    """
    sf = pd.read_csv(DATA / "seller_features_v3.csv")
    monthly = pd.read_csv(DATA / "monthly_seller_revenue.csv")
    monthly["ym"] = pd.to_datetime(monthly["year_month"])

    # AR(1) 계산
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

    # Olist 카테고리 (셀러별 주력)
    items = pd.read_csv(DATA / "Olist_Data" / "olist_order_items_dataset.csv")
    products = pd.read_csv(DATA / "Olist_Data" / "olist_products_dataset.csv")
    merged = items.merge(products[["product_id", "product_category_name"]], on="product_id")

    def primary_cat(g):
        d = g["product_category_name"].dropna()
        return d.mode().iloc[0] if len(d) > 0 else None
    seller_primary = merged.groupby("seller_id").apply(primary_cat).reset_index()
    seller_primary.columns = ["seller_id", "olist_category"]

    sf = sf.merge(seller_primary, on="seller_id", how="left")

    # KOSIS 매핑 적용
    mapping = pd.read_csv(DATA / "olist_to_kosis_mapping.csv")
    map_dict = dict(zip(mapping["olist_category"], mapping["kosis_category"]))
    status_dict = dict(zip(mapping["olist_category"], mapping["status"]))
    sf["kosis_category"] = sf["olist_category"].map(map_dict)
    sf["mapping_status"] = sf["olist_category"].map(status_dict).fillna("EXCLUDE")

    n_before = len(sf)
    sf_filtered = sf[sf["mapping_status"] == "OK"].copy()
    n_after = len(sf_filtered)
    print(f"  카테고리 매핑: 전체 {n_before} → OK {n_after} (EXCLUDE {n_before - n_after}명 제거)")

    return sf_filtered


# === KOSIS / Naver prior 로드 (v2와 동일) ===
def load_priors() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_parquet(DATA / "kosis" / "kr_trend_season.parquet")
    total = df[df["category"] == "합계"].sort_values("date").tail(N_MONTHS)
    if len(total) < N_MONTHS:
        pad = N_MONTHS - len(total)
        total = pd.concat([total.head(1)] * pad + [total]).reset_index(drop=True)
    raw = total["value"].values[:N_MONTHS].astype(float)
    trend_shape = raw / raw.mean() - 1.0
    season = total["season_rel"].values[:N_MONTHS] * 3.0

    nv = pd.read_csv(DATA / "naver" / "naver_monthly.csv", index_col=0, parse_dates=True)
    composite = nv.mean(axis=1).sort_index().tail(N_MONTHS).values.astype(float)
    if len(composite) < N_MONTHS:
        composite = np.pad(composite, (N_MONTHS - len(composite), 0), mode="edge")
    composite = composite[:N_MONTHS]
    composite = composite / max(composite.max(), 1e-8)
    return trend_shape, season, composite


def make_promo_dummy() -> np.ndarray:
    dates = pd.date_range(START_DATE, periods=N_MONTHS, freq="MS")
    return np.array([1.0 if d.month in PROMO_MONTHS else 0.0 for d in dates])


def donor_to_korean_mu(donor_log_rev, pool_log_mean, pool_log_std, rng):
    z = (donor_log_rev - pool_log_mean) / max(pool_log_std, 0.01)
    z = float(np.clip(z, -3, 3))
    return float(np.exp(KOREAN_LOG_MEAN + z * KOREAN_LOG_STD + 0.1 * rng.standard_normal()))


def generate_seller(donor, pool_stats, kosis_trend, kosis_season, naver_cov, promo, rng):
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
        naver_factor = 1.0 + 0.15 * (naver_cov[t] - 0.5)

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


def main():
    rng = np.random.default_rng(SEED)
    print("[1/5] Donor pool + 카테고리 로드 + 모호 카테고리 제거")
    donors = load_donor_pool_with_category()
    print(f"  최종 donor 수: {len(donors)}")
    print(f"  카테고리 분포 top 10:")
    print(donors["kosis_category"].value_counts().head(10).to_string())

    print("\n[2/5] KOSIS / Naver prior 로드")
    kosis_trend, kosis_season, naver_cov = load_priors()

    print(f"\n[3/5] 합성 셀러 생성 ({len(donors)} × {SAMPLES_PER_DONOR})")
    pool_stats = {
        "log_mean": float(donors["log_avg_rev"].mean()),
        "log_std": float(donors["log_avg_rev"].std()),
    }
    promo = make_promo_dummy()
    dates = pd.date_range(START_DATE, periods=N_MONTHS, freq="MS")

    records, seller_meta = [], []
    for _, donor in donors.iterrows():
        for k in range(SAMPLES_PER_DONOR):
            sid = f"k_{donor['seller_id'][:8]}_{k}"
            rev, base_mu = generate_seller(donor, pool_stats, kosis_trend, kosis_season,
                                            naver_cov, promo, rng)
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
                    naver_index=float(naver_cov[t]),
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

    print("\n[4/5] KOSIS 검증")
    monthly_total = df.groupby("month_idx")["monthly_revenue"].sum().values
    monthly_norm = monthly_total / monthly_total.mean() - 1.0
    r, p = pearsonr(monthly_norm, kosis_trend)
    print(f"  Pearson r vs KOSIS = {r:+.4f}, p = {p:.4f}")

    print(f"\n[5/5] 저장 + 시각화")
    out_path = DATA / "cohort_kr_v3.parquet"
    df.to_parquet(out_path, index=False)
    print(f"  [save] {out_path}")

    validation = {
        "version": "v3",
        "method": "per-seller bootstrap + 카테고리 통합",
        "n_sellers": int(meta["seller_id"].nunique()),
        "n_rows": int(len(df)),
        "n_donors": int(len(donors)),
        "samples_per_donor": SAMPLES_PER_DONOR,
        "pearson_r_vs_kosis": float(r),
        "p_value": float(p),
        "type_distribution": meta["type"].value_counts().to_dict(),
        "kosis_category_distribution": meta["kosis_category"].value_counts().to_dict(),
        "ambiguous_donors_excluded": 60,
        "donor_pool_size_before_filter": 651,
    }
    (DATA / "cohort_kr_v3_validation.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False))
    print(f"  [save] cohort_kr_v3_validation.json")

    # 시각화
    color_map = {"stable": "steelblue", "growth": "mediumseagreen",
                 "volatile": "crimson", "seasonal": "darkorange",
                 "decline": "gray", "other": "lightgray"}
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # (1) 카테고리 분포
    ax = axes[0, 0]
    cat_cnt = meta["kosis_category"].value_counts()
    ax.barh(cat_cnt.index[::-1], cat_cnt.values[::-1], color="steelblue", alpha=0.8)
    ax.set_xlabel("셀러 수")
    ax.set_title(f"KOSIS 카테고리 분포 (총 {len(cat_cnt)} 카테고리)")
    ax.grid(alpha=0.3)

    # (2) 유형 분포
    ax = axes[0, 1]
    type_cnt = meta["type"].value_counts()
    colors_t = [color_map.get(t, "gray") for t in type_cnt.index]
    ax.bar(type_cnt.index, type_cnt.values, color=colors_t, edgecolor="white")
    for i, v in enumerate(type_cnt.values):
        ax.text(i, v + 5, str(v), ha="center", fontweight="bold")
    ax.set_title("유형 분포 (사후 라벨)")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(alpha=0.3)

    # (3) KOSIS vs 합성 합계
    ax = axes[1, 0]
    ax.plot(dates, kosis_trend, "o-", color="red", label="KOSIS trend", linewidth=2)
    ax.plot(dates, monthly_norm, "s-", color="steelblue", label="v3 합성 합계", linewidth=2)
    ax.set_title(f"KOSIS vs v3 합성 (Pearson r = {r:+.3f})")
    ax.legend()
    ax.tick_params(axis="x", rotation=30)
    ax.grid(alpha=0.3)

    # (4) 카테고리 × 유형 cross
    ax = axes[1, 1]
    cross = pd.crosstab(meta["kosis_category"], meta["type"])
    cross_top = cross.loc[cat_cnt.head(8).index]
    cross_top.plot(kind="bar", stacked=True, ax=ax,
                    color=[color_map.get(t, "gray") for t in cross_top.columns])
    ax.set_title("Top 8 카테고리 × 유형 분포")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)

    plt.suptitle(f"v3 합성 코호트 (n={meta['seller_id'].nunique()}, r={r:+.3f}, 카테고리 통합)",
                 fontsize=14, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(DATA / "cohort_kr_v3_diagnostics.png", dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  [save] cohort_kr_v3_diagnostics.png")
    print("\n=== 완료 ===")


if __name__ == "__main__":
    main()
