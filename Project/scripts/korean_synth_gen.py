"""한국형 합성 판매자 코호트 생성기.

v4 Layer 3: 4유형 × 1,000명 × 24개월 합성 매출 시계열.

생성식 (v4_commitment.md Section 1):
    안정형:  μ·(1+trend_KR_t) + σ_low·AR(1)
    계절형:  μ·(1+trend_KR_t)·(1+season_KR_t) + σ_mid·ε
    성장형:  μ·exp(g·t)·(1+season_KR_t) + σ_mid·ε,  g∈[3%,8%] monthly
    불안정형: μ·(1+trend_KR_t) + σ_high·StudentT(df=3) + shock_t

추가:
    m_i: 유형별 Beta 분포에서 판매자 단위 샘플 후 고정
    Olist에서 학습한 AR(1), CV, spike_ratio 반영
    네이버 카테고리 지수를 외생 covariate로 첨부
    프로모션 더미: 블프(11월), 추석(9월), 설(1~2월)

출력:
    data/cohort_kr.parquet   (4000 sellers × 24 months ≈ 96,000 rows)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
N_PER_TYPE = 1000
N_MONTHS = 24
START_DATE = "2023-01-01"

# --- Load priors ---
KOSIS_TS = ROOT / "data" / "kosis" / "kr_trend_season.parquet"
NAVER = ROOT / "data" / "naver" / "naver_monthly.csv"
VAR_PARAMS = ROOT / "data" / "variance_params.json"
OUT = ROOT / "data"


def load_kr_prior() -> tuple[np.ndarray, np.ndarray]:
    """Load KOSIS trend_rel and season_rel for '합계' category, last 24 months.

    trend_rel은 누적 상대 변화 (0.50~0.72) → 월별 변화율로 변환해 de-mean.
    season_rel은 ±5% 수준이므로 3x 증폭해 합성 데이터에 충분히 반영.
    """
    df = pd.read_parquet(KOSIS_TS)
    total = df[df["category"] == "합계"].sort_values("date").tail(N_MONTHS)
    if len(total) < N_MONTHS:
        total = df[df["category"] == "합계"].sort_values("date").tail(len(total))
        pad = N_MONTHS - len(total)
        total = pd.concat([total.head(1)] * pad + [total]).reset_index(drop=True)

    # Use raw value normalized to get the KOSIS shape directly
    raw = total["value"].values[:N_MONTHS].astype(float)
    trend_shape = raw / raw.mean() - 1.0  # centered around 0

    season = total["season_rel"].values[:N_MONTHS] * 3.0  # amplify season

    return trend_shape, season


def load_naver_cov() -> np.ndarray:
    """Load naver composite index (mean of all categories), last 24 months, normalized 0-1."""
    nv = pd.read_csv(NAVER, index_col=0, parse_dates=True)
    composite = nv.mean(axis=1).sort_index().tail(N_MONTHS).values
    composite = composite / composite.max()
    if len(composite) < N_MONTHS:
        composite = np.pad(composite, (N_MONTHS - len(composite), 0), mode="edge")
    return composite[:N_MONTHS]


def load_var_params() -> dict:
    return json.loads(VAR_PARAMS.read_text())


def promo_dummy(n_months: int) -> np.ndarray:
    """1=프로모션 월 (1,2=설, 9=추석, 11=블프), else 0."""
    dates = pd.date_range(START_DATE, periods=n_months, freq="MS")
    return np.array([1.0 if d.month in (1, 2, 9, 11) else 0.0 for d in dates])


def generate_type(
    type_name: str,
    n: int,
    trend: np.ndarray,
    season: np.ndarray,
    vp: dict,
    rng: np.random.Generator,
) -> list[dict]:
    bp = vp["m_i_beta"]
    cv_median = vp["median_cv"]
    ar1_mean = vp["mean_ar1"]
    t_df = max(vp["noise_dist"]["t_df"], 3.0)  # clamp minimum df

    # Revenue scale: Korean e-commerce small seller (만원 단위, 월 200~2000만원)
    mu_range = {
        "stable":   (800, 2000),
        "seasonal": (400, 1500),
        "growth":   (200, 800),
        "volatile": (100, 600),
    }[type_name]

    promo = promo_dummy(N_MONTHS)
    records = []

    for i in range(n):
        sid = f"{type_name}_{i:04d}"
        m_i = float(beta_dist.rvs(bp["alpha"], bp["beta"], random_state=rng))
        mu = rng.uniform(*mu_range)
        sigma = mu * cv_median * 0.3  # scale down CV (Olist CV is extreme due to zeros)
        ar1 = ar1_mean

        rev = np.zeros(N_MONTHS)
        noise_prev = 0.0

        if type_name == "stable":
            for t in range(N_MONTHS):
                base = mu * (1.0 + trend[t])
                noise = ar1 * noise_prev + sigma * 0.5 * rng.standard_normal()
                noise_prev = noise
                promo_boost = 0.1 * mu * promo[t]
                rev[t] = max(base + noise + promo_boost, 0)

        elif type_name == "seasonal":
            for t in range(N_MONTHS):
                base = mu * (1.0 + trend[t]) * (1.0 + season[t])
                noise = sigma * rng.standard_normal()
                promo_boost = 0.15 * mu * promo[t]
                rev[t] = max(base + noise + promo_boost, 0)

        elif type_name == "growth":
            g = rng.uniform(0.03, 0.08) / 12  # monthly growth rate
            for t in range(N_MONTHS):
                base = mu * np.exp(g * t) * (1.0 + season[t])
                noise = sigma * rng.standard_normal()
                promo_boost = 0.12 * mu * promo[t]
                rev[t] = max(base + noise + promo_boost, 0)

        elif type_name == "volatile":
            for t in range(N_MONTHS):
                base = mu * (1.0 + trend[t])
                noise = sigma * rng.standard_t(df=t_df)
                shock = 0.0
                if rng.random() < 0.05:
                    shock = mu * rng.uniform(-0.5, 1.5)
                promo_boost = 0.08 * mu * promo[t]
                rev[t] = max(base + noise + shock + promo_boost, 0)

        for t in range(N_MONTHS):
            records.append({
                "seller_id": sid,
                "type": type_name,
                "month_idx": t,
                "m_i": m_i,
                "mu": mu,
                "monthly_revenue": rev[t],
            })

    return records


def main():
    rng = np.random.default_rng(SEED)
    trend, season = load_kr_prior()
    naver_cov = load_naver_cov()
    vp = load_var_params()
    promo = promo_dummy(N_MONTHS)
    dates = pd.date_range(START_DATE, periods=N_MONTHS, freq="MS")

    print(f"[prior] trend range: {trend.min():.4f} ~ {trend.max():.4f}")
    print(f"[prior] season range: {season.min():.4f} ~ {season.max():.4f}")
    print(f"[prior] naver cov range: {naver_cov.min():.4f} ~ {naver_cov.max():.4f}")

    all_records = []
    for typ in ["stable", "seasonal", "growth", "volatile"]:
        print(f"\n[gen] {typ} × {N_PER_TYPE} ...")
        recs = generate_type(typ, N_PER_TYPE, trend, season, vp[typ], rng)
        all_records.extend(recs)
        rev_arr = np.array([r["monthly_revenue"] for r in recs])
        print(f"  total rows: {len(recs)}")
        print(f"  rev: mean={rev_arr.mean():.1f} std={rev_arr.std():.1f} "
              f"min={rev_arr.min():.1f} max={rev_arr.max():.1f}")
        mi_arr = np.array([r["m_i"] for r in recs[::N_MONTHS]])
        print(f"  m_i: mean={mi_arr.mean():.4f} std={mi_arr.std():.4f}")

    df = pd.DataFrame(all_records)
    # Add date, promo dummy, naver covariate
    df["date"] = df["month_idx"].apply(lambda t: dates[t])
    df["promo"] = df["month_idx"].apply(lambda t: promo[t])
    df["naver_index"] = df["month_idx"].apply(lambda t: naver_cov[t])

    # Add month dummies
    df["month"] = df["date"].dt.month

    out_path = OUT / "cohort_kr.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\n[save] {out_path}")
    print(f"[shape] {df.shape}")
    print(f"[sellers] {df['seller_id'].nunique()}")
    print(f"[types] {df['type'].value_counts().to_dict()}")

    # Quick sanity
    print("\n[sanity] Revenue stats by type:")
    print(df.groupby("type")["monthly_revenue"].describe()[["mean", "std", "min", "max"]].to_string())

    print("\n[sanity] m_i stats by type:")
    mi = df.drop_duplicates("seller_id").groupby("type")["m_i"]
    print(mi.describe()[["mean", "std", "min", "max"]].to_string())


if __name__ == "__main__":
    main()
