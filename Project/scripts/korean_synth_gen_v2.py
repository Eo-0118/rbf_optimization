"""한국형 합성 코호트 생성기 v2 — Per-seller 부트스트랩

v1 (cohort_kr.parquet)과의 차이:
- v1: KMeans 4유형 분류 → 유형별 평균 분산 → 4유형 × 1000명 수식 합성
- v2: 클러스터링 없음. 651명 Olist donor 각자의 개별 분산 지문 사용
       → donor × k batch 한국 합성 셀러 (이질성 보존)

v2.1 변경 (2026-05-04): 외생 변수 영향 강화
- Naver 영향: ±2.5% → ±15% (시계열 예측 모델이 학습 가능한 수준으로)
- Promo 영향: +10% → +30% (한국 이커머스 프로모션 현실에 가깝게)
- 사유: weak exo 버전(archive_v1/cohort_kr_v2_weak_exo.parquet)에서
       Prophet 외생 변수 추가가 효과 거의 없었음 → 합성 데이터에 신호 부재 확인
- 영향: KOSIS Pearson r 다소 변동 가능 (외생 변수 추가 변동성 때문)

방법:
  1. seller_features_v3.csv (651 donor pool) 로드
  2. monthly_seller_revenue.csv에서 각 donor의 AR(1) 추출
  3. 각 donor의 (CV, AR(1), trend, seasonality, spike, zero, scale) 지문 + KOSIS 한국 trend/season prior + Naver covariate
     → k명의 한국 합성 셀러 생성
  4. 사후 라벨링: v3 임계 룰을 합성 데이터에 적용해 4유형 라벨 부여 (실험 설계용)
  5. 검증: 합성 데이터 합계 vs KOSIS 합계 패턴 → Pearson r
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
SAMPLES_PER_DONOR = 2  # 1 donor → 2명 한국 셀러 (총 651 × 2 = 1302)
NOISE_SCALE = 0.3      # CV 적용 시 스케일 조정 (Olist CV가 한국보다 큼 → 깎음)

# Korean market scale (만원 단위, log-normal)
KOREAN_LOG_MEAN = np.log(500)   # 평균 500만원/월 가정
KOREAN_LOG_STD = 0.8

PROMO_MONTHS = [1, 2, 9, 11]  # 설(1-2월), 추석(9월), 블프(11월)

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False


# === Donor pool 로드 ===
def load_donor_pool() -> pd.DataFrame:
    """seller_features_v3.csv + AR(1) 추가."""
    sf = pd.read_csv(DATA / "seller_features_v3.csv")
    monthly = pd.read_csv(DATA / "monthly_seller_revenue.csv")
    monthly["ym"] = pd.to_datetime(monthly["year_month"])

    # 각 donor의 활성 구간 내 AR(1) (lag-1 자기상관) 계산
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
    return sf


# === KOSIS / Naver prior 로드 ===
def load_priors() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """KOSIS 합계 trend shape, season, Naver composite covariate."""
    df = pd.read_parquet(DATA / "kosis" / "kr_trend_season.parquet")
    total = df[df["category"] == "합계"].sort_values("date").tail(N_MONTHS)
    if len(total) < N_MONTHS:
        pad = N_MONTHS - len(total)
        total = pd.concat([total.head(1)] * pad + [total]).reset_index(drop=True)
    raw = total["value"].values[:N_MONTHS].astype(float)
    trend_shape = raw / raw.mean() - 1.0   # centered around 0
    season = total["season_rel"].values[:N_MONTHS] * 3.0  # KOSIS season은 ±5% 수준 → 3x 증폭

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


# === 매출 스케일 매핑 (BRL 분포 → 한국 만원 분포) ===
def donor_to_korean_mu(donor_log_rev: float, pool_log_mean: float,
                        pool_log_std: float, rng: np.random.Generator) -> float:
    """donor의 매출 percentile 위치를 보존하면서 한국 시장 스케일로 변환."""
    z = (donor_log_rev - pool_log_mean) / max(pool_log_std, 0.01)
    z = float(np.clip(z, -3, 3))
    return float(np.exp(KOREAN_LOG_MEAN + z * KOREAN_LOG_STD + 0.1 * rng.standard_normal()))


# === 핵심: donor → 한국 합성 셀러 1명 생성 ===
def generate_seller(donor: pd.Series, pool_stats: dict,
                    kosis_trend: np.ndarray, kosis_season: np.ndarray,
                    naver_cov: np.ndarray, promo: np.ndarray,
                    rng: np.random.Generator) -> tuple[np.ndarray, float]:
    base_mu = donor_to_korean_mu(donor["log_avg_rev"], pool_stats["log_mean"],
                                  pool_stats["log_std"], rng)

    cv = float(donor["cv"])
    ar1 = float(donor["ar1"])
    season_strength = float(donor["seasonality"])  # |ACF lag-12|, [0, 1]
    spike_ratio = float(donor["spike_ratio"])
    zero_ratio = float(donor["zero_ratio"])
    trend_per_month = (float(donor["trend_slope"]) / N_MONTHS) * float(donor["trend_r2"])

    rev = np.zeros(N_MONTHS)
    noise_prev = 0.0

    for t in range(N_MONTHS):
        # 한국 시장 트렌드 + donor 개별 트렌드 + 계절성
        market_factor = 1.0 + kosis_trend[t]
        # donor의 계절 강도가 0이면 KOSIS season 30%만 반영, 1이면 100% 반영
        season_factor = 1.0 + kosis_season[t] * (0.3 + 0.7 * season_strength)
        donor_growth = 1.0 + trend_per_month * t

        base_t = base_mu * market_factor * season_factor * donor_growth

        # AR(1) 노이즈, donor의 CV로 스케일링 (NOISE_SCALE로 한국 수준에 맞춤)
        sigma = base_t * cv * NOISE_SCALE
        noise = ar1 * noise_prev + sigma * rng.standard_normal()
        noise_prev = noise

        # Spike (donor의 spike_ratio 큰 경우 드물게 발생)
        spike = 0.0
        if spike_ratio > 2.0 and rng.random() < 0.04:
            spike = base_t * (spike_ratio - 1) * rng.uniform(0.2, 0.5)

        # Korean promo boost (블프, 추석, 설) — v2.2: 중간값 (10% → 20%)
        # v2.1의 +30%는 KOSIS trend를 압도해 r=0.22로 떨어짐 → 균형값 +20%
        promo_boost = 0.20 * base_mu * promo[t]

        # Naver 외생 covariate — v2.2: 중간값 (±2.5% → 약 ±7-15%)
        # v2.1의 0.30은 KOSIS와 충돌 → 0.15로 균형 (외생 신호 의미있되 KOSIS trend 보존)
        naver_factor = 1.0 + 0.15 * (naver_cov[t] - 0.5)

        rev[t] = max((base_t + noise + spike + promo_boost) * naver_factor, 0.0)

        # Zero event (donor의 zero_ratio 비례, 0.6 강도 조절)
        if rng.random() < zero_ratio * 0.6:
            rev[t] = 0.0

    return rev, base_mu


# === 사후 라벨링 (v3 임계 룰) ===
def calc_synth_features(rev: np.ndarray) -> dict | None:
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
                seasonality=seasonality, zero_ratio=zero, density=density,
                mean_rev=mu)


def post_hoc_label(feats: dict | None) -> str:
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


# === 메인 ===
def main():
    rng = np.random.default_rng(SEED)

    print("[1/5] Donor pool 로드 (AR(1) 계산 포함)")
    donors = load_donor_pool()
    print(f"  donor 수: {len(donors)}")
    print(f"  AR(1) 평균={donors['ar1'].mean():.3f}, 중앙값={donors['ar1'].median():.3f}")

    print("\n[2/5] KOSIS / Naver prior 로드")
    kosis_trend, kosis_season, naver_cov = load_priors()
    print(f"  KOSIS trend range: {kosis_trend.min():+.4f} ~ {kosis_trend.max():+.4f}")
    print(f"  KOSIS season range: {kosis_season.min():+.4f} ~ {kosis_season.max():+.4f}")
    print(f"  Naver cov range: {naver_cov.min():.4f} ~ {naver_cov.max():.4f}")

    print(f"\n[3/5] 합성 셀러 생성 ({len(donors)} donor × {SAMPLES_PER_DONOR})")
    pool_stats = {
        "log_mean": float(donors["log_avg_rev"].mean()),
        "log_std": float(donors["log_avg_rev"].std()),
    }
    promo = make_promo_dummy()
    dates = pd.date_range(START_DATE, periods=N_MONTHS, freq="MS")

    records = []
    seller_meta = []
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
                    month_idx=t,
                    date=dates[t],
                    monthly_revenue=float(rev[t]),
                    type=label,
                    mu=base_mu,
                    promo=float(promo[t]),
                    naver_index=float(naver_cov[t]),
                ))
            seller_meta.append(dict(seller_id=sid, donor_id=donor["seller_id"],
                                    type=label, mu=base_mu))

    df = pd.DataFrame(records)
    meta = pd.DataFrame(seller_meta)
    print(f"  합성 셀러: {meta['seller_id'].nunique()} (총 {len(df)} 행)")
    print(f"  유형 분포:")
    print(meta["type"].value_counts().to_string())

    print("\n[4/5] KOSIS 검증")
    monthly_total = df.groupby("month_idx")["monthly_revenue"].sum().values
    monthly_norm = monthly_total / monthly_total.mean() - 1.0
    r, p = pearsonr(monthly_norm, kosis_trend)
    print(f"  Pearson r vs KOSIS trend = {r:+.4f}, p = {p:.4f}")

    print(f"\n[5/5] 저장 + 시각화")
    out_path = DATA / "cohort_kr_v2.parquet"
    df.to_parquet(out_path, index=False)
    print(f"  [save] {out_path}")

    validation = {
        "version": "v2",
        "method": "per-seller bootstrap",
        "n_sellers": int(meta["seller_id"].nunique()),
        "n_rows": int(len(df)),
        "n_months": N_MONTHS,
        "samples_per_donor": SAMPLES_PER_DONOR,
        "n_donors": int(len(donors)),
        "pearson_r_vs_kosis": float(r),
        "p_value": float(p),
        "type_distribution": meta["type"].value_counts().to_dict(),
        "rev_stats": {
            "mean": float(df["monthly_revenue"].mean()),
            "std": float(df["monthly_revenue"].std()),
            "min": float(df["monthly_revenue"].min()),
            "max": float(df["monthly_revenue"].max()),
            "p25": float(df["monthly_revenue"].quantile(0.25)),
            "p50": float(df["monthly_revenue"].quantile(0.5)),
            "p75": float(df["monthly_revenue"].quantile(0.75)),
        },
        "donor_pool_stats": pool_stats,
    }
    val_path = DATA / "cohort_kr_v2_validation.json"
    val_path.write_text(json.dumps(validation, indent=2, ensure_ascii=False))
    print(f"  [save] {val_path}")

    # 시각화: 4-panel diagnostics
    color_map = {
        "stable": "steelblue", "seasonal": "darkorange",
        "growth": "mediumseagreen", "volatile": "crimson",
        "decline": "gray", "other": "lightgray",
    }
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # (1) Type distribution
    ax = axes[0, 0]
    counts = meta["type"].value_counts()
    colors = [color_map.get(t, "gray") for t in counts.index]
    ax.bar(counts.index, counts.values, color=colors, edgecolor="white")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 5, str(v), ha="center", fontweight="bold")
    ax.set_title("v2 합성 셀러 유형 분포 (사후 라벨)")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(alpha=0.3)

    # (2) KOSIS shape vs synthetic aggregate
    ax = axes[0, 1]
    ax.plot(dates, kosis_trend, "o-", color="red", label="KOSIS trend (정규화)", linewidth=2)
    ax.plot(dates, monthly_norm, "s-", color="steelblue", label="v2 합성 합계 (정규화)", linewidth=2)
    ax.set_title(f"KOSIS vs v2 합성 합계 (Pearson r = {r:+.3f})")
    ax.legend()
    ax.tick_params(axis="x", rotation=30)
    ax.grid(alpha=0.3)

    # (3) Sample seller time series (4 sellers, different types)
    ax = axes[1, 0]
    for typ in ["stable", "seasonal", "growth", "volatile"]:
        sub = meta[meta["type"] == typ]
        if len(sub) == 0:
            continue
        sid = sub.iloc[0]["seller_id"]
        ts = df[df["seller_id"] == sid].sort_values("month_idx")
        ax.plot(ts["date"], ts["monthly_revenue"], "o-",
                color=color_map.get(typ, "gray"), label=f"{typ}", linewidth=2, alpha=0.8)
    ax.set_title("유형별 합성 셀러 샘플 시계열")
    ax.set_ylabel("월 매출 (만원)")
    ax.legend()
    ax.tick_params(axis="x", rotation=30)
    ax.grid(alpha=0.3)

    # (4) Revenue distribution
    ax = axes[1, 1]
    ax.hist(np.log1p(df["monthly_revenue"]), bins=60, color="steelblue", edgecolor="white", alpha=0.8)
    ax.set_xlabel("log(1 + 월매출) — 만원 단위")
    ax.set_ylabel("빈도")
    ax.set_title("v2 매출 분포 (log scale)")
    ax.grid(alpha=0.3)

    plt.suptitle(f"v2 합성 코호트 진단 (n={meta['seller_id'].nunique()}, r={r:+.3f})",
                 fontsize=14, fontweight="bold", y=1.00)
    plt.tight_layout()
    fig_path = DATA / "cohort_kr_v2_diagnostics.png"
    plt.savefig(fig_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  [save] {fig_path}")

    print("\n=== 완료 ===")


if __name__ == "__main__":
    main()
