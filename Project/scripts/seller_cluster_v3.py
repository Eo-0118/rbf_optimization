"""판매자 유형 클러스터링 v3 — 분리도 개선

v2 대비 변경:
1. 활성 구간 >= 12개월 필터 (1년 미만 셀러 제외, 651명 남음)
2. seasonality 재정의: ACF lag-12 (12개월 자기상관) — 기존 std-ratio 방식이 1.0에 수렴하던 문제 해결
3. K=3,4,5,6 silhouette 비교 후 최적 K 선택
4. 라벨 매핑: 불안정 -> 성장 -> 계절 -> 안정 우선순위, CV/trend/season으로 명시 분기
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

DATA = Path("/Users/eoseungyun/Desktop/project/SW_Capstone/Project/Data")
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

MIN_ACTIVE_MONTHS = 12


def acf_at_lag(ts: np.ndarray, lag: int) -> float:
    if len(ts) <= lag:
        return 0.0
    a, b = ts[:-lag], ts[lag:]
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def calc_features(monthly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sid, grp in monthly.groupby("seller_id"):
        grp = grp.sort_values("ym").reset_index(drop=True)
        ts_full = grp["monthly_revenue"].values
        nonzero = np.where(ts_full > 0)[0]
        if len(nonzero) < 2:
            continue
        first, last = nonzero[0], nonzero[-1]
        ts = ts_full[first:last + 1]
        n_active = len(ts)
        if n_active < MIN_ACTIVE_MONTHS:
            continue

        n_nonzero = len(nonzero)
        mu = ts.mean()
        sd = ts.std()
        cv = sd / (mu + 1e-8)
        zero_ratio = (ts == 0).mean()
        density = n_nonzero / n_active

        x = np.arange(n_active)
        if sd > 0:
            slope, _, r, _, _ = linregress(x, ts)
            trend_slope = slope / (mu + 1e-8) * n_active
            trend_r2 = r ** 2
        else:
            trend_slope = 0.0
            trend_r2 = 0.0

        acf12 = acf_at_lag(ts, 12)
        acf6 = acf_at_lag(ts, 6)
        seasonality = max(abs(acf12), 0.0)

        nonzero_vals = ts[ts > 0]
        spike = (ts.max() / (np.median(nonzero_vals) + 1e-8)) if len(nonzero_vals) else 1.0

        rows.append(dict(
            seller_id=sid,
            n_active=n_active,
            n_nonzero=n_nonzero,
            density=density,
            cv=cv,
            zero_ratio=zero_ratio,
            trend_slope=trend_slope,
            trend_r2=trend_r2,
            acf12=acf12,
            acf6=acf6,
            seasonality=seasonality,
            spike_ratio=spike,
            log_avg_rev=np.log1p(mu),
            mean_rev=mu,
        ))
    return pd.DataFrame(rows)


def label_clusters(profile: pd.DataFrame) -> dict:
    """임계 기반 라벨링. 임계 미달 클러스터는 '기타'로 빠짐 (잘못된 라벨 방지).

    임계:
      Volatile: CV >= 1.2 AND zero >= 0.3
      Growth  : trend >= 0.3 AND r2 >= 0.2
      Stable  : CV < 1.0 AND density >= 0.7
      Seasonal: season(|ACF lag-12|) >= 0.4 AND not labeled above
    """
    remaining = set(profile.index)
    out = {}

    cand = profile.loc[list(remaining)]
    mask = (cand["cv"] >= 1.2) & (cand["zero"] >= 0.3)
    if mask.any():
        c = (cand.loc[mask, "cv"] + cand.loc[mask, "zero"]).idxmax()
        out[c] = "불안정형 (Volatile)"
        remaining.discard(c)

    cand = profile.loc[list(remaining)]
    mask = (cand["trend"] >= 0.3) & (cand["r2"] >= 0.2)
    if mask.any():
        c = (cand.loc[mask, "trend"] * cand.loc[mask, "r2"]).idxmax()
        out[c] = "성장형 (Growth)"
        remaining.discard(c)

    cand = profile.loc[list(remaining)]
    mask = (cand["cv"] < 1.0) & (cand["density"] >= 0.7)
    if mask.any():
        score = -cand.loc[mask, "cv"] + cand.loc[mask, "density"]
        c = score.idxmax()
        out[c] = "안정형 (Stable)"
        remaining.discard(c)

    cand = profile.loc[list(remaining)]
    mask = cand["season"] >= 0.4
    if mask.any():
        c = cand.loc[mask, "season"].idxmax()
        out[c] = "계절형 (Seasonal)"
        remaining.discard(c)

    for c in remaining:
        prof = profile.loc[c]
        if prof["trend"] < -0.2:
            out[c] = "쇠퇴형 (Decline)"
        else:
            out[c] = f"기타 (Other-{c})"
    return out


def main():
    print("[load] monthly_seller_revenue.csv")
    monthly = pd.read_csv(DATA / "monthly_seller_revenue.csv")
    monthly["ym"] = pd.to_datetime(monthly["year_month"])
    n_total = monthly["seller_id"].nunique()
    print(f"  총 셀러: {n_total}")

    print(f"[feat] 활성 구간 >= {MIN_ACTIVE_MONTHS}개월 필터 + ACF 기반 seasonality 계산")
    sf = calc_features(monthly)
    print(f"  필터 후: {len(sf)} 명 ({len(sf)/n_total*100:.1f}%)")

    feat_cols = ["cv", "trend_slope", "trend_r2", "seasonality",
                 "zero_ratio", "density", "spike_ratio", "log_avg_rev"]
    X = sf[feat_cols].fillna(0).values
    Xs = StandardScaler().fit_transform(X)

    print("\n[K 비교]")
    results = {}
    for K in [3, 4, 5, 6]:
        km = KMeans(n_clusters=K, random_state=42, n_init=20)
        labels = km.fit_predict(Xs)
        sil = silhouette_score(Xs, labels)
        sil_s = silhouette_samples(Xs, labels)
        neg_ratio = (sil_s < 0).mean()
        results[K] = dict(km=km, labels=labels, sil=sil, neg=neg_ratio)
        print(f"  K={K}: silhouette={sil:.4f}, 음수 silhouette 비율={neg_ratio*100:.1f}%")

    best_K = max(results.keys(), key=lambda k: results[k]["sil"])
    print(f"\n[선택] K={best_K} (silhouette {results[best_K]['sil']:.4f})")

    labels = results[best_K]["labels"]
    sf["cluster"] = labels

    profile = sf.groupby("cluster").agg(
        cv=("cv", "mean"),
        trend=("trend_slope", "mean"),
        r2=("trend_r2", "mean"),
        season=("seasonality", "mean"),
        zero=("zero_ratio", "mean"),
        density=("density", "mean"),
        log_rev=("log_avg_rev", "mean"),
        n=("seller_id", "count"),
    ).round(3)
    print("\n[클러스터 프로파일]")
    print(profile.to_string())

    label_map = label_clusters(profile)
    print("\n[라벨 매핑]")
    for c in sorted(label_map.keys()):
        print(f"  cluster {c} -> {label_map[c]}")

    sf["seller_type"] = sf["cluster"].map(label_map)
    print("\n[유형별 셀러 수]")
    print(sf["seller_type"].value_counts().to_string())

    sil_samples = silhouette_samples(Xs, labels)
    print("\n[유형별 silhouette]")
    for typ in sf["seller_type"].unique():
        mask = sf["seller_type"] == typ
        s = sil_samples[mask]
        print(f"  {typ:25s}  mean={s.mean():+.3f}  음수={ (s<0).mean()*100:.1f}%  n={mask.sum()}")

    print("\n[v2 vs v3 비교]")
    print(f"  v2: 1814 sellers, silhouette=0.2395, 성장형 22.6% / 불안정형 15.8% 음수")
    print(f"  v3: {len(sf)} sellers, silhouette={results[best_K]['sil']:.4f}")

    sf.to_csv(DATA / "seller_features_v3.csv", index=False)
    print(f"\n[save] seller_features_v3.csv")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    Ks = sorted(results.keys())
    sils = [results[k]["sil"] for k in Ks]
    ax = axes[0, 0]
    ax.plot(Ks, sils, "o-", color="steelblue", linewidth=2)
    ax.axvline(best_K, color="red", linestyle="--", alpha=0.7, label=f"선택 K={best_K}")
    ax.set_title("K별 Silhouette")
    ax.set_xlabel("K"); ax.set_ylabel("Silhouette"); ax.legend(); ax.grid(alpha=0.3)

    color_map = {
        "안정형 (Stable)": "steelblue",
        "성장형 (Growth)": "mediumseagreen",
        "계절형 (Seasonal)": "darkorange",
        "불안정형 (Volatile)": "crimson",
    }
    pca = PCA(n_components=2, random_state=42)
    P = pca.fit_transform(Xs)
    ax = axes[0, 1]
    for typ, color in color_map.items():
        mask = sf["seller_type"] == typ
        if mask.sum():
            ax.scatter(P[mask, 0], P[mask, 1], c=color, label=typ, alpha=0.6, s=22)
    ax.set_title(f"PCA (PC1={pca.explained_variance_ratio_[0]*100:.1f}%, "
                 f"PC2={pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[0, 2]
    counts = sf["seller_type"].value_counts()
    colors_bar = [color_map.get(t, "gray") for t in counts.index]
    ax.bar(counts.index, counts.values, color=colors_bar, edgecolor="white")
    ax.set_title("유형별 셀러 수")
    ax.tick_params(axis="x", rotation=15)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 1, str(v), ha="center", fontweight="bold")

    order = [t for t in ["안정형 (Stable)", "성장형 (Growth)", "계절형 (Seasonal)", "불안정형 (Volatile)"]
             if t in sf["seller_type"].values]
    plot_feats = [("cv", "CV (변동계수)"),
                  ("trend_slope", "Trend Slope (정규화)"),
                  ("seasonality", "Seasonality (|ACF lag-12|)")]
    for i, (col, title) in enumerate(plot_feats):
        ax = axes[1, i]
        data_lst = [sf[sf["seller_type"] == t][col].values for t in order]
        bp = ax.boxplot(data_lst, labels=[t.split(" ")[0] for t in order], patch_artist=True)
        for patch, t in zip(bp["boxes"], order):
            patch.set_facecolor(color_map.get(t, "gray"))
            patch.set_alpha(0.7)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=10)
        ax.grid(alpha=0.3)

    plt.suptitle(
        f"클러스터링 v3 진단 (n={len(sf)}, K={best_K}, silhouette={results[best_K]['sil']:.3f})",
        fontsize=14, fontweight="bold", y=1.00,
    )
    plt.tight_layout()
    plt.savefig(DATA / "clustering_v3_diagnostics.png", dpi=130, bbox_inches="tight")
    plt.close()
    print(f"[save] clustering_v3_diagnostics.png")

    print("\n=== 완료 ===")


if __name__ == "__main__":
    main()
