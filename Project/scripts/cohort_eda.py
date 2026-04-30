"""합성 코호트 EDA + KOSIS shape 비교 검증.

v4 Phase 1 (D6): 합성 4유형 각 10명 플롯 + KOSIS 집계 vs 합성 집계 Pearson r 확인.
Target: Pearson r ≥ 0.7 (v4 Section 12, Phase 6a).

출력:
    data/cohort_eda_samples.png        유형별 10명 시계열 플롯
    data/cohort_eda_aggregate.png      합성 집계 vs KOSIS shape 비교
    data/cohort_eda_stats.csv          유형별 요약 통계
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[1]
COHORT = ROOT / "data" / "cohort_kr.parquet"
KOSIS_TS = ROOT / "data" / "kosis" / "kr_trend_season.parquet"
OUT = ROOT / "data"

try:
    plt.rcParams["font.family"] = "AppleGothic"
    plt.rcParams["axes.unicode_minus"] = False
except Exception:
    pass


def plot_samples(df: pd.DataFrame, n_sample: int = 10) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    types = ["stable", "seasonal", "growth", "volatile"]
    for ax, typ in zip(axes.flat, types):
        sub = df[df["type"] == typ]
        sellers = sub["seller_id"].unique()
        chosen = np.random.choice(sellers, min(n_sample, len(sellers)), replace=False)
        for sid in chosen:
            s = sub[sub["seller_id"] == sid].sort_values("month_idx")
            ax.plot(s["month_idx"], s["monthly_revenue"], alpha=0.6, lw=1)
        ax.set_title(typ)
        ax.set_xlabel("month")
        ax.set_ylabel("revenue")
    fig.suptitle("Synthetic Cohort — Sample Sellers per Type", y=1.01)
    fig.tight_layout()
    fig.savefig(OUT / "cohort_eda_samples.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] cohort_eda_samples.png")


def compare_kosis(df: pd.DataFrame) -> None:
    # Aggregate synthetic: monthly total revenue
    synth_monthly = df.groupby("month_idx")["monthly_revenue"].sum().values

    # KOSIS: 합계 category last 24 months actual value
    kosis = pd.read_parquet(KOSIS_TS)
    total = kosis[kosis["category"] == "합계"].sort_values("date").tail(24)
    kosis_vals = total["value"].values

    # Normalize both to 0-1
    def norm(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-12)

    synth_n = norm(synth_monthly)
    n_compare = min(len(synth_n), len(kosis_vals))
    kosis_n = norm(kosis_vals[-n_compare:])
    synth_n = synth_n[-n_compare:]

    r, p = pearsonr(synth_n, kosis_n)
    print(f"\n[Pearson r] synth_agg vs KOSIS 합계: r={r:.4f}, p={p:.6f}")
    print(f"  → {'PASS' if r >= 0.7 else 'FAIL'} (threshold: r≥0.7)")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    months = np.arange(n_compare)
    axes[0].plot(months, kosis_n, "o-", label="KOSIS (normalized)", lw=2)
    axes[0].plot(months, synth_n, "s--", label="Synthetic agg (normalized)", lw=2)
    axes[0].set_title(f"Shape Comparison (Pearson r={r:.3f})")
    axes[0].legend()
    axes[0].set_xlabel("month index")
    axes[0].set_ylabel("normalized value")

    axes[1].scatter(kosis_n, synth_n, alpha=0.7)
    axes[1].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[1].set_xlabel("KOSIS (normalized)")
    axes[1].set_ylabel("Synthetic (normalized)")
    axes[1].set_title("Scatter: KOSIS vs Synthetic")

    fig.tight_layout()
    fig.savefig(OUT / "cohort_eda_aggregate.png", dpi=120)
    plt.close(fig)
    print(f"[save] cohort_eda_aggregate.png")
    return r


def type_stats(df: pd.DataFrame) -> None:
    stats = []
    for typ in ["stable", "seasonal", "growth", "volatile"]:
        sub = df[df["type"] == typ]
        sellers = sub.drop_duplicates("seller_id")
        rev = sub["monthly_revenue"]
        stats.append({
            "type": typ,
            "n_sellers": sellers.shape[0],
            "rev_mean": rev.mean(),
            "rev_std": rev.std(),
            "rev_cv": rev.std() / rev.mean(),
            "rev_min": rev.min(),
            "rev_max": rev.max(),
            "zero_pct": (rev == 0).mean() * 100,
            "m_i_mean": sellers["m_i"].mean(),
            "m_i_std": sellers["m_i"].std(),
        })
    sdf = pd.DataFrame(stats)
    sdf.to_csv(OUT / "cohort_eda_stats.csv", index=False)
    print(f"\n[save] cohort_eda_stats.csv")
    print(sdf.to_string(index=False))


def main():
    np.random.seed(42)
    df = pd.read_parquet(COHORT)
    print(f"[load] cohort: {df.shape}")

    plot_samples(df)
    r = compare_kosis(df)
    type_stats(df)

    print(f"\n=== D6 Verification ===")
    print(f"Pearson r = {r:.4f} → {'✅ PASS' if r >= 0.7 else '❌ FAIL'}")


if __name__ == "__main__":
    main()
