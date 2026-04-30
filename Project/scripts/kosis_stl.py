"""KOSIS 카테고리별 월 거래액 → STL 분해 → trend_KR / season_KR prior 저장.

입력: data/kosis/kosis_DT_1KE10051_T20.csv  (5,544 rows)
    C1 = 상품군(category), C2 = 운영형태(00=계), PRD_DE = YYYYMM, DT = 거래액(백만원)

출력:
    data/kosis/kr_monthly_by_category.csv    wide pivot (PRD_DE × category)
    data/kosis/kr_trend_season.parquet        카테고리별 STL trend/season/resid
    data/kosis/kr_trend_season_summary.csv    요약 (카테고리별 season 진폭 등)
    data/kosis/kr_stl_plots.png               시각화
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.seasonal import STL

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "kosis" / "kosis_DT_1KE10051_T20.csv"
OUT = ROOT / "data" / "kosis"
OUT.mkdir(parents=True, exist_ok=True)


def load_totals() -> pd.DataFrame:
    df = pd.read_csv(SRC, dtype={"C1": str, "C2": str, "PRD_DE": str})
    tot = df[df["C2_NM"] == "계"].copy()
    tot["PRD_DE"] = tot["PRD_DE"].astype(str)
    tot["date"] = pd.to_datetime(tot["PRD_DE"], format="%Y%m")
    tot["value"] = pd.to_numeric(tot["DT"], errors="coerce")
    tot["category"] = tot["C1_NM"].astype(str)
    return tot[["date", "category", "value"]]


def pivot_wide(tot: pd.DataFrame) -> pd.DataFrame:
    w = tot.pivot_table(index="date", columns="category", values="value", aggfunc="sum")
    w = w.sort_index()
    w = w.asfreq("MS")
    return w


def stl_decompose(wide: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    summary = []
    for cat in wide.columns:
        s = wide[cat].dropna()
        if len(s) < 24:
            continue
        try:
            res = STL(s, period=12, robust=True).fit()
        except Exception as e:
            print(f"  [skip] {cat}: {e}")
            continue
        tr = res.trend
        se = res.seasonal
        rs = res.resid
        base = tr.mean() if tr.mean() != 0 else 1.0
        trend_rel = (tr - tr.iloc[0]) / base
        season_rel = se / base
        for dt in s.index:
            records.append(
                {
                    "date": dt,
                    "category": cat,
                    "value": s.loc[dt],
                    "trend": tr.loc[dt],
                    "season": se.loc[dt],
                    "resid": rs.loc[dt],
                    "trend_rel": trend_rel.loc[dt],
                    "season_rel": season_rel.loc[dt],
                }
            )
        summary.append(
            {
                "category": cat,
                "n_obs": len(s),
                "mean_value": float(s.mean()),
                "trend_slope_rel": float((tr.iloc[-1] - tr.iloc[0]) / base / max(len(s), 1)),
                "season_amplitude_rel": float((se.max() - se.min()) / base),
                "resid_std_rel": float(rs.std() / base),
            }
        )
    return pd.DataFrame(records), pd.DataFrame(summary)


def plot_selected(long_df: pd.DataFrame, categories: list[str], path: Path) -> None:
    n = len(categories)
    fig, axes = plt.subplots(n, 3, figsize=(15, 2.6 * n), sharex=True)
    if n == 1:
        axes = axes.reshape(1, -1)
    for i, cat in enumerate(categories):
        sub = long_df[long_df["category"] == cat].set_index("date").sort_index()
        axes[i, 0].plot(sub.index, sub["value"], lw=1)
        axes[i, 0].plot(sub.index, sub["trend"], lw=1.5, color="red")
        axes[i, 0].set_title(f"{cat} — value + trend")
        axes[i, 0].tick_params(axis="x", rotation=30)
        axes[i, 1].plot(sub.index, sub["season"], lw=1, color="green")
        axes[i, 1].axhline(0, color="k", lw=0.5)
        axes[i, 1].set_title("seasonal")
        axes[i, 1].tick_params(axis="x", rotation=30)
        axes[i, 2].plot(sub.index, sub["resid"], lw=1, color="gray")
        axes[i, 2].axhline(0, color="k", lw=0.5)
        axes[i, 2].set_title("residual")
        axes[i, 2].tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    print(f"[load] {SRC}")
    tot = load_totals()
    print(f"[rows] {len(tot)}  categories={tot['category'].nunique()}  "
          f"date_range={tot['date'].min().date()}~{tot['date'].max().date()}")

    wide = pivot_wide(tot)
    wide_path = OUT / "kr_monthly_by_category.csv"
    wide.to_csv(wide_path)
    print(f"[save] {wide_path}  shape={wide.shape}")

    long_df, summary = stl_decompose(wide)
    long_path = OUT / "kr_trend_season.parquet"
    long_df.to_parquet(long_path, index=False)
    sum_path = OUT / "kr_trend_season_summary.csv"
    summary.sort_values("season_amplitude_rel", ascending=False).to_csv(sum_path, index=False)
    print(f"[save] {long_path}  rows={len(long_df)}")
    print(f"[save] {sum_path}")

    print("\n[summary top 10 by season amplitude]")
    print(summary.sort_values("season_amplitude_rel", ascending=False).head(10).to_string(index=False))
    print("\n[summary top 10 by trend slope]")
    print(summary.sort_values("trend_slope_rel", ascending=False).head(10).to_string(index=False))

    # Plot a representative set: 합계, 의복, 음·식료품, 가전·전자·통신기기, 여행 및 교통서비스
    targets = [c for c in ["합계", "의복", "음·식료품", "가전·전자·통신기기", "여행 및 교통서비스"] if c in wide.columns]
    if targets:
        plot_path = OUT / "kr_stl_plots.png"
        plot_selected(long_df, targets, plot_path)
        print(f"[save] {plot_path}")


if __name__ == "__main__":
    main()
