"""외생변수 검증: KOSIS 카테고리별 매출 ↔ Naver 카테고리별 검색 트렌드 상관

목적:
  현재 합성 데이터에서 매출-Naver 상관이 0.002 (사실상 무관)였음.
  원인이 (a) 외생변수 자체가 무의미 vs (b) 합성 process의 약한 주입 중 어느 것인지 검증.

  → 한국 원본 데이터 (KOSIS, Naver)에서 매출-트렌드 상관이 실제로 존재하는지 확인.

산출:
  Data/exog_correlation_validation.json
  Data/exog_correlation_validation.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

ROOT = Path("/Users/eoseungyun/Desktop/project/SW_Capstone/Project")
DATA = ROOT / "Data"
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

# KOSIS 24 → Naver 11 매핑 (cohort_kr_v4와 동일 정책)
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
    # 아래는 의도적으로 제외 (매핑 모호): 자동차, 애완용품, 기타, 사무·문구, 통신기기 등
}


def load_data():
    ks = pd.read_parquet(DATA / "kosis" / "kr_trend_season.parquet")
    ks["date"] = pd.to_datetime(ks["date"])

    nv = pd.read_csv(DATA / "naver" / "naver_monthly.csv")
    nv["date"] = pd.to_datetime(nv["date"])
    nv = nv.set_index("date")
    return ks, nv


def compute_correlations(ks: pd.DataFrame, nv: pd.DataFrame) -> list[dict]:
    rows = []
    for kosis_cat, naver_cat in KOSIS_TO_NAVER.items():
        sub_ks = ks[ks["category"] == kosis_cat].sort_values("date").copy()
        if len(sub_ks) == 0:
            continue
        if naver_cat not in nv.columns:
            continue

        merged = sub_ks.set_index("date")[["value", "trend", "season_rel"]].join(
            nv[[naver_cat]].rename(columns={naver_cat: "naver"}), how="inner"
        ).dropna()

        if len(merged) < 24:
            rows.append(dict(kosis=kosis_cat, naver=naver_cat,
                             n=len(merged), note="기간 부족"))
            continue

        # (a) 매출 vs Naver — raw
        r0, p0 = pearsonr(merged["value"].values, merged["naver"].values)
        # (b) 매출 1차 차분 (계절성/추세 제거 효과) vs Naver 1차 차분
        v_diff = merged["value"].diff().dropna()
        n_diff = merged["naver"].diff().dropna()
        r_diff, p_diff = pearsonr(v_diff.values, n_diff.values)
        # (c) detrend된 매출 (resid에서) vs Naver
        # resid는 STL 분해 잔차 — Naver 변동과 직접 비교 가능
        ks_resid = sub_ks.set_index("date")["value"].values - sub_ks.set_index("date")["trend"].values
        ks_resid_s = pd.Series(ks_resid, index=sub_ks["date"].values)
        merged_r = merged[["naver"]].join(ks_resid_s.to_frame("resid"), how="inner").dropna()
        r_resid, p_resid = pearsonr(merged_r["resid"].values, merged_r["naver"].values)

        # Lag: Naver가 1~3개월 선행할 때 매출과 상관
        lag_results = {}
        for lag in [-2, -1, 0, 1, 2, 3]:  # 음수 = Naver가 매출보다 뒤, 양수 = 앞
            if lag == 0:
                lag_results[lag] = float(r0)
                continue
            n_shift = merged["naver"].shift(lag)
            tmp = pd.concat([merged["value"], n_shift], axis=1).dropna()
            if len(tmp) < 24:
                lag_results[lag] = None
                continue
            r_l, _ = pearsonr(tmp.iloc[:, 0].values, tmp.iloc[:, 1].values)
            lag_results[lag] = float(r_l)

        rows.append(dict(
            kosis=kosis_cat, naver=naver_cat, n=len(merged),
            r_raw=float(r0), p_raw=float(p0),
            r_diff=float(r_diff), p_diff=float(p_diff),
            r_resid=float(r_resid), p_resid=float(p_resid),
            lag=lag_results,
        ))
    return rows


def main():
    print("[1/3] KOSIS + Naver 로드")
    ks, nv = load_data()
    print(f"  KOSIS: {len(ks['category'].unique())} 카테고리, {len(ks['date'].unique())} months")
    print(f"  Naver: {nv.shape[1]} 카테고리, {len(nv.index)} months")

    print(f"\n[2/3] 매핑 쌍 {len(KOSIS_TO_NAVER)}개 상관 계산")
    rows = compute_correlations(ks, nv)

    print(f"\n[3/3] 결과 ({len(rows)} 쌍)")
    print(f"\n{'KOSIS':22s} → {'Naver':14s} | n  | r_raw(p)        | r_diff(p)       | r_resid(p)      | best_lag")
    print("-" * 130)

    valid = [r for r in rows if "r_raw" in r]
    for r in valid:
        lags = {k: v for k, v in r["lag"].items() if v is not None}
        best_lag = max(lags, key=lambda k: abs(lags[k])) if lags else None
        bl_val = lags[best_lag] if best_lag is not None else None
        bl_str = f"lag={best_lag:+d}, r={bl_val:+.3f}" if best_lag is not None else "N/A"
        print(f"{r['kosis']:22s} → {r['naver']:14s} | {r['n']:2d} | "
              f"{r['r_raw']:+.3f}({r['p_raw']:.3f}) | "
              f"{r['r_diff']:+.3f}({r['p_diff']:.3f}) | "
              f"{r['r_resid']:+.3f}({r['p_resid']:.3f}) | {bl_str}")

    # 종합 통계
    if valid:
        r_raws = np.array([r["r_raw"] for r in valid])
        r_diffs = np.array([r["r_diff"] for r in valid])
        r_resids = np.array([r["r_resid"] for r in valid])
        sig_raw = sum(1 for r in valid if r["p_raw"] < 0.05)
        sig_diff = sum(1 for r in valid if r["p_diff"] < 0.05)
        sig_resid = sum(1 for r in valid if r["p_resid"] < 0.05)
        print(f"\n=== 종합 ===")
        print(f"  매핑 쌍: {len(valid)}")
        print(f"  raw 매출 vs Naver:   |r| 평균={np.abs(r_raws).mean():.3f}  유의(p<0.05): {sig_raw}/{len(valid)}")
        print(f"  diff 매출 vs Naver:  |r| 평균={np.abs(r_diffs).mean():.3f}  유의(p<0.05): {sig_diff}/{len(valid)}")
        print(f"  resid 매출 vs Naver: |r| 평균={np.abs(r_resids).mean():.3f}  유의(p<0.05): {sig_resid}/{len(valid)}")

    # 저장
    out = {
        "method": "KOSIS 24 카테고리 매출 ↔ Naver 11 카테고리 검색 트렌드 상관",
        "period": "2019-01 ~ 2024-12 (72개월)",
        "n_pairs": len(valid),
        "summary": {
            "abs_r_raw_mean": float(np.abs(r_raws).mean()) if valid else None,
            "abs_r_diff_mean": float(np.abs(r_diffs).mean()) if valid else None,
            "abs_r_resid_mean": float(np.abs(r_resids).mean()) if valid else None,
            "significant_raw": int(sig_raw) if valid else 0,
            "significant_diff": int(sig_diff) if valid else 0,
            "significant_resid": int(sig_resid) if valid else 0,
        } if valid else {},
        "pairs": rows,
    }
    (DATA / "exog_correlation_validation.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False, default=str))
    print(f"\n[save] exog_correlation_validation.json")

    # 시각화
    if valid:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        ax = axes[0, 0]
        ax.barh([r["kosis"] for r in valid], [r["r_raw"] for r in valid],
                color=["green" if r["p_raw"] < 0.05 else "lightgray" for r in valid])
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_xlabel("Pearson r (매출 raw)")
        ax.set_title("KOSIS 매출 vs Naver 검색 (raw)\n초록 = p<0.05")
        ax.grid(alpha=0.3)

        ax = axes[0, 1]
        ax.barh([r["kosis"] for r in valid], [r["r_diff"] for r in valid],
                color=["green" if r["p_diff"] < 0.05 else "lightgray" for r in valid])
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_xlabel("Pearson r (1차 차분)")
        ax.set_title("KOSIS 매출 vs Naver (1차 차분)\n계절성 제거 효과")
        ax.grid(alpha=0.3)

        ax = axes[1, 0]
        ax.barh([r["kosis"] for r in valid], [r["r_resid"] for r in valid],
                color=["green" if r["p_resid"] < 0.05 else "lightgray" for r in valid])
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_xlabel("Pearson r (STL 잔차)")
        ax.set_title("KOSIS 매출 잔차 vs Naver")
        ax.grid(alpha=0.3)

        # Lag analysis (top 3 매핑)
        ax = axes[1, 1]
        top3 = sorted(valid, key=lambda r: -abs(r["r_raw"]))[:5]
        for r in top3:
            lags = sorted(r["lag"].keys())
            vals = [r["lag"][k] for k in lags]
            ax.plot(lags, vals, "o-", label=f"{r['kosis']} ↔ {r['naver']}", alpha=0.7)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_xlabel("Lag (개월, 양수 = Naver 선행)")
        ax.set_ylabel("Pearson r")
        ax.set_title("Lag 분석 (top 5 매핑)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        plt.suptitle("외생변수 검증: KOSIS 매출 ↔ Naver 검색 트렌드 상관",
                     fontsize=13, fontweight="bold", y=1.00)
        plt.tight_layout()
        plt.savefig(DATA / "exog_correlation_validation.png", dpi=130, bbox_inches="tight")
        plt.close()
        print(f"[save] exog_correlation_validation.png")

    print("\n=== 검증 완료 ===")


if __name__ == "__main__":
    main()
