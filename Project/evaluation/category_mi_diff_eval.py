"""Day 12: 카테고리별 m_i 차별화 — 거절율 분석

배경:
  Day 11+/11++ 결과: m_i=0.10 단일값에서 거절율 87.7% (또는 R_p10에서 93.8%)
  한국 영세 셀러의 87%+가 RBF 부적합 — 사회적 가치 의문

가설:
  카테고리별 영업이익률 차별화로 거절율 낮출 수 있음:
  - 화장품, 패션: m_i 0.15~0.20 (마진 큼)
  - 가전·전자, 도서: m_i 0.05~0.08 (마진 작음)

m_i 출처 (추정, 본 연구 가정):
  - 한국 스마트스토어 셀러 가이드
  - 카테고리별 gross/operating margin 일반론
  - 공식 통계 부재 → 보고서 한계로 명시

산출:
  Data/category_mi_classification.csv
  Data/category_mi_summary.json
  Data/category_mi_analysis.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

ROOT = PROJECT_ROOT
DATA = ROOT / "Data"

# 카테고리별 m_i (추정, 본 연구 가정)
# 출처: 스마트스토어 셀러 가이드 + 일반 마진 추정. 공식 통계 부재.
CATEGORY_M_I = {
    "화장품": 0.20,                      # 고마진 (화장품 업계)
    "패션용품 및 액세서리": 0.15,
    "의복": 0.15,
    "아동·유아용품": 0.15,
    "가방": 0.12,
    "신발": 0.12,
    "스포츠·레저용품": 0.12,
    "애완용품": 0.12,
    "생활용품": 0.10,                    # 중마진 (기본값)
    "가구": 0.10,
    "음·식료품": 0.10,
    "사무·문구": 0.10,
    "자동차 및 자동차용품": 0.08,
    "컴퓨터 및 주변기기": 0.06,
    "가전·전자": 0.05,
    "통신기기": 0.05,
    "농축수산물": 0.05,                   # 저마진
    "서적": 0.03,
    "기타": 0.10,
}

DEFAULT_M_I = 0.10
L_PERSONAL = 128.21
T_MAX = 36
CAP_BASE = 1.10
L_MIN = 100.0


def compute_eligibility(R: float, m_i: float, seller_type: str, cv: float,
                          L_personal: float = L_PERSONAL,
                          T_max: int = T_MAX, cap_base: float = CAP_BASE,
                          L_min: float = L_MIN) -> dict:
    risk_premium = {"stable": -0.05, "growth": 0.0, "other": 0.05,
                     "seasonal": 0.10, "decline": 0.15, "volatile": 0.20}.get(seller_type, 0.05)
    if cv > 1.0:
        risk_premium += 0.05
    cap_star = cap_base + risk_premium

    monthly_safe = R * m_i - L_personal
    if monthly_safe <= 0:
        return dict(eligible=False, reject_reason="가계비 충당 불가",
                    L_star=0, cap_star=cap_star, monthly_safe=monthly_safe)
    L_max = T_max * monthly_safe / cap_star
    if L_max < L_min:
        return dict(eligible=False, reject_reason="L_max < 100만",
                    L_star=L_max, cap_star=cap_star, monthly_safe=monthly_safe)
    return dict(eligible=True, reject_reason=None,
                L_star=L_max, cap_star=cap_star, monthly_safe=monthly_safe)


def main():
    print("[1/4] cohort_kr_v3 로드 (카테고리 매핑 포함)")
    df = pd.read_parquet(DATA / "cohort_kr_v3.parquet")
    print(f"  {df['seller_id'].nunique()} 셀러, {len(df)} 행")

    sellers = []
    for sid, sdf in df.groupby("seller_id"):
        revs = sdf["monthly_revenue"].values
        revs_nz = revs[revs > 0]
        if len(revs_nz) == 0:
            continue
        R_mean = float(np.mean(revs_nz))
        R_p10 = float(np.percentile(revs_nz, 10))
        cv = float(np.std(revs) / max(R_mean, 1e-6))
        kosis_cat = sdf["kosis_category"].iloc[0]
        seller_type = sdf["type"].iloc[0]
        sellers.append(dict(
            seller_id=sid, type=seller_type, cv=cv,
            R_mean=R_mean, R_p10=R_p10,
            kosis_category=kosis_cat,
            m_i_uniform=DEFAULT_M_I,
            m_i_category=CATEGORY_M_I.get(kosis_cat, DEFAULT_M_I),
        ))
    meta_df = pd.DataFrame(sellers)
    print(f"  유효 셀러: {len(meta_df)}")

    print(f"\n[2/4] 두 가지 m_i 시나리오에서 (L*, cap*) 산출 (R_p10 보수 기준)")

    for scen_name, m_i_col in [("Uniform_m_i_0.10", "m_i_uniform"),
                                 ("Category_m_i", "m_i_category")]:
        results = []
        for _, row in meta_df.iterrows():
            r = compute_eligibility(row["R_p10"], row[m_i_col], row["type"], row["cv"])
            results.append(r)
        results_df = pd.DataFrame(results)
        meta_df[f"{scen_name}_eligible"] = results_df["eligible"].values
        meta_df[f"{scen_name}_L_star"] = results_df["L_star"].values
        meta_df[f"{scen_name}_cap_star"] = results_df["cap_star"].values
        meta_df[f"{scen_name}_reject_reason"] = results_df["reject_reason"].values

        n_elig = results_df["eligible"].sum()
        n_total = len(meta_df)
        print(f"  {scen_name}: 적합 {n_elig} ({n_elig/n_total*100:.1f}%) "
              f"/ 거절 {n_total-n_elig} ({(1-n_elig/n_total)*100:.1f}%)")

    meta_df.to_csv(DATA / "category_mi_classification.csv", index=False)
    print(f"  [save] category_mi_classification.csv")

    print(f"\n[3/4] 카테고리별 분석")
    by_cat = meta_df.groupby("kosis_category").agg(
        n=("seller_id", "count"),
        m_i_used=("m_i_category", "first"),
        uniform_eligible=("Uniform_m_i_0.10_eligible", "sum"),
        category_eligible=("Category_m_i_eligible", "sum"),
        R_mean_avg=("R_mean", "mean"),
    ).sort_values("category_eligible", ascending=False)
    by_cat["uniform_pct"] = (by_cat["uniform_eligible"] / by_cat["n"] * 100).round(1)
    by_cat["category_pct"] = (by_cat["category_eligible"] / by_cat["n"] * 100).round(1)
    by_cat["pct_delta"] = by_cat["category_pct"] - by_cat["uniform_pct"]

    print(by_cat.to_string())

    print(f"\n  [요약]")
    n_uniform = meta_df["Uniform_m_i_0.10_eligible"].sum()
    n_category = meta_df["Category_m_i_eligible"].sum()
    print(f"  Uniform m_i=0.10:  적합 {n_uniform}/{len(meta_df)} ({n_uniform/len(meta_df)*100:.1f}%)")
    print(f"  Category m_i (차별): 적합 {n_category}/{len(meta_df)} ({n_category/len(meta_df)*100:.1f}%)")
    delta_pct = (n_category - n_uniform) / len(meta_df) * 100
    print(f"  → 카테고리 차별화 효과: +{delta_pct:.1f}%p ({n_category - n_uniform:+d}명 추가 적합)")

    print(f"\n[4/4] 시각화")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. m_i 가정 차이
    ax = axes[0, 0]
    cats_sorted = by_cat.index.tolist()
    m_is = [CATEGORY_M_I.get(c, DEFAULT_M_I) for c in cats_sorted]
    bars = ax.barh(cats_sorted[::-1], m_is[::-1],
                    color=["mediumseagreen" if m >= 0.12 else
                           "khaki" if m >= 0.08 else "salmon" for m in m_is[::-1]])
    ax.axvline(DEFAULT_M_I, color="red", linestyle="--", label=f"기본 m_i={DEFAULT_M_I}")
    ax.set_xlabel("카테고리별 m_i 추정")
    ax.set_title("카테고리별 영업이익률 m_i 차별화")
    ax.legend(); ax.grid(alpha=0.3)

    # 2. 적합 셀러 수 비교
    ax = axes[0, 1]
    x = np.arange(len(cats_sorted))
    width = 0.35
    u_counts = by_cat["uniform_eligible"].values
    c_counts = by_cat["category_eligible"].values
    ax.barh(x - width/2, u_counts, width, label="Uniform m_i=0.10",
            color="lightcoral", alpha=0.8)
    ax.barh(x + width/2, c_counts, width, label="Category m_i",
            color="mediumseagreen", alpha=0.8)
    ax.set_yticks(x); ax.set_yticklabels(cats_sorted, fontsize=8)
    ax.set_xlabel("적합 셀러 수")
    ax.set_title("카테고리별 적합 셀러 수 비교 (R_p10 보수)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="x")

    # 3. 적합률 전후 비교
    ax = axes[1, 0]
    ax.bar(["Uniform\n(m_i=0.10)", "Category 차별\n(0.03~0.20)"],
            [n_uniform/len(meta_df)*100, n_category/len(meta_df)*100],
            color=["lightcoral", "mediumseagreen"], alpha=0.8)
    ax.text(0, n_uniform/len(meta_df)*100 + 0.3, f"{n_uniform}명",
             ha="center", fontweight="bold")
    ax.text(1, n_category/len(meta_df)*100 + 0.3, f"{n_category}명",
             ha="center", fontweight="bold")
    ax.set_ylabel("적합 셀러 비율 (%)")
    ax.set_title(f"적합률: {n_uniform/len(meta_df)*100:.1f}% → "
                 f"{n_category/len(meta_df)*100:.1f}% "
                 f"({delta_pct:+.1f}%p)")
    ax.grid(alpha=0.3, axis="y")

    # 4. 카테고리별 R_mean 분포
    ax = axes[1, 1]
    big_cats = by_cat.nlargest(8, "n").index.tolist()
    box_data = [meta_df[meta_df["kosis_category"] == c]["R_mean"].values
                for c in big_cats]
    bp = ax.boxplot(box_data, labels=big_cats, patch_artist=True)
    for patch, c in zip(bp["boxes"], big_cats):
        m = CATEGORY_M_I.get(c, DEFAULT_M_I)
        color = "mediumseagreen" if m >= 0.12 else "khaki" if m >= 0.08 else "salmon"
        patch.set_facecolor(color); patch.set_alpha(0.7)
    ax.axhline(L_PERSONAL / DEFAULT_M_I, color="red", linestyle="--",
                label=f"기준 임계 R={L_PERSONAL/DEFAULT_M_I:.0f}만 (m_i=0.10)")
    ax.set_ylabel("매출 R (만원)")
    ax.set_title("Top 8 카테고리별 매출 분포")
    ax.tick_params(axis="x", rotation=25, labelsize=8)
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

    plt.suptitle(
        f"Day 12: 카테고리별 m_i 차별화 — 거절율 {(1-n_uniform/len(meta_df))*100:.0f}% → "
        f"{(1-n_category/len(meta_df))*100:.0f}%",
        fontsize=13, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(DATA / "category_mi_analysis.png", dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  [save] category_mi_analysis.png")

    summary = {
        "config": {
            "default_m_i": DEFAULT_M_I,
            "category_m_i_map": CATEGORY_M_I,
            "L_personal": L_PERSONAL,
            "T_max": T_MAX, "cap_base": CAP_BASE,
            "R_used": "R_p10 (보수 산출)",
            "m_i_disclaimer": "스마트스토어 가이드 + 일반 마진 추정. 공식 통계 부재.",
        },
        "results": {
            "uniform": {
                "n_eligible": int(n_uniform),
                "rejection_pct": float((1 - n_uniform/len(meta_df))*100),
            },
            "category": {
                "n_eligible": int(n_category),
                "rejection_pct": float((1 - n_category/len(meta_df))*100),
            },
            "delta_pct": float(delta_pct),
        },
        "by_category": by_cat.reset_index().to_dict(orient="records"),
    }
    (DATA / "category_mi_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    print(f"  [save] category_mi_summary.json")
    print("\n=== Day 12 완료 ===")


if __name__ == "__main__":
    main()
