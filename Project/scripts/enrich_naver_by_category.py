"""A-1: 카테고리별 Naver 트렌드 매칭 — cohort_kr_v3 → v4

문제 진단:
  cohort_kr_v3.parquet의 naver_index는 모든 셀러에 동일한 글로벌 시그널
  (같은 month에서 std=0). Prophet v3가 v2보다 나빠진 원인.

해결:
  KOSIS 19 카테고리 ↔ Naver 11 카테고리 매핑 후
  각 셀러의 kosis_category에 맞는 naver 트렌드로 교체.

매핑 정책:
  - 명확한 매핑이 있으면 1:1 매칭
  - 애매한 카테고리("기타", "사무·문구", "애완용품")는 11개 평균 사용
    → 향후 더 정밀하게 다시 매핑할 수 있음

산출:
  Data/cohort_kr_v4.parquet
  Data/kosis_to_naver_mapping.csv
  Data/cohort_kr_v4_validation.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/eoseungyun/Desktop/project/SW_Capstone/Project")
DATA = ROOT / "Data"

# KOSIS 19 → Naver 11 매핑
# Naver 카테고리: 패션의류, 패션잡화, 화장품/미용, 디지털/가전, 가구/인테리어,
#                출산/육아, 식품, 스포츠/레저, 생활/건강, 여가/생활편의, 도서
KOSIS_TO_NAVER = {
    "생활용품": "생활/건강",
    "가구": "가구/인테리어",
    "스포츠·레저용품": "스포츠/레저",
    "화장품": "화장품/미용",
    "아동·유아용품": "출산/육아",
    "컴퓨터 및 주변기기": "디지털/가전",
    "가방": "패션잡화",
    "자동차 및 자동차용품": None,        # 애매 → 평균
    "가전·전자": "디지털/가전",
    "통신기기": "디지털/가전",
    "애완용품": None,                   # 애매 → 평균
    "기타": None,                       # 명시적 평균
    "서적": "도서",
    "패션용품 및 액세서리": "패션잡화",
    "사무·문구": None,                   # 애매 → 평균
    "음·식료품": "식품",
    "의복": "패션의류",
    "신발": "패션잡화",
    "농축수산물": "식품",
}


def load_naver_monthly() -> pd.DataFrame:
    nv = pd.read_csv(DATA / "naver" / "naver_monthly.csv")
    nv["date"] = pd.to_datetime(nv["date"])
    return nv


def normalize_per_category(nv: pd.DataFrame) -> pd.DataFrame:
    """카테고리별 min-max 정규화 (0~1). 셀러간 비교 가능하도록."""
    out = nv.copy()
    cat_cols = [c for c in nv.columns if c != "date"]
    for c in cat_cols:
        x = nv[c].values.astype(float)
        x_min, x_max = x.min(), x.max()
        out[c] = (x - x_min) / (x_max - x_min + 1e-9)
    return out


def build_seller_naver_lookup(nv_norm: pd.DataFrame) -> dict:
    """date → {naver_category: normalized_value} dict. NaN은 자동 제외."""
    lookup = {}
    cat_cols = [c for c in nv_norm.columns if c != "date"]
    for _, row in nv_norm.iterrows():
        d = row["date"]
        lookup[d] = {c: row[c] for c in cat_cols}
        vals = np.array([row[c] for c in cat_cols], dtype=float)
        lookup[d]["_mean"] = float(np.nanmean(vals))   # NaN 제외 평균
    return lookup


def enrich(v3: pd.DataFrame, nv_norm: pd.DataFrame) -> pd.DataFrame:
    """v3에 카테고리별 naver_index 다시 채우기.

    정책: naver_cat 값이 NaN(예: 도서 2024-08~)이면 평균으로 fallback.
    """
    v3 = v3.copy()
    v3["date"] = pd.to_datetime(v3["date"])
    lookup = build_seller_naver_lookup(nv_norm)

    fallback_count = 0

    def get_naver(row):
        nonlocal fallback_count
        d = row["date"]
        kosis_cat = row["kosis_category"]
        naver_cat = KOSIS_TO_NAVER.get(kosis_cat)
        if d not in lookup:
            return np.nan
        if naver_cat is None:
            return lookup[d]["_mean"]
        val = lookup[d][naver_cat]
        if pd.isna(val):                # 카테고리값 자체가 NaN이면 평균 fallback
            fallback_count += 1
            return lookup[d]["_mean"]
        return val

    v3["naver_index"] = v3.apply(get_naver, axis=1)
    if fallback_count > 0:
        print(f"  ⚠ NaN fallback (카테고리값 결측 → 평균 사용): {fallback_count} rows")
    return v3


def main():
    print("[1/5] cohort_kr_v3 + Naver 원본 로드")
    v3 = pd.read_parquet(DATA / "cohort_kr_v3.parquet")
    nv = load_naver_monthly()
    print(f"  v3 rows: {len(v3)}, sellers: {v3['seller_id'].nunique()}")
    print(f"  Naver: {len(nv)} months × {len(nv.columns)-1} categories")

    print("\n[2/5] Naver 카테고리별 0~1 정규화")
    nv_norm = normalize_per_category(nv)

    print("\n[3/5] KOSIS → Naver 매핑 적용")
    mapping_df = pd.DataFrame([
        {"kosis_category": k, "naver_category": v or "(평균)",
         "n_sellers": int((v3["kosis_category"] == k).groupby(v3["seller_id"]).any().sum())}
        for k, v in KOSIS_TO_NAVER.items()
    ])
    print(mapping_df.to_string(index=False))
    mapping_df.to_csv(DATA / "kosis_to_naver_mapping.csv", index=False)

    print("\n[4/5] cohort에 셀러별 naver_index 적용")
    v4 = enrich(v3, nv_norm)

    # 검증: 같은 month에서 naver_index std > 0 이어야 함
    sample_month = v4["date"].iloc[0]
    sub = v4[v4["date"] == sample_month][["kosis_category", "naver_index"]]
    print(f"\n  같은 month ({sample_month.date()}) 카테고리별 naver_index:")
    print(sub.groupby("kosis_category")["naver_index"].first().to_string())

    overall_std = v4.groupby("date")["naver_index"].std().mean()
    print(f"\n  같은 month 내 셀러간 naver_index std (전체 평균): {overall_std:.4f}")
    print(f"  (v3에서는 0, v4에서는 > 0 이어야 매칭 성공)")

    print("\n[5/5] v4 저장")
    v4.to_parquet(DATA / "cohort_kr_v4.parquet", index=False)
    print(f"  saved: cohort_kr_v4.parquet ({len(v4)} rows)")

    validation = {
        "source": "cohort_kr_v3 + naver_monthly per-category normalize",
        "n_rows": int(len(v4)),
        "n_sellers": int(v4["seller_id"].nunique()),
        "n_kosis_categories": int(v4["kosis_category"].nunique()),
        "mapping_explicit": sum(1 for v in KOSIS_TO_NAVER.values() if v is not None),
        "mapping_average_fallback": sum(1 for v in KOSIS_TO_NAVER.values() if v is None),
        "naver_index_intra_month_std_mean": float(overall_std),
        "naver_index_min": float(v4["naver_index"].min()),
        "naver_index_max": float(v4["naver_index"].max()),
    }
    (DATA / "cohort_kr_v4_validation.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False))

    print("\n=== A-1 완료 ===")
    print("다음: Prophet v4 학습 (cohort_kr_v4 사용)")


if __name__ == "__main__":
    main()
