"""네이버 데이터랩 쇼핑인사이트 — 카테고리별 월별 검색 트렌드 수집.

v4 Layer 2: 카테고리별 월 지수 → 합성 코호트 외생 covariate x_t.

API: POST https://openapi.naver.com/v1/datalab/shopping/categories
헤더: X-Naver-Client-Id, X-Naver-Client-Secret
바디:
    {
      "startDate": "YYYY-MM-DD",
      "endDate":   "YYYY-MM-DD",
      "timeUnit":  "month",
      "category": [{"name": "...", "param": ["<catId>"]}, ...]  (최대 3개)
    }

응답은 0~100 정규화된 월별 ratio 값.
최대 3개씩 배치, 결과를 wide CSV로 저장.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")
assert CLIENT_ID and CLIENT_SECRET, "NAVER_CLIENT_ID/SECRET missing in .env"

OUT = ROOT / "data" / "naver"
OUT.mkdir(parents=True, exist_ok=True)

URL = "https://openapi.naver.com/v1/datalab/shopping/categories"
START = "2019-01-01"
END = "2024-12-31"

# 네이버 쇼핑 1차 카테고리 (cid)
CATEGORIES = [
    ("패션의류", "50000000"),
    ("패션잡화", "50000001"),
    ("화장품/미용", "50000002"),
    ("디지털/가전", "50000003"),
    ("가구/인테리어", "50000004"),
    ("출산/육아", "50000005"),
    ("식품", "50000006"),
    ("스포츠/레저", "50000007"),
    ("생활/건강", "50000008"),
    ("여가/생활편의", "50000009"),
    ("도서", "50000011"),
]


def batch(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def call(cats: list[tuple[str, str]]) -> dict:
    body = {
        "startDate": START,
        "endDate": END,
        "timeUnit": "month",
        "category": [{"name": name, "param": [cid]} for name, cid in cats],
    }
    headers = {
        "X-Naver-Client-Id": CLIENT_ID,
        "X-Naver-Client-Secret": CLIENT_SECRET,
        "Content-Type": "application/json",
    }
    r = requests.post(URL, headers=headers, data=json.dumps(body), timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text}")
    return r.json()


def main():
    all_series = {}
    raw_dump = []
    for i, chunk in enumerate(batch(CATEGORIES, 3)):
        print(f"[batch {i+1}] {[c[0] for c in chunk]}")
        try:
            data = call(chunk)
        except Exception as e:
            print(f"  [ERROR] {e}")
            continue
        raw_dump.append(data)
        for r in data.get("results", []):
            name = r["title"]
            rows = r["data"]
            s = pd.Series(
                {pd.to_datetime(d["period"]): float(d["ratio"]) for d in rows},
                name=name,
            )
            all_series[name] = s
            print(f"  {name}: {len(s)} months, range={s.min():.1f}~{s.max():.1f}")
        time.sleep(0.3)

    if not all_series:
        print("[FAIL] 데이터 없음")
        return

    wide = pd.DataFrame(all_series).sort_index()
    wide.index.name = "date"
    csv_path = OUT / "naver_monthly.csv"
    wide.to_csv(csv_path)
    print(f"\n[save] {csv_path}  shape={wide.shape}")
    print(f"[range] {wide.index.min().date()} ~ {wide.index.max().date()}")
    print("\n[head]")
    print(wide.head().to_string())
    print("\n[summary]")
    print(wide.describe().T[["mean", "std", "min", "max"]].to_string())

    raw_path = OUT / "naver_raw.json"
    raw_path.write_text(json.dumps(raw_dump, ensure_ascii=False, indent=2))
    print(f"[save] {raw_path}")


if __name__ == "__main__":
    main()
