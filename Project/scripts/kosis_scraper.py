"""KOSIS 온라인쇼핑동향조사 스크래퍼.

타겟: DT_1KE10051 (온라인쇼핑몰 운영형태별/상품군별 거래액, 월)
v4 Layer 2 용도 — 카테고리별 월별 거래액 → STL로 trend_KR_t, season_KR_t 추출.

KOSIS 데이터 endpoint는 itmId가 ALL을 지원하지 않음. 실제 코드(예: T20=거래액)를 지정해야 함.
우선 itmId 후보를 순차로 시도하고, 성공 시 전체 데이터 저장.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
API_KEY = os.getenv("KOSIS_API_KEY")
assert API_KEY, "KOSIS_API_KEY missing from .env"

BASE = "https://kosis.kr/openapi/Param/statisticsParameterData.do"
OUT_DIR = ROOT / "data" / "kosis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TBL_TARGETS = [
    ("101", "DT_1KE10051", "운영형태별/상품군별 거래액"),
    ("101", "DT_1KE10071", "판매매체별/상품군별 거래액"),
]
ITM_CANDIDATES = ["T20", "T1", "T2", "T10", "13103114T2", "13103114T1"]


def fetch(org, tbl, itm, start="201901", end="202412"):
    params = {
        "method": "getList",
        "apiKey": API_KEY,
        "orgId": org,
        "tblId": tbl,
        "format": "json",
        "jsonVD": "Y",
        "prdSe": "M",
        "startPrdDe": start,
        "endPrdDe": end,
        "itmId": itm,
        "objL1": "ALL",
        "objL2": "ALL",
        "objL3": "",
        "objL4": "",
        "objL5": "",
        "objL6": "",
        "objL7": "",
        "objL8": "",
    }
    r = requests.get(BASE, params=params, timeout=60)
    return r.status_code, r.text


def try_combo(org, tbl, desc):
    print(f"\n== {tbl} ({desc}) ==")
    for itm in ITM_CANDIDATES:
        status, body = fetch(org, tbl, itm)
        head = body[:140].replace("\n", " ")
        print(f"  itmId={itm:15s} status={status} len={len(body):6d} head={head}")
        try:
            data = json.loads(body)
        except Exception:
            continue
        if isinstance(data, dict) and data.get("err"):
            continue
        if isinstance(data, list) and len(data) > 0:
            print(f"  [HIT] itmId={itm} records={len(data)}")
            return itm, data
    return None, None


def save(tbl, itm, data):
    df = pd.DataFrame(data)
    raw = OUT_DIR / f"kosis_{tbl}_{itm}_raw.json"
    raw.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    csv = OUT_DIR / f"kosis_{tbl}_{itm}.csv"
    df.to_csv(csv, index=False)
    print(f"  saved: {csv}  shape={df.shape}")
    print(f"  cols: {list(df.columns)}")
    print(df.head(3).to_string())


def main():
    any_hit = False
    for org, tbl, desc in TBL_TARGETS:
        itm, data = try_combo(org, tbl, desc)
        if data:
            save(tbl, itm, data)
            any_hit = True
    if not any_hit:
        print("\n[FAIL] 데이터 pull 실패. 다음 단계: itmId 메타 조회 필요.")
        sys.exit(1)


if __name__ == "__main__":
    main()
