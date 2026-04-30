"""KOSIS 통계표 검색 — '온라인쇼핑' 키워드로 유효한 tblId 찾기."""
from __future__ import annotations

import json
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
API_KEY = os.getenv("KOSIS_API_KEY")


def try_endpoint(name, url, params):
    print(f"\n=== {name} ===")
    print(f"URL: {url}")
    print(f"params: { {k:v for k,v in params.items() if k!='apiKey'} }")
    r = requests.get(url, params=params, timeout=30)
    print(f"status={r.status_code} len={len(r.text)}")
    print(f"head: {r.text[:500]}")
    try:
        data = json.loads(r.text)
        if isinstance(data, list):
            print(f"[list len={len(data)}]")
            for d in data[:5]:
                print("  ", d)
        elif isinstance(data, dict):
            print(f"[dict keys={list(data.keys())[:10]}]")
    except Exception as e:
        print("parse err:", e)


# 1. statisticsSearch with keyword
try_endpoint(
    "statisticsSearch — 온라인쇼핑",
    "https://kosis.kr/openapi/statisticsSearch.do",
    {
        "method": "getList",
        "apiKey": API_KEY,
        "searchNm": "온라인쇼핑동향",
        "format": "json",
        "jsonVD": "Y",
    },
)

# 2. Try statisticsList with vwCd search
try_endpoint(
    "statisticsList — 주제별 root",
    "https://kosis.kr/openapi/statisticsList.do",
    {
        "method": "getList",
        "apiKey": API_KEY,
        "vwCd": "MT_ZTITLE",
        "format": "json",
        "jsonVD": "Y",
    },
)
