# Next Steps — Phase 5+ 계획

> 작성: 2026-05-20 (Phase 4 완료, m_i=0.10 / L_personal=128만 기준)
> 전제: 현재 시계열 예측과 RBF 환경이 분리되어 있음. 보고서 원안([interim_report.md:14](interim_report.md))은 "시계열 예측 분포 → RBF state 통합"이지만 v1 MVP에서 단순화함.

---

## A. 시계열 예측 정확도 개선

### 현재 성능 (test 6개월)
| 모델 | WAPE mean | WAPE median | WAPE < 20% |
|---|---|---|---|
| Prophet v2 (외생 X) | 104% | 77% | **17.8%** |
| Prophet v3 (Naver+Promo) | 112% | 77% | 14.1% |
| LSTM v2 (글로벌) | 77% | 50% | 2.9% |

**v3 < v2 진단**: cohort_kr_v3의 `naver_index`가 모든 셀러에 동일한 글로벌 시그널(같은 month에서 std=0). 카테고리별로 분리 안 됨 → 외생 회귀계수가 의미 없음.

### 개선 옵션 (우선순위)

| # | 옵션 | 예상 효과 | 작업량 | 비고 |
|---|---|---|---|---|
| **1** | **카테고리별 Naver 트렌드 매칭** | 중-대 | 0.5일 | KOSIS 19 ↔ Naver 11 매핑. v3 회복 |
| **2** | **LSTM에 category embedding 추가** | 중 | 0.5일 | 현재 type_emb만. category_emb(19) 추가 |
| **3** | **분위수 예측 도입 (Prophet interval + LightGBM Quantile)** | 중 | 0.5~1일 | **분포 통합 준비** — 정확도 못 올려도 활용 가치 |
| 4 | 셀러별 best-of 앙상블 (Prophet vs LSTM) | 중 | 0.5일 | 유형별 우위 모델 다름 |
| 5 | lag/rolling feature + LightGBM | 중-대 | 1~1.5일 | 짧은 시계열에 강한 tabular 접근 |
| 6 | TFT / N-HiTS (PyTorch Forecasting) | 대 | 2~3일 | 보고서 원안. high-risk high-reward |
| 7 | Prophet yearly_seasonality 활성화 | 소 | 0.2일 | 24개월 학습 데이터로는 한계 |

**추천 진행**: 1 → 2 → 3까지 (1.5일) 끝나면 **분포 통합 준비** 완료.

---

## B. 시계열 → RBF State 통합

목표: 보고서 원안([interim_report.md:548](interim_report.md))대로 "**시계열 예측 분포를 RL state로 통합**".

### 통합 방식 (현재 정확도 수준 고려)

**(2) 분포 통합** 채택 — 정확도 낮아도 불확실성 정보로 의미 있음:
```python
# 기존 state (13-dim)
state = [t/T, recovery%, 최근3개월매출(3), type_oh(6), m_i, log_scale]

# 통합 후 state (19-dim 예시)
state = [..., R_hat_mean, R_hat_p10, R_hat_p90,
              R_hat_{t+1}_mean, R_hat_{t+2}_mean, R_hat_{t+3}_mean]
```

### 단계
1. 시계열 모델이 매월 시점 t에서 미래 N개월 분위수 예측 출력
2. RBF env가 그 예측을 state에 추가
3. PPO 재학습
4. 통합 전후 비교 (Phase 4 민감도 동일 그리드)

**작업량**: 1~1.5일 (env 수정 + PPO 재학습)

---

## C. D-4 / D-5 (이전 보류 작업)

시계열 통합과 **독립적** — 분리된 현 구조 위에서 진행 가능.

| # | 작업 | 작업량 | 비고 |
|---|---|---|---|
| D-4 | 카테고리 state input PPO 재학습 | 1일 | state에 category one-hot 추가. cohort v3 활용 |
| D-5 | 조건별 분할 평가 | 0.5일 | 유형/카테고리/노이즈별 PPO 성능 분석 |

---

## D. 웹페이지 (실제 운영 인터페이스)

**방향성 변화 아님 — 학습 vs 운영 패러다임을 구체화**.

### 구조
```
[입력]                       [처리]                    [출력]
새 셀러 정보         →  학습된 PPO inference   →    대출 가능성
- 매출 이력 (3~24개월)       + N회 몬테카를로 시뮬       - 가능/불가능
- 카테고리                   - 시계열 예측 분포 생성     - 예상 회수율
- (선택) m_i, 가구 정보       - 각 시나리오 PPO step     - 가계 침범 확률
                            - 회수율/디폴트 집계        - 추천 r_t 경로
```

### 기술 스택
- **Streamlit** (Python 단일 파일, 1-2일 구현)
- 학습된 PPO 모델 load + 시뮬레이션 N회
- 디스클레이머: "학술 연구용 시뮬레이션, 실제 금융 의사결정 근거 아님"

**작업량**: 1~2일 (B 완료 후)

---

## E. (조건부) Olist 데이터 대체 옵션

### 트리거 조건
**시계열 정확도가 너무 낮아 무의미한 경우** — A 옵션 1~5 시도 후에도 WAPE < 20% 비율이 30%를 못 넘기면 데이터 자체의 한계로 판단.

### 후보 데이터셋

| 후보 | 장점 | 단점 |
|---|---|---|
| **공정거래위원회 가맹사업 정보공개서** | 한국 직접 데이터, 카테고리 풍부 | 가맹점만, 이커머스 아님, 월 매출 단위 |
| **소상공인시장진흥공단 BizMap / SBIZ** | 한국 소상공인, 매출 추정치 제공 | 추정치라 노이즈 |
| **GS Shop / 11번가 / Coupang 공개 보고서** | 한국 이커머스 직접 | 셀러 단위 데이터 거의 없음 |
| **M5 Forecasting (Walmart, Kaggle)** | 30,490 SKU × 5년, 시계열 품질 ↑ | 미국 데이터, 한국 맥락 ↓ |
| **Amazon SP-API 공개 통계** | 글로벌 이커머스 표준 | 셀러 비공개, 메타데이터만 |
| **PSE 한국 e-commerce panel (가능 시 학교 통해 신청)** | 한국 셀러 panel | 접근성 불확실 |

### 추가 검토 방향
- **Olist 유지 + KOSIS 무게 ↑**: bootstrap 시 KOSIS 한국 prior 비중을 더 강하게
- **Olist + 한국 카테고리별 매출 분포 (KOSIS 24개월)** 결합 — donor는 유지하되 한국 통계로 보정 강화
- **합성 데이터 generative process 자체 재설계** — Olist donor 제거하고 KOSIS + Naver만으로 시계열 생성 (단순화)

### 결정 기준
A 옵션 1~5 완료 후 다음 체크리스트:
- [ ] WAPE < 20% 비율 30% 이상 도달? (Yes → Olist 유지)
- [ ] decline/volatile 유형 WAPE 80% 미만 도달? (No → 데이터 구조 문제)
- [ ] 카테고리별 매핑 정확도 충분? (No → Olist→KOSIS 매핑 한계)
- 3개 중 2개 이상 No → **Olist 대체 검토**

---

## F. 진행 순서 (제안)

```
[현재] Phase 4 완료
   ↓
A. 시계열 개선 (옵션 1~3, 1.5일)
   ↓
   체크포인트: WAPE < 20% 비율 개선 측정
   ├─ 충분히 개선 → B 진행
   └─ 불충분 → E. Olist 대체 검토
   ↓
B. 시계열 → RBF state 통합 + PPO 재학습 (1.5일)
   ↓
   비교: 분리 vs 통합 성능 (Phase 4 그리드 재사용)
   ↓
C. D-4 / D-5 (선택, 1.5일)
   ↓
D. 웹페이지 배포 (1~2일)
   ↓
최종 보고서 마무리
```

**총 예상 작업량**: 6~8일 (Olist 대체 시 +3~5일)
