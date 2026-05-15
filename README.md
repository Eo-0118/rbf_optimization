# 시계열 예측 및 강화학습 기반 이커머스 RBF 최적화

> 한국 이커머스 판매자를 위한 **현실적·윤리적 매출 연동 동적 상환(Revenue-Based Financing) 정책** 연구
---

## 1. 핵심 문제

이커머스 판매자(특히 신용 이력이 짧은 씬파일러)의 금융 소외 문제를 **매출 연동 동적 상환(RBF)** 으로 해결한다. 기존 RBF의 한계:

- **과거 매출 기반 1회 결정 + 고정 비율 상환** → 미래 불확실성 미반영
- **사업체 차원만 고려** → 셀러 본인의 가계 생활비 침범 위험

본 연구는 **시계열 예측 + CVaR 정적 최적화 + 강화학습 동적 조정**을 결합하여 회수율과 셀러 가계 보호를 동시 개선하는 의사결정 시스템을 제안한다.

---

## 2. 4단계 파이프라인

### Phase 1: 데이터 수집 + 한국형 합성 코호트
- **KOSIS** 「온라인쇼핑동향조사」 (DT_1KE10051): 26개 상품군 × 72개월 (2019-01 ~ 2024-12) → STL 분해
- **네이버 데이터랩** 쇼핑 트렌드: 11개 카테고리 × 72개월 (외생 covariate)
- **Olist** 이커머스 데이터 (Kaggle): 3,095 셀러 분산 구조 학습 → 651명 donor pool
- **Per-seller 부트스트랩 합성**: 1,302명 한국형 합성 셀러 × 24개월 (KOSIS Pearson r=0.55, p<0.01)

### Phase 2: 시계열 예측 베이스라인
- **Prophet** 3변형 (외생 변수 포함/제외, robust 지표)
- **LSTM** 글로벌 모델 2변형 (PyTorch, weighted sampling)
- 평가 지표: MAPE, SMAPE, WAPE
- 결과: trend·seasonality 명확한 셀러는 Prophet 우위, 노이즈 큰 셀러는 LSTM 우위

### Phase 3: RBF 시뮬레이션 + 정책 최적화
- **Gymnasium 환경** (`envs/rbf_env.py`): State 13-dim, 연속 Action r_t ∈ [0.03, 0.25]
- **2-Tier Burden 모델**: 사업 차원(영업이익) + 가계 차원(L_personal)
  - L_personal_min = 1,720,612원 (통계청 「2024년 가계동향조사」 1인가구 평균 소비지출)
  - m_i = 0.25 (한국 자영업 영업이익률 추정)
- **CVaR 정적 최적화** (`optim/cvar_optimizer.py`): CVXPY로 셀러별 최적 r* 산출
- **PPO 강화학습** (`agents/train_ppo.py`): StableBaselines3, MPS 가속

### Phase 4: 비교 평가 + 민감도 분석
- **3-정책 비교**: Fixed-0.15 (회수 우선) vs CVaR (가계 보호 우선) vs PPO (동적 균형)
- **민감도 분석**: L_personal ±50%, m_i 변동에 대한 정책 robustness 검증

---

## 3. 핵심 결과

### 3-정책 비교 (평가 셀러 260명)

| 정책 | Recovery | Completion | Burden | HH 안전 셀러 % | HH 침범 개월/24 |
|---|---|---|---|---|---|
| Fixed-0.15 | 1.000 | **80%** | 0.118 | 2.3% | 18.8 |
| CVaR (정적) | 0.659 | 9% | **0.067** | 2.1% | 18.7 |
| **PPO (RL)** | 0.807 | 70% | 0.174 | **9.6%** | **10.0** |

### PPO의 차별화된 학습 전략

> **"침범 빈도 ↓ + 침범 시 강도 ↑"**

- **가계 안전 셀러 비율 4.2배 향상** (2.3% → 9.6%)
- **평균 가계 침범 개월 47% 감소** (18.8 → 10.0개월)
- → 단일 r로 불가능한 **셀러별 차별화 정책** 학습 = RL의 핵심 가치

### 민감도 분석 (35 가정 조합)
- PPO의 가계 침범 개월 변동 폭 가장 작음 (3-4개월)
- 모든 가정 조합에서 PPO 우위 일관 → **결과 robust**

---

## 4. 학술적 차별점

| 기존 RBF 연구 | 본 연구 |
|---|---|
| 회수율·디폴트율만 최적화 | + **셀러 가계 생활 보호** (2-tier burden) |
| 단일 burden 지표 | **2-tier burden** (사업/가계 분리) |
| 셀러 = 사업체 | **셀러 = 사업체 + 개인사업자 가구** |
| 정적 OR 동적 | **정적 CVaR + 동적 RL 하이브리드** |

---

## 5. 폴더 구조

```
SW_Capstone/
└── Project/
    ├── scripts/                  # 데이터 수집·합성 (KOSIS, Naver, Olist, v2/v3)
    ├── models/                   # 시계열 예측 (Prophet v1-v3, LSTM v1-v2)
    ├── envs/                     # RBF Gymnasium 환경 + 베이스라인 정책
    ├── optim/                    # CVXPY CVaR 정적 최적화
    ├── agents/                   # PPO 강화학습 (StableBaselines3)
    ├── evaluation/               # 3-정책 비교 + 민감도 분석
    ├── notebooks/                # Jupyter 분석 노트북
    ├── docs/                     # 설계 문서, 카테고리 매핑, 중간 보고서
    └── Data/                     # 원본·전처리·합성·결과 데이터
        ├── kosis/                # KOSIS 통계청 데이터
        ├── naver/                # 네이버 데이터랩
        ├── Olist_Data/           # Olist 원본
        └── archive_v1/           # 이전 버전 보관
```

---

## 6. 기술 스택

| 분야 | 도구 |
|---|---|
| **언어·환경** | Python 3.12, .venv |
| **시계열 예측** | Prophet, PyTorch (LSTM) |
| **최적화** | CVXPY (CVaR) |
| **강화학습** | Stable-Baselines3 (PPO), Gymnasium |
| **데이터** | pandas, numpy, scipy, scikit-learn |
| **시각화** | matplotlib |

---

## 7. 설치 + 실행

```bash
# 가상환경 생성
cd Project
python3.12 -m venv .venv
source .venv/bin/activate

# 의존성 설치
pip install -r requirements.txt

# .env 파일 작성 (KOSIS, Naver API 키)
cp .env.example .env
# 편집기로 .env 열어서 API 키 입력
```

### 주요 스크립트 실행

```bash
# Phase 1: 합성 코호트 생성
python -m scripts.korean_synth_gen_v2

# Phase 2: 시계열 예측 (Prophet, LSTM)
python -m models.prophet_baseline_v2
python -m models.lstm_baseline_v2

# Phase 3: RBF env + 정책 평가
python -m envs.baselines
python -m optim.cvar_optimizer
python -m agents.train_ppo

# Phase 4: 통합 평가 + 민감도 분석
python -m evaluation.compare_policies
python -m evaluation.sensitivity_analysis
```

---

## 8. 데이터 출처 (재현 가능)

| 데이터 | 출처 |
|---|---|
| **온라인쇼핑동향조사** | [KOSIS DT_1KE10051](https://kosis.kr/statHtml/statHtml.do?orgId=101&tblId=DT_1KE10051) (통계청) |
| **쇼핑 검색 트렌드** | [네이버 데이터랩 OpenAPI](https://developers.naver.com/docs/serviceapi/datalab/shopping/shopping.md) |
| **이커머스 거래** | [Olist Brazilian E-Commerce Public Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (Kaggle) |
| **1인가구 소비지출** | 통계청 「2024년 가계동향조사」 (1,720,612원) |

---

## 9. 진행 상태

- [x] **Phase 1**: 데이터 수집 + 합성 코호트 (per-seller bootstrap)
- [x] **Phase 2**: 시계열 예측 베이스라인 (Prophet, LSTM)
- [x] **Phase 3 v1**: RBF env + CVaR + PPO + 평가 (1인가구 가정)
- [x] **Phase 4**: 민감도 분석 (PPO robust 입증)
- [ ] **Phase 3 v2**: Olist 카테고리 통합 (진행 중)
- [ ] 최종 보고서 작성 (6월 예정)

---

## 10. 정직한 한계

본 연구는 학술적 정직성을 우선하며, 다음 한계를 명시한다:

1. **합성 데이터 의존**: 한국 셀러 실데이터 부재로 합성 코호트 사용. KOSIS 패턴 r=0.55로 시장 반영했으나 카테고리별 분기는 단순화
2. **1인 가구 단순화**: Phase 3 v1은 모든 셀러를 1인 자영업자로 가정. 가구 다양성 미반영 (한국 셀러의 매출-가구 동시 측정 데이터 부재)
3. **m_i 단일값 가정**: 카테고리별 영업이익률 차이 미반영 (Phase 3 v2에서 통합 예정)
4. **시계열 예측 절대 정확도 한계**: WAPE < 20% 셀러 비율 < 20% (단기·고노이즈 한계). Phase 3에서 예측 분포로 활용
5. **외생 변수 영향 강도 가정**: KOSIS r과 외생 신호 강도의 trade-off (균형값 채택)

---

## 11. 진행 일정

- 3~4월: 데이터 수집·전처리·합성 환경 설계
- 4~5월: 시계열 예측 모델 학습 (Prophet, LSTM)
- 5월: CVaR 최적화 + RL 에이전트 구현 + 민감도 분석
- 5~6월: 카테고리 통합 + 최종 비교 실험
- 6월: 최종 보고서

---

## License

본 프로젝트는 학술 연구 목적의 캡스톤 프로젝트입니다. 코드는 비상업적 학술 용도로 자유롭게 참고할 수 있습니다.
