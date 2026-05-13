# 캡스톤 중간 보고서

**과제명**: 시계열 예측 및 강화학습 기반 이커머스 RBF 최적화
**학생**: 어승윤 (2021105709), 소프트웨어융합학과
**작성일**: 2026-05-04
**진행 단계**: Phase 3 시작 (3/4 단계, 약 60% 완료)

---

## 1. 프로젝트 개요

### 1.1 핵심 문제

이커머스 판매자(특히 신용 이력이 짧은 씬파일러)의 금융 소외 문제를 **매출 연동 동적 상환(Revenue-Based Financing, RBF)** 으로 해결한다. 기존 RBF의 한계인 **"과거 매출 기반 1회 결정 + 고정 비율 상환"** 구조를 극복하기 위해 시계열 예측 모델로 미래 매출 분포를 산출하고, **CVaR 최적화 + 강화학습**으로 대출자-셀러 trade-off를 최적화하는 AI 프레임워크를 제안한다.

### 1.2 4단계 파이프라인

| 단계 | 내용 | 상태 |
|---|---|---|
| **Phase 1** | 대안 데이터 수집 + 한국형 합성 코호트 생성 | ✅ 완료 |
| **Phase 2** | 시계열 예측 모델 베이스라인 (Prophet, LSTM) | ✅ 완료 |
| **Phase 3** | RBF 시뮬레이션 환경 + CVaR 최적화 + RL 에이전트 | 🚧 진행 중 |
| **Phase 4** | 비교 실험 + 평가 + 최종 보고서 | ⏳ 예정 |

### 1.3 주요 산출물 현황

```
Project/
├── Data/                               (45+ 산출물, 분석/검증 데이터)
│   ├── cohort_kr_v2.parquet           (1,302 합성 셀러 × 24개월)
│   ├── kosis/, naver/, Olist_Data/    (원본 + 전처리)
│   ├── prophet_baseline_v{1,2,3}_*    (시계열 베이스라인 결과)
│   ├── lstm_baseline_v{1,2}_*          (LSTM 결과)
│   └── baselines_*                     (RBF 정책 베이스라인)
├── scripts/                            (데이터 수집·합성 9개)
├── models/                             (Prophet 3변형 + LSTM 2변형)
├── envs/                               (Phase 3: RBF 환경 + 정책)
├── optim/, agents/, evaluation/       (Phase 3: 작성 예정)
└── docs/                               (설계 문서, 본 보고서)
```

---

## 2. Phase 1: 데이터 수집 및 한국형 합성 코호트 생성

### 2.1 다중 데이터 소스 수집

| 데이터 | 출처 | 규모 | 활용 |
|---|---|---|---|
| **KOSIS 온라인쇼핑동향** | 통계청 OpenAPI (DT_1KE10051) | 26개 상품군 × 72개월 (2019-01~2024-12) | 한국 시장 trend·season prior |
| **네이버 데이터랩** | 네이버 OpenAPI | 11개 카테고리 × 72개월 검색 트렌드 | 외생 covariate (시계열 예측 입력) |
| **Olist 이커머스** | Kaggle 공개 데이터 | 99,441 주문 / 3,095 셀러 | 분산 구조 추출 (donor pool) |

### 2.2 KOSIS STL 분해

KOSIS 합계 카테고리(한국 이커머스 전체)에 대해 **STL 분해(period=12, robust=True)** 적용하여 trend·seasonal·residual 추출. 합성 코호트의 한국 시장 prior로 사용.

> KOSIS는 통계청 공식 통계로 출처 명확, 누구나 검증 가능. 본 프로젝트에서 가장 학술적으로 방어 가능한 한국 데이터 소스.

### 2.3 Olist 셀러 클러스터링 (v3)

Olist 셀러를 4가지 유형(안정형/계절형/성장형/불안정형)으로 분류하기 위한 K-Means 클러스터링 진행.

**v2 → v3 개선**:
- **활성 구간 ≥ 12개월 필터**: 1년 미만 단기 셀러 제외 → 651명
- **`seasonality_strength` 재정의**: 기존 std-ratio (모든 셀러 1.0 수렴) → **ACF lag-12** (12개월 자기상관)
- **임계 기반 라벨링**: 무조건 4유형 강제하지 않음, 임계 미달은 "기타/쇠퇴형"

**v3 결과**:
- Silhouette score: **0.248** (약한 분리)
- 안정형 음수 silhouette **0.0%** (v2의 4.0%에서 개선)
- "Olist 데이터에는 진짜 계절형 셀러가 거의 없음" 객관적으로 확인 (모든 클러스터의 ACF lag-12가 0.33~0.41로 변별력 약함)
- **결론**: Olist 분류는 분산 파라미터 추출용으로만 사용, 4유형 정의는 KOSIS+수식 기반으로 정당화

### 2.4 한국형 합성 코호트 생성 (Per-seller 부트스트랩, v2)

기존 클러스터 기반 합성(v1)의 한계(클러스터링 silhouette 0.25, 이질성 손실)를 극복하기 위해 **per-seller 비모수 부트스트랩** 방식으로 변경.

**방법**:
1. 651명 Olist donor 각각의 통계 지문 추출 (CV, AR(1), trend, seasonality, spike, zero ratio, scale)
2. 각 donor마다 KOSIS 한국 trend·season prior + Naver covariate + 한국 프로모션 캘린더(블프/추석/설) 적용
3. donor × 2 batch = **1,302명 합성 셀러 × 24개월 = 31,248행**
4. v3 임계 룰 사후 적용으로 4유형 라벨 부여 (Phase 3·4 비교 실험용)

**검증**:
- KOSIS 합계 패턴 vs 합성 합계 **Pearson r = 0.5506** (p=0.0053, 유의)
- 외생 변수 영향 강도 균형값 (Naver ±7.5%, Promo +20%)
- 유형 분포: stable 1018, growth 101, decline 54, other 56, volatile 40, seasonal 33

> **방법론적 정당성**: per-seller bootstrap은 healthcare(synthpop), census(SYNTHEA) 등에서 사용되는 표준 합성 데이터 기법. 클러스터링 평균화로 인한 이질성 손실을 회피.

### 2.5 Phase 1 학술적 한계

- **Olist 분류 silhouette 0.25**: 약한 분리, 단순화 가정 명시 필요
- **KOSIS r 0.55**: 외생 변수 강도와 trade-off로 이전 0.815 대비 하락 (외생 신호 의미있게 들어가는 대신)
- **단순화 가정**: 모든 셀러가 한국 이커머스 평균 trend를 따른다고 가정 (카테고리별 분기 미반영)
- **데이터 기간 한계**: Olist 23개월 → 자연 계절성 추출 한계

---

## 3. Phase 2: 시계열 예측 베이스라인

### 3.1 데이터 분할 + 평가 지표

- **분할**: 24개월 → 학습 18 / 검증 3 / 테스트 3
- **샘플**: 유형별 50명 = 271 셀러 (1차), 1,302 셀러 전체 (LSTM)
- **지표**: MAPE, SMAPE, WAPE (3가지 robust 지표)

### 3.2 Prophet 베이스라인 (3변형)

| 변형 | 변경 사항 | WAPE test mean | < 20% 셀러 비율 |
|---|---|---|---|
| **v1** | 매출만 (외생 X) | 140.0% | 14.2% |
| **v2** | + 음수 예측 차단 + Robust 지표 (SMAPE, WAPE) | 104.5% | 17.8% |
| **v3** | + 외생 변수 (Naver, Promo) | 141.9% | 12.3% (악화) |

**주요 발견**:
- 음수 차단으로 outlier 제거 효과 (mean 140 → 104%)
- 외생 변수 추가는 Prophet에서 오히려 악화 (18개월 train으로 자유도 증가 → 과적합)
- **유형별 명확한 패턴**:
  - growth: WAPE 17.5% (목표 ≤ 20% 달성!)
  - stable: 26.3% (양호)
  - decline/seasonal/volatile/other: 95-103% (Prophet 한계)

### 3.3 LSTM 글로벌 모델 (2변형)

PyTorch 기반 Encoder-Decoder LSTM, type embedding + 외생 변수 입력.

| 변형 | 변경 사항 | WAPE test mean | < 20% 셀러 비율 |
|---|---|---|---|
| **v1** | hidden=64, dropout=0.1, uniform sampling | 75.0% | 2.2% |
| **v2** | hidden=128, dropout=0.25, **WeightedRandomSampler** + cosine LR + 80 epochs | 76.8% | 2.9% |

**v1 vs v2 비교**:
- v1 vs v2 셀러별 산점도: **54% 셀러 개선**
- growth: WAPE 52.2 → **35.9%** (큰 개선, weighted sampling 효과)
- decline: 100.1 → 92.6%, seasonal: 91.6 → 87.2% (소수 유형 학습 강화 효과)
- **Overfitting**: epoch 11에 best val 도달 후 정체 (모델 capacity > 데이터 신호)

### 3.4 Phase 2 종합 결론

**모델별 강점/한계** (WAPE median, 외생 X 기준):

| 유형 | Prophet v2 | LSTM v2 | 우위 |
|---|---|---|---|
| growth | **17.5%** ✅ | 35.9% | Prophet (셀러별 trend 학습) |
| stable | 26.3% | 48.6% | Prophet |
| decline | 100.0% | **92.6%** | LSTM |
| seasonal | 100.0% | **87.2%** | LSTM |
| volatile | 103.2% | 122.1% | Prophet |
| other | 95.0% | **82.5%** | LSTM |

> **학술적 결론**: "Trend·Seasonality 신호가 명확한 유형(growth, stable)은 셀러별 모델인 Prophet이, 노이즈가 큰 유형(decline, seasonal, other)은 글로벌 LSTM이 평균값 예측에 우위. 합성 데이터의 24개월 단기·high noise 한계로 두 모델 모두 < 20% 셀러 비율 < 20%."

이 패턴은 **"단순 예측 정확도가 아닌 모델별 데이터 특성 매칭"** 을 보여주며, Phase 3에서 시계열 예측 분포를 의사결정 입력으로 사용할 때 각 모델의 장점을 활용할 수 있는 근거가 된다.

### 3.5 Phase 2 학술적 한계

- **합성 데이터의 high-noise**: 어떤 모델로도 < 20% 비율 20% 미만
- **외생 변수 효과 제한적**: 강하게 합성하면 KOSIS r 무너짐, 약하게 하면 모델 학습 불가 (trade-off)
- **단기 학습 데이터(18개월)**: yearly seasonality 추출 불가, 외생 변수 과적합 위험

---

## 4. Phase 3 (진행 중): RBF 시뮬레이션 환경 + 베이스라인

### 4.1 가계 보호 설계 (사용자 핵심 요구사항)

> 사용자 제기: "판매자들이 실제 생활에서 생활하는 현금 흐름까지 침해하지 않는 수준에서 회수율을 가져가야 한다"

**2-tier burden 모델** ([design_household_protection.md](Project/docs/design_household_protection.md) 참조):

```
매출 (R_t)
  │
  ├── 운전자금 (1 − m_i) × R_t          ← Tier 1: 사업 차원 (매출 비례)
  │
  └── 영업이익 (m_i × R_t)
        │
        ├── 가계 생활비 (L_personal)     ← Tier 2: 개인 차원 (거의 고정)
        │
        └── RBF 가용 = m_i × R_t − L_personal
```

**구현 단계**: v1 (현재)는 단순 m_i burden만, v2에서 가계 보호 통합 예정. L_personal_min은 통계청 가계동향조사·최저임금 자료로 검증 후 결정.

### 4.2 RBF 시뮬레이션 환경 (envs/rbf_env.py)

**설계 (v1 MVP)**:

| 항목 | 값 |
|---|---|
| **데이터 소스** | cohort_kr_v2.parquet (1,302 셀러) |
| **State (13-dim)** | [t/T, recovery_progress, recent_3mo_rev, type_one-hot(6), m_i, log_scale_norm] |
| **Action** | 연속 r_t ∈ [0.03, 0.25] (수수료율) |
| **Reward** | α·recovery_inc − β·burden + 종료 보상 |
| **대출 조건** | L = 3 × mean_revenue, cap = 1.2, T = 24개월 |
| **m_i** | 0.15 (셀러 영업이익률 가정) |
| **가중치** | α=1.0, β=2.0, γ=5.0(완납 보너스), δ=10.0(디폴트 페널티) |

**검증**: env reset/step 정상 작동, 1,302 셀러 모두 로드 확인.

### 4.3 베이스라인 정책 평가

6개 베이스라인 정책 × 1,302 셀러 평가 (총 7,812 에피소드):

| 정책 | Completion Rate | Mean Recovery | Mean Burden | Mean Reward |
|---|---|---|---|---|
| **Random** (균등) | 27.8% | 0.928 | 0.0499 | -0.73 |
| **Fixed r=0.05** | 0.0% | 0.333 | 0.0000 | -6.33 |
| **Fixed r=0.10** | 0.0% | 0.667 | 0.0000 | -2.67 |
| **Fixed r=0.15** | **79.7%** | 1.000 | ~0.0000 | **+1.04** |
| **Fixed r=0.20** | **100.0%** | 1.037 | 0.0500 | +0.64 |
| **RevProportional** | 0.1% | 0.637 | 0.0020 | -3.00 |

**해석**:
- Fixed-0.15가 **균형점**: 80% 완납 + 거의 0 침해 + 가장 높은 reward
- Fixed-0.20은 100% 회수지만 burden 발생 (셀러 침해)
- Random은 변동성 자체가 회수에 도움 (27.8% 완납)
- 단순 RevProportional 휴리스틱은 비효율 (0.1% 완납)

> **이 결과는 RL의 가치를 정량화할 기준점**이 됨. CVaR/RL이 Fixed-0.15 대비 더 나은 trade-off (높은 reward + 낮은 burden)를 달성해야 의미 있음.

### 4.4 Phase 3 진행 현황

| 컴포넌트 | 상태 | 산출물 |
|---|---|---|
| `envs/rbf_env.py` | ✅ 완료 | Gymnasium 환경 클래스 |
| `envs/baselines.py` | ✅ 완료 | 6개 정책 평가 + 시각화 |
| `optim/cvar_optimizer.py` | ⏳ 다음 | CVXPY로 정적 L*, r0* 최적화 |
| `agents/train_ppo.py` | ⏳ 예정 | StableBaselines3 PPO 학습 |
| `evaluation/compare_policies.py` | ⏳ 예정 | 3-정책 비교 (Baseline vs CVaR vs CVaR+RL) |

---

## 5. 다음 단계 계획 (Phase 3 후반 + Phase 4)

### 5.1 단기 (5월 중순까지)

1. **CVaR 정적 최적화**: CVXPY로 셀러별 L*, r0* 산정
   - Monte Carlo 1,000 시나리오 (합성 데이터에서 샘플)
   - 디폴트 위험 제약 (CVaR_5% ≤ τ)
2. **PPO 동적 RL 에이전트**: 매월 r_t 조정
   - 시계열 예측 분포(Phase 2 산출)를 state에 통합
   - 가계 보호(2-tier burden) 통합

### 5.2 중기 (5월 말 ~ 6월 초)

3. **3-정책 비교 실험**:
   - Baseline (Fixed-0.15)
   - CVaR-only (정적 최적화만)
   - CVaR + RL (정적 + 동적, 본 연구 제안)
4. **유형별 성능 측정**: 4유형(또는 6유형 with decline/other) 각각의 trade-off 분석

### 5.3 보고서 (6월)

5. **민감도 분석**: m_i, L_personal 등 추정 불확실성에 대한 정책 robustness
6. **최종 보고서 작성**

---

## 6. 학술적 기여 및 차별점

| 기존 RBF 연구 | 본 연구 |
|---|---|
| 회수율·디폴트율만 최적화 | + **셀러 가계 생활 보호** (2-tier burden) |
| 단일 burden 지표 | **2-tier burden** (사업/가계 분리) |
| 셀러 = 사업체 | **셀러 = 사업체 + 가구** (개인사업자 현실 반영) |
| 정적 최적화 OR 동적 RL | **정적 CVaR + 동적 RL 하이브리드** (안전 + 적응) |
| 시계열 예측 별도 | **시계열 예측 분포를 RL state로 통합** |

> **핵심 서사**: "현실적이고 윤리적인 RBF 시스템" — 한국 이커머스 셀러의 사회적 맥락(개인사업자 다수, 가계 직격 위험)을 정량적으로 반영한 의사결정 프레임워크

---

## 7. 정직한 한계 인정

본 보고서는 **사실 확인 기반**으로 작성되었으며, 학술적 약점도 명시한다.

1. **데이터 합성 의존**: 한국 셀러 실데이터 부재로 합성 코호트 사용. KOSIS r=0.55로 시장 패턴 반영했으나 카테고리별 분기 미반영
2. **Olist 클러스터링 약점**: Silhouette 0.25, 4유형 자연 분리 어려움 → 분산 파라미터 추출용으로만 사용
3. **시계열 예측 절대 정확도 한계**: WAPE < 20% 셀러 비율 < 20% (단기·고노이즈 한계). Phase 3에서 예측 분포로 활용하는 방향으로 보완
4. **m_i, L_personal 추정 불확실성**: 정확한 값 미확정 → Phase 4 민감도 분석으로 robustness 입증 예정
5. **외생 변수 영향 강도 가정**: KOSIS r과 외생 신호 강도 사이 trade-off, 균형값 채택

---

## 8. 진행률 요약

```
Phase 1: 데이터 + 합성    ████████████████████  100%
Phase 2: 시계열 예측      ████████████████████  100%
Phase 3: RBF 환경 + RL    ████████░░░░░░░░░░░░   40%
Phase 4: 평가 + 보고서    ░░░░░░░░░░░░░░░░░░░░    0%
─────────────────────────────────────────────────
전체 진행률                                       ~60%
```

**남은 일정 (예상)**:
- 5월 중순: CVaR + PPO 구현 완료
- 5월 말: 3-정책 비교 실험
- 6월 초: 민감도 분석
- 6월 중순: 최종 보고서

---

## 부록: 주요 산출 파일

### Phase 1 (데이터 + 합성)
- `Data/cohort_kr_v2.parquet` — 합성 코호트 (1,302 × 24개월)
- `Data/seller_features_v3.csv` — donor pool (651명)
- `Data/kosis/kr_trend_season.parquet` — KOSIS STL 분해
- `Data/cohort_kr_v2_diagnostics.png` — 합성 검증 시각화

### Phase 2 (시계열 예측)
- `Data/prophet_baseline_v2_results.csv` — Prophet 결과
- `Data/lstm_baseline_v2_results.csv` — LSTM 결과
- `Data/{prophet,lstm}_baseline_v2_diagnostics.png` — 시각화

### Phase 3 (현재)
- `envs/rbf_env.py` — Gymnasium RBF 환경
- `envs/baselines.py` — 베이스라인 정책
- `Data/baselines_summary.csv` — 정책 비교
- `Data/baselines_diagnostics.png` — 4-panel 시각화

### 설계 문서
- `docs/design_household_protection.md` — 가계 보호 설계
- `docs/interim_report.md` — 본 보고서
