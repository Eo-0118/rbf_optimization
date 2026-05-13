# 가계 생활비 보호 (Household Protection) 설계 문서

**상태**: Phase 3 진입 시 구현 예정 (Phase 2 시계열 예측은 본 기능 없이 진행)
**작성일**: 2026-05-03
**목적**: RBF 정책에 셀러의 개인/가계 생활비 보호 계층을 추가하여 사회적·윤리적 RBF 시스템으로 차별화

---

## 1. 동기 (Why)

기존 RBF 모델의 한계:
- m_i (영업이익률) 기반 침해 측정 = **사업체 차원**의 보호만 다룸
- 한국 이커머스 셀러 다수가 **개인사업자** → 사업소득 = 가계소득
- 매출 폭락 시 운전자금만 보호해도 셀러 본인의 **월세·식비·자녀교육비** 등 가계 생활은 위협

**핵심 통찰** (사용자 제기):
> "판매자들이 실제 생활에서 생활하는 현금 흐름까지 침해하지 않는 수준에서 회수율을 가져가야 한다"

---

## 2. 두 층 보호 모델 (Two-Tier Burden Model)

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
              (이 부분만 안전하게 상환에 사용 가능)
```

**핵심 차이**:
- 운전자금: **매출 비례** (매출 적으면 원가도 적음)
- 가계 생활비: **거의 고정** (매출 적어도 월세는 그대로) → 매출 폭락 시 진짜 위협

---

## 3. 침해 측정 함수

```python
def two_tier_burden(P_t, R_t, m_i, L_personal_min):
    """
    Tier 1 (사업): RBF가 운전자금 침범
    Tier 2 (가계): RBF가 가계 생활비 침범 — 더 심각
    """
    operating_profit = R_t * m_i
    safe_rbf_cap = max(0, operating_profit - L_personal_min)

    if P_t <= safe_rbf_cap:
        return 0.0  # 완전 안전

    elif P_t <= operating_profit:
        # Tier 2 (가계 침범): 무거운 페널티
        excess = P_t - safe_rbf_cap
        return (excess / R_t) * WEIGHT_HOUSEHOLD  # WEIGHT_HOUSEHOLD ~ 5

    else:
        # Tier 1 + Tier 2 동시 침범 (디폴트 임박)
        excess_household = operating_profit - safe_rbf_cap
        excess_business = P_t - operating_profit
        return ((excess_household * WEIGHT_HOUSEHOLD +
                 excess_business * WEIGHT_BUSINESS) / R_t)
```

---

## 4. RBF 정책에 통합

### 4.1 CVaR 정적 최적화 (Project/optim/)

```python
# CVXPY hard constraint 추가
P[k, t] <= revenues[k, t] * m_i - L_personal_min   # 모든 시나리오 k, 모든 월 t
```

→ 수수료율 r은 가계 생활비를 침범하지 않는 한도 내에서만 결정

### 4.2 RL 환경 (Project/envs/)

```python
def step(self, action):
    # ... 매출 실현, 상환 계산 ...
    burden = two_tier_burden(P_t, R_t, seller.m_i, seller.L_personal_min)

    # 가계 침범 시 큰 페널티
    if P_t > R_t * seller.m_i - seller.L_personal_min:
        household_violation_penalty = -PENALTY_LARGE
    else:
        household_violation_penalty = 0

    reward = alpha * recovery_inc - beta * burden + household_violation_penalty
```

### 4.3 평가 지표 (Project/evaluation/)

신규 지표:
- **가계 침범 비율**: 24개월 중 P_t > safe_rbf_cap 횟수 / 24
- **누적 가계 침범액**: Σ max(0, P_t − safe_rbf_cap_t)

목표: "본 정책은 가계 침범 0건" 달성

---

## 5. L_personal_min 확정값 (2026-05-07 결정)

### Phase 3 v1: 1인 가구 단일값 ⭐ 현재 적용

**L_personal_min = 1,720,612원**

**출처**: 통계청 「2024년 가계동향조사」 1인가구 월평균 소비지출
- 한국 공식 통계 (정부 발표)
- 검증 가능: KOSIS 또는 통계청 보도자료
- 비교 자료:
  - 1인가구 평균 소비지출: 1,720,612원 (가계동향조사 2024)
  - 2026 기준 중위소득 100% (1인): 2,564,238원 (보건복지부)
  - 2026 최저임금 월 환산: 2,156,880원 (고용노동부, 209시간)
  - 2025 1분위 가구 평균: 약 1,349,000원 (저소득 가구)

**채택 이유**:
1. "소비지출"은 가계 생활 침해 측정에 가장 직접적
2. 1인 가구 평균은 자영업자 셀러 가정에 적합
3. 출처 명확하고 학술 인용 표준

### Phase 3 v2 (시간 남으면): 가구 분포 + 카테고리 통합

**계획**: 합성 코호트에 가구 크기 분포 + Olist 카테고리 추가 → 단일 모델이 조건별 차별 정책 학습
- 가구 크기별 L_personal:
  - 1인: 1,720,612원 (가계동향조사 2024)
  - 4인: 4,676,996원 (가계동향조사 2024)
  - 2-3인: 추가 검색 필요
- 가구 크기 분포: 통계청 인구주택총조사 (추가 검색 필요)
- Olist 71 카테고리 ↔ KOSIS 26 카테고리 매핑 (수작업)

### Phase 4 민감도 분석 범위

기준값 1,720,612원 ± 50% 변동:
- 84.5만 (-50%): 매우 적은 생활비
- 126.7만 (-25%): 1분위 가구
- **172만 (기준)**: 1인가구 평균 (2024)
- 211만 (+25%): 최저임금 근사
- 253만 (+50%): 다인 가구 추정

→ ±50% 범위에서 정책 robustness 측정

---

## 6. 합성 코호트에 추가할 필드

`korean_synth_gen_v2.py` 작성 시 셀러별로:

```python
seller = {
    # 기존 필드
    'seller_id': ..., 'mean_rev': ..., 'cv': ..., 'm_i': ...,

    # ★ 가계 보호 신규 필드
    'household_size': sample_household(rng),       # 가구 크기 (1/2/3+)
    'L_personal_min': sample_living_cost(...),     # 월 가계 생활비
}
```

---

## 7. 학술적 의의

| 기존 RBF 연구 | 본 연구 (확장) |
|---|---|
| 회수율·디폴트율만 최적화 | + 셀러 가계 생활 보호 |
| 단일 burden 지표 | 2-tier burden (사업/가계) |
| 셀러 = 사업체 | 셀러 = 사업체 + 가구 (개인사업자 현실 반영) |
| 추상적 "셀러 보호" | 측정 가능 목표: "가계 침범 0건" |

→ "현실적이고 윤리적인 RBF 시스템"이라는 차별화 서사

---

## 8. 다음 단계

**Phase 3 시작 시점에 다시 보아야 할 것**:
1. L_personal_min 출처 확정 (통계청·최저임금 자료 검증)
2. WEIGHT_HOUSEHOLD, PENALTY_LARGE 같은 가중치 튜닝
3. 합성 코호트 재생성 시 household_size, L_personal_min 필드 추가
4. CVaR·RL·평가 코드에 본 함수 통합
5. Phase 4 민감도 분석 설계
