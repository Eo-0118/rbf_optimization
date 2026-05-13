"""RBF (Revenue-Based Financing) 시뮬레이션 환경 — Gymnasium API.

설계 (Phase 3 v1 — 1인 가구 가정 + 가계 보호 통합):
- Episode = 1 셀러 × T개월. reset()마다 새 셀러 샘플링.
- State: 시간 진행, 회수 진행, 최근 3개월 매출, 셀러 유형 one-hot, m_i, log scale
- Action: 연속 r_t ∈ [0.03, 0.25] (수수료율)
- Reward:
    매월: α·recovery_inc - β·burden + household_violation_penalty
    종료:
      완납 시 +γ·(T-t)/T (조기 완납 보너스)
      만기 디폴트 시 -δ·(1 - cumulative_recovery/target)

대출 조건 (v1, 고정 비례):
  L = loan_multiplier × mean_revenue (3개월치 매출)
  target = L × cap (cap=1.2: 20% 마진)

가계 보호 (v1 통합, 2026-05-07):
  m_i = 0.25 (한국 자영업 영업이익률 추정, Phase 4 민감도 검증)
  L_personal_min = 1,720,612원 (통계청 「2024년 가계동향조사」 1인가구 평균 소비지출)
  safe_rbf_cap_t = max(0, R_t × m_i - L_personal_min)
  burden_t = max(0, P_t - safe_rbf_cap_t) / R_t
  household_violation: P_t > safe_rbf_cap → 가계 생활비 침범

Phase 3 v2 (시간 남으면): 가구 분포 + 카테고리 통합 (design_household_protection.md 참조)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

DEFAULT_DATA = Path("/Users/eoseungyun/Desktop/project/SW_Capstone/Project/Data/cohort_kr_v2.parquet")

TYPE_TO_ID = {"stable": 0, "growth": 1, "seasonal": 2, "volatile": 3,
              "decline": 4, "other": 5}
N_TYPES = len(TYPE_TO_ID)


class RBFEnv(gym.Env):
    """Single-seller RBF episode environment."""
    metadata = {"render_modes": []}

    def __init__(
        self,
        cohort_path: Path | str = DEFAULT_DATA,
        T: int = 24,
        cap: float = 1.2,
        loan_multiplier: float = 3.0,
        m_i: float = 0.25,                # v1: 0.15 → 0.25 (한국 자영업 마진 추정)
        L_personal_min: float = 172.0612, # v1 신규: 1,720,612원 = 172.0612 만원 (합성 데이터 만원 단위와 일관)
        r_min: float = 0.03,
        r_max: float = 0.25,
        alpha: float = 1.0,        # 회수율 보상
        beta: float = 2.0,         # 침해 페널티
        gamma: float = 5.0,        # 만기내 완납 보너스
        delta: float = 10.0,       # 디폴트 페널티
        eta: float = 1.0,          # v1 신규: 가계 침범 페널티 (soft, 매월 누적)
        seed: Optional[int] = None,
        seller_ids: Optional[list[str]] = None,  # 특정 subset만 학습/평가
    ):
        super().__init__()
        self.cohort_path = Path(cohort_path)
        self.T = T
        self.cap = cap
        self.loan_multiplier = loan_multiplier
        self.m_i = m_i
        self.L_personal_min = L_personal_min   # v1 신규
        self.r_min = r_min
        self.r_max = r_max
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.eta = eta                         # v1 신규: 가계 침범 페널티

        # 데이터 로드 (셀러 단위 dict로 캐시)
        self._load_cohort(seller_ids)

        # log_scale 정규화용 통계
        log_scales = np.array([s["mean_rev"] for s in self.sellers.values()])
        log_scales = np.log1p(log_scales)
        self.log_scale_mean = float(log_scales.mean())
        self.log_scale_std = float(log_scales.std() + 1e-8)

        # Spaces
        # state dim: 1 (t/T) + 1 (recovery_progress) + 3 (recent rev) + N_TYPES + 1 (m_i) + 1 (log_scale_norm)
        self.state_dim = 1 + 1 + 3 + N_TYPES + 1 + 1
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(self.state_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([r_min], dtype=np.float32),
            high=np.array([r_max], dtype=np.float32),
            dtype=np.float32,
        )

        self._rng = np.random.default_rng(seed)
        self._seller_id_list = list(self.sellers.keys())

        # 에피소드 상태 (reset에서 채워짐)
        self.current_seller = None
        self.t = 0
        self.cumulative_repaid = 0.0
        self.recent_revs = [0.0, 0.0, 0.0]
        self.L = 0.0
        self.target = 0.0
        self._episode_log = []

    def _load_cohort(self, seller_ids: Optional[list[str]] = None) -> None:
        df = pd.read_parquet(self.cohort_path)
        if seller_ids is not None:
            df = df[df["seller_id"].isin(seller_ids)]
        self.sellers = {}
        for sid, sdf in df.groupby("seller_id"):
            sdf = sdf.sort_values("month_idx").reset_index(drop=True)
            rev = sdf["monthly_revenue"].values.astype(np.float32)
            mean_rev = float(rev.mean())
            self.sellers[sid] = {
                "revenues": rev,
                "type": sdf["type"].iloc[0],
                "type_id": TYPE_TO_ID.get(sdf["type"].iloc[0], 5),
                "mean_rev": mean_rev,
                "mu": float(sdf["mu"].iloc[0]),
            }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # 셀러 샘플링 — options에서 강제 지정 가능
        if options and "seller_id" in options:
            sid = options["seller_id"]
        else:
            sid = self._rng.choice(self._seller_id_list)

        self.current_seller = self.sellers[sid]
        self.current_seller_id = sid
        self.t = 0
        self.cumulative_repaid = 0.0
        self.recent_revs = [0.0, 0.0, 0.0]
        # 대출 금액 결정 (v1: 고정 비례)
        self.L = self.loan_multiplier * self.current_seller["mean_rev"]
        self.target = self.L * self.cap
        self._episode_log = []

        info = {
            "seller_id": sid,
            "type": self.current_seller["type"],
            "L": self.L,
            "target": self.target,
            "m_i": self.m_i,
        }
        return self._get_state(), info

    def _get_state(self) -> np.ndarray:
        s = self.current_seller
        # 정규화된 state 벡터
        t_norm = self.t / self.T
        recovery_progress = self.cumulative_repaid / max(self.target, 1.0)
        # 최근 3개월 매출 (mean_rev로 정규화)
        scale = max(s["mean_rev"], 1.0)
        recent_norm = [r / scale for r in self.recent_revs]
        # type one-hot
        type_oh = np.zeros(N_TYPES, dtype=np.float32)
        type_oh[s["type_id"]] = 1.0
        # m_i (이미 [0,1])
        m_i_n = self.m_i
        # log scale z-score
        log_scale_norm = (np.log1p(s["mean_rev"]) - self.log_scale_mean) / self.log_scale_std

        state = np.concatenate([
            [t_norm, recovery_progress],
            recent_norm,
            type_oh,
            [m_i_n, log_scale_norm],
        ]).astype(np.float32)
        return state

    def step(self, action: np.ndarray):
        # Action 처리 (clip, 단일 값 추출)
        r_t = float(np.clip(action[0], self.r_min, self.r_max))

        # 매출 실현 (합성 데이터에서)
        revenue_t = float(self.current_seller["revenues"][self.t])

        # 상환금 계산
        payment_t = revenue_t * r_t
        self.cumulative_repaid += payment_t

        # 보상 계산
        recovery_inc = payment_t / max(self.target, 1.0)

        # Two-tier burden (v1: 가계 보호 통합)
        # Tier 1 (사업): 영업이익 = R_t × m_i 초과 시 운전자금 침범
        # Tier 2 (가계): 영업이익 - L_personal 초과 시 가계 생활비 침범
        operating_profit = revenue_t * self.m_i
        safe_rbf_cap = max(0.0, operating_profit - self.L_personal_min)  # 가계 보호 후 RBF 가용 한도
        excess = max(0.0, payment_t - safe_rbf_cap)
        burden = excess / max(revenue_t, 1.0) if revenue_t > 0 else 0.0

        # 가계 침범 여부 — P_t > safe_rbf_cap이면 가계 생활비 영역 침투
        household_violated = (payment_t > safe_rbf_cap) and (revenue_t > 0)
        household_penalty = -self.eta if household_violated else 0.0

        step_reward = self.alpha * recovery_inc - self.beta * burden + household_penalty

        # 시간 진행
        self.t += 1
        self.recent_revs = self.recent_revs[1:] + [revenue_t]

        # 종료 판정
        completed = self.cumulative_repaid >= self.target
        time_up = self.t >= self.T
        terminated = completed or time_up

        # 종료 보상/페널티
        terminal_bonus = 0.0
        if terminated:
            if completed:
                # 조기 완납 보너스: 만기 대비 남은 개월 비율
                remain_ratio = (self.T - self.t) / self.T
                terminal_bonus = self.gamma * remain_ratio
            else:  # time_up이지만 미완납 → 디폴트
                shortfall = 1.0 - (self.cumulative_repaid / max(self.target, 1.0))
                terminal_bonus = -self.delta * shortfall

        total_reward = step_reward + terminal_bonus

        info = {
            "revenue": revenue_t,
            "payment": payment_t,
            "cumulative_repaid": self.cumulative_repaid,
            "recovery_progress": self.cumulative_repaid / max(self.target, 1.0),
            "burden": burden,
            "operating_profit": operating_profit,
            "safe_rbf_cap": safe_rbf_cap,
            "household_violated": bool(household_violated),
            "household_penalty": household_penalty,
            "step_reward": step_reward,
            "terminal_bonus": terminal_bonus,
            "completed": completed,
            "time_up": time_up,
            "r_t": r_t,
        }
        self._episode_log.append(info)

        truncated = False  # gymnasium API: 외부 cutoff 없음 (T 도달은 terminated)
        return self._get_state(), float(total_reward), terminated, truncated, info

    def get_episode_log(self) -> list[dict]:
        return self._episode_log

    def render(self):
        pass  # 시각화는 evaluation에서 별도 처리

    def close(self):
        pass


# === Sanity check (단독 실행 시) ===
if __name__ == "__main__":
    env = RBFEnv(seed=42)
    print(f"Sellers: {len(env.sellers)}")
    print(f"State dim: {env.state_dim}, Action range: [{env.r_min}, {env.r_max}]")

    state, info = env.reset()
    print(f"\nReset → seller={info['seller_id'][:20]} type={info['type']} L={info['L']:.0f} target={info['target']:.0f}")
    print(f"Initial state shape: {state.shape}")

    # 임의 action 5번
    total_reward = 0.0
    for step in range(5):
        action = np.array([0.10], dtype=np.float32)  # 10% 수수료율 고정
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(f"  step {env.t}: rev={info['revenue']:.0f} pay={info['payment']:.0f} "
              f"recovery={info['recovery_progress']:.3f} burden={info['burden']:.3f} reward={reward:+.4f}")
        if terminated:
            print(f"  → terminated! total reward={total_reward:.3f}")
            break
