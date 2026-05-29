"""Lv4 Week 2 (Day 9-10): Streamlit 웹페이지 — RBF 의사결정 시연

학습/운영 패러다임 구체화:
  학습 (본 연구): 합성 데이터 1,302명 + KOSIS/Naver prior로 PPO 학습
  운영 (본 웹페이지): 새 셀러 매출 + 카테고리 입력 → 학습된 PPO로 시뮬레이션 → 대출 가능성 + 위험 평가

페이지 구성:
  1. 셀러 정보 입력 (사이드바)
     - 매출 이력 (3~24개월)
     - 카테고리 (KOSIS 19개)
     - 영업이익률 추정 (m_i, 기본 0.10)
     - 가구 정보 (L_personal, 기본 128만)
  2. 시뮬레이션 결과 (메인)
     - PPO 정책 적용 → 매월 r_t 시퀀스
     - 회수율 + 침해 분석
     - 5-정책 비교 (Fixed/CVaR/PPO/X1 비례)
  3. 위험 평가 + 추천
     - 대출 적합도 등급 (A/B/C/D)
     - 사전 적합도 평가 (Day 8 발견 활용)

디스클레이머: 학술 연구용 시뮬레이션, 실제 금융 의사결정 근거 아님.

실행:
  source Project/.venv/bin/activate
  streamlit run Project/webapp/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.rbf_env import RBFEnv

ROOT = Path("/Users/eoseungyun/Desktop/project/SW_Capstone/Project")
DATA = ROOT / "Data"
MODELS = ROOT / "models"

KOSIS_CATEGORIES = [
    "생활용품", "가구", "스포츠·레저용품", "화장품", "아동·유아용품",
    "컴퓨터 및 주변기기", "가전·전자", "가방", "통신기기", "서적",
    "패션용품 및 액세서리", "음·식료품", "의복", "신발", "농축수산물",
    "자동차 및 자동차용품", "애완용품", "기타", "사무·문구",
]

TYPE_LABELS = {"stable": "안정형", "growth": "성장형", "seasonal": "계절형",
                "volatile": "변동형", "decline": "쇠퇴형", "other": "기타"}

st.set_page_config(page_title="RBF 의사결정 시뮬레이션", page_icon=":bar_chart:", layout="wide")

# 모델 로드 (캐시)
@st.cache_resource
def load_ppo_models():
    from stable_baselines3 import PPO
    models = {}
    for name, path in [("PPO_base", MODELS / "ppo" / "ppo_final.zip"),
                        ("PPO_forecast", MODELS / "ppo_forecast" / "ppo_final.zip"),
                        ("PPO_x1_only", MODELS / "ppo_x1_only" / "ppo_final.zip")]:
        if path.exists():
            try:
                models[name] = PPO.load(str(path))
            except Exception:
                pass
    return models


def classify_type(revenues: np.ndarray) -> str:
    """매출 이력에서 셀러 유형 추정 (cohort 라벨 로직 간소화)."""
    nz = np.where(revenues > 0)[0]
    if len(nz) < 6:
        return "other"
    ts = revenues[nz[0]:nz[-1] + 1]
    if len(ts) < 6:
        return "other"
    mu = ts.mean(); sd = ts.std()
    cv = sd / (mu + 1e-8)
    zero = (ts == 0).mean()
    x = np.arange(len(ts))
    if sd > 0:
        slope = np.polyfit(x, ts, 1)[0]
        trend = slope / (mu + 1e-8) * len(ts)
    else:
        trend = 0
    if cv >= 1.2 and zero >= 0.3:
        return "volatile"
    if trend >= 0.3:
        return "growth"
    if cv < 1.0 and zero < 0.3:
        return "stable"
    if trend < -0.2:
        return "decline"
    return "other"


def synthesize_future(history: np.ndarray, T: int = 24, seed: int = 42) -> np.ndarray:
    """매출 이력으로 미래 (T - len(history))개월 합성.
    간단한 AR(1) + 노이즈 기반.
    """
    rng = np.random.default_rng(seed)
    out = list(history.astype(float))
    n_remain = T - len(out)
    if n_remain <= 0:
        return np.array(out[:T])

    mu = float(np.mean([x for x in out if x > 0])) if any(x > 0 for x in out) else 100.0
    sd = float(np.std(out)) if len(out) > 1 else mu * 0.3
    ar1 = float(np.corrcoef(out[:-1], out[1:])[0, 1]) if len(out) > 1 and np.std(out) > 0 else 0.3
    if np.isnan(ar1):
        ar1 = 0.3

    last = out[-1]
    noise_prev = 0.0
    for _ in range(n_remain):
        noise = ar1 * noise_prev + sd * rng.standard_normal() * 0.5
        noise_prev = noise
        v = max(0.0, mu * 0.95 + 0.05 * last + noise)  # 평균 회귀 + 일부 carry
        out.append(v); last = v
    return np.array(out)


def simulate_policy(revenues: np.ndarray, policy_name: str, m_i: float, L_personal: float,
                     model=None, fixed_r: float | None = None,
                     use_forecast_state: bool = False,
                     eta: float = 1.0, eta_proportional: bool = False) -> dict:
    """단일 셀러에 정책 적용 + 시뮬레이션. 결과 dict 반환."""
    # 임시 env (단일 셀러 데이터로 in-memory)
    class _SingleSellerEnv(RBFEnv):
        def __init__(self_inner):
            # 부모 __init__ 호출하지 않고 직접 설정
            import gymnasium as gym
            from gymnasium import spaces
            from envs.rbf_env import N_TYPES
            self_inner.T = len(revenues)
            self_inner.cap = 1.2
            self_inner.loan_multiplier = 3.0
            self_inner.m_i = m_i
            self_inner.L_personal_min = L_personal
            self_inner.r_min = 0.03; self_inner.r_max = 0.25
            self_inner.alpha = 1.0; self_inner.beta = 2.0
            self_inner.gamma = 5.0; self_inner.delta = 10.0
            self_inner.eta = eta
            self_inner.eta_proportional = eta_proportional
            self_inner.r_clip_to_safe = False
            self_inner.use_forecast_state = use_forecast_state
            self_inner.forecast_lag = 6

            typ = classify_type(revenues)
            from envs.rbf_env import TYPE_TO_ID
            mean_rev = float(np.mean([r for r in revenues if r > 0])) if any(r > 0 for r in revenues) else 100.0
            self_inner.sellers = {
                "user": {
                    "revenues": revenues.astype(np.float32),
                    "type": typ,
                    "type_id": TYPE_TO_ID.get(typ, 5),
                    "mean_rev": mean_rev,
                    "mu": mean_rev,
                }
            }
            # log_scale 통계 (단일 셀러라 더미)
            self_inner.log_scale_mean = float(np.log1p(mean_rev))
            self_inner.log_scale_std = 1.0

            self_inner.state_dim = 1 + 1 + 3 + N_TYPES + 1 + 1
            if use_forecast_state:
                self_inner.state_dim += 3
            self_inner.observation_space = spaces.Box(low=-10.0, high=10.0,
                                                       shape=(self_inner.state_dim,), dtype=np.float32)
            self_inner.action_space = spaces.Box(
                low=np.array([self_inner.r_min], dtype=np.float32),
                high=np.array([self_inner.r_max], dtype=np.float32), dtype=np.float32)

            self_inner._rng = np.random.default_rng(42)
            self_inner._seller_id_list = ["user"]
            self_inner.current_seller = None
            self_inner.t = 0
            self_inner.cumulative_repaid = 0.0
            self_inner.recent_revs = [0.0, 0.0, 0.0]
            self_inner.L = 0.0; self_inner.target = 0.0
            self_inner._episode_log = []

    env = _SingleSellerEnv()
    obs, info_init = env.reset(options={"seller_id": "user"})

    r_history = []
    revenue_history = []
    payment_history = []
    safe_cap_history = []
    violation_amount_history = []

    while True:
        if policy_name == "Fixed":
            action = np.array([fixed_r or 0.10], dtype=np.float32)
        elif policy_name == "Naive_R_proportional":
            # 단순 매출 비례
            recent = env.recent_revs
            avg = np.mean(recent) if any(recent) else env.current_seller["mean_rev"]
            ratio = recent[-1] / max(avg, 1e-6) if any(recent) else 1.0
            r = float(np.clip(0.10 * ratio, env.r_min, env.r_max))
            action = np.array([r], dtype=np.float32)
        else:
            # PPO models
            action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)
        r_history.append(info["r_t"])
        revenue_history.append(info["revenue"])
        payment_history.append(info["payment"])
        safe_cap_history.append(info["safe_rbf_cap"])
        violation_amount_history.append(info.get("household_violation_amount", 0.0))
        if terminated:
            break

    return {
        "L": env.L, "target": env.target,
        "final_recovery": env.cumulative_repaid / max(env.target, 1.0),
        "completed": env.cumulative_repaid >= env.target,
        "r_history": r_history,
        "revenue_history": revenue_history,
        "payment_history": payment_history,
        "safe_cap_history": safe_cap_history,
        "violation_amount_history": violation_amount_history,
        "total_violation_amount": sum(violation_amount_history),
        "violation_months": sum(1 for v in violation_amount_history if v > 0),
        "max_violation_ratio": max([v / max(L_personal, 1e-6) for v in violation_amount_history], default=0.0),
        "type": info_init["type"],
    }


def assess_loan_grade(result: dict, L_personal: float) -> tuple[str, str, str]:
    """대출 적합도 등급 + 색상 + 설명."""
    completed = result["completed"]
    max_ratio = result["max_violation_ratio"]
    violation_months = result["violation_months"]

    if completed and max_ratio < 0.3 and violation_months < 6:
        return "A", "green", "RBF 적합 — 회수 가능 + 가계 보호 양호"
    elif completed and max_ratio < 1.0 and violation_months < 12:
        return "B", "blue", "RBF 가능 — 회수 가능하나 일부 침해 발생"
    elif completed and max_ratio < 2.0:
        return "C", "orange", "RBF 권장 안 함 — 회수 가능하나 가계비 침해 큼"
    elif not completed and result["final_recovery"] > 0.5:
        return "D", "red", "RBF 부적합 — 회수 부족 (50% 이상은 회수)"
    else:
        return "F", "darkred", "RBF 거절 — 매출 부족으로 회수 거의 불가능"


def main():
    st.title("💼 RBF 의사결정 시뮬레이션")
    st.markdown("*시계열 예측 및 강화학습 기반 이커머스 RBF 최적화 — 학습 vs 운영 패러다임 시연*")
    st.warning("⚠️ **학술 연구용 시뮬레이션**. 합성 데이터 기반 시스템이며 실제 금융 의사결정의 근거가 될 수 없습니다.")

    # 사이드바: 셀러 정보 입력
    with st.sidebar:
        st.header("셀러 정보 입력")

        st.subheader("매출 이력 (만원/월)")
        n_months = st.slider("입력 개월 수", 3, 18, 12)
        st.caption("미입력 미래는 자동 합성")

        # 매출 이력 입력 — 기본값 sample
        default_revs = [500, 550, 480, 600, 650, 700, 580, 620, 700, 750, 680, 720][:n_months]
        revenues_str = st.text_area(
            f"최근 {n_months}개월 매출 (쉼표 구분, 만원)",
            value=", ".join(map(str, default_revs)),
            height=80,
        )
        try:
            history = np.array([float(x.strip()) for x in revenues_str.split(",")])[:n_months]
        except Exception:
            st.error("매출 형식 오류 — 쉼표로 구분된 숫자")
            return

        st.subheader("카테고리")
        kosis_cat = st.selectbox("KOSIS 분류", KOSIS_CATEGORIES, index=0)

        st.subheader("RBF 환경 파라미터")
        m_i = st.slider("영업이익률 m_i", 0.03, 0.30, 0.10, 0.01,
                         help="기본 0.10 (스마트스토어 gross 20% - 운영비 10% 추정)")
        L_personal = st.slider("월 최소 가계비 (만원)", 50.0, 400.0, 128.21, 5.0,
                                help="기본 128만 (KB 1인가구 / 보건복지부 중위소득 50%)")

        st.subheader("미래 합성")
        seed = st.number_input("시드", 1, 9999, 42)
        T = 24
        revenues_full = synthesize_future(history, T=T, seed=seed)

        st.markdown(f"**전체 24개월** = 입력 {len(history)} + 합성 {T - len(history)}")
        st.markdown(f"**평균 매출**: {np.mean(revenues_full):.1f} 만원")
        st.markdown(f"**영업이익 평균**: {np.mean(revenues_full) * m_i:.1f} 만원")
        st.markdown(f"**가계 보호 후 RBF 가용**: {max(0, np.mean(revenues_full) * m_i - L_personal):.1f} 만원/월")

        run_sim = st.button("시뮬레이션 실행", type="primary")

    if not run_sim:
        st.info("← 사이드바에서 셀러 정보 입력 후 [시뮬레이션 실행] 클릭")
        st.markdown("### 본 시스템 개요")
        st.markdown("""
        1. **입력**: 새 셀러의 매출 이력 + 카테고리 + 환경 파라미터
        2. **처리**:
           - 미래 매출 합성 (24개월까지)
           - 5가지 정책으로 RBF 시뮬레이션
           - 회수율 + 가계 침해 분석
        3. **출력**: 대출 적합도 등급 (A/B/C/D/F) + 정책별 비교
        """)
        return

    # ===== 시뮬레이션 =====
    st.header("📊 시뮬레이션 결과")
    st.markdown(f"**셀러 유형 추정**: {TYPE_LABELS.get(classify_type(history), '기타')} / **카테고리**: {kosis_cat}")

    # 매출 시계열 (입력 + 합성)
    chart_df = pd.DataFrame({
        "month": np.arange(T) + 1,
        "revenue": revenues_full,
        "is_input": ["입력"] * len(history) + ["합성"] * (T - len(history)),
    })
    st.subheader("매출 이력 (입력 + 합성)")
    st.line_chart(chart_df.set_index("month")["revenue"])

    # PPO 모델 로드
    models = load_ppo_models()
    if not models:
        st.error("PPO 모델이 학습되지 않음. agents/train_ppo.py 등 실행 필요.")
        return

    # 5-정책 시뮬레이션
    st.subheader("정책별 시뮬레이션")
    results = {}
    with st.spinner("시뮬레이션 중..."):
        results["Fixed-0.15"] = simulate_policy(revenues_full, "Fixed", m_i, L_personal, fixed_r=0.15)
        if "PPO_base" in models:
            results["PPO base"] = simulate_policy(revenues_full, "PPO_base", m_i, L_personal,
                                                   model=models["PPO_base"])
        if "PPO_forecast" in models:
            results["PPO + 분포"] = simulate_policy(revenues_full, "PPO_forecast", m_i, L_personal,
                                                    model=models["PPO_forecast"],
                                                    use_forecast_state=True)
        if "PPO_x1_only" in models:
            results["PPO + 비례 페널티"] = simulate_policy(
                revenues_full, "PPO_x1_only", m_i, L_personal,
                model=models["PPO_x1_only"], eta=3.0, eta_proportional=True)

    # 결과 표
    rows = []
    for name, r in results.items():
        grade, color, desc = assess_loan_grade(r, L_personal)
        rows.append({
            "정책": name,
            "회수율": f"{r['final_recovery']:.1%}",
            "완납": "✓" if r["completed"] else "✗",
            "침해 월수": r["violation_months"],
            "누적 침해액(만원)": f"{r['total_violation_amount']:.0f}",
            "단일월 최대 침해비": f"{r['max_violation_ratio']:.2f}",
            "등급": grade,
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # ===== 추천 정책 =====
    st.header("🎯 적합도 평가")
    # PPO base 기준
    if "PPO base" in results:
        r = results["PPO base"]
        grade, color, desc = assess_loan_grade(r, L_personal)
        col1, col2, col3 = st.columns([1, 2, 1])
        col1.metric("등급", grade)
        col2.markdown(f"### :{color}[{desc}]")
        col3.metric("회수율", f"{r['final_recovery']:.0%}")

        # 상세 분석
        st.subheader("PPO base 정책 적용 시 상세")
        sim_df = pd.DataFrame({
            "month": np.arange(T) + 1,
            "매출 (만원)": r["revenue_history"],
            "상환 (만원)": r["payment_history"],
            "가계 보호선 (만원)": r["safe_cap_history"],
            "침해액 (만원)": r["violation_amount_history"],
        }).set_index("month")
        st.line_chart(sim_df[["매출 (만원)", "상환 (만원)", "가계 보호선 (만원)"]])

        if r["total_violation_amount"] > 0:
            st.warning(f"⚠️ 24개월 누적 가계 침해액: **{r['total_violation_amount']:.0f}만원**, "
                       f"단일 월 최대 침해비 **가계비의 {r['max_violation_ratio']:.2f}배**")

        # r 시퀀스
        st.subheader("PPO 매월 r_t 결정 (동적 조정)")
        rt_df = pd.DataFrame({"month": np.arange(T) + 1, "r_t": r["r_history"]}).set_index("month")
        st.line_chart(rt_df)

    # ===== Day 8 발견 반영: 사전 적합도 평가 =====
    st.header("🛡️ 사전 적합도 평가 (학술적 발견 적용)")
    st.markdown("""
    **본 연구 발견 (Lv4)**: 모든 셀러를 RBF 대상으로 보면 "회수율 + 침해 빈도 + 침해 강도" 동시 최적화 불가능.
    영업이익이 가계비를 충당 못하는 영세 셀러에서는 침해 없이 회수 불가능.
    → **사전 적합도 평가로 부적합 셀러를 제외해야 함**.
    """)

    avg_revenue = np.mean(revenues_full)
    avg_operating_profit = avg_revenue * m_i
    coverage_ratio = avg_operating_profit / L_personal

    if coverage_ratio < 1.0:
        st.error(f"❌ **사전 적합도 미달**: 영업이익(평균 {avg_operating_profit:.0f}만) < 가계비({L_personal:.0f}만). "
                 f"커버리지 비율 {coverage_ratio:.2f} (< 1.0). RBF 대출 권장 불가.")
    elif coverage_ratio < 2.0:
        st.warning(f"⚠️ **경계 영역**: 커버리지 비율 {coverage_ratio:.2f}. 매출 변동성에 따라 위험.")
    else:
        st.success(f"✅ **사전 적합도 통과**: 커버리지 비율 {coverage_ratio:.2f}. RBF 가능 셀러 군.")

    st.caption("Coverage = 평균 영업이익 / 가계비. 1.0 미만이면 가계비조차 충당 못함.")

    st.markdown("---")
    st.caption(f"본 시스템 v1.0 — 학습된 PPO 모델 ({len(models)}개) 기반. "
                f"세션 시각: {datetime.now().strftime('%Y-%m-%d %H:%M')}")


if __name__ == "__main__":
    main()
