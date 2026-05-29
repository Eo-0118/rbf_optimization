"""Day 12 (Phase D): PPO v2 — Multi-condition 학습 + State 확장 + 비례 페널티

핵심 개선 (Day 11++ 발견 대응):
  Day 11++ 결과: 적합 셀러 81명(R_p10)에서 PPO 50.6% < CVaR 100% (회수율)
  원인: PPO가 학습 환경(L=3R, cap=1.2, T=24)과 평가 환경(L*, cap*, T=36) 미스매치

v2 변경:
  1. State 확장: + L_norm, cap_norm, T_norm (use_lct_state=True)
     → PPO가 현재 대출 조건 인식 → 적응적 r 결정 가능
  2. Multi-condition 학습: 매 reset마다 (L, cap) 다양한 조건 random
     → 학습된 정책이 평가 환경에 일반화
  3. 비례 페널티 (X1): η × violation_ratio
     → 침해 강도까지 회피하도록 학습
  4. Gamma 조정: 0.99 → 0.95 (후반 보상 가중)

목표 (R_p10 81명에서):
  - Completion ≥ 80% (CVaR 100% 수준 도전)
  - 단일월 최대비 ≤ 1.0 (CVaR 0.99 수준 도전)

산출:
  Project/models/ppo_v2/ppo_final.zip
  Data/ppo_v2_eval_results.csv (평가 셀러 260명, 기존 환경)
  Data/ppo_v2_train_summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from envs.rbf_env import RBFEnv

ROOT = Path("/Users/eoseungyun/Desktop/project/SW_Capstone/Project")
DATA = ROOT / "Data"
MODELS = ROOT / "models" / "ppo_v2"
MODELS.mkdir(parents=True, exist_ok=True)

SEED = 42

# Multi-condition 학습 분포 (실제 RBF 시장 + 본 연구 (L*, cap*) 범위 커버)
L_MULTIPLIER_RANGE = (1.5, 3.5)   # L = R × U(1.5, 3.5) — 작은~큰 대출 모두
CAP_RANGE = (1.0, 1.3)             # cap U(1.0, 1.3) — 무이자~30%이자

# Reward (비례 페널티)
ETA = 3.0
GAMMA = 0.95   # 0.99 → 0.95


class MultiConditionWrapper(gym.Wrapper):
    """학습 중 매 reset마다 (L, cap) random sampling. T는 self.T 그대로."""
    def __init__(self, env, L_multiplier_range=L_MULTIPLIER_RANGE,
                 cap_range=CAP_RANGE, seed=None):
        super().__init__(env)
        self.L_multiplier_range = L_multiplier_range
        self.cap_range = cap_range
        self.rng = np.random.default_rng(seed)

    def reset(self, *, seed=None, options=None):
        options = dict(options) if options else {}
        # eval 시 명시적 L_override가 있으면 그대로 사용
        if "L_override" not in options:
            # 셀러 ID 결정 (env가 정함, 우리는 mean_rev 기반 random L)
            # env.reset 이전이라 current_seller 미정 → 일단 sampling, env가 적용
            # 트릭: env reset 후 L 재계산. 간단히 multiplier만 전달
            mult = float(self.rng.uniform(*self.L_multiplier_range))
            options["_L_multiplier_random"] = mult
        if "cap_override" not in options:
            options["cap_override"] = float(self.rng.uniform(*self.cap_range))
        return self.env.reset(seed=seed, options=options)


class RBFEnvMultiCondition(RBFEnv):
    """reset에서 _L_multiplier_random 처리 (셀러 결정 후 L 적용)."""
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        opts = dict(options) if options else {}
        L_mult_random = opts.pop("_L_multiplier_random", None)

        # 부모 reset이 sellers 선정 + L 계산까지 함
        state, info = super().reset(seed=seed, options=opts)

        # L_mult_random 있으면 L 재설정
        if L_mult_random is not None:
            self.L = L_mult_random * self.current_seller["mean_rev"]
            self.target = self.L * self.episode_cap
            info["L"] = self.L
            info["target"] = self.target
            # state 재산출 (L 변경 반영)
            state = self._get_state()
        return state, info


def split_sellers(test_ratio=0.2, seed=SEED):
    df = pd.read_parquet(DATA / "cohort_kr_v2.parquet")
    sids = df["seller_id"].unique().tolist()
    rng = np.random.default_rng(seed)
    rng.shuffle(sids)
    n_test = int(len(sids) * test_ratio)
    return sids[n_test:], sids[:n_test]


def make_train_env(seller_ids, seed=SEED, monitor_path=None):
    def _init():
        env = RBFEnvMultiCondition(
            seller_ids=seller_ids, seed=seed,
            eta=ETA,
            eta_proportional=True,    # X1
            use_lct_state=True,        # Day 12 state 확장
        )
        env = MultiConditionWrapper(env, seed=seed + 999)
        if monitor_path:
            env = Monitor(env, str(monitor_path))
        return env
    return _init


def make_eval_env(seller_ids, seed=SEED):
    """평가용: 기본 (L=3R, cap=1.2) — overfit 진단용."""
    def _init():
        return RBFEnvMultiCondition(
            seller_ids=seller_ids, seed=seed,
            eta=ETA,
            eta_proportional=True,
            use_lct_state=True,
        )
    return _init


def train(timesteps=300_000):
    print(f"[1/4] 셀러 분할 + Multi-condition 환경")
    train_ids, eval_ids = split_sellers()
    print(f"  학습: {len(train_ids)}, 평가: {len(eval_ids)}")
    print(f"  ★ State 확장 (use_lct_state=True)")
    print(f"  ★ Multi-condition: L=R×U{L_MULTIPLIER_RANGE}, cap=U{CAP_RANGE}")
    print(f"  ★ 비례 페널티 (eta_proportional=True, η={ETA})")
    print(f"  ★ gamma={GAMMA}")

    print(f"\n[2/4] Vec env")
    train_env = DummyVecEnv([make_train_env(train_ids, seed=SEED,
                                              monitor_path=MODELS / "monitor.csv")])
    eval_env = DummyVecEnv([make_eval_env(eval_ids, seed=SEED + 1)])

    obs = train_env.reset()
    print(f"  state shape: {obs.shape}")

    print(f"\n[3/4] PPO 학습 ({timesteps:,} timesteps)")
    model = PPO(
        "MlpPolicy", train_env,
        learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10,
        gamma=GAMMA, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
        verbose=0, seed=SEED, device="auto",
    )
    eval_cb = EvalCallback(
        eval_env, best_model_save_path=str(MODELS / "best"),
        log_path=str(MODELS / "eval_logs"),
        eval_freq=5_000, n_eval_episodes=50,
        deterministic=True, render=False,
    )
    t0 = time.time()
    model.learn(total_timesteps=timesteps, callback=eval_cb, progress_bar=False)
    elapsed = time.time() - t0
    print(f"  학습 시간: {elapsed/60:.1f}분")

    print(f"\n[4/4] 모델 저장 + 평가 (기본 환경 (L=3R, cap=1.2))")
    model.save(str(MODELS / "ppo_final.zip"))

    eval_df = pd.DataFrame(evaluate_full(model, eval_ids))
    eval_df.to_csv(DATA / "ppo_v2_eval_results.csv", index=False)
    print(f"  [save] ppo_v2_eval_results.csv")

    summary = summarize(eval_df, train_ids, eval_ids, elapsed, timesteps)
    (DATA / "ppo_v2_train_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))

    base_path = DATA / "ppo_train_summary.json"
    if base_path.exists():
        base = json.loads(base_path.read_text())["eval"]
        print(f"\n=== PPO v2 vs PPO base (기본 환경 평가 셀러 260명) ===")
        cur = summary["eval"]
        print(f"  {'지표':32s} {'base':>10s} {'v2':>10s} {'변화':>10s}")
        for k, label in [("completion_rate", "Completion %"),
                          ("mean_recovery", "Mean Recovery"),
                          ("mean_burden", "Mean Burden"),
                          ("mean_household_violation_months", "HH 침해 월수"),
                          ("household_violation_zero_rate", "HH 안전 %"),
                          ("mean_violation_ratio_max", "[A-2] 단일월 최대비")]:
            b = base.get(k, float("nan"))
            c = cur.get(k, float("nan"))
            diff = c - b if not (np.isnan(b) or np.isnan(c)) else float("nan")
            print(f"  {label:32s} {b:10.3f} {c:10.3f} {diff:+10.3f}")

    print(f"\n=== PPO v2 학습 완료 — 다음: optimal_lt_cap 평가에서 비교 ===")


def evaluate_full(model, seller_ids):
    env = RBFEnvMultiCondition(
        seller_ids=seller_ids, seed=SEED + 100,
        eta=ETA, eta_proportional=True, use_lct_state=True,
    )
    rows = []
    for sid in seller_ids:
        obs, info_init = env.reset(options={"seller_id": sid})
        total_reward = 0.0
        burden_sum, burden_count = 0.0, 0
        hh_v, hh_amt, hh_ratio_max = 0, 0.0, 0.0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if info["burden"] > 0:
                burden_sum += info["burden"]; burden_count += 1
            if info.get("household_violated", False):
                hh_v += 1
            hh_amt += info.get("household_violation_amount", 0.0)
            hh_ratio_max = max(hh_ratio_max, info.get("household_violation_ratio", 0.0))
            if terminated:
                break
        log = env.get_episode_log()
        rows.append(dict(
            seller_id=sid, type=info_init["type"],
            final_recovery=float(log[-1]["recovery_progress"]),
            completed=bool(log[-1]["recovery_progress"] >= 1.0),
            burden_mean=burden_sum / max(burden_count, 1),
            burden_months=burden_count,
            household_violation_count=hh_v,
            household_violation_amount_total=hh_amt,
            household_violation_ratio_max=hh_ratio_max,
            total_reward=total_reward,
        ))
    return rows


def summarize(df, train_ids, eval_ids, elapsed, timesteps):
    return {
        "config": {
            "n_train": len(train_ids), "n_eval": len(eval_ids),
            "timesteps": timesteps, "elapsed_min": round(elapsed/60, 2),
            "eta": ETA, "eta_proportional": True, "gamma": GAMMA,
            "use_lct_state": True,
            "multi_condition": True,
            "L_multiplier_range": list(L_MULTIPLIER_RANGE),
            "cap_range": list(CAP_RANGE),
            "reward_design": "PPO_v2_multi_condition_state_ext",
        },
        "eval": {
            "n": int(len(df)),
            "mean_recovery": float(df["final_recovery"].mean()),
            "completion_rate": float(df["completed"].mean() * 100),
            "mean_burden": float(df["burden_mean"].mean()),
            "mean_household_violation_months": float(df["household_violation_count"].mean()),
            "household_violation_zero_rate": float((df["household_violation_count"] == 0).mean() * 100),
            "mean_violation_amount_total": float(df["household_violation_amount_total"].mean()),
            "mean_violation_ratio_max": float(df["household_violation_ratio_max"].mean()),
            "p95_violation_ratio_max": float(df["household_violation_ratio_max"].quantile(0.95)),
            "mean_reward": float(df["total_reward"].mean()),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=300_000)
    args = parser.parse_args()
    train(timesteps=args.timesteps)


if __name__ == "__main__":
    main()
