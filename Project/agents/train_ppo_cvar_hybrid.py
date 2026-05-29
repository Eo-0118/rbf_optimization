"""Phase E (Day 12+): CVaR-PPO 하이브리드

가설:
  CVaR이 정적으로 우수 → CVaR을 baseline으로 두고 PPO가 매월 작은 delta로 조정
  → "정적 CVaR 안전성" + "동적 PPO 적응성" 결합

구현:
  Action wrapper: PPO action ∈ [-1, 1]
    → effective r_t = clip(cvar_r*(seller) + action × delta_scale, r_min, r_max)
  delta_scale = 0.05 (CVaR r* 주변 ±5%p 조정 허용)

목표:
  - CVaR 안전성 (단일월 최대비 ≤ 1.0) 유지
  - 매월 매출 변동에 따라 적응 → completion 또는 침해 추가 개선
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
from gymnasium import spaces

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from envs.rbf_env import RBFEnv
from agents.train_ppo_v2 import (RBFEnvMultiCondition, MultiConditionWrapper,
                                    split_sellers, L_MULTIPLIER_RANGE, CAP_RANGE)

ROOT = Path("/Users/eoseungyun/Desktop/project/SW_Capstone/Project")
DATA = ROOT / "Data"
MODELS = ROOT / "models" / "ppo_cvar_hybrid"
MODELS.mkdir(parents=True, exist_ok=True)

SEED = 42
DELTA_SCALE = 0.05    # PPO action × delta_scale = r 조정 (±5%p)
GAMMA = 0.95


class CVaRHybridWrapper(gym.Wrapper):
    """PPO action을 CVaR r*에 대한 delta로 변환.

    Action space: [-1, 1] (정규화)
    Effective r_t = clip(cvar_r* + action × delta_scale, r_min, r_max)
    """
    def __init__(self, env, cvar_r_lookup: dict, delta_scale: float = DELTA_SCALE):
        super().__init__(env)
        self.cvar_r = cvar_r_lookup
        self.delta_scale = delta_scale
        # Override action space: [-1, 1]
        self.action_space = spaces.Box(low=np.array([-1.0], dtype=np.float32),
                                         high=np.array([1.0], dtype=np.float32),
                                         dtype=np.float32)
        self._default_r = 0.10

    def step(self, action):
        sid = self.env.unwrapped.current_seller_id
        base_r = self.cvar_r.get(sid, self._default_r)
        delta = float(action[0]) * self.delta_scale
        r_min = self.env.unwrapped.r_min
        r_max = self.env.unwrapped.r_max
        effective_r = float(np.clip(base_r + delta, r_min, r_max))
        return self.env.step(np.array([effective_r], dtype=np.float32))


def load_cvar_lookup():
    df = pd.read_csv(DATA / "cvar_optimizer_results.csv")
    return dict(zip(df["seller_id"], df["r_star"]))


def make_train_env(seller_ids, cvar_lookup, seed=SEED, monitor_path=None):
    def _init():
        env = RBFEnvMultiCondition(
            seller_ids=seller_ids, seed=seed,
            eta=1.0,
            eta_proportional=False,
            use_lct_state=True,
        )
        env = MultiConditionWrapper(env, seed=seed + 999)
        env = CVaRHybridWrapper(env, cvar_lookup)
        if monitor_path:
            env = Monitor(env, str(monitor_path))
        return env
    return _init


def make_eval_env(seller_ids, cvar_lookup, seed=SEED):
    def _init():
        env = RBFEnvMultiCondition(
            seller_ids=seller_ids, seed=seed,
            eta=1.0, eta_proportional=False,
            use_lct_state=True,
        )
        env = CVaRHybridWrapper(env, cvar_lookup)
        return env
    return _init


def train(timesteps=300_000):
    print(f"[CVaR-PPO 하이브리드 학습]")
    print(f"  delta_scale = {DELTA_SCALE} (PPO action × ±0.05 = r 조정)")

    cvar_lookup = load_cvar_lookup()
    print(f"  CVaR r* 로드: {len(cvar_lookup)} 셀러")
    print(f"  CVaR r* 분포: min={min(cvar_lookup.values()):.3f}, "
          f"mean={np.mean(list(cvar_lookup.values())):.3f}, "
          f"max={max(cvar_lookup.values()):.3f}")

    train_ids, eval_ids = split_sellers()
    print(f"  학습: {len(train_ids)}, 평가: {len(eval_ids)}")

    train_env = DummyVecEnv([make_train_env(train_ids, cvar_lookup, seed=SEED,
                                              monitor_path=MODELS / "monitor.csv")])
    eval_env = DummyVecEnv([make_eval_env(eval_ids, cvar_lookup, seed=SEED + 1)])

    model = PPO("MlpPolicy", train_env,
                learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10,
                gamma=GAMMA, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
                verbose=0, seed=SEED, device="auto")
    eval_cb = EvalCallback(eval_env, best_model_save_path=str(MODELS / "best"),
                            log_path=str(MODELS / "eval_logs"),
                            eval_freq=5_000, n_eval_episodes=50,
                            deterministic=True, render=False)
    t0 = time.time()
    model.learn(total_timesteps=timesteps, callback=eval_cb, progress_bar=False)
    elapsed = time.time() - t0
    print(f"  학습 시간: {elapsed/60:.1f}분")

    model.save(str(MODELS / "ppo_final.zip"))

    # 평가 (기본 환경)
    env_eval = make_eval_env(eval_ids, cvar_lookup, seed=SEED + 100)()
    rows = []
    for sid in eval_ids:
        obs, info_init = env_eval.reset(options={"seller_id": sid})
        burden_sum, burden_count = 0.0, 0
        hh_v, hh_amt, hh_ratio_max = 0, 0.0, 0.0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env_eval.step(action)
            if info["burden"] > 0:
                burden_sum += info["burden"]; burden_count += 1
            if info.get("household_violated", False):
                hh_v += 1
            hh_amt += info.get("household_violation_amount", 0.0)
            hh_ratio_max = max(hh_ratio_max, info.get("household_violation_ratio", 0.0))
            if terminated:
                break
        log = env_eval.get_episode_log()
        rows.append(dict(
            seller_id=sid, type=info_init["type"],
            final_recovery=float(log[-1]["recovery_progress"]),
            completed=bool(log[-1]["recovery_progress"] >= 1.0),
            burden_mean=burden_sum / max(burden_count, 1),
            household_violation_count=hh_v,
            household_violation_amount_total=hh_amt,
            household_violation_ratio_max=hh_ratio_max,
        ))
    eval_df = pd.DataFrame(rows)
    eval_df.to_csv(DATA / "ppo_cvar_hybrid_eval_results.csv", index=False)

    summary = {
        "config": {
            "delta_scale": DELTA_SCALE, "gamma": GAMMA,
            "timesteps": timesteps, "elapsed_min": round(elapsed/60, 2),
        },
        "eval": {
            "n": int(len(eval_df)),
            "completion_rate": float(eval_df["completed"].mean() * 100),
            "mean_recovery": float(eval_df["final_recovery"].mean()),
            "mean_burden": float(eval_df["burden_mean"].mean()),
            "mean_household_violation_months": float(eval_df["household_violation_count"].mean()),
            "mean_violation_amount_total": float(eval_df["household_violation_amount_total"].mean()),
            "mean_violation_ratio_max": float(eval_df["household_violation_ratio_max"].mean()),
            "p95_violation_ratio_max": float(eval_df["household_violation_ratio_max"].quantile(0.95)),
        },
    }
    (DATA / "ppo_cvar_hybrid_train_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))

    base = json.loads((DATA / "ppo_train_summary.json").read_text())["eval"]
    print(f"\n=== CVaR-PPO Hybrid vs PPO base vs CVaR ===")
    cur = summary["eval"]
    print(f"  {'지표':30s} {'PPO base':>10s} {'CVaR':>8s} {'Hybrid':>10s}")
    # CVaR 통계는 기본 env에서. cvar_lookup이 매월 동일 r 적용 → CVaR과 동등.
    # baseline_summary.csv 또는 hardcoded
    print(f"  Completion %                  {base['completion_rate']:10.1f}  (참고) {cur['completion_rate']:10.1f}")
    print(f"  Mean Burden                   {base['mean_burden']:10.3f}        {cur['mean_burden']:10.3f}")
    print(f"  HH 침해 월수                    {base['mean_household_violation_months']:10.1f}        {cur['mean_household_violation_months']:10.1f}")
    print(f"  단일월 최대비                    {'nan':>10s}        {cur['mean_violation_ratio_max']:10.3f}")
    print(f"\n=== CVaR-PPO Hybrid 학습 완료 ===")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=300_000)
    args = parser.parse_args()
    train(timesteps=args.timesteps)


if __name__ == "__main__":
    main()
