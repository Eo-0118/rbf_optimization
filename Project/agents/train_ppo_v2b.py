"""Day 12 (Phase D): PPO v2b — Multi-condition + State 확장 (비례 페널티 제거)

v2 결과: 비례 페널티(X1) + multi-condition 결합이 너무 보수적 → completion 0%
v2b 변경: 비례 페널티 제거. State 확장 + multi-condition만 활성.
  → PPO가 환경 인식 + 다양한 조건 일반화. 회수 적극성 유지.

목표 (R_p10 81명에서):
  - Completion ≥ 80%
  - 단일월 최대비 ≤ 1.5 (현실적, CVaR 0.99 도전)

산출:
  Project/models/ppo_v2b/ppo_final.zip
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
from agents.train_ppo_v2 import (RBFEnvMultiCondition, MultiConditionWrapper,
                                    split_sellers, evaluate_full,
                                    L_MULTIPLIER_RANGE, CAP_RANGE)

ROOT = Path("/Users/eoseungyun/Desktop/project/SW_Capstone/Project")
DATA = ROOT / "Data"
MODELS = ROOT / "models" / "ppo_v2b"
MODELS.mkdir(parents=True, exist_ok=True)

SEED = 42
ETA = 1.0   # 기본 (비례 X)
GAMMA = 0.95


def make_train_env(seller_ids, seed=SEED, monitor_path=None):
    def _init():
        env = RBFEnvMultiCondition(
            seller_ids=seller_ids, seed=seed,
            eta=ETA,
            eta_proportional=False,   # ★ v2 vs v2b 차이: X1 비활성
            use_lct_state=True,
        )
        env = MultiConditionWrapper(env, seed=seed + 999)
        if monitor_path:
            env = Monitor(env, str(monitor_path))
        return env
    return _init


def make_eval_env(seller_ids, seed=SEED):
    def _init():
        return RBFEnvMultiCondition(
            seller_ids=seller_ids, seed=seed,
            eta=ETA, eta_proportional=False,
            use_lct_state=True,
        )
    return _init


def train(timesteps=300_000):
    print(f"[v2b] State 확장 + Multi-condition + 기본 페널티 (X1=False)")
    train_ids, eval_ids = split_sellers()
    print(f"  학습: {len(train_ids)}, 평가: {len(eval_ids)}")

    train_env = DummyVecEnv([make_train_env(train_ids, seed=SEED,
                                              monitor_path=MODELS / "monitor.csv")])
    eval_env = DummyVecEnv([make_eval_env(eval_ids, seed=SEED + 1)])

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

    # 평가
    env_eval = RBFEnvMultiCondition(
        seller_ids=eval_ids, seed=SEED + 100,
        eta=ETA, eta_proportional=False, use_lct_state=True,
    )
    rows = []
    for sid in eval_ids:
        obs, info_init = env_eval.reset(options={"seller_id": sid})
        total_reward = 0.0
        burden_sum, burden_count = 0.0, 0
        hh_v, hh_amt, hh_ratio_max = 0, 0.0, 0.0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env_eval.step(action)
            total_reward += reward
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
            burden_months=burden_count,
            household_violation_count=hh_v,
            household_violation_amount_total=hh_amt,
            household_violation_ratio_max=hh_ratio_max,
            total_reward=total_reward,
        ))
    eval_df = pd.DataFrame(rows)
    eval_df.to_csv(DATA / "ppo_v2b_eval_results.csv", index=False)

    summary = {
        "config": {
            "n_train": len(train_ids), "n_eval": len(eval_ids),
            "timesteps": timesteps, "elapsed_min": round(elapsed/60, 2),
            "eta": ETA, "eta_proportional": False, "gamma": GAMMA,
            "use_lct_state": True,
            "multi_condition": True,
            "L_multiplier_range": list(L_MULTIPLIER_RANGE),
            "cap_range": list(CAP_RANGE),
            "reward_design": "PPO_v2b_state_ext_multi_no_x1",
        },
        "eval": {
            "n": int(len(eval_df)),
            "mean_recovery": float(eval_df["final_recovery"].mean()),
            "completion_rate": float(eval_df["completed"].mean() * 100),
            "mean_burden": float(eval_df["burden_mean"].mean()),
            "mean_household_violation_months": float(eval_df["household_violation_count"].mean()),
            "mean_violation_amount_total": float(eval_df["household_violation_amount_total"].mean()),
            "mean_violation_ratio_max": float(eval_df["household_violation_ratio_max"].mean()),
            "p95_violation_ratio_max": float(eval_df["household_violation_ratio_max"].quantile(0.95)),
            "mean_reward": float(eval_df["total_reward"].mean()),
        },
    }
    (DATA / "ppo_v2b_train_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))

    base = json.loads((DATA / "ppo_train_summary.json").read_text())["eval"]
    print(f"\n=== PPO v2b vs PPO base (기본 환경 260명) ===")
    cur = summary["eval"]
    print(f"  {'지표':30s} {'base':>10s} {'v2b':>10s} {'변화':>10s}")
    for k, label in [("completion_rate", "Completion %"),
                      ("mean_burden", "Mean Burden"),
                      ("mean_household_violation_months", "HH 침해 월수"),
                      ("mean_violation_ratio_max", "최대 침해비")]:
        b = base.get(k, float("nan"))
        c = cur.get(k, float("nan"))
        diff = c - b if not (np.isnan(b) or np.isnan(c)) else float("nan")
        print(f"  {label:30s} {b:10.3f} {c:10.3f} {diff:+10.3f}")
    print(f"\n=== PPO v2b 학습 완료 ===")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=300_000)
    args = parser.parse_args()
    train(timesteps=args.timesteps)


if __name__ == "__main__":
    main()
