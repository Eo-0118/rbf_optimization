"""Phase E-3 (마지막 시도): SAC 알고리즘 학습 — PPO 대안

가설:
  PPO가 8가지 시도 모두 CVaR 못 능가 → SAC (entropy regularization)이 다를까?
  SAC 특성:
    - Continuous control에 PPO보다 강함
    - Entropy 항으로 탐색 강함 (local optima 회피)
    - Sample efficient (off-policy)

설정:
  환경: cohort_kr_v2 + state 확장 + multi-condition (PPO v2b와 동일)
  비례 페널티 X (X1=False, eta=1.0 기본)
  학습: 300k step

목표 (R_p10 81명):
  Completion ≥ 80% (CVaR 100% 도전)
  단일월 최대비 ≤ 1.5 (CVaR 0.99 도전)

산출:
  Project/models/sac/sac_final.zip
  Data/sac_eval_results.csv
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from envs.rbf_env import RBFEnv
from agents.train_ppo_v2 import (RBFEnvMultiCondition, MultiConditionWrapper,
                                    split_sellers, L_MULTIPLIER_RANGE, CAP_RANGE)

ROOT = Path("/Users/eoseungyun/Desktop/project/SW_Capstone/Project")
DATA = ROOT / "Data"
MODELS = ROOT / "models" / "sac"
MODELS.mkdir(parents=True, exist_ok=True)

SEED = 42
ETA = 1.0
GAMMA = 0.95


def make_train_env(seller_ids, seed=SEED, monitor_path=None):
    def _init():
        env = RBFEnvMultiCondition(
            seller_ids=seller_ids, seed=seed,
            eta=ETA, eta_proportional=False,
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
    print(f"[SAC 학습] State 확장 + Multi-condition + 기본 페널티")
    train_ids, eval_ids = split_sellers()
    print(f"  학습: {len(train_ids)}, 평가: {len(eval_ids)}")
    print(f"  ★ Algorithm: SAC (entropy regularization)")

    train_env = DummyVecEnv([make_train_env(train_ids, seed=SEED,
                                              monitor_path=MODELS / "monitor.csv")])
    eval_env = DummyVecEnv([make_eval_env(eval_ids, seed=SEED + 1)])

    model = SAC(
        "MlpPolicy", train_env,
        learning_rate=3e-4,
        buffer_size=100_000,
        batch_size=256,
        gamma=GAMMA,
        tau=0.005,
        train_freq=1,
        gradient_steps=1,
        verbose=0,
        seed=SEED,
        device="auto",
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

    model.save(str(MODELS / "sac_final.zip"))
    print(f"  [save] {MODELS / 'sac_final.zip'}")

    # 평가 (기본 환경)
    env_eval = RBFEnvMultiCondition(
        seller_ids=eval_ids, seed=SEED + 100,
        eta=ETA, eta_proportional=False, use_lct_state=True,
    )
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
            burden_months=burden_count,
            household_violation_count=hh_v,
            household_violation_amount_total=hh_amt,
            household_violation_ratio_max=hh_ratio_max,
        ))
    eval_df = pd.DataFrame(rows)
    eval_df.to_csv(DATA / "sac_eval_results.csv", index=False)

    summary = {
        "config": {
            "n_train": len(train_ids), "n_eval": len(eval_ids),
            "timesteps": timesteps, "elapsed_min": round(elapsed/60, 2),
            "algo": "SAC", "gamma": GAMMA, "eta": ETA,
            "use_lct_state": True, "multi_condition": True,
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
    (DATA / "sac_train_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))

    base = json.loads((DATA / "ppo_train_summary.json").read_text())["eval"]
    print(f"\n=== SAC vs PPO base (기본 환경 260명) ===")
    cur = summary["eval"]
    print(f"  {'지표':30s} {'PPO base':>10s} {'SAC':>10s} {'변화':>10s}")
    for k, label in [("completion_rate", "Completion %"),
                      ("mean_burden", "Mean Burden"),
                      ("mean_household_violation_months", "HH 침해 월수"),
                      ("mean_violation_ratio_max", "단일월 최대비")]:
        b = base.get(k, float("nan"))
        c = cur.get(k, float("nan"))
        diff = c - b if not (np.isnan(b) or np.isnan(c)) else float("nan")
        print(f"  {label:30s} {b:10.3f} {c:10.3f} {diff:+10.3f}")
    print(f"\n=== SAC 학습 완료 ===")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=300_000)
    args = parser.parse_args()
    train(timesteps=args.timesteps)


if __name__ == "__main__":
    main()
