"""Lv4 (Day 4): A-3 Reward 재설계 — eta (가계 침범 페널티) 강화

배경 (A-2 발견):
  PPO: 침해 월수 12.2 (낮음) but 단일월 최대비 1.73 (높음), 상위 5% 셀러 3.67
  → "침해 빈도는 줄이되 침해 정도는 강한" 정책 학습
  원인: reward 가중치에서 가계 침범 페널티 약함 (eta=1.0, terminal -10 디폴트 대비)

A-3 변경:
  eta: 1.0 → 5.0 (5배 강화)
  → 가계 침범 단일 월 시 -5.0 페널티 → PPO가 침해 정도까지 회피하도록 유도

비교:
  - 기존 PPO (eta=1.0): completion 91.2%, 침해월 12.2, 최대비 1.73
  - 본 (eta=5.0): 목표 침해 정도 ↓, 회수율 일부 손해 trade-off 예상

산출:
  - Project/models/ppo_eta_strong/ppo_final.zip
  - Data/ppo_eta_strong_eval_results.csv
  - Data/ppo_eta_strong_train_summary.json
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

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from envs.rbf_env import RBFEnv

ROOT = Path("/Users/eoseungyun/Desktop/project/SW_Capstone/Project")
DATA = ROOT / "Data"
MODELS = ROOT / "models" / "ppo_eta_strong"
MODELS.mkdir(parents=True, exist_ok=True)

SEED = 42
ETA_STRONG = 5.0   # 1.0 → 5.0


def split_sellers(test_ratio: float = 0.2, seed: int = SEED):
    df = pd.read_parquet(DATA / "cohort_kr_v2.parquet")
    sids = df["seller_id"].unique().tolist()
    rng = np.random.default_rng(seed)
    rng.shuffle(sids)
    n_test = int(len(sids) * test_ratio)
    return sids[n_test:], sids[:n_test]


def make_env(seller_ids, seed=SEED, monitor_path=None):
    def _init():
        env = RBFEnv(seller_ids=seller_ids, seed=seed, eta=ETA_STRONG)
        if monitor_path:
            env = Monitor(env, str(monitor_path))
        return env
    return _init


def train(timesteps=200_000, learning_rate=3e-4, n_steps=2048, batch_size=64, gamma=0.99):
    print(f"[1/4] 셀러 분할")
    train_ids, eval_ids = split_sellers()
    print(f"  학습: {len(train_ids)}, 평가: {len(eval_ids)}")
    print(f"  ★ eta = {ETA_STRONG} (기존 1.0 → 5배 강화)")

    print(f"\n[2/4] Vec env (eta 강화)")
    train_env = DummyVecEnv([make_env(train_ids, seed=SEED, monitor_path=MODELS / "monitor.csv")])
    eval_env = DummyVecEnv([make_env(eval_ids, seed=SEED + 1)])

    print(f"\n[3/4] PPO 학습 ({timesteps:,} timesteps)")
    model = PPO(
        "MlpPolicy", train_env,
        learning_rate=learning_rate, n_steps=n_steps, batch_size=batch_size,
        n_epochs=10, gamma=gamma, gae_lambda=0.95, clip_range=0.2,
        ent_coef=0.01, verbose=1, seed=SEED, device="auto",
    )
    eval_callback = EvalCallback(
        eval_env, best_model_save_path=str(MODELS / "best"),
        log_path=str(MODELS / "eval_logs"),
        eval_freq=5_000, n_eval_episodes=50,
        deterministic=True, render=False,
    )
    t0 = time.time()
    model.learn(total_timesteps=timesteps, callback=eval_callback, progress_bar=False)
    elapsed = time.time() - t0
    print(f"  학습 시간: {elapsed/60:.1f}분")

    print(f"\n[4/4] 평가")
    model.save(str(MODELS / "ppo_final.zip"))

    eval_df = pd.DataFrame(evaluate_policy_full(model, eval_ids))
    eval_df.to_csv(DATA / "ppo_eta_strong_eval_results.csv", index=False)
    print(f"  [save] ppo_eta_strong_eval_results.csv")

    summary = summarize(eval_df, train_ids, eval_ids, elapsed, timesteps)
    (DATA / "ppo_eta_strong_train_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))

    # 즉시 비교
    base_path = DATA / "ppo_train_summary.json"
    if base_path.exists():
        base = json.loads(base_path.read_text())["eval"]
        print(f"\n=== eta 강화 vs 기존 PPO 비교 ===")
        cur = summary["eval"]
        print(f"  {'지표':32s} {'기존(eta=1)':>14s} {'강화(eta=5)':>14s} {'변화':>10s}")
        for k, label in [("completion_rate", "Completion %"),
                         ("mean_recovery", "Mean Recovery"),
                         ("mean_burden", "Mean Burden"),
                         ("mean_household_violation_months", "HH 침해 월수"),
                         ("household_violation_zero_rate", "HH 안전 셀러 %"),
                         ("mean_violation_amount_total", "[A-2] 누적 침해액"),
                         ("mean_violation_ratio_max", "[A-2] 단일월 최대비"),
                         ("mean_reward", "Mean Reward")]:
            b = base.get(k, cur.get(k, float("nan")))
            c = cur.get(k, float("nan"))
            print(f"  {label:32s} {b:14.3f} {c:14.3f} {c-b:+10.3f}")

    print(f"\n=== 학습 완료 ===")


def evaluate_policy_full(model, seller_ids):
    env = RBFEnv(seller_ids=seller_ids, seed=SEED + 100, eta=ETA_STRONG)
    rows = []
    for sid in seller_ids:
        obs, info_init = env.reset(options={"seller_id": sid})
        total_reward = 0.0
        burden_sum, burden_count = 0.0, 0
        hh_violations = 0
        hh_amt_sum = 0.0
        hh_ratio_max = 0.0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if info["burden"] > 0:
                burden_sum += info["burden"]
                burden_count += 1
            if info.get("household_violated", False):
                hh_violations += 1
            hh_amt_sum += info.get("household_violation_amount", 0.0)
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
            household_violation_count=hh_violations,
            household_violation_amount_total=hh_amt_sum,
            household_violation_ratio_max=hh_ratio_max,
            total_reward=total_reward,
        ))
    return rows


def summarize(df, train_ids, eval_ids, elapsed, timesteps):
    return {
        "config": {
            "n_train": len(train_ids), "n_eval": len(eval_ids),
            "timesteps": timesteps, "elapsed_min": round(elapsed/60, 2),
            "eta": ETA_STRONG, "reward_design": "eta_strong_5x",
        },
        "eval": {
            "n": int(len(df)),
            "mean_recovery": float(df["final_recovery"].mean()),
            "completion_rate": float(df["completed"].mean() * 100),
            "mean_burden": float(df["burden_mean"].mean()),
            "mean_burden_months": float(df["burden_months"].mean()),
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
    parser.add_argument("--timesteps", type=int, default=200_000)
    args = parser.parse_args()
    train(timesteps=args.timesteps)


if __name__ == "__main__":
    main()
