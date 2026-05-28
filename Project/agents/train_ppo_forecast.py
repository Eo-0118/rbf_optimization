"""Lv4 (Day 2-3): 시계열 분포 → RBF state 통합 PPO 학습

목적:
  보고서 원안 ([interim_report.md:14, 548]) 완성:
    "시계열 예측 분포 → RBF state로 통합"

  현재 PPO state는 과거 정보만 (t/T, recovery, 최근 3개월 매출, type, m_i, scale).
  여기에 lag-based 분포 (P10, P50, P90) 추가 → PPO가 매출 불확실성 정보로 활용.

분포 통합 방식:
  - Lag-based 분위수 (모델 학습 없이 즉시 계산)
  - rbf_env.py의 use_forecast_state=True
  - state_dim: 13 → 16 (P10/P50/P90 추가)

비교 baseline:
  - 기존 PPO (Data/ppo_eval_results.csv, Data/ppo_eval_full_results.csv)
  - 본 학습 결과 (Data/ppo_forecast_eval_results.csv)

산출:
  - Project/models/ppo_forecast/ppo_final.zip (학습된 모델)
  - Data/ppo_forecast_eval_results.csv (평가 셀러 260명)
  - Data/ppo_forecast_train_summary.json
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
MODELS = ROOT / "models" / "ppo_forecast"
MODELS.mkdir(parents=True, exist_ok=True)

SEED = 42


def split_sellers(test_ratio: float = 0.2, seed: int = SEED) -> tuple[list[str], list[str]]:
    """train_ppo.py와 동일한 분할 (재현)."""
    df = pd.read_parquet(DATA / "cohort_kr_v2.parquet")
    sids = df["seller_id"].unique().tolist()
    rng = np.random.default_rng(seed)
    rng.shuffle(sids)
    n_test = int(len(sids) * test_ratio)
    test_ids = sids[:n_test]
    train_ids = sids[n_test:]
    return train_ids, test_ids


def make_env(seller_ids: list[str], seed: int = SEED, monitor_path: Path | None = None):
    def _init():
        env = RBFEnv(seller_ids=seller_ids, seed=seed,
                     use_forecast_state=True,    # Lv4 신규
                     forecast_lag=6)
        if monitor_path:
            env = Monitor(env, str(monitor_path))
        return env
    return _init


def train(timesteps: int = 200_000, learning_rate: float = 3e-4, n_steps: int = 2048,
          batch_size: int = 64, gamma: float = 0.99):
    print(f"[1/4] 셀러 분할 (80/20)")
    train_ids, eval_ids = split_sellers()
    print(f"  학습: {len(train_ids)} 셀러 / 평가: {len(eval_ids)} 셀러")
    print(f"  ★ use_forecast_state=True (state_dim 13 → 16)")

    print(f"\n[2/4] Vec env 생성 (forecast state 활성)")
    train_env = DummyVecEnv([make_env(train_ids, seed=SEED, monitor_path=MODELS / "monitor.csv")])
    eval_env = DummyVecEnv([make_env(eval_ids, seed=SEED + 1)])

    # state_dim 확인
    obs = train_env.reset()
    print(f"  state shape: {obs.shape}")

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

    print(f"\n[4/4] 모델 저장 + 평가")
    final_path = MODELS / "ppo_final.zip"
    model.save(str(final_path))
    print(f"  [save] {final_path}")

    eval_results = evaluate_policy_full(model, eval_ids)
    eval_df = pd.DataFrame(eval_results)
    eval_df.to_csv(DATA / "ppo_forecast_eval_results.csv", index=False)
    print(f"  [save] ppo_forecast_eval_results.csv")

    summary = summarize_eval(eval_df, train_ids, eval_ids, elapsed, timesteps)
    (DATA / "ppo_forecast_train_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))

    # 기존 PPO와 즉시 비교
    base_path = DATA / "ppo_train_summary.json"
    if base_path.exists():
        base = json.loads(base_path.read_text())["eval"]
        print(f"\n=== 분포 통합 vs 기존 PPO 비교 ===")
        cur = summary["eval"]
        print(f"  {'지표':32s} {'기존':>10s} {'분포통합':>10s} {'변화':>10s}")
        for k, label in [("completion_rate", "Completion %"),
                         ("mean_recovery", "Mean Recovery"),
                         ("mean_burden", "Mean Burden"),
                         ("mean_household_violation_months", "HH 침해 월수"),
                         ("household_violation_zero_rate", "HH 안전 셀러 %"),
                         ("mean_reward", "Mean Reward")]:
            b = base.get(k, float("nan"))
            c = cur.get(k, float("nan"))
            print(f"  {label:32s} {b:10.3f} {c:10.3f} {c - b:+10.3f}")

    print(f"\n=== 학습 완료 ===")


def evaluate_policy_full(model, seller_ids: list[str]) -> list[dict]:
    """학습된 PPO 모델을 평가 셀러에 deterministic 적용.
    A-2 침해 액도 지표도 함께 수집.
    """
    env = RBFEnv(seller_ids=seller_ids, seed=SEED + 100,
                 use_forecast_state=True, forecast_lag=6)
    rows = []
    for sid in seller_ids:
        obs, info_init = env.reset(options={"seller_id": sid})
        total_reward = 0.0
        burden_sum, burden_count = 0.0, 0
        hh_violations = 0
        hh_violation_amount_sum = 0.0
        hh_violation_ratio_max = 0.0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if info["burden"] > 0:
                burden_sum += info["burden"]
                burden_count += 1
            if info.get("household_violated", False):
                hh_violations += 1
            hh_violation_amount_sum += info.get("household_violation_amount", 0.0)
            hh_violation_ratio_max = max(hh_violation_ratio_max,
                                          info.get("household_violation_ratio", 0.0))
            if terminated:
                break
        log = env.get_episode_log()
        final_recovery = log[-1]["recovery_progress"]
        completed = final_recovery >= 1.0
        rows.append(dict(
            seller_id=sid, type=info_init["type"],
            final_recovery=final_recovery, completed=completed,
            burden_mean=burden_sum / max(burden_count, 1),
            burden_months=burden_count,
            household_violation_count=hh_violations,
            household_violation_amount_total=hh_violation_amount_sum,
            household_violation_ratio_max=hh_violation_ratio_max,
            total_reward=total_reward,
        ))
    return rows


def summarize_eval(df: pd.DataFrame, train_ids, eval_ids, elapsed, timesteps) -> dict:
    return {
        "config": {
            "n_train": len(train_ids), "n_eval": len(eval_ids),
            "timesteps": timesteps, "elapsed_min": round(elapsed / 60, 2),
            "use_forecast_state": True, "forecast_lag": 6,
            "state_dim": 16,
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
        "by_type": {
            typ: dict(
                n=int(len(g)),
                completion_rate=float(g["completed"].mean() * 100),
                mean_recovery=float(g["final_recovery"].mean()),
                mean_burden=float(g["burden_mean"].mean()),
                mean_violation_amount_total=float(g["household_violation_amount_total"].mean()),
            )
            for typ, g in df.groupby("type")
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()
    train(timesteps=args.timesteps, learning_rate=args.lr)


if __name__ == "__main__":
    main()
