"""PPO RL agent 학습 (Phase 3 v1)

목적:
- env(RBFEnv)에서 매월 r_t를 동적 결정하는 정책 학습
- 학습 셀러로 학습, 평가 셀러로 일반화 측정
- CVaR(정적, 가계 보호 우선) + Fixed-0.15(정적, 회수 우선) 대비
  RL이 동시에 둘 다 개선하는지 검증

학습 흐름:
1. 1302 셀러 80/20 분할 (학습 1041 / 평가 261, seed=42)
2. PPO 학습 (~수십만 timesteps)
3. 학습 완료 후 평가 셀러에 대해 deterministic 정책 평가
4. 결과 + 모델 저장

사용법 (사용자 실행):
  python -m agents.train_ppo
  python -m agents.train_ppo --timesteps 200000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# 어디서 실행해도 envs/ 모듈 찾을 수 있도록 Project 루트 추가
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
MODELS = ROOT / "models" / "ppo"
MODELS.mkdir(parents=True, exist_ok=True)

SEED = 42


def split_sellers(test_ratio: float = 0.2, seed: int = SEED) -> tuple[list[str], list[str]]:
    """셀러를 학습/평가로 분할 (재현 가능)."""
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
        env = RBFEnv(seller_ids=seller_ids, seed=seed)
        if monitor_path:
            env = Monitor(env, str(monitor_path))
        return env
    return _init


def train(timesteps: int = 200_000, learning_rate: float = 3e-4, n_steps: int = 2048,
          batch_size: int = 64, gamma: float = 0.99):
    print(f"[1/4] 셀러 분할 (80/20)")
    train_ids, eval_ids = split_sellers()
    print(f"  학습: {len(train_ids)} 셀러 / 평가: {len(eval_ids)} 셀러")

    print(f"\n[2/4] Vec env 생성")
    train_env = DummyVecEnv([make_env(train_ids, seed=SEED, monitor_path=MODELS / "monitor.csv")])
    eval_env = DummyVecEnv([make_env(eval_ids, seed=SEED + 1)])

    print(f"\n[3/4] PPO 학습 ({timesteps:,} timesteps, lr={learning_rate}, n_steps={n_steps}, batch={batch_size})")
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=10,
        gamma=gamma,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        seed=SEED,
        device="auto",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(MODELS / "best"),
        log_path=str(MODELS / "eval_logs"),
        eval_freq=max(5_000 // 1, 5_000),
        n_eval_episodes=50,
        deterministic=True,
        render=False,
    )

    t0 = time.time()
    model.learn(total_timesteps=timesteps, callback=eval_callback, progress_bar=False)
    elapsed = time.time() - t0
    print(f"  학습 시간: {elapsed/60:.1f}분")

    print(f"\n[4/4] 모델 저장 + 최종 평가")
    final_path = MODELS / "ppo_final.zip"
    model.save(str(final_path))
    print(f"  [save] {final_path}")

    # 평가 (deterministic, 평가 셀러 전체)
    eval_results = evaluate_policy_full(model, eval_ids)
    eval_df = pd.DataFrame(eval_results)
    eval_df.to_csv(DATA / "ppo_eval_results.csv", index=False)
    print(f"  [save] ppo_eval_results.csv")

    summary = summarize_eval(eval_df, train_ids, eval_ids, elapsed, timesteps)
    (DATA / "ppo_train_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"\n=== PPO 학습 완료 ===")
    print(f"  학습 시간: {elapsed/60:.1f}분 / {timesteps:,} timesteps")
    print(f"\n  [평가 셀러 {len(eval_ids)}명 결과]")
    print(f"   completion: {summary['eval']['completion_rate']:.1f}%")
    print(f"   mean recovery: {summary['eval']['mean_recovery']:.3f}")
    print(f"   mean burden: {summary['eval']['mean_burden']:.4f}")
    print(f"   household safe: {summary['eval']['household_violation_zero_rate']:.1f}%")
    print(f"   mean reward: {summary['eval']['mean_reward']:+.3f}")
    print(f"\n  비교 (Fixed-0.15 / CVaR / PPO):")
    print(f"   Fixed-0.15: completion 79.7%, burden 0.118, hh_safe 2.3%")
    print(f"   CVaR:       completion 9.1%,  burden 0.067, hh_safe 2.1%")
    print(f"   PPO (RL):   completion {summary['eval']['completion_rate']:.1f}%, "
          f"burden {summary['eval']['mean_burden']:.3f}, hh_safe {summary['eval']['household_violation_zero_rate']:.1f}%")


def evaluate_policy_full(model, seller_ids: list[str]) -> list[dict]:
    """학습된 PPO 모델을 모든 평가 셀러에 deterministic 적용."""
    env = RBFEnv(seller_ids=seller_ids, seed=SEED + 100)
    rows = []
    for sid in seller_ids:
        obs, info_init = env.reset(options={"seller_id": sid})
        total_reward = 0.0
        burden_sum = 0.0
        burden_count = 0
        hh_violations = 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if info["burden"] > 0:
                burden_sum += info["burden"]
                burden_count += 1
            if info.get("household_violated", False):
                hh_violations += 1
            if terminated:
                break
        log = env.get_episode_log()
        final_recovery = log[-1]["recovery_progress"]
        completed = final_recovery >= 1.0
        rows.append(dict(
            seller_id=sid,
            type=info_init["type"],
            final_recovery=final_recovery,
            completed=completed,
            burden_mean=burden_sum / max(burden_count, 1),
            burden_months=burden_count,
            household_violation_count=hh_violations,
            total_reward=total_reward,
        ))
    return rows


def summarize_eval(df: pd.DataFrame, train_ids, eval_ids, elapsed, timesteps) -> dict:
    summary = {
        "config": {
            "n_train": len(train_ids),
            "n_eval": len(eval_ids),
            "timesteps": timesteps,
            "elapsed_min": round(elapsed / 60, 2),
        },
        "eval": {
            "n": int(len(df)),
            "mean_recovery": float(df["final_recovery"].mean()),
            "median_recovery": float(df["final_recovery"].median()),
            "completion_rate": float(df["completed"].mean() * 100),
            "mean_burden": float(df["burden_mean"].mean()),
            "mean_burden_months": float(df["burden_months"].mean()),
            "mean_household_violation_months": float(df["household_violation_count"].mean()),
            "household_violation_zero_rate": float((df["household_violation_count"] == 0).mean() * 100),
            "mean_reward": float(df["total_reward"].mean()),
        },
        "by_type": {},
    }
    for typ, g in df.groupby("type"):
        summary["by_type"][typ] = dict(
            n=int(len(g)),
            completion_rate=float(g["completed"].mean() * 100),
            mean_recovery=float(g["final_recovery"].mean()),
            mean_burden=float(g["burden_mean"].mean()),
            mean_household_violation_months=float(g["household_violation_count"].mean()),
        )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=200_000, help="Total RL timesteps")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--n_steps", type=int, default=2048, help="PPO n_steps per update")
    parser.add_argument("--batch_size", type=int, default=64, help="PPO batch size")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    args = parser.parse_args()

    train(timesteps=args.timesteps, learning_rate=args.lr, n_steps=args.n_steps,
          batch_size=args.batch_size, gamma=args.gamma)


if __name__ == "__main__":
    main()
