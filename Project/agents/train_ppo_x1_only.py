"""Day 8-2: X1 단독 PPO — η × violation_ratio (binary → 비례)

X1+X2 결과: 회수율 0% (X2가 너무 강력)
→ X1만 적용: 비례 페널티로 침해 강도 줄이되 r는 자유

η 비례 페널티가 충분히 강하면 PPO가 강한 침해를 학습적으로 회피해야 함.
침해 강도 줄이면서 회수율 일정 수준 유지 가능성 측정.
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
MODELS = ROOT / "models" / "ppo_x1_only"
MODELS.mkdir(parents=True, exist_ok=True)

SEED = 42
ETA = 3.0  # 침해비 1.0 (가계비 정확히 1배 침해) 시 -3.0 페널티


def split_sellers(test_ratio=0.2, seed=SEED):
    df = pd.read_parquet(DATA / "cohort_kr_v2.parquet")
    sids = df["seller_id"].unique().tolist()
    rng = np.random.default_rng(seed)
    rng.shuffle(sids)
    n_test = int(len(sids) * test_ratio)
    return sids[n_test:], sids[:n_test]


def make_env(seller_ids, seed=SEED, monitor_path=None):
    def _init():
        env = RBFEnv(
            seller_ids=seller_ids, seed=seed,
            eta=ETA,
            eta_proportional=True,    # X1만
            r_clip_to_safe=False,
        )
        if monitor_path:
            env = Monitor(env, str(monitor_path))
        return env
    return _init


def train(timesteps=200_000):
    train_ids, eval_ids = split_sellers()
    print(f"[학습 {len(train_ids)} / 평가 {len(eval_ids)}, X1만 활성, η={ETA}]")

    train_env = DummyVecEnv([make_env(train_ids, seed=SEED, monitor_path=MODELS / "monitor.csv")])
    eval_env = DummyVecEnv([make_env(eval_ids, seed=SEED + 1)])

    model = PPO("MlpPolicy", train_env, learning_rate=3e-4, n_steps=2048, batch_size=64,
                n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
                verbose=0, seed=SEED, device="auto")
    eval_cb = EvalCallback(eval_env, best_model_save_path=str(MODELS / "best"),
                            log_path=str(MODELS / "eval_logs"),
                            eval_freq=5_000, n_eval_episodes=50, deterministic=True, render=False)
    t0 = time.time()
    model.learn(total_timesteps=timesteps, callback=eval_cb, progress_bar=False)
    elapsed = time.time() - t0
    print(f"학습 시간: {elapsed/60:.1f}분")

    model.save(str(MODELS / "ppo_final.zip"))

    eval_df = pd.DataFrame(evaluate_full(model, eval_ids))
    eval_df.to_csv(DATA / "ppo_x1_only_eval_results.csv", index=False)

    summary = summarize(eval_df, train_ids, eval_ids, elapsed, timesteps)
    (DATA / "ppo_x1_only_train_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))

    # 비교 (base + X1+X2)
    print(f"\n=== X1만 vs base vs X1+X2 ===")
    base = json.loads((DATA / "ppo_train_summary.json").read_text())["eval"]
    x1x2 = json.loads((DATA / "ppo_x1x2_train_summary.json").read_text())["eval"]
    cur = summary["eval"]
    print(f"  {'지표':28s} {'base':>10s} {'X1 only':>10s} {'X1+X2':>10s}")
    for k, label in [("completion_rate", "Completion %"),
                      ("mean_burden", "Mean Burden"),
                      ("mean_household_violation_months", "HH 침해 월수"),
                      ("household_violation_zero_rate", "HH 안전 셀러 %"),
                      ("mean_violation_amount_total", "[A-2] 누적 침해액"),
                      ("mean_violation_ratio_max", "[A-2] 단일월 최대비")]:
        b = base.get(k, float("nan"))
        c = cur.get(k, float("nan"))
        x = x1x2.get(k, float("nan"))
        print(f"  {label:28s} {b:10.3f} {c:10.3f} {x:10.3f}")
    print(f"\n=== 완료 ===")


def evaluate_full(model, seller_ids):
    env = RBFEnv(seller_ids=seller_ids, seed=SEED + 100,
                 eta=ETA, eta_proportional=True, r_clip_to_safe=False)
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
        "config": {"n_train": len(train_ids), "n_eval": len(eval_ids),
                   "timesteps": timesteps, "elapsed_min": round(elapsed/60, 2),
                   "eta": ETA, "x1_eta_proportional": True, "x2_r_clip_to_safe": False,
                   "reward_design": "X1_only"},
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
