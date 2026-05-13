"""Phase 2: LSTM 글로벌 시계열 예측 v2 — 튜닝

v1 결과 진단:
- WAPE mean 75% / median 51% (Prophet v2 mean 104% 대비 개선)
- 그러나 < 20% 셀러 비율 2.2%만 (Prophet v2 17.8% 대비 악화)
- 원인: stable 78% 데이터 편향 → 학습이 stable 평균에 수렴
       overfitting signal (train 0.12 vs val 0.16)
       모든 셀러에서 보수적 평균값 예측

v2 변경:
1. WeightedRandomSampler: minority type oversample (각 type 균등 가중)
2. HIDDEN_DIM 64 → 128 (모델 capacity ↑)
3. Dropout 0.1 → 0.25 (overfitting 완화)
4. Cosine LR schedule (후반 미세조정)
5. EPOCHS 50 → 80 (best val 추적은 동일)

산출:
  Data/lstm_baseline_v2_results.csv
  Data/lstm_baseline_v2_summary.json
  Data/lstm_baseline_v2_diagnostics.png
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

warnings.filterwarnings("ignore")

ROOT = Path("/Users/eoseungyun/Desktop/project/SW_Capstone/Project")
DATA = ROOT / "Data"

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

# === Config (v2 튜닝값) ===
SEED = 42
TRAIN_MONTHS = 18
VAL_MONTHS = 3
TEST_MONTHS = 3
CONTEXT_LEN = 9
PRED_LEN = 6
HIDDEN_DIM = 128            # ↑ 64→128
NUM_LAYERS = 2
TYPE_EMB_DIM = 8
DROPOUT = 0.25              # ↑ 0.1→0.25
EPOCHS = 80                 # ↑ 50→80
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

torch.manual_seed(SEED)
np.random.seed(SEED)

TYPE_TO_ID = {"stable": 0, "growth": 1, "seasonal": 2, "volatile": 3,
              "decline": 4, "other": 5}
N_TYPES = len(TYPE_TO_ID)


# === Robust 지표 ===
def mape(actual, pred):
    a, p = np.asarray(actual, dtype=float), np.asarray(pred, dtype=float)
    mask = a > 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((a[mask] - p[mask]) / a[mask])) * 100)


def smape(actual, pred):
    a, p = np.asarray(actual, dtype=float), np.asarray(pred, dtype=float)
    denom = (np.abs(a) + np.abs(p)) / 2.0
    safe = denom > 0
    if safe.sum() == 0:
        return float("nan")
    err = np.zeros_like(a)
    err[safe] = np.abs(a[safe] - p[safe]) / denom[safe]
    return float(np.mean(err) * 100)


def wape(actual, pred):
    a, p = np.asarray(actual, dtype=float), np.asarray(pred, dtype=float)
    s = np.abs(a).sum()
    if s == 0:
        return float("nan")
    return float(np.abs(a - p).sum() / s * 100)


# === Dataset (v1과 동일) ===
class SellerWindowDataset(Dataset):
    def __init__(self, df: pd.DataFrame, mode: str = "train"):
        self.windows = []
        for sid, sdf in df.groupby("seller_id"):
            sdf = sdf.sort_values("month_idx").reset_index(drop=True)
            rev = sdf["monthly_revenue"].values.astype(np.float32)
            naver = sdf["naver_index"].values.astype(np.float32)
            promo = sdf["promo"].values.astype(np.float32)
            typ = TYPE_TO_ID.get(sdf["type"].iloc[0], 5)
            mu = sdf["mu"].iloc[0]

            if mode == "train":
                max_ctx_end = TRAIN_MONTHS - PRED_LEN
                for ctx_end in range(CONTEXT_LEN, max_ctx_end + 1):
                    ctx_start = ctx_end - CONTEXT_LEN
                    pred_start = ctx_end
                    pred_end = pred_start + PRED_LEN
                    if pred_end > TRAIN_MONTHS:
                        break
                    self._add_window(sid, rev, naver, promo, typ, mu,
                                      ctx_start, ctx_end, pred_start, pred_end)
            else:
                ctx_start = TRAIN_MONTHS - CONTEXT_LEN
                ctx_end = TRAIN_MONTHS
                pred_start = TRAIN_MONTHS
                pred_end = TRAIN_MONTHS + PRED_LEN
                self._add_window(sid, rev, naver, promo, typ, mu,
                                  ctx_start, ctx_end, pred_start, pred_end)

    def _add_window(self, sid, rev, naver, promo, typ, mu,
                     ctx_start, ctx_end, pred_start, pred_end):
        if ctx_end > len(rev) or pred_end > len(rev):
            return
        scale = max(mu, 1.0)
        ctx_rev = rev[ctx_start:ctx_end] / scale
        ctx_naver = naver[ctx_start:ctx_end]
        ctx_promo = promo[ctx_start:ctx_end]
        future_naver = naver[pred_start:pred_end]
        future_promo = promo[pred_start:pred_end]
        target = rev[pred_start:pred_end] / scale

        self.windows.append(dict(
            seller_id=sid,
            ctx=np.stack([ctx_rev, ctx_naver, ctx_promo], axis=-1),
            future_exog=np.stack([future_naver, future_promo], axis=-1),
            target=target,
            typ=typ,
            scale=scale,
            mu=mu,
            actual_pred=rev[pred_start:pred_end],
        ))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        w = self.windows[i]
        return (
            torch.from_numpy(w["ctx"]).float(),
            torch.from_numpy(w["future_exog"]).float(),
            torch.tensor(w["typ"]).long(),
            torch.from_numpy(w["target"]).float(),
        )


# === Model (HIDDEN, DROPOUT 변경) ===
class GlobalLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.type_emb = nn.Embedding(N_TYPES, TYPE_EMB_DIM)
        self.encoder = nn.LSTM(input_size=3, hidden_size=HIDDEN_DIM,
                               num_layers=NUM_LAYERS, batch_first=True, dropout=DROPOUT)
        self.decoder = nn.LSTM(input_size=2 + TYPE_EMB_DIM, hidden_size=HIDDEN_DIM,
                               num_layers=NUM_LAYERS, batch_first=True, dropout=DROPOUT)
        self.head = nn.Linear(HIDDEN_DIM, 1)

    def forward(self, ctx, future_exog, typ):
        _, (h, c) = self.encoder(ctx)
        emb = self.type_emb(typ)
        emb_exp = emb.unsqueeze(1).expand(-1, PRED_LEN, -1)
        dec_in = torch.cat([future_exog, emb_exp], dim=-1)
        dec_out, _ = self.decoder(dec_in, (h, c))
        pred = self.head(dec_out).squeeze(-1)
        return torch.relu(pred)


# === Train (Weighted sampler + Cosine LR 추가) ===
def make_weighted_sampler(dataset_subset, parent_dataset):
    """Type 균등 가중 샘플러: 각 type sampling 확률 = 1/n_type."""
    indices = dataset_subset.indices if hasattr(dataset_subset, "indices") else range(len(dataset_subset))
    types = [parent_dataset.windows[i]["typ"] for i in indices]
    type_counts = np.bincount(types, minlength=N_TYPES)
    # 가중치: type별 1/count (drawn 빈도 균등화)
    weights = np.array([1.0 / type_counts[t] if type_counts[t] > 0 else 0 for t in types],
                       dtype=np.float64)
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True), type_counts


def train_model(train_loader, val_loader):
    model = GlobalLSTM().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.SmoothL1Loss()

    history = {"train_loss": [], "val_loss": [], "lr": []}
    best_val = float("inf")
    best_state = None
    best_epoch = 0

    for epoch in range(EPOCHS):
        model.train()
        train_losses = []
        for ctx, fexog, typ, target in train_loader:
            ctx, fexog, typ, target = ctx.to(DEVICE), fexog.to(DEVICE), typ.to(DEVICE), target.to(DEVICE)
            pred = model(ctx, fexog, typ)
            loss = criterion(pred, target)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for ctx, fexog, typ, target in val_loader:
                ctx, fexog, typ, target = ctx.to(DEVICE), fexog.to(DEVICE), typ.to(DEVICE), target.to(DEVICE)
                pred = model(ctx, fexog, typ)
                pred_v = pred[:, :VAL_MONTHS].contiguous()
                target_v = target[:, :VAL_MONTHS].contiguous()
                val_losses.append(criterion(pred_v, target_v).item())

        train_mean = float(np.mean(train_losses))
        val_mean = float(np.mean(val_losses))
        history["train_loss"].append(train_mean)
        history["val_loss"].append(val_mean)
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))

        if val_mean < best_val:
            best_val = val_mean
            best_epoch = epoch + 1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  epoch {epoch+1}/{EPOCHS}: train={train_mean:.4f}  val={val_mean:.4f}  lr={history['lr'][-1]:.5f}")

    model.load_state_dict(best_state)
    print(f"  Best val_loss: {best_val:.4f} (epoch {best_epoch})")
    return model, history


def evaluate(model, dataset):
    model.eval()
    rows = []
    detailed = []
    with torch.no_grad():
        for w in dataset.windows:
            ctx = torch.from_numpy(w["ctx"]).float().unsqueeze(0).to(DEVICE)
            fexog = torch.from_numpy(w["future_exog"]).float().unsqueeze(0).to(DEVICE)
            typ = torch.tensor([w["typ"]]).long().to(DEVICE)
            pred = model(ctx, fexog, typ).squeeze(0).cpu().numpy()
            pred_rescaled = pred * w["scale"]

            actual = w["actual_pred"]
            val_actual = actual[:VAL_MONTHS]
            val_pred = pred_rescaled[:VAL_MONTHS]
            test_actual = actual[VAL_MONTHS:]
            test_pred = pred_rescaled[VAL_MONTHS:]

            typ_name = [k for k, v in TYPE_TO_ID.items() if v == w["typ"]][0]
            rows.append(dict(
                seller_id=w["seller_id"], type=typ_name,
                mape_val=mape(val_actual, val_pred),
                mape_test=mape(test_actual, test_pred),
                smape_val=smape(val_actual, val_pred),
                smape_test=smape(test_actual, test_pred),
                wape_val=wape(val_actual, val_pred),
                wape_test=wape(test_actual, test_pred),
            ))
            if len(detailed) < 6:
                detailed.append(dict(
                    seller_id=w["seller_id"], type=typ_name,
                    actual_pred=actual, pred=pred_rescaled,
                    ctx_actual=w["ctx"][:, 0] * w["scale"],
                ))
    return pd.DataFrame(rows), detailed


def main():
    print(f"[Device] {DEVICE}")
    print("\n[1/5] Cohort v2 로드")
    df = pd.read_parquet(DATA / "cohort_kr_v2.parquet")
    df["date"] = pd.to_datetime(df["date"])
    print(f"  {df['seller_id'].nunique()} sellers, {len(df)} rows")

    print("\n[2/5] Sliding window 데이터셋 생성")
    train_ds = SellerWindowDataset(df, mode="train")
    eval_ds = SellerWindowDataset(df, mode="eval")
    print(f"  train windows: {len(train_ds)}")
    print(f"  eval windows: {len(eval_ds)}")

    n_train = len(train_ds)
    val_idx = np.random.RandomState(SEED).choice(n_train, size=n_train // 10, replace=False)
    val_mask = np.zeros(n_train, dtype=bool)
    val_mask[val_idx] = True
    train_subset = torch.utils.data.Subset(train_ds, np.where(~val_mask)[0])
    val_subset = torch.utils.data.Subset(train_ds, np.where(val_mask)[0])

    # Weighted sampler (train만)
    sampler, type_counts = make_weighted_sampler(train_subset, train_ds)
    print(f"  train subset type counts: {dict(zip(TYPE_TO_ID.keys(), type_counts.tolist()))}")
    print(f"  → WeightedRandomSampler 활성: 각 type 균등 sampling")

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"\n[3/5] LSTM v2 학습 ({EPOCHS} epochs, hidden={HIDDEN_DIM}, dropout={DROPOUT}, cosine LR)")
    model, history = train_model(train_loader, val_loader)

    print("\n[4/5] 평가 (test set 6개월 예측)")
    res_df, detailed = evaluate(model, eval_ds)
    res_df.to_csv(DATA / "lstm_baseline_v2_results.csv", index=False)
    print(f"  결과: {len(res_df)} sellers")

    # === Summary ===
    summary = {
        "config": {
            "context_len": CONTEXT_LEN, "pred_len": PRED_LEN,
            "hidden": HIDDEN_DIM, "layers": NUM_LAYERS, "dropout": DROPOUT,
            "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LEARNING_RATE,
            "type_emb_dim": TYPE_EMB_DIM,
            "exog_vars": ["naver_index", "promo"],
            "loss": "SmoothL1Loss (Huber)",
            "weighted_sampler": True,
            "lr_schedule": "cosine",
        },
        "training": {
            "final_train_loss": float(history["train_loss"][-1]),
            "final_val_loss": float(history["val_loss"][-1]),
            "best_val_loss": float(min(history["val_loss"])),
        },
        "overall": {
            "n_total": int(len(res_df)),
            "mape_test_mean": float(res_df["mape_test"].mean()),
            "mape_test_median": float(res_df["mape_test"].median()),
            "smape_test_mean": float(res_df["smape_test"].mean()),
            "smape_test_median": float(res_df["smape_test"].median()),
            "wape_test_mean": float(res_df["wape_test"].mean()),
            "wape_test_median": float(res_df["wape_test"].median()),
            "mape_test_pct_under_20": float((res_df["mape_test"] < 20).mean() * 100),
            "smape_test_pct_under_20": float((res_df["smape_test"] < 20).mean() * 100),
            "wape_test_pct_under_20": float((res_df["wape_test"] < 20).mean() * 100),
        },
        "by_type": {},
    }
    for typ, g in res_df.groupby("type"):
        summary["by_type"][typ] = dict(
            n=int(len(g)),
            mape_test_median=float(g["mape_test"].median()),
            smape_test_median=float(g["smape_test"].median()),
            wape_test_median=float(g["wape_test"].median()),
            wape_test_pct_under_20=float((g["wape_test"] < 20).mean() * 100),
        )

    (DATA / "lstm_baseline_v2_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"\n=== LSTM v2 결과 ===")
    print(f"  [test set 평균]")
    print(f"   MAPE:  mean={summary['overall']['mape_test_mean']:.1f}%  median={summary['overall']['mape_test_median']:.1f}%")
    print(f"   SMAPE: mean={summary['overall']['smape_test_mean']:.1f}%  median={summary['overall']['smape_test_median']:.1f}%")
    print(f"   WAPE:  mean={summary['overall']['wape_test_mean']:.1f}%  median={summary['overall']['wape_test_median']:.1f}%")
    print(f"\n  [< 20% 셀러 비율]")
    print(f"   MAPE:  {summary['overall']['mape_test_pct_under_20']:.1f}%")
    print(f"   SMAPE: {summary['overall']['smape_test_pct_under_20']:.1f}%")
    print(f"   WAPE:  {summary['overall']['wape_test_pct_under_20']:.1f}%")
    print(f"\n  [유형별 WAPE test median]")
    for typ, s in summary["by_type"].items():
        print(f"    {typ:10s}: WAPE={s['wape_test_median']:6.1f}%  SMAPE={s['smape_test_median']:5.1f}%  MAPE={s['mape_test_median']:5.1f}%  (n={s['n']})")

    # === [5/5] 시각화 (v1 비교 포함) ===
    color_map = {"stable": "steelblue", "growth": "mediumseagreen",
                 "volatile": "crimson", "seasonal": "darkorange",
                 "decline": "gray", "other": "lightgray"}
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # (1) 학습 곡선 + LR
    ax = axes[0, 0]
    ax2 = ax.twinx()
    ax.plot(history["train_loss"], label="train", color="steelblue")
    ax.plot(history["val_loss"], label="val", color="darkorange")
    ax2.plot(history["lr"], label="lr", color="green", alpha=0.4, linestyle="--")
    ax.set_xlabel("epoch"); ax.set_ylabel("Huber loss")
    ax2.set_ylabel("learning rate", color="green")
    ax.set_title("학습 곡선 (v2: cosine LR)")
    ax.legend(loc="upper left"); ax.grid(alpha=0.3)

    # (2) 지표 분포
    ax = axes[0, 1]
    bins = np.linspace(0, 200, 41)
    ax.hist(res_df["mape_test"].clip(upper=200), bins=bins, alpha=0.5, label="MAPE", color="crimson")
    ax.hist(res_df["smape_test"].clip(upper=200), bins=bins, alpha=0.5, label="SMAPE", color="steelblue")
    ax.hist(res_df["wape_test"].clip(upper=200), bins=bins, alpha=0.5, label="WAPE", color="mediumseagreen")
    ax.axvline(20, color="black", linestyle="--", label="목표 20%")
    ax.set_xlabel("error %"); ax.set_ylabel("셀러 수")
    ax.set_title("LSTM v2 지표 분포 (test)")
    ax.legend(); ax.grid(alpha=0.3)

    # (3) v1 vs v2 비교 (가능하면)
    ax = axes[0, 2]
    v1_path = DATA / "lstm_baseline_results.csv"
    if v1_path.exists():
        v1 = pd.read_csv(v1_path)
        merged = v1.merge(res_df, on="seller_id", suffixes=("_v1", "_v2"))
        ax.scatter(merged["wape_test_v1"].clip(upper=200),
                   merged["wape_test_v2"].clip(upper=200),
                   c=[color_map.get(t, "gray") for t in merged["type_v2"]],
                   alpha=0.6, s=25)
        lim = 200
        ax.plot([0, lim], [0, lim], "k--", alpha=0.5, label="동일")
        improved_pct = (merged["wape_test_v2"] < merged["wape_test_v1"]).mean() * 100
        ax.set_xlabel("LSTM v1 WAPE")
        ax.set_ylabel("LSTM v2 WAPE")
        ax.set_title(f"v1 vs v2 ({improved_pct:.0f}% 개선)")
        ax.legend(); ax.grid(alpha=0.3)
    else:
        # fallback: 유형별 WAPE box
        types_ordered = sorted(res_df["type"].unique())
        data_lst = [res_df[res_df["type"] == t]["wape_test"].clip(upper=200).values for t in types_ordered]
        bp = ax.boxplot(data_lst, labels=types_ordered, patch_artist=True)
        for patch, t in zip(bp["boxes"], types_ordered):
            patch.set_facecolor(color_map.get(t, "gray"))
            patch.set_alpha(0.7)
        ax.axhline(20, color="red", linestyle="--", alpha=0.5)
        ax.set_title("유형별 WAPE 분포")
        ax.tick_params(axis="x", rotation=15)
        ax.grid(alpha=0.3)

    # (4-6) 샘플 시계열
    for i, d in enumerate(detailed[:3]):
        ax = axes[1, i]
        ctx_idx = np.arange(CONTEXT_LEN)
        pred_idx = np.arange(CONTEXT_LEN, CONTEXT_LEN + PRED_LEN)
        ax.plot(ctx_idx, d["ctx_actual"], "o-", color="steelblue", label="과거 (실제)")
        ax.plot(pred_idx, d["actual_pred"], "o-", color="darkorange", label="미래 (실제)")
        ax.plot(pred_idx, d["pred"], "x--", color="crimson", alpha=0.8, label="예측")
        ax.axvline(CONTEXT_LEN - 0.5, color="gray", linestyle=":", alpha=0.5)
        ax.axvline(CONTEXT_LEN + VAL_MONTHS - 0.5, color="gray", linestyle=":", alpha=0.3)
        ax.set_title(f"{d['seller_id'][:24]} [{d['type']}]")
        ax.set_xlabel("월 (윈도우 기준)")
        ax.set_ylabel("매출 (만원)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle(f"LSTM v2 (n={summary['overall']['n_total']}, "
                 f"WAPE 평균 {summary['overall']['wape_test_mean']:.1f}%, "
                 f"weighted+hidden128+dropout25+cosineLR)",
                 fontsize=12, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(DATA / "lstm_baseline_v2_diagnostics.png", dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\n[save] lstm_baseline_v2_diagnostics.png")
    print("\n=== 완료 ===")


if __name__ == "__main__":
    main()
