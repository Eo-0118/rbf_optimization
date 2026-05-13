"""Phase 2: LSTM 글로벌 시계열 예측 베이스라인

Prophet 베이스라인 한계:
- growth/stable 외 유형(decline/seasonal/volatile)에서 WAPE 95%+
- 단일 셀러 학습으로 노이즈에 취약
- 외생 변수 추가 시 오히려 악화 (과적합)

LSTM 글로벌 모델:
- 1,302 셀러 전체로 1개 모델 학습 (셀러별 학습 X)
- 외생 변수와 매출의 비선형 관계 학습 가능
- Seller type embedding으로 유형 정보 활용

설계:
  Input: 12개월 윈도우 (revenue, naver, promo) + type embedding
  Output: 다음 6개월 매출 예측
  분할: train 18 / val 3 / test 3 (Prophet과 동일하게 비교)
  학습: sliding window로 train 데이터 증강 (12→6 윈도우 여러 개)

산출:
  Data/lstm_baseline_results.csv
  Data/lstm_baseline_summary.json
  Data/lstm_baseline_diagnostics.png
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
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")

ROOT = Path("/Users/eoseungyun/Desktop/project/SW_Capstone/Project")
DATA = ROOT / "Data"

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

# === Config ===
SEED = 42
TRAIN_MONTHS = 18
VAL_MONTHS = 3
TEST_MONTHS = 3
CONTEXT_LEN = 9        # 입력 윈도우 (9개월) — TRAIN_MONTHS=18, PRED_LEN=6 이므로 ctx_start ∈ [0,3] → 4 windows/seller
PRED_LEN = 6           # 예측 윈도우 (val 3 + test 3)
HIDDEN_DIM = 64
NUM_LAYERS = 2
TYPE_EMB_DIM = 8
EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

torch.manual_seed(SEED)
np.random.seed(SEED)


# === Robust 지표 (Prophet과 동일) ===
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


# === Dataset ===
TYPE_TO_ID = {"stable": 0, "growth": 1, "seasonal": 2, "volatile": 3,
              "decline": 4, "other": 5}
N_TYPES = len(TYPE_TO_ID)


class SellerWindowDataset(Dataset):
    """Sliding window dataset.
    각 셀러에서 (CONTEXT_LEN → PRED_LEN) 윈도우를 추출.
    train 모드: 학습 가능한 모든 윈도우 사용 (data augmentation)
    eval 모드: 마지막 윈도우만 (실제 평가 시점)
    """
    def __init__(self, df: pd.DataFrame, mode: str = "train"):
        self.windows = []
        for sid, sdf in df.groupby("seller_id"):
            sdf = sdf.sort_values("month_idx").reset_index(drop=True)
            rev = sdf["monthly_revenue"].values.astype(np.float32)
            naver = sdf["naver_index"].values.astype(np.float32)
            promo = sdf["promo"].values.astype(np.float32)
            typ = TYPE_TO_ID.get(sdf["type"].iloc[0], 5)
            mu = sdf["mu"].iloc[0]  # 셀러 base scale (정규화용)

            if mode == "train":
                # train 영역(0~17)에서 가능한 윈도우들 (context_len → pred_len)
                # train만 보고 학습. context end <= 11 이어야 pred end <= 17
                max_ctx_end = TRAIN_MONTHS - PRED_LEN  # 12
                for ctx_end in range(CONTEXT_LEN, max_ctx_end + 1):
                    ctx_start = ctx_end - CONTEXT_LEN
                    pred_start = ctx_end
                    pred_end = pred_start + PRED_LEN
                    if pred_end > TRAIN_MONTHS:
                        break
                    self._add_window(sid, rev, naver, promo, typ, mu,
                                      ctx_start, ctx_end, pred_start, pred_end)
            else:
                # eval: 평가용 마지막 윈도우 (context: 6~17, pred: 18~23)
                ctx_start = TRAIN_MONTHS - CONTEXT_LEN  # 6
                ctx_end = TRAIN_MONTHS                   # 18
                pred_start = TRAIN_MONTHS                # 18
                pred_end = TRAIN_MONTHS + PRED_LEN       # 24
                self._add_window(sid, rev, naver, promo, typ, mu,
                                  ctx_start, ctx_end, pred_start, pred_end)

    def _add_window(self, sid, rev, naver, promo, typ, mu,
                     ctx_start, ctx_end, pred_start, pred_end):
        if ctx_end > len(rev) or pred_end > len(rev):
            return
        # 정규화: 각 셀러 mu 기준 (mu는 generation 시 base, 모든 셀러에 다름)
        scale = max(mu, 1.0)
        ctx_rev = rev[ctx_start:ctx_end] / scale
        ctx_naver = naver[ctx_start:ctx_end]
        ctx_promo = promo[ctx_start:ctx_end]
        # decoder 입력용 미래 외생 변수 (실세계에서도 알 수 있다고 가정)
        future_naver = naver[pred_start:pred_end]
        future_promo = promo[pred_start:pred_end]
        target = rev[pred_start:pred_end] / scale

        self.windows.append(dict(
            seller_id=sid,
            ctx=np.stack([ctx_rev, ctx_naver, ctx_promo], axis=-1),  # [12, 3]
            future_exog=np.stack([future_naver, future_promo], axis=-1),  # [6, 2]
            target=target,    # [6]
            typ=typ,
            scale=scale,
            mu=mu,
            actual_pred=rev[pred_start:pred_end],  # 원본 스케일 (평가용)
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


# === Model ===
class GlobalLSTM(nn.Module):
    """Encoder-decoder LSTM with type embedding + future exogenous.
    Encoder: past 12 months (rev, naver, promo) → hidden
    Decoder: hidden + future exog (naver, promo) → 6 month rev
    """
    def __init__(self):
        super().__init__()
        self.type_emb = nn.Embedding(N_TYPES, TYPE_EMB_DIM)
        self.encoder = nn.LSTM(input_size=3, hidden_size=HIDDEN_DIM,
                               num_layers=NUM_LAYERS, batch_first=True, dropout=0.1)
        # decoder input: 2 (future exog) + TYPE_EMB_DIM
        self.decoder = nn.LSTM(input_size=2 + TYPE_EMB_DIM, hidden_size=HIDDEN_DIM,
                               num_layers=NUM_LAYERS, batch_first=True, dropout=0.1)
        self.head = nn.Linear(HIDDEN_DIM, 1)

    def forward(self, ctx, future_exog, typ):
        # ctx: [B, 12, 3], future_exog: [B, 6, 2], typ: [B]
        _, (h, c) = self.encoder(ctx)  # [num_layers, B, hidden]

        emb = self.type_emb(typ)        # [B, emb_dim]
        emb_exp = emb.unsqueeze(1).expand(-1, PRED_LEN, -1)   # [B, 6, emb_dim]
        dec_in = torch.cat([future_exog, emb_exp], dim=-1)    # [B, 6, 2+emb_dim]

        dec_out, _ = self.decoder(dec_in, (h, c))             # [B, 6, hidden]
        pred = self.head(dec_out).squeeze(-1)                 # [B, 6]
        # ReLU로 음수 차단
        return torch.relu(pred)


# === Train ===
def train_model(train_loader, val_loader):
    model = GlobalLSTM().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.SmoothL1Loss()  # Huber loss — outlier에 robust

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_state = None

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

        # val
        model.eval()
        val_losses = []
        with torch.no_grad():
            for ctx, fexog, typ, target in val_loader:
                ctx, fexog, typ, target = ctx.to(DEVICE), fexog.to(DEVICE), typ.to(DEVICE), target.to(DEVICE)
                pred = model(ctx, fexog, typ)
                # val loss는 첫 3개월만 (val period). .contiguous()로 view 호환성 보장
                pred_v = pred[:, :VAL_MONTHS].contiguous()
                target_v = target[:, :VAL_MONTHS].contiguous()
                val_losses.append(criterion(pred_v, target_v).item())

        train_mean = float(np.mean(train_losses))
        val_mean = float(np.mean(val_losses))
        history["train_loss"].append(train_mean)
        history["val_loss"].append(val_mean)

        if val_mean < best_val:
            best_val = val_mean
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  epoch {epoch+1}/{EPOCHS}: train_loss={train_mean:.4f}, val_loss={val_mean:.4f}")

    model.load_state_dict(best_state)
    print(f"  Best val_loss: {best_val:.4f}")
    return model, history


def evaluate(model, dataset):
    """eval 데이터셋 전체에 대해 예측 + 지표 계산."""
    model.eval()
    rows = []
    detailed = []
    with torch.no_grad():
        for w in dataset.windows:
            ctx = torch.from_numpy(w["ctx"]).float().unsqueeze(0).to(DEVICE)
            fexog = torch.from_numpy(w["future_exog"]).float().unsqueeze(0).to(DEVICE)
            typ = torch.tensor([w["typ"]]).long().to(DEVICE)
            pred = model(ctx, fexog, typ).squeeze(0).cpu().numpy()  # [6]
            pred_rescaled = pred * w["scale"]

            actual = w["actual_pred"]
            val_actual = actual[:VAL_MONTHS]
            val_pred = pred_rescaled[:VAL_MONTHS]
            test_actual = actual[VAL_MONTHS:]
            test_pred = pred_rescaled[VAL_MONTHS:]

            typ_name = [k for k, v in TYPE_TO_ID.items() if v == w["typ"]][0]
            rows.append(dict(
                seller_id=w["seller_id"],
                type=typ_name,
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
    print(f"  유형: {df.groupby('seller_id')['type'].first().value_counts().to_dict()}")

    print("\n[2/5] Sliding window 데이터셋 생성")
    train_ds = SellerWindowDataset(df, mode="train")
    eval_ds = SellerWindowDataset(df, mode="eval")
    print(f"  train windows: {len(train_ds)}")
    print(f"  eval windows: {len(eval_ds)} (= 셀러 수)")

    # train의 일부를 val로 (학습 중 monitoring)
    n_train = len(train_ds)
    val_idx = np.random.RandomState(SEED).choice(n_train, size=n_train // 10, replace=False)
    val_mask = np.zeros(n_train, dtype=bool)
    val_mask[val_idx] = True
    train_subset = torch.utils.data.Subset(train_ds, np.where(~val_mask)[0])
    val_subset = torch.utils.data.Subset(train_ds, np.where(val_mask)[0])
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"\n[3/5] LSTM 학습 ({EPOCHS} epochs, batch={BATCH_SIZE}, hidden={HIDDEN_DIM})")
    model, history = train_model(train_loader, val_loader)

    print("\n[4/5] 평가 (test set 6개월 예측)")
    res_df, detailed = evaluate(model, eval_ds)
    res_df.to_csv(DATA / "lstm_baseline_results.csv", index=False)
    print(f"  결과: {len(res_df)} sellers")

    # === 요약 ===
    summary = {
        "config": {
            "context_len": CONTEXT_LEN, "pred_len": PRED_LEN,
            "hidden": HIDDEN_DIM, "layers": NUM_LAYERS,
            "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LEARNING_RATE,
            "type_emb_dim": TYPE_EMB_DIM,
            "exog_vars": ["naver_index", "promo"],
            "loss": "SmoothL1Loss (Huber)",
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

    (DATA / "lstm_baseline_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"\n=== LSTM Baseline 결과 ===")
    print(f"  학습 데이터: {n_train - len(val_idx)} windows")
    print(f"  평가: {len(res_df)} sellers")
    print(f"\n  [test set 평균]")
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

    # === [5/5] 시각화 ===
    color_map = {"stable": "steelblue", "growth": "mediumseagreen",
                 "volatile": "crimson", "seasonal": "darkorange",
                 "decline": "gray", "other": "lightgray"}
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # (1) 학습 곡선
    ax = axes[0, 0]
    ax.plot(history["train_loss"], label="train")
    ax.plot(history["val_loss"], label="val")
    ax.set_xlabel("epoch"); ax.set_ylabel("Huber loss")
    ax.set_title("학습 곡선")
    ax.legend(); ax.grid(alpha=0.3)

    # (2) 지표 분포
    ax = axes[0, 1]
    bins = np.linspace(0, 200, 41)
    ax.hist(res_df["mape_test"].clip(upper=200), bins=bins, alpha=0.5, label="MAPE", color="crimson")
    ax.hist(res_df["smape_test"].clip(upper=200), bins=bins, alpha=0.5, label="SMAPE", color="steelblue")
    ax.hist(res_df["wape_test"].clip(upper=200), bins=bins, alpha=0.5, label="WAPE", color="mediumseagreen")
    ax.axvline(20, color="black", linestyle="--", label="목표 20%")
    ax.set_xlabel("error %"); ax.set_ylabel("셀러 수")
    ax.set_title("LSTM 지표 분포 (test)")
    ax.legend(); ax.grid(alpha=0.3)

    # (3) 유형별 WAPE
    ax = axes[0, 2]
    types_ordered = sorted(res_df["type"].unique())
    data_lst = [res_df[res_df["type"] == t]["wape_test"].clip(upper=200).values for t in types_ordered]
    bp = ax.boxplot(data_lst, labels=types_ordered, patch_artist=True)
    for patch, t in zip(bp["boxes"], types_ordered):
        patch.set_facecolor(color_map.get(t, "gray"))
        patch.set_alpha(0.7)
    ax.axhline(20, color="red", linestyle="--", alpha=0.5, label="목표 20%")
    ax.set_ylabel("WAPE (test) %")
    ax.set_title("유형별 WAPE 분포")
    ax.tick_params(axis="x", rotation=15)
    ax.legend(); ax.grid(alpha=0.3)

    # (4-6) 샘플 예측 시계열 3개
    for i, d in enumerate(detailed[:3]):
        ax = axes[1, i]
        ctx_idx = np.arange(CONTEXT_LEN)
        pred_idx = np.arange(CONTEXT_LEN, CONTEXT_LEN + PRED_LEN)
        ax.plot(ctx_idx, d["ctx_actual"], "o-", color="steelblue", label="과거 (실제)")
        ax.plot(pred_idx, d["actual_pred"], "o-", color="darkorange", label="미래 (실제)")
        ax.plot(pred_idx, d["pred"], "x--", color="crimson", alpha=0.8, label="예측")
        ax.axvline(CONTEXT_LEN - 0.5, color="gray", linestyle=":", alpha=0.5)
        # val/test 경계
        ax.axvline(CONTEXT_LEN + VAL_MONTHS - 0.5, color="gray", linestyle=":", alpha=0.3)
        ax.set_title(f"{d['seller_id'][:24]} [{d['type']}]")
        ax.set_xlabel("월 (윈도우 기준)")
        ax.set_ylabel("매출 (만원)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle(f"LSTM Baseline (n={summary['overall']['n_total']}, "
                 f"WAPE 평균 {summary['overall']['wape_test_mean']:.1f}%)",
                 fontsize=13, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(DATA / "lstm_baseline_diagnostics.png", dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\n[save] lstm_baseline_diagnostics.png")
    print("\n=== 완료 ===")


if __name__ == "__main__":
    main()
