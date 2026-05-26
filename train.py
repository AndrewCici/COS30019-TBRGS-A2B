import os
import sys
import math
import copy
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Resolve paths relative to this script regardless of cwd
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(SCRIPT_DIR, "data", "scats_sample.csv")
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
PLOTS_DIR  = os.path.join(SCRIPT_DIR, "plots")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)

# Allow importing sibling modules when run from a different cwd
sys.path.insert(0, SCRIPT_DIR)

from data_processing import TrafficDataProcessor
from models import TrafficLSTM, TrafficGRU, TrafficTransformer

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[train] Using device: {DEVICE}")


# ---------------------------------------------------------------------------
# Training loop with early stopping
# ---------------------------------------------------------------------------
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    patience: int = 10,
    lr: float = 1e-3,
) -> dict:
    """
    Train *model* with MSELoss + Adam, using early stopping on val loss.

    Parameters
    ----------
    model        : nn.Module   — one of TrafficLSTM, TrafficGRU, TrafficTransformer
    train_loader : DataLoader
    val_loader   : DataLoader
    epochs       : int         — maximum training epochs
    patience     : int         — early-stopping patience (epochs without improvement)
    lr           : float       — Adam learning rate

    Returns
    -------
    history : dict with keys 'train_loss', 'val_loss', 'best_epoch'
    """
    model = model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = math.inf
    best_weights  = copy.deepcopy(model.state_dict())
    no_improve    = 0
    history       = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        # ---- Training phase ------------------------------------------------
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()
            preds = model(X_batch)
            loss  = criterion(preds, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * X_batch.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        # ---- Validation phase ----------------------------------------------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                preds    = model(X_batch)
                val_loss += criterion(preds, y_batch).item() * X_batch.size(0)
        val_loss /= len(val_loader.dataset)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(
            f"  Epoch [{epoch:03d}/{epochs}]  "
            f"Train MSE: {train_loss:.6f}  Val MSE: {val_loss:.6f}"
        )

        # ---- Early stopping ------------------------------------------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights  = copy.deepcopy(model.state_dict())
            no_improve    = 0
            history["best_epoch"] = epoch
        else:
            no_improve += 1
            if no_improve >= patience:
                print(
                    f"  Early stopping triggered at epoch {epoch} "
                    f"(best epoch: {history['best_epoch']}, "
                    f"best val MSE: {best_val_loss:.6f})"
                )
                break

    # Restore best weights before returning
    model.load_state_dict(best_weights)
    return history


# ---------------------------------------------------------------------------
# Evaluation — RMSE and MAE on a DataLoader split
# ---------------------------------------------------------------------------
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    scaler,
) -> dict:
    """
    Compute RMSE and MAE in the *original* (inverse-scaled) unit.

    Parameters
    ----------
    model  : trained nn.Module
    loader : DataLoader (test split)
    scaler : fitted MinMaxScaler used in TrafficDataProcessor

    Returns
    -------
    dict with 'rmse' and 'mae'
    """
    model.eval()
    all_preds, all_true = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            preds = model(X_batch).cpu().numpy()
            all_preds.append(preds)
            all_true.append(y_batch.numpy())

    all_preds = np.concatenate(all_preds, axis=0)   # (N, 1)
    all_true  = np.concatenate(all_true,  axis=0)   # (N, 1)

    # Inverse-transform back to vehicle counts
    all_preds_inv = scaler.inverse_transform(all_preds)
    all_true_inv  = scaler.inverse_transform(all_true)

    rmse = float(np.sqrt(np.mean((all_preds_inv - all_true_inv) ** 2)))
    mae  = float(np.mean(np.abs(all_preds_inv - all_true_inv)))

    return {"rmse": rmse, "mae": mae}


# ---------------------------------------------------------------------------
# Loss curve plotting
# ---------------------------------------------------------------------------
def plot_loss_curves(history: dict, model_name: str) -> None:
    """Save a train-vs-val loss curve PNG to PLOTS_DIR."""
    fig, ax = plt.subplots(figsize=(9, 5))
    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], label="Train MSE", linewidth=2)
    ax.plot(epochs, history["val_loss"],   label="Val MSE",   linewidth=2, linestyle="--")
    best_ep = history.get("best_epoch")
    if best_ep:
        ax.axvline(x=best_ep, color="red", linestyle=":", linewidth=1.5,
                   label=f"Best epoch ({best_ep})")
    ax.set_title(f"{model_name} — Train vs Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, f"{model_name.lower()}_loss.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Loss curve saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    # ---- 1. Data ----------------------------------------------------------
    print("\n[1/4] Loading and preprocessing data …")
    processor = TrafficDataProcessor(
        filepath=DATA_PATH,
        time_step=12,
        batch_size=32,
    )
    train_loader, val_loader, test_loader = processor.build_loaders()
    scaler = processor.scaler

    # ---- 2. Model registry ------------------------------------------------
    model_configs = [
        (
            "LSTM",
            TrafficLSTM(input_size=1, hidden_size=64, num_layers=2, dropout=0.2),
            os.path.join(MODELS_DIR, "lstm_best.pth"),
        ),
        (
            "GRU",
            TrafficGRU(input_size=1, hidden_size=64, num_layers=2, dropout=0.2),
            os.path.join(MODELS_DIR, "gru_best.pth"),
        ),
        (
            "Transformer",
            TrafficTransformer(
                input_size=1, d_model=64, nhead=4,
                num_encoder_layers=2, dim_feedforward=128, dropout=0.1,
            ),
            os.path.join(MODELS_DIR, "transformer_best.pth"),
        ),
    ]

    results = {}

    for model_name, model, save_path in model_configs:
        print(f"\n{'='*60}")
        print(f"[2/4] Training {model_name} …")
        print(f"{'='*60}")

        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=50,
            patience=10,
            lr=1e-3,
        )

        # ---- 3. Save best weights -----------------------------------------
        torch.save(model.state_dict(), save_path)
        print(f"  Best weights saved → {save_path}")

        # ---- 4. Evaluate on test set ---------------------------------------
        metrics = evaluate_model(model, test_loader, scaler)
        results[model_name] = metrics
        print(
            f"  Test  RMSE: {metrics['rmse']:.4f} vehicles  |  "
            f"MAE: {metrics['mae']:.4f} vehicles"
        )

        # ---- 5. Plot loss curves -------------------------------------------
        plot_loss_curves(history, model_name)

    # ---- Summary -----------------------------------------------------------
    print(f"\n{'='*60}")
    print("Final Test Metrics Summary")
    print(f"{'='*60}")
    print(f"  {'Model':<15} {'RMSE':>10} {'MAE':>10}")
    print(f"  {'-'*37}")
    for name, m in results.items():
        print(f"  {name:<15} {m['rmse']:>10.4f} {m['mae']:>10.4f}")
    print(f"\nAll model weights saved in : {MODELS_DIR}")
    print(f"All loss plots saved in    : {PLOTS_DIR}")


if __name__ == "__main__":
    main()
