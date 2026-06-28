"""
evaluate.py
-----------
Unified evaluation utilities: metrics, plots, and final report generation.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime


# ─────────────────────────────────────────────
# 1. Core metrics
# ─────────────────────────────────────────────

def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str = "") -> dict:
    """
    Returns a comprehensive dict of regression + trading metrics.
    """
    mae  = np.mean(np.abs(y_true - y_pred))
    mse  = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-9)

    actual_dir    = np.sign(np.diff(y_true))
    predicted_dir = np.sign(np.diff(y_pred))
    dir_acc = np.mean(actual_dir == predicted_dir) * 100

    return {
        "Model":                label,
        "MAE":                  round(mae,     4),
        "RMSE":                 round(rmse,    4),
        "MAPE (%)":             round(mape,    4),
        "R²":                   round(r2,      4),
        "Directional_Acc (%)":  round(dir_acc, 2),
    }


# ─────────────────────────────────────────────
# 2. Summary table
# ─────────────────────────────────────────────

def metrics_table(results: list[dict]) -> pd.DataFrame:
    """
    Takes a list of metric dicts (one per model) and returns a formatted DataFrame.

    Usage:
        table = metrics_table([
            compute_all_metrics(y_true, y_lstm, label="LSTM"),
            compute_all_metrics(y_true, y_xgb,  label="XGBoost"),
            compute_all_metrics(y_true, y_ens,  label="Ensemble"),
        ])
    """
    df = pd.DataFrame(results).set_index("Model")
    print("\n╔══════════════════════════════════════════════════╗")
    print("║           Final Model Comparison Table           ║")
    print("╠══════════════════════════════════════════════════╣")
    print(df.to_string())
    print("╚══════════════════════════════════════════════════╝\n")
    return df


# ─────────────────────────────────────────────
# 3. Dashboard plot
# ─────────────────────────────────────────────

def plot_dashboard(y_true:      np.ndarray,
                   y_pred_lstm: np.ndarray,
                   y_pred_xgb:  np.ndarray,
                   y_pred_ens:  np.ndarray,
                   ticker:      str = "STOCK",
                   save_dir:    str = "results/plots") -> None:
    """
    4-panel dashboard:
      [0] Actual vs all predictions
      [1] Residuals for each model
      [2] Error distribution (histogram)
      [3] Scatter: predicted vs actual
    """
    os.makedirs(save_dir, exist_ok=True)
    fig = plt.figure(figsize=(18, 12))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    colors = {"LSTM": "steelblue", "XGBoost": "tomato", "Ensemble": "mediumseagreen"}

    # ── Panel 0: Predictions vs Actual ──────────
    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(y_true,      color="black",                      linewidth=2.0, label="Actual")
    ax0.plot(y_pred_lstm, color=colors["LSTM"],    linestyle="--", linewidth=1.2, alpha=0.8, label="LSTM")
    ax0.plot(y_pred_xgb,  color=colors["XGBoost"], linestyle="--", linewidth=1.2, alpha=0.8, label="XGBoost")
    ax0.plot(y_pred_ens,  color=colors["Ensemble"],linestyle="-.", linewidth=1.8, label="Ensemble")
    ax0.set_title(f"{ticker} — Actual vs Predictions", fontsize=13)
    ax0.set_xlabel("Time Step"); ax0.set_ylabel("Price (USD)")
    ax0.legend(); ax0.grid(alpha=0.3)

    # ── Panel 1: Residuals ───────────────────────
    ax1 = fig.add_subplot(gs[1, 0])
    for label, y_pred, c in [("LSTM", y_pred_lstm, colors["LSTM"]),
                               ("XGBoost", y_pred_xgb, colors["XGBoost"]),
                               ("Ensemble", y_pred_ens, colors["Ensemble"])]:
        ax1.plot(y_true - y_pred, color=c, linewidth=0.9, alpha=0.75, label=label)
    ax1.axhline(0, color="black", linewidth=1.2, linestyle="--")
    ax1.set_title("Residuals (Actual − Predicted)", fontsize=12)
    ax1.set_xlabel("Time Step"); ax1.set_ylabel("Residual")
    ax1.legend(); ax1.grid(alpha=0.3)

    # ── Panel 2: Error Distribution ─────────────
    ax2 = fig.add_subplot(gs[1, 1])
    for label, y_pred, c in [("LSTM", y_pred_lstm, colors["LSTM"]),
                               ("XGBoost", y_pred_xgb, colors["XGBoost"]),
                               ("Ensemble", y_pred_ens, colors["Ensemble"])]:
        errors = y_true - y_pred
        ax2.hist(errors, bins=40, alpha=0.55, color=c, label=label, edgecolor="white")
    ax2.axvline(0, color="black", linewidth=1.5, linestyle="--")
    ax2.set_title("Error Distribution", fontsize=12)
    ax2.set_xlabel("Error"); ax2.set_ylabel("Frequency")
    ax2.legend(); ax2.grid(alpha=0.3)

    fig.suptitle(f"Stock Price Prediction Dashboard — {ticker}", fontsize=15, fontweight="bold")
    path = os.path.join(save_dir, f"{ticker}_dashboard.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[evaluate] Dashboard saved → {path}")


# ─────────────────────────────────────────────
# 4. Save metrics to CSV
# ─────────────────────────────────────────────

def save_metrics_csv(df_metrics: pd.DataFrame,
                     save_dir: str = "results",
                     filename: str = "metrics.csv") -> None:
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    df_metrics.to_csv(path)
    print(f"[evaluate] Metrics saved → {path}")


# ─────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(3)
    N = 150
    y_true = np.cumsum(np.random.randn(N)) + 200
    y_lstm = y_true + np.random.randn(N) * 3
    y_xgb  = y_true + np.random.randn(N) * 4
    y_ens  = 0.5 * y_lstm + 0.5 * y_xgb

    results = [
        compute_all_metrics(y_true, y_lstm, "LSTM"),
        compute_all_metrics(y_true, y_xgb,  "XGBoost"),
        compute_all_metrics(y_true, y_ens,  "Ensemble"),
    ]
    table = metrics_table(results)
    plot_dashboard(y_true, y_lstm, y_xgb, y_ens, ticker="TEST")
    save_metrics_csv(table)
