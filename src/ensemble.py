"""
ensemble.py
-----------
Combines LSTM and XGBoost predictions via weighted averaging.
Also supports a simple meta-learner (Ridge regression) for learned blending.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ─────────────────────────────────────────────
# 1. Weighted Average Ensemble
# ─────────────────────────────────────────────

def weighted_ensemble(y_pred_lstm: np.ndarray,
                      y_pred_xgb:  np.ndarray,
                      lstm_weight: float = 0.5) -> np.ndarray:
    """
    Simple weighted average of two model predictions.

    Args:
        y_pred_lstm  : LSTM predictions (original price scale)
        y_pred_xgb   : XGBoost predictions (original price scale)
        lstm_weight  : Weight for LSTM (XGBoost weight = 1 - lstm_weight)

    Returns:
        Blended prediction array
    """
    xgb_weight = 1.0 - lstm_weight
    blended = lstm_weight * y_pred_lstm + xgb_weight * y_pred_xgb
    print(f"[ensemble] Weighted average  →  LSTM: {lstm_weight:.2f} | XGB: {xgb_weight:.2f}")
    return blended


# ─────────────────────────────────────────────
# 2. Find optimal weights via grid search on validation set
# ─────────────────────────────────────────────

def find_optimal_weights(y_val_true:  np.ndarray,
                         y_val_lstm:  np.ndarray,
                         y_val_xgb:   np.ndarray,
                         metric: str = "rmse") -> float:
    """
    Grid search over LSTM weight in [0, 1] (step 0.05).
    Returns the LSTM weight that minimises the chosen metric on validation set.

    metric: 'rmse' | 'mae'
    """
    best_weight = 0.5
    best_score  = float("inf")
    results = []

    for w in np.arange(0.0, 1.05, 0.05):
        blended = w * y_val_lstm + (1 - w) * y_val_xgb
        if metric == "rmse":
            score = np.sqrt(mean_squared_error(y_val_true, blended))
        else:
            score = mean_absolute_error(y_val_true, blended)
        results.append((round(w, 2), round(score, 6)))
        if score < best_score:
            best_score  = score
            best_weight = round(w, 2)

    print(f"[ensemble] Optimal LSTM weight: {best_weight}  |  "
          f"Best val {metric.upper()}: {best_score:.6f}")
    return best_weight


# ─────────────────────────────────────────────
# 3. Meta-learner (stacking) — optional advanced blend
# ─────────────────────────────────────────────

def train_meta_learner(y_val_true: np.ndarray,
                       y_val_lstm: np.ndarray,
                       y_val_xgb:  np.ndarray,
                       alpha: float = 1.0) -> Ridge:
    """
    Trains a Ridge regression meta-learner on validation predictions.
    Uses [lstm_pred, xgb_pred] as features to predict the true price.

    Alpha: L2 regularisation strength.
    """
    X_meta = np.column_stack([y_val_lstm, y_val_xgb])
    meta   = Ridge(alpha=alpha)
    meta.fit(X_meta, y_val_true)

    coef = meta.coef_
    print(f"[ensemble] Meta-learner coefficients → LSTM: {coef[0]:.4f} | XGB: {coef[1]:.4f}")
    return meta


def predict_meta(meta: Ridge,
                 y_pred_lstm: np.ndarray,
                 y_pred_xgb:  np.ndarray) -> np.ndarray:
    X_meta = np.column_stack([y_pred_lstm, y_pred_xgb])
    return meta.predict(X_meta)


# ─────────────────────────────────────────────
# 4. Evaluation
# ─────────────────────────────────────────────

def evaluate_ensemble(y_true:      np.ndarray,
                      y_pred_lstm: np.ndarray,
                      y_pred_xgb:  np.ndarray,
                      y_pred_ens:  np.ndarray,
                      save_dir:    str = "results") -> pd.DataFrame:
    """
    Computes metrics for all three models side-by-side and returns a DataFrame.
    Also saves a comparison plot.
    """
    rows = []
    for label, y_pred in [("LSTM", y_pred_lstm),
                           ("XGBoost", y_pred_xgb),
                           ("Ensemble", y_pred_ens)]:
        m = _compute_metrics(y_true, y_pred)
        m["Model"] = label
        rows.append(m)

    df_metrics = pd.DataFrame(rows).set_index("Model")
    print("\n── Ensemble Comparison ────────────────────────")
    print(df_metrics.to_string())
    print()

    # Save plot
    os.makedirs(os.path.join(save_dir, "plots"), exist_ok=True)
    plot_path = os.path.join(save_dir, "plots", "ensemble_comparison.png")
    _plot_comparison(y_true, y_pred_lstm, y_pred_xgb, y_pred_ens, save_path=plot_path)

    return df_metrics


# ─────────────────────────────────────────────
# 5. Plotting
# ─────────────────────────────────────────────

def _plot_comparison(y_true, y_lstm, y_xgb, y_ens, save_path: str) -> None:
    plt.figure(figsize=(16, 6))
    plt.plot(y_true, label="Actual",   color="black",      linewidth=2.0)
    plt.plot(y_lstm, label="LSTM",     color="steelblue",  linewidth=1.2, linestyle="--", alpha=0.8)
    plt.plot(y_xgb,  label="XGBoost", color="tomato",     linewidth=1.2, linestyle="--", alpha=0.8)
    plt.plot(y_ens,  label="Ensemble", color="mediumseagreen", linewidth=1.8, linestyle="-.")
    plt.title("Model Comparison — Actual vs Predictions")
    plt.xlabel("Time Step")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[ensemble] Comparison plot saved → {save_path}")


# ─────────────────────────────────────────────
# 6. Metrics
# ─────────────────────────────────────────────

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100

    actual_dir    = np.sign(np.diff(y_true))
    predicted_dir = np.sign(np.diff(y_pred))
    dir_acc = np.mean(actual_dir == predicted_dir) * 100

    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4),
            "MAPE": round(mape, 4), "Directional_Accuracy": round(dir_acc, 2)}


# ─────────────────────────────────────────────
# 7. Save / Load meta-learner
# ─────────────────────────────────────────────

def save_meta(meta: Ridge, path: str = "results/meta_learner.pkl") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(meta, path)
    print(f"[ensemble] Meta-learner saved → {path}")


def load_meta(path: str = "results/meta_learner.pkl") -> Ridge:
    meta = joblib.load(path)
    print(f"[ensemble] Meta-learner loaded ← {path}")
    return meta


# ─────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(7)
    N = 200
    y_true = np.cumsum(np.random.randn(N)) + 150
    y_lstm = y_true + np.random.randn(N) * 3
    y_xgb  = y_true + np.random.randn(N) * 4

    # Optimal weight
    best_w = find_optimal_weights(y_true[:100], y_lstm[:100], y_xgb[:100])

    # Blend
    y_ens = weighted_ensemble(y_lstm, y_xgb, lstm_weight=best_w)

    # Evaluate
    df_metrics = evaluate_ensemble(y_true, y_lstm, y_xgb, y_ens)
    print(df_metrics)
