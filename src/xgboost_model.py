"""
xgboost_model.py
----------------
Trains, evaluates, and saves an XGBoost model for stock price prediction.
Works on tabular features (technical indicators + price features).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ─────────────────────────────────────────────
# 1. Build model
# ─────────────────────────────────────────────

def build_xgboost(n_estimators: int = 500,
                  max_depth: int = 6,
                  learning_rate: float = 0.05,
                  subsample: float = 0.8,
                  colsample_bytree: float = 0.8,
                  min_child_weight: int = 3,
                  reg_alpha: float = 0.1,
                  reg_lambda: float = 1.0,
                  early_stopping_rounds: int = 30,
                  random_state: int = 42) -> XGBRegressor:
    """
    Returns a configured XGBRegressor.

    Key hyperparameters:
      n_estimators     : number of boosting rounds
      max_depth        : tree depth — controls overfitting
      learning_rate    : shrinkage per step
      subsample        : row sampling per tree
      colsample_bytree : feature sampling per tree
      reg_alpha/lambda : L1/L2 regularisation
    """
    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=min_child_weight,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        random_state=random_state,
        tree_method="hist",         # fast histogram method
        objective="reg:squarederror",
        eval_metric="rmse",
        verbosity=0,
    )
    return model


# ─────────────────────────────────────────────
# 2. Prepare flat (tabular) arrays from DataFrame
# ─────────────────────────────────────────────

def prepare_tabular(df: pd.DataFrame,
                    feature_cols: list[str],
                    target_col: str = "Target") -> tuple[np.ndarray, np.ndarray]:
    """
    Extracts X (2-D) and y (1-D) arrays from a DataFrame.
    Used for XGBoost which does NOT need 3-D sequences.
    """
    X = df[feature_cols].values
    y = df[target_col].values
    return X, y


# ─────────────────────────────────────────────
# 3. Training with early stopping
# ─────────────────────────────────────────────

# def train_xgboost(model: XGBRegressor,
#                   X_train: np.ndarray,
#                   y_train: np.ndarray,
#                   X_val:   np.ndarray,
#                   y_val:   np.ndarray,
#                   early_stopping_rounds: int = 30) -> XGBRegressor:

def train_xgboost(model, X_train, y_train, X_val, y_val):

    """
    Fits the XGBoost model with early stopping on validation RMSE.

    Returns:
        Fitted model
    """
    print(f"[xgboost] Training on {X_train.shape[0]} samples | "
          f"Validating on {X_val.shape[0]} samples | "
          f"Features: {X_train.shape[1]}")

    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=50,
    )

    evals = model.evals_result()
    val_rmse = evals['validation_1']['rmse']
    best = int(val_rmse.index(min(val_rmse)))
    print(f"[xgboost] Best iteration: {best}  |  Val RMSE: {min(val_rmse):.6f}")

    return model


# ─────────────────────────────────────────────
# 4. Evaluation
# ─────────────────────────────────────────────

def evaluate_xgboost(model: XGBRegressor,
                     X_test: np.ndarray,
                     y_test: np.ndarray,
                     target_scaler=None,
                     save_dir: str = "results") -> dict:
    """
    Predicts on test set, optionally inverse-transforms, computes metrics.

    If target_scaler is provided, predictions are in original price scale.
    """
    y_pred_raw = model.predict(X_test)

    if target_scaler is not None:
        y_pred = target_scaler.inverse_transform(y_pred_raw.reshape(-1, 1)).flatten()
        y_true = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    else:
        y_pred = y_pred_raw
        y_true = y_test

    metrics = _compute_metrics(y_true, y_pred)
    _print_metrics(metrics, label="XGBoost")

    os.makedirs(os.path.join(save_dir, "plots"), exist_ok=True)
    plot_path = os.path.join(save_dir, "plots", "xgboost_predictions.png")
    _plot_predictions(y_true, y_pred,
                      title="XGBoost — Actual vs Predicted",
                      save_path=plot_path)

    return {"y_true": y_true, "y_pred": y_pred, "metrics": metrics}


# ─────────────────────────────────────────────
# 5. Feature importance
# ─────────────────────────────────────────────

def plot_feature_importance(model: XGBRegressor,
                             feature_cols: list[str],
                             top_n: int = 20,
                             save_dir: str = "results/plots") -> None:
    """
    Plots and saves the top-N most important features by XGBoost gain score.
    """
    os.makedirs(save_dir, exist_ok=True)

    importance = model.feature_importances_
    pairs = sorted(zip(feature_cols, importance), key=lambda x: x[1], reverse=True)[:top_n]
    names, scores = zip(*pairs)

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(names)), scores[::-1], color="steelblue")
    plt.yticks(range(len(names)), names[::-1])
    plt.xlabel("Feature Importance (Gain)")
    plt.title(f"XGBoost Top-{top_n} Feature Importances")
    plt.tight_layout()
    path = os.path.join(save_dir, "xgboost_feature_importance.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[xgboost] Feature importance plot saved → {path}")


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

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "Directional_Accuracy": dir_acc}


def _print_metrics(metrics: dict, label: str = "") -> None:
    header = f"── {label} Metrics " if label else "── Metrics "
    print(f"\n{header}{'─' * (40 - len(header))}")
    for k, v in metrics.items():
        print(f"  {k:<25}: {v:.4f}")
    print()


def _plot_predictions(y_true, y_pred, title: str, save_path: str) -> None:
    plt.figure(figsize=(14, 5))
    plt.plot(y_true, label="Actual",    color="steelblue",  linewidth=1.5)
    plt.plot(y_pred, label="Predicted", color="tomato",     linewidth=1.5, linestyle="--")
    plt.title(title)
    plt.xlabel("Time Step")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[xgboost] Prediction plot saved → {save_path}")


# ─────────────────────────────────────────────
# 7. Save / Load
# ─────────────────────────────────────────────

def save_xgboost(model: XGBRegressor, path: str = "results/xgboost_final.pkl") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"[xgboost] Model saved → {path}")


def load_xgboost(path: str = "results/xgboost_final.pkl") -> XGBRegressor:
    model = joblib.load(path)
    print(f"[xgboost] Model loaded ← {path}")
    return model


# ─────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(0)
    N, F = 800, 25

    X = np.random.randn(N, F).astype(np.float32)
    y = np.random.randn(N).astype(np.float32)

    split_tr = int(0.7 * N)
    split_val = int(0.85 * N)

    X_tr,  y_tr  = X[:split_tr],        y[:split_tr]
    X_val, y_val = X[split_tr:split_val], y[split_tr:split_val]
    X_te,  y_te  = X[split_val:],        y[split_val:]

    model = build_xgboost(n_estimators=200)
    model = train_xgboost(model, X_tr, y_tr, X_val, y_val)

    feature_names = [f"feat_{i}" for i in range(F)]
    result = evaluate_xgboost(model, X_te, y_te)
    plot_feature_importance(model, feature_names)

    print("XGBoost smoke test complete.")
