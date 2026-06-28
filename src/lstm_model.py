"""
lstm_model.py
-------------
Defines, trains, evaluates, and saves the LSTM model for stock price prediction.
Uses TensorFlow / Keras.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam


# ─────────────────────────────────────────────
# 1. Model architecture
# ─────────────────────────────────────────────

def build_lstm(input_shape: tuple,
               lstm_units: list[int] = [128, 64],
               dropout_rate: float = 0.2,
               learning_rate: float = 1e-3) -> tf.keras.Model:
    """
    Stacked LSTM with Dropout + BatchNorm.

    Args:
        input_shape   : (seq_len, n_features)
        lstm_units    : Number of units per LSTM layer
        dropout_rate  : Dropout fraction
        learning_rate : Adam optimizer lr

    Returns:
        Compiled Keras model
    """
    model = Sequential(name="StockLSTM")

    # First LSTM layer — return sequences so next LSTM can consume them
    model.add(LSTM(units=lstm_units[0],
                   return_sequences=len(lstm_units) > 1,
                   input_shape=input_shape,
                   name="lstm_1"))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    # Additional LSTM layers
    for i, units in enumerate(lstm_units[1:], start=2):
        return_seq = (i < len(lstm_units))   # False for the last layer
        model.add(LSTM(units=units, return_sequences=return_seq, name=f"lstm_{i}"))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))

    # Dense output
    model.add(Dense(32, activation="relu", name="dense_hidden"))
    model.add(Dense(1,  activation="linear", name="output"))

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="mean_squared_error",
        metrics=["mae"],
    )

    return model


# ─────────────────────────────────────────────
# 2. Training
# ─────────────────────────────────────────────

def train_lstm(model: tf.keras.Model,
               X_train: np.ndarray,
               y_train: np.ndarray,
               X_val:   np.ndarray,
               y_val:   np.ndarray,
               epochs: int = 100,
               batch_size: int = 32,
               save_dir: str = "results") -> dict:
    """
    Trains the LSTM with early stopping and LR reduction.

    Returns:
        Dict with keys: model, history
    """
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, "lstm_best.keras")

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6, verbose=1),
        ModelCheckpoint(filepath=checkpoint_path, monitor="val_loss",
                        save_best_only=True, verbose=0),
    ]

    print(f"[lstm] Training on {X_train.shape[0]} samples | Validating on {X_val.shape[0]} samples")
    print(f"[lstm] Input shape: {X_train.shape[1:]}")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    print(f"[lstm] Best model saved → {checkpoint_path}")
    return {"model": model, "history": history}


# ─────────────────────────────────────────────
# 3. Evaluation & inverse-transform
# ─────────────────────────────────────────────

def evaluate_lstm(model: tf.keras.Model,
                  X_test: np.ndarray,
                  y_test: np.ndarray,
                  target_scaler,
                  save_dir: str = "results") -> dict:
    """
    Predicts on test set, inverse-transforms to original price scale,
    and returns evaluation metrics + saves a plot.
    """
    # Raw predictions (scaled)
    y_pred_scaled = model.predict(X_test).flatten()

    # Inverse transform
    y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_true = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    metrics = _compute_metrics(y_true, y_pred)
    _print_metrics(metrics, label="LSTM")

    # Plot
    os.makedirs(os.path.join(save_dir, "plots"), exist_ok=True)
    plot_path = os.path.join(save_dir, "plots", "lstm_predictions.png")
    _plot_predictions(y_true, y_pred, title="LSTM — Actual vs Predicted", save_path=plot_path)

    return {"y_true": y_true, "y_pred": y_pred, "metrics": metrics}


# ─────────────────────────────────────────────
# 4. Plotting helpers
# ─────────────────────────────────────────────

def plot_training_history(history, save_dir: str = "results/plots") -> None:
    """Plots train vs validation loss over epochs."""
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["loss"],     label="Train Loss")
    axes[0].plot(history.history["val_loss"], label="Val Loss")
    axes[0].set_title("Loss (MSE)")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["mae"],      label="Train MAE")
    axes[1].plot(history.history["val_mae"],  label="Val MAE")
    axes[1].set_title("MAE")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(save_dir, "lstm_training_history.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[lstm] Training history plot saved → {path}")


def _plot_predictions(y_true, y_pred, title: str, save_path: str) -> None:
    plt.figure(figsize=(14, 5))
    plt.plot(y_true, label="Actual",    color="steelblue",  linewidth=1.5)
    plt.plot(y_pred, label="Predicted", color="darkorange", linewidth=1.5, linestyle="--")
    plt.title(title)
    plt.xlabel("Time Step")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[lstm] Prediction plot saved → {save_path}")


# ─────────────────────────────────────────────
# 5. Metrics
# ─────────────────────────────────────────────

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae  = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100

    # Directional accuracy
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


# ─────────────────────────────────────────────
# 6. Save / Load
# ─────────────────────────────────────────────

def save_lstm(model: tf.keras.Model, path: str = "results/lstm_final.keras") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)
    print(f"[lstm] Model saved → {path}")


def load_lstm(path: str = "results/lstm_final.keras") -> tf.keras.Model:
    model = load_model(path)
    print(f"[lstm] Model loaded ← {path}")
    return model


# ─────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    SEQ_LEN    = 60
    N_FEATURES = 20
    N_SAMPLES  = 500

    # Dummy data
    X = np.random.randn(N_SAMPLES, SEQ_LEN, N_FEATURES).astype(np.float32)
    y = np.random.randn(N_SAMPLES).astype(np.float32)

    split = int(0.8 * N_SAMPLES)
    X_tr, X_val = X[:split], X[split:]
    y_tr, y_val = y[:split], y[split:]

    model = build_lstm(input_shape=(SEQ_LEN, N_FEATURES))
    model.summary()

    result = train_lstm(model, X_tr, y_tr, X_val, y_val, epochs=5, batch_size=32)
    print("Training complete.")
