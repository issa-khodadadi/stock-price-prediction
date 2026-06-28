"""
data_loader.py
--------------
Downloads historical stock data using yfinance and prepares it for modeling.
"""

import os
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib


# ─────────────────────────────────────────────
# 1. Download raw OHLCV data
# ─────────────────────────────────────────────

def download_data(ticker: str, start: str, end: str, save_dir: str = "data/raw") -> pd.DataFrame:
    """
    Download historical OHLCV data from Yahoo Finance.

    Args:
        ticker  : Stock symbol, e.g. 'AAPL', 'GOOGL'
        start   : Start date string 'YYYY-MM-DD'
        end     : End date string   'YYYY-MM-DD'
        save_dir: Directory to save raw CSV

    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume
    """
    print(f"[data_loader] Downloading {ticker} from {start} to {end} ...")

    df = yf.download(ticker, start=start, end=end, auto_adjust=True)

    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'. Check symbol or date range.")

    # Keep only OHLCV columns
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.index.name = "Date"

    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f"{ticker}_{start}_{end}.csv")
    df.to_csv(file_path)
    print(f"[data_loader] Saved raw data → {file_path}  (shape: {df.shape})")

    return df


# ─────────────────────────────────────────────
# 2. Basic cleaning
# ─────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Drop rows with NaN values
    - Ensure index is DatetimeIndex sorted ascending
    - Remove duplicate dates
    """
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    before = len(df)
    df = df.dropna()
    after = len(df)
    if before != after:
        print(f"[data_loader] Dropped {before - after} rows with NaN values.")
    return df


# ─────────────────────────────────────────────
# 3. Train / Validation / Test split
# ─────────────────────────────────────────────

def split_data(df: pd.DataFrame,
               train_ratio: float = 0.70,
               val_ratio: float = 0.15) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Time-series aware split (no shuffling).

    Returns:
        train_df, val_df, test_df
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end   = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end]
    val_df   = df.iloc[train_end:val_end]
    test_df  = df.iloc[val_end:]

    print(f"[data_loader] Split → train: {len(train_df)} | val: {len(val_df)} | test: {len(test_df)}")
    return train_df, val_df, test_df


# ─────────────────────────────────────────────
# 4. Scaling (fit on train only)
# ─────────────────────────────────────────────

def scale_data(train_df: pd.DataFrame,
               val_df: pd.DataFrame,
               test_df: pd.DataFrame,
               feature_cols: list[str],
               target_col: str = "Close",
               save_dir: str = "data/processed") -> dict:
    """
    Fit MinMaxScaler on training data only, then transform all splits.
    Saves scalers for later inverse-transform at inference time.

    Returns a dict with keys:
        train, val, test         → scaled DataFrames
        feature_scaler           → fitted scaler for features
        target_scaler            → fitted scaler for target (Close)
    """
    os.makedirs(save_dir, exist_ok=True)

    feature_scaler = MinMaxScaler()
    target_scaler  = MinMaxScaler()

    # Fit on train
    feature_scaler.fit(train_df[feature_cols])
    target_scaler.fit(train_df[[target_col]])

    def _transform(df):
        scaled = df.copy()
        scaled[feature_cols] = feature_scaler.transform(df[feature_cols])
        scaled[[target_col]] = target_scaler.transform(df[[target_col]])
        return scaled

    train_scaled = _transform(train_df)
    val_scaled   = _transform(val_df)
    test_scaled  = _transform(test_df)

    # Persist scalers
    joblib.dump(feature_scaler, os.path.join(save_dir, "feature_scaler.pkl"))
    joblib.dump(target_scaler,  os.path.join(save_dir, "target_scaler.pkl"))
    print(f"[data_loader] Scalers saved → {save_dir}")

    return {
        "train": train_scaled,
        "val":   val_scaled,
        "test":  test_scaled,
        "feature_scaler": feature_scaler,
        "target_scaler":  target_scaler,
    }


# ─────────────────────────────────────────────
# 5. Sequence builder for LSTM
# ─────────────────────────────────────────────

def build_sequences(df: pd.DataFrame,
                    feature_cols: list[str],
                    target_col: str = "Close",
                    seq_len: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """
    Converts a DataFrame into (X, y) arrays for LSTM input.

    X shape: (samples, seq_len, n_features)
    y shape: (samples,)  — next-day Close price (scaled)
    """
    X, y = [], []
    data = df[feature_cols].values
    target = df[target_col].values

    for i in range(seq_len, len(df)):
        X.append(data[i - seq_len: i])
        y.append(target[i])

    return np.array(X), np.array(y)


# ─────────────────────────────────────────────
# 6. Quick-run pipeline
# ─────────────────────────────────────────────

def load_and_prepare(ticker: str,
                     start: str,
                     end: str,
                     feature_cols: list[str] | None = None,
                     seq_len: int = 60) -> dict:
    """
    End-to-end convenience function:
      download → clean → split → scale → build sequences

    feature_cols defaults to OHLCV if not provided.
    Call this after features.py has added technical indicators.
    """
    df_raw = download_data(ticker, start, end)
    df     = clean_data(df_raw)

    if feature_cols is None:
        feature_cols = ["Open", "High", "Low", "Close", "Volume"]

    train_df, val_df, test_df = split_data(df)

    scaled = scale_data(train_df, val_df, test_df,
                        feature_cols=feature_cols)

    X_train, y_train = build_sequences(scaled["train"], feature_cols, seq_len=seq_len)
    X_val,   y_val   = build_sequences(scaled["val"],   feature_cols, seq_len=seq_len)
    X_test,  y_test  = build_sequences(scaled["test"],  feature_cols, seq_len=seq_len)

    print(f"[data_loader] Sequences ready → "
          f"X_train: {X_train.shape} | X_val: {X_val.shape} | X_test: {X_test.shape}")

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val":   X_val,   "y_val":   y_val,
        "X_test":  X_test,  "y_test":  y_test,
        "df":      df,
        **{k: v for k, v in scaled.items() if k not in ("train", "val", "test")},
    }


# ─────────────────────────────────────────────
# Quick smoke-test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    result = load_and_prepare(
        ticker="AAPL",
        start="2018-01-01",
        end="2024-01-01",
    )
    print("\nDone! Keys:", list(result.keys()))
