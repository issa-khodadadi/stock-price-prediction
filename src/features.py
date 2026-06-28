"""
features.py
-----------
Adds technical indicators and engineered features to raw OHLCV data.
All indicators are computed with pandas only (no TA-Lib dependency).
"""

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────
# 1. Moving Averages
# ─────────────────────────────────────────────

def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple Moving Average (SMA) and Exponential Moving Average (EMA)
    for windows: 7, 21, 50 days.
    """
    df = df.copy()
    for window in [7, 21, 50]:
        df[f"SMA_{window}"] = df["Close"].rolling(window=window).mean()
        df[f"EMA_{window}"] = df["Close"].ewm(span=window, adjust=False).mean()
    return df


# ─────────────────────────────────────────────
# 2. RSI — Relative Strength Index
# ─────────────────────────────────────────────

def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    RSI measures momentum. Values > 70 = overbought, < 30 = oversold.
    """
    df = df.copy()
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)

    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    rs = avg_gain / (avg_loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))
    return df


# ─────────────────────────────────────────────
# 3. MACD — Moving Average Convergence Divergence
# ─────────────────────────────────────────────

def add_macd(df: pd.DataFrame,
             fast: int = 12,
             slow: int = 26,
             signal: int = 9) -> pd.DataFrame:
    """
    MACD = EMA(fast) - EMA(slow)
    Signal line = EMA(MACD, signal)
    Histogram = MACD - Signal
    """
    df = df.copy()
    ema_fast   = df["Close"].ewm(span=fast,   adjust=False).mean()
    ema_slow   = df["Close"].ewm(span=slow,   adjust=False).mean()
    df["MACD"]            = ema_fast - ema_slow
    df["MACD_Signal"]     = df["MACD"].ewm(span=signal, adjust=False).mean()
    df["MACD_Histogram"]  = df["MACD"] - df["MACD_Signal"]
    return df


# ─────────────────────────────────────────────
# 4. Bollinger Bands
# ─────────────────────────────────────────────

def add_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """
    Upper Band = SMA + (std * num_std)
    Lower Band = SMA - (std * num_std)
    Width      = (Upper - Lower) / Middle  → volatility measure
    """
    df = df.copy()
    sma = df["Close"].rolling(window=window).mean()
    std = df["Close"].rolling(window=window).std()

    df["BB_Upper"]  = sma + (std * num_std)
    df["BB_Middle"] = sma
    df["BB_Lower"]  = sma - (std * num_std)
    df["BB_Width"]  = (df["BB_Upper"] - df["BB_Lower"]) / (df["BB_Middle"] + 1e-9)
    # df["BB_Pct"]    = (df["Close"] - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"] + 1e-9)
    
    bb_range = (df["BB_Upper"] - df["BB_Lower"]).squeeze()
    df["BB_Pct"] = (df["Close"].squeeze() - df["BB_Lower"].squeeze()) / (bb_range + 1e-9)
    
    
    return df


# ─────────────────────────────────────────────
# 5. Volume Indicators
# ─────────────────────────────────────────────

def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Volume SMA (20-day)
    - Volume Ratio: today vs 20-day average
    - On-Balance Volume (OBV)
    """
    df = df.copy()
    df["Volume_SMA20"]  = df["Volume"].rolling(window=20).mean()
    df["Volume_Ratio"]  = df["Volume"] / (df["Volume_SMA20"] + 1e-9)

    # OBV
    obv = [0]
    for i in range(1, len(df)):
        if df["Close"].iloc[i] > df["Close"].iloc[i - 1]:
            obv.append(obv[-1] + df["Volume"].iloc[i])
        elif df["Close"].iloc[i] < df["Close"].iloc[i - 1]:
            obv.append(obv[-1] - df["Volume"].iloc[i])
        else:
            obv.append(obv[-1])
    df["OBV"] = obv
    return df


# ─────────────────────────────────────────────
# 6. Price-derived Features
# ─────────────────────────────────────────────

def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Daily Return: percentage change Close-to-Close
    - High-Low Range: (High - Low) / Close
    - Open-Close Body: (Close - Open) / Open
    - Lag features: Close at t-1, t-2, t-3, t-5
    - Rolling Volatility: std of returns over 7 and 21 days
    """
    df = df.copy()
    df["Daily_Return"]  = df["Close"].pct_change()
    df["HL_Range"]      = (df["High"] - df["Low"]) / (df["Close"] + 1e-9)
    df["OC_Body"]       = (df["Close"] - df["Open"]) / (df["Open"] + 1e-9)

    for lag in [1, 2, 3, 5]:
        df[f"Close_Lag{lag}"] = df["Close"].shift(lag)

    df["Volatility_7"]  = df["Daily_Return"].rolling(7).std()
    df["Volatility_21"] = df["Daily_Return"].rolling(21).std()

    return df


# ─────────────────────────────────────────────
# 7. Target column
# ─────────────────────────────────────────────

def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Target = next-day Close price (shift -1).
    Last row will be NaN and must be dropped before training.
    """
    df = df.copy()
    df["Target"] = df["Close"].shift(-1)
    return df


# ─────────────────────────────────────────────
# 8. Master pipeline
# ─────────────────────────────────────────────

def build_features(df: pd.DataFrame, drop_na: bool = True) -> pd.DataFrame:
    """
    Applies all feature engineering steps in order.
    Drops NaN rows introduced by rolling windows (if drop_na=True).

    Returns enriched DataFrame ready for modeling.
    """
    df = add_moving_averages(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger_bands(df)
    df = add_volume_features(df)
    df = add_price_features(df)
    df = add_target(df)

    if drop_na:
        before = len(df)
        df = df.dropna()
        print(f"[features] Dropped {before - len(df)} NaN rows after feature engineering.")

    print(f"[features] Final shape: {df.shape}  |  Columns: {len(df.columns)}")
    return df


def get_feature_columns(df: pd.DataFrame, exclude: list[str] | None = None) -> list[str]:
    """
    Returns list of feature column names (everything except 'Target').
    Optionally exclude raw OHLCV columns if you only want engineered features.
    """
    exclude = exclude or []
    exclude.append("Target")
    return [col for col in df.columns if col not in exclude]


# ─────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Simulate raw OHLCV data
    np.random.seed(42)
    dates  = pd.date_range("2020-01-01", periods=300, freq="B")
    close  = np.cumsum(np.random.randn(300) * 2) + 150
    dummy  = pd.DataFrame({
        "Open":   close + np.random.randn(300) * 0.5,
        "High":   close + np.abs(np.random.randn(300)),
        "Low":    close - np.abs(np.random.randn(300)),
        "Close":  close,
        "Volume": np.random.randint(1_000_000, 10_000_000, 300).astype(float),
    }, index=dates)

    enriched = build_features(dummy)
    print(enriched.tail(3))
    print("\nFeature columns:", get_feature_columns(enriched))
