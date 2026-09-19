"""Feature engineering for ML signals, with leakage designed out.

TWO RULES
1. Every feature at bar ``t`` uses only bars ``<= t``. Enforced by building
   everything from :mod:`algotrade.indicators`, and verified in
   ``tests/test_causality.py``.
2. Features should be roughly **stationary**. Raw price is not: a model
   trained on BTC at \\$8k has never seen \\$60k and will extrapolate
   nonsense. Returns, z-scores and ratios are; levels are not.

Rule 2 is the one that silently destroys models. If a feature's mean drifts
over the sample, a chronological train/test split hands the model a test set
drawn from a different distribution -- and the model's failure will look like
"the strategy stopped working" rather than "the feature was never valid".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as ind

__all__ = ["make_features", "FEATURE_DESCRIPTIONS", "check_stationarity"]

FEATURE_DESCRIPTIONS = {
    "ret_1": "1-bar log return. The rawest signal there is.",
    "ret_6": "6-bar log return. Short-horizon momentum.",
    "ret_24": "24-bar log return. Daily momentum on hourly bars.",
    "ret_168": "168-bar log return. Weekly momentum.",
    "vol_24": "Trailing 24-bar realised volatility, annualised.",
    "vol_168": "Trailing 168-bar realised volatility, annualised.",
    "vol_ratio": "Short vol / long vol. Above 1 means volatility is rising.",
    "zscore_24": "Log price vs its trailing 24-bar mean, in trailing sds.",
    "zscore_168": "Same over a week. Mean-reversion signal.",
    "rsi_14": "RSI, rescaled to [-1, 1]. Overbought/oversold.",
    "macd_hist": "MACD histogram, normalised by price. Trend acceleration.",
    "bb_position": "Where price sits inside its Bollinger band (-1 to +1).",
    "atr_pct": "ATR as a fraction of price. Volatility in comparable units.",
    "volume_z": "Volume vs its trailing mean, in sds. Attention proxy.",
    "hl_range": "(high-low)/close. Intrabar range.",
    "close_loc": "Where the close sits within the bar's range (0=low, 1=high).",
    "hour_sin": "Hour of day, sine-encoded. Crypto has weak intraday seasonality.",
    "hour_cos": "Hour of day, cosine-encoded.",
}


def make_features(df: pd.DataFrame, *, include_time: bool = True) -> pd.DataFrame:
    """Build the workshop's standard feature matrix from an OHLCV frame.

    Every column is causal and (with the exception of the time encodings)
    approximately stationary. Returns a frame aligned to ``df.index``; the
    first rows are NaN during indicator warm-up and should be dropped
    **together with their labels**, not separately.
    """
    close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]
    log_close = np.log(close)
    ret1 = log_close.diff()

    f = pd.DataFrame(index=df.index)
    f["ret_1"] = ret1
    f["ret_6"] = log_close.diff(6)
    f["ret_24"] = log_close.diff(24)
    f["ret_168"] = log_close.diff(168)

    f["vol_24"] = ind.realized_volatility(ret1, 24)
    f["vol_168"] = ind.realized_volatility(ret1, 168)
    f["vol_ratio"] = f["vol_24"] / f["vol_168"].replace(0, np.nan)

    f["zscore_24"] = ind.zscore(log_close, 24)
    f["zscore_168"] = ind.zscore(log_close, 168)

    f["rsi_14"] = (ind.rsi(close, 14) - 50.0) / 50.0

    macd = ind.macd(close)
    f["macd_hist"] = macd["histogram"] / close

    bb = ind.bollinger(close, 20, 2.0)
    span = (bb["upper"] - bb["lower"]).replace(0, np.nan)
    f["bb_position"] = 2.0 * (close - bb["lower"]) / span - 1.0

    f["atr_pct"] = ind.atr(high, low, close, 14) / close
    f["volume_z"] = ind.zscore(volume, 168)
    f["hl_range"] = (high - low) / close
    f["close_loc"] = ((close - low) / (high - low).replace(0, np.nan)).fillna(0.5)

    if include_time:
        hour = df.index.hour.to_numpy()
        f["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        f["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    return f.replace([np.inf, -np.inf], np.nan)


def check_stationarity(features: pd.DataFrame, n_chunks: int = 4) -> pd.DataFrame:
    """Crude drift check: compare each feature's mean and sd across time chunks.

    Not a formal unit-root test -- just a fast way to spot the feature whose
    mean triples between the first quarter of the sample and the last. That
    feature will not survive a chronological split, and it is better to find
    out here than after training.

    ``mean_drift`` is the spread of chunk means expressed in units of the
    feature's own overall standard deviation. Anything above ~1 deserves a
    hard look.
    """
    clean = features.dropna()
    if len(clean) < n_chunks * 2:
        raise ValueError(f"need at least {n_chunks * 2} complete rows, got {len(clean)}")
    # NOTE: np.array_split on a DataFrame returns bare ndarrays and loses the
    # column names, which silently produces an all-NaN result. Slice instead.
    edges = np.linspace(0, len(clean), n_chunks + 1).astype(int)
    chunks = [clean.iloc[edges[k] : edges[k + 1]] for k in range(n_chunks)]
    means = pd.DataFrame([c.mean() for c in chunks])
    stds = pd.DataFrame([c.std() for c in chunks])
    overall_sd = features.std()
    return pd.DataFrame(
        {
            "mean_first": means.iloc[0],
            "mean_last": means.iloc[-1],
            "mean_drift": (means.max() - means.min()) / overall_sd.replace(0, np.nan),
            "sd_ratio": stds.iloc[-1] / stds.iloc[0].replace(0, np.nan),
        }
    ).sort_values("mean_drift", ascending=False)
