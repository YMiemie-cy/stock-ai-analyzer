"""Hybrid signal generation blending model outputs with technical indicators."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _indicator_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Compute discrete indicator signals in a vectorised fashion."""
    sma_cross = np.where(df["sma_10"] > df["sma_50"], 1, np.where(df["sma_10"] < df["sma_50"], -1, 0))
    rsi_score = np.where(df["rsi_14"] < 30, 1, np.where(df["rsi_14"] > 70, -1, 0))
    macd_score = np.where(df["macd"] > df["macd_signal"], 1, np.where(df["macd"] < df["macd_signal"], -1, 0))
    bollinger_score = np.where(df["bb_pct"] < 0.1, 1, np.where(df["bb_pct"] > 0.9, -1, 0))
    volume_score = np.where(df["volume_zscore"] > 1.0, 1, np.where(df["volume_zscore"] < -1.0, -1, 0))

    scores = pd.DataFrame(
        {
            "indicator_sma": sma_cross,
            "indicator_rsi": rsi_score,
            "indicator_macd": macd_score,
            "indicator_bollinger": bollinger_score,
            "indicator_volume": volume_score,
        },
        index=df.index,
    )
    scores = scores.where(np.isfinite(scores), 0)
    scores["indicator_bias"] = scores.mean(axis=1)
    return scores


def generate_signal_report(dataset: pd.DataFrame) -> pd.DataFrame:
    if dataset.empty:
        return pd.DataFrame()

    working = dataset.copy()
    indicator_df = _indicator_scores(working)

    probs = working[["prob_buy", "prob_hold", "prob_sell"]].to_numpy()
    sorted_probs = np.sort(probs, axis=1)
    probability_gap = sorted_probs[:, -1] - sorted_probs[:, -2]
    prob_gap_series = pd.Series(probability_gap, index=working.index)

    buy_score = working["prob_buy"] + 0.15 * indicator_df["indicator_bias"]
    sell_score = working["prob_sell"] - 0.15 * indicator_df["indicator_bias"]
    final_score = buy_score - sell_score

    decision_threshold = 0.12 + np.clip(0.08 - probability_gap, 0.0, None)
    threshold_series = pd.Series(decision_threshold, index=working.index)
    threshold_distance = threshold_series - np.abs(final_score)
    short_vol = working["return_1d"].rolling(window=5, min_periods=1).std()
    mid_vol = working["return_1d"].rolling(window=10, min_periods=1).std()

    diff = final_score.values
    decisions = np.where(
        diff > decision_threshold,
        "buy",
        np.where((-diff) > decision_threshold, "sell", "hold"),
    )

    price_series = working["Adj Close"].where(working["Adj Close"].notna(), working.get("Close"))

    report = pd.DataFrame(
        {
            "decision": decisions,
            "score": final_score,
            "decision_threshold": threshold_series,
            "threshold_distance": threshold_distance,
            "prob_buy": working["prob_buy"],
            "prob_hold": working["prob_hold"],
            "prob_sell": working["prob_sell"],
            "prob_gap": prob_gap_series,
            "prob_diff_buy_sell": working["prob_buy"] - working["prob_sell"],
            "indicator_sma": indicator_df["indicator_sma"],
            "indicator_rsi": indicator_df["indicator_rsi"],
            "indicator_macd": indicator_df["indicator_macd"],
            "indicator_bollinger": indicator_df["indicator_bollinger"],
            "indicator_volume": indicator_df["indicator_volume"],
            "indicator_bias": indicator_df["indicator_bias"],
            "price": price_series,
            "model_signal": working.get("model_signal"),
            "return_1d": working.get("return_1d"),
            "return_3d": working.get("return_3d"),
            "return_5d": working.get("return_5d"),
            "return_10d": working.get("return_10d"),
            "return_20d": working.get("return_20d"),
            "return_60d": working.get("return_60d"),
            "volatility_5d": short_vol,
            "volatility_10d": mid_vol,
            "volatility_20d": working.get("volatility_20d"),
            "volatility_60d": working.get("volatility_60d"),
            "trend_strength_20d": working.get("trend_strength_20d"),
            "trend_strength_60d": working.get("trend_strength_60d"),
            "drawdown": working.get("drawdown"),
        },
        index=working.index,
    )
    optional_cols = [
        "prob_buy_local",
        "prob_hold_local",
        "prob_sell_local",
        "prob_buy_deepseek",
        "prob_hold_deepseek",
        "prob_sell_deepseek",
        "deepseek_label",
        "deepseek_confidence",
        "deepseek_reason",
    ]
    for col in optional_cols:
        if col in working.columns:
            report[col] = working[col]
    return report
