"""Lightweight backtesting utilities for model signal evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    horizon: int
    average_signal_return: float
    hit_ratio: float
    cumulative_return: float
    trades: int


def naive_signal_backtest(report_df: pd.DataFrame, horizon: int = 4) -> Optional[BacktestResult]:
    """Compute simple forward returns for generated signals.

    The backtest assumes that `price` 列为 signal 生成时的价格，使用 `horizon`
    周后的价格评估买/卖建议。该函数不考虑交易成本，主要用于模型迭代时
    快速比较信号质量。
    """

    if "price" not in report_df.columns or report_df.empty:
        return None
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    df = report_df.sort_index().copy()
    df["future_price"] = df["price"].shift(-horizon)
    df["future_return"] = df["future_price"] / df["price"] - 1
    df.dropna(subset=["future_return"], inplace=True)
    if df.empty:
        return None

    def signal_return(row: pd.Series) -> float:
        ret = float(row["future_return"])
        if row["decision"] == "buy":
            return ret
        if row["decision"] == "sell":
            return -ret
        return 0.0

    df["strategy_return"] = df.apply(signal_return, axis=1)
    executed = df[df["decision"].isin({"buy", "sell"})]

    if executed.empty:
        return BacktestResult(horizon=horizon, average_signal_return=0.0, hit_ratio=0.0, cumulative_return=0.0, trades=0)

    avg_ret = executed["strategy_return"].mean()
    hit_ratio = (executed["strategy_return"] > 0).mean()
    cumulative = float(np.prod(1 + executed["strategy_return"]) - 1)

    return BacktestResult(
        horizon=horizon,
        average_signal_return=float(avg_ret),
        hit_ratio=float(hit_ratio),
        cumulative_return=cumulative,
        trades=len(executed),
    )
