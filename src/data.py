"""Data handling utilities for the stock AI analyzer."""

from __future__ import annotations

import datetime as dt
import atexit
import logging
import os
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from ta.momentum import StochRSIIndicator, WilliamsRIndicator
from ta.trend import ADXIndicator, CCIIndicator
from ta.volume import ChaikinMoneyFlowIndicator, OnBalanceVolumeIndicator

CACHE_DIR = Path(__file__).resolve().parent.parent / "data_cache"
CACHE_DIR.mkdir(exist_ok=True)

logger = logging.getLogger(__name__)

# Reuse a shared thread pool for I/O heavy ticker fetches to avoid per-request overhead.
MAX_FETCH_WORKERS = max(4, min(8, (os.cpu_count() or 4)))
_PRICE_EXECUTOR = ThreadPoolExecutor(max_workers=MAX_FETCH_WORKERS)
_PRICE_DATA_CACHE: "OrderedDict[tuple[str, int], PriceData]" = OrderedDict()
_RESAMPLED_CACHE: "OrderedDict[tuple[str, int, str], PriceData]" = OrderedDict()
_PRICE_CACHE_LOCK = Lock()
PRICE_CACHE_LIMIT = 20
RESAMPLE_CACHE_LIMIT = 24
atexit.register(_PRICE_EXECUTOR.shutdown, wait=False, cancel_futures=True)


def _copy_price_data(data: PriceData) -> PriceData:
    return PriceData(ticker=data.ticker, prices=data.prices.copy(), frequency=data.frequency)


def _remember_price_data(key: tuple[str, int], data: PriceData) -> None:
    with _PRICE_CACHE_LOCK:
        _PRICE_DATA_CACHE[key] = _copy_price_data(data)
        _PRICE_DATA_CACHE.move_to_end(key, last=True)
        while len(_PRICE_DATA_CACHE) > PRICE_CACHE_LIMIT:
            _PRICE_DATA_CACHE.popitem(last=False)


def _remember_resampled_data(key: tuple[str, int, str], data: PriceData) -> None:
    with _PRICE_CACHE_LOCK:
        _RESAMPLED_CACHE[key] = _copy_price_data(data)
        _RESAMPLED_CACHE.move_to_end(key, last=True)
        while len(_RESAMPLED_CACHE) > RESAMPLE_CACHE_LIMIT:
            _RESAMPLED_CACHE.popitem(last=False)


def _get_cached_price_data(key: tuple[str, int]) -> Optional[PriceData]:
    with _PRICE_CACHE_LOCK:
        cached = _PRICE_DATA_CACHE.get(key)
        if cached is None:
            return None
        _PRICE_DATA_CACHE.move_to_end(key, last=True)
        return _copy_price_data(cached)


def _get_cached_resampled_data(key: tuple[str, int, str]) -> Optional[PriceData]:
    with _PRICE_CACHE_LOCK:
        cached = _RESAMPLED_CACHE.get(key)
        if cached is None:
            return None
        _RESAMPLED_CACHE.move_to_end(key, last=True)
        return _copy_price_data(cached)

BENCHMARK_MAP = {
    "SS": "000300.SS",  # CSI 300
    "SZ": "399001.SZ",  # Shenzhen Component
    "BJ": "000300.SS",
    "HK": "^HSI",
    "US": "^GSPC",
    "CA": "^GSPTSE",
    "JP": "^N225",
    "EU": "^STOXX50E",
}

_BENCHMARK_CACHE: Dict[str, pd.Series] = {}
_MACRO_CACHE: Dict[str, pd.Series] = {}
_SECURITY_META_CACHE: Dict[str, Dict[str, object]] = {}

MACRO_SERIES = {
    "vix": "^VIX",
    "dxy": "DX-Y.NYB",
    "us10y": "^TNX",
}

REGION_MACRO_SERIES: Dict[str, Dict[str, str]] = {
    "SS": {
        "csi300": "000300.SS",
        "shanghai": "000001.SS",
        "cnh": "CNH=X",
    },
    "SZ": {
        "csi300": "000300.SS",
        "shanghai": "000001.SS",
        "cnh": "CNH=X",
    },
    "BJ": {
        "csi300": "000300.SS",
        "shanghai": "000001.SS",
        "cnh": "CNH=X",
    },
    "HK": {
        "hang_seng": "^HSI",
        "cnh": "CNH=X",
    },
    "US": {
        "nasdaq": "^IXIC",
        "russell": "^RUT",
    },
    "default": {
        "sp500": "^GSPC",
    },
}

REGION_FEATURE_KEYS: Dict[str, set[str]] = {}
for region, mapping in REGION_MACRO_SERIES.items():
    for key in mapping.keys():
        REGION_FEATURE_KEYS.setdefault(region, set()).add(key)

SECTOR_SERIES: Dict[str, str] = {
    "tech": "XLK",
    "finance": "XLF",
    "energy": "XLE",
    "healthcare": "XLV",
    "industry": "XLI",
}


def _sanitize_price_frame(prices: pd.DataFrame) -> pd.DataFrame:
    if prices is None or prices.empty:
        return prices
    working = prices.copy()
    if not working.index.is_monotonic_increasing:
        working = working.sort_index()
    working = working[~working.index.duplicated(keep="last")]
    numeric_cols = [col for col in working.columns if working[col].dtype.kind in {"i", "u", "f"}]
    working[numeric_cols] = working[numeric_cols].replace([np.inf, -np.inf], np.nan)
    required_cols = [col for col in ["Open", "High", "Low", "Close", "Adj Close"] if col in working.columns]
    if required_cols:
        working.dropna(subset=required_cols, inplace=True)
    if "Volume" in working.columns:
        working["Volume"] = working["Volume"].fillna(0.0)
        working = working[working["Volume"] >= 0.0]
    working = working.fillna(method="ffill").fillna(method="bfill")
    return working


def _ticker_region(ticker: str) -> str:
    if "." in ticker:
        suffix = ticker.split(".")[-1].upper()
        if suffix in {"SS", "SH"}:
            return "SS"
        if suffix in {"SZ"}:
            return "SZ"
        if suffix in {"BJ"}:
            return "BJ"
        if suffix in {"HK"}:
            return "HK"
        if suffix in {"TO"}:
            return "CA"
        if suffix in {"L"}:
            return "EU"
    else:
        # assume US ticker if letters only
        if ticker.isalpha():
            return "US"
    return "US"


def _benchmark_symbol_for(ticker: str) -> Optional[str]:
    region = _ticker_region(ticker)
    return BENCHMARK_MAP.get(region)


def _load_security_metadata(ticker: str) -> Dict[str, object]:
    cached = _SECURITY_META_CACHE.get(ticker)
    if cached is not None:
        return cached
    try:
        info = yf.Ticker(ticker)
        fast_info = getattr(info, "fast_info", {}) or {}
        basic_info = getattr(info, "info", {}) or {}
    except Exception:
        fast_info = {}
        basic_info = {}
    meta = {
        "sector": basic_info.get("sector") or basic_info.get("industry") or "unknown",
        "industry": basic_info.get("industry") or "unknown",
        "market_cap": fast_info.get("market_cap") or basic_info.get("marketCap"),
        "pe_ratio": fast_info.get("pe_ratio") or basic_info.get("trailingPE"),
        "forward_pe_ratio": basic_info.get("forwardPE"),
        "price_to_book": basic_info.get("priceToBook"),
        "beta": fast_info.get("beta") or basic_info.get("beta"),
        "dividend_yield": basic_info.get("dividendYield") or fast_info.get("dividend_yield"),
    }
    _SECURITY_META_CACHE[ticker] = meta
    return meta


def load_security_metadata(ticker: str) -> Dict[str, object]:
    """Public helper used by other modules."""
    return _load_security_metadata(ticker)


def _load_benchmark_series(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> Optional[pd.Series]:
    if not symbol:
        return None
    cached = _BENCHMARK_CACHE.get(symbol)
    if cached is not None:
        if cached.index.min() <= start and cached.index.max() >= end:
            return cached
    try:
        data = yf.download(symbol, start=start - pd.Timedelta(days=10), end=end + pd.Timedelta(days=10), progress=False, auto_adjust=True, threads=False)
    except Exception:
        return cached
    if data.empty:
        return cached
    column = "Adj Close" if "Adj Close" in data.columns else "Close"
    series = data[column]
    if isinstance(series, pd.DataFrame):
        if series.shape[1] == 0:
            return cached
        series = series.iloc[:, 0]
    series = series.copy()
    series.name = symbol
    series = series.sort_index()
    _BENCHMARK_CACHE[symbol] = series
    return series


def _load_macro_series(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> Optional[pd.Series]:
    if not symbol:
        return None
    cached = _MACRO_CACHE.get(symbol)
    if cached is not None:
        if cached.index.min() <= start and cached.index.max() >= end:
            return cached
    try:
        data = yf.download(
            symbol,
            start=start - pd.Timedelta(days=10),
            end=end + pd.Timedelta(days=10),
            progress=False,
            auto_adjust=True,
            threads=False,
        )
    except Exception:
        return cached
    if data.empty:
        return cached
    column = "Adj Close" if "Adj Close" in data.columns else "Close"
    series = data[column]
    if isinstance(series, pd.DataFrame):
        if series.shape[1] == 0:
            return cached
        series = series.iloc[:, 0]
    series = series.copy()
    series.name = symbol
    series = series.sort_index()
    _MACRO_CACHE[symbol] = series
    return series


def _normalize_price_columns(prices: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if isinstance(prices.columns, pd.MultiIndex):
        try:
            prices = prices.xs(ticker, axis=1, level=-1)
        except (KeyError, ValueError):
            prices.columns = prices.columns.get_level_values(0)
    prices.columns = [str(col) for col in prices.columns]
    return prices


@dataclass
class PriceData:
    ticker: str
    prices: pd.DataFrame
    frequency: str = "D"

    def with_returns(self, horizon: int = 5) -> pd.DataFrame:
        df = self.prices.copy()
        df[f"return_{horizon}d"] = df["Adj Close"].pct_change(periods=horizon).shift(-horizon)
        return df


def _download_price_history(
    ticker: str,
    start: dt.date,
    end: dt.date,
) -> pd.DataFrame:
    """Download OHLCV data from yfinance and normalize column layout."""
    data = yf.download(
        ticker,
        start=start.isoformat(),
        end=end.isoformat(),
        progress=False,
        auto_adjust=False,
        group_by="column",
        threads=False,
    )
    normalized = _normalize_price_columns(data, ticker)
    return _sanitize_price_frame(normalized)


def _ensure_adjusted_close(prices: pd.DataFrame, ticker: str, start: dt.date, end: dt.date, cache_file: Path) -> pd.DataFrame:
    """Re-fetch the price history when adjusted close is missing."""
    if "Adj Close" in prices.columns:
        return prices

    refreshed = _download_price_history(ticker, start, end)
    if "Adj Close" not in refreshed.columns:
        raise ValueError(f"Adjusted close price missing for {ticker}")
    refreshed.to_parquet(cache_file)
    return refreshed


def fetch_price_history(ticker: str, lookback_years: int = 5) -> PriceData:
    """Fetch historical OHLCV data with local caching."""
    cache_key = (ticker, lookback_years)
    cached = _get_cached_price_data(cache_key)
    if cached is not None:
        return cached
    end = dt.date.today()
    start = end - dt.timedelta(days=lookback_years * 365)
    start_ts = pd.Timestamp(start)

    cache_file = CACHE_DIR / f"{ticker}.parquet"
    if cache_file.exists():
        prices = pd.read_parquet(cache_file)
        prices = _normalize_price_columns(prices, ticker)
        prices = _sanitize_price_frame(prices)
    else:
        prices = pd.DataFrame()

    need_full_refresh = prices.empty
    if not need_full_refresh:
        prices.index = pd.to_datetime(prices.index)
        prices.sort_index(inplace=True)
        existing_start = prices.index.min().date()
        existing_end = prices.index.max().date()
        tolerance = dt.timedelta(days=3)
        if start < existing_start - tolerance:
            need_full_refresh = True
        elif end > existing_end + tolerance:
            incremental_start = existing_end + dt.timedelta(days=1)
            try:
                incremental = _download_price_history(
                    ticker,
                    incremental_start,
                    end,
                )
            except Exception:
                incremental = pd.DataFrame()
            if not incremental.empty:
                incremental.index = pd.to_datetime(incremental.index)
                prices = (
                    pd.concat([prices, incremental])
                    .sort_index()
                    .groupby(level=0)
                    .last()
                )
                prices.to_parquet(cache_file)
        prices = prices.loc[prices.index >= start_ts]

    if need_full_refresh:
        prices = _download_price_history(ticker, start, end)
        prices.to_parquet(cache_file)

    prices = _normalize_price_columns(prices, ticker)
    prices = _sanitize_price_frame(prices)
    prices = _ensure_adjusted_close(prices, ticker, start, end, cache_file)

    prices.index = pd.to_datetime(prices.index)
    prices.sort_index(inplace=True)
    prices = prices.loc[prices.index >= start_ts]
    result = PriceData(ticker=ticker, prices=prices, frequency="D")
    _remember_price_data(cache_key, result)
    return _copy_price_data(result)


def resample_price_data(price_data: PriceData, frequency: str) -> PriceData:
    """Aggregate OHLCV to a coarser frequency (e.g., weekly) for长期策略."""
    if frequency.upper() == price_data.frequency.upper():
        return price_data

    if price_data.prices.empty:
        return replace(price_data, frequency=frequency.upper())

    cache_key = (
        price_data.ticker,
        price_data.prices.index[0].date().toordinal(),
        price_data.prices.index[-1].date().toordinal(),
        frequency.upper(),
    )
    cached = _get_cached_resampled_data(cache_key)
    if cached is not None:
        return cached

    agg_map = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Adj Close": "last",
        "Volume": "sum",
    }
    label = "left"
    closed = "left"
    resampled = (
        price_data.prices.resample(frequency, label=label, closed=closed)
        .apply(agg_map)
        .dropna()
    )
    resampled.index = pd.to_datetime(resampled.index).tz_localize(None)
    if frequency.upper().startswith("W"):
        resampled.index = resampled.index - pd.Timedelta(days=6)
    resampled = _sanitize_price_frame(resampled)
    result = replace(price_data, prices=resampled, frequency=frequency.upper())
    _remember_resampled_data(cache_key, result)
    return _copy_price_data(result)


def build_feature_frame(price_data: PriceData, horizon: int = 5) -> pd.DataFrame:
    """Generate a feature-rich DataFrame for modeling."""
    df = price_data.with_returns(horizon=horizon)

    # Price momentum & volatility context
    daily_return = df["Adj Close"].pct_change()
    df["return_1d"] = daily_return
    df["return_5d"] = df["Adj Close"].pct_change(periods=5)
    df["return_20d"] = df["Adj Close"].pct_change(periods=20)
    df["return_60d"] = df["Adj Close"].pct_change(periods=60)
    df["return_120d"] = df["Adj Close"].pct_change(periods=120)
    df["return_3d"] = df["Adj Close"].pct_change(periods=3)
    df["return_10d"] = df["Adj Close"].pct_change(periods=10)
    df["return_horizon_past"] = df["Adj Close"].pct_change(periods=horizon)
    df["volatility_20d"] = daily_return.rolling(window=20).std()
    df["volatility_60d"] = daily_return.rolling(window=60).std()
    df["volatility_120d"] = daily_return.rolling(window=120).std()

    # Rolling moments
    df["return_zscore_20d"] = (df["return_1d"] - df["return_1d"].rolling(20).mean()) / df["return_1d"].rolling(20).std()
    df["skew_60d"] = daily_return.rolling(60).skew()
    df["kurtosis_60d"] = daily_return.rolling(60).kurt()

    # Volume surprises help filter low-conviction signals
    volume_mean = df["Volume"].rolling(window=20).mean()
    volume_std = df["Volume"].rolling(window=20).std()
    volume_zscore = (df["Volume"] - volume_mean) / volume_std
    volume_zscore = volume_zscore.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["volume_zscore"] = volume_zscore
    df["volume_ratio_20"] = (df["Volume"] / volume_mean).replace([np.inf, -np.inf], np.nan)
    df["volume_ratio_5"] = (df["Volume"] / df["Volume"].rolling(window=5).mean()).replace([np.inf, -np.inf], np.nan)

    df["sma_10"] = df["Adj Close"].rolling(window=10).mean()
    df["sma_50"] = df["Adj Close"].rolling(window=50).mean()
    df["sma_100"] = df["Adj Close"].rolling(window=100).mean()
    df["sma_200"] = df["Adj Close"].rolling(window=200).mean()
    df["ema_20"] = df["Adj Close"].ewm(span=20, adjust=False).mean()
    df["ema_50"] = df["Adj Close"].ewm(span=50, adjust=False).mean()
    df["ema_100"] = df["Adj Close"].ewm(span=100, adjust=False).mean()
    df["ema_200"] = df["Adj Close"].ewm(span=200, adjust=False).mean()
    df["ema_20_slope_5d"] = df["ema_20"].pct_change(periods=5)
    df["ema_50_slope_5d"] = df["ema_50"].pct_change(periods=5)
    df["ema_200_slope_10d"] = df["ema_200"].pct_change(periods=10)
    df["rsi_7"] = compute_rsi(df["Adj Close"], window=7)
    df["rsi_14"] = compute_rsi(df["Adj Close"], window=14)
    df["rsi_28"] = compute_rsi(df["Adj Close"], window=28)
    df["macd"] = df["Adj Close"].ewm(span=12, adjust=False).mean() - df["Adj Close"].ewm(span=26, adjust=False).mean()
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    df["bb_up"], df["bb_mid"], df["bb_low"], df["bb_pct"] = compute_bollinger(df["Adj Close"], window=20)
    bollinger_span = df["bb_up"] - df["bb_low"]
    df["bollinger_width"] = (bollinger_span / df["bb_mid"]).replace([np.inf, -np.inf], np.nan)
    df["atr_14"] = compute_atr(df, window=14)
    df["atr_ratio"] = df["atr_14"] / df["Adj Close"]
    keltner_upper = df["ema_20"] + df["atr_14"] * 1.5
    keltner_lower = df["ema_20"] - df["atr_14"] * 1.5
    df["keltner_channel_width"] = ((keltner_upper - keltner_lower) / df["ema_20"]).replace([np.inf, -np.inf], np.nan)

    # Additional technical indicators from `ta` library
    adx_indicator = ADXIndicator(high=df["High"], low=df["Low"], close=df["Close"], window=14, fillna=True)
    df["adx"] = adx_indicator.adx()
    df["adx_pos"] = adx_indicator.adx_pos()
    df["adx_neg"] = adx_indicator.adx_neg()

    cci_indicator = CCIIndicator(high=df["High"], low=df["Low"], close=df["Close"], window=20, constant=0.015, fillna=True)
    df["cci"] = cci_indicator.cci()

    stoch_rsi_indicator = StochRSIIndicator(close=df["Adj Close"], window=14, smooth1=3, smooth2=3, fillna=True)
    df["stoch_rsi"] = stoch_rsi_indicator.stochrsi()
    df["stoch_rsi_k"] = stoch_rsi_indicator.stochrsi_k()
    df["stoch_rsi_d"] = stoch_rsi_indicator.stochrsi_d()

    williams_r_indicator = WilliamsRIndicator(high=df["High"], low=df["Low"], close=df["Close"], lbp=14, fillna=True)
    df["williams_r"] = williams_r_indicator.williams_r()

    chaikin_indicator = ChaikinMoneyFlowIndicator(
        high=df["High"], low=df["Low"], close=df["Close"], volume=df["Volume"], window=20, fillna=True
    )
    df["chaikin_mf"] = chaikin_indicator.chaikin_money_flow()

    obv_indicator = OnBalanceVolumeIndicator(close=df["Adj Close"], volume=df["Volume"], fillna=True)
    df["obv"] = obv_indicator.on_balance_volume()
    df["obv_pct_change"] = df["obv"].pct_change().replace([np.inf, -np.inf], np.nan)

    df["atr_trend"] = df["atr_14"] / df["atr_14"].rolling(window=30, min_periods=10).mean() - 1.0
    df["volatility_ratio"] = df["volatility_20d"] / df["volatility_60d"].replace(0.0, np.nan)
    df["return_momentum_diff"] = df["return_20d"] - df["return_60d"]
    with np.errstate(divide="ignore", invalid="ignore"):
        df["trend_strength_20d"] = df["return_20d"] / df["volatility_20d"].replace(0.0, np.nan)
        df["trend_strength_60d"] = df["return_60d"] / df["volatility_60d"].replace(0.0, np.nan)
    df["trend_strength_20d"] = df["trend_strength_20d"].replace([np.inf, -np.inf], np.nan)
    df["trend_strength_60d"] = df["trend_strength_60d"].replace([np.inf, -np.inf], np.nan)
    df["range_pct"] = (df["High"] - df["Low"]) / df["Adj Close"]
    df["range_pct"] = df["range_pct"].replace([np.inf, -np.inf], np.nan)
    df["gap_pct"] = df["Open"] / df["Adj Close"].shift(1) - 1.0
    df["gap_pct"] = df["gap_pct"].replace([np.inf, -np.inf], np.nan)
    volume_mean_60 = df["Volume"].rolling(window=60, min_periods=20).mean()
    df["volume_trend_60d"] = (df["Volume"] / volume_mean_60) - 1.0
    df["volume_trend_60d"] = df["volume_trend_60d"].replace([np.inf, -np.inf], np.nan)
    df["volume_pct_rank_60d"] = (
        df["Volume"]
        .rolling(window=60, min_periods=20)
        .apply(lambda window: window.rank(pct=True).iloc[-1], raw=False)
    )

    df["obv_pct_change"] = df["obv_pct_change"].fillna(0.0)
    df["volatility_ratio"] = df["volatility_ratio"].replace([np.inf, -np.inf], np.nan)
    df["atr_trend"] = df["atr_trend"].replace([np.inf, -np.inf], np.nan)

    rolling_max = df["Adj Close"].rolling(window=max(20, horizon)).max()
    df["drawdown"] = (df["Adj Close"] / rolling_max) - 1.0
    df["price_sma_50_ratio"] = df["Adj Close"] / df["sma_50"] - 1.0
    df["price_ema_50_ratio"] = df["Adj Close"] / df["ema_50"] - 1.0
    df["price_ema_100_ratio"] = df["Adj Close"] / df["ema_100"] - 1.0

    high_252 = df["Adj Close"].rolling(window=252, min_periods=60).max()
    low_252 = df["Adj Close"].rolling(window=252, min_periods=60).min()
    df["pct_from_52w_high"] = df["Adj Close"] / high_252 - 1.0
    df["pct_from_52w_low"] = df["Adj Close"] / low_252 - 1.0
    price_mean_60 = df["Adj Close"].rolling(window=60, min_periods=20).mean()
    price_std_60 = df["Adj Close"].rolling(window=60, min_periods=20).std()
    df["price_zscore_60d"] = (df["Adj Close"] - price_mean_60) / price_std_60
    df["price_zscore_60d"] = df["price_zscore_60d"].replace([np.inf, -np.inf], np.nan)
    df["price_to_ema_20"] = df["Adj Close"] / df["ema_20"] - 1.0
    df["price_to_ema_100"] = df["Adj Close"] / df["ema_100"] - 1.0
    df["price_to_bollinger_band"] = (df["Adj Close"] - df["bb_mid"]) / (df["bb_up"] - df["bb_low"])
    df["price_to_bollinger_band"] = df["price_to_bollinger_band"].replace([np.inf, -np.inf], np.nan)

    benchmark_symbol = _benchmark_symbol_for(price_data.ticker)
    benchmark_series = _load_benchmark_series(benchmark_symbol, df.index.min(), df.index.max()) if benchmark_symbol else None
    if benchmark_series is not None and not benchmark_series.empty:
        bench = benchmark_series.reindex(df.index).ffill().bfill()
        bench_returns_5d = bench.pct_change(5, fill_method=None)
        bench_returns_20d = bench.pct_change(20, fill_method=None)
        bench_returns_60d = bench.pct_change(60, fill_method=None)
        df["bench_return_5d"] = bench_returns_5d
        df["bench_return_20d"] = bench_returns_20d
        df["bench_return_60d"] = bench_returns_60d
        df["relative_return_20d"] = df["return_20d"] - bench_returns_20d
        df["relative_return_60d"] = df["return_60d"] - bench_returns_60d
        df["relative_strength_pct"] = df["Adj Close"] / bench - 1.0
    else:
        df["bench_return_5d"] = np.nan
        df["bench_return_20d"] = np.nan
        df["bench_return_60d"] = np.nan
        df["relative_return_20d"] = np.nan
        df["relative_return_60d"] = np.nan
        df["relative_strength_pct"] = np.nan

    df["benchmark_spread"] = df["relative_return_20d"] - df["relative_return_60d"]
    df["relative_strength_pct"] = df["relative_strength_pct"].replace([np.inf, -np.inf], np.nan)

    # Macro context (volatility, rates, FX) supplies broader regime signals.
    macro_start = df.index.min()
    macro_end = df.index.max()
    for macro_key, symbol in MACRO_SERIES.items():
        series = _load_macro_series(symbol, macro_start, macro_end)
        if series is None or series.empty:
            df[f"macro_{macro_key}_level"] = np.nan
            df[f"macro_{macro_key}_pct_change_5d"] = np.nan
            df[f"macro_{macro_key}_zscore_60d"] = np.nan
            df[f"macro_{macro_key}_ema_ratio_30"] = np.nan
            continue
        aligned = series.reindex(df.index).ffill().bfill()
        df[f"macro_{macro_key}_level"] = aligned
        pct_change_5d = aligned.pct_change(5, fill_method=None)
        pct_change_5d = pct_change_5d.replace([np.inf, -np.inf], np.nan)
        df[f"macro_{macro_key}_pct_change_5d"] = pct_change_5d
        rolling_mean = aligned.rolling(60, min_periods=20).mean()
        rolling_std = aligned.rolling(60, min_periods=20).std()
        zscore = (aligned - rolling_mean) / rolling_std
        df[f"macro_{macro_key}_zscore_60d"] = zscore.replace([np.inf, -np.inf], np.nan)
        ema_30 = aligned.ewm(span=30, adjust=False, min_periods=10).mean()
        ratio_ema = aligned / ema_30 - 1.0
        df[f"macro_{macro_key}_ema_ratio_30"] = ratio_ema.replace([np.inf, -np.inf], np.nan)

    region_key = _ticker_region(price_data.ticker)
    region_macros = REGION_MACRO_SERIES.get(region_key) or REGION_MACRO_SERIES.get("default", {})
    for macro_key, symbol in region_macros.items():
        series = _load_macro_series(symbol, macro_start, macro_end)
        base = f"macro_region_{macro_key}"
        if series is None or series.empty:
            df[f"{base}_level"] = np.nan
            df[f"{base}_pct_change_5d"] = np.nan
            df[f"{base}_zscore_60d"] = np.nan
            df[f"{base}_ema_ratio_30"] = np.nan
            continue
        aligned = series.reindex(df.index).ffill().bfill()
        df[f"{base}_level"] = aligned
        pct_change_5d = aligned.pct_change(5, fill_method=None)
        pct_change_5d = pct_change_5d.replace([np.inf, -np.inf], np.nan)
        df[f"{base}_pct_change_5d"] = pct_change_5d
        rolling_mean = aligned.rolling(60, min_periods=20).mean()
        rolling_std = aligned.rolling(60, min_periods=20).std()
        zscore = (aligned - rolling_mean) / rolling_std
        df[f"{base}_zscore_60d"] = zscore.replace([np.inf, -np.inf], np.nan)
        ema_30 = aligned.ewm(span=30, adjust=False, min_periods=10).mean()
        ratio_ema = aligned / ema_30 - 1.0
        df[f"{base}_ema_ratio_30"] = ratio_ema.replace([np.inf, -np.inf], np.nan)

    df["macro_liquidity_spread"] = (
        df.get("macro_us10y_pct_change_5d", 0.0) - df.get("macro_vix_pct_change_5d", 0.0)
    )
    df["macro_risk_spread"] = (
        df.get("macro_vix_zscore_60d", 0.0) - df.get("macro_dxy_zscore_60d", 0.0)
    )
    df["macro_vix_to_us10y"] = (
        df.get("macro_vix_zscore_60d", 0.0) - df.get("macro_us10y_zscore_60d", 0.0)
    )

    sector_start = df.index.min()
    sector_end = df.index.max()
    for sector_key, symbol in SECTOR_SERIES.items():
        base = f"sector_{sector_key}"
        series = _load_macro_series(symbol, sector_start, sector_end)
        if series is None or series.empty:
            df[f"{base}_level"] = np.nan
            df[f"{base}_pct_change_5d"] = np.nan
            df[f"{base}_zscore_60d"] = np.nan
            df[f"{base}_ema_ratio_30"] = np.nan
            continue
        aligned = series.reindex(df.index).ffill().bfill()
        df[f"{base}_level"] = aligned
        pct_change_5d = aligned.pct_change(5, fill_method=None).replace([np.inf, -np.inf], np.nan)
        df[f"{base}_pct_change_5d"] = pct_change_5d
        rolling_mean = aligned.rolling(60, min_periods=20).mean()
        rolling_std = aligned.rolling(60, min_periods=20).std()
        zscore = (aligned - rolling_mean) / rolling_std
        df[f"{base}_zscore_60d"] = zscore.replace([np.inf, -np.inf], np.nan)
        ema_30 = aligned.ewm(span=30, adjust=False, min_periods=10).mean()
        ratio_ema = aligned / ema_30 - 1.0
        df[f"{base}_ema_ratio_30"] = ratio_ema.replace([np.inf, -np.inf], np.nan)

    vol_ratio = df["volatility_ratio"].replace([np.inf, -np.inf], np.nan).fillna(1.0)
    trend_strength = df["trend_strength_20d"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    risk_on = (-df.get("macro_vix_zscore_60d", pd.Series(0.0, index=df.index))).fillna(0.0)

    df["regime_volatility"] = np.clip(vol_ratio - 1.0, -2.5, 2.5)
    df["regime_trend"] = np.clip(trend_strength, -3.0, 3.0)
    df["regime_risk_on"] = np.clip(risk_on, -3.0, 3.0)
    df["regime_score"] = (
        0.4 * df["regime_volatility"]
        + 0.35 * df["regime_trend"]
        + 0.25 * df["regime_risk_on"]
    )
    df["trend_alignment_score"] = np.sign(df["return_20d"].fillna(0.0)) * np.sign(trend_strength)

    meta_snapshot = _load_security_metadata(price_data.ticker)

    def _meta_value(key: str) -> float | None:
        value = meta_snapshot.get(key)
        if value in (None, ""):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    df["fundamental_pe_ratio"] = _meta_value("pe_ratio")
    df["fundamental_forward_pe"] = _meta_value("forward_pe_ratio")
    df["fundamental_price_to_book"] = _meta_value("price_to_book")
    df["fundamental_beta"] = _meta_value("beta")
    df["fundamental_dividend_yield"] = _meta_value("dividend_yield")
    market_cap = _meta_value("market_cap")
    if market_cap and market_cap > 0:
        df["fundamental_market_cap_log"] = float(np.log(market_cap))
    else:
        df["fundamental_market_cap_log"] = np.nan

    fill_zero_cols = [
        "bench_return_5d",
        "bench_return_20d",
        "bench_return_60d",
        "relative_return_20d",
        "relative_return_60d",
        "relative_strength_pct",
    ]
    for col in fill_zero_cols:
        df[col] = df[col].fillna(0.0)

    df["volume_ratio_20"] = df["volume_ratio_20"].fillna(1.0)
    df["volume_ratio_5"] = df["volume_ratio_5"].fillna(1.0)
    df["return_zscore_20d"] = df["return_zscore_20d"].fillna(0.0)

    zero_fill_cols = [
        "return_20d",
        "return_60d",
        "return_120d",
        "return_3d",
        "return_10d",
        "volatility_20d",
        "volatility_60d",
        "volatility_120d",
        "skew_60d",
        "kurtosis_60d",
        "pct_from_52w_high",
        "pct_from_52w_low",
        "adx",
        "adx_pos",
        "adx_neg",
        "cci",
        "stoch_rsi",
        "stoch_rsi_k",
        "stoch_rsi_d",
        "williams_r",
        "chaikin_mf",
        "obv",
        "obv_pct_change",
        "atr_trend",
        "volatility_ratio",
        "return_momentum_diff",
        "benchmark_spread",
        "range_pct",
        "gap_pct",
        "volume_trend_60d",
        "volume_pct_rank_60d",
        "trend_strength_20d",
        "trend_strength_60d",
        "price_zscore_60d",
        "price_to_ema_20",
        "price_to_ema_100",
        "price_to_bollinger_band",
        "ema_20_slope_5d",
        "ema_50_slope_5d",
        "ema_200",
        "ema_200_slope_10d",
        "bollinger_width",
        "keltner_channel_width",
        "macro_liquidity_spread",
        "macro_risk_spread",
        "macro_vix_to_us10y",
        "regime_volatility",
        "regime_trend",
        "regime_risk_on",
        "regime_score",
        "trend_alignment_score",
        "fundamental_pe_ratio",
        "fundamental_forward_pe",
        "fundamental_price_to_book",
        "fundamental_beta",
        "fundamental_dividend_yield",
        "fundamental_market_cap_log",
    ]
    for macro_key in MACRO_SERIES:
        zero_fill_cols.extend(
            [
                f"macro_{macro_key}_pct_change_5d",
                f"macro_{macro_key}_zscore_60d",
                f"macro_{macro_key}_ema_ratio_30",
            ]
        )
    region_macro_keys = set()
    for mapping in REGION_MACRO_SERIES.values():
        region_macro_keys.update(mapping.keys())
    for macro_key in sorted(region_macro_keys):
        zero_fill_cols.extend(
            [
                f"macro_region_{macro_key}_level",
                f"macro_region_{macro_key}_pct_change_5d",
                f"macro_region_{macro_key}_zscore_60d",
                f"macro_region_{macro_key}_ema_ratio_30",
            ]
        )
    for sector_key in SECTOR_SERIES.keys():
        zero_fill_cols.extend(
            [
                f"sector_{sector_key}_level",
                f"sector_{sector_key}_pct_change_5d",
                f"sector_{sector_key}_zscore_60d",
                f"sector_{sector_key}_ema_ratio_30",
            ]
        )
    for col in zero_fill_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)
        else:
            df[col] = 0.0

    feature_cols = [col for col in df.columns if col != f"return_{horizon}d"]
    df[feature_cols] = df[feature_cols].fillna(0.0)

    return df


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_bollinger(series: pd.Series, window: int = 20, num_std: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    upper_band = rolling_mean + num_std * rolling_std
    lower_band = rolling_mean - num_std * rolling_std
    pct = (series - lower_band) / (upper_band - lower_band)
    return upper_band, rolling_mean, lower_band, pct


def compute_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)

    tr_components = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    true_range = tr_components.max(axis=1)
    atr = true_range.rolling(window=window).mean()
    return atr


def label_signals(
    df: pd.DataFrame,
    horizon: int = 5,
    threshold: float = 0.02,
    adaptive_threshold: bool = False,
    min_threshold: float = 0.01,
    max_threshold: float = 0.06,
    meta_quality_floor: float = 0.25,
) -> pd.DataFrame:
    """Label rows with buy/hold/sell classes based on future returns.

    When ``adaptive_threshold`` is True the absolute return requirement
    scales with recent volatility (ATR relative to price) so that highly
    volatile tickers (如 TSLL) do not overwhelm calmer holdings (如 VTI).
    """
    df = df.copy()
    future_return = df[f"return_{horizon}d"]
    label_available = future_return.notna()

    if adaptive_threshold:
        atr_fraction = (df["atr_14"] / df["Adj Close"]).clip(lower=min_threshold, upper=max_threshold)
        threshold_series = atr_fraction.fillna(threshold)
    else:
        threshold_series = pd.Series(threshold, index=df.index)

    df["label"] = "hold"
    df.loc[future_return >= threshold_series, "label"] = "buy"
    df.loc[future_return <= -threshold_series, "label"] = "sell"
    df["label_available"] = label_available

    abs_future = future_return.abs()
    meta_threshold = threshold_series * 1.15
    meta_threshold = meta_threshold.replace(0.0, threshold)
    confidence_ratio = (abs_future / meta_threshold.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    meta_confidence = confidence_ratio.fillna(0.0).clip(0.0, 5.0)
    positive_conf = meta_confidence[meta_confidence > 0]
    dynamic_floor = float(meta_quality_floor)
    if not positive_conf.empty:
        quantile_floor = float(positive_conf.quantile(0.7))
        if np.isfinite(quantile_floor):
            dynamic_floor = max(meta_quality_floor, min(quantile_floor, 0.95))
    quality_mask = meta_confidence >= dynamic_floor
    df["meta_quality_dynamic_floor"] = dynamic_floor
    df["meta_signal_active"] = (label_available & quality_mask & (abs_future >= meta_threshold)).astype(int)
    df["meta_signal_magnitude"] = future_return
    df["meta_signal_confidence"] = meta_confidence
    df["meta_signal_direction"] = np.sign(future_return).fillna(0.0)
    high_cut = min(dynamic_floor + 0.25, 0.95)
    medium_cut = max(dynamic_floor - 0.1, 0.0)
    df["meta_signal_quality_bucket"] = np.where(
        meta_confidence >= high_cut,
        "high",
        np.where(meta_confidence >= medium_cut, "medium", "low"),
    )
    return df


def build_dataset(
    tickers: List[str],
    lookback_years: int = 5,
    horizon: int = 5,
    threshold: float = 0.02,
    adaptive_threshold: bool = False,
    resample_frequency: str = "D",
    min_threshold: float = 0.01,
    max_threshold: float = 0.06,
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    freq_code = resample_frequency.upper()
    if freq_code not in {"D", "W"}:
        raise ValueError("resample_frequency must be 'D' (daily) or 'W' (weekly)")

    def _process_ticker(symbol: str) -> Optional[pd.DataFrame]:
        try:
            price_data = fetch_price_history(symbol, lookback_years=lookback_years)
            if freq_code != "D":
                price_data = resample_price_data(price_data, freq_code)
            features = build_feature_frame(price_data, horizon=horizon)
            if features.empty:
                return None
            labeled = label_signals(
                features,
                horizon=horizon,
                threshold=threshold,
                adaptive_threshold=adaptive_threshold,
                min_threshold=min_threshold,
                max_threshold=max_threshold,
            )
            labeled["ticker"] = symbol
            labeled["date"] = labeled.index
            return labeled
        except Exception:
            logger.exception("Failed to build dataset for %s", symbol)
            return None

    if tickers:
        future_map = {ticker: _PRICE_EXECUTOR.submit(_process_ticker, ticker) for ticker in tickers}
        for ticker in tickers:
            future = future_map[ticker]
            try:
                result = future.result()
            except Exception:
                logger.exception("Unhandled exception while processing %s", ticker)
                continue
            if result is not None:
                frames.append(result)

    if not frames:
        return pd.DataFrame()

    dataset = pd.concat(frames)
    dataset.dropna(subset=["label"], inplace=True)
    return dataset
