"""FastAPI server exposing analysis endpoints and simple frontend."""

from __future__ import annotations

import json
import asyncio
import os
import time
from collections import OrderedDict
from datetime import datetime, timezone
from numbers import Number
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.chatbot import (
    DEFAULT_BOT,
    SUPPORTED_BOTS,
    build_analysis_summary,
    build_chat_messages,
    collect_response_text,
)
from src.core import run_analysis

BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"
PORTFOLIO_PATH = BASE_DIR / "portfolio.json"

ANALYSIS_CACHE_TTL = 180  # seconds
REALTIME_CACHE_TTL = 60  # seconds
MAX_ANALYSIS_CACHE_ENTRIES = 12
HISTORY_EXPORT_LIMIT = 180

_ANALYSIS_CACHE: "OrderedDict[tuple, Dict[str, Any]]" = OrderedDict()
_REALTIME_CACHE: Dict[str, Dict[str, Any]] = {}

LATEST_RESULT_FIELDS = {
    "timestamp",
    "decision",
    "score",
    "decision_threshold",
    "threshold_distance",
    "prob_buy",
    "prob_hold",
    "prob_sell",
    "prob_gap",
    "prob_diff_buy_sell",
    "indicator_bias",
    "price",
    "model_signal",
    "return_1d",
    "return_3d",
    "return_5d",
    "return_10d",
    "return_20d",
    "return_60d",
    "volatility_5d",
    "volatility_10d",
    "volatility_20d",
    "volatility_60d",
    "trend_strength_20d",
    "trend_strength_60d",
    "drawdown",
    "confidence_score",
    "signal_changed_at",
    "signal_age_days",
    "signal_is_recent",
    "is_near_threshold",
    "risk_flags",
    "key_drivers",
    "return_since_signal",
    "analysis_frequency",
    "action_hint",
    "realtime_price",
    "realtime_price_timestamp",
}

HISTORY_RESULT_FIELDS = {
    "decision",
    "score",
    "prob_buy",
    "prob_hold",
    "prob_sell",
    "prob_gap",
    "price",
    "return_1d",
    "return_3d",
    "return_5d",
    "return_10d",
    "return_20d",
    "return_60d",
}


class AnalyzeRequest(BaseModel):
    tickers: List[str] = Field(..., description="List of tickers to analyze")
    lookback_years: int = 5
    horizon: int = 12
    threshold: float = 0.05
    adaptive_threshold: bool = True
    min_threshold: float = 0.01
    max_threshold: float = 0.06
    resample_frequency: str = "weekly"
    resample_frequencies: Optional[List[str]] = None
    model_name: str = "default_model"
    model_type: str = "auto"
    train: bool = False


def _download_realtime_batch(
    tickers: List[str],
    *,
    period: str,
    interval: str,
) -> Dict[str, Dict[str, Any]]:
    if not tickers:
        return {}

    try:
        data = yf.download(
            tickers=tickers if len(tickers) > 1 else tickers[0],
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=False,
            group_by="ticker",
            threads=False,
            timeout=15,
        )
    except Exception:
        return {}

    if data.empty:
        return {}

    frames: Dict[str, pd.DataFrame] = {}
    if isinstance(data.columns, pd.MultiIndex):
        # group_by="ticker" yields {ticker -> columns}
        for ticker in tickers:
            try:
                frames[ticker] = data[ticker]
            except (KeyError, TypeError):
                continue
    else:
        frames[tickers[0]] = data

    results: Dict[str, Dict[str, Any]] = {}
    for ticker, frame in frames.items():
        if frame is None or frame.empty:
            continue
        if isinstance(frame.columns, pd.MultiIndex):
            frame = frame.droplevel(0, axis=1)
        frame = frame.dropna(how="all")
        if frame.empty:
            continue
        last_row = frame.iloc[-1]
        price = last_row.get("Close")
        if price is None or pd.isna(price):
            price = last_row.get("Adj Close")
        if price is None or pd.isna(price):
            continue
        ts = frame.index[-1]
        timestamp = None
        if isinstance(ts, pd.Timestamp):
            dt_obj = ts.to_pydatetime()
            if ts.tzinfo is None:
                dt_obj = dt_obj.replace(tzinfo=timezone.utc)
            timestamp = dt_obj
        results[ticker] = {"price": float(price), "timestamp": timestamp}
    return results


def fetch_realtime_prices(tickers: List[str]) -> Dict[str, Dict[str, Any]]:
    """Fetch near-real-time prices for provided tickers using yfinance with caching."""
    prices: Dict[str, Dict[str, Any]] = {}
    unique = [t for t in dict.fromkeys(tickers) if t]
    if not unique:
        return prices

    now = time.time()
    stale: List[str] = []

    for ticker in unique:
        cached = _REALTIME_CACHE.get(ticker)
        if cached and (now - cached.get("fetched_at", 0)) < REALTIME_CACHE_TTL:
            prices[ticker] = {"price": cached.get("price"), "timestamp": cached.get("timestamp")}
        else:
            stale.append(ticker)

    fresh: Dict[str, Dict[str, Any]] = {}
    if stale:
        fresh.update(
            _download_realtime_batch(
                stale,
                period="1d",
                interval="1m",
            )
        )
        missing = [ticker for ticker in stale if ticker not in fresh]
        if missing:
            fresh.update(
                _download_realtime_batch(
                    missing,
                    period="5d",
                    interval="1d",
                )
            )

    for ticker in stale:
        fetched_at = time.time()
        entry = fresh.get(ticker, {})
        price_entry: Dict[str, Any] = {
            "price": entry.get("price"),
            "timestamp": entry.get("timestamp"),
        }
        if price_entry["price"] is None:
            cached = _REALTIME_CACHE.get(ticker)
            if cached:
                price_entry["price"] = cached.get("price")
                price_entry["timestamp"] = cached.get("timestamp")
        _REALTIME_CACHE[ticker] = {
            "price": price_entry.get("price"),
            "timestamp": price_entry.get("timestamp"),
            "fetched_at": fetched_at,
        }
        prices[ticker] = price_entry

    return prices


def dataframe_to_history(
    report_df,
    *,
    limit: Optional[int] = None,
    allowed_fields: Optional[set[str]] = None,
) -> List[Dict[str, Any]]:
    history = []
    frame = report_df
    if limit is not None and len(frame) > limit:
        frame = frame.tail(limit)
    for timestamp, row in frame.iterrows():
        row_dict = {}
        for key, value in row.to_dict().items():
            if allowed_fields is not None and key not in allowed_fields:
                continue
            if isinstance(value, bool):
                row_dict[key] = value
            elif isinstance(value, Number):
                row_dict[key] = float(value)
            else:
                row_dict[key] = value
        row_dict["timestamp"] = timestamp.isoformat()
        history.append(row_dict)
    return history


def _analysis_cache_key(
    *,
    tickers: List[str],
    lookback_years: int,
    horizon: int,
    threshold: float,
    adaptive_threshold: bool,
    min_threshold: float,
    max_threshold: float,
    resample_frequency: str,
    model_name: str,
    model_type: str,
) -> tuple:
    return (
        tuple(tickers),
        lookback_years,
        horizon,
        round(threshold, 6),
        adaptive_threshold,
        round(min_threshold, 6),
        round(max_threshold, 6),
        resample_frequency,
        model_name,
        model_type,
    )


def get_or_run_analysis(
    *,
    tickers: List[str],
    lookback_years: int,
    horizon: int,
    threshold: float,
    adaptive_threshold: bool,
    min_threshold: float,
    max_threshold: float,
    resample_frequency: str,
    model_name: str,
    model_type: str,
    train: bool,
) -> Dict[str, Any]:
    freq_norm = (resample_frequency or "weekly").lower()
    model_name_effective = model_name
    if freq_norm.startswith("d"):
        if not model_name or model_name == "default_model":
            model_name_effective = "default_model_daily"
    elif freq_norm.startswith("w") and model_name == "default_model_daily":
        model_name_effective = "default_model"

    cache_key = _analysis_cache_key(
        tickers=tickers,
        lookback_years=lookback_years,
        horizon=horizon,
        threshold=threshold,
        adaptive_threshold=adaptive_threshold,
        min_threshold=min_threshold,
        max_threshold=max_threshold,
        resample_frequency=resample_frequency,
        model_name=model_name_effective,
        model_type=model_type,
    )

    now = time.time()
    if not train:
        cached = _ANALYSIS_CACHE.get(cache_key)
        if cached and (now - cached.get("timestamp", 0)) < ANALYSIS_CACHE_TTL:
            _ANALYSIS_CACHE.move_to_end(cache_key, last=True)
            return cached["results"]

    results = run_analysis(
        tickers=tickers,
        lookback_years=lookback_years,
        horizon=horizon,
        threshold=threshold,
        adaptive_threshold=adaptive_threshold,
        min_threshold=min_threshold,
        max_threshold=max_threshold,
        resample_frequency=resample_frequency,
        model_name=model_name_effective,
        model_type=model_type,
        train=train,
        console=None,
    )

    if not train:
        cached_results = {
            "meta": results["meta"],
            "reports": results["reports"],
            "latest": results["latest"],
            "backtests": results.get("backtests", {}),
        }
        _ANALYSIS_CACHE[cache_key] = {"results": cached_results, "timestamp": now}
        _ANALYSIS_CACHE.move_to_end(cache_key, last=True)
        while len(_ANALYSIS_CACHE) > MAX_ANALYSIS_CACHE_ENTRIES:
            _ANALYSIS_CACHE.popitem(last=False)

    return results


def transform_results(
    results: Dict[str, Any],
    *,
    realtime_cache: Optional[Dict[str, Dict[str, Any]]] = None,
    realtime_timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    meta = results["meta"]
    normalized = meta.get("normalized_tickers", [])
    ticker_mapping = meta.get("ticker_mapping", {})
    reverse_mapping = {display: norm for norm, display in ticker_mapping.items()}
    realtime_raw = realtime_cache if realtime_cache is not None else fetch_realtime_prices(normalized)
    realtime_fetch_ts = (
        realtime_timestamp
        if realtime_timestamp is not None
        else datetime.now(tz=timezone.utc).isoformat()
    )

    data = {
        "meta": {
            "data_start": str(meta.get("data_start")),
            "data_end": str(meta.get("data_end")),
            "today": str(meta.get("today")),
            "resample_frequency": meta.get("resample_frequency"),
            "selected_frequencies": meta.get("selected_frequencies", []),
            "horizon": meta.get("horizon"),
            "lookback_years": meta.get("lookback_years"),
            "available_tickers": meta.get("available_tickers", []),
            "missing_tickers": meta.get("missing_tickers", []),
            "invalid_inputs": meta.get("invalid_inputs", []),
            "model_type": meta.get("model_type"),
            "model_type_by_market": meta.get("model_type_by_market", {}),
            "model_name": meta.get("model_name"),
            "model_name_by_market": meta.get("model_name_by_market", {}),
            "threshold": meta.get("threshold"),
            "min_threshold": meta.get("min_threshold"),
            "max_threshold": meta.get("max_threshold"),
            "adaptive_threshold": meta.get("adaptive_threshold"),
            "requested_tickers": meta.get("requested_tickers", []),
            "realtime_timestamp": realtime_fetch_ts,
        },
        "tickers": [],
    }

    backtests = results.get("backtests", {})

    for ticker, report_df in results["reports"].items():
        latest = results["latest"].get(ticker, {})
        normalized_key = reverse_mapping.get(ticker, ticker)
        realtime_entry = realtime_raw.get(normalized_key, {})
        latest_copy: Dict[str, Any] = {}
        for key, value in latest.items():
            if key not in LATEST_RESULT_FIELDS:
                continue
            if isinstance(value, datetime):
                latest_copy[key] = value.isoformat()
            elif isinstance(value, bool):
                latest_copy[key] = value
            elif isinstance(value, Number):
                latest_copy[key] = float(value)
            else:
                latest_copy[key] = value
        if realtime_entry:
            price_val = realtime_entry.get("price")
            ts_val = realtime_entry.get("timestamp")
            if price_val is not None:
                latest_copy["realtime_price"] = float(price_val)
            if ts_val is not None:
                latest_copy["realtime_price_timestamp"] = ts_val.isoformat()
        history = dataframe_to_history(
            report_df,
            limit=HISTORY_EXPORT_LIMIT,
            allowed_fields=HISTORY_RESULT_FIELDS,
        )
        bt = backtests.get(ticker)
        data["tickers"].append(
            {
                "ticker": ticker,
                "latest": latest_copy,
                "history": history,
                "backtest": bt,
            }
        )
    return data


def load_portfolio_config() -> Dict[str, Any]:
    if not PORTFOLIO_PATH.exists():
        raise FileNotFoundError("portfolio.json not found.")
    with PORTFOLIO_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


app = FastAPI(title="Stock AI Analyzer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

POE_API_KEY = os.getenv("POE_API_KEY")


class ChatHistoryItem(BaseModel):
    role: str
    content: str


class ChatRequest(AnalyzeRequest):
    question: str
    bot_name: Optional[str] = None
    history: List[ChatHistoryItem] = []


def _select_bot(requested: Optional[str]) -> str:
    if requested and requested in SUPPORTED_BOTS:
        return requested
    return DEFAULT_BOT


@app.get("/", response_class=HTMLResponse)
async def serve_index() -> HTMLResponse:
    index_path = WEB_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


@app.post("/api/analyze")
async def analyze(request: AnalyzeRequest) -> Dict[str, Any]:
    if not request.tickers:
        raise HTTPException(status_code=400, detail="tickers list cannot be empty")

    try:
        results = await asyncio.to_thread(
            get_or_run_analysis,
            tickers=request.tickers,
            lookback_years=request.lookback_years,
            horizon=request.horizon,
            threshold=request.threshold,
            adaptive_threshold=request.adaptive_threshold,
            min_threshold=request.min_threshold,
            max_threshold=request.max_threshold,
            resample_frequency=request.resample_frequency,
            model_name=request.model_name,
            model_type=request.model_type,
            train=request.train,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return transform_results(results)


@app.get("/api/portfolio")
async def portfolio(train: Optional[bool] = False, frequency: Optional[str] = None) -> Dict[str, Any]:
    try:
        config = load_portfolio_config()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    options = config.get("options", {})
    try:
        results = await asyncio.to_thread(
            get_or_run_analysis,
            tickers=config.get("tickers", []),
            lookback_years=options.get("lookback_years", 5),
            horizon=options.get("horizon", 12),
            threshold=options.get("threshold", 0.05),
            adaptive_threshold=options.get("adaptive_threshold", True),
            min_threshold=options.get("min_threshold", 0.01),
            max_threshold=options.get("max_threshold", 0.06),
            resample_frequency=frequency or options.get("resample_frequency", "weekly"),
            model_name=options.get("model_name", "default_model"),
            model_type=options.get("model_type", "auto"),
            train=bool(train),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return transform_results(results)


@app.post("/api/chat")
async def chat(request: ChatRequest) -> Dict[str, Any]:
    if not request.tickers:
        raise HTTPException(status_code=400, detail="tickers list cannot be empty")
    if not POE_API_KEY:
        raise HTTPException(status_code=500, detail="POE_API_KEY 未配置，无法调用智能对话功能")

    try:
        results = await asyncio.to_thread(
            get_or_run_analysis,
            tickers=request.tickers,
            lookback_years=request.lookback_years,
            horizon=request.horizon,
            threshold=request.threshold,
            adaptive_threshold=request.adaptive_threshold,
            min_threshold=request.min_threshold,
            max_threshold=request.max_threshold,
            resample_frequency=request.resample_frequency,
            model_name=request.model_name,
            model_type=request.model_type,
            train=request.train,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    normalized = results["meta"].get("normalized_tickers", [])
    realtime_cache = fetch_realtime_prices(normalized)
    realtime_timestamp = datetime.now(tz=timezone.utc).isoformat()
    summary = build_analysis_summary(results, realtime_cache)

    history_payload = [item.dict() for item in request.history if item.content]
    messages = build_chat_messages(summary=summary, question=request.question, history=history_payload)
    bot_name = _select_bot(request.bot_name)

    try:
        answer = await asyncio.to_thread(collect_response_text, messages, bot_name, POE_API_KEY)
    except Exception as exc:  # pragma: no cover - rely on runtime
        raise HTTPException(status_code=502, detail=f"AI 助手调用失败：{exc}") from exc

    analysis_payload = transform_results(
        results,
        realtime_cache=realtime_cache,
        realtime_timestamp=realtime_timestamp,
    )

    return {
        "answer": answer,
        "bot_name": bot_name,
        "analysis": analysis_payload,
        "summary": summary,
    }
