"""FastAPI server exposing analysis endpoints and simple frontend."""

from __future__ import annotations

import asyncio
import json
import os
import time
import urllib.parse
from collections import OrderedDict
from datetime import datetime, timezone
from numbers import Number
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import httpx
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException, Query
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
from src.tickers import normalize_tickers

BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"
PORTFOLIO_PATH = BASE_DIR / "portfolio.json"

ANALYSIS_CACHE_TTL = 180  # seconds
REALTIME_CACHE_TTL = 60  # seconds
MAX_ANALYSIS_CACHE_ENTRIES = 12
HISTORY_EXPORT_LIMIT = 180
LOOKUP_CACHE_TTL = 600  # seconds
LOOKUP_LIMIT = 12
YAHOO_LOOKUP_URL = "https://query2.finance.yahoo.com/v1/finance/search"
YAHOO_LOOKUP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://finance.yahoo.com",
    "Referer": "https://finance.yahoo.com/",
}
SINA_LOOKUP_TEMPLATE = "https://suggest3.sinajs.cn/suggest/type=11,12,13,14,15&key={key}"

_ANALYSIS_CACHE: "OrderedDict[tuple, Dict[str, Any]]" = OrderedDict()
_REALTIME_CACHE: Dict[str, Dict[str, Any]] = {}
_LOOKUP_CACHE: Dict[str, Tuple[float, List[Dict[str, Any]]]] = {}

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


def _normalize_sina_symbol(raw: str) -> Optional[str]:
    if not raw:
        return None
    lowered = raw.strip().lower()
    if not lowered:
        return None
    if lowered.startswith("sh"):
        code = lowered[2:]
        if code.isdigit():
            return f"{code}.SS"
    elif lowered.startswith("sz"):
        code = lowered[2:]
        if code.isdigit():
            return f"{code}.SZ"
    elif lowered.startswith("bj"):
        code = lowered[2:]
        if code.isdigit():
            return f"{code}.BJ"
    elif lowered.startswith("hk"):
        code = lowered[2:]
        if code.isdigit():
            return f"{code}.HK"
    elif lowered.startswith("us"):
        code = lowered[2:]
        if code:
            return code.upper()
    return None


async def _lookup_yahoo_source(query: str, seen_symbols: Set[str]) -> List[Dict[str, Any]]:
    params = {
        "q": query,
        "quotesCount": LOOKUP_LIMIT,
        "newsCount": 0,
        "listsCount": 0,
    }
    try:
        async with httpx.AsyncClient(timeout=8.0, headers=YAHOO_LOOKUP_HEADERS) as client:
            response = await client.get(YAHOO_LOOKUP_URL, params=params)
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:  # pragma: no cover - network error
        status = exc.response.status_code
        if status == 429:
            raise HTTPException(status_code=429, detail="搜索频率过高，请稍后再试") from exc
        raise HTTPException(status_code=502, detail=f"代码搜索失败：{status}") from exc
    except httpx.HTTPError as exc:  # pragma: no cover - network error
        raise HTTPException(status_code=502, detail=f"代码搜索失败：{exc}") from exc

    data = response.json()
    quotes = data.get("quotes") or []
    results: List[Dict[str, Any]] = []
    for quote in quotes:
        symbol = quote.get("symbol")
        if not symbol or not isinstance(symbol, str):
            continue
        normalized, _, _ = normalize_tickers([symbol])
        if not normalized:
            continue
        normalized_symbol = normalized[0]
        if normalized_symbol in seen_symbols:
            continue
        seen_symbols.add(normalized_symbol)
        results.append(
            {
                "symbol": normalized_symbol,
                "display_symbol": symbol,
                "short_name": quote.get("shortname") or quote.get("longname"),
                "long_name": quote.get("longname"),
                "exchange": quote.get("exchDisp") or quote.get("exchange"),
                "type": quote.get("quoteType"),
                "score": quote.get("score"),
            }
        )
        if len(results) >= LOOKUP_LIMIT:
            break
    return results


async def _lookup_sina_source(query: str, seen_symbols: Set[str]) -> List[Dict[str, Any]]:
    try:
        encoded = urllib.parse.quote_from_bytes(query.encode("gbk"))
    except UnicodeEncodeError:
        encoded = urllib.parse.quote(query)
    url = SINA_LOOKUP_TEMPLATE.format(key=encoded)
    try:
        async with httpx.AsyncClient(
            timeout=6.0,
            headers={
                "User-Agent": YAHOO_LOOKUP_HEADERS["User-Agent"],
                "Accept": "*/*",
            },
        ) as client:
            response = await client.get(url)
            response.raise_for_status()
    except httpx.HTTPError as exc:  # pragma: no cover - network error
        raise HTTPException(status_code=502, detail=f"代码搜索失败：{exc}") from exc

    text = response.content.decode("gbk", errors="ignore")
    if "suggestvalue" not in text:
        return []
    parts = text.split('"', 2)
    if len(parts) < 2:
        return []
    payload = parts[1]
    entries = [entry for entry in payload.split(";") if entry]
    results: List[Dict[str, Any]] = []
    for entry in entries:
        fields = entry.split(",")
        if len(fields) < 4:
            continue
        normalized_symbol = _normalize_sina_symbol(fields[3])
        if not normalized_symbol or normalized_symbol in seen_symbols:
            continue
        short_name = fields[4] or fields[0]
        long_name = fields[0] or short_name
        exchange = normalized_symbol.split(".")[-1] if "." in normalized_symbol else None
        seen_symbols.add(normalized_symbol)
        results.append(
            {
                "symbol": normalized_symbol,
                "display_symbol": normalized_symbol,
                "short_name": short_name,
                "long_name": long_name,
                "exchange": exchange,
                "type": "EQUITY",
                "score": 75.0,
            }
        )
        if len(results) >= LOOKUP_LIMIT:
            break
    return results


async def _perform_symbol_lookup(query: str) -> List[Dict[str, Any]]:
    cache_key = query.strip().lower()
    now = time.time()
    cached = _LOOKUP_CACHE.get(cache_key)
    if cached and now - cached[0] < LOOKUP_CACHE_TTL:
        return cached[1]

    results: List[Dict[str, Any]] = []
    seen_symbols: set[str] = set()

    direct_symbols, _, _ = normalize_tickers([query])
    if direct_symbols:
        direct_symbol = direct_symbols[0]
        seen_symbols.add(direct_symbol)
        results.append(
            {
                "symbol": direct_symbol,
                "display_symbol": direct_symbol,
                "short_name": query.strip() or direct_symbol,
                "long_name": None,
                "exchange": direct_symbol.split(".")[-1] if "." in direct_symbol else None,
                "type": "DIRECT",
                "score": 1.0,
            }
        )

    errors: List[HTTPException] = []
    is_ascii_query = query.isascii()

    if is_ascii_query:
        try:
            results.extend(await _lookup_yahoo_source(query, seen_symbols))
        except HTTPException as exc:
            errors.append(exc)
    else:
        try:
            results.extend(await _lookup_sina_source(query, seen_symbols))
        except HTTPException as exc:
            if exc.status_code == 429 and results:
                _LOOKUP_CACHE[cache_key] = (now, results)
                return results
            errors.append(exc)
        if len(results) < LOOKUP_LIMIT:
            try:
                results.extend(await _lookup_yahoo_source(query, seen_symbols))
            except HTTPException as exc:
                errors.append(exc)

    if not results:
        if errors:
            raise errors[-1]
        raise HTTPException(status_code=404, detail="未找到匹配的代码")

    trimmed_results = results[:LOOKUP_LIMIT]
    _LOOKUP_CACHE[cache_key] = (now, trimmed_results)
    return trimmed_results


@app.get("/api/lookup")
async def lookup(q: str = Query(..., min_length=1, description="股票代码或名称")) -> Dict[str, Any]:
    query = q.strip()
    if not query:
        raise HTTPException(status_code=400, detail="查询内容不能为空")
    results = await _perform_symbol_lookup(query)
    return {"query": query, "results": results}


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
