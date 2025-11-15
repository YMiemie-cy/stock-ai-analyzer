"""Reusable analysis pipeline helpers for CLI and API."""

from __future__ import annotations

import datetime as dt
import json
import math
import os
from collections import OrderedDict
from numbers import Number
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
from rich.console import Console

from .data import build_dataset, load_security_metadata
from .model import (
    ModelArtifacts,
    get_market_bias_factors,
    load_artifacts,
    predict_signals,
    train_model,
)
from .signals import generate_signal_report
from .backtest import naive_signal_backtest
from .tickers import normalize_tickers, market_for_ticker
from .insights import (
    ensure_scenario_shocks,
    build_daily_briefing,
    detect_event_alerts,
    build_scenario_matrix,
    build_portfolio_health,
    build_suggested_prompts,
    build_market_summary,
    build_performance_snapshot,
)


_MODEL_CACHE: Dict[tuple[str, str, str], ModelArtifacts] = {}
_SECTOR_META_CACHE: Dict[str, Dict[str, Any]] = {}
def _load_sector_metadata(ticker: str) -> Dict[str, Any]:
    cached = _SECTOR_META_CACHE.get(ticker)
    if cached is not None:
        return cached
    try:
        meta = load_security_metadata(ticker)
    except Exception:
        meta = {}
    if not isinstance(meta, dict):
        meta = {}
    _SECTOR_META_CACHE[ticker] = meta
    return meta



MARKET_DATA_OVERRIDES: Dict[str, Dict[str, object]] = {
    "china_a": {
        "threshold": 0.07,
        "min_threshold": 0.025,
        "max_threshold": 0.11,
        "hold_target_ratio": 0.28,
    },
    "global": {
        "threshold": 0.058,
        "min_threshold": 0.02,
        "max_threshold": 0.10,
        "hold_target_ratio": 0.2,
    },
}

LOW_QUALITY_HOLD_THRESHOLD = 0.32


def _rebalance_hold_samples(
    dataset: pd.DataFrame,
    *,
    target_ratio: float,
    random_state: int = 42,
) -> pd.DataFrame:
    if dataset.empty or target_ratio <= 0.0:
        return dataset
    counts = dataset["label"].value_counts()
    hold_count = counts.get("hold", 0)
    total = len(dataset)
    if total == 0:
        return dataset
    current_ratio = hold_count / total
    if current_ratio >= target_ratio or hold_count == 0:
        return dataset

    required_hold = int(target_ratio * total)
    additional = required_hold - hold_count
    if additional <= 0:
        return dataset

    hold_rows = dataset[dataset["label"] == "hold"]
    sampled = hold_rows.sample(n=additional, replace=True, random_state=random_state)
    rebalanced = pd.concat([dataset, sampled], axis=0)
    return rebalanced.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def ensure_model(
    model_name: str,
    dataset: pd.DataFrame,
    *,
    train: bool = False,
    console: Optional[Console] = None,
    market: str = "global",
    model_type: str = "auto",
) -> ModelArtifacts:
    """Load or train model artifacts."""
    artifacts_path = Path(__file__).resolve().parent.parent / "models" / f"{model_name}.joblib"
    should_train = train or not artifacts_path.exists()
    cache_key = (model_name, market, model_type)

    if not should_train:
        cached = _MODEL_CACHE.get(cache_key)
        if cached is not None:
            return cached

    if should_train:
        if console:
            console.print("[bold yellow]Training model...[/bold yellow]")
        artifacts = train_model(dataset, market=market, model_type=model_type)
        artifacts.save(model_name)
        if console:
            console.print(f"Model saved to {artifacts_path}")
        artifacts.market = market
        _MODEL_CACHE[cache_key] = artifacts
        if model_type == "auto":
            _MODEL_CACHE[(model_name, market, artifacts.model_type)] = artifacts
        return artifacts

    if console:
        console.print(f"Loading model artifacts from {artifacts_path}")
    artifacts = load_artifacts(model_name)
    artifacts.market = market
    if model_type != "auto" and getattr(artifacts, "model_type", model_type) != model_type:
        if console:
            console.print(
                f"[yellow]Existing artifact model_type={artifacts.model_type} "
                f"does not match requested {model_type}, retraining...[/yellow]"
            )
        artifacts = train_model(dataset, market=market, model_type=model_type)
        artifacts.save(model_name)
    if model_type == "auto":
        _MODEL_CACHE[(model_name, market, artifacts.model_type)] = artifacts
    if not isinstance(getattr(artifacts, "bias_factors", None), dict):
        artifacts.bias_factors = get_market_bias_factors(market)
    _MODEL_CACHE[cache_key] = artifacts
    return artifacts


def run_analysis(
    *,
    tickers: Iterable[str],
    lookback_years: int = 5,
    horizon: int = 5,
    threshold: float = 0.02,
    adaptive_threshold: bool = False,
    min_threshold: float = 0.01,
    max_threshold: float = 0.06,
    resample_frequency: str = "daily",
    model_name: str = "default_model",
    model_type: str = "auto",
    train: bool = False,
    console: Optional[Console] = None,
    deepseek_options: Optional[Dict[str, Any]] = None,
    risk_profile: str = "balanced",
    risk_limits: Optional[Dict[str, Optional[float]]] = None,
    scenario_shocks: Optional[Iterable[float]] = None,
    include_briefing: bool = True,
    briefing_top: int = 5,
    portfolio_path: Optional[str] = None,
) -> Dict[str, object]:
    """Execute the full analysis pipeline and return structured results."""
    freq_code = "D" if resample_frequency.lower() == "daily" else "W"
    deepseek_options = deepseek_options or {}
    deepseek_enabled = False
    deepseek_config = None
    deepseek_client = None
    deepseek_owned_client = False
    deepseek_error: str | None = None
    deepseek_stats_snapshot: Dict[str, Any] | None = None
    apply_deepseek_fusion = None

    if deepseek_options:
        try:
            from .deepseek import (  # type: ignore import-not-found
                DEFAULT_DEEPSEEK_MODEL,
                DeepseekClient,
                DeepseekConfig,
                apply_deepseek_fusion as _apply_deepseek_fusion,
            )
        except Exception as exc:  # pragma: no cover - import failure
            deepseek_error = f"Failed to import DeepSeek helpers: {exc}"
        else:
            apply_deepseek_fusion = _apply_deepseek_fusion
            options = dict(deepseek_options)
            config_candidate = options.get("config")
            if isinstance(config_candidate, DeepseekConfig):
                deepseek_config = config_candidate
            else:
                api_key = options.get("api_key") or os.environ.get("DEEPSEEK_API_KEY")
                if api_key:
                    weight = options.get("weight", 0.35)
                    try:
                        weight = float(weight)
                    except (TypeError, ValueError):
                        weight = 0.35
                    latest_only = bool(options.get("latest_only", True))
                    timeout = options.get("timeout", 25.0)
                    try:
                        timeout = float(timeout)
                    except (TypeError, ValueError):
                        timeout = 25.0
                    max_rows = options.get("max_rows", 60)
                    try:
                        max_rows = int(max_rows)
                    except (TypeError, ValueError):
                        max_rows = 60
                    trigger_prob_gap = options.get("trigger_prob_gap", 0.08)
                    trigger_quality = options.get("trigger_quality", 0.45)
                    try:
                        trigger_prob_gap = float(trigger_prob_gap)
                    except (TypeError, ValueError):
                        trigger_prob_gap = 0.08
                    try:
                        trigger_quality = float(trigger_quality)
                    except (TypeError, ValueError):
                        trigger_quality = 0.45
                    deepseek_config = DeepseekConfig(
                        api_key=api_key,
                        model=str(options.get("model") or DEFAULT_DEEPSEEK_MODEL),
                        base_url=str(options.get("base_url") or "https://api.deepseek.com"),
                        temperature=float(options.get("temperature", 0.2) or 0.2),
                        top_p=float(options.get("top_p", 0.85) or 0.85),
                        timeout=timeout,
                        weight=max(0.0, min(1.0, weight)),
                        latest_only=latest_only,
                        max_rows=max_rows,
                        apply_only_when_low_confidence=bool(options.get("apply_low_confidence", True)),
                        trigger_prob_gap=max(0.0, trigger_prob_gap),
                        trigger_quality=max(0.0, min(1.0, trigger_quality)),
                    )
            client_candidate = options.get("client")
            if deepseek_config:
                if isinstance(client_candidate, DeepseekClient):
                    deepseek_client = client_candidate
                elif client_candidate is None:
                    try:
                        deepseek_client = DeepseekClient(deepseek_config)
                        deepseek_owned_client = True
                    except Exception as exc:  # pragma: no cover - network init
                        deepseek_error = f"Failed to initialise DeepSeek client: {exc}"
                        deepseek_client = None
                else:
                    deepseek_error = "Invalid DeepSeek client provided."
            elif options.get("api_key") or os.environ.get("DEEPSEEK_API_KEY"):
                deepseek_error = "DeepSeek API key provided but configuration could not be constructed."

            deepseek_enabled = bool(deepseek_client and deepseek_config and apply_deepseek_fusion)

    def _clean_limit(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        return numeric if math.isfinite(numeric) else None

    risk_limits_dict = risk_limits or {}
    risk_limits_clean = {
        "max_drawdown": _clean_limit(risk_limits_dict.get("max_drawdown")),
        "target_volatility": _clean_limit(risk_limits_dict.get("target_volatility")),
    }
    scenario_shock_values = ensure_scenario_shocks(scenario_shocks)
    try:
        briefing_top = int(briefing_top)
    except (TypeError, ValueError):
        briefing_top = 5
    briefing_top = max(1, min(briefing_top, 15))
    normalized_tickers, ticker_mapping, invalid_inputs = normalize_tickers(tickers)
    if not normalized_tickers:
        raise ValueError("未能识别任何有效的股票代码，请输入正确的股票代码。")

    market_groups: "OrderedDict[str, List[str]]" = OrderedDict()
    market_lookup: Dict[str, str] = {}
    for norm in normalized_tickers:
        market = market_for_ticker(norm)
        market_lookup[norm] = market
        market_groups.setdefault(market, []).append(norm)

    def _model_name_for_market(base: str, market: str) -> str:
        if market == "china_a":
            return f"{base}_china"
        return base

    today = dt.date.today()
    datasets: List[pd.DataFrame] = []
    predictions_list: List[pd.DataFrame] = []
    available_norm: "set[str]" = set()
    model_name_by_market: Dict[str, str] = {}
    model_type_by_market: Dict[str, str] = {}


    def _sector_key(ticker: str) -> str:
        meta = _load_sector_metadata(ticker)
        sector = meta.get("sector") or meta.get("industry") or "unknown"
        return str(sector).lower()

    for market, market_tickers in market_groups.items():
        actual_model_name = _model_name_for_market(model_name, market)
        model_name_by_market[market] = actual_model_name

        sector_groups: "OrderedDict[str, List[str]]" = OrderedDict()
        for ticker in market_tickers:
            sector = _sector_key(ticker) if market == "china_a" else "global"
            sector_groups.setdefault(sector, []).append(ticker)

        for sector_key, sector_tickers in sector_groups.items():
            group_label = f"{market}:{sector_key}"
            data_overrides = MARKET_DATA_OVERRIDES.get(market, {})
            threshold_market = float(data_overrides.get("threshold", threshold))
            min_threshold_market = float(data_overrides.get("min_threshold", min_threshold))
            max_threshold_market = float(data_overrides.get("max_threshold", max_threshold))

            dataset_market = build_dataset(
                sector_tickers,
                lookback_years=lookback_years,
                horizon=horizon,
                threshold=threshold_market,
                adaptive_threshold=adaptive_threshold,
                resample_frequency=freq_code,
                min_threshold=min_threshold_market,
                max_threshold=max_threshold_market,
            )
            if dataset_market.empty:
                continue

            hold_target_ratio = float(data_overrides.get("hold_target_ratio", 0.0))
            train_dataset = _rebalance_hold_samples(
                dataset_market,
                target_ratio=hold_target_ratio,
                random_state=42,
            )

            datasets.append(dataset_market)
            available_norm.update(dataset_market["ticker"].unique())

            model_label = f"{actual_model_name}_{sector_key}" if sector_key != "global" else actual_model_name
            artifacts = ensure_model(
                model_name=model_label,
                dataset=train_dataset,
                train=train,
                console=console,
                market=market,
                model_type=model_type,
            )
            predictions = predict_signals(dataset_market, artifacts)
            if deepseek_enabled and apply_deepseek_fusion and deepseek_client and deepseek_config:
                ds_meta_context = {
                    "horizon": horizon,
                    "resample_frequency": resample_frequency,
                    "threshold": threshold_market,
                    "market": market,
                    "sector": sector_key,
                }
                predictions = apply_deepseek_fusion(
                    predictions,
                    client=deepseek_client,
                    config=deepseek_config,
                    meta=ds_meta_context,
                )
            predictions_list.append(predictions)
            model_type_by_market[group_label] = getattr(artifacts, "model_type", model_type)

    if not datasets:
        missing_all = [ticker_mapping.get(t, t) for t in normalized_tickers] + invalid_inputs
        raise ValueError(
            "未能获取任何有效数据，请检查股票代码或参数设置。"
            + (f"（尝试的代码：{', '.join(missing_all)}）" if missing_all else "")
        )

    dataset = pd.concat(datasets)
    predictions = pd.concat(predictions_list)

    if deepseek_client:
        stats_candidate = getattr(deepseek_client, "stats", None)
        if isinstance(stats_candidate, dict) and stats_candidate:
            deepseek_stats_snapshot = dict(stats_candidate)
    if deepseek_owned_client and deepseek_client and hasattr(deepseek_client, "close"):
        try:
            deepseek_client.close()
        except Exception:  # pragma: no cover - close failures are non-critical
            pass

    data_start = dataset["date"].min()
    data_end = dataset["date"].max()

    available_norm = sorted(set(available_norm))
    available_display = [ticker_mapping.get(t, t) for t in available_norm]
    requested_display = [ticker_mapping.get(t, t) for t in normalized_tickers]
    missing_norm = [t for t in normalized_tickers if t not in available_norm]
    missing_display = [ticker_mapping.get(t, t) for t in missing_norm]

    report_map: "OrderedDict[str, pd.DataFrame]" = OrderedDict()
    latest_map: "OrderedDict[str, Dict[str, object]]" = OrderedDict()
    backtest_map: "OrderedDict[str, Dict[str, object]]" = OrderedDict()

    def _compute_confidence(prob_gap: Optional[Number], indicator_bias: Optional[Number], quality_prob: Optional[Number] = None) -> float:
        gap_val = float(prob_gap) if isinstance(prob_gap, Number) and math.isfinite(prob_gap) else 0.0
        bias_val = float(indicator_bias) if isinstance(indicator_bias, Number) and math.isfinite(indicator_bias) else 0.0
        quality_val = float(quality_prob) if isinstance(quality_prob, Number) and math.isfinite(quality_prob) else 0.0
        prob_strength = max(gap_val, 0.0)
        indicator_strength = min(max(abs(bias_val) / 2.0, 0.0), 1.0)
        quality_strength = min(max(quality_val, 0.0), 1.0)
        return round(prob_strength * 0.55 + indicator_strength * 0.25 + quality_strength * 0.2, 4)

    def _build_driver_summary(row: pd.Series) -> List[str]:
        drivers: List[str] = []
        sma_10 = row.get("sma_10")
        sma_50 = row.get("sma_50")
        if isinstance(sma_10, Number) and isinstance(sma_50, Number):
            if sma_10 > sma_50:
                drivers.append("短期均线仍在中期均线之上")
            elif sma_10 < sma_50:
                drivers.append("短期均线已跌破中期均线")
        rsi_14 = row.get("rsi_14")
        if isinstance(rsi_14, Number) and math.isfinite(rsi_14):
            if rsi_14 >= 70:
                drivers.append("RSI 进入超买区间")
            elif rsi_14 <= 35:
                drivers.append("RSI 接近超卖")
        macd_val = row.get("macd")
        macd_sig = row.get("macd_signal")
        if isinstance(macd_val, Number) and isinstance(macd_sig, Number):
            if macd_val > macd_sig:
                drivers.append("MACD 柱线偏多")
            elif macd_val < macd_sig:
                drivers.append("MACD 柱线偏空")
        trend_20 = row.get("trend_strength_20d")
        if isinstance(trend_20, Number) and math.isfinite(trend_20):
            if trend_20 >= 0.7:
                drivers.append("20 日趋势强劲向上")
            elif trend_20 <= -0.7:
                drivers.append("20 日趋势显著走弱")
        indicator_bias = row.get("indicator_bias")
        if not drivers and isinstance(indicator_bias, Number):
            if indicator_bias >= 0.5:
                drivers.append("多项指标偏向做多")
            elif indicator_bias <= -0.5:
                drivers.append("多项指标偏向做空")
        return drivers[:3]

    for norm in normalized_tickers:
        display_label = ticker_mapping.get(norm, norm)
        ticker_df = predictions[predictions["ticker"] == norm]
        if ticker_df.empty:
            continue
        report_df = generate_signal_report(ticker_df)
        report_df = report_df.sort_index()
        report_map[display_label] = report_df
        if not report_df.empty:
            quality_mask = None
            if "meta_signal_prob" in report_df.columns:
                quality_mask = report_df["meta_signal_prob"].astype(float) < LOW_QUALITY_HOLD_THRESHOLD
                report_df["quality_override"] = quality_mask.fillna(False)
            latest_series = report_df.iloc[-1]
            latest_row = {
                key: (float(value) if isinstance(value, Number) else value)
                for key, value in latest_series.to_dict().items()
            }
            latest_timestamp = report_df.index[-1]
            latest_row["timestamp"] = latest_timestamp
            if isinstance(latest_timestamp, pd.Timestamp):
                latest_row["data_age_days"] = max(0, (today - latest_timestamp.date()).days)

            decision_series = report_df["decision"]
            change_mask = decision_series.ne(decision_series.shift())
            if change_mask.any():
                last_change_ts = report_df.index[change_mask].max()
            else:
                last_change_ts = report_df.index[0]
            if isinstance(last_change_ts, pd.Timestamp):
                latest_row["signal_changed_at"] = last_change_ts.to_pydatetime()
                age_delta = report_df.index[-1] - last_change_ts
                age_days = int(age_delta.days) if isinstance(age_delta, pd.Timedelta) else 0
                latest_row["signal_age_days"] = max(age_days, 0)
                latest_row["signal_is_recent"] = last_change_ts == report_df.index[-1]

            source_candidates = ticker_df[ticker_df.index == report_df.index[-1]]
            source_row = source_candidates.iloc[-1] if not source_candidates.empty else None
            if source_row is not None:
                metrics_fields = [
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
                ]
                for field in metrics_fields:
                    if field in source_row.index:
                        value = source_row.get(field)
                        latest_row[field] = (
                            float(value)
                            if isinstance(value, Number) and math.isfinite(value)
                            else None
                        )
                for field in ("rsi_14", "macd", "macd_signal", "sma_10", "sma_50"):
                    if field in source_row.index:
                        value = source_row.get(field)
                        latest_row[field] = (
                            float(value)
                            if isinstance(value, Number) and math.isfinite(value)
                            else None
                        )
                latest_row["key_drivers"] = _build_driver_summary(source_row)

            confidence_score = _compute_confidence(
                latest_series.get("prob_gap"),
                latest_series.get("indicator_bias"),
                latest_series.get("meta_signal_prob"),
            )
            latest_row["confidence_score"] = confidence_score

            prob_gap_value = latest_row.get("prob_gap")
            threshold_distance = latest_row.get("threshold_distance")
            decision_value = str(latest_row.get("decision", "")).lower()
            is_near_threshold = False
            if decision_value == "hold" and isinstance(threshold_distance, Number) and math.isfinite(threshold_distance):
                is_near_threshold = threshold_distance <= 0.0200005
            latest_row["is_near_threshold"] = is_near_threshold

            risk_flags: List[str] = []
            if confidence_score < 0.35 or (
                isinstance(prob_gap_value, Number) and math.isfinite(prob_gap_value) and prob_gap_value < 0.04
            ):
                risk_flags.append("low_confidence")
            quality_prob = latest_row.get("meta_signal_prob")
            if isinstance(quality_prob, Number) and math.isfinite(quality_prob) and quality_prob < 0.35:
                risk_flags.append("low_quality")
            if latest_row.get("signal_is_recent"):
                risk_flags.append("recent_flip")
            if is_near_threshold:
                risk_flags.append("near_threshold")
            latest_row["risk_flags"] = risk_flags
            if "key_drivers" not in latest_row:
                latest_row["key_drivers"] = []

            quality_override = False
            quality_threshold = LOW_QUALITY_HOLD_THRESHOLD
            if isinstance(quality_prob, Number) and math.isfinite(quality_prob):
                if quality_prob < quality_threshold:
                    quality_override = True
                    original_decision = str(latest_row.get("decision"))
                    latest_row["decision_raw"] = original_decision
                    latest_row["decision"] = "hold"
                    latest_row["action_hint"] = "保持观望（信号质量偏低）"
                    if "low_quality" not in latest_row["risk_flags"]:
                        latest_row["risk_flags"].append("low_quality")
            latest_row["quality_override"] = quality_override

            latest_price = latest_series.get("price")
            base_price = None
            if "signal_changed_at" in latest_row and latest_row["signal_changed_at"]:
                change_time = pd.Timestamp(latest_row["signal_changed_at"])
                if change_time in report_df.index:
                    change_rows = report_df.loc[change_time]
                    if isinstance(change_rows, pd.DataFrame):
                        change_series = change_rows.iloc[-1]
                    else:
                        change_series = change_rows
                    base_price = change_series.get("price")
            if isinstance(base_price, Number) and isinstance(latest_price, Number) and base_price not in (0, None):
                latest_row["return_since_signal"] = (float(latest_price) - float(base_price)) / float(base_price)
            else:
                latest_row["return_since_signal"] = None

            prob_diff_val = latest_row.get("prob_diff_buy_sell")
            latest_row["prob_diff_buy_sell"] = (
                float(prob_diff_val)
                if isinstance(prob_diff_val, Number) and math.isfinite(prob_diff_val)
                else None
            )
            latest_row["analysis_frequency"] = resample_frequency.lower()
            latest_row["action_hint"] = {
                "buy": "建议加仓",
                "sell": "建议减仓",
            }.get(decision_value, "保持观望")

            latest_map[display_label] = latest_row
        bt = naive_signal_backtest(report_df, horizon=max(1, horizon // 3))
        if bt is not None:
            backtest_map[display_label] = {
                "horizon": bt.horizon,
                "average_signal_return": bt.average_signal_return,
                "hit_ratio": bt.hit_ratio,
                "cumulative_return": bt.cumulative_return,
                "trades": bt.trades,
            }

    portfolio_holdings: List[str] = []
    portfolio_source: Optional[str] = None
    portfolio_source_error: Optional[str] = None
    if portfolio_path:
        candidate = Path(str(portfolio_path)).expanduser()
        if candidate.exists():
            portfolio_source = str(candidate)
            try:
                with candidate.open("r", encoding="utf-8") as fh:
                    portfolio_config = json.load(fh)
            except Exception as exc:  # pragma: no cover - rare path
                portfolio_source_error = f"读取持仓文件失败：{exc}"
            else:
                holdings = portfolio_config.get("tickers") or portfolio_config.get("holdings") or []
                if isinstance(holdings, list):
                    portfolio_holdings = [str(item).upper() for item in holdings if item]
        else:
            portfolio_source_error = f"未找到指定的持仓文件：{candidate}"
    else:
        default_portfolio = Path(__file__).resolve().parent.parent / "portfolio.json"
        if default_portfolio.exists():
            try:
                with default_portfolio.open("r", encoding="utf-8") as fh:
                    default_config = json.load(fh)
            except Exception as exc:  # pragma: no cover - rare path
                portfolio_source_error = f"读取默认持仓文件失败：{exc}"
            else:
                holdings = default_config.get("tickers") or default_config.get("holdings") or []
                if isinstance(holdings, list):
                    portfolio_holdings = [str(item).upper() for item in holdings if item]
                    portfolio_source = str(default_portfolio)

    meta = {
        "tickers": list(tickers),
        "normalized_tickers": normalized_tickers,
        "ticker_mapping": ticker_mapping,
        "lookback_years": lookback_years,
        "data_start": data_start.date() if pd.notna(data_start) else None,
        "data_end": data_end.date() if pd.notna(data_end) else None,
        "today": today,
        "resample_frequency": resample_frequency.lower(),
        "selected_frequencies": [resample_frequency.lower()],
        "horizon": horizon,
        "threshold": threshold,
        "min_threshold": min_threshold,
        "max_threshold": max_threshold,
        "adaptive_threshold": adaptive_threshold,
        "model_name": model_name,
        "model_type": model_type,
        "model_name_by_market": model_name_by_market,
        "model_type_by_market": model_type_by_market,
        "available_tickers": available_display,
        "requested_tickers": requested_display,
        "missing_tickers": missing_display,
        "invalid_inputs": invalid_inputs,
        "market_by_ticker": {
            ticker_mapping.get(t, t): market_lookup.get(t, "global") for t in normalized_tickers
        },
        "risk_profile": risk_profile,
        "risk_limits": risk_limits_clean,
        "scenario_shocks": scenario_shock_values,
        "briefing_top": briefing_top,
        "include_briefing": include_briefing,
        "portfolio_source": portfolio_source,
        "portfolio_source_error": portfolio_source_error,
    }
    meta["deepseek"] = {
        "enabled": deepseek_enabled,
        "model": getattr(deepseek_config, "model", None) if deepseek_enabled else None,
        "weight": getattr(deepseek_config, "weight", None) if deepseek_enabled else None,
        "latest_only": getattr(deepseek_config, "latest_only", None) if deepseek_enabled else None,
        "error": deepseek_error,
    }
    if deepseek_stats_snapshot:
        meta["deepseek"]["stats"] = deepseek_stats_snapshot

    insights: Dict[str, Any] = {}
    meta_for_briefing = dict(meta)
    if include_briefing:
        insights["daily_briefing"] = build_daily_briefing(
            latest_map,
            report_map,
            meta=meta_for_briefing,
            top_n=briefing_top,
            risk_limits=risk_limits_clean,
        )
    insights["event_alerts"] = detect_event_alerts(latest_map, risk_limits=risk_limits_clean)
    scenario_matrix = build_scenario_matrix(latest_map, scenario_shock_values)
    insights["scenario_matrix"] = scenario_matrix
    insights["portfolio_health"] = build_portfolio_health(
        latest_map,
        portfolio_holdings,
        risk_limits=risk_limits_clean,
    )
    insights["market_summary"] = build_market_summary(latest_map, meta_for_briefing)
    insights["performance_snapshot"] = build_performance_snapshot(latest_map, scenario_matrix)
    insights["suggested_prompts"] = build_suggested_prompts(
        latest_map,
        top_n=max(3, briefing_top),
    )
    insights["meta"] = {
        "scenario_shocks": scenario_shock_values,
        "risk_profile": risk_profile,
        "risk_limits": risk_limits_clean,
        "briefing_top": briefing_top,
        "include_briefing": include_briefing,
        "portfolio_source": portfolio_source,
        "portfolio_source_error": portfolio_source_error,
        "portfolio_holdings_count": len(portfolio_holdings),
    }

    return {
        "meta": meta,
        "reports": report_map,
        "latest": latest_map,
        "backtests": backtest_map,
        "dataset": dataset,
        "predictions": predictions,
        "insights": insights,
    }
