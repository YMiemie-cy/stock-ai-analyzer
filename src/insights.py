"""High-level insight builders for daily briefings, alerts, and stress tests."""

from __future__ import annotations

import datetime as dt
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

DEFAULT_SCENARIO_SHOCKS: Tuple[float, ...] = (-0.06, -0.03, 0.0, 0.03, 0.06)
DEFAULT_ALIGNMENT_WEIGHTS = {
    "buy": 1.0,
    "sell": -1.0,
    "hold": 0.0,
}


def ensure_scenario_shocks(shocks: Optional[Iterable[float]]) -> List[float]:
    """Validate and normalise the scenario shock list."""
    if shocks is None:
        return list(DEFAULT_SCENARIO_SHOCKS)

    cleaned: List[float] = []
    for value in shocks:
        try:
            num = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(num):
            cleaned.append(num)
    if not cleaned:
        cleaned = list(DEFAULT_SCENARIO_SHOCKS)
    if 0.0 not in cleaned:
        cleaned.append(0.0)
    cleaned = sorted(set(round(val, 4) for val in cleaned))
    return cleaned


def _safe_float(value: Any) -> Optional[float]:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    return num if math.isfinite(num) else None


def _confidence_key(latest: Mapping[str, Any]) -> float:
    confidence = _safe_float(latest.get("confidence_score"))
    if confidence is not None:
        return confidence
    prob_gap = _safe_float(latest.get("prob_gap"))
    return prob_gap if prob_gap is not None else 0.0


def build_daily_briefing(
    latest_map: Mapping[str, Mapping[str, Any]],
    reports: Mapping[str, pd.DataFrame],
    *,
    meta: Mapping[str, Any],
    top_n: int = 5,
    risk_limits: Optional[Mapping[str, Optional[float]]] = None,
) -> Dict[str, Any]:
    if not latest_map:
        return {
            "generated_at": dt.datetime.utcnow().isoformat(),
            "headline": "暂无信号",
            "top_signals": [],
            "watchlist": [],
            "market_snapshot": {},
        }

    decisions = {"buy": 0, "sell": 0, "hold": 0}
    sorted_entries: List[Tuple[str, Mapping[str, Any]]] = []
    for ticker, latest in latest_map.items():
        decision = str(latest.get("decision", "hold")).lower()
        decisions[decision] = decisions.get(decision, 0) + 1
        sorted_entries.append((ticker, latest))

    order_map = {"buy": 3, "hold": 2, "sell": 1}
    sorted_entries.sort(
        key=lambda item: (order_map.get(str(item[1].get("decision", "hold")).lower(), 0), _confidence_key(item[1])),
        reverse=True,
    )

    top_signals: List[Dict[str, Any]] = []
    for ticker, latest in sorted_entries[: max(1, top_n)]:
        entry = {
            "ticker": ticker,
            "decision": latest.get("decision"),
            "price": _safe_float(latest.get("price")),
            "confidence": _confidence_key(latest),
            "prob_gap": _safe_float(latest.get("prob_gap")),
            "action_hint": latest.get("action_hint"),
            "risk_flags": list(latest.get("risk_flags", [])),
            "key_drivers": latest.get("key_drivers", []),
            "signal_age_days": latest.get("signal_age_days"),
        }
        quality_prob = _safe_float(latest.get("meta_signal_prob"))
        if quality_prob is not None:
            entry["quality"] = quality_prob
        backtest = reports.get(ticker)
        if isinstance(backtest, pd.DataFrame) and not backtest.empty:
            recent_returns = backtest["return_5d"].dropna().tail(5)
            if not recent_returns.empty:
                entry["avg_return_5d"] = float(recent_returns.mean())
        top_signals.append(entry)

    watchlist: List[Dict[str, Any]] = []
    limit_drawdown = _safe_float((risk_limits or {}).get("max_drawdown"))
    limit_vol = _safe_float((risk_limits or {}).get("target_volatility"))

    for ticker, latest in sorted_entries:
        issues: List[str] = []
        risk_flags = latest.get("risk_flags", [])
        if "recent_flip" in risk_flags:
            issues.append("信号刚刚翻转")
        if "low_confidence" in risk_flags:
            issues.append("置信度偏低")
        if "near_threshold" in risk_flags:
            issues.append("逼近操作阈值")
        quality_prob = _safe_float(latest.get("meta_signal_prob"))
        if "low_quality" in risk_flags or (quality_prob is not None and quality_prob < 0.35):
            issues.append(f"信号质量较弱（约 {quality_prob:.0%}）")
        drawdown = _safe_float(latest.get("drawdown"))
        if limit_drawdown is not None and drawdown is not None and abs(drawdown) > limit_drawdown:
            issues.append(f"回撤 {drawdown:.2%} 超过容忍度")
        vol_20d = _safe_float(latest.get("volatility_20d"))
        if limit_vol is not None and vol_20d is not None and vol_20d > limit_vol:
            issues.append(f"20日波动 {vol_20d:.2%} 超过目标")
        if issues:
            watchlist.append(
                {
                    "ticker": ticker,
                    "decision": latest.get("decision"),
                    "issues": issues,
                    "confidence": _confidence_key(latest),
                    "risk_flags": list(risk_flags),
                }
            )

    headline = (
        f"{decisions.get('buy', 0)} 个买入 | {decisions.get('hold', 0)} 个观望 | "
        f"{decisions.get('sell', 0)} 个卖出"
    )

    return {
        "generated_at": dt.datetime.utcnow().isoformat(),
        "headline": headline,
        "top_signals": top_signals,
        "watchlist": watchlist,
        "market_snapshot": {
            "today": str(meta.get("today")),
            "horizon": meta.get("horizon"),
            "threshold": meta.get("threshold"),
            "lookback_years": meta.get("lookback_years"),
            "risk_profile": meta.get("risk_profile"),
        },
    }


def detect_event_alerts(
    latest_map: Mapping[str, Mapping[str, Any]],
    *,
    risk_limits: Optional[Mapping[str, Optional[float]]] = None,
) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    if not latest_map:
        return events

    limit_drawdown = _safe_float((risk_limits or {}).get("max_drawdown"))
    limit_vol = _safe_float((risk_limits or {}).get("target_volatility"))

    for ticker, latest in latest_map.items():
        return_1d = _safe_float(latest.get("return_1d"))
        return_3d = _safe_float(latest.get("return_3d"))
        drawdown = _safe_float(latest.get("drawdown"))
        vol_5d = _safe_float(latest.get("volatility_5d"))
        vol_20d = _safe_float(latest.get("volatility_20d"))
        prob_gap = _safe_float(latest.get("prob_gap"))
        risk_flags = list(latest.get("risk_flags", []))

        if return_1d is not None and abs(return_1d) >= 0.035:
            severity = "critical" if abs(return_1d) >= 0.06 else "warning"
            events.append(
                {
                    "ticker": ticker,
                    "type": "price_move",
                    "severity": severity,
                    "message": f"单日波动 {return_1d:.2%}",
                    "evidence": {"return_1d": return_1d},
                }
            )
        if return_3d is not None and abs(return_3d) >= 0.08:
            events.append(
                {
                    "ticker": ticker,
                    "type": "swing_move",
                    "severity": "warning",
                    "message": f"近3日累计波动 {return_3d:.2%}",
                    "evidence": {"return_3d": return_3d},
                }
            )
        if "recent_flip" in risk_flags:
            events.append(
                {
                    "ticker": ticker,
                    "type": "signal_flip",
                    "severity": "info",
                    "message": "模型信号刚刚翻转，请关注确认力度。",
                }
            )
        if prob_gap is not None and prob_gap < 0.03:
            events.append(
                {
                    "ticker": ticker,
                    "type": "low_confidence",
                    "severity": "info",
                    "message": "置信度较低，建议等待确认。",
                    "evidence": {"prob_gap": prob_gap},
                }
            )
        if limit_drawdown is not None and drawdown is not None and abs(drawdown) > limit_drawdown:
            events.append(
                {
                    "ticker": ticker,
                    "type": "drawdown_limit",
                    "severity": "critical",
                    "message": f"持仓回撤 {drawdown:.2%} 超出个性化阈值。",
                }
            )
        if (
            limit_vol is not None
            and vol_20d is not None
            and vol_20d > limit_vol
            and (vol_5d is not None and vol_5d > limit_vol)
        ):
            events.append(
                {
                    "ticker": ticker,
                    "type": "vol_spike",
                    "severity": "warning",
                    "message": f"短中期波动率 {vol_5d:.2%}/{vol_20d:.2%} 高于目标。",
                }
            )
        if "near_threshold" in risk_flags:
            events.append(
                {
                    "ticker": ticker,
                    "type": "threshold_watch",
                    "severity": "warning",
                    "message": "价格逼近模型阈值，留意触发条件。",
                }
            )
        quality_prob = _safe_float(latest.get("meta_signal_prob"))
        if "low_quality" in risk_flags or (quality_prob is not None and quality_prob < 0.35):
            quality_display = f"{quality_prob:.0%}" if quality_prob is not None else "--"
            events.append(
                {
                    "ticker": ticker,
                    "type": "low_quality",
                    "severity": "info",
                    "message": f"信号质量较弱（约 {quality_display}），建议等待确认。",
                    "evidence": {"quality_prob": quality_prob},
                }
            )

    return events


def build_scenario_matrix(
    latest_map: Mapping[str, Mapping[str, Any]],
    shocks: Sequence[float],
) -> Dict[str, Any]:
    matrix: List[Dict[str, Any]] = []
    if not latest_map:
        return {"shocks": list(shocks), "scenarios": matrix, "portfolio_expected_returns": []}

    portfolio_expected_returns: List[Dict[str, Any]] = []
    for shock in shocks:
        buy_pressure = 0
        sell_recovery = 0
        neutral_breakout = 0
        entries: List[Dict[str, Any]] = []
        alignment_sum = 0.0
        valid_alignment_count = 0
        for ticker, latest in latest_map.items():
            decision = str(latest.get("decision", "hold")).lower()
            threshold = _safe_float(latest.get("decision_threshold")) or 0.1
            price = _safe_float(latest.get("price"))
            shock_price = price * (1.0 + shock) if price is not None else None
            status = "steady"
            note = ""
            if decision == "buy":
                if shock <= -threshold:
                    status = "risk"
                    note = "跌幅或触发止损/减仓"
                    buy_pressure += 1
                elif shock <= -(threshold / 2):
                    status = "watch"
                    note = "跌幅逼近阈值，关注确认"
            elif decision == "sell":
                if shock >= threshold:
                    status = "opportunity"
                    note = "反弹可能确认止盈"
                    sell_recovery += 1
                elif shock >= (threshold / 2):
                    status = "watch"
                    note = "反弹接近阈值"
            else:
                if abs(shock) >= threshold:
                    status = "swing"
                    note = "波动可能触发买入/卖出信号"
                    neutral_breakout += 1

            alignment_weight = DEFAULT_ALIGNMENT_WEIGHTS.get(decision, 0.0)
            if alignment_weight != 0.0:
                alignment = alignment_weight * shock
                alignment_sum += alignment
                valid_alignment_count += 1
            elif decision == "hold":
                alignment_sum += -abs(shock) * 0.25
                valid_alignment_count += 1

            entries.append(
                {
                    "ticker": ticker,
                    "decision": decision,
                    "shock_price": shock_price,
                    "status": status,
                    "note": note,
                    "threshold": threshold,
                    "shock": shock,
                    "alignment_score": alignment_weight * shock,
                }
            )
        expected_return = alignment_sum / valid_alignment_count if valid_alignment_count else 0.0
        portfolio_expected_returns.append(
            {
                "shock": shock,
                "expected_alignment": expected_return,
                "buy_under_pressure": buy_pressure,
                "sell_recovery": sell_recovery,
                "neutral_breakout": neutral_breakout,
            }
        )
        matrix.append(
            {
                "shock": shock,
                "summary": {
                    "buy_under_pressure": buy_pressure,
                    "sell_recovery": sell_recovery,
                    "neutral_breakout": neutral_breakout,
                    "expected_alignment": expected_return,
                },
                "entries": entries,
            }
        )
    return {"shocks": list(shocks), "scenarios": matrix, "portfolio_expected_returns": portfolio_expected_returns}


def build_market_summary(
    latest_map: Mapping[str, Mapping[str, Any]],
    meta: Mapping[str, Any],
) -> Dict[str, Any]:
    market_lookup = meta.get("market_by_ticker", {})
    summary: Dict[str, Dict[str, Any]] = {}

    for ticker, latest in latest_map.items():
        market = market_lookup.get(ticker, market_lookup.get(ticker.upper(), "unknown")) or "unknown"
        entry = summary.setdefault(
            market,
            {
                "market": market,
                "total": 0,
                "buy": 0,
                "hold": 0,
                "sell": 0,
                "confidence_sum": 0.0,
                "low_confidence": 0,
                "near_threshold": 0,
                "recent_flip": 0,
                "low_quality": 0,
            },
        )
        decision = str(latest.get("decision", "hold")).lower()
        entry["total"] += 1
        if decision in ("buy", "hold", "sell"):
            entry[decision] += 1
        entry["confidence_sum"] += _confidence_key(latest)
        risk_flags = latest.get("risk_flags", [])
        if "low_confidence" in risk_flags:
            entry["low_confidence"] += 1
        if "near_threshold" in risk_flags:
            entry["near_threshold"] += 1
        if "recent_flip" in risk_flags:
            entry["recent_flip"] += 1
        quality_prob = _safe_float(latest.get("meta_signal_prob"))
        if "low_quality" in risk_flags or (quality_prob is not None and quality_prob < 0.35):
            entry["low_quality"] += 1

    results: List[Dict[str, Any]] = []
    for market, entry in summary.items():
        total = max(entry["total"], 1)
        avg_confidence = entry["confidence_sum"] / total
        results.append(
            {
                "market": market,
                "total": entry["total"],
                "buy": entry["buy"],
                "hold": entry["hold"],
                "sell": entry["sell"],
                "avg_confidence": round(avg_confidence, 3),
                "low_confidence": entry["low_confidence"],
                "near_threshold": entry["near_threshold"],
                "recent_flip": entry["recent_flip"],
                "low_quality": entry["low_quality"],
            }
        )

    results.sort(key=lambda item: (item["sell"], -item["avg_confidence"]), reverse=True)
    return {
        "items": results,
        "markets": [item["market"] for item in results],
    }


def build_performance_snapshot(
    latest_map: Mapping[str, Mapping[str, Any]],
    scenario_matrix: Dict[str, Any],
) -> Dict[str, Any]:
    decisions = {"buy": 0, "hold": 0, "sell": 0}
    total_confidence = 0.0
    total_quality = 0.0
    for latest in latest_map.values():
        decision = str(latest.get("decision", "hold")).lower()
        if decision in decisions:
            decisions[decision] += 1
        total_confidence += _confidence_key(latest)
        quality_prob = _safe_float(latest.get("meta_signal_prob"))
        total_quality += quality_prob if quality_prob is not None else 0.0
    total = max(len(latest_map), 1)
    avg_confidence = total_confidence / total
    avg_quality = total_quality / total

    best_scenario = None
    if scenario_matrix:
        scenarios = scenario_matrix.get("portfolio_expected_returns") or []
        if scenarios:
            best_scenario = max(scenarios, key=lambda item: item.get("expected_alignment", 0.0))

    return {
        "decisions": decisions,
        "average_confidence": round(avg_confidence, 3),
        "average_quality": round(avg_quality, 3),
        "best_scenario": best_scenario,
        "total": len(latest_map),
    }


def build_portfolio_health(
    latest_map: Mapping[str, Mapping[str, Any]],
    holdings: Sequence[str],
    *,
    risk_limits: Optional[Mapping[str, Optional[float]]] = None,
) -> Dict[str, Any]:
    if not holdings:
        return {"holdings": [], "missing": [], "summary": {}}

    normalized_holdings = [ticker.upper() for ticker in holdings]
    limit_drawdown = _safe_float((risk_limits or {}).get("max_drawdown"))
    limit_vol = _safe_float((risk_limits or {}).get("target_volatility"))

    items: List[Dict[str, Any]] = []
    missing: List[str] = []

    for ticker in normalized_holdings:
        latest = latest_map.get(ticker)
        if latest is None:
            latest = next((v for key, v in latest_map.items() if key.upper() == ticker), None)
        if latest is None:
            missing.append(ticker)
            continue

        decision = str(latest.get("decision", "hold")).lower()
        confidence = _confidence_key(latest)
        risk_flags = list(latest.get("risk_flags", []))
        drawdown = _safe_float(latest.get("drawdown"))
        vol_20d = _safe_float(latest.get("volatility_20d"))

        rating = "green"
        notes: List[str] = []
        if decision == "sell":
            rating = "red"
        elif decision == "hold":
            rating = "yellow"
        if "low_confidence" in risk_flags or "near_threshold" in risk_flags:
            rating = "yellow"
            notes.append("置信度或阈值提示观察")
        if limit_drawdown is not None and drawdown is not None and abs(drawdown) > limit_drawdown:
            rating = "red"
            notes.append(f"回撤 {drawdown:.2%} 超过容忍度")
        if limit_vol is not None and vol_20d is not None and vol_20d > limit_vol:
            rating = "yellow" if rating != "red" else rating
            notes.append(f"波动率 {vol_20d:.2%} 高于目标")

        items.append(
            {
                "ticker": ticker,
                "decision": decision,
                "rating": rating,
                "confidence": confidence,
                "price": _safe_float(latest.get("price")),
                "drawdown": drawdown,
                "volatility_20d": vol_20d,
                "risk_flags": risk_flags,
                "notes": notes,
            }
        )

    summary = {
        "green": sum(1 for item in items if item["rating"] == "green"),
        "yellow": sum(1 for item in items if item["rating"] == "yellow"),
        "red": sum(1 for item in items if item["rating"] == "red"),
        "covered": len(items),
        "missing": len(missing),
    }

    return {"holdings": items, "missing": missing, "summary": summary}


def build_suggested_prompts(
    latest_map: Mapping[str, Mapping[str, Any]],
    *,
    top_n: int = 5,
) -> List[str]:
    if not latest_map:
        return [
            "今天哪些标的最值得关注？",
            "如果市场突然大跌 3%，我的持仓应该如何调整？",
        ]

    sorted_entries = sorted(
        latest_map.items(),
        key=lambda item: _confidence_key(item[1]),
        reverse=True,
    )

    prompts: List[str] = []
    for ticker, latest in sorted_entries[: max(1, top_n)]:
        decision = str(latest.get("decision", "hold")).lower()
        confidence = _confidence_key(latest)
        if decision == "buy":
            prompts.append(f"{ticker} 今天适合加仓吗？置信度 {confidence:.0%}")
            prompts.append(f"如果 {ticker} 盘中回落 3%，是否需要减仓？")
        elif decision == "sell":
            prompts.append(f"{ticker} 是否到了分批止盈的时机？")
            prompts.append(f"{ticker} 若突然反弹 4%，策略需要调整吗？")
        else:
            prompts.append(f"{ticker} 目前为何建议观望？何时考虑介入？")
    prompts.append("今天整体的买入/卖出机会主要集中在哪些行业？")
    prompts.append("若出现超预期事件（财报/政策），策略如何快速调整？")

    # remove duplicates while preserving order
    seen: set[str] = set()
    unique_prompts: List[str] = []
    for prompt in prompts:
        if prompt in seen:
            continue
        seen.add(prompt)
        unique_prompts.append(prompt)
    return unique_prompts[: top_n * 2]
