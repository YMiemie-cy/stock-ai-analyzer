import datetime as dt

import pandas as pd

from src.insights import (
    build_daily_briefing,
    build_portfolio_health,
    build_scenario_matrix,
    detect_event_alerts,
    ensure_scenario_shocks,
    build_market_summary,
    build_performance_snapshot,
)


def test_ensure_scenario_shocks_includes_zero_sorted():
    shocks = ensure_scenario_shocks([0.04, -0.02, 0.04])
    assert shocks[0] == -0.02
    assert shocks[-1] == 0.04
    assert 0.0 in shocks


def test_portfolio_health_ratings_with_risk_limits():
    latest_map = {
        "AAPL": {
            "decision": "buy",
            "confidence_score": 0.62,
            "drawdown": -0.05,
            "volatility_20d": 0.18,
            "risk_flags": [],
            "price": 120.0,
        },
        "TSLA": {
            "decision": "sell",
            "confidence_score": 0.58,
            "drawdown": -0.14,
            "volatility_20d": 0.25,
            "risk_flags": ["low_confidence"],
            "price": 220.0,
        },
    }
    holdings = ["AAPL", "TSLA", "MSFT"]
    health = build_portfolio_health(
        latest_map,
        holdings,
        risk_limits={"max_drawdown": 0.1, "target_volatility": 0.2},
    )
    assert health["summary"]["covered"] == 2
    assert "MSFT" in health["missing"]
    ratings = {item["ticker"]: item["rating"] for item in health["holdings"]}
    assert ratings["TSLA"] == "red"
    assert ratings["AAPL"] == "green"


def test_daily_briefing_headline_and_watchlist():
    index = pd.date_range(dt.date(2024, 4, 1), periods=3, freq="D")
    reports = {
        "AAPL": pd.DataFrame({"return_5d": [0.02, 0.03, 0.04]}, index=index),
    }
    latest_map = {
        "AAPL": {
            "decision": "buy",
            "confidence_score": 0.7,
            "prob_gap": 0.12,
            "action_hint": "建议加仓",
            "risk_flags": ["recent_flip"],
        }
    }
    meta = {
        "today": dt.date(2024, 4, 3),
        "horizon": 5,
        "threshold": 0.02,
        "lookback_years": 5,
        "risk_profile": "balanced",
    }
    briefing = build_daily_briefing(latest_map, reports, meta=meta, top_n=3)
    assert "买入" in briefing["headline"]
    assert briefing["top_signals"][0]["ticker"] == "AAPL"
    assert any(item["ticker"] == "AAPL" for item in briefing["watchlist"])


def test_detect_event_alerts_flags_large_move():
    latest_map = {
        "AAPL": {
            "return_1d": 0.08,
            "return_3d": 0.1,
            "prob_gap": 0.02,
            "risk_flags": ["near_threshold"],
            "drawdown": -0.12,
            "volatility_5d": 0.15,
            "volatility_20d": 0.2,
        }
    }
    alerts = detect_event_alerts(latest_map, risk_limits={"max_drawdown": 0.1, "target_volatility": 0.18})
    alert_types = {alert["type"] for alert in alerts}
    assert "price_move" in alert_types
    assert "threshold_watch" in alert_types
    assert "drawdown_limit" in alert_types


def test_scenario_matrix_counts_pressure():
    latest_map = {
        "AAPL": {
            "decision": "buy",
            "decision_threshold": 0.05,
            "price": 100.0,
        },
        "TSLA": {
            "decision": "sell",
            "decision_threshold": 0.04,
            "price": 200.0,
        },
        "MSFT": {
            "decision": "hold",
            "decision_threshold": 0.03,
            "price": 150.0,
        },
    }
    matrix = build_scenario_matrix(latest_map, shocks=[-0.06, 0.05])
    first_summary = matrix["scenarios"][0]["summary"]
    assert first_summary["buy_under_pressure"] == 1
    assert matrix["scenarios"][1]["summary"]["sell_recovery"] == 1
    assert matrix["portfolio_expected_returns"], "Expected portfolio expected returns to be populated"
    assert "alignment_score" in matrix["scenarios"][0]["entries"][0]


def test_market_summary_and_performance_snapshot():
    meta = {"market_by_ticker": {"AAPL": "usa", "TSLA": "usa", "600519.SS": "china"}}
    latest_map = {
        "AAPL": {"decision": "buy", "prob_gap": 0.2, "risk_flags": ["low_confidence"], "meta_signal_prob": 0.2},
        "TSLA": {"decision": "sell", "prob_gap": 0.3, "risk_flags": ["recent_flip"]},
        "600519.SS": {"decision": "hold", "prob_gap": 0.05, "risk_flags": ["near_threshold"]},
    }
    market_summary = build_market_summary(latest_map, meta)
    assert len(market_summary["items"]) == 2
    usa_summary = next(item for item in market_summary["items"] if item["market"] == "usa")
    assert usa_summary["buy"] == 1 and usa_summary["sell"] == 1
    assert usa_summary["low_quality"] >= 1

    scenario_matrix = build_scenario_matrix(latest_map, shocks=[-0.03, 0.04])
    snapshot = build_performance_snapshot(latest_map, scenario_matrix)
    assert snapshot["decisions"]["buy"] == 1
    assert snapshot["total"] == 3
    assert snapshot["best_scenario"]
    assert snapshot["average_quality"] >= 0


def test_meta_label_generation():
    from src.data import label_signals

    index = pd.date_range("2024-01-01", periods=6, freq="D")
    frame = pd.DataFrame(
        {
            "Adj Close": [100, 101, 102, 104, 107, 111],
            "atr_14": [1.2, 1.3, 1.25, 1.4, 1.5, 1.55],
        },
        index=index,
    )
    frame["return_3d"] = frame["Adj Close"].pct_change(3)
    labeled = label_signals(frame, horizon=3, threshold=0.02, adaptive_threshold=True, min_threshold=0.01, max_threshold=0.06)
    assert "meta_signal_active" in labeled.columns
    assert labeled["meta_signal_confidence"].max() >= 0.0
    assert labeled["meta_signal_active"].dtype in (int, "int64", "int32", "int")
