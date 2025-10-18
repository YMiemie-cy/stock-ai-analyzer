import pandas as pd
from types import SimpleNamespace

import app_server
from app_server import _ANALYSIS_CACHE


def test_daily_model_selection(monkeypatch):
    _ANALYSIS_CACHE.clear()

    captured = {}

    def fake_run_analysis(**kwargs):
        captured["model_name"] = kwargs.get("model_name")
        captured["resample_frequency"] = kwargs.get("resample_frequency")
        frequency = kwargs.get("resample_frequency")
        model = kwargs.get("model_name")
        return {
            "meta": {
                "resample_frequency": frequency,
                "selected_frequencies": [frequency],
                "model_name": model,
            },
            "reports": {},
            "latest": {},
            "backtests": {},
        }

    monkeypatch.setattr(app_server, "run_analysis", fake_run_analysis)

    result = app_server.get_or_run_analysis(
        tickers=["AAPL"],
        lookback_years=1,
        horizon=4,
        threshold=0.05,
        adaptive_threshold=True,
        min_threshold=0.01,
        max_threshold=0.06,
        resample_frequency="daily",
        model_name="default_model",
        model_type="auto",
        train=False,
    )

    assert captured["model_name"] == "default_model_daily"
    assert result["meta"]["model_name"] == "default_model_daily"


def test_run_analysis_sets_risk_flags(monkeypatch):
    import src.core as core

    def fake_normalize_tickers(tickers):
        return ["AAPL"], {"AAPL": "AAPL"}, []

    monkeypatch.setattr(core, "normalize_tickers", fake_normalize_tickers)
    monkeypatch.setattr(core, "market_for_ticker", lambda _: "global")

    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    data = pd.DataFrame(
        {
            "prob_buy": [0.62, 0.58, 0.50],
            "prob_hold": [0.28, 0.30, 0.10],
            "prob_sell": [0.10, 0.12, 0.40],
            "Adj Close": [100.0, 101.0, 102.0],
            "return_1d": [0.01, 0.012, 0.008],
            "return_3d": [0.02, 0.021, 0.019],
            "return_5d": [0.03, 0.031, 0.029],
            "return_10d": [0.04, 0.039, 0.037],
            "return_20d": [0.05, 0.048, 0.045],
            "return_60d": [0.06, 0.058, 0.056],
            "volatility_5d": [0.008, 0.009, 0.010],
            "volatility_10d": [0.010, 0.011, 0.012],
            "volatility_20d": [0.013, 0.014, 0.015],
            "volatility_60d": [0.018, 0.019, 0.020],
            "trend_strength_20d": [0.6, 0.55, 0.52],
            "trend_strength_60d": [0.4, 0.35, 0.32],
            "drawdown": [-0.05, -0.04, -0.03],
            "indicator_bias": [0.0, 0.0, 0.0],
            "sma_10": [100, 100.5, 101],
            "sma_50": [99, 99.5, 100],
            "rsi_14": [55, 57, 59],
            "macd": [0.12, 0.09, 0.07],
            "macd_signal": [0.11, 0.08, 0.06],
            "bb_pct": [0.5, 0.55, 0.6],
            "volume_zscore": [0.0, 0.0, 0.0],
            "ticker": ["AAPL", "AAPL", "AAPL"],
            "label": ["hold", "buy", "hold"],
        },
        index=dates,
    )
    data["date"] = dates

    def fake_build_dataset(*args, **kwargs):
        return data.copy()

    monkeypatch.setattr(core, "build_dataset", fake_build_dataset)
    monkeypatch.setattr(core, "ensure_model", lambda **kwargs: SimpleNamespace(model_type="mock"))
    monkeypatch.setattr(core, "predict_signals", lambda dataset, artifacts: dataset)
    monkeypatch.setattr(core, "naive_signal_backtest", lambda report_df, horizon: None)

    result = core.run_analysis(
        tickers=["AAPL"],
        lookback_years=1,
        horizon=4,
        threshold=0.05,
        adaptive_threshold=True,
        min_threshold=0.01,
        max_threshold=0.06,
        resample_frequency="daily",
        model_name="default_model_daily",
        model_type="auto",
        train=False,
    )

    latest = result["latest"]["AAPL"]
    assert latest["threshold_distance"] <= 0.02
    assert "near_threshold" in latest["risk_flags"]
    assert "low_confidence" in latest["risk_flags"]
