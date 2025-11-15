import math
import numpy as np
import pandas as pd
import pytest

from src import data as data_module
from src.data import PriceData, build_dataset, build_feature_frame, label_signals
from src.model import FEATURE_COLUMNS, ModelArtifacts, train_model
from src.signals import generate_signal_report


@pytest.fixture
def synthetic_price_data():
    index = pd.date_range("2023-01-01", periods=260, freq="D")
    base = np.linspace(100, 150, len(index))
    frame = pd.DataFrame(
        {
            "Open": base * 0.99,
            "High": base * 1.01,
            "Low": base * 0.98,
            "Close": base,
            "Adj Close": base,
            "Volume": np.linspace(1_000_000, 1_200_000, len(index)),
        },
        index=index,
    )
    return PriceData(ticker="AAPL", prices=frame)


@pytest.fixture(autouse=True)
def mock_external_series(monkeypatch, synthetic_price_data):
    def _series_generator(symbol, start, end):
        idx = pd.date_range(start, end, freq="D")
        values = np.linspace(1.0, 1.0 + 0.001 * len(idx), len(idx))
        return pd.Series(values, index=idx)

    monkeypatch.setattr(data_module, "_load_benchmark_series", _series_generator)
    monkeypatch.setattr(data_module, "_load_macro_series", _series_generator)
    monkeypatch.setattr(
        data_module,
        "_load_security_metadata",
        lambda ticker: {
            "pe_ratio": 18.5,
            "forwardPE": 16.2,
            "priceToBook": 6.1,
            "beta": 1.1,
            "dividendYield": 0.012,
            "market_cap": 1.2e12,
        },
    )
    return None


def test_build_feature_frame_produces_core_columns(synthetic_price_data):
    horizon = 5
    frame = build_feature_frame(synthetic_price_data, horizon=horizon)
    required = {
        f"return_{horizon}d",
        "sma_10",
        "macro_vix_level",
        "macro_region_nasdaq_level",
        "fundamental_pe_ratio",
        "ema_200",
        "sector_tech_level",
    }
    assert required.issubset(frame.columns)
    feature_only = frame.drop(columns=[f"return_{horizon}d"])
    assert feature_only.isna().sum().sum() == 0


def test_build_dataset_labels_all_tickers(monkeypatch):
    index = pd.date_range("2024-01-01", periods=8, freq="D")

    def fake_fetch_price_history(symbol, lookback_years):
        base = np.linspace(100, 105, len(index)) + (0 if symbol == "AAA" else 5)
        prices = pd.DataFrame(
            {
                "Open": base,
                "High": base * 1.01,
                "Low": base * 0.99,
                "Close": base,
                "Adj Close": base,
                "Volume": 1_000_000,
            },
            index=index,
        )
        return PriceData(symbol, prices)

    def fake_feature_frame(price_data, horizon):
        values = np.linspace(0.05, -0.05, len(price_data.prices))
        df = pd.DataFrame(
            {
                f"return_{horizon}d": values,
                "Adj Close": price_data.prices["Adj Close"],
                "atr_14": 0.02,
                "Volume": 1_000_000,
            },
            index=price_data.prices.index,
        )
        return df

    monkeypatch.setattr(data_module, "fetch_price_history", fake_fetch_price_history)
    monkeypatch.setattr(data_module, "build_feature_frame", fake_feature_frame)
    dataset = build_dataset(
        ["AAA", "BBB"],
        lookback_years=1,
        horizon=2,
        threshold=0.02,
        adaptive_threshold=True,
    )
    assert set(dataset["ticker"].unique()) == {"AAA", "BBB"}
    assert dataset["label"].isin({"buy", "hold", "sell"}).all()


def test_train_model_returns_artifacts():
    rng = np.random.default_rng(42)
    rows = 160
    data = {col: rng.normal(size=rows).astype(float) for col in FEATURE_COLUMNS}
    data["label"] = rng.choice(["buy", "hold", "sell"], size=rows)
    data["date"] = pd.date_range("2023-01-01", periods=rows, freq="D")
    data["ticker"] = rng.choice(["AAA", "BBB"], size=rows)
    data["label_available"] = True
    data["meta_signal_confidence"] = rng.random(rows)
    data["meta_signal_active"] = rng.integers(0, 2, size=rows)
    dataset = pd.DataFrame(data)

    artifacts = train_model(dataset, market="global", model_type="hist_gb")
    assert isinstance(artifacts, ModelArtifacts)
    assert artifacts.model_type == "hist_gb"
    assert set(artifacts.bias_factors.keys()) >= {"buy", "hold", "sell"}
    assert math.isclose(sum(artifacts.class_priors.values()), 1.0, rel_tol=1e-6)


def test_generate_signal_report_reacts_to_quality_weighting():
    index = pd.date_range("2024-01-01", periods=2, freq="D")
    df = pd.DataFrame(
        {
            "prob_buy": [0.6, 0.25],
            "prob_hold": [0.3, 0.30],
            "prob_sell": [0.1, 0.45],
            "meta_signal_prob": [0.9, 0.05],
            "Adj Close": [100.0, 98.0],
            "Open": [99.5, 98.5],
            "High": [101.0, 99.0],
            "Low": [98.5, 97.0],
            "Volume": [1_000_000, 900_000],
            "sma_10": [102.0, 97.0],
            "sma_50": [100.0, 100.0],
            "rsi_14": [65.0, 32.0],
            "macd": [0.4, -0.4],
            "macd_signal": [0.1, -0.1],
            "bb_pct": [0.2, 0.85],
            "volume_zscore": [0.5, -0.6],
            "return_1d": [0.01, -0.02],
            "return_3d": [0.02, -0.03],
            "return_5d": [0.03, -0.04],
            "return_10d": [0.05, -0.05],
            "return_20d": [0.08, -0.06],
            "return_60d": [0.12, -0.08],
        },
        index=index,
    )
    report = generate_signal_report(df)
    first = report.iloc[0]
    second = report.iloc[1]

    assert first["decision"] == "buy"
    assert second["decision"] == "sell"
    assert first["prob_buy"] > second["prob_buy"]
    assert first["indicator_bias"] > 0
    assert second["indicator_bias"] < 0


def test_label_signals_quality_bucket():
    index = pd.date_range("2024-01-01", periods=10, freq="D")
    frame = pd.DataFrame(
        {
            "Adj Close": np.linspace(100, 110, len(index)),
            "Open": np.linspace(99, 109, len(index)),
            "High": np.linspace(101, 111, len(index)),
            "Low": np.linspace(98, 108, len(index)),
            "Volume": 1_000_000,
            "atr_14": 0.02,
        },
        index=index,
    )
    frame["return_3d"] = frame["Adj Close"].pct_change(3)
    labeled = label_signals(frame, horizon=3, threshold=0.02, adaptive_threshold=False, meta_quality_floor=0.3)
    assert "meta_signal_quality_bucket" in labeled.columns
    assert set(labeled["meta_signal_quality_bucket"].unique()) <= {"high", "medium", "low", None}
    assert (labeled["meta_signal_active"] <= 1).all()

