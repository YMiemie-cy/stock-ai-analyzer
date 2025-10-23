import pandas as pd
import pytest

from src.deepseek import DeepseekConfig, DeepseekDecision, apply_deepseek_fusion


class StubClient:
    def __init__(self, decision: DeepseekDecision):
        self.decision = decision
        self.calls = []

    def classify_row(self, row, *, meta=None):
        self.calls.append((row, meta))
        return self.decision


def test_apply_deepseek_fusion_latest_only():
    index = pd.date_range("2024-01-01", periods=2, freq="D")
    df = pd.DataFrame(
        {
            "ticker": ["AAPL", "AAPL"],
            "prob_buy": [0.6, 0.55],
            "prob_hold": [0.25, 0.3],
            "prob_sell": [0.15, 0.15],
            "prob_gap": [0.35, 0.25],
            "indicator_bias": [0.1, 0.2],
            "return_1d": [0.01, 0.012],
        },
        index=index,
    )
    config = DeepseekConfig(api_key="dummy", weight=0.5, latest_only=True)
    decision = DeepseekDecision(label="buy", confidence=0.8, reason="Momentum improving")
    client = StubClient(decision)

    fused = apply_deepseek_fusion(df, client=client, config=config, meta={"horizon": 4})

    # First row untouched
    assert fused.loc[index[0], "prob_buy"] == fused.loc[index[0], "prob_buy_local"]
    # Latest row adjusted towards DeepSeek suggestion
    assert fused.loc[index[1], "prob_buy"] > fused.loc[index[1], "prob_buy_local"]
    assert fused.loc[index[1], "prob_buy_deepseek"] == pytest.approx(0.8)
    assert fused.loc[index[1], "deepseek_label"] == "buy"
    assert fused.loc[index[1], "deepseek_confidence"] == pytest.approx(0.8)
    assert abs(fused.loc[index[1], "prob_buy"] + fused.loc[index[1], "prob_hold"] + fused.loc[index[1], "prob_sell"] - 1.0) < 1e-6


def test_apply_deepseek_fusion_respects_max_rows():
    index = pd.date_range("2024-01-01", periods=3, freq="D")
    df = pd.DataFrame(
        {
            "ticker": ["MSFT"] * 3,
            "prob_buy": [0.4, 0.45, 0.5],
            "prob_hold": [0.4, 0.35, 0.3],
            "prob_sell": [0.2, 0.2, 0.2],
            "prob_gap": [0.2, 0.1, 0.2],
            "indicator_bias": [0.0, 0.1, 0.2],
            "return_1d": [0.008, 0.009, 0.01],
        },
        index=index,
    )
    config = DeepseekConfig(api_key="dummy", weight=0.4, latest_only=False, max_rows=1)
    decision = DeepseekDecision(label="sell", confidence=0.7, reason="Weakness detected")
    client = StubClient(decision)

    fused = apply_deepseek_fusion(df, client=client, config=config, meta={"horizon": 5})

    # Only the most recent row should be updated when max_rows=1
    assert len(client.calls) == 1
    assert fused.loc[index[-1], "deepseek_label"] == "sell"
    assert fused.loc[index[-2], "prob_sell"] == fused.loc[index[-2], "prob_sell_local"]
