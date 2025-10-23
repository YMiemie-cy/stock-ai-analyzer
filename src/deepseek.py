"""Utilities for fusing DeepSeek LLM guidance with local model probabilities."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import httpx
import numpy as np
import pandas as pd


DEEPSEEK_ENDPOINT = "/v1/chat/completions"
DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(num):
        return default
    return num


def _extract_json_blob(text: str) -> Dict[str, Any]:
    """Attempt to parse JSON from model response body."""
    text = text.strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        if isinstance(parsed, Mapping):
            return dict(parsed)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return {}
    snippet = match.group(0)
    try:
        parsed = json.loads(snippet)
        if isinstance(parsed, Mapping):
            return dict(parsed)
    except json.JSONDecodeError:
        return {}
    return {}


@dataclass
class DeepseekConfig:
    api_key: str
    model: str = DEFAULT_DEEPSEEK_MODEL
    base_url: str = "https://api.deepseek.com"
    temperature: float = 0.2
    top_p: float = 0.85
    timeout: float = 25.0
    weight: float = 0.35
    latest_only: bool = True
    max_rows: int = 60


@dataclass
class DeepseekDecision:
    label: str
    confidence: float
    reason: str = ""
    raw: Dict[str, Any] | None = None


class DeepseekClient:
    """Lightweight wrapper around DeepSeek chat completion API."""

    def __init__(
        self,
        config: DeepseekConfig,
        *,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.config = config
        base = config.base_url.rstrip("/")
        self._endpoint = f"{base}{DEEPSEEK_ENDPOINT}"
        headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        }
        if http_client is None:
            self._client = httpx.Client(timeout=config.timeout, headers=headers)
            self._owns_client = True
        else:
            self._client = http_client
            self._owns_client = False

    def close(self) -> None:
        """Close the internal HTTP client if we created it."""
        if self._owns_client and isinstance(self._client, httpx.Client):
            try:
                self._client.close()
            except Exception:
                pass

    def classify_row(
        self,
        row: Mapping[str, Any],
        *,
        meta: Optional[Mapping[str, Any]] = None,
    ) -> DeepseekDecision | None:
        payload = self._build_payload(row, meta=meta)
        try:
            response = self._client.post(self._endpoint, json=payload)
        except Exception:
            return None
        if response.status_code >= 500:
            return None
        if response.status_code == 401:
            # Invalid credential, bail out permanently for this run.
            raise PermissionError("DeepSeek API returned 401 - invalid API key.")
        if response.status_code >= 400:
            return None
        try:
            data = response.json()
        except ValueError:
            return None
        choices: Sequence[Dict[str, Any]] = data.get("choices") or []
        first = choices[0] if choices else {}
        message = first.get("message") or {}
        content = message.get("content", "")
        parsed = _extract_json_blob(content)
        label = str(parsed.get("label", "")).strip().lower()
        if label not in {"buy", "hold", "sell"}:
            label = ""
        confidence = _safe_float(parsed.get("confidence"), default=0.0)
        confidence = max(0.0, min(1.0, confidence))
        reason = str(parsed.get("reason", "")).strip()
        if not label:
            return None
        return DeepseekDecision(label=label, confidence=confidence, reason=reason, raw=parsed)

    def _build_payload(
        self,
        row: Mapping[str, Any],
        *,
        meta: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        instruction = (
            "你是一位量化分析师，需要根据给定的特征数据判断当前时点的操作建议，"
            "仅在 \"buy\"、\"hold\"、\"sell\" 之间选择，并提供 0~1 的置信度和一句理由。"
            "请严格输出 JSON 对象，例如："
            '{"label": "buy", "confidence": 0.72, "reason": "动量与量能同步走强"}'
        )
        meta_text = ""
        if meta:
            horizon = meta.get("horizon")
            frequency = meta.get("resample_frequency")
            threshold = meta.get("threshold")
            parts = []
            if horizon is not None:
                parts.append(f"预测周期: {horizon}")
            if frequency:
                parts.append(f"频率: {frequency}")
            if threshold is not None:
                parts.append(f"收益阈值: {threshold}")
            if parts:
                meta_text = "；".join(parts)

        feature_lines = []
        focus_keys = [
            "prob_buy",
            "prob_hold",
            "prob_sell",
            "prob_gap",
            "indicator_bias",
            "return_1d",
            "return_5d",
            "return_20d",
            "return_60d",
            "trend_strength_20d",
            "trend_strength_60d",
            "volatility_20d",
            "volatility_60d",
            "rsi_14",
            "macd",
            "macd_signal",
            "bb_pct",
            "volume_zscore",
            "atr_ratio",
            "drawdown",
            "price",
        ]
        for key in focus_keys:
            if key not in row:
                continue
            value = row.get(key)
            if isinstance(value, (int, float)) and math.isfinite(value):
                feature_lines.append(f"{key}: {value:.4f}")
            elif value is not None:
                feature_lines.append(f"{key}: {value}")

        fallback_features = {
            k: v
            for k, v in row.items()
            if k in {"ticker", "date", "market", "model_signal"}
        }
        focus_block = "\n".join(feature_lines)
        fallback_block = json.dumps(fallback_features, ensure_ascii=False)

        user_prompt = (
            f"{meta_text}\n核心特征如下：\n{focus_block}\n"
            f"上下文：{fallback_block}\n"
            "请输出 JSON，字段包括 label、confidence、reason。"
        ).strip()

        payload = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "messages": [
                {"role": "system", "content": instruction},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
        }
        return payload


def _label_to_distribution(label: str, confidence: float) -> Dict[str, float]:
    label = (label or "").lower()
    if label not in {"buy", "hold", "sell"}:
        return {"buy": 1 / 3, "hold": 1 / 3, "sell": 1 / 3}
    confidence = max(0.0, min(1.0, confidence))
    residual = max(0.0, 1.0 - confidence)
    rest = residual / 2.0
    distribution = {"buy": rest, "hold": rest, "sell": rest}
    distribution[label] = confidence
    return distribution


def apply_deepseek_fusion(
    predictions: pd.DataFrame,
    *,
    client: DeepseekClient,
    config: DeepseekConfig,
    meta: Optional[Mapping[str, Any]] = None,
) -> pd.DataFrame:
    if predictions is None or predictions.empty:
        return predictions

    df = predictions.copy()
    df["prob_buy_local"] = df["prob_buy"]
    df["prob_hold_local"] = df["prob_hold"]
    df["prob_sell_local"] = df["prob_sell"]

    if config.latest_only:
        row_iter: List[Tuple[pd.Timestamp, pd.Series]] = [
            (group.index[-1], group.iloc[-1]) for _, group in df.groupby("ticker", sort=False)
        ]
    else:
        row_iter = list(df.iterrows())

    if config.max_rows and len(row_iter) > config.max_rows:
        row_iter = row_iter[-config.max_rows :]

    applied_any = False
    for idx, row in row_iter:
        decision = client.classify_row(row.to_dict(), meta=meta)
        if decision is None:
            continue
        applied_any = True
        dist = _label_to_distribution(decision.label, decision.confidence)
        weight = max(0.0, min(1.0, config.weight))
        for label_key, prob_key in [("buy", "prob_buy"), ("hold", "prob_hold"), ("sell", "prob_sell")]:
            local_prob = _safe_float(df.at[idx, prob_key], 0.0)
            fused = (1.0 - weight) * local_prob + weight * dist[label_key]
            df.at[idx, prob_key] = fused
            df.at[idx, f"{prob_key}_deepseek"] = dist[label_key]
        df.at[idx, "deepseek_label"] = decision.label
        df.at[idx, "deepseek_confidence"] = decision.confidence
        df.at[idx, "deepseek_reason"] = decision.reason

    if applied_any:
        df["prob_buy"] = df["prob_buy"].clip(0.0, 1.0)
        df["prob_hold"] = df["prob_hold"].clip(0.0, 1.0)
        df["prob_sell"] = df["prob_sell"].clip(0.0, 1.0)
        total = df["prob_buy"] + df["prob_hold"] + df["prob_sell"]
        total = total.replace(0, 1.0)
        df["prob_buy"] = df["prob_buy"] / total
        df["prob_hold"] = df["prob_hold"] / total
        df["prob_sell"] = df["prob_sell"] / total
        matrix = df[["prob_buy", "prob_hold", "prob_sell"]].to_numpy(copy=True)
        sorted_probs = np.sort(matrix, axis=1)
        df["prob_gap"] = sorted_probs[:, -1] - sorted_probs[:, -2]

    return df
