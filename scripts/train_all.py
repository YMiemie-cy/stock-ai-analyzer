"""Batch training helper to refresh US and China models with wider coverage."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import sys

import numpy as np
from rich.console import Console
from sklearn.metrics import classification_report

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.core import run_analysis
from src.model import load_artifacts

console = Console()
REPORT_PATH = Path(__file__).resolve().parent.parent / "reports" / "model_metrics_latest.json"


TRAINING_PLANS: List[Dict[str, object]] = [
    {
        "label": "us_default_weekly",
        "tickers": [
            "AAPL",
            "MSFT",
            "NVDA",
            "VTI",
            "QQQ",
            "SPY",
            "TSLA",
            "AMD",
            "META",
            "GOOGL",
            "AMZN",
            "AVGO",
            "IWM",
            "XLF",
            "XLK",
            "TSLL",
            "ROKU",
            "BB",
        ],
        "params": {
            "lookback_years": 12,
            "horizon": 12,
            "threshold": 0.055,
            "min_threshold": 0.02,
            "max_threshold": 0.09,
            "adaptive_threshold": True,
            "resample_frequency": "weekly",
            "model_name": "default_model",
            "model_type": "auto",
        },
    },
    {
        "label": "us_diversified_weekly",
        "tickers": [
            "SPY",
            "VTI",
            "QQQ",
            "IWD",
            "DVY",
            "XLV",
            "XLE",
            "XLU",
            "XLI",
            "GLD",
        ],
        "params": {
            "lookback_years": 12,
            "horizon": 12,
            "threshold": 0.05,
            "min_threshold": 0.018,
            "max_threshold": 0.08,
            "adaptive_threshold": True,
            "resample_frequency": "weekly",
            "model_name": "default_model_diversified",
            "model_type": "auto",
        },
    },
    {
        "label": "china_default_weekly",
        "tickers": [
            "600519.SS",
            "601318.SS",
            "600036.SS",
            "000001.SZ",
            "600276.SS",
            "300750.SZ",
            "002415.SZ",
            "601012.SS",
            "600887.SS",
            "000858.SZ",
            "600050.SS",
        ],
        "params": {
            "lookback_years": 9,
            "horizon": 12,
            "threshold": 0.06,
            "min_threshold": 0.02,
            "max_threshold": 0.10,
            "adaptive_threshold": True,
            "resample_frequency": "weekly",
            "model_name": "default_model",
            "model_type": "auto",
        },
    },
    {
        "label": "china_growth_weekly",
        "tickers": [
            "300750.SZ",
            "002475.SZ",
            "002594.SZ",
            "688981.SS",
            "600438.SS",
            "603259.SS",
            "300014.SZ",
            "300015.SZ",
            "002371.SZ",
            "002230.SZ",
        ],
        "params": {
            "lookback_years": 9,
            "horizon": 12,
            "threshold": 0.065,
            "min_threshold": 0.025,
            "max_threshold": 0.11,
            "adaptive_threshold": True,
            "resample_frequency": "weekly",
            "model_name": "default_model_china_growth",
            "model_type": "auto",
        },
    },
]


def _class_report(dataset, predictions) -> Dict[str, float]:
    if dataset is None or dataset.empty or predictions is None or predictions.empty:
        return {}
    report = classification_report(
        dataset["label"],
        predictions["model_signal"],
        zero_division=0,
        output_dict=True,
    )
    return {
        "accuracy": float(report.get("accuracy", 0.0)),
        "macro_precision": float(report.get("macro avg", {}).get("precision", 0.0)),
        "macro_recall": float(report.get("macro avg", {}).get("recall", 0.0)),
        "macro_f1": float(report.get("macro avg", {}).get("f1-score", 0.0)),
    }


def main() -> None:
    results: List[Dict[str, object]] = []
    for plan in TRAINING_PLANS:
        label = plan["label"]
        tickers = plan["tickers"]
        params = dict(plan.get("params", {}))
        model_type = params.pop("model_type", "auto")
        params_for_entry = dict(plan.get("params", {}))

        console.rule(f"[bold cyan]Training {label}")
        analysis = run_analysis(
            tickers=tickers,
            train=True,
            model_type=model_type,
            **params,
        )

        dataset = analysis.get("dataset")
        predictions = analysis.get("predictions")
        class_metrics = _class_report(dataset, predictions)
        meta = analysis["meta"]
        artifact_metrics: Dict[str, Dict[str, object]] = {}
        for market, model_name in meta.get("model_name_by_market", {}).items():
            artifact = load_artifacts(model_name)
            artifact_metrics[market] = {
                "model_type": getattr(artifact, "model_type", "hist_gb"),
                "cv_metrics": artifact.cv_metrics or {},
            }

        backtests = analysis.get("backtests", {})
        per_ticker_backtests: List[Dict[str, float]] = []
        for ticker, bt in backtests.items():
            if not bt:
                continue
            per_ticker_backtests.append(
                {
                    "ticker": ticker,
                    "horizon": bt.get("horizon"),
                    "average_signal_return": bt.get("average_signal_return"),
                    "hit_ratio": bt.get("hit_ratio"),
                    "cumulative_return": bt.get("cumulative_return"),
                    "trades": bt.get("trades"),
                }
            )

        backtest_summary = {}
        if per_ticker_backtests:
            hit_ratios = [bt["hit_ratio"] for bt in per_ticker_backtests if bt.get("hit_ratio") is not None]
            avg_returns = [
                bt["average_signal_return"] for bt in per_ticker_backtests if bt.get("average_signal_return") is not None
            ]
            cumulative_returns = [
                bt["cumulative_return"] for bt in per_ticker_backtests if bt.get("cumulative_return") is not None
            ]
            trades = [bt["trades"] for bt in per_ticker_backtests if bt.get("trades") is not None]
            if hit_ratios:
                backtest_summary["hit_ratio_mean"] = float(np.mean(hit_ratios))
                backtest_summary["hit_ratio_median"] = float(np.median(hit_ratios))
            if avg_returns:
                backtest_summary["average_signal_return_mean"] = float(np.mean(avg_returns))
            if cumulative_returns:
                backtest_summary["cumulative_return_mean"] = float(np.mean(cumulative_returns))
            if trades:
                backtest_summary["total_trades"] = int(np.sum(trades))
                backtest_summary["avg_trades_per_ticker"] = float(np.mean(trades))

        entry = {
            "label": label,
            "tickers": tickers,
            "params": params_for_entry,
            "training_time": datetime.utcnow().isoformat(),
            "available_tickers": meta.get("available_tickers", []),
            "missing_tickers": meta.get("missing_tickers", []),
            "model_type_by_market": meta.get("model_type_by_market", {}),
            "classification_metrics": class_metrics,
            "oof_metrics": artifact_metrics,
            "backtests": {
                "per_ticker": per_ticker_backtests,
                "summary": backtest_summary,
            },
        }
        results.append(entry)

        console.print(
            f"[green]Finished {label}[/green] "
            f"| accuracy: {class_metrics.get('accuracy', 0.0):.3f} "
            f"| macro_f1: {class_metrics.get('macro_f1', 0.0):.3f} "
            f"| backtest hit⌀: {backtest_summary.get('hit_ratio_mean', float('nan')):.3f}"
        )

    timestamp = datetime.utcnow().isoformat()
    payload = {"generated_at": timestamp, "runs": results}
    REPORT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    history_path = REPORT_PATH.with_name("model_metrics_history.jsonl")
    with history_path.open("a", encoding="utf-8") as history_file:
        for run in results:
            history_record = {"generated_at": timestamp, **run}
            history_file.write(json.dumps(history_record, ensure_ascii=False) + "\n")
    console.print(f"[bold green]Saved metrics to {REPORT_PATH}[/bold green]")


if __name__ == "__main__":
    main()
