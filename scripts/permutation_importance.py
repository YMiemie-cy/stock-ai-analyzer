"""Compute permutation feature importance for the trained stock model.

This utility loads the cached model artifacts, builds a dataset for the
specified tickers, and evaluates permutation importance on a sampled slice of
the feature matrix. Results are saved to `reports/permutation_importance.csv`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import sys

import pandas as pd
from rich.console import Console
from sklearn.inspection import permutation_importance

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data import build_dataset
from src.model import FEATURE_COLUMNS, LABEL, load_artifacts
from src.tickers import normalize_tickers

console = Console()
REPORT_PATH = Path(__file__).resolve().parent.parent / "reports" / "permutation_importance.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Permutation feature importance for stock model")
    parser.add_argument("--tickers", nargs="+", required=True, help="Tickers to include in the dataset")
    parser.add_argument("--lookback-years", type=int, default=5, help="History window for data download")
    parser.add_argument("--horizon", type=int, default=12, help="Labeling horizon in periods")
    parser.add_argument("--threshold", type=float, default=0.05, help="Base buy/sell threshold")
    parser.add_argument("--min-threshold", type=float, default=0.01, help="Adaptive threshold lower bound")
    parser.add_argument("--max-threshold", type=float, default=0.06, help="Adaptive threshold upper bound")
    parser.add_argument("--adaptive-threshold", action="store_true", help="Use ATR-adjusted thresholds when labeling")
    parser.add_argument("--resample-frequency", choices=["daily", "weekly"], default="weekly", help="Price aggregation frequency")
    parser.add_argument("--model-name", default="default_model", help="Which saved model artifacts to load")
    parser.add_argument("--sample-per-ticker", type=int, default=400, help="Limit number of rows per ticker for the analysis")
    parser.add_argument("--n-repeats", type=int, default=5, help="Permutation repeats")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


def prepare_dataset(
    tickers: Iterable[str],
    lookback_years: int,
    horizon: int,
    threshold: float,
    adaptive_threshold: bool,
    min_threshold: float,
    max_threshold: float,
    resample_frequency: str,
    sample_per_ticker: Optional[int],
) -> pd.DataFrame:
    normalized, _, invalid = normalize_tickers(tickers)
    if not normalized:
        raise ValueError(f"None of the provided tickers are valid: {invalid}")

    dataset = build_dataset(
        normalized,
        lookback_years=lookback_years,
        horizon=horizon,
        threshold=threshold,
        adaptive_threshold=adaptive_threshold,
        resample_frequency="W" if resample_frequency.lower() == "weekly" else "D",
        min_threshold=min_threshold,
        max_threshold=max_threshold,
    )
    if dataset.empty:
        raise ValueError("No data available for the requested configuration.")

    dataset = dataset.dropna(subset=[LABEL])

    if sample_per_ticker is not None and sample_per_ticker > 0:
        dataset = (
            dataset.sort_values(["ticker", "date"])
            .groupby("ticker", group_keys=False)
            .tail(sample_per_ticker)
        )

    return dataset


def compute_importance(
    dataset: pd.DataFrame,
    model_name: str,
    n_repeats: int,
    random_state: int,
) -> pd.DataFrame:
    artifacts = load_artifacts(model_name)
    X = dataset[FEATURE_COLUMNS]
    y = dataset[LABEL]
    X_scaled = artifacts.scaler.transform(X)

    result = permutation_importance(
        artifacts.model,
        X_scaled,
        y,
        scoring="accuracy",
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )

    importance_df = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    return importance_df


def main() -> None:
    args = parse_args()

    console.print(
        f"Building dataset for tickers: {', '.join(args.tickers)} | resample={args.resample_frequency} | adaptive={args.adaptive_threshold}"
    )
    dataset = prepare_dataset(
        tickers=args.tickers,
        lookback_years=args.lookback_years,
        horizon=args.horizon,
        threshold=args.threshold,
        adaptive_threshold=args.adaptive_threshold,
        min_threshold=args.min_threshold,
        max_threshold=args.max_threshold,
        resample_frequency=args.resample_frequency,
        sample_per_ticker=args.sample_per_ticker,
    )

    console.print(f"Dataset rows: {len(dataset)} | tickers: {dataset['ticker'].nunique()}")
    importance_df = compute_importance(
        dataset=dataset,
        model_name=args.model_name,
        n_repeats=args.n_repeats,
        random_state=args.random_state,
    )

    REPORT_PATH.parent.mkdir(exist_ok=True)
    importance_df.to_csv(REPORT_PATH, index=False)
    console.print(f"[green]Permutation importance saved to {REPORT_PATH}[/green]")
    console.print(importance_df.head(20))


if __name__ == "__main__":
    main()
