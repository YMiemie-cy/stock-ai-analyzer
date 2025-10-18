"""Report generation for buy/sell recommendations."""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Iterable, List, Tuple

import pandas as pd
from rich.console import Console
from rich.table import Table

console = Console()
REPORT_DIR = Path(__file__).resolve().parent.parent / "reports"
REPORT_DIR.mkdir(exist_ok=True)


def compute_confidence(row: pd.Series) -> float:
    prob_gap = row.get("prob_gap", 0.0)
    indicator_bias = row.get("indicator_bias", 0.0)
    prob_strength = max(prob_gap, 0.0)
    indicator_strength = min(max(abs(indicator_bias) / 2.0, 0.0), 1.0)
    return round(prob_strength * 0.7 + indicator_strength * 0.3, 2)


def format_price(value: float) -> str:
    if value is None or (isinstance(value, float) and (pd.isna(value))):
        return "--"
    return f"{value:.2f}"


def render_signal_summary(report_df: pd.DataFrame, ticker: str) -> None:
    if report_df.empty:
        console.print(f"[bold red]{ticker}[/bold red] No data available for summary.")
        return

    latest = report_df.iloc[-1]
    timestamp = getattr(latest, "name", None)
    timestamp_str = timestamp.strftime("%Y-%m-%d") if isinstance(timestamp, pd.Timestamp) else "N/A"
    console.print(f"\n[bold cyan]{ticker} Signal Summary[/bold cyan]")
    console.print(
        f"As of: [bold]{timestamp_str}[/bold] | Price: [bold]{format_price(latest.get('price'))}[/bold] | Decision: [bold]{latest['decision'].upper()}[/bold] | Score: {latest['score']:.2f} | "
        f"Probabilities -> Buy: {latest['prob_buy']:.2f}, Hold: {latest['prob_hold']:.2f}, Sell: {latest['prob_sell']:.2f} | "
        f"Prob Gap: {latest.get('prob_gap', 0.0):.2f} | Confidence: {compute_confidence(latest):.2f}"
    )
    console.print(
        f"Indicators -> SMA: {latest['indicator_sma']:+}, RSI: {latest['indicator_rsi']:+}, "
        f"MACD: {latest['indicator_macd']:+}, Bollinger: {latest['indicator_bollinger']:+}, "
        f"Volume: {latest.get('indicator_volume', 0):+} | Bias Avg: {latest.get('indicator_bias', 0.0):+.2f}"
    )


def render_historical_table(report_df: pd.DataFrame, limit: int = 10) -> None:
    if report_df.empty:
        console.print("No historical signal data to display.")
        return

    table = Table(title="Recent Signals", show_lines=True)
    table.add_column("Date", justify="left", overflow="fold")
    table.add_column("Decision", justify="center")
    table.add_column("Score", justify="right")
    table.add_column("Price", justify="right")
    table.add_column("Prob Buy", justify="right")
    table.add_column("Prob Hold", justify="right")
    table.add_column("Prob Sell", justify="right")
    table.add_column("Prob Gap", justify="right")
    table.add_column("Confidence", justify="right")
    table.add_column("Bias Avg", justify="right")

    recent = report_df.tail(limit)
    for idx, row in recent.iterrows():
        table.add_row(
            idx.strftime("%Y-%m-%d"),
            row["decision"],
            f"{row['score']:.2f}",
            format_price(row.get("price")),
            f"{row['prob_buy']:.2f}",
            f"{row['prob_hold']:.2f}",
            f"{row['prob_sell']:.2f}",
            f"{row.get('prob_gap', 0.0):.2f}",
            f"{compute_confidence(row):.2f}",
            f"{row.get('indicator_bias', 0.0):+.2f}",
        )

    console.print(table)


def export_signal_csv(report_df: pd.DataFrame, ticker: str) -> Path:
    """Persist signal history for轻量回测或日志审计."""
    path = REPORT_DIR / f"{ticker}_signals.csv"
    report_df.to_csv(path, index=True)
    return path


def export_markdown_summary(
    entries: List[Tuple[str, pd.DataFrame]],
    metadata: dict,
    limit: int = 5,
    filename: str | None = None,
) -> Path:
    """Generate a consolidated Markdown report."""
    if not entries:
        raise ValueError("No entries provided for markdown summary.")

    run_time = metadata.get("run_time", datetime.utcnow())
    summary_path = (
        REPORT_DIR / filename
        if filename
        else REPORT_DIR / f"summary_{run_time.strftime('%Y%m%d_%H%M%S')}.md"
    )

    lines: List[str] = []
    lines.append("# Stock AI Analyzer Report")
    lines.append("")
    lines.append(f"- Generated at: {run_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    lines.append(
        f"- Data coverage: {metadata.get('data_start')} → {metadata.get('data_end')} "
        f"(frequency: {metadata.get('frequency', 'daily')}, horizon: {metadata.get('horizon')} periods)"
    )
    lines.append(f"- Today: {metadata.get('today')}")
    if metadata.get("adaptive_threshold"):
        lines.append(
            f"- Adaptive threshold enabled | Base threshold: {metadata.get('threshold')} | "
            f"Range: {metadata.get('min_threshold')} - {metadata.get('max_threshold')}"
        )
    else:
        lines.append(f"- Fixed threshold: {metadata.get('threshold')}")
    lines.append("")

    for ticker, report_df in entries:
        if report_df.empty:
            lines.append(f"## {ticker}")
            lines.append("_No signal data available._")
            lines.append("")
            continue

        latest = report_df.iloc[-1]
        latest_ts = latest.name.strftime("%Y-%m-%d") if isinstance(latest.name, pd.Timestamp) else str(latest.name)
        lines.append(f"## {ticker}")
        lines.append(
            f"- Latest date: **{latest_ts}** | Price: **{format_price(latest.get('price'))}** | "
            f"Decision: **{latest['decision'].upper()}** | Score: {latest['score']:.2f}"
        )
        lines.append(
            f"- Probabilities: buy {latest['prob_buy']:.2f} / hold {latest['prob_hold']:.2f} / sell {latest['prob_sell']:.2f} "
            f"(gap {latest.get('prob_gap', 0.0):.2f}) | Confidence: {compute_confidence(latest):.2f}"
        )
        lines.append(
            f"- Indicator bias: SMA {latest['indicator_sma']:+}, RSI {latest['indicator_rsi']:+}, "
            f"MACD {latest['indicator_macd']:+}, Bollinger {latest['indicator_bollinger']:+}, "
            f"Volume {latest.get('indicator_volume', 0):+} | Mean {latest.get('indicator_bias', 0.0):+.2f}"
        )
        lines.append("")

        recent = report_df.tail(limit)
        lines.append(
            "| Date | Decision | Price | Score | Prob Buy | Prob Hold | Prob Sell | Prob Gap | Confidence | Bias Avg |"
        )
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for idx, row in recent.iterrows():
            date_str = idx.strftime("%Y-%m-%d") if isinstance(idx, pd.Timestamp) else str(idx)
            lines.append(
                f"| {date_str} | {row['decision']} | {format_price(row.get('price'))} | {row['score']:.2f} | "
                f"{row['prob_buy']:.2f} | {row['prob_hold']:.2f} | {row['prob_sell']:.2f} | "
                f"{row.get('prob_gap', 0.0):.2f} | {compute_confidence(row):.2f} | {row.get('indicator_bias', 0.0):+.2f} |"
            )
        lines.append("")

    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return summary_path
