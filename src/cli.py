"""Command-line interface for the stock AI analyzer prototype."""

from __future__ import annotations

import argparse
import datetime as dt

from rich.console import Console
from rich.table import Table
from sklearn.metrics import classification_report, confusion_matrix

from .core import run_analysis
from .report import export_markdown_summary, export_signal_csv, render_historical_table, render_signal_summary

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI-assisted buy/sell signal analyzer")
    parser.add_argument("--tickers", nargs="+", required=True, help="Tickers to analyze")
    parser.add_argument("--lookback-years", type=int, default=5, help="Years of history to download")
    parser.add_argument("--horizon", type=int, default=5, help="Days ahead for return labeling")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.02,
        help="Return threshold for buy/sell labeling (e.g. 0.02 = 2%)",
    )
    parser.add_argument(
        "--min-threshold",
        type=float,
        default=0.01,
        help="Adaptive 阈值的下限（默认 1%）",
    )
    parser.add_argument(
        "--max-threshold",
        type=float,
        default=0.06,
        help="Adaptive 阈值的上限（默认 6%）",
    )
    parser.add_argument(
        "--adaptive-threshold",
        action="store_true",
        help="Scale阈值随ATR波动调整（高波动标的需要更大涨跌幅才触发）。",
    )
    parser.add_argument("--model-name", default="default_model", help="Saved model name")
    parser.add_argument(
        "--model-type",
        choices=["auto", "hist_gb", "random_forest"],
        default="auto",
        help="选择模型架构；auto 会在候选模型之间选出验证表现最佳的方案。",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Force retraining even if saved model exists",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=15,
        help="Number of recent signals to display",
    )
    parser.add_argument(
        "--export-signals",
        action="store_true",
        help="将每个标的的信号明细导出到 reports/ 目录，方便事后回测或复盘。",
    )
    parser.add_argument(
        "--summary-report",
        action="store_true",
        help="生成 Markdown 总览报告（reports/summary_*.md）。",
    )
    parser.add_argument(
        "--summary-filename",
        default=None,
        help="自定义 Markdown 报告文件名（默认自动带时间戳）。",
    )
    parser.add_argument(
        "--resample-frequency",
        choices=["daily", "weekly"],
        default="daily",
        help="按日或按周聚合价格数据，周频更适合长期加减仓节奏。",
    )
    parser.add_argument(
        "--report-metrics",
        action="store_true",
        help="输出模型在当前数据集上的分类报告与混淆矩阵，供效果评估。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    console.print(
        f"Preparing dataset for tickers: {', '.join(args.tickers)} | Lookback: {args.lookback_years} years"
    )

    analysis = run_analysis(
        tickers=args.tickers,
        lookback_years=args.lookback_years,
        horizon=args.horizon,
        threshold=args.threshold,
        adaptive_threshold=args.adaptive_threshold,
        min_threshold=args.min_threshold,
        max_threshold=args.max_threshold,
        resample_frequency=args.resample_frequency,
        model_name=args.model_name,
        model_type=args.model_type,
        train=args.train,
        console=console,
    )

    meta = analysis["meta"]
    console.print(
        f"Data coverage: {meta.get('data_start')} → {meta.get('data_end')} | Today: {meta.get('today')}"
    )
    model_types = meta.get("model_type_by_market")
    if isinstance(model_types, dict) and model_types:
        model_info = ", ".join(f"{market}: {m_type}" for market, m_type in model_types.items())
        console.print(f"Model selection | {model_info}")

    missing = meta.get("missing_tickers") or []
    if missing:
        console.print(f"[yellow]未获取到数据的代码：{', '.join(missing)}[/yellow]")

    reports = analysis["reports"]
    backtests = analysis.get("backtests", {})
    if not reports:
        console.print("[bold red]No signals available for the requested tickers.[/bold red]")
        return

    summary_entries = []
    for ticker, report_df in reports.items():
        if report_df is None or report_df.empty:
            console.print(f"[bold red]{ticker}[/bold red] No data after preprocessing.")
            continue

        render_signal_summary(report_df, ticker)
        render_historical_table(report_df, limit=args.limit)
        bt = backtests.get(ticker)
        if bt:
            console.print(
                "  Backtest | horizon: {h} | avg_ret: {avg:.3f} | hit_ratio: {hit:.2%} |"
                " cumulative: {cum:.3f} | trades: {trades}".format(
                    h=bt.get("horizon"),
                    avg=bt.get("average_signal_return"),
                    hit=bt.get("hit_ratio", 0.0),
                    cum=bt.get("cumulative_return"),
                    trades=bt.get("trades"),
                )
            )
        if args.export_signals:
            path = export_signal_csv(report_df, ticker)
            console.print(f"[green]Saved signal history to {path}[/green]")
        if args.summary_report:
            summary_entries.append((ticker, report_df))

    if args.summary_report and summary_entries:
        summary_meta = {
            "run_time": dt.datetime.utcnow(),
            "data_start": meta.get("data_start"),
            "data_end": meta.get("data_end"),
            "today": meta.get("today"),
            "frequency": meta.get("resample_frequency"),
            "horizon": meta.get("horizon"),
            "threshold": meta.get("threshold"),
            "min_threshold": meta.get("min_threshold"),
            "max_threshold": meta.get("max_threshold"),
            "adaptive_threshold": meta.get("adaptive_threshold"),
        }
        summary_path = export_markdown_summary(
            summary_entries,
            metadata=summary_meta,
            limit=args.limit,
            filename=args.summary_filename,
        )
        console.print(f"[green]Summary report saved to {summary_path}[/green]")

    if args.report_metrics:
        dataset = analysis.get("dataset")
        predictions = analysis.get("predictions")
        if dataset is None or dataset.empty or predictions is None or predictions.empty:
            console.print("[yellow]无法生成分类报告：缺少数据集或预测结果。[/yellow]")
        else:
            y_true = dataset["label"]
            y_pred = predictions["model_signal"]
            console.print("\n[bold cyan]Classification Report[/bold cyan]")
            console.print(classification_report(y_true, y_pred, zero_division=0))

            labels = ["buy", "hold", "sell"]
            matrix = confusion_matrix(y_true, y_pred, labels=labels)
            table = Table(title="Confusion Matrix", show_lines=True)
            table.add_column("Actual \\ Pred", justify="left")
            for pred_label in labels:
                table.add_column(pred_label, justify="right")
            for idx, actual_label in enumerate(labels):
                row_values = [str(matrix[idx][j]) for j in range(len(labels))]
                table.add_row(actual_label, *row_values)
            console.print(table)


if __name__ == "__main__":
    main()
