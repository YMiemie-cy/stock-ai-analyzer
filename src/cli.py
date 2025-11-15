"""Command-line interface for the stock AI analyzer prototype."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
from pathlib import Path
from typing import Dict, List, Sequence

from rich.console import Console
from rich.table import Table
from sklearn.metrics import classification_report, confusion_matrix

from .core import run_analysis
from .report import export_markdown_summary, export_signal_csv, render_historical_table, render_signal_summary

console = Console()

DEFAULT_THRESHOLD = 0.02
DEFAULT_MIN_THRESHOLD = 0.01
DEFAULT_MAX_THRESHOLD = 0.06
DEFAULT_HORIZON = 5
DEFAULT_LOOKBACK = 5

RISK_PROFILE_PRESETS: Dict[str, Dict[str, float]] = {
    "conservative": {
        "threshold": 0.028,
        "min_threshold": 0.015,
        "max_threshold": 0.065,
        "horizon": 7,
        "lookback_years": 7,
    },
    "balanced": {
        "threshold": 0.022,
        "min_threshold": 0.012,
        "max_threshold": 0.06,
        "horizon": 5,
        "lookback_years": 5,
    },
    "aggressive": {
        "threshold": 0.016,
        "min_threshold": 0.008,
        "max_threshold": 0.05,
        "horizon": 3,
        "lookback_years": 4,
    },
}

DEFAULT_SCENARIO_SHOCKS: Sequence[float] = (-0.06, -0.03, 0.0, 0.03, 0.06)
DEFAULT_METRICS_LOG_PATH = Path("reports/metrics_log.csv")


def _apply_risk_profile(args: argparse.Namespace) -> Dict[str, List[str]]:
    profile = getattr(args, "risk_profile", "balanced")
    preset = RISK_PROFILE_PRESETS.get(profile)
    notes: List[str] = []
    if not preset:
        return {"profile": profile, "notes": notes}

    if getattr(args, "threshold", DEFAULT_THRESHOLD) == DEFAULT_THRESHOLD:
        args.threshold = float(preset["threshold"])
        notes.append(f"阈值调整为 {args.threshold:.3f}")

    if getattr(args, "min_threshold", DEFAULT_MIN_THRESHOLD) == DEFAULT_MIN_THRESHOLD:
        args.min_threshold = float(preset["min_threshold"])
    if getattr(args, "max_threshold", DEFAULT_MAX_THRESHOLD) == DEFAULT_MAX_THRESHOLD:
        args.max_threshold = float(preset["max_threshold"])

    if getattr(args, "horizon", DEFAULT_HORIZON) == DEFAULT_HORIZON:
        args.horizon = int(preset["horizon"])
        notes.append(f"预测周期调整为 {args.horizon} 天")

    if getattr(args, "lookback_years", DEFAULT_LOOKBACK) == DEFAULT_LOOKBACK:
        args.lookback_years = int(preset["lookback_years"])

    return {"profile": profile, "notes": notes}


def _build_metrics_record(meta: dict, insights: dict) -> Dict[str, object]:
    snapshot = insights.get("performance_snapshot") or {}
    decisions = snapshot.get("decisions") or {}
    market_items = (insights.get("market_summary") or {}).get("items", [])
    total_low_confidence = sum(int(item.get("low_confidence", 0)) for item in market_items)
    total_near_threshold = sum(int(item.get("near_threshold", 0)) for item in market_items)
    total_recent_flip = sum(int(item.get("recent_flip", 0)) for item in market_items)
    total_low_quality = sum(int(item.get("low_quality", 0)) for item in market_items)
    best_scenario = snapshot.get("best_scenario") or {}

    tickers = meta.get("requested_tickers") or meta.get("available_tickers") or []
    record = {
        "timestamp": dt.datetime.utcnow().isoformat(),
        "tickers": "|".join(tickers),
        "risk_profile": meta.get("risk_profile"),
        "lookback_years": meta.get("lookback_years"),
        "horizon": meta.get("horizon"),
        "buy": decisions.get("buy", 0),
        "hold": decisions.get("hold", 0),
        "sell": decisions.get("sell", 0),
        "average_confidence": snapshot.get("average_confidence"),
        "low_confidence_flags": total_low_confidence,
        "near_threshold_flags": total_near_threshold,
        "recent_flip_flags": total_recent_flip,
        "low_quality_flags": total_low_quality,
        "best_scenario_shock": best_scenario.get("shock"),
        "best_scenario_expected_alignment": best_scenario.get("expected_alignment"),
        "best_scenario_buy_pressure": best_scenario.get("buy_under_pressure"),
        "best_scenario_sell_recovery": best_scenario.get("sell_recovery"),
        "best_scenario_neutral_breakout": best_scenario.get("neutral_breakout"),
    }
    return record


def _write_metrics_log(path: Path, record: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp",
        "tickers",
        "risk_profile",
        "lookback_years",
        "horizon",
        "buy",
        "hold",
        "sell",
        "average_confidence",
        "low_confidence_flags",
        "near_threshold_flags",
        "recent_flip_flags",
        "low_quality_flags",
        "best_scenario_shock",
        "best_scenario_expected_alignment",
        "best_scenario_buy_pressure",
        "best_scenario_sell_recovery",
        "best_scenario_neutral_breakout",
    ]
    file_exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI-assisted buy/sell signal analyzer")
    parser.add_argument("--tickers", nargs="+", required=True, help="Tickers to analyze")
    parser.add_argument("--lookback-years", type=int, default=DEFAULT_LOOKBACK, help="Years of history to download")
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON, help="Days ahead for return labeling")
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Return threshold for buy/sell labeling (e.g. 0.02 = 2%)",
    )
    parser.add_argument(
        "--min-threshold",
        type=float,
        default=DEFAULT_MIN_THRESHOLD,
        help="Adaptive 阈值的下限（默认 1%）",
    )
    parser.add_argument(
        "--max-threshold",
        type=float,
        default=DEFAULT_MAX_THRESHOLD,
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
        choices=["auto", "hist_gb", "random_forest", "lightgbm"],
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
    parser.add_argument(
        "--risk-profile",
        choices=tuple(RISK_PROFILE_PRESETS.keys()) + ("custom",),
        default="balanced",
        help="根据风险偏好自动调节阈值与预测周期（conservative/balanced/aggressive/custom）。",
    )
    parser.add_argument(
        "--risk-max-drawdown",
        type=float,
        default=None,
        help="自定义最大回撤容忍度（例如 0.12 表示 12%），用于持仓健康评估。",
    )
    parser.add_argument(
        "--risk-target-volatility",
        type=float,
        default=None,
        help="自定义目标波动率（例如 0.25），用于持仓健康评估。",
    )
    parser.add_argument(
        "--scenario-shocks",
        type=float,
        nargs="+",
        default=None,
        help="场景压力测试的价格冲击百分比（例如 -0.05 -0.03 0.03 0.07）。",
    )
    parser.add_argument(
        "--briefing-top",
        type=int,
        default=5,
        help="开盘摘要中展示的重点标的数量。",
    )
    parser.add_argument(
        "--skip-briefing",
        action="store_true",
        help="跳过开盘摘要输出，仅展示原有信号与表格。",
    )
    parser.add_argument(
        "--portfolio-path",
        default=None,
        help="持仓文件（JSON）路径，默认读取项目根目录下的 portfolio.json。",
    )
    parser.add_argument(
        "--metrics-log",
        nargs="?",
        const=str(DEFAULT_METRICS_LOG_PATH),
        default=None,
        help="记录本次运行的指标概览到 CSV（默认 reports/metrics_log.csv）。",
    )
    parser.add_argument(
        "--use-deepseek",
        action="store_true",
        help="启用 DeepSeek 模型融合（需提供 DEEPSEEK_API_KEY 或 --deepseek-api-key）。",
    )
    parser.add_argument(
        "--deepseek-api-key",
        default=None,
        help="DeepSeek API key，若省略则回退到环境变量 DEEPSEEK_API_KEY。",
    )
    parser.add_argument(
        "--deepseek-model",
        default=None,
        help="DeepSeek 模型名称（默认 deepseek-chat）。",
    )
    parser.add_argument(
        "--deepseek-weight",
        type=float,
        default=0.35,
        help="DeepSeek 结果在概率融合中的权重（0-1，默认 0.35）。",
    )
    parser.add_argument(
        "--deepseek-process-all",
        action="store_true",
        help="对每个标的的全部样本调用 DeepSeek（默认只处理最新一条记录）。",
    )
    parser.add_argument(
        "--deepseek-max-rows",
        type=int,
        default=60,
        help="DeepSeek 单次最多处理的样本数（默认 60）。",
    )
    parser.add_argument(
        "--deepseek-timeout",
        type=float,
        default=25.0,
        help="DeepSeek API 请求超时时间（秒）。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenario_shocks = args.scenario_shocks or list(DEFAULT_SCENARIO_SHOCKS)
    risk_notes = _apply_risk_profile(args)

    console.print(
        f"Preparing dataset for tickers: {', '.join(args.tickers)} | Lookback: {args.lookback_years} years"
    )
    if risk_notes.get("notes"):
        joined = "；".join(risk_notes["notes"])
        console.print(f"[cyan]风险档位：{risk_notes['profile']}（{joined}）[/cyan]")

    deepseek_options = None
    if args.use_deepseek or args.deepseek_api_key:
        deepseek_options = {
            "api_key": args.deepseek_api_key,
            "weight": args.deepseek_weight,
            "latest_only": not args.deepseek_process_all,
            "max_rows": args.deepseek_max_rows,
            "timeout": args.deepseek_timeout,
        }
        if args.deepseek_model:
            deepseek_options["model"] = args.deepseek_model

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
        deepseek_options=deepseek_options,
        risk_profile=args.risk_profile,
        risk_limits={
            "max_drawdown": args.risk_max_drawdown,
            "target_volatility": args.risk_target_volatility,
        },
        scenario_shocks=scenario_shocks,
        portfolio_path=args.portfolio_path,
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

    insights = analysis.get("insights", {})
    if not args.skip_briefing:
        briefing = insights.get("daily_briefing") or {}
        top_signals = briefing.get("top_signals") or []
        if top_signals:
            console.print("\n[bold magenta]开盘智能摘要[/bold magenta]")
            console.print(briefing.get("headline", ""))
            table = Table(title="重点标的", show_lines=False)
            table.add_column("Ticker", style="bold")
            table.add_column("决策", justify="center")
            table.add_column("置信度", justify="right")
            table.add_column("价格", justify="right")
            table.add_column("提示", justify="left")
            for entry in top_signals[: args.briefing_top]:
                confidence = entry.get("confidence")
                confidence_str = f"{confidence:.0%}" if isinstance(confidence, float) else "--"
                price = entry.get("price")
                price_str = f"{price:.2f}" if isinstance(price, float) else "--"
                action = entry.get("action_hint") or ""
                table.add_row(
                    str(entry.get("ticker")),
                    str(entry.get("decision", "")).upper(),
                    confidence_str,
                    price_str,
                    action,
                )
            console.print(table)
            watchlist = briefing.get("watchlist") or []
            if watchlist:
                console.print("[yellow]风险观察[/yellow]")
                for item in watchlist[: args.briefing_top]:
                    issues = "；".join(item.get("issues", []))
                    console.print(f" • {item.get('ticker')}: {issues}")

    event_alerts = insights.get("event_alerts") or []
    if event_alerts:
        console.print("\n[bold magenta]事件驱动提醒[/bold magenta]")
        severity_color = {"critical": "red", "warning": "yellow", "info": "cyan"}
        for alert in event_alerts[: args.briefing_top]:
            color = severity_color.get(alert.get("severity"), "cyan")
            console.print(
                f"[{color}]{alert.get('ticker')} | {alert.get('message')} ({alert.get('type')})[/{color}]"
            )

    scenario_matrix = insights.get("scenario_matrix") or {}
    scenarios = scenario_matrix.get("scenarios") or []
    if scenarios:
        console.print("\n[bold magenta]场景压力测试[/bold magenta]")
        for scenario in scenarios[:3]:
            shock = scenario.get("shock", 0.0)
            summary = scenario.get("summary", {})
            console.print(
                f"{shock:+.1%}：买入承压 {summary.get('buy_under_pressure', 0)} | "
                f"卖出回补 {summary.get('sell_recovery', 0)} | "
                f"观望波动 {summary.get('neutral_breakout', 0)} | "
                f"对齐收益 {summary.get('expected_alignment', 0.0):+.2%}"
            )
        best_scenario = (scenario_matrix.get("portfolio_expected_returns") or [])
        if best_scenario:
            top = max(best_scenario, key=lambda item: item.get("expected_alignment", 0.0))
            console.print(
                f"[cyan]最优情景：冲击 {top.get('shock', 0.0):+.1%} 时，对齐收益约 {top.get('expected_alignment', 0.0):+.2%}[/cyan]"
            )

    market_summary = insights.get("market_summary") or {}
    market_items = market_summary.get("items") or []
    if market_items:
        console.print("\n[bold magenta]市场布控[/bold magenta]")
        table = Table(title="按市场信号统计", show_lines=False)
        table.add_column("市场", justify="left")
        table.add_column("买/持/卖", justify="center")
        table.add_column("平均信心", justify="right")
        table.add_column("风险计数", justify="left")
        for item in market_items[:6]:
            risk_str = (
                f"LC {item.get('low_confidence', 0)}｜NT {item.get('near_threshold', 0)}"
                f"｜Flip {item.get('recent_flip', 0)}｜LQ {item.get('low_quality', 0)}"
            )
            table.add_row(
                str(item.get("market")),
                f"{item.get('buy', 0)}/{item.get('hold', 0)}/{item.get('sell', 0)}",
                f"{item.get('avg_confidence', 0.0):.2f}",
                risk_str,
            )
        console.print(table)

    performance_snapshot = insights.get("performance_snapshot") or {}
    if performance_snapshot:
        console.print(
            "\n[bold magenta]模型表现快照[/bold magenta]\n"
            f"信号统计（买/持/卖）："
            f"{performance_snapshot.get('decisions', {}).get('buy', 0)} / "
            f"{performance_snapshot.get('decisions', {}).get('hold', 0)} / "
            f"{performance_snapshot.get('decisions', {}).get('sell', 0)} | "
            f"平均置信度 {performance_snapshot.get('average_confidence', 0.0):.2f} | "
            f"平均质量 {performance_snapshot.get('average_quality', 0.0):.2f}"
        )

    portfolio_health = insights.get("portfolio_health") or {}
    holdings = portfolio_health.get("holdings") or []
    if holdings:
        console.print("\n[bold magenta]持仓健康面板[/bold magenta]")
        summary = portfolio_health.get("summary") or {}
        console.print(
            f"绿 {summary.get('green', 0)}｜黄 {summary.get('yellow', 0)}｜红 {summary.get('red', 0)}"
            f" ｜缺失 {summary.get('missing', 0)}"
        )
        rating_order = {"red": 2, "yellow": 1, "green": 0}
        sorted_holdings = sorted(
            holdings,
            key=lambda item: (rating_order.get(item.get("rating"), 0), -(item.get("confidence") or 0.0)),
            reverse=True,
        )
        table = Table(title="持仓状态", show_lines=False)
        table.add_column("Ticker", style="bold")
        table.add_column("评级", justify="center")
        table.add_column("决策", justify="center")
        table.add_column("置信度", justify="right")
        table.add_column("提示", justify="left")
        for item in sorted_holdings[: args.briefing_top]:
            rating = item.get("rating", "green")
            rating_label = {"green": "GREEN", "yellow": "YELLOW", "red": "RED"}.get(rating, rating.upper())
            confidence = item.get("confidence")
            confidence_str = f"{confidence:.0%}" if isinstance(confidence, float) else "--"
            notes = "；".join(item.get("notes") or [])
            table.add_row(
                str(item.get("ticker")),
                rating_label,
                str(item.get("decision", "")).upper(),
                confidence_str,
                notes or "--",
            )
        console.print(table)
        missing = portfolio_health.get("missing") or []
        if missing:
            console.print(f"[yellow]未覆盖持仓：{', '.join(missing)}[/yellow]")

    prompts = insights.get("suggested_prompts") or []
    if prompts:
        console.print("\n[bold magenta]快捷提问建议[/bold magenta]")
        for prompt in prompts[: args.briefing_top + 2]:
            console.print(f" - {prompt}")

    if args.metrics_log:
        metrics_path = Path(args.metrics_log)
        record = _build_metrics_record(meta, insights)
        _write_metrics_log(metrics_path, record)
        console.print(f"[green]Metrics snapshot appended to {metrics_path}[/green]")

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
