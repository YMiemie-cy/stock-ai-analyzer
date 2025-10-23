"""Utilities for composing AI assistant prompts and calling Poe API."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence

import fastapi_poe as fp

SUPPORTED_BOTS: Sequence[str] = (
    "GPT-5",
    "Claude-Sonnet-4.5",
    "GPT-4o",
    "Gemini-2.5-Pro",
    "DeepSeek-R1",
)
DEFAULT_BOT: str = SUPPORTED_BOTS[0]


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "--"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def build_analysis_summary(results: Dict[str, Any], realtime: Dict[str, Dict[str, Any]]) -> str:
    meta = results.get("meta", {})
    mapping = meta.get("ticker_mapping", {})
    reverse_mapping = {display: norm for norm, display in mapping.items()}
    lines: List[str] = []
    lines.append(
        "分析概览："
        f"数据区间 {meta.get('data_start')} ~ {meta.get('data_end')}，"
        f"预测周期 {meta.get('horizon')} 周，频率 {meta.get('resample_frequency')},"
        f" 自适应阈值 {'开启' if meta.get('adaptive_threshold') else '关闭'}。"
    )

    backtests = results.get("backtests", {})
    for display_ticker, report_df in results.get("reports", {}).items():
        latest = results.get("latest", {}).get(display_ticker, {})
        norm_ticker = reverse_mapping.get(display_ticker, display_ticker)
        realtime_entry = realtime.get(norm_ticker, {})
        decision = str(latest.get("decision", "hold")).upper()
        model_price = _fmt(latest.get("price"))
        realtime_price = _fmt(realtime_entry.get("price"))
        prob_buy = _fmt(latest.get("prob_buy"))
        prob_hold = _fmt(latest.get("prob_hold"))
        prob_sell = _fmt(latest.get("prob_sell"))
        prob_gap = _fmt(latest.get("prob_gap"))
        confidence = _fmt(latest.get("confidence"))

        bt = backtests.get(display_ticker) or {}
        bt_avg = _fmt(bt.get("average_signal_return"), digits=3)
        bt_hit = f"{bt.get('hit_ratio', 0.0):.2%}" if bt else "--"
        bt_cum = _fmt(bt.get("cumulative_return"), digits=3)
        bt_trades = bt.get("trades", "--")

        history_lines: List[str] = []
        try:
            tail_df = report_df.tail(3)
            for ts, row in tail_df.iterrows():
                history_lines.append(
                    f"{ts.date()}: {row['decision']} "
                    f"(价格 {_fmt(row.get('price'))}, 分数 {_fmt(row.get('score'))})"
                )
        except Exception:
            pass

        ticker_line = (
            f"{display_ticker}: 决策 {decision} | 模型收盘价 {model_price} | 实时价 {realtime_price} | "
            f"概率 buy {prob_buy} / hold {prob_hold} / sell {prob_sell} | prob_gap {prob_gap} | 置信度 {confidence}"
        )
        lines.append(ticker_line)
        if history_lines:
            lines.append("  最近信号：" + "； ".join(history_lines))
        lines.append(
            "  回测：平均收益 "
            f"{bt_avg} | 命中率 {bt_hit} | 累计收益 {bt_cum} | 交易次数 {bt_trades}"
        )

    lines.append("注意：以上为模型信号，仅供研究与教育用途，不构成投资建议。")
    return "\n".join(lines)


def build_chat_messages(
    *,
    summary: str,
    question: str,
    history: Iterable[Dict[str, str]] | None = None,
) -> List[fp.ProtocolMessage]:
    system_prompt = (
        "你是 Stock AI Analyzer 的投顾助手，需要根据提供的模型信号和实时行情，"
        "为用户提供中文的投资分析和建议。\n"
        "务必遵循以下原则：\n"
        "1. 优先引用数据摘要中的事实，并在回答中明确说明结论来源。\n"
        "2. 若信息不足或结论不确定，要说明局限性和潜在风险。\n"
        "3. 不要输出任何与股票无关或不确定的内容，不做承诺。\n"
        "4. 在建议中可给出多种方案（如增持/观望/减持），并说明理由。\n"
        "5. 回答必须使用中文，保持条理清晰。\n\n"
        f"以下是最新的数据摘要：\n{summary}\n"
        "---- 摘要结束 ----"
    )
    messages: List[fp.ProtocolMessage] = [
        fp.ProtocolMessage(role="system", content=system_prompt)
    ]
    if history:
        for msg in history:
            role = msg.get("role")
            content = msg.get("content")
            if not content:
                continue
            if role == "assistant":
                mapped_role = "bot"
            elif role in {"user", "bot"}:
                mapped_role = role
            else:
                continue
            messages.append(fp.ProtocolMessage(role=mapped_role, content=content))
    messages.append(fp.ProtocolMessage(role="user", content=question))
    return messages


def collect_response_text(messages: Sequence[fp.ProtocolMessage], bot_name: str, api_key: str) -> str:
    chunks = []
    generator = fp.get_bot_response_sync(messages=messages, bot_name=bot_name, api_key=api_key)
    for partial in generator:
        text = getattr(partial, "text", "")
        if text:
            chunks.append(text)
    return "".join(chunks).strip()
