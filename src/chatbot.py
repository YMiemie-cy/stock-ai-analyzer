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

BOT_PROFILES: Dict[str, Dict[str, str]] = {
    "GPT-5": {
        "persona": "你是 GPT-5 版本的投顾助手，擅长快速梳理关键信息并给出结构化的建议。",
        "style": "回答结构建议：先给出总体结论，再拆分成【信号概览】【操作建议】【风险提示】三部分。",
        "tone": "保持专业、简洁，突出可执行步骤。",
        "format": "请严格按照以下顺序输出：\n【信号概览】\n- …\n【操作建议】\n- …\n【风险提示】\n- …",
    },
    "Claude-Sonnet-4.5": {
        "persona": "你是 Claude Sonnet 4.5，擅长细腻的中文表达与深度推理，适合解释背景与逻辑。",
        "style": "回答结构建议：按【背景洞察】【信号含义】【操作思路】【注意事项】展开，适度引用历史。",
        "tone": "语气稳健、具有同理心，可提示用户从长期视角看待。",
        "format": "请按照以下结构输出：\n【背景洞察】\n- …\n【信号含义】\n- …\n【操作思路】\n- …\n【注意事项】\n- …",
    },
    "GPT-4o": {
        "persona": "你是 GPT-4o，善于从多维数据中提炼洞察并给出清晰的行动路线。",
        "style": "回答结构建议：使用分点或表格，让用户迅速理解差异与优先级。",
        "tone": "保持客观、效率导向，可给出不同场景下的行动方案。",
        "format": "请按照以下格式回答：\n【结论】\n- …\n【数据要点】\n- …\n【操作步骤】\n- …\n【风险控制】\n- …",
    },
    "Gemini-2.5-Pro": {
        "persona": "你是 Gemini 2.5 Pro，擅长跨市场与跨资产的关联分析。",
        "style": "回答结构建议：强调关联因素、相关资产以及可能的联动影响。",
        "tone": "保持理性、兼顾宏观与微观视角。",
        "format": "输出格式：\n【核心观点】\n- …\n【关联资产】\n- …\n【操作建议】\n- …\n【风险提示】\n- …",
    },
    "DeepSeek-R1": {
        "persona": "你是 DeepSeek R1，以量化交易员的思路给出严谨的分析，注重风险控制与数字证明。",
        "style": "回答结构建议：使用【信号拆解】【量化评估】【风险对冲】【执行建议】四段式。",
        "tone": "语言直接、重数据、不夸大收益，提醒潜在风险。",
        "format": "请固定使用以下结构：\n【信号拆解】\n- …\n【量化评估】\n- …\n【风险对冲】\n- …\n【执行建议】\n- …",
    },
}
DEFAULT_BOT_PROFILE: Dict[str, str] = {
    "persona": "你是 Stock AI Analyzer 的投顾助手。",
    "style": "回答时保持条理清晰，覆盖信号、建议与风险。",
    "tone": "语气稳健、客观。",
    "format": "请按照以下结构输出：\n【信号概览】\n- …\n【操作建议】\n- …\n【风险提示】\n- …",
}


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
    insights = results.get("insights", {})
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
        quality_prob = latest.get("meta_signal_prob")
        quality_text = f"{quality_prob:.0%}" if isinstance(quality_prob, (int, float)) else "--"
        quality_conf = latest.get("meta_signal_confidence")
        quality_conf_text = _fmt(quality_conf)
        risk_flags = latest.get("risk_flags") or []
        risk_text = ", ".join(risk_flags) if risk_flags else "无明显风险标记"

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
        lines.append(
            f"  信号质量：{quality_text}（置信 {quality_conf_text}）| 风险标记：{risk_text}"
        )
        if history_lines:
            lines.append("  最近信号：" + "； ".join(history_lines))
        lines.append(
            "  回测：平均收益 "
            f"{bt_avg} | 命中率 {bt_hit} | 累计收益 {bt_cum} | 交易次数 {bt_trades}"
        )

    perf = insights.get("performance_snapshot") or {}
    if perf:
        decisions = perf.get("decisions", {})
        lines.append(
            "模型整体："
            f"买/持/卖 {decisions.get('buy', 0)} / {decisions.get('hold', 0)} / {decisions.get('sell', 0)} | "
            f"平均置信度 {_fmt(perf.get('average_confidence'))} | 平均质量 {_fmt(perf.get('average_quality'))}"
        )
        best = perf.get("best_scenario") or {}
        if best:
            lines.append(
                f"最佳情景：价格冲击 {best.get('shock', 0.0):+.1%}，对齐收益约 {best.get('expected_alignment', 0.0):+.2%}"
            )

    alerts = insights.get("event_alerts") or []
    if alerts:
        top_alerts = alerts[:4]
        formatted = [f"{a.get('ticker')}: {a.get('message')}" for a in top_alerts]
        lines.append("事件提醒：" + "； ".join(formatted))

    market_summary = insights.get("market_summary", {}).get("items") or []
    if market_summary:
        summary_parts = []
        for item in market_summary[:4]:
            summary_parts.append(
                f"{item.get('market')}: 买/持/卖 {item.get('buy')}/{item.get('hold')}/{item.get('sell')} | "
                f"低质量 {item.get('low_quality', 0)}"
            )
        lines.append("市场布控：" + "； ".join(summary_parts))

    lines.append("注意：以上为模型信号，仅供研究与教育用途，不构成投资建议。")
    return "\n".join(lines)


def build_chat_messages(
    *,
    summary: str,
    question: str,
    history: Iterable[Dict[str, str]] | None = None,
    bot_name: str = DEFAULT_BOT,
) -> List[fp.ProtocolMessage]:
    profile = BOT_PROFILES.get(bot_name, DEFAULT_BOT_PROFILE)
    system_sections = [
        profile.get("persona", DEFAULT_BOT_PROFILE["persona"]),
        profile.get("style", DEFAULT_BOT_PROFILE["style"]),
        profile.get("tone", DEFAULT_BOT_PROFILE["tone"]),
        "可用的本地数据包括：\n"
        "- 逐标的模型信号：决策、价格、概率、置信度、信号质量（meta prob）、风险标记；\n"
        "- 回测表现：平均收益、命中率、累计收益、交易次数；\n"
        "- 事件提醒与市场布控：重要风险、低质量信号、市场整体状态；\n"
        "- 压力测试：不同冲击下的对齐收益与买卖承压情况。",
        "通用守则：\n"
        "1. 优先引用数据摘要中的事实，并指出来源；\n"
        "2. 结论不确定时说明局限与风险，不做承诺；\n"
        "3. 给出可执行的建议（增持/减持/观望等）并解释原因；\n"
        "4. 若用户问题超出数据范围，要明确告知；\n"
        "5. 始终使用中文作答。",
        "免责声明：模型信号仅供研究与教育用途，不构成投资建议。",
        f"---- 数据摘要 ----\n{summary}\n---- 摘要结束 ----",
    ]
    system_prompt = "\n\n".join(system_sections)
    format_hint = profile.get("format", DEFAULT_BOT_PROFILE["format"])
    system_prompt = f"{system_prompt}\n\n请严格遵循下面的输出格式：\n{format_hint}"
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
