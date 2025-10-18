const navButtons = document.querySelectorAll("nav button");
const views = document.querySelectorAll(".view");
const searchForm = document.getElementById("search-form");
const tickerInput = document.getElementById("tickers-input");
const tickerSuggestions = document.getElementById("ticker-suggestions");
const searchStatus = document.getElementById("search-status");
const searchResultsContainer = document.getElementById("search-results");
const portfolioStatus = document.getElementById("portfolio-status");
const portfolioResultsContainer = document.getElementById("portfolio-results");
const refreshPortfolioBtn = document.getElementById("refresh-portfolio");
const chatPanel = document.getElementById("chat-panel");
const chatStatus = document.getElementById("chat-status");
const chatLog = document.getElementById("chat-log");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const chatBotSelect = document.getElementById("chat-bot");
const portfolioChatPanel = document.getElementById("portfolio-chat-panel");
const portfolioChatStatus = document.getElementById("portfolio-chat-status");
const portfolioChatLog = document.getElementById("portfolio-chat-log");
const portfolioChatForm = document.getElementById("portfolio-chat-form");
const portfolioChatInput = document.getElementById("portfolio-chat-input");
const portfolioChatBotSelect = document.getElementById("portfolio-chat-bot");
const decisionFilter = document.getElementById("decision-filter");
const sortSelect = document.getElementById("sort-select");
const frequencySelect = document.getElementById("frequency-select");
const refreshSearchBtn = document.getElementById("refresh-search");
const searchUpdatedAt = document.getElementById("search-updated");
const resultControls = document.getElementById("result-controls");
const quickPromptButtons = document.querySelectorAll("[data-prompt][data-target='search']");
const portfolioQuickPromptButtons = document.querySelectorAll("[data-prompt][data-target='portfolio']");
const portfolioFrequencySelect = document.getElementById("portfolio-frequency-select");
const portfolioSortSelect = document.getElementById("portfolio-sort-select");
const portfolioFilterSelect = document.getElementById("portfolio-filter-select");
const portfolioControls = document.getElementById("portfolio-controls");

let latestSearchResults = [];
let latestSearchMeta = null;
let latestSearchPayload = null;
let latestPortfolioResults = [];
let latestPortfolioMeta = null;
let latestPortfolioPayload = null;
let latestSearchResultsByFrequency = {};
let latestSearchMetaByFrequency = {};
let latestPortfolioResultsByFrequency = {};
let latestPortfolioMetaByFrequency = {};
let currentSearchFrequency = "weekly";
let currentPortfolioFrequency = "weekly";
let lastSearchBasePayload = null;

const DEFAULT_SORT = "origin";
const FREQUENCY_LABELS = {
  daily: "日线",
  weekly: "周线",
};
const FREQUENCY_ORDER = {
  weekly: 0,
  daily: 1,
};

function normalizeFrequency(value) {
  if (!value) {
    return "weekly";
  }
  const normalized = String(value).toLowerCase();
  if (normalized.startsWith("d")) {
    return "daily";
  }
  if (normalized.startsWith("w")) {
    return "weekly";
  }
  return normalized;
}

function frequencyLabel(freq) {
  const key = normalizeFrequency(freq);
  return FREQUENCY_LABELS[key] || key;
}

if (searchResultsContainer) {
  searchResultsContainer.innerHTML = "<div class=\"empty-state\">请输入股票代码，以获取即时买卖参考。</div>";
}

let lookupTimer = null;
let lookupAbortController = null;
let lastLookupQuery = "";
let portfolioTimer = null;
let chatHistory = [];
let chatSending = false;
let portfolioChatHistory = [];
let portfolioChatSending = false;

function switchView(target) {
  views.forEach((view) => view.classList.toggle("active", view.id === `${target}-view`));
  navButtons.forEach((btn) => btn.classList.toggle("active", btn.dataset.view === target));
}

navButtons.forEach((button) => {
  button.addEventListener("click", () => {
    switchView(button.dataset.view);
    if (button.dataset.view === "portfolio") {
      loadPortfolio();
      setupPortfolioAutoRefresh();
    } else {
      clearPortfolioAutoRefresh();
      if (button.dataset.view === "search") {
        renderSearchResults();
      }
    }
  });
});

if (tickerInput) {
  tickerInput.addEventListener("input", handleTickerInputChange);
  tickerInput.addEventListener("focus", handleTickerInputChange);
  tickerInput.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      hideTickerSuggestions();
    }
  });
}

if (tickerSuggestions) {
  tickerSuggestions.addEventListener("mousedown", (event) => {
    const target = event.target.closest(".suggestions__item");
    if (!target) {
      return;
    }
    event.preventDefault();
    const symbol = target.dataset.symbol;
    applyTickerSuggestion(symbol);
  });
}

document.addEventListener("click", (event) => {
  if (!tickerSuggestions || tickerSuggestions.classList.contains("hidden")) {
    return;
  }
  if (tickerSuggestions.contains(event.target) || event.target === tickerInput) {
    return;
  }
  hideTickerSuggestions();
});

function clearPortfolioAutoRefresh() {
  if (portfolioTimer) {
    clearInterval(portfolioTimer);
    portfolioTimer = null;
  }
}

function setupPortfolioAutoRefresh(interval = 60_000) {
  clearPortfolioAutoRefresh();
  portfolioTimer = setInterval(() => loadPortfolio(), interval);
}

function formatDecision(decision) {
  const cls = `decision-${decision}`;
  const label = decision.toUpperCase();
  return `<span class="${cls}">${label}</span>`;
}

function formatPercent(value, digits = 0) {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "--";
  }
  return `${(value * 100).toFixed(digits)}%`;
}

function formatPercentWithSign(value, digits = 1) {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "--";
  }
  const percent = (value * 100).toFixed(digits);
  const sign = value > 0 ? "+" : "";
  return `${sign}${percent}%`;
}

function formatDecimal(value, digits = 2) {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "--";
  }
  return value.toFixed(digits);
}

function formatDateTime(value) {
  if (!value) {
    return "--";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "--";
  }
  return date.toISOString().replace("T", " ").slice(0, 16);
}

function ensureArray(value) {
  return Array.isArray(value) ? value : [];
}

function setLoadingState(element, isLoading) {
  if (!element) {
    return;
  }
  if (isLoading) {
    element.setAttribute("data-loading", "true");
  } else {
    element.removeAttribute("data-loading");
  }
}

function buildRiskBadges(latest) {
  const flags = ensureArray(latest?.risk_flags);
  const badges = [];
  const probGap = typeof latest?.prob_gap === "number" ? latest.prob_gap : null;
  const thresholdDistance = typeof latest?.threshold_distance === "number" ? latest.threshold_distance : null;
  const signalAge = typeof latest?.signal_age_days === "number" ? latest.signal_age_days : null;
  if (flags.includes("low_confidence")) {
    const detail = probGap !== null ? `概率差 ${formatDecimal(probGap, 2)}` : "概率差偏低";
    badges.push(`<span class="risk-badge risk-badge--low" title="${detail}">信心不足</span>`);
  }
  if (flags.includes("near_threshold")) {
    const distanceText = thresholdDistance !== null ? `距阈值 ${formatDecimal(thresholdDistance, 3)}` : "接近阈值";
    badges.push(`<span class="risk-badge risk-badge--edge" title="${distanceText}">临界</span>`);
  }
  if (flags.includes("recent_flip")) {
    const recency = signalAge === 0 ? "当日刚翻转" : signalAge ? `${signalAge} 日前翻转` : "近期信号翻转";
    badges.push(`<span class="risk-badge risk-badge--flip" title="${recency}">刚刚反转</span>`);
  }
  return badges.join("");
}

function computeConfidenceScore(row) {
  if (row && typeof row.confidence_score === "number" && !Number.isNaN(row.confidence_score)) {
    return row.confidence_score;
  }
  const probGap = Number(row?.prob_gap ?? 0);
  const indicatorBias = Number(row?.indicator_bias ?? 0);
  const probStrength = Math.max(probGap, 0);
  const indicatorStrength = Math.min(Math.max(Math.abs(indicatorBias) / 2, 0), 1);
  return Number((probStrength * 0.7 + indicatorStrength * 0.3).toFixed(2));
}

function getConfidenceFromRow(row) {
  if (!row) {
    return 0;
  }
  const score = computeConfidenceScore(row);
  return Number.isFinite(score) ? score : 0;
}

function renderTable(tickerData, limit = 5) {
  const history = tickerData.history ?? [];
  const latest = tickerData.latest ?? {};
  const frequency = normalizeFrequency(latest.analysis_frequency ?? tickerData.frequency ?? currentSearchFrequency);
  const frequencyBadge = `<span class="freq-badge freq-${frequency}">${frequencyLabel(frequency)}</span>`;
  const riskBadges = buildRiskBadges(latest);
  const decision = (latest?.decision ?? "hold").toLowerCase();
  const confidence = getConfidenceFromRow(latest);
  const probBuy = typeof latest.prob_buy === "number" ? latest.prob_buy : null;
  const probSell = typeof latest.prob_sell === "number" ? latest.prob_sell : null;
  const probGap = typeof latest.prob_gap === "number" ? latest.prob_gap : null;
  const probDiff = typeof latest.prob_diff_buy_sell === "number" ? latest.prob_diff_buy_sell : null;
  const decisionLabel = decision.toUpperCase();
  const latestDate = latest.timestamp?.split("T")[0] ?? "--";
  const realtimeTimestamp = formatDateTime(latest.realtime_price_timestamp);
  const realtimePrice =
    typeof latest.realtime_price === "number" && !Number.isNaN(latest.realtime_price)
      ? latest.realtime_price.toFixed(2)
      : null;
  const modelPrice =
    typeof latest.price === "number" && !Number.isNaN(latest.price) ? latest.price.toFixed(2) : "--";
  const displayPrice = realtimePrice ?? modelPrice;
  const hasRealtime = Boolean(realtimePrice);
  const actionHint = latest.action_hint ?? "";
  const returnSinceSignal = formatPercentWithSign(latest.return_since_signal);
  const signalAgeLabel = latest.signal_is_recent
    ? "刚刚更新"
    : typeof latest.signal_age_days === "number"
    ? `${latest.signal_age_days} 日前`
    : "--";
  const signalChangedAt = latest.signal_changed_at ? formatDateTime(latest.signal_changed_at) : "--";
  const metricsRow = (label, value) =>
    `<div class="metric"><span class="metric__label">${label}</span><span class="metric__value">${value}</span></div>`;
  const drivers = ensureArray(latest.key_drivers);
  const driverMarkup = drivers.length
    ? `<ul class="result-card__drivers">${drivers.map((item) => `<li>${item}</li>`).join("")}</ul>`
    : `<div class="result-card__drivers result-card__drivers--empty">暂无关键驱动说明。</div>`;
  const cardClasses = ["result-card", `summary-${decision}`, `result-card--${frequency}`];
  const riskFlags = ensureArray(latest.risk_flags);
  if (riskFlags.includes("low_confidence")) {
    cardClasses.push("result-card--low-confidence");
  }
  if (riskFlags.includes("near_threshold")) {
    cardClasses.push("result-card--near-threshold");
  }
  if (riskFlags.includes("recent_flip")) {
    cardClasses.push("result-card--recent-flip");
  }
  const rows = history.slice(-limit).map((row) => {
    const confidence = computeConfidenceScore(row);
    const return1d = formatPercentWithSign(row.return_1d);
    const return5d = formatPercentWithSign(row.return_5d);
    const return10d = formatPercentWithSign(row.return_10d);
    return `
      <tr>
        <td>${row.timestamp.split("T")[0]}</td>
        <td>${formatDecision(row.decision)}</td>
        <td>${row.price?.toFixed?.(2) ?? "--"}</td>
        <td>${row.score.toFixed(2)}</td>
        <td>${row.prob_buy.toFixed(2)}</td>
        <td>${row.prob_sell.toFixed(2)}</td>
        <td>${row.prob_gap.toFixed(2)}</td>
        <td>${confidence.toFixed(2)}</td>
        <td>${return1d}</td>
        <td>${return5d}</td>
        <td>${return10d}</td>
      </tr>
    `;
  });

  return `
    <article class="${cardClasses.join(" ")}">
      <header class="result-card__header">
        <div class="result-card__header-left">
          <div class="result-card__ticker">${tickerData.ticker}</div>
          <div class="result-card__badges">
            ${frequencyBadge}
            ${riskBadges}
          </div>
        </div>
        <div class="result-card__prices">
          <div class="price-badge live${hasRealtime ? "" : " disabled"}">
            <span class="label">现价</span>
            <span class="value">${displayPrice}</span>
          </div>
          <div class="price-badge snapshot">
            <span class="label">收盘价</span>
            <span class="value">${modelPrice}</span>
          </div>
        </div>
      </header>
      <p class="result-card__meta">
        最新日期：${latestDate} ｜ 建议：${decisionLabel} ｜ ${actionHint || "暂无操作建议"}
      </p>
      <div class="result-card__summary">
        <div class="decision-chip">${decisionLabel}</div>
        <div class="summary-metrics">
          <span>信心 <strong>${formatPercent(confidence, 0)}</strong></span>
          <span>买入 <strong>${formatPercent(probBuy, 0)}</strong></span>
          <span>卖出 <strong>${formatPercent(probSell, 0)}</strong></span>
          <span>概率差 <strong>${formatDecimal(probGap, 2)}</strong></span>
          <span>买卖差 <strong>${formatDecimal(probDiff, 2)}</strong></span>
        </div>
      </div>
      <div class="result-card__insights">
        <div class="metric-group">
          <h4>短期收益</h4>
          ${metricsRow("1 日", formatPercentWithSign(latest.return_1d))}
          ${metricsRow("3 日", formatPercentWithSign(latest.return_3d))}
          ${metricsRow("5 日", formatPercentWithSign(latest.return_5d))}
          ${metricsRow("10 日", formatPercentWithSign(latest.return_10d))}
        </div>
        <div class="metric-group">
          <h4>风险指示</h4>
          ${metricsRow("波动 (5 日)", formatPercent(Math.abs(latest.volatility_5d ?? NaN), 1))}
          ${metricsRow("波动 (10 日)", formatPercent(Math.abs(latest.volatility_10d ?? NaN), 1))}
          ${metricsRow("趋势强度 (20 日)", formatDecimal(latest.trend_strength_20d, 2))}
          ${metricsRow("趋势强度 (60 日)", formatDecimal(latest.trend_strength_60d, 2))}
        </div>
      </div>
      <div class="result-card__signal">
        <div>信号最后更新：<strong>${signalChangedAt}</strong>（${signalAgeLabel}）</div>
        <div>自信号以来收益：<strong>${returnSinceSignal}</strong></div>
        <div>实时行情时间：<strong>${realtimeTimestamp}</strong></div>
      </div>
      ${driverMarkup}
      <table>
        <thead>
          <tr>
            <th>日期</th>
            <th>信号</th>
            <th>价格</th>
            <th>分数</th>
            <th>Prob Buy</th>
            <th>Prob Sell</th>
            <th>Prob Gap</th>
            <th>Confidence</th>
            <th>1 日</th>
            <th>5 日</th>
            <th>10 日</th>
          </tr>
        </thead>
        <tbody>
          ${rows.join("")}
        </tbody>
      </table>
      ${renderBacktest(tickerData.backtest)}
    </article>
  `;
}

function renderBacktest(backtest) {
  if (!backtest) {
    return "";
  }
  const { horizon, average_signal_return, hit_ratio, cumulative_return, trades } = backtest;
  return `
    <div class="backtest">
      <strong>回测</strong>：周期 ${horizon} 周 |
      平均收益 ${average_signal_return?.toFixed?.(3) ?? "--"} |
      命中率 ${(hit_ratio || 0).toFixed(2)} |
      累计收益 ${(cumulative_return || 0).toFixed(3)} |
      交易次数 ${trades ?? 0}
    </div>
  `;
}

function enrichResultItem(item, index = 0) {
  const latest = item.latest ?? {};
  const frequency = normalizeFrequency(latest.analysis_frequency ?? item.frequency ?? currentSearchFrequency);
  return {
    ...item,
    frequency,
    _decision: (latest.decision ?? "hold").toLowerCase(),
    _confidence: getConfidenceFromRow(latest),
    _probGap: typeof latest.prob_gap === "number" ? latest.prob_gap : 0,
    _probBuy: typeof latest.prob_buy === "number" ? latest.prob_buy : 0,
    _probSell: typeof latest.prob_sell === "number" ? latest.prob_sell : 0,
    _hitRatio: item.backtest?.hit_ratio ?? 0,
    _order: index,
  };
}

function groupResultsByTicker(items) {
  const map = new Map();
  items.forEach((item) => {
    const key = item.ticker;
    if (!map.has(key)) {
      map.set(key, []);
    }
    map.get(key).push(item);
  });
  return Array.from(map.entries()).map(([ticker, variations]) => ({
    ticker,
    variations,
    _order: Math.min(...variations.map((entry) => entry._order ?? 0)),
  }));
}

function sortResultItems(items, sortValue) {
  const cloned = [...items];
  switch (sortValue) {
    case "confidence_desc":
      return cloned.sort((a, b) => b._confidence - a._confidence);
    case "prob_gap_desc":
      return cloned.sort((a, b) => b._probGap - a._probGap);
    case "prob_buy_desc":
      return cloned.sort((a, b) => b._probBuy - a._probBuy);
    case "prob_sell_desc":
      return cloned.sort((a, b) => b._probSell - a._probSell);
    case "hit_ratio_desc":
      return cloned.sort((a, b) => (b._hitRatio ?? 0) - (a._hitRatio ?? 0));
    case "alpha_asc":
      return cloned.sort((a, b) => a.ticker.localeCompare(b.ticker));
    default:
      return cloned.sort((a, b) => (a._order ?? 0) - (b._order ?? 0));
  }
}

function computeGroupMetric(group, sortValue) {
  const metrics = group.variations.map((item) => ({
    confidence: item._confidence,
    probGap: item._probGap,
    probBuy: item._probBuy,
    probSell: item._probSell,
    hitRatio: item._hitRatio ?? 0,
  }));
  switch (sortValue) {
    case "confidence_desc":
      return Math.max(...metrics.map((m) => m.confidence));
    case "prob_gap_desc":
      return Math.max(...metrics.map((m) => m.probGap));
    case "prob_buy_desc":
      return Math.max(...metrics.map((m) => m.probBuy));
    case "prob_sell_desc":
      return Math.max(...metrics.map((m) => m.probSell));
    case "hit_ratio_desc":
      return Math.max(...metrics.map((m) => m.hitRatio));
    case "alpha_asc":
      return group.ticker;
    default:
      return group._order ?? 0;
  }
}

function sortResultGroups(groups, sortValue) {
  const cloned = [...groups];
  switch (sortValue) {
    case "alpha_asc":
      return cloned.sort((a, b) => a.ticker.localeCompare(b.ticker));
    case "confidence_desc":
    case "prob_gap_desc":
    case "prob_buy_desc":
    case "prob_sell_desc":
    case "hit_ratio_desc":
      return cloned.sort((a, b) => computeGroupMetric(b, sortValue) - computeGroupMetric(a, sortValue));
    default:
      return cloned.sort((a, b) => computeGroupMetric(a, sortValue) - computeGroupMetric(b, sortValue));
  }
}

function buildGroupMarkup(group) {
  const sorted = group.variations
    .slice()
    .sort(
      (a, b) =>
        (FREQUENCY_ORDER[a.frequency] ?? Number.MAX_SAFE_INTEGER) -
        (FREQUENCY_ORDER[b.frequency] ?? Number.MAX_SAFE_INTEGER),
    );
  const cards = sorted.map((item) => renderTable(item, 8)).join("");
  const summaryBadges = Array.from(new Set(sorted.map((item) => item.frequency))).map(
    (freq) => `<span class="freq-badge freq-${freq}">${frequencyLabel(freq)}</span>`,
  );
  return `
    <details class="result-card-group" open id="group-${group.ticker}">
      <summary>
        <span class="group-ticker">${group.ticker}</span>
        <span class="group-badges">${summaryBadges.join(" ")}</span>
        <span class="group-meta">共 ${group.variations.length} 条信号</span>
      </summary>
      <div class="result-card-group__body">${cards}</div>
    </details>
  `;
}

function renderPortfolioChatMessages() {
  if (!portfolioChatLog) {
    return;
  }
  const content = portfolioChatHistory
    .map((msg) => {
      const roleLabel = msg.role === "assistant" ? "AI" : "我";
      return `
        <div class="chat-message ${msg.role}">
          <div class="chat-message__avatar">${roleLabel}</div>
          <div class="chat-message__bubble">${escapeHtml(msg.content)}</div>
        </div>
      `;
    })
    .join("");
  portfolioChatLog.innerHTML = content || '<div class="chat-empty">尚未与 AI 沟通持仓，试着提问吧。</div>';
  portfolioChatLog.scrollTop = portfolioChatLog.scrollHeight;
}

function buildChatPayloadFromMeta(meta = {}, tickers = [], frequencies = []) {
  const requested = Array.isArray(meta.requested_tickers) && meta.requested_tickers.length
    ? meta.requested_tickers
    : tickers.map((item) => item.ticker).filter(Boolean);
  const freqList = frequencies.length
    ? frequencies.map((freq) => normalizeFrequency(freq))
    : [normalizeFrequency(meta.resample_frequency ?? "weekly")];
  return {
    tickers: requested,
    lookback_years: meta.lookback_years ?? 5,
    horizon: meta.horizon ?? 12,
    threshold: meta.threshold ?? 0.05,
    min_threshold: meta.min_threshold ?? 0.01,
    max_threshold: meta.max_threshold ?? 0.06,
    adaptive_threshold: meta.adaptive_threshold ?? true,
    resample_frequency: freqList[0],
    resample_frequencies: freqList,
    model_name: meta.model_name ?? "default_model",
    model_type: meta.model_type ?? "auto",
    train: false,
  };
}

function escapeHtml(str = "") {
  return str.replace(/[&<>"']/g, (ch) => {
    switch (ch) {
      case "&":
        return "&amp;";
      case "<":
        return "&lt;";
      case ">":
        return "&gt;";
      case '"':
        return "&quot;";
      case "'":
        return "&#39;";
      default:
        return ch;
    }
  });
}

function getActiveTickerToken(value = "") {
  const parts = value.split(",");
  const activeIndex = Math.max(parts.length - 1, 0);
  const activeRaw = parts[activeIndex] ?? "";
  return {
    parts,
    activeIndex,
    token: activeRaw.trim(),
  };
}

function hideTickerSuggestions() {
  if (!tickerSuggestions) {
    return;
  }
  tickerSuggestions.innerHTML = "";
  tickerSuggestions.classList.add("hidden");
  delete tickerSuggestions.dataset.hasResults;
}

function setTickerSuggestionsContent(content) {
  if (!tickerSuggestions) {
    return;
  }
  if (!content) {
    hideTickerSuggestions();
    return;
  }
  tickerSuggestions.innerHTML = `<div class="suggestions__panel">${content}</div>`;
  tickerSuggestions.classList.remove("hidden");
}

function renderTickerSuggestionResults(results, query) {
  if (!tickerSuggestions) {
    return;
  }
  if (!results?.length) {
    setTickerSuggestionsContent(
      `<div class="suggestions__item suggestions__item--empty">未找到与“${escapeHtml(query)}”匹配的代码</div>`
    );
    tickerSuggestions.dataset.hasResults = "0";
    return;
  }
  const items = results
    .map((item) => {
      const symbol = escapeHtml(item.symbol ?? item.display_symbol ?? "");
      const display = escapeHtml(item.display_symbol ?? item.symbol ?? "");
      const labelParts = [];
      if (item.short_name) {
        labelParts.push(escapeHtml(item.short_name));
      } else if (item.long_name) {
        labelParts.push(escapeHtml(item.long_name));
      }
      if (item.exchange) {
        labelParts.push(escapeHtml(item.exchange));
      }
      const meta = labelParts.join(" · ");
      return `
        <div class="suggestions__item" data-symbol="${symbol}">
          <span class="suggestions__item-symbol">${display}</span>
          <span class="suggestions__item-meta">${meta}</span>
        </div>
      `;
    })
    .join("");
  setTickerSuggestionsContent(items);
  tickerSuggestions.dataset.hasResults = "1";
}

function applyTickerSuggestion(symbol) {
  if (!tickerInput || !symbol) {
    hideTickerSuggestions();
    return;
  }
  const { parts } = getActiveTickerToken(tickerInput.value);
  const normalized = symbol.trim().toUpperCase();
  if (!normalized) {
    hideTickerSuggestions();
    return;
  }
  if (parts.length === 0) {
    parts.push(normalized);
  } else {
    parts[parts.length - 1] = normalized;
  }
  const sanitized = parts
    .map((part) => part.trim())
    .filter(Boolean);
  const unique = [];
  const seen = new Set();
  sanitized.forEach((item) => {
    if (!seen.has(item)) {
      seen.add(item);
      unique.push(item);
    }
  });
  tickerInput.value = unique.join(", ");
  hideTickerSuggestions();
  tickerInput.focus({ preventScroll: true });
}

async function requestTickerLookup(query) {
  if (!tickerSuggestions) {
    return;
  }
  const trimmed = query.trim();
  if (!trimmed) {
    hideTickerSuggestions();
    return;
  }
  const cacheKey = trimmed.toLowerCase();
  lastLookupQuery = cacheKey;
  if (lookupAbortController) {
    lookupAbortController.abort();
  }
  lookupAbortController = new AbortController();
  setTickerSuggestionsContent(
    `<div class="suggestions__item">正在搜索“${escapeHtml(trimmed)}”...</div>`
  );
  try {
    const response = await fetch(`/api/lookup?q=${encodeURIComponent(trimmed)}`, {
      signal: lookupAbortController.signal,
    });
    if (!response.ok) {
      let message = response.statusText || response.status;
      try {
        const errorData = await response.json();
        if (errorData?.detail) {
          message = errorData.detail;
        }
      } catch (_) {
        // ignore
      }
      throw new Error(`搜索失败：${message}`);
    }
    const data = await response.json();
    if (lastLookupQuery !== cacheKey) {
      return;
    }
    const results = Array.isArray(data?.results) ? data.results : [];
    renderTickerSuggestionResults(results, trimmed);
  } catch (error) {
    if (error?.name === "AbortError") {
      return;
    }
    setTickerSuggestionsContent(
      `<div class="suggestions__item suggestions__item--empty">${escapeHtml(
        error?.message || "搜索失败，请稍后重试。"
      )}</div>`
    );
    console.error(error);
  }
}

function scheduleTickerLookup(query) {
  if (lookupTimer) {
    clearTimeout(lookupTimer);
  }
  lookupTimer = setTimeout(() => requestTickerLookup(query), 260);
}

function handleTickerInputChange() {
  if (!tickerInput) {
    return;
  }
  const { token } = getActiveTickerToken(tickerInput.value);
  const trimmed = token.trim();
  if (!trimmed) {
    hideTickerSuggestions();
    return;
  }
  const asciiOnly = /^[0-9a-z.\-]+$/i.test(trimmed);
  if (asciiOnly && trimmed.length < 2) {
    hideTickerSuggestions();
    return;
  }
  scheduleTickerLookup(trimmed);
}

function renderChatMessages() {
  if (!chatLog) {
    return;
  }
  const content = chatHistory
    .map((msg) => {
      const roleLabel = msg.role === "assistant" ? "AI" : "我";
      return `
        <div class="chat-message ${msg.role}">
          <div class="chat-message__avatar">${roleLabel}</div>
          <div class="chat-message__bubble">${escapeHtml(msg.content)}</div>
        </div>
      `;
    })
    .join("");
  chatLog.innerHTML = content || '<div class="chat-empty">暂无对话，欢迎向 AI 提问。</div>';
  chatLog.scrollTop = chatLog.scrollHeight;
}

async function requestAnalysis(payload) {
  const response = await fetch("/api/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    let message = `分析失败：${response.status} ${response.statusText}`;
    try {
      const errData = await response.json();
      if (errData?.detail) {
        message = `分析失败：${errData.detail}`;
      }
    } catch (parseErr) {
      // ignore
    }
    throw new Error(message);
  }
  return response.json();
}

function annotateTickers(tickers, frequency) {
  const freq = normalizeFrequency(frequency);
  return (tickers ?? []).map((item) => ({
    ...item,
    frequency: freq,
  }));
}

function mergeSearchResultsForDisplay(frequencyMode) {
  if (frequencyMode === "combined") {
    const weekly = latestSearchResultsByFrequency.weekly ?? [];
    const daily = latestSearchResultsByFrequency.daily ?? [];
    return [...weekly, ...daily];
  }
  return latestSearchResultsByFrequency[normalizeFrequency(frequencyMode)] ?? [];
}

function updateSearchDisplay() {
  latestSearchResults = mergeSearchResultsForDisplay(currentSearchFrequency);
  renderSearchResults();
}

function updateSearchChatContext(frequencies) {
  const normalized = frequencies.map((freq) => normalizeFrequency(freq));
  const preferred = normalized.includes("daily") ? "daily" : normalized[0] ?? "weekly";
  latestSearchMeta = latestSearchMetaByFrequency[preferred] ?? null;
  return { preferred, normalized };
}

function buildSearchStatusText(summaries, preferredMeta) {
  if (!summaries.length) {
    return "未获取到分析结果。";
  }
  const segments = summaries.map(({ frequency, meta }) => {
    const missing = ensureArray(meta?.missing_tickers ?? []);
    const invalid = ensureArray(meta?.invalid_inputs ?? []);
    const base = `${frequencyLabel(frequency)}：${meta?.data_start ?? "--"} ~ ${meta?.data_end ?? "--"}`;
    const extra = [];
    if (missing.length) {
      extra.push(`缺失：${missing.join(", ")}`);
    }
    if (invalid.length) {
      extra.push(`无法识别：${invalid.join(", ")}`);
    }
    return extra.length ? `${base}（${extra.join("；")}）` : base;
  });
  const realtime = preferredMeta?.realtime_timestamp ?? "N/A";
  return `分析完成（${segments.join(" ｜ ")}，实时行情获取于 ${realtime}）`;
}

async function executeSearchAnalysis(basePayload, frequencyMode, { silent = false } = {}) {
  if (!basePayload || !Array.isArray(basePayload.tickers) || !basePayload.tickers.length) {
    searchStatus.textContent = "请输入至少一个股票代码";
    searchStatus.className = "status error";
    return;
  }

  const frequencies = frequencyMode === "combined" ? ["weekly", "daily"] : [frequencyMode];
  const normalizedFrequencies = frequencies.map((freq) => normalizeFrequency(freq));

  if (!silent) {
    searchStatus.textContent = "分析中，请稍候...";
    searchStatus.className = "status";
    setLoadingState(searchResultsContainer, true);
    if (resultControls) {
      resultControls.classList.add("hidden");
    }
  }

  latestSearchResultsByFrequency = {};
  latestSearchMetaByFrequency = {};

  const summaries = [];

  try {
    const payloads = normalizedFrequencies.map((freq) => ({ ...basePayload, resample_frequency: freq }));
    const responses = await Promise.all(payloads.map((payload) => requestAnalysis(payload)));
    responses.forEach((data, idx) => {
      const freq = normalizedFrequencies[idx];
      latestSearchResultsByFrequency[freq] = annotateTickers(data.tickers, freq);
      latestSearchMetaByFrequency[freq] = data.meta ?? {};
      summaries.push({ frequency: freq, meta: data.meta ?? {} });
    });

    currentSearchFrequency = frequencyMode;
    const { preferred: chatPreferredFrequency, normalized: normalizedForChat } = updateSearchChatContext(normalizedFrequencies);
    updateSearchDisplay();

    if (!latestSearchResults.length) {
      searchStatus.textContent = "未获取到分析结果。";
      searchStatus.className = "status error";
      latestSearchPayload = null;
      setLoadingState(searchResultsContainer, false);
      if (chatPanel) {
        chatPanel.classList.add("hidden");
        chatHistory = [];
        renderChatMessages();
      }
      return;
    }

    if (decisionFilter) {
      decisionFilter.value = "all";
    }
    if (sortSelect) {
      sortSelect.value = DEFAULT_SORT;
    }
    if (searchUpdatedAt) {
      searchUpdatedAt.textContent = `最后更新：${new Date().toLocaleString()}`;
    }

    const preferredMeta = latestSearchMeta ?? {};
    latestSearchMeta = preferredMeta;
    const chatFrequency = normalizeFrequency(preferredMeta?.resample_frequency ?? chatPreferredFrequency);
    const chatSource = latestSearchResultsByFrequency[chatFrequency] ?? [];
    latestSearchPayload = buildChatPayloadFromMeta(preferredMeta, chatSource, normalizedForChat);
    searchStatus.textContent = buildSearchStatusText(summaries, preferredMeta);
    searchStatus.className = "status success";

    if (chatPanel) {
      chatPanel.classList.remove("hidden");
      chatStatus.textContent = "已根据最新数据准备好，可以向 AI 咨询策略或风险。";
      chatStatus.className = "status";
      chatHistory = [];
      renderChatMessages();
      if (chatInput) {
        chatInput.value = "";
      }
    }
  } catch (error) {
    console.error(error);
    searchStatus.textContent = error.message || "分析过程中出现错误";
    searchStatus.className = "status error";
    latestSearchResultsByFrequency = {};
    latestSearchMetaByFrequency = {};
    latestSearchResults = [];
    renderSearchResults();
    if (resultControls) {
      resultControls.classList.add("hidden");
    }
    if (chatPanel) {
      chatPanel.classList.add("hidden");
      chatHistory = [];
      renderChatMessages();
    }
    latestSearchMeta = null;
    latestSearchPayload = null;
  } finally {
    if (!silent) {
      setLoadingState(searchResultsContainer, false);
    }
  }
}

function mergePortfolioResultsForDisplay(frequencyMode) {
  if (frequencyMode === "combined") {
    const weekly = latestPortfolioResultsByFrequency.weekly ?? [];
    const daily = latestPortfolioResultsByFrequency.daily ?? [];
    return [...weekly, ...daily];
  }
  return latestPortfolioResultsByFrequency[normalizeFrequency(frequencyMode)] ?? [];
}

function updatePortfolioDisplay() {
  latestPortfolioResults = mergePortfolioResultsForDisplay(currentPortfolioFrequency);
  renderPortfolioResults();
}

function updatePortfolioChatContext(frequencies) {
  const normalized = frequencies.map((freq) => normalizeFrequency(freq));
  const preferred = normalized.includes("daily") ? "daily" : normalized[0] ?? "weekly";
  latestPortfolioMeta = latestPortfolioMetaByFrequency[preferred] ?? null;
  const chatSource = latestPortfolioResultsByFrequency[preferred] ?? [];
  latestPortfolioPayload = buildChatPayloadFromMeta(latestPortfolioMeta ?? {}, chatSource, normalized);
}

function buildPortfolioStatusText(summaries) {
  if (!summaries.length) {
    return "未获取到持仓分析结果。";
  }
  const segments = summaries.map(({ frequency, meta }) => {
    const missing = ensureArray(meta?.missing_tickers ?? []);
    const invalid = ensureArray(meta?.invalid_inputs ?? []);
    const base = `${frequencyLabel(frequency)}：${meta?.data_start ?? "--"} ~ ${meta?.data_end ?? "--"}`;
    const extra = [];
    if (missing.length) {
      extra.push(`缺失：${missing.join(", ")}`);
    }
    if (invalid.length) {
      extra.push(`无法识别：${invalid.join(", ")}`);
    }
    return extra.length ? `${base}（${extra.join("；")}）` : base;
  });
  return `更新完成（${segments.join(" ｜ ")}）`;
}

async function analyzeTickers(event) {
  event.preventDefault();
  hideTickerSuggestions();
  const rawTickers = tickerInput ? tickerInput.value : "";
  const tickers = rawTickers
    .split(",")
    .map((t) => t.trim().toUpperCase())
    .filter(Boolean);
  if (!tickers.length) {
    searchStatus.textContent = "请输入至少一个股票代码";
    searchStatus.className = "status error";
    return;
  }

  const basePayload = {
    tickers,
    lookback_years: Number(document.getElementById("lookback-input").value) || 5,
    horizon: Number(document.getElementById("horizon-input").value) || 12,
    threshold: Number(document.getElementById("threshold-input").value) || 0.05,
    adaptive_threshold: document.getElementById("adaptive-checkbox").checked,
    train: false,
  };
  lastSearchBasePayload = { ...basePayload };
  const frequencyMode = frequencySelect?.value ?? "weekly";
  currentSearchFrequency = frequencyMode;
  await executeSearchAnalysis(basePayload, frequencyMode);
}

function renderSearchResults() {
  if (!searchResultsContainer) {
    return;
  }

  const enriched = latestSearchResults.map((item, index) => enrichResultItem(item, index));

  if (!enriched.length) {
    if (resultControls) {
      resultControls.classList.add("hidden");
    }
    searchResultsContainer.innerHTML = "<div class=\"empty-state\">尚未分析任何标的，或当前筛选条件下没有结果。</div>";
    return;
  }

  if (resultControls) {
    resultControls.classList.remove("hidden");
  }

  const filterValue = decisionFilter?.value ?? "all";
  const sortValue = sortSelect?.value ?? DEFAULT_SORT;
  const frequencyMode = currentSearchFrequency;

  if (frequencyMode === "combined") {
    let groups = groupResultsByTicker(enriched);
    if (filterValue !== "all") {
      groups = groups.filter((group) => group.variations.some((item) => item._decision === filterValue));
    }
    groups = sortResultGroups(groups, sortValue);
    if (!groups.length) {
      searchResultsContainer.innerHTML = "<div class=\"empty-state\">筛选条件下暂无标的。</div>";
      return;
    }
    searchResultsContainer.innerHTML = groups.map((group) => buildGroupMarkup(group)).join("");
    return;
  }

  let filtered = enriched;
  if (filterValue !== "all") {
    filtered = enriched.filter((item) => item._decision === filterValue);
  }
  filtered = sortResultItems(filtered, sortValue);

  if (!filtered.length) {
    searchResultsContainer.innerHTML = "<div class=\"empty-state\">筛选条件下暂无标的。</div>";
    return;
  }

  searchResultsContainer.innerHTML = filtered.map((ticker) => renderTable(ticker, 8)).join("");
}

function renderPortfolioResults() {
  if (!portfolioResultsContainer) {
    return;
  }

  const enriched = latestPortfolioResults.map((item, index) => enrichResultItem(item, index));

  if (!enriched.length) {
    if (portfolioControls) {
      portfolioControls.classList.add("hidden");
    }
    portfolioResultsContainer.innerHTML = '<div class="empty-state">暂无持仓信号，请检查 portfolio.json 配置。</div>';
    return;
  }

  if (portfolioControls) {
    portfolioControls.classList.remove("hidden");
  }

  const filterValue = portfolioFilterSelect?.value ?? "all";
  const sortValue = portfolioSortSelect?.value ?? DEFAULT_SORT;
  const frequencyMode = currentPortfolioFrequency;

  if (frequencyMode === "combined") {
    let groups = groupResultsByTicker(enriched);
    if (filterValue !== "all") {
      groups = groups.filter((group) => group.variations.some((item) => item._decision === filterValue));
    }
    groups = sortResultGroups(groups, sortValue);
    if (!groups.length) {
      portfolioResultsContainer.innerHTML = '<div class="empty-state">筛选条件下暂无持仓。</div>';
      return;
    }
    portfolioResultsContainer.innerHTML = groups.map((group) => buildGroupMarkup(group)).join("");
    return;
  }

  let filtered = enriched;
  if (filterValue !== "all") {
    filtered = enriched.filter((item) => item._decision === filterValue);
  }

  filtered = sortResultItems(filtered, sortValue);

  if (!filtered.length) {
    portfolioResultsContainer.innerHTML = '<div class="empty-state">筛选条件下暂无持仓。</div>';
    return;
  }

  portfolioResultsContainer.innerHTML = filtered.map((ticker) => renderTable(ticker, 5)).join("");
}

async function requestPortfolioAnalysis(frequency, { train = false } = {}) {
  const params = new URLSearchParams();
  if (frequency) {
    params.set("frequency", normalizeFrequency(frequency));
  }
  if (train) {
    params.set("train", "true");
  }
  const query = params.toString();
  const response = await fetch(`/api/portfolio${query ? `?${query}` : ""}`);
  if (!response.ok) {
    let message = `获取持仓数据失败：${response.status} ${response.statusText}`;
    try {
      const errData = await response.json();
      if (errData?.detail) {
        message = `获取持仓数据失败：${errData.detail}`;
      }
    } catch (parseErr) {
      // ignore
    }
    throw new Error(message);
  }
  return response.json();
}

async function loadPortfolio({ train = false } = {}) {
  if (portfolioStatus) {
    portfolioStatus.textContent = "刷新中...";
    portfolioStatus.className = "status";
  }
  setLoadingState(portfolioResultsContainer, true);

  const frequencyMode = portfolioFrequencySelect?.value ?? currentPortfolioFrequency ?? "weekly";
  currentPortfolioFrequency = frequencyMode;
  const frequencies = frequencyMode === "combined" ? ["weekly", "daily"] : [frequencyMode];
  const normalizedFrequencies = frequencies.map((freq) => normalizeFrequency(freq));

  latestPortfolioResultsByFrequency = {};
  latestPortfolioMetaByFrequency = {};

  const summaries = [];

  try {
    const tasks = normalizedFrequencies.map((freq) => requestPortfolioAnalysis(freq, { train }));
    const responses = await Promise.all(tasks);
    responses.forEach((data, idx) => {
      const freq = normalizedFrequencies[idx];
      latestPortfolioResultsByFrequency[freq] = annotateTickers(data.tickers, freq);
      latestPortfolioMetaByFrequency[freq] = data.meta ?? {};
      summaries.push({ frequency: freq, meta: data.meta ?? {} });
    });

    updatePortfolioChatContext(normalizedFrequencies);
    updatePortfolioDisplay();

    if (!latestPortfolioResults.length) {
      if (portfolioStatus) {
        portfolioStatus.textContent = "未获取到持仓信号，请检查配置。";
        portfolioStatus.className = "status error";
      }
      latestPortfolioMeta = latestPortfolioMetaByFrequency[normalizedFrequencies[0]] ?? null;
      latestPortfolioPayload = null;
      setLoadingState(portfolioResultsContainer, false);
      if (portfolioChatPanel) {
        portfolioChatPanel.classList.add("hidden");
        portfolioChatHistory = [];
        renderPortfolioChatMessages();
      }
      return;
    }

    if (portfolioFilterSelect) {
      portfolioFilterSelect.value = "all";
    }
    if (portfolioSortSelect) {
      portfolioSortSelect.value = DEFAULT_SORT;
    }

    if (portfolioStatus) {
      portfolioStatus.textContent = buildPortfolioStatusText(summaries);
      portfolioStatus.className = "status success";
    }

    if (portfolioChatPanel) {
      if (latestPortfolioResults.length) {
        portfolioChatPanel.classList.remove("hidden");
        if (portfolioChatStatus) {
          portfolioChatStatus.textContent = "AI 已根据最新持仓准备就绪，可以提问。";
          portfolioChatStatus.className = "status";
        }
        portfolioChatHistory = [];
        renderPortfolioChatMessages();
        if (portfolioChatInput) {
          portfolioChatInput.value = "";
        }
      } else {
        portfolioChatPanel.classList.add("hidden");
        portfolioChatHistory = [];
        renderPortfolioChatMessages();
      }
    }
  } catch (error) {
    console.error(error);
    if (portfolioStatus) {
      portfolioStatus.textContent = error.message || "获取持仓数据失败";
      portfolioStatus.className = "status error";
    }
    latestPortfolioResultsByFrequency = {};
    latestPortfolioMetaByFrequency = {};
    latestPortfolioResults = [];
    latestPortfolioMeta = null;
    latestPortfolioPayload = null;
    renderPortfolioResults();
    if (portfolioChatPanel) {
      portfolioChatPanel.classList.add("hidden");
      portfolioChatHistory = [];
      renderPortfolioChatMessages();
    }
  } finally {
    setLoadingState(portfolioResultsContainer, false);
  }
}

searchForm.addEventListener("submit", analyzeTickers);
if (refreshSearchBtn) {
  refreshSearchBtn.addEventListener("click", async () => {
    if (!lastSearchBasePayload) {
      searchStatus.textContent = "请先完成一次分析后再刷新。";
      searchStatus.className = "status error";
      return;
    }
    const frequencyMode = frequencySelect?.value ?? currentSearchFrequency;
    currentSearchFrequency = frequencyMode;
    await executeSearchAnalysis({ ...lastSearchBasePayload }, frequencyMode);
  });
}

if (decisionFilter) {
  decisionFilter.addEventListener("change", renderSearchResults);
}

if (sortSelect) {
  sortSelect.addEventListener("change", renderSearchResults);
}

if (frequencySelect) {
  frequencySelect.addEventListener("change", async () => {
    const frequencyMode = frequencySelect.value ?? "weekly";
    currentSearchFrequency = frequencyMode;
    if (!lastSearchBasePayload) {
      updateSearchDisplay();
      return;
    }
    await executeSearchAnalysis({ ...lastSearchBasePayload }, frequencyMode);
  });
}

quickPromptButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    const prompt = btn.dataset.prompt;
    if (!prompt || !chatInput) {
      return;
    }
    chatInput.value = prompt;
    chatInput.focus();
  });
});

portfolioQuickPromptButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    const prompt = btn.dataset.prompt;
    if (!prompt || !portfolioChatInput) {
      return;
    }
    portfolioChatInput.value = prompt;
    portfolioChatInput.focus();
  });
});

if (portfolioFilterSelect) {
  portfolioFilterSelect.addEventListener("change", renderPortfolioResults);
}

if (portfolioSortSelect) {
  portfolioSortSelect.addEventListener("change", renderPortfolioResults);
}

if (portfolioFrequencySelect) {
  portfolioFrequencySelect.addEventListener("change", () => {
    loadPortfolio();
  });
}

refreshPortfolioBtn.addEventListener("click", () => {
  loadPortfolio();
  setupPortfolioAutoRefresh();
});

if (chatForm) {
  chatForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!latestSearchPayload) {
      chatStatus.textContent = "请先完成一次分析后再提问";
      chatStatus.className = "status error";
      return;
    }
    const question = chatInput.value.trim();
    if (!question || chatSending) {
      return;
    }

    chatSending = true;
    const submitButton = chatForm.querySelector("button[type='submit']");
    if (submitButton) {
      submitButton.disabled = true;
    }

    chatHistory.push({ role: "user", content: question });
    renderChatMessages();
    chatInput.value = "";
    chatStatus.textContent = "AI 正在生成回答，请稍候...";
    chatStatus.className = "status";

    const historyForRequest = chatHistory.slice(0, -1);
    const requestPayload = {
      ...latestSearchPayload,
      question,
      bot_name: chatBotSelect?.value,
      history: historyForRequest,
    };

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestPayload),
      });
      if (!response.ok) {
        let message = `AI 请求失败：${response.status} ${response.statusText}`;
        try {
          const errData = await response.json();
          if (errData?.detail) {
            message = `AI 请求失败：${errData.detail}`;
          }
        } catch (parseErr) {
          // ignore
        }
        throw new Error(message);
      }
      const data = await response.json();
      const answer = data.answer || "(AI 未返回内容)";
      chatHistory.push({ role: "assistant", content: answer });
      renderChatMessages();
      chatStatus.textContent = `回复来自 ${data.bot_name ?? "AI 助手"}`;
      chatStatus.className = "status success";

      if (data.analysis?.tickers) {
        const analysis = data.analysis;
        const freq = normalizeFrequency(analysis.meta?.resample_frequency ?? currentSearchFrequency);
        latestSearchResultsByFrequency[freq] = annotateTickers(analysis.tickers, freq);
        latestSearchMetaByFrequency[freq] = analysis.meta ?? {};
        const summaries = [{ frequency: freq, meta: analysis.meta ?? {} }];
        currentSearchFrequency = currentSearchFrequency === "combined" ? currentSearchFrequency : freq;
        const { normalized: normalizedForChat } = updateSearchChatContext([freq]);
        updateSearchDisplay();
        if (decisionFilter) {
          decisionFilter.value = "all";
        }
        if (sortSelect) {
          sortSelect.value = DEFAULT_SORT;
        }
        if (searchUpdatedAt) {
          searchUpdatedAt.textContent = `最后更新：${new Date().toLocaleString()}`;
        }
        const preferredMeta = latestSearchMeta ?? {};
        latestSearchPayload = buildChatPayloadFromMeta(
          preferredMeta,
          latestSearchResultsByFrequency[freq] ?? [],
          normalizedForChat,
        );
        searchStatus.textContent = buildSearchStatusText(summaries, preferredMeta);
        searchStatus.className = "status success";
        chatStatus.textContent = `AI 已自动刷新至 ${frequencyLabel(freq)} 信号 — 回复来自 ${data.bot_name ?? "AI 助手"}`;
      }
    } catch (error) {
      console.error(error);
      const message = error.message || "AI 回答失败";
      chatStatus.textContent = message;
      chatStatus.className = "status error";
      chatHistory.push({ role: "assistant", content: message });
      renderChatMessages();
    } finally {
      chatSending = false;
      if (submitButton) {
        submitButton.disabled = false;
      }
    }
  });
}

if (portfolioChatForm) {
  portfolioChatForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!latestPortfolioPayload) {
      if (portfolioChatStatus) {
        portfolioChatStatus.textContent = "请先刷新持仓数据后再提问。";
        portfolioChatStatus.className = "status error";
      }
      return;
    }
    const question = portfolioChatInput.value.trim();
    if (!question || portfolioChatSending) {
      return;
    }

    portfolioChatSending = true;
    const submitButton = portfolioChatForm.querySelector("button[type='submit']");
    if (submitButton) {
      submitButton.disabled = true;
    }

    portfolioChatHistory.push({ role: "user", content: question });
    renderPortfolioChatMessages();
    portfolioChatInput.value = "";
    if (portfolioChatStatus) {
      portfolioChatStatus.textContent = "AI 正在分析持仓，请稍候...";
      portfolioChatStatus.className = "status";
    }

    const historyForRequest = portfolioChatHistory.slice(0, -1);
    const requestPayload = {
      ...latestPortfolioPayload,
      question,
      bot_name: portfolioChatBotSelect?.value,
      history: historyForRequest,
    };

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestPayload),
      });
      if (!response.ok) {
        let message = `AI 请求失败：${response.status} ${response.statusText}`;
        try {
          const errData = await response.json();
          if (errData?.detail) {
            message = `AI 请求失败：${errData.detail}`;
          }
        } catch (_) {
          // ignore
        }
        throw new Error(message);
      }
      const data = await response.json();
      const answer = data.answer || "(AI 未返回内容)";
      portfolioChatHistory.push({ role: "assistant", content: answer });
      renderPortfolioChatMessages();
      if (portfolioChatStatus) {
        portfolioChatStatus.textContent = `回复来自 ${data.bot_name ?? "AI 助手"}`;
        portfolioChatStatus.className = "status success";
      }

      if (data.analysis?.tickers) {
        const analysis = data.analysis;
        const freq = normalizeFrequency(analysis.meta?.resample_frequency ?? currentPortfolioFrequency);
        latestPortfolioResultsByFrequency[freq] = annotateTickers(analysis.tickers, freq);
        latestPortfolioMetaByFrequency[freq] = analysis.meta ?? {};
        const summaries = [{ frequency: freq, meta: analysis.meta ?? {} }];
        currentPortfolioFrequency = currentPortfolioFrequency === "combined" ? currentPortfolioFrequency : freq;
        updatePortfolioChatContext([freq]);
        updatePortfolioDisplay();
        if (portfolioFilterSelect) {
          portfolioFilterSelect.value = "all";
        }
        if (portfolioSortSelect) {
          portfolioSortSelect.value = DEFAULT_SORT;
        }
        if (portfolioStatus) {
          portfolioStatus.textContent = buildPortfolioStatusText(summaries);
          portfolioStatus.className = "status success";
        }
      }
    } catch (error) {
      console.error(error);
      const message = error.message || "AI 回答失败";
      portfolioChatHistory.push({ role: "assistant", content: message });
      renderPortfolioChatMessages();
      if (portfolioChatStatus) {
        portfolioChatStatus.textContent = message;
        portfolioChatStatus.className = "status error";
      }
    } finally {
      portfolioChatSending = false;
      if (submitButton) {
        submitButton.disabled = false;
      }
    }
  });
}

// 默认展示即时信号
switchView("search");
renderSearchResults();
renderPortfolioResults();
loadPortfolio();
