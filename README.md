# Stock AI Analyzer

一个用于分析美国股票持仓买入与卖出时机的原型工具包，结合 AI 信号和技术指标，为投资者提供可读性强的决策参考。

## 功能特性

- 下载一个或多个股票代码的历史 OHLCV 数据。
- 使用经典技术指标结合动量、波动率、ATR、自适应阈值、成交量异常与基准相对强弱等辅助特征构建数据集，支持日/周频分析。
- 新增宏观因子（VIX、美元指数、十年期美债收益率等）作为市场环境特征，提升跨市场泛化能力。
- 增补趋势强度、均线斜率、价格 Z-Score、布林带偏移、短周期收益等派生特征，兼顾趋势行情与震荡行情的信号稳定性。
- 训练轻量级机器学习分类模型，以预测短周期收益表现。
- 支持自动模型选择，在 HistGradientBoosting 与 RandomForest 等候选模型中基于验证集指标择优。
- 将模型输出与基于规则的技术指标评分融合，生成直观的买入 / 持有 / 卖出信号。
- 输出简明报告，总结信心水平、支撑指标以及近期表现。
- Web 控制台支持日/周频切换及并排展示，强调短期收益、概率差、临界风险提示，并输出关键驱动因子与回测表现。
- 引入时间序列交叉验证与梯度提升模型（HistGradientBoosting）评估信号质量，并提供基础回测结果接口。

## 为什么要做这个工具？

许多开源项目聚焦回测或单指标策略，而本原型强调：

- **以持仓为中心**：输入关注的股票代码，即刻获得可执行的信号。
- **可解释性**：AI 预测结合易读的指标背景，便于理解信号来源。
- **可扩展性**：可接入另类数据、升级模型，或对接通知系统。

## 快速开始

1. 激活仓库随附的虚拟环境：
   ```bash
   source .venv/bin/activate  # Windows 用户运行 .\.venv\Scripts\activate
   ```

   若需重新创建虚拟环境，可执行 `python -m venv .venv` 后再激活。
2. 通过 `requirements.txt` 安装依赖（首次运行或依赖更新时）：
   > 建议保持 `scikit-learn` 版本为 `1.6.1`（与预训练模型一致）。若使用 `pip`, 可执行 `pip install -r requirements.txt`，如需重新指定版本可在安装后运行 `pip install scikit-learn==1.6.1` 以避免模型反序列化告警。
3. 使用 CLI 分析股票代码（首次运行会训练模型并缓存历史行情）：
  ```bash
  python -m src.cli --tickers AAPL MSFT NVDA --lookback-years 5 --train --resample-frequency weekly --horizon 12 --threshold 0.06 --adaptive-threshold --export-signals --summary-report
   ```

   - `--adaptive-threshold`：利用 ATR 调整买卖标签的收益阈值，兼顾高/低波动标的。
   - `--export-signals`：将信号明细导出到 `reports/` 目录，便于后续回测或人工复盘。
   - `--resample-frequency weekly`：以周为单位聚合 OHLCV，更贴合长期加减仓节奏；`--horizon 12` 即预测约 12 周后的收益。
   - `--threshold` / `--min-threshold` / `--max-threshold`：控制买卖阈值区间，适配不同持仓目标。
   - `--summary-report`：额外生成 Markdown 总览（默认命名为 `reports/summary_时间戳.md`），可打印成简洁报告。
   - 支持 A 股 6 位代码（如 `600519`、`000001`）和美股/港股等带交易所后缀的代码；若输入公司名称请改用对应的交易所代码，也可在 `ticker_aliases.json` 中维护中文/英文别名。
4. 后续运行会复用已训练模型和缓存数据，除非再次传入 `--train`。
5. 查看终端中的分析摘要与近期信号表；训练阶段默认采用时间顺序切分验证集，避免未来数据泄露。CSV/JSON 导出与提醒功能仍在规划中。
6. 启动 Web 控制台获取交互式图形界面：
   ```bash
   uvicorn app_server:app --reload
   ```

   打开 [http://localhost:8000](http://localhost:8000)，即可使用“即时信号”搜索单支股票或在“持仓监控”页面实时查看 `portfolio.json` 中维护的持仓信号；前端支持日/周频切换、折叠对比、信心/概率差排序及临界风险提示，也可快速查询“今天该做什么”。若输入中文或别名需要映射，可编辑 `ticker_aliases.json` 添加自定义映射（例如 `"中国联通": "600050.SS"`）。

### 训练日频模型

默认仓库仅携带周频示例模型，如需充分利用日线分析，可通过 CLI 重新训练：

```bash
python -m src.cli \
  --tickers AAPL MSFT NVDA \
  --lookback-years 5 \
  --resample-frequency daily \
  --model-name default_model_daily \
  --horizon 12 --threshold 0.05 --adaptive-threshold --train
```

训练完成后会生成 `models/default_model_daily.joblib`，后端会在请求日频信号时自动加载该模型。

## 进阶分析

- **更广泛的回归测试**：对更大持仓集合执行 `--train` 可以验证模型参数选择是否稳定，例：
  ```bash
  .venv/bin/python -m src.cli \
    --tickers 600050.SS AAPL TSLL NVDA VTI ROKU BB \
    --lookback-years 8 --resample-frequency weekly \
    --horizon 12 --threshold 0.065 --adaptive-threshold --train
  ```
- **特征重要度**：使用 `scripts/permutation_importance.py` 调用 sklearn permutation importance（默认保存至 `reports/permutation_importance.csv`），便于洞察混合特征的贡献。
- **批量训练**：通过 `python scripts/train_all.py` 一次性刷新美股/中概/A 股等预设组合的模型，脚本同时输出多组模型（默认/成长/红利等）的分类指标、OOF 验证结果与回测统计到 `reports/model_metrics_latest.json`，并在 `reports/model_metrics_history.jsonl` 中累计留档，便于纵向对比。
- **集成通知**：CLI 的 `--export-signals` 与 `--summary-report` 生成的 CSV/Markdown 已存在于 `reports/` 目录，可按需接入 Webhook 或自动化发布流程，无需重复计算信号。

## 规划路线

- [X] 设计整体架构与项目结构。
- [X] 实现数据抓取与缓存模块。
- [X] 完成基于技术指标与标签的特征工程。
- [X] 构建 AI + 指标混合信号生成器。
- [X] 提供用于批量分析与报告的 CLI。
- [ ] 输出信号到 CSV/JSON 并集成通知渠道。
- [ ] 试验更高级的 AI 模型（Transformer、强化学习等）。

## 免责声明

本工具仅用于研究与教育目的，不构成投资建议。在做出任何交易决策前，请务必自行进行充分调研。
# stock-ai-analyzer
