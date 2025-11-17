#!/usr/bin/env python3
"""
快速训练 portfolio.json 中配置的股票模型
专用于 Render 部署时的构建阶段
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add parent directory to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.core import run_analysis

def main():
    """读取 portfolio.json 并快速训练需要的模型"""
    portfolio_path = project_root / "portfolio.json"
    
    if not portfolio_path.exists():
        print("⚠️ portfolio.json 不存在,跳过训练")
        return
    
    try:
        with portfolio_path.open("r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception as exc:
        print(f"❌ 读取 portfolio.json 失败: {exc}")
        return
    
    tickers = config.get("tickers", [])
    if not tickers:
        print("⚠️ portfolio.json 中没有股票代码,跳过训练")
        return
    
    options = config.get("options", {})
    
    print(f"🚀 开始训练模型 for: {tickers}")
    print(f"📋 配置: lookback_years={options.get('lookback_years', 5)}, "
          f"horizon={options.get('horizon', 12)}, "
          f"frequency={options.get('resample_frequency', 'weekly')}")
    
    try:
        run_analysis(
            tickers=tickers,
            lookback_years=options.get("lookback_years", 5),
            horizon=options.get("horizon", 12),
            threshold=options.get("threshold", 0.05),
            adaptive_threshold=options.get("adaptive_threshold", True),
            min_threshold=options.get("min_threshold", 0.01),
            max_threshold=options.get("max_threshold", 0.06),
            resample_frequency=options.get("resample_frequency", "weekly"),
            model_name=options.get("model_name", "default_model"),
            model_type=options.get("model_type", "auto"),
            train=True,  # 强制训练
            console=None,
            include_briefing=False,  # 跳过简报生成以节省时间
        )
        print("✅ 模型训练完成")
    except Exception as exc:
        print(f"❌ 模型训练失败: {exc}")
        print("⚠️ 警告:模型将在首次请求时自动训练")

if __name__ == "__main__":
    main()

