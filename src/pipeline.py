"""端到端运行入口：采集 -> NLP -> 趋势预测 -> 文件输出。"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .nlp_engine import SentimentAnalyzer
from .predictor import forecast_heat, hourly_metrics
from .scraper import SocialMediaCollector
from .utils import PROJECT_ROOT, ensure_project_dirs, load_config


def run_pipeline(
    keyword: str,
    hours: int = 48,
    mode: str | None = None,
    config_path: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """运行采集、情感分析和预测，并返回三个阶段的数据。"""
    if not keyword or not keyword.strip():
        raise ValueError("keyword 不能为空")
    if hours <= 0:
        raise ValueError("hours 必须大于 0")
    config = load_config(config_path)
    ensure_project_dirs()
    selected_mode = mode or config.get("collector", {}).get("default_mode", "mock")
    raw = SocialMediaCollector(config=config).collect(keyword, hours, selected_mode)
    processed = SentimentAnalyzer(config).analyze(raw)
    processed.to_csv(PROJECT_ROOT / "data/03_processed/sentiment_posts.csv", index=False)
    metrics = hourly_metrics(processed)
    forecast = forecast_heat(metrics, int(config.get("forecast", {}).get("horizon_hours", 24)))
    metrics.to_csv(PROJECT_ROOT / "reports/tables/hourly_metrics.csv", index=False)
    forecast.to_csv(PROJECT_ROOT / "reports/tables/hourly_forecast.csv", index=False)
    return {"processed": processed, "metrics": metrics, "forecast": forecast}


def main() -> None:
    parser = argparse.ArgumentParser(description="运行社交媒体情绪分析流水线")
    parser.add_argument("--keyword", default="新能源汽车")
    parser.add_argument("--hours", type=int, default=48)
    parser.add_argument("--mode", choices=["mock", "api"], default=None)
    parser.add_argument("--config", default=None, help="可选的 YAML 配置文件路径")
    args = parser.parse_args()
    result = run_pipeline(args.keyword, args.hours, args.mode, args.config)
    print(f"已处理 {len(result['processed'])} 条帖子，输出目录：{Path(PROJECT_ROOT) / 'reports'}")


if __name__ == "__main__":
    main()
