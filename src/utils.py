"""项目公共工具：路径、配置和数据目录管理。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - 仅用于极简离线环境
    yaml = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CONFIG: dict[str, Any] = {
    "project_name": "social-sentiment-monitor",
    "timezone": "Asia/Shanghai",
    "collector": {
        "default_mode": "mock",
        "default_hours": 48,
        "posts_per_hour": 24,
        "sources": ["微博", "新闻评论", "视频平台", "论坛"],
    },
    "nlp": {"positive_threshold": 0.15, "negative_threshold": -0.15, "top_keywords": 15},
    "forecast": {"horizon_hours": 24, "history_hours": 48, "negative_alert_ratio": 0.45, "heat_alert_multiplier": 1.8},
}


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """读取 YAML 配置，并允许用环境变量覆盖 Redis 地址。"""
    config_path = Path(path) if path else PROJECT_ROOT / "config" / "config.yaml"
    if yaml is None:
        config = DEFAULT_CONFIG.copy()
    else:
        with config_path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}
    if os.getenv("REDIS_URL"):
        config.setdefault("collector", {})["redis_url"] = os.environ["REDIS_URL"]
    return config


def ensure_project_dirs() -> None:
    """确保运行时输出目录存在。"""
    for relative_path in (
        "data/01_raw", "data/02_intermediate", "data/03_processed",
        "models", "reports/figures", "reports/tables",
    ):
        (PROJECT_ROOT / relative_path).mkdir(parents=True, exist_ok=True)
