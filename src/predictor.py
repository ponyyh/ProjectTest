"""热度趋势聚合与未来 24 小时基线预测。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def hourly_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """按小时聚合帖子量、互动量和负面比例。"""
    columns = ["hour", "post_count", "engagement", "negative_ratio", "sentiment_score"]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    data = frame.copy()
    data["published_at"] = pd.to_datetime(data["published_at"], utc=True)
    data["hour"] = data["published_at"].dt.floor("h")
    data["is_negative"] = data["sentiment"].eq("negative").astype(int)
    return (
        data.groupby("hour", as_index=False)
        .agg(
            post_count=("post_id", "count"), engagement=("engagement", "sum"),
            negative_ratio=("is_negative", "mean"), sentiment_score=("sentiment_score", "mean"),
        )
        .sort_values("hour")
    )


def forecast_heat(metrics: pd.DataFrame, horizon: int = 24) -> pd.DataFrame:
    """使用线性回归趋势预测热度，返回历史与预测数据。"""
    if metrics.empty:
        return pd.DataFrame(columns=["hour", "post_count", "kind"])
    history = metrics[["hour", "post_count"]].copy()
    history["kind"] = "history"
    values = history["post_count"].to_numpy(dtype=float)
    window = min(len(values), 24)
    x = np.arange(window, dtype=float)
    y = values[-window:]
    slope, intercept = np.polyfit(x, y, 1) if window >= 2 else (0.0, y[0])
    last_hour = pd.to_datetime(history["hour"].iloc[-1])
    future_x = np.arange(window, window + horizon, dtype=float)
    predictions = np.maximum(0, slope * future_x + intercept)
    future = pd.DataFrame({
        "hour": [last_hour + pd.Timedelta(hours=i) for i in range(1, horizon + 1)],
        "post_count": np.round(predictions, 2), "kind": "forecast",
    })
    return pd.concat([history, future], ignore_index=True)

