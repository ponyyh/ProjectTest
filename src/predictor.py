"""热度趋势聚合与未来 24 小时基线预测。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def hourly_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """按小时聚合帖子量、互动量和负面比例。"""
    columns = ["hour", "post_count", "engagement", "negative_ratio", "sentiment_score"]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    required = {"post_id", "published_at", "engagement", "sentiment", "sentiment_score"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"输入数据缺少字段: {', '.join(sorted(missing))}")
    data = frame.copy()
    data["published_at"] = pd.to_datetime(data["published_at"], utc=True)
    data["hour"] = data["published_at"].dt.floor("h")
    data["is_negative"] = data["sentiment"].eq("negative").astype(int)
    grouped = (
        data.groupby("hour", as_index=False)
        .agg(
            post_count=("post_id", "count"), engagement=("engagement", "sum"),
            negative_ratio=("is_negative", "mean"), sentiment_score=("sentiment_score", "mean"),
        )
        .sort_values("hour")
    )
    # 补齐没有帖子的小时时段，避免折线图把“无数据”误读成“未监控”。
    all_hours = pd.date_range(grouped["hour"].min(), grouped["hour"].max(), freq="h", tz="UTC")
    result = grouped.set_index("hour").reindex(all_hours)
    result.index.name = "hour"
    result = result.reset_index()
    result["post_count"] = result["post_count"].fillna(0).astype(int)
    result["engagement"] = result["engagement"].fillna(0).astype(int)
    result["negative_ratio"] = result["negative_ratio"].fillna(0.0)
    result["sentiment_score"] = result["sentiment_score"].fillna(0.0)
    return result[columns]


def forecast_heat(metrics: pd.DataFrame, horizon: int = 24) -> pd.DataFrame:
    """使用线性回归趋势预测热度，返回历史与预测数据。"""
    if horizon <= 0:
        raise ValueError("horizon 必须大于 0")
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
