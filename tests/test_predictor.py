import pandas as pd

from src.predictor import forecast_heat, hourly_metrics


def test_hourly_metrics_and_forecast() -> None:
    frame = pd.DataFrame({
        "post_id": ["1", "2", "3", "4"],
        "published_at": pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC"),
        "engagement": [1, 2, 3, 4],
        "sentiment": ["positive", "negative", "neutral", "negative"],
        "sentiment_score": [0.5, -0.5, 0, -0.5],
    })
    metrics = hourly_metrics(frame)
    result = forecast_heat(metrics, horizon=3)
    assert len(metrics) == 4
    assert len(result) == 7
    assert set(result["kind"]) == {"history", "forecast"}

