"""社交媒体数据采集、标准化和去重模块。"""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

import pandas as pd

from .utils import PROJECT_ROOT, ensure_project_dirs, load_config

POSITIVE_TEXTS = (
    "体验很好，服务专业，值得推荐", "新品表现惊喜，大家反馈非常满意",
    "解决问题很快，效率和质量都不错", "这次活动很成功，用户评价持续上升",
)
NEGATIVE_TEXTS = (
    "体验很差，问题一直没有解决", "大量用户反馈失望，希望尽快改进",
    "服务响应太慢，影响了正常使用", "质量问题引发争议，大家都在担心",
)
NEUTRAL_TEXTS = (
    "官方发布了最新进展和说明", "目前正在收集用户反馈，后续会同步结果",
    "相关话题登上热榜，评论数量持续增加", "产品将在本周更新，具体时间以公告为准",
)


@dataclass
class SocialMediaCollector:
    """采集器，默认使用可重复的模拟数据。"""

    config: dict[str, Any] | None = None
    seed: int = 42

    def __post_init__(self) -> None:
        self.config = self.config or load_config()
        self._seen_urls: set[str] = set()
        self._random = random.Random(self.seed)

    def fetch_from_api(self, keyword: str, hours: int) -> list[dict[str, Any]]:
        """真实 API 接入占位，需按目标平台授权和分页协议实现。"""
        raise NotImplementedError(
            "请在 SocialMediaCollector.fetch_from_api 中接入目标平台官方 API。"
        )

    def generate_mock_posts(self, keyword: str, hours: int = 48) -> list[dict[str, Any]]:
        """生成用于本地演示的模拟帖子。"""
        collector_config = self.config.get("collector", {})
        sources = collector_config.get("sources", ["微博", "新闻评论", "论坛"])
        per_hour = int(collector_config.get("posts_per_hour", 24))
        now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        rows: list[dict[str, Any]] = []
        for hour_offset in range(hours, 0, -1):
            timestamp = now - timedelta(hours=hour_offset)
            wave = max(0, int((hours - hour_offset) / max(hours, 1) * 10))
            count = max(1, per_hour + self._random.randint(-5, 5) + wave)
            for index in range(count):
                sentiment_bucket = self._random.choices(
                    ["positive", "neutral", "negative"], weights=[35, 38, 27]
                )[0]
                text_pool = {
                    "positive": POSITIVE_TEXTS,
                    "neutral": NEUTRAL_TEXTS,
                    "negative": NEGATIVE_TEXTS,
                }[sentiment_bucket]
                text = f"{keyword}：{self._random.choice(text_pool)}"
                post_id = f"{timestamp.isoformat()}-{index}-{self._random.randint(1000, 9999)}"
                rows.append({
                    "source": self._random.choice(sources),
                    "author": f"user_{self._random.randint(10000, 99999)}",
                    "text": text,
                    "url": f"https://example.com/posts/{hashlib.md5(post_id.encode()).hexdigest()}",
                    "published_at": timestamp + timedelta(minutes=self._random.randint(0, 59)),
                    "engagement": self._random.randint(1, 5000),
                })
        return rows

    def collect(self, keyword: str, hours: int = 48, mode: str = "mock") -> pd.DataFrame:
        """采集并返回去重、清洗后的标准化 DataFrame。"""
        raw_rows = self.generate_mock_posts(keyword, hours) if mode == "mock" else self.fetch_from_api(keyword, hours)
        frame = self._normalize(raw_rows)
        ensure_project_dirs()
        frame.to_csv(PROJECT_ROOT / "data/01_raw" / "social_posts.csv", index=False)
        return frame

    def _normalize(self, rows: Iterable[dict[str, Any]]) -> pd.DataFrame:
        normalized: list[dict[str, Any]] = []
        for row in rows:
            url = str(row.get("url", "")).strip()
            text = re.sub(r"\s+", " ", str(row.get("text", ""))).strip()
            if not text or not url or url in self._seen_urls:
                continue
            self._seen_urls.add(url)
            published_at = pd.to_datetime(row.get("published_at"), utc=True, errors="coerce")
            if pd.isna(published_at):
                continue
            normalized.append({
                "post_id": hashlib.sha1(url.encode("utf-8")).hexdigest()[:16],
                "source": str(row.get("source", "unknown")),
                "author": str(row.get("author", "anonymous")),
                "text": text,
                "url": url,
                "published_at": published_at,
                "engagement": max(0, int(row.get("engagement", 0))),
            })
        return pd.DataFrame(normalized, columns=[
            "post_id", "source", "author", "text", "url", "published_at", "engagement"
        ])


def collect_posts(keyword: str, hours: int = 48, mode: str = "mock") -> pd.DataFrame:
    """便捷采集入口。"""
    return SocialMediaCollector().collect(keyword, hours, mode)

