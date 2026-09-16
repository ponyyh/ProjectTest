"""轻量级中英文文本情感分析与关键词提取。"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

import pandas as pd

POSITIVE_WORDS = {
    "好": 1.0, "满意": 1.0, "喜欢": 0.9, "推荐": 0.8, "成功": 0.8,
    "惊喜": 0.8, "专业": 0.6, "优秀": 1.0, "很好": 0.9, "great": 0.8,
    "good": 0.7, "love": 1.0, "excellent": 1.0, "happy": 0.8,
}
NEGATIVE_WORDS = {
    "差": -1.0, "失望": -0.9, "问题": -0.5, "担心": -0.6, "争议": -0.6,
    "慢": -0.5, "失败": -1.0, "糟糕": -1.0, "投诉": -0.8, "bad": -0.8,
    "poor": -0.8, "hate": -1.0, "angry": -0.8, "disappointed": -0.9,
}
STOPWORDS = {"的", "了", "是", "和", "在", "有", "都", "就", "也", "将", "目前", "官方", "用户"}


class SentimentAnalyzer:
    """可解释、零模型文件依赖的情感分析器。"""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        nlp_config = (config or {}).get("nlp", {})
        self.positive_threshold = float(nlp_config.get("positive_threshold", 0.15))
        self.negative_threshold = float(nlp_config.get("negative_threshold", -0.15))

    def score(self, text: str) -> float:
        tokens = self._tokens(text)
        if not tokens:
            return 0.0
        raw = sum(POSITIVE_WORDS.get(token, 0) + NEGATIVE_WORDS.get(token, 0) for token in tokens)
        return max(-1.0, min(1.0, raw / max(2.0, len(tokens) / 2)))

    def label(self, score: float) -> str:
        if score >= self.positive_threshold:
            return "positive"
        if score <= self.negative_threshold:
            return "negative"
        return "neutral"

    def analyze(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        result["sentiment_score"] = result["text"].fillna("").map(self.score)
        result["sentiment"] = result["sentiment_score"].map(self.label)
        return result

    def keywords(self, texts: list[str] | pd.Series, top_n: int = 15) -> list[tuple[str, int]]:
        counts: Counter[str] = Counter()
        for text in texts:
            for token in self._tokens(str(text)):
                if token not in STOPWORDS and len(token) > 1 and not token.isdigit():
                    counts[token] += 1
        ranked = sorted(counts.items(), key=lambda item: (-item[1], -len(item[0]), item[0]))
        return ranked[:top_n]

    @staticmethod
    def _tokens(text: str) -> list[str]:
        tokens: list[str] = []
        for segment in re.findall(r"[a-zA-Z][a-zA-Z'-]+|[\u4e00-\u9fff]+", text.lower()):
            if re.fullmatch(r"[a-zA-Z][a-zA-Z'-]+", segment):
                tokens.append(segment)
                continue
            # 保留单字以匹配“好/差”等情感词，并增加二元、三元词组用于关键词统计。
            tokens.extend(segment)
            tokens.extend(segment[index:index + size] for size in (2, 3) for index in range(len(segment) - size + 1))
        return tokens
