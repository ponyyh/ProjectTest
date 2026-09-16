"""轻量级中英文文本情感分析与关键词提取。

该模块不依赖模型文件，适合本地演示和离线运行。生产环境可以保留同样的
``analyze`` 接口，替换成 BERT/RoBERTa 推理实现。
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Iterable

import pandas as pd

POSITIVE_WORDS = {
    "好": 1.0, "满意": 1.0, "喜欢": 0.9, "推荐": 0.8, "成功": 0.8,
    "惊喜": 0.8, "专业": 0.6, "优秀": 1.0, "很好": 0.9,
    "great": 0.8, "good": 0.7, "love": 1.0, "excellent": 1.0, "happy": 0.8,
}
NEGATIVE_WORDS = {
    "差": -1.0, "失望": -0.9, "问题": -0.5, "担心": -0.6, "争议": -0.6,
    "慢": -0.5, "失败": -1.0, "糟糕": -1.0, "投诉": -0.8,
    "bad": -0.8, "poor": -0.8, "hate": -1.0, "angry": -0.8, "disappointed": -0.9,
}
NEGATION_WORDS = {"不", "没", "没有", "未", "别", "不是", "not", "never", "no"}
STOPWORDS = {
    "的", "了", "是", "和", "在", "有", "都", "就", "也", "将", "目前", "官方", "用户",
    "这次", "大家", "希望", "后续", "相关", "具体", "一个", "非常",
}

# 没有引入 jieba，使用小型领域词典做最长匹配；未命中的连续中文则回退为二字词。
CHINESE_TERMS = {
    "新能源汽车", "新能源", "产品", "服务", "体验", "很好", "满意", "喜欢", "推荐",
    "成功", "惊喜", "专业", "优秀", "新品", "表现", "反馈", "解决问题", "解决",
    "很快", "效率", "质量", "不错", "活动", "评价", "持续", "上升", "差", "问题",
    "一直", "没有", "大量", "失望", "尽快", "改进", "响应", "太慢", "影响", "正常使用",
    "引发", "争议", "担心", "发布", "最新进展", "说明", "收集", "同步", "结果",
    "话题", "热榜", "评论数量", "增加", "更新", "时间", "公告", "关键词", "测试",
}
TOKEN_TERMS = sorted(
    CHINESE_TERMS | set(POSITIVE_WORDS) | set(NEGATIVE_WORDS) | NEGATION_WORDS,
    key=len,
    reverse=True,
)


class SentimentAnalyzer:
    """可解释、零模型文件依赖的情感分析器。"""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        nlp_config = (config or {}).get("nlp", {})
        self.positive_threshold = float(nlp_config.get("positive_threshold", 0.15))
        self.negative_threshold = float(nlp_config.get("negative_threshold", -0.15))

    def score(self, text: str) -> float:
        """计算 -1 到 1 的极性分数，并处理相邻的简单否定词。"""
        tokens = self._tokens(str(text or ""))
        scores: list[float] = []
        for index, token in enumerate(tokens):
            value = POSITIVE_WORDS.get(token, 0.0) + NEGATIVE_WORDS.get(token, 0.0)
            if not value:
                continue
            previous = tokens[max(0, index - 2):index]
            if any(item in NEGATION_WORDS for item in previous):
                value = -value
            scores.append(value)
        if not scores:
            return 0.0
        return max(-1.0, min(1.0, sum(scores) / len(scores)))

    def label(self, score: float) -> str:
        if score >= self.positive_threshold:
            return "positive"
        if score <= self.negative_threshold:
            return "negative"
        return "neutral"

    def analyze(self, frame: pd.DataFrame) -> pd.DataFrame:
        """为明细数据增加 ``sentiment_score`` 和 ``sentiment`` 两列。"""
        if "text" not in frame.columns:
            raise ValueError("输入数据必须包含 text 列")
        result = frame.copy()
        result["sentiment_score"] = result["text"].fillna("").map(self.score)
        result["sentiment"] = result["sentiment_score"].map(self.label)
        return result

    def keywords(self, texts: Iterable[str], top_n: int = 15) -> list[tuple[str, int]]:
        """提取领域词频，过滤停用词和情感词，避免返回单字碎片。"""
        if top_n <= 0:
            return []
        sentiment_terms = set(POSITIVE_WORDS) | set(NEGATIVE_WORDS) | NEGATION_WORDS
        counts: Counter[str] = Counter()
        for text in texts:
            for token in self._tokens(str(text or "")):
                if token in STOPWORDS or token in sentiment_terms:
                    continue
                if len(token) > 1 and not token.isdigit():
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
            index = 0
            while index < len(segment):
                match = next((term for term in TOKEN_TERMS if segment.startswith(term, index)), None)
                if match:
                    tokens.append(match)
                    index += len(match)
                else:
                    # 未知中文按二字切分，保留少量可用信息而不是产生所有重叠 n-gram。
                    tokens.append(segment[index:index + 2])
                    index += 2
        return tokens
