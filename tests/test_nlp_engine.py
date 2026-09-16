import pandas as pd

from src.nlp_engine import SentimentAnalyzer


def test_sentiment_score_and_label() -> None:
    analyzer = SentimentAnalyzer()
    assert analyzer.label(analyzer.score("体验很好，值得推荐")) == "positive"
    assert analyzer.label(analyzer.score("体验很差，用户失望")) == "negative"
    assert analyzer.label(analyzer.score("官方发布了最新进展")) == "neutral"


def test_analyze_adds_columns() -> None:
    result = SentimentAnalyzer().analyze(pd.DataFrame({"text": ["服务专业", "质量问题"]}))
    assert {"sentiment_score", "sentiment"}.issubset(result.columns)
    assert len(result) == 2


def test_keywords() -> None:
    keywords = SentimentAnalyzer().keywords(["新能源汽车 新能源汽车 体验很好", "新能源汽车"])
    assert keywords[0] == ("新能源", 3)

