from src.scraper import SocialMediaCollector


def test_normalize_removes_duplicate_urls() -> None:
    collector = SocialMediaCollector(seed=1)
    rows = [
        {"source": "test", "text": " hello  world ", "url": "https://a", "published_at": "2026-01-01T00:00:00Z"},
        {"source": "test", "text": "duplicate", "url": "https://a", "published_at": "2026-01-01T00:00:00Z"},
        {"source": "test", "text": "", "url": "https://b", "published_at": "2026-01-01T00:00:00Z"},
    ]
    result = collector._normalize(rows)
    assert len(result) == 1
    assert result.iloc[0]["text"] == "hello world"


def test_mock_collection_has_expected_columns() -> None:
    result = SocialMediaCollector(seed=1).collect("测试", hours=2)
    assert len(result) > 0
    assert {"post_id", "text", "url", "published_at"}.issubset(result.columns)
    assert result["url"].is_unique


def test_normalize_canonicalizes_urls_and_bad_engagement() -> None:
    collector = SocialMediaCollector(seed=1)
    result = collector._normalize([
        {"text": "a", "url": "HTTPS://EXAMPLE.COM/post/1/", "published_at": "2026-01-01T00:00:00Z", "engagement": "bad"},
        {"text": "duplicate", "url": "https://example.com/post/1", "published_at": "2026-01-01T00:00:00Z", "engagement": 10},
    ])
    assert len(result) == 1
    assert result.iloc[0]["engagement"] == 0


def test_invalid_collection_arguments() -> None:
    collector = SocialMediaCollector()
    for kwargs in ({"keyword": "", "hours": 1}, {"keyword": "test", "hours": 0}):
        try:
            collector.collect(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("无效采集参数应抛出 ValueError")
