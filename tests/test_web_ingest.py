"""
Unit tests for scripts/web_ingest.py — Day 10.

Tests cover Wikipedia fetching, article filtering, Kafka payload
construction, and error handling. No live HTTP or Kafka calls.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

os.environ.setdefault("CONFLUENT_BOOTSTRAP_SERVERS", "test-broker:9092")
os.environ.setdefault("CONFLUENT_API_KEY",           "test-key")
os.environ.setdefault("CONFLUENT_API_SECRET",        "test-secret")
os.environ.setdefault("KAFKA_RAW_TOPIC",             "raw-documents")


def _mock_wiki_response(title: str, extract: str, full_url: str = "https://en.wikipedia.org/wiki/Test"):
    page_id = "12345"
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status.return_value = None
    resp.json.return_value = {
        "query": {
            "pages": {
                page_id: {
                    "title":   title,
                    "extract": extract,
                    "fullurl": full_url,
                }
            }
        }
    }
    return resp


class TestFetchArticle:
    def test_returns_dict_on_success(self):
        from scripts.web_ingest import fetch_article
        content = " ".join(["word"] * 100)
        resp = _mock_wiki_response("Test Article", content)
        with patch("requests.get", return_value=resp):
            result = fetch_article("Test_Article")
        assert result is not None
        assert result["title"] == "Test Article"
        assert result["content"] == content
        assert result["word_count"] == 100

    def test_returns_none_for_missing_article(self):
        from scripts.web_ingest import fetch_article
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status.return_value = None
        resp.json.return_value = {
            "query": {"pages": {"-1": {"title": "Missing", "missing": ""}}}
        }
        with patch("requests.get", return_value=resp):
            result = fetch_article("NonExistent_Article")
        assert result is None

    def test_returns_none_for_too_short_article(self):
        from scripts.web_ingest import fetch_article
        content = "Too short."
        resp = _mock_wiki_response("Short Article", content)
        with patch("requests.get", return_value=resp):
            result = fetch_article("Short_Article")
        assert result is None

    def test_returns_none_on_network_error(self):
        from scripts.web_ingest import fetch_article
        with patch("requests.get", side_effect=Exception("network error")):
            result = fetch_article("Any_Article")
        assert result is None

    def test_url_is_populated(self):
        from scripts.web_ingest import fetch_article
        content = " ".join(["word"] * 100)
        url = "https://en.wikipedia.org/wiki/Test_Article"
        resp = _mock_wiki_response("Test Article", content, full_url=url)
        with patch("requests.get", return_value=resp):
            result = fetch_article("Test_Article")
        assert result["url"] == url

    def test_word_count_is_accurate(self):
        from scripts.web_ingest import fetch_article
        words = ["alpha", "beta", "gamma"] * 40  # 120 words
        content = " ".join(words)
        resp = _mock_wiki_response("Word Count Test", content)
        with patch("requests.get", return_value=resp):
            result = fetch_article("Word_Count_Test")
        assert result["word_count"] == 120


class TestWebIngestPayload:
    """Tests that Kafka payloads are correctly structured."""

    def _run_ingest_with_mock(self, article: dict):
        """Run run_web_ingest() with one mocked article and capture produced payloads."""
        import io
        import json
        import fastavro
        from pathlib import Path

        schema_path = Path(__file__).parent.parent / "schemas" / "document_event.avsc"
        with schema_path.open() as f:
            schema = fastavro.parse_schema(json.load(f))

        produced = []

        def fake_produce(topic, key, value, on_delivery=None):
            buf = io.BytesIO(value)
            record = fastavro.schemaless_reader(buf, schema)
            produced.append(record)

        fake_producer = MagicMock()
        fake_producer.produce.side_effect = fake_produce
        fake_producer.flush.return_value = 0

        with (
            patch("scripts.web_ingest.fetch_article", return_value=article),
            patch("scripts.web_ingest.ARTICLES", ["Fake_Article"]),
            patch("scripts.web_ingest._build_producer", return_value=fake_producer),
        ):
            from scripts.web_ingest import run_web_ingest
            run_web_ingest()

        return produced

    def test_source_type_is_web(self):
        article = {"title": "Test", "url": "https://en.wikipedia.org/wiki/Test",
                   "content": "content text here", "word_count": 3}
        produced = self._run_ingest_with_mock(article)
        assert produced[0]["source_type"] == "WEB"

    def test_content_matches_article(self):
        content = "The quick brown fox jumps over the lazy dog."
        article = {"title": "Test", "url": "https://en.wikipedia.org/wiki/Test",
                   "content": content, "word_count": 9}
        produced = self._run_ingest_with_mock(article)
        assert produced[0]["content"] == content

    def test_metadata_contains_word_count(self):
        article = {"title": "Test", "url": "https://en.wikipedia.org/wiki/Test",
                   "content": "some content words here now", "word_count": 5}
        produced = self._run_ingest_with_mock(article)
        assert produced[0]["metadata"]["word_count"] == "5"

    def test_metadata_contains_title(self):
        article = {"title": "Apache Kafka", "url": "https://en.wikipedia.org/wiki/Apache_Kafka",
                   "content": " ".join(["word"] * 60), "word_count": 60}
        produced = self._run_ingest_with_mock(article)
        assert produced[0]["metadata"]["title"] == "Apache Kafka"

    def test_source_uri_is_wikipedia_url(self):
        url = "https://en.wikipedia.org/wiki/Vector_database"
        article = {"title": "Vector database", "url": url,
                   "content": " ".join(["word"] * 60), "word_count": 60}
        produced = self._run_ingest_with_mock(article)
        assert produced[0]["source_uri"] == url

    def test_document_id_is_uuid(self):
        import uuid
        article = {"title": "Test", "url": "https://en.wikipedia.org/wiki/Test",
                   "content": " ".join(["word"] * 60), "word_count": 60}
        produced = self._run_ingest_with_mock(article)
        uuid.UUID(produced[0]["document_id"])  # raises if not valid UUID

    def test_schema_version_is_1_0(self):
        article = {"title": "Test", "url": "https://en.wikipedia.org/wiki/Test",
                   "content": " ".join(["word"] * 60), "word_count": 60}
        produced = self._run_ingest_with_mock(article)
        assert produced[0]["schema_version"] == "1.0"

    def test_failed_fetch_produces_nothing(self):
        with (
            patch("scripts.web_ingest.fetch_article", return_value=None),
            patch("scripts.web_ingest.ARTICLES", ["Missing_Article"]),
            patch("scripts.web_ingest._build_producer", return_value=MagicMock()),
        ):
            from scripts.web_ingest import run_web_ingest
            run_web_ingest()  # should complete without error and produce 0 messages
