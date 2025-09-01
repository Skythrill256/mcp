"""Tests for Cohere reranking service."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from core.config.settings import AppSettings
from core.services.reranking import CohereReranker


def test_reranker_init_requires_key(monkeypatch: pytest.MonkeyPatch):
    cfg = AppSettings()
    cfg.cohere_api_key = None
    # No env either
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    with pytest.raises(ValueError):
        CohereReranker(cfg)


def test_reranker_success(monkeypatch: pytest.MonkeyPatch):
    cfg = AppSettings()
    cfg.cohere_api_key = "k"

    class MockCohere:
        def __init__(self, api_key: str):
            self.api_key = api_key

        def rerank(self, **kwargs):
            # Return minimal structure expected
            res = SimpleNamespace(
                results=[
                    SimpleNamespace(
                        index=0,
                        relevance_score=0.9,
                        document=SimpleNamespace(text="d0"),
                    ),
                    SimpleNamespace(
                        index=1,
                        relevance_score=0.8,
                        document=SimpleNamespace(text="d1"),
                    ),
                ]
            )
            return res

    with patch("core.services.reranking.cohere.Client", MockCohere):
        rr = CohereReranker(cfg)
        docs = [
            {"text": "A"},
            {"text": "B"},
        ]
        out = rr.rerank_documents("q", docs, top_n=2)
        assert len(out) == 2 and out[0]["index"] == 0
