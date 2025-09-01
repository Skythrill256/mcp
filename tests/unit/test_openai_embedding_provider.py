"""Additional tests to raise coverage for OpenAIEmbeddingProvider."""

from unittest.mock import patch, Mock

import pytest

from core.config.settings import AppSettings
from core.providers.openai_embedding import OpenAIEmbeddingProvider


@pytest.mark.asyncio
async def test_generate_query_embedding_calls_model(monkeypatch):
    cfg = AppSettings()
    cfg.openai_api_key = "k"
    cfg.embedding_model = "text-embedding-3-small"
    cfg.embedding_dimensions = 1536

    with patch("core.providers.openai_embedding.OpenAIEmbedding") as m_cls:
        inst = Mock()
        inst.get_text_embedding.return_value = [0.1, 0.2]
        m_cls.return_value = inst

        prov = OpenAIEmbeddingProvider(cfg)
        vec = await prov.generate_query_embedding("hello")
        assert vec == [0.1, 0.2]


def test_validate_embedding_config_warns(monkeypatch, caplog):
    cfg = AppSettings()
    cfg.embedding_model = "text-embedding-3-small"
    cfg.embedding_dimensions = 1024  # mismatch

    with patch("core.providers.openai_embedding.OpenAIEmbedding") as m_cls:
        m_cls.return_value = Mock(get_text_embedding=lambda x: [0.1])
        prov = OpenAIEmbeddingProvider(cfg)
        # Trigger validation explicitly
        prov._validate_embedding_config()
        assert any("Dimension mismatch" in r.message for r in caplog.records)
