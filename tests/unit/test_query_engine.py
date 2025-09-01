"""Tests for query engine providers."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from core.config.settings import AppSettings
from core.providers.query_engine import QueryEngineProvider


def test_query_engine_provider_contract():
    class Impl(QueryEngineProvider):
        async def query(self, **kwargs):
            return {"ok": True}

        async def similarity_search(self, **kwargs):
            return []

    impl = Impl(AppSettings())
    assert impl.config is not None


@pytest.mark.asyncio
async def test_llamaindex_query_engine_query(monkeypatch: pytest.MonkeyPatch):
    from core.providers import llamaindex_query_engine as mod

    cfg = AppSettings()
    cfg.embedding_provider = "openai"
    cfg.openai_api_key = "k"
    cfg.qdrant_url = "http://localhost:6333"
    cfg.collection_name = "c"

    # Mock DB to choose qdrant
    monkeypatch.setenv("QDRANT_URL", cfg.qdrant_url)
    with (
        patch("core.providers.llamaindex_query_engine.load_db_settings") as m_db,
        patch("core.providers.llamaindex_query_engine.AsyncQdrantClient"),
        patch("core.providers.llamaindex_query_engine.QdrantVectorStore"),
        patch("core.providers.llamaindex_query_engine.VectorStoreIndex") as m_index,
        patch("core.providers.llamaindex_query_engine.VectorIndexRetriever") as m_retr,
        patch(
            "core.providers.llamaindex_query_engine.get_response_synthesizer"
        ) as m_syn,
        patch(
            "core.providers.llamaindex_query_engine.RetrieverQueryEngine"
        ) as m_engine,
    ):
        m_db.return_value = SimpleNamespace(db_type="qdrant", index_name="c")
        mock_engine = AsyncMock()
        # Simulate response shape from LlamaIndex
        resp = SimpleNamespace(
            response="answer",
            confidence=0.5,
            source_nodes=[],
        )
        mock_engine.aquery = AsyncMock(return_value=resp)
        m_engine.return_value = mock_engine
        m_index.from_vector_store.return_value = SimpleNamespace(vector_store=object())
        m_retr.return_value = SimpleNamespace()
        m_syn.return_value = SimpleNamespace()

        qe = mod.LlamaIndexQueryEngine(cfg)
        out = await qe.query("What?")
        assert out["answer"] == "answer"
        assert "sources" in out


@pytest.mark.asyncio
async def test_llamaindex_similarity_search(monkeypatch: pytest.MonkeyPatch):
    from core.providers import llamaindex_query_engine as mod

    cfg = AppSettings()
    cfg.embedding_provider = "openai"
    cfg.openai_api_key = "k"
    cfg.qdrant_url = "http://localhost:6333"
    cfg.collection_name = "c"

    with (
        patch("core.providers.llamaindex_query_engine.load_db_settings") as m_db,
        patch("core.providers.llamaindex_query_engine.AsyncQdrantClient"),
        patch("core.providers.llamaindex_query_engine.QdrantVectorStore"),
        patch("core.providers.llamaindex_query_engine.VectorStoreIndex") as m_index,
        patch("core.providers.llamaindex_query_engine.VectorIndexRetriever") as m_retr,
    ):
        m_db.return_value = SimpleNamespace(db_type="qdrant", index_name="c")
        m_index.from_vector_store.return_value = SimpleNamespace(vector_store=object())
        # Mock retriever.aretrieve to return nodes
        node = SimpleNamespace(
            node=SimpleNamespace(
                get_text=lambda: "t",
                metadata={"url": "u", "title": "ti", "chunk_id": "id"},
                node_id="id",
            ),
            score=0.9,
        )
        m_retr.return_value.aretrieve = AsyncMock(return_value=[node])

        qe = mod.LlamaIndexQueryEngine(cfg)
        res = await qe.similarity_search("q", top_k=1)
        assert isinstance(res, list) and res[0]["text"] == "t"
