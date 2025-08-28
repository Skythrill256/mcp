"""Tests for the ingestion models."""

import pytest
from pydantic import ValidationError
from core.models.ingestion import IngestRequest, IngestResponse, EmbeddingResult
from llama_index.core.schema import TextNode


def test_ingest_request_valid():
    """Test that a valid IngestRequest can be created."""
    request = IngestRequest(
        url="https://example.com",
        recreate=True,
        collection_prefix="test",
        max_pages=10,
        max_depth=3,
        include_external=True,
        keywords=["test", "example"],
        url_patterns=["/test/*"]
    )
    
    assert str(request.url) == "https://example.com/"
    assert request.recreate is True
    assert request.collection_prefix == "test"
    assert request.max_pages == 10
    assert request.max_depth == 3
    assert request.include_external is True
    assert request.keywords == ["test", "example"]
    assert request.url_patterns == ["/test/*"]


def test_ingest_request_defaults():
    """Test that default values are correctly applied."""
    request = IngestRequest(url="https://example.com")
    
    assert request.recreate is False
    assert request.collection_prefix == "site"
    assert request.max_pages is None
    assert request.max_depth is None
    assert request.include_external is None
    assert request.keywords is None
    assert request.url_patterns is None


def test_ingest_request_invalid_url():
    """Test that an invalid URL raises a ValidationError."""
    with pytest.raises(ValidationError):
        IngestRequest(url="not-a-url")


def test_ingest_response_creation():
    """Test that an IngestResponse can be created correctly."""
    response = IngestResponse(
        site="https://example.com",
        collection="test_collection",
        ingestion={"stored": {"count": 5}},
        mcp={"host": "localhost", "port": 8000, "path": "/mcp"}
    )
    
    assert response.site == "https://example.com"
    assert response.collection == "test_collection"
    assert response.ingestion == {"stored": {"count": 5}}
    assert response.mcp == {"host": "localhost", "port": 8000, "path": "/mcp"}


def test_embedding_result_to_node():
    """Test that an EmbeddingResult can be converted to a TextNode."""
    result = EmbeddingResult(
        text="Test content",
        embedding=[0.1, 0.2, 0.3],
        token_count=5,
        chunk_id="test_chunk_1",
        metadata={"source": "test"}
    )
    
    node = result.to_node()
    
    assert isinstance(node, TextNode)
    assert node.text == "Test content"
    assert node.embedding == [0.1, 0.2, 0.3]
    assert node.metadata == {"source": "test"}
    assert node.id_ == "test_chunk_1"