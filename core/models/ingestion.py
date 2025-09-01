from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, HttpUrl
from llama_index.core.schema import TextNode


class IngestRequest(BaseModel):
    url: HttpUrl
    recreate: bool = False
    collection_prefix: str = "site"
    max_pages: Optional[int] = None
    max_depth: Optional[int] = None
    include_external: Optional[bool] = None
    keywords: Optional[list[str]] = None
    url_patterns: Optional[list[str]] = None


class IngestResponse(BaseModel):
    site: str
    collection: str
    ingestion: dict[str, Any]
    mcp: dict[str, Any]


class EmbeddingResult(BaseModel):
    """Result of embedding generation."""

    text: str
    embedding: list[float]
    token_count: int
    chunk_id: str
    metadata: dict[str, Any]

    def to_node(self) -> TextNode:
        """Convert to LlamaIndex TextNode."""
        return TextNode(
            text=self.text,
            embedding=self.embedding,
            metadata=self.metadata,
            id_=self.chunk_id,
        )
