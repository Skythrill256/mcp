"""
Embedding generation using LlamaIndex's OpenAIEmbedding.
"""

import asyncio
import logging
import os
from contextlib import suppress
from typing import Any

# Third-party imports (kept at top to satisfy import ordering rules)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Document, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.vector_stores.postgres import PGVectorStore  # type: ignore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from core.config.settings import AppSettings
from core.config.database import load_db_settings
from core.errors.exceptions import EmbeddingError
from core.providers.embedding import EmbeddingProvider, EmbeddingResult

logger = logging.getLogger(__name__)


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Manages embedding generation using LlamaIndex OpenAIEmbedding."""

    def __init__(self, config: AppSettings):
        """Initialize the embedding manager."""
        super().__init__(config)

        # Ensure API key is available for LlamaIndex/OpenAI
        if self.config.openai_api_key and not os.environ.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = self.config.openai_api_key

        # Initialize LlamaIndex embedding model
        try:
            self.embed_model = OpenAIEmbedding(
                model=self.config.embedding_model,
                dimensions=self.config.embedding_dimensions,
                embed_batch_size=self.config.batch_size,
            )
        except TypeError:
            # Fallback for older versions without dimensions/embed_batch_size kwargs
            self.embed_model = OpenAIEmbedding(model=self.config.embedding_model)

        # Validate embedding model and dimensions
        self._validate_embedding_config()

        # Placeholders for LlamaIndex components
        self._splitter = None
        self._vector_store = None
        self._index = None
        self._query_engine = None

    def _ensure_llamaindex_ready(self):
        """Initialize LlamaIndex ServiceContext and VectorStore based on DB config."""
        if Document is None:
            raise ImportError(
                "llama-index core/vector-store packages are required. Install with: pip install llama-index llama-index-vector-stores-qdrant"
            )

        # Configure global Settings for LlamaIndex (embed model and parser)
        if Settings is not None:
            Settings.embed_model = self.embed_model
            if self._splitter is None:
                self._splitter = SentenceSplitter(
                    chunk_size=self.config.chunk_size,
                    chunk_overlap=self.config.chunk_overlap,
                )
            Settings.node_parser = self._splitter

        if self._vector_store is None:
            db = load_db_settings()
            # Default to qdrant
            backend = (db.db_type or "qdrant").lower()
            if backend in ("qdrant",):
                client = QdrantClient(url=self.config.qdrant_url)
                # Ensure collection exists with correct vector params
                if VectorParams is not None and Distance is not None:
                    # Prefer the per-site collection name from the WebVectorConfig so
                    # each site's embeddings live in a separate collection. Fall
                    # back to the DB-configured index_name only if the site-specific
                    # name is not set.
                    collection_name = self.config.collection_name or db.index_name
                    try:
                        client.get_collection(collection_name)
                    except Exception:
                        with suppress(Exception):
                            client.create_collection(
                                collection_name=collection_name,
                                vectors_config=VectorParams(
                                    size=self.config.embedding_dimensions,
                                    distance=Distance.COSINE,
                                ),
                            )
                    self._vector_store = QdrantVectorStore(
                        client=client,
                        collection_name=collection_name,
                    )
            elif backend in ("postgres", "pgvector"):
                if PGVectorStore is None:
                    raise ImportError(
                        "PGVector backend requested but llama-index-vector-stores-postgres is not installed.\n"
                        "Install with: pip install 'llama-index-vector-stores-postgres' 'psycopg[binary]' sqlalchemy"
                    )
                conn = db.connection
                if not conn:
                    raise EnvironmentError(
                        f"POSTGRES connection string not found in env '{db.api_endpoint_env}'."
                    )
                # PGVectorStore will create tables as needed. Prefer per-site name.
                table_name = self.config.collection_name or db.index_name
                self._vector_store = PGVectorStore(
                    connection_string=conn,
                    table_name=table_name,
                )
            else:
                raise ValueError(f"Unsupported db_type: {backend}")

    async def generate_embeddings(
        self, scraped_data: list[dict[str, Any]]
    ) -> list[EmbeddingResult]:
        """
        Generate embeddings for scraped content using LlamaIndex's splitter and embedder.

        Args:
            scraped_data: List of scraped page data

        Returns:
            List of embedding results

        Raises:
            EmbeddingError: If embedding generation fails
        """
        try:
            # Ensure LlamaIndex components
            self._ensure_llamaindex_ready()

            # Build Documents
            documents: list[Document] = []  # type: ignore[assignment]
            for page_data in scraped_data:
                content = page_data.get("content", "")
                metadata = page_data.get("metadata", {}) or {}
                if not content.strip():
                    logger.warning(
                        f"No content found for {metadata.get('url', 'unknown URL')}"
                    )
                    continue
                documents.append(Document(text=content, metadata=metadata))  # type: ignore[call-arg]

            if not documents:
                logger.warning("No chunks to embed")
                return []

            # Split into nodes using LlamaIndex's splitter
            splitter = self._splitter or SentenceSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
            )

            def _split_docs():
                return splitter.get_nodes_from_documents(documents)

            nodes = await asyncio.to_thread(_split_docs)

            logger.info(f"Generating embeddings for {len(nodes)} chunks via LlamaIndex")

            # Embed each node text via LlamaIndex embedding model
            def _embed_nodes(texts: list[str]) -> list[list[float]]:
                return [self.embed_model.get_text_embedding(t) for t in texts]

            texts = [
                n.get_text() if hasattr(n, "get_text") else getattr(n, "text", "")
                for n in nodes
            ]
            vectors = await asyncio.to_thread(_embed_nodes, texts)

            # Assemble EmbeddingResult; compute per-document chunk indices
            embedding_results: list[EmbeddingResult] = []
            doc_counters: dict[str, int] = {}
            for node, vec in zip(nodes, vectors):
                # Derive a doc key from metadata URL if present, else a generic key
                meta = getattr(node, "metadata", {}) or {}
                doc_key = meta.get("url") or meta.get("source") or "doc"
                idx = doc_counters.get(doc_key, 0)
                doc_counters[doc_key] = idx + 1

                # Build metadata with chunk info
                merged_meta = dict(meta)
                merged_meta.update(
                    {
                        "chunk_index": idx,
                    }
                )

                chunk_id = getattr(node, "node_id", None) or f"{doc_key}_{idx}"

                embedding_results.append(
                    EmbeddingResult(
                        text=texts[embedding_results.__len__()],
                        embedding=vec,
                        token_count=0,
                        chunk_id=str(chunk_id),
                        metadata=merged_meta,
                    )
                )

            logger.info(f"Successfully generated {len(embedding_results)} embeddings")
            return embedding_results

        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise EmbeddingError(f"Failed to generate embeddings: {str(e)}") from e

    async def generate_query_embedding(self, query: str) -> list[float]:
        """
        Generate embedding for a query string using LlamaIndex.
        """
        try:
            # Run sync LlamaIndex call in a thread
            return await asyncio.to_thread(self.embed_model.get_text_embedding, query)
        except Exception as e:
            logger.error(f"Error generating query embedding: {str(e)}")
            raise EmbeddingError(f"Failed to generate query embedding: {str(e)}") from e

    def _validate_embedding_config(self):
        """Validate embedding model and dimensions compatibility."""
        model_dimensions = {
            "text-embedding-3-large": 3072,
            "text-embedding-3-small": 1536,
            "text-embedding-ada-002": 1536,
        }

        if self.config.embedding_model in model_dimensions:
            expected_dim = model_dimensions[self.config.embedding_model]
            if self.config.embedding_dimensions != expected_dim:
                logger.warning(
                    f"Dimension mismatch: {self.config.embedding_model} "
                    f"typically uses {expected_dim} dimensions, "
                    f"but config specifies {self.config.embedding_dimensions}"
                )
