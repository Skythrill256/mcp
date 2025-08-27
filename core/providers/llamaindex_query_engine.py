"""
Query functionality using LlamaIndex integration with OpenAI models.
"""

import logging
from typing import Any, Optional

try:
    from llama_index.core import Settings, VectorStoreIndex
    from llama_index.core.query_engine import RetrieverQueryEngine
    from llama_index.core.response_synthesizers import get_response_synthesizer
    from llama_index.core.retrievers import VectorIndexRetriever
    from llama_index.core.schema import NodeWithScore
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.llms.openai import OpenAI
    from llama_index.vector_stores.qdrant import QdrantVectorStore
    from qdrant_client import AsyncQdrantClient
    # Optional PGVector
    try:
        from llama_index.vector_stores.postgres import PGVectorStore  # type: ignore
    except Exception:
        PGVectorStore = None  # type: ignore
except ImportError:
    raise ImportError(
        "llama-index packages are required. Install with: "
        "pip install llama-index llama-index-vector-stores-qdrant "
        "llama-index-embeddings-openai llama-index-llms-openai"
    ) from None

from core.config.settings import AppSettings
from core.config.database import load_db_settings
from core.errors.exceptions import QueryError
from core.providers.query_engine import QueryEngineProvider

logger = logging.getLogger(__name__)


class LlamaIndexQueryEngine(QueryEngineProvider):
    """Advanced query engine using LlamaIndex with OpenAI integration."""

    def __init__(self, config: AppSettings):
        """Initialize the query engine."""
        super().__init__(config)

        # Initialize LlamaIndex components
        self._setup_llamaindex()

        # Query engine components
        self.vector_store = None
        self.index = None
        self.query_engine = None

    def _setup_llamaindex(self):
        """Setup LlamaIndex global settings."""
        # Configure OpenAI LLM
        Settings.llm = OpenAI(
            api_key=self.config.openai_api_key,
            model="gpt-3.5-turbo",
            temperature=0.1,
            max_tokens=1000
        )

        # Configure OpenAI embeddings
        Settings.embed_model = OpenAIEmbedding(
            api_key=self.config.openai_api_key,
            model=self.config.embedding_model,
            dimensions=self.config.embedding_dimensions
        )

        # Set chunk size for text processing
        Settings.chunk_size = self.config.chunk_size
        Settings.chunk_overlap = self.config.chunk_overlap

    async def initialize(self):
        """Initialize the query engine with vector store connection."""
        try:
            # Choose vector store based on database config
            db = load_db_settings()
            backend = (db.db_type or "qdrant").lower()
            if backend in ("qdrant",):
                qdrant_client = AsyncQdrantClient(url=self.config.qdrant_url)
                # Prefer per-site collection name if available
                collection_name = self.config.collection_name or db.index_name
                self.vector_store = QdrantVectorStore(
                    aclient=qdrant_client,
                    collection_name=collection_name,
                    enable_hybrid=False,
                )
            elif backend in ("pgvector",):
                if PGVectorStore is None:
                    raise ImportError(
                        "PGVector backend requested but llama-index-vector-stores-postgres is not installed.\n"
                        "Install with: pip install 'llama-index-vector-stores-postgres' 'psycopg[binary]' sqlalchemy"
                    )
                if not self.config.postgres_connection_string:
                    raise QueryError(
                        f"POSTGRES connection string not found in env '{db.api_endpoint_env}'."
                    )
                # Prefer per-site collection/table name if available
                table_name = self.config.collection_name or db.index_name
                self.vector_store = PGVectorStore.from_params(
                    connection_string=self.config.postgres_connection_string,
                    table_name=table_name,
                    embed_dim=self.config.embedding_dimensions,
                )
            else:
                raise QueryError(f"Unsupported db_type: {backend}")

            # Create index from existing vector store
            self.index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store
            )

            # Create query engine
            retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=10,
                vector_store_query_mode="default"
            )

            response_synthesizer = get_response_synthesizer(
                response_mode="tree_summarize",
                use_async=True
            )

            self.query_engine = RetrieverQueryEngine(
                retriever=retriever,
                response_synthesizer=response_synthesizer
            )

            logger.info("Query engine initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing query engine: {str(e)}")
            raise QueryError(f"Failed to initialize query engine: {str(e)}") from e

    async def query(
        self,
        question: str,
        top_k: int = 5,
        similarity_threshold: float = 0.7,
        include_sources: bool = True,
        filters: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        """
        Query the vector database using natural language.

        Args:
            question: Natural language question
            top_k: Number of top results to retrieve
            similarity_threshold: Minimum similarity score
            include_sources: Whether to include source information
            filters: Optional filters for search

        Returns:
            Query response with answer and sources

        Raises:
            QueryError: If query execution fails
        """
        if not self.query_engine:
            await self.initialize()

        try:
            logger.info(f"Processing query: {question}")

            # Execute query using LlamaIndex
            response = await self.query_engine.aquery(question)

            # Format response
            result = {
                'question': question,
                'answer': str(response.response),
                'confidence_score': getattr(response, 'confidence', 0.0),
                'sources': []
            }

            # Extract source information if requested
            if include_sources and hasattr(response, 'source_nodes'):
                for node in response.source_nodes:
                    if isinstance(node, NodeWithScore):
                        source_info = {
                            'text': node.node.text,
                            'score': node.score,
                            'metadata': node.node.metadata,
                            'node_id': node.node.node_id,
                            'url': node.node.metadata.get('url', ''),
                            'title': node.node.metadata.get('title', ''),
                            'chunk_id': node.node.metadata.get('chunk_id', '')
                        }
                        result['sources'].append(source_info)

            logger.info(f"Query completed with {len(result['sources'])} sources")
            return result

        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            raise QueryError(f"Failed to execute query: {str(e)}") from e

    async def similarity_search(
        self,
        query: str,
        top_k: int = 10,
        similarity_threshold: float = 0.7,
        filters: Optional[dict[str, Any]] = None
    ) -> list[dict[str, Any]]:
        """
        Perform similarity search without LLM processing.

        Args:
            query: Search query
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score
            filters: Optional search filters

        Returns:
            List of similar documents with scores

        Raises:
            QueryError: If search fails
        """
        try:
            if not self.index:
                await self.initialize()
            # Use retriever directly for semantic similarity results
            retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=top_k,
                vector_store_query_mode="default",
            )
            nodes = await retriever.aretrieve(query)
            out: list[dict[str, Any]] = []
            for n in nodes:
                try:
                    out.append({
                        "text": n.node.get_text() if hasattr(n.node, "get_text") else getattr(n.node, "text", ""),
                        "score": getattr(n, "score", None),
                        "metadata": getattr(n.node, "metadata", {}),
                        "node_id": getattr(n.node, "node_id", None),
                        "url": getattr(n.node, "metadata", {}).get("url", ""),
                        "title": getattr(n.node, "metadata", {}).get("title", ""),
                        "chunk_id": getattr(n.node, "metadata", {}).get("chunk_id", ""),
                    })
                except Exception:
                    continue
            return out

        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            raise QueryError(f"Failed to perform similarity search: {str(e)}") from e
