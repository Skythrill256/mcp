"""
Vector storage functionality using PostgreSQL with pgvector extension.
"""

import logging
from typing import Any, Optional

import sqlalchemy
from sqlalchemy.engine.url import make_url
from llama_index.vector_stores.postgres import PGVectorStore

from core.config.settings import AppSettings
from core.embeddings.embedding import EmbeddingResult
from core.errors.exceptions import StorageError
from core.providers.vector_store import VectorStoreProvider

logger = logging.getLogger(__name__)


class PgVectorStoreProvider(VectorStoreProvider):
    """Manages vector storage operations with PostgreSQL/pgvector."""

    def __init__(self, config: AppSettings):
        super().__init__(config)
        self.connection_string = config.postgres_connection_string
        self.table_name = config.collection_name
        self._vector_store: Optional[PGVectorStore] = None

        # Validate connection string early
        if (
            not isinstance(self.connection_string, str)
            or not self.connection_string.strip()
        ):
            raise StorageError(
                "POSTGRES_CONNECTION_STRING environment variable is not set or empty. "
                "Please set it to a valid PostgreSQL connection string, e.g., "
                "'postgresql://user:password@host:port/dbname'."
            )
        try:
            # Try to parse the URL to give an early, clear error
            make_url(self.connection_string)
        except Exception as e:
            # Mask password when reporting back
            try:
                s = str(self.connection_string)
                masked = s
                if ":" in s and "@" in s:
                    # Simple mask: hide between : and @ (password)
                    before, after = s.split("@", 1)
                    if ":" in before:
                        userpart = before.split(":", 1)[0]
                        masked = f"{userpart}:*****@{after}"
            except Exception:
                masked = "<redacted>"
            raise StorageError(
                f"Invalid POSTGRES_CONNECTION_STRING (masked): {masked}. "
                f"Error: {e}. Original cause: {e.__cause__}. Expected format: 'postgresql://user:password@host:port/dbname'."
            ) from e

    @property
    def vector_store(self) -> PGVectorStore:
        if self._vector_store is None:
            # Process the connection string for asyncpg
            # Remove sslmode and channel_binding parameters for asyncpg
            async_connection_string = self.connection_string.replace(
                "postgresql://", "postgresql+asyncpg://"
            )
            if "asyncpg" in async_connection_string:
                # asyncpg doesn't support sslmode and channel_binding parameters in the same way
                # Remove them to prevent the "unexpected keyword argument 'sslmode'" error
                async_connection_string = async_connection_string.replace(
                    "?sslmode=require&channel_binding=require", ""
                )
                async_connection_string = async_connection_string.replace(
                    "&channel_binding=require", ""
                )
                async_connection_string = async_connection_string.replace(
                    "?sslmode=require", ""
                )

                # Ensure there are no trailing ? characters
                if async_connection_string.endswith("?"):
                    async_connection_string = async_connection_string[:-1]

                # Ensure the connection string has an explicit port to prevent 'None' port issue
                try:
                    parsed_url = make_url(async_connection_string)
                    if parsed_url.port is None:
                        # Add default PostgreSQL port (5432) if not specified
                        host_part = f"://{parsed_url.username}:{parsed_url.password}@{parsed_url.host}"
                        host_with_port = f"://{parsed_url.username}:{parsed_url.password}@{parsed_url.host}:5432"
                        async_connection_string = async_connection_string.replace(
                            host_part, host_with_port, 1
                        )
                except Exception as e:
                    # If URL parsing fails, proceed with original string
                    logger.debug(
                        "Async connection URL parse failed; using original string: %s",
                        e,
                    )

            # Pass both the original connection string for the sync engine and the processed one for the async engine
            self._vector_store = PGVectorStore(
                connection_string=self.connection_string,
                async_connection_string=async_connection_string,
                table_name=self.table_name,
                embed_dim=self.config.embedding_dimensions,
            )
        return self._vector_store

    async def __aenter__(self):
        """Enter async context manager (no-op)."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context manager (no-op)."""
        return None

    async def create_collection(self, recreate: bool = False) -> bool:
        """Create or recreate the table used for storing vectors.

        Prefers to use the PGVectorStore internal metadata table when present.
        Falls back to creating the `vector` extension and a minimal SQL table.
        """
        try:
            engine = sqlalchemy.create_engine(self.connection_string)
            with engine.begin() as conn:
                inspector = sqlalchemy.inspect(engine)
                if inspector.has_table(self.table_name):
                    if recreate:
                        logger.info(f"Deleting existing table: {self.table_name}")
                        if hasattr(self.vector_store, "_metadata_table"):
                            try:
                                self.vector_store._metadata_table.drop(bind=conn)
                            except Exception:
                                conn.execute(
                                    sqlalchemy.text(
                                        f'DROP TABLE IF EXISTS "{self.table_name}" CASCADE'
                                    )
                                )  # nosec B608
                        else:
                            conn.execute(
                                sqlalchemy.text(
                                    f'DROP TABLE IF EXISTS "{self.table_name}" CASCADE'
                                )
                            )  # nosec B608
                    else:
                        logger.info(f"Table {self.table_name} already exists")
                        return False

                logger.info(f"Creating table: {self.table_name}")

                # Try to create pgvector extension if possible (graceful if not permitted)
                try:
                    conn.execute(
                        sqlalchemy.text("CREATE EXTENSION IF NOT EXISTS vector")
                    )
                except Exception as ee:
                    logger.debug(
                        f"Could not create pgvector extension automatically: {ee}"
                    )

                # Use PGVectorStore internals when available
                if hasattr(self.vector_store, "_metadata_table"):
                    try:
                        self.vector_store._metadata_table.create(bind=conn)
                        logger.info(
                            f"Successfully created table via PGVectorStore internals: {self.table_name}"
                        )
                        return True
                    except Exception as e:
                        logger.warning(
                            f"PGVectorStore internal creation failed, falling back to SQL: {e}"
                        )

                # Fallback: create a minimal table using the native `vector` type
                try:
                    create_sql = (
                        f"CREATE TABLE IF NOT EXISTS {self.table_name} ("
                        f"id SERIAL PRIMARY KEY, "
                        f"embedding vector({self.config.embedding_dimensions}), "
                        f"metadata jsonb"
                        f");"
                    )
                    conn.execute(sqlalchemy.text(create_sql))
                    logger.info(
                        f"Successfully created table via SQL: {self.table_name}"
                    )
                    return True
                except Exception as e:
                    logger.error(f"Error creating table via SQL: {e}")
                    # Provide a helpful message when the vector type is missing
                    if 'type "vector" does not exist' in str(
                        e
                    ) or "does not exist" in str(e):
                        raise StorageError(
                            "Postgres 'vector' type is missing. Ensure the 'pgvector' extension is installed on the database and the user has permission to create extensions, or create the extension manually: CREATE EXTENSION pgvector;"
                        ) from e
                    raise StorageError(f"Failed to create table: {e}") from e

        except Exception as e:
            logger.error(f"Error creating table: {e}")
            raise StorageError(f"Failed to create table: {e}") from e

    async def store_embeddings(self, embeddings: list[EmbeddingResult]) -> int:
        """Store embeddings and return the number stored."""
        if not embeddings:
            logger.warning("No embeddings to store")
            return 0

        try:
            await self.create_collection(recreate=False)

            # Convert embeddings to LlamaIndex nodes and add
            nodes = [emb.to_node() for emb in embeddings]
            await self.vector_store.async_add(nodes)

            logger.info(f"Successfully stored {len(nodes)} embeddings")
            return len(nodes)

        except Exception as e:
            logger.error(f"Error storing embeddings: {e}")
            raise StorageError(f"Failed to store embeddings: {e}") from e

    async def search_similar(
        self,
        query_embedding: list[float],
        limit: int = 10,
        score_threshold: float = 0.7,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """Search for similar vectors and return filtered results."""
        try:
            from llama_index.core.vector_stores import VectorStoreQuery

            query = VectorStoreQuery(
                query_embedding=query_embedding,
                similarity_top_k=limit,
                mode="default",
            )
            results = self.vector_store.query(query)

            formatted_results: list[dict[str, Any]] = []
            if results.nodes:
                for node, similarity in zip(results.nodes, results.similarities or []):
                    if similarity >= score_threshold:
                        formatted_results.append(
                            {
                                "id": node.node_id,
                                "score": similarity,
                                "text": node.get_content(),
                                "metadata": node.metadata,
                                "url": node.metadata.get("url", ""),
                                "title": node.metadata.get("title", ""),
                            }
                        )
            logger.info(f"Found {len(formatted_results)} similar results")
            return formatted_results

        except Exception as e:
            logger.error(f"Error searching vectors: {e}")
            raise StorageError(f"Failed to search vectors: {e}") from e

    async def get_collection_info(self) -> dict[str, Any]:
        """Return collection name and points_count."""
        try:
            engine = sqlalchemy.create_engine(self.connection_string)
            metadata = sqlalchemy.MetaData()
            table = sqlalchemy.Table(self.table_name, metadata, autoload_with=engine)
            with engine.connect() as conn:
                count = conn.execute(
                    sqlalchemy.select(sqlalchemy.func.count()).select_from(table)
                ).scalar_one()
            return {"name": self.table_name, "points_count": int(count)}
        except Exception as e:
            logger.error(f"Error getting table info: {e}")
            raise StorageError(f"Failed to get table info: {e}") from e

    async def delete_collection(self) -> bool:
        """Delete the collection/table if it exists."""
        try:
            engine = sqlalchemy.create_engine(self.connection_string)
            with engine.begin() as conn:
                if hasattr(self.vector_store, "_metadata_table"):
                    try:
                        self.vector_store._metadata_table.drop(
                            bind=conn, checkfirst=True
                        )
                        logger.info(
                            f"Successfully deleted table via metadata: {self.table_name}"
                        )
                        return True
                    except Exception:
                        metadata = sqlalchemy.MetaData()
                        table = sqlalchemy.Table(
                            self.table_name, metadata, autoload_with=conn
                        )
                        table.drop(bind=conn, checkfirst=True)
                        logger.info(
                            f"Successfully deleted table via SQL: {self.table_name}"
                        )
                        return True
                else:
                    metadata = sqlalchemy.MetaData()
                    table = sqlalchemy.Table(
                        self.table_name, metadata, autoload_with=conn
                    )
                    table.drop(bind=conn, checkfirst=True)
                    logger.info(
                        f"Successfully deleted table via SQL: {self.table_name}"
                    )
                    return True
        except Exception as e:
            logger.error(f"Error deleting table: {e}")
            raise StorageError(f"Failed to delete table: {e}") from e

    async def count_points(self, filters: Optional[dict[str, Any]] = None) -> int:
        """Return number of points in the collection, ignoring filters for now."""
        try:
            info = await self.get_collection_info()
            return info.get("points_count", 0)
        except Exception as e:
            logger.error(f"Error counting points: {e}")
            raise StorageError(f"Failed to count points: {e}") from e
