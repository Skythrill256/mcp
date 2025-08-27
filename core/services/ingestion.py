from __future__ import annotations

from core.config.settings import AppSettings
from core.config.database import load_db_settings
from core.errors.exceptions import ScrapingError, EmbeddingError, StorageError
from core.models.ingestion import IngestRequest, IngestResponse
from core.providers.openai_embedding import OpenAIEmbeddingProvider
from core.providers.huggingface_embedding import HuggingFaceEmbeddingProvider
from core.providers.qdrant import QdrantVectorStore
from core.providers.pgvector import PgVectorStoreProvider
from core.services.mcp_server import spawn_site_mcp
from core.services.scraper import WebScraper


class IngestionService:
    def __init__(self, base_config: AppSettings):
        self.base_config = base_config

    async def ingest(self, req: IngestRequest) -> IngestResponse:
        # Apply optional overrides from request
        if req.max_pages is not None:
            self.base_config.max_pages = req.max_pages
        if req.max_depth is not None:
            self.base_config.max_depth = req.max_depth
        if req.include_external is not None:
            self.base_config.include_external = req.include_external
        if req.keywords is not None:
            self.base_config.keywords = req.keywords
        if req.url_patterns is not None:
            self.base_config.url_patterns = req.url_patterns

        cfg = self.base_config.for_site(str(req.url), collection_prefix=req.collection_prefix)

        # 1) Scrape
        try:
            async with WebScraper(cfg) as scraper:
                scraped = await scraper.scrape_website(str(req.url))
        except ScrapingError as e:
            raise e
        except Exception as e:
            raise ScrapingError(f"Unexpected scraping error: {e}") from e

        # 2) Ingest into vector store
        try:
            db_settings = load_db_settings()
            if db_settings.db_type == "qdrant":
                vector_store = QdrantVectorStore(cfg)
            elif db_settings.db_type == "pgvector":
                vector_store = PgVectorStoreProvider(cfg)
            else:
                raise StorageError(f"Unsupported db_type: {db_settings.db_type}")

            if self.base_config.embedding_provider == "openai":
                embed_mgr = OpenAIEmbeddingProvider(cfg)
            elif self.base_config.embedding_provider == "huggingFace":
                embed_mgr = HuggingFaceEmbeddingProvider(cfg)
            else:
                raise EmbeddingError(f"Unsupported embedding provider: {self.base_config.embedding_provider}")

            await vector_store.create_collection(recreate=req.recreate)
            embeddings = await embed_mgr.generate_embeddings(scraped)
            ingestion_info = await vector_store.store_embeddings(embeddings)
        except (EmbeddingError, StorageError) as e:
            raise e
        except Exception as e:
            raise EmbeddingError(f"Unexpected embedding error: {e}") from e

        # 3) Spawn MCP server for this site with tools
        try:
            mcp_info = await spawn_site_mcp(str(req.url), self.base_config)
        except Exception as e:
            raise Exception(f"Failed to start MCP server: {e}") from e

        return IngestResponse(
            site=str(req.url),
            collection=cfg.collection_name,
            ingestion={"stored": ingestion_info},
            mcp={
                **mcp_info,
                "http_url": f"http://{mcp_info['host']}:{mcp_info['port']}{mcp_info['path']}",
            },
        )
