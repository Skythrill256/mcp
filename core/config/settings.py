from __future__ import annotations

import yaml  # type: ignore[import-untyped]
from typing import List, Optional
from pydantic import BaseModel, Field
import os


class AppSettings(BaseModel):
    """Configuration for scraping, embeddings, and vector storage."""

    # Database
    postgres_connection_string: str = Field(
        default_factory=lambda: os.getenv(
            "POSTGRES_CONNECTION_STRING",
        )
    )
    qdrant_url: Optional[str] = Field(default_factory=lambda: os.getenv("QDRANT_URL"))
    collection_name: str = Field(default="web_vectors")

    # Embeddings
    embedding_provider: str = Field(default="openai")
    embedding_model: str = Field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    )
    embedding_dimensions: int = Field(
        default_factory=lambda: int(os.getenv("EMBEDDING_DIM", "1536"))
    )
    batch_size: int = Field(default=32)
    chunk_size: int = Field(default=1200)
    chunk_overlap: int = Field(default=100)
    openai_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY")
    )
    huggingface_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("HF_API_KEY")
    )
    cohere_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("COHERE_API_KEY")
    )

    # Scraping/deep crawling
    keywords: List[str] = Field(default_factory=list)
    url_patterns: List[str] = Field(default_factory=list)
    include_external: bool = Field(default=False)
    max_pages: int = Field(default=50)
    max_depth: int = Field(default=2)

    # MCP / server
    mcp_port: Optional[int] = None

    # Logging
    log_level: str = Field(default="INFO")
    log_file: Optional[str] = Field(default=None)

    def for_site(
        self,
        url: str,
        collection_prefix: str = "site",
        collection_name: Optional[str] = None,
    ) -> "AppSettings":
        """Return a copy with a namespaced collection name derived from the URL host or use provided name."""
        from urllib.parse import urlparse

        host = urlparse(url).netloc.replace(":", "_").replace(".", "_") or "default"
        new = self.model_copy()
        # Use the provided collection name if available, otherwise generate one
        new.collection_name = (
            collection_name if collection_name else f"{collection_prefix}_{host}"
        )
        return new

    @classmethod
    def from_yaml(cls, path: str) -> "AppSettings":
        with open(path, "r") as f:
            config = yaml.safe_load(f) or {}

        provider = config.get("preferred_provider", "openai")
        providers_cfg = config.get("providers", {}) or {}
        provider_config = providers_cfg.get(provider, {}) or {}

        # Build kwargs and only include optional overrides when present so
        # pydantic will use the model's defaults when YAML doesn't specify them.
        kwargs: dict = {}
        kwargs["embedding_provider"] = provider

        if provider_config.get("model") is not None:
            kwargs["embedding_model"] = provider_config.get("model")

        if provider_config.get("dimensions") is not None:
            try:
                dim_val = provider_config.get("dimensions")
                kwargs["embedding_dimensions"] = int(str(dim_val))
            except (TypeError, ValueError):
                # ignore invalid value and fall back to model default
                pass
        else:
            # Set default dimensions based on model if not specified
            model_name = provider_config.get("model", "")
            if "jina-embeddings-v4" in model_name:
                kwargs["embedding_dimensions"] = 2048
            elif "all-mpnet-base-v2" in model_name:
                kwargs["embedding_dimensions"] = 3072
            elif "all-MiniLM-L6-v2" in model_name:
                kwargs["embedding_dimensions"] = 384

        openai_env = providers_cfg.get("openai", {}).get("api_key_env")
        hugging_env = providers_cfg.get("huggingFace", {}).get("api_key_env")

        kwargs["openai_api_key"] = os.getenv(openai_env) if openai_env else None
        kwargs["huggingface_api_key"] = os.getenv(hugging_env) if hugging_env else None

        return cls(**kwargs)
