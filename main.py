from __future__ import annotations

import warnings
from dotenv import load_dotenv
from typing import List
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from core.models.ingestion import IngestRequest, IngestResponse
from core.services.ingestion import IngestionService

# Load environment from .env early so modules that import config pick up values
load_dotenv()

# Suppress Pydantic v1-style config deprecation warnings from dependencies
warnings.filterwarnings(
    "ignore",
    message="Support for class-based `config` is deprecated",
    category=DeprecationWarning,
)

from core.config.settings import AppSettings  # noqa: E402
from core import configure_logging, info  # noqa: E402

# Configure logging
settings = AppSettings()
configure_logging(settings)

info("Starting pebblify")


class ScrapedData(BaseModel):
    url: str
    title: str
    description: str
    content: str


class EmbeddingRequest(BaseModel):
    data: List[ScrapedData]
    collection_name: str = None


app = FastAPI(title="Web Ingestion + MCP Spawner", version="1.0.0")


@app.get("/healthz")
async def healthz():
    info("Health check requested")
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest):
    info(f"Ingestion request received for URL: {req.url}")
    base_cfg = AppSettings.from_yaml("core/config/config-embedding-model.yaml")
    service = IngestionService(base_cfg)
    try:
        result = await service.ingest(req)
        info(f"Ingestion completed successfully for URL: {req.url}")
        return result
    except Exception as e:
        info(f"Ingestion failed for URL: {req.url} with error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embedding", response_model=IngestResponse)
async def embedding(req: EmbeddingRequest):
    # Generate collection name with site_ prefix if not provided
    collection_name = req.collection_name
    if not collection_name and req.data:
        # Generate collection name from the first URL
        from urllib.parse import urlparse

        first_url = req.data[0].url
        host = urlparse(first_url).netloc.replace(":", "_").replace(".", "_")
        collection_name = f"site_{host}"

    info(
        f"Embedding request received for {len(req.data)} documents with collection name: {collection_name}"
    )
    base_cfg = AppSettings.from_yaml("core/config/config-embedding-model.yaml")
    service = IngestionService(base_cfg)
    try:
        # Pass the collection name to a new method
        result = await service.embed_scraped_data_with_collection(
            req.data, collection_name
        )
        info("Embedding completed successfully")
        return result
    except Exception as e:
        info(f"Embedding failed with error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Optional local runner for development: uvicorn main:app --reload --port 8000
