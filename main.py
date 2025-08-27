from __future__ import annotations

from dotenv import load_dotenv

# Load environment from .env early so modules that import config pick up values
load_dotenv()

from core.config.settings import AppSettings
from core.logging import configure_logging

# Configure logging
settings = AppSettings()
configure_logging(settings)

from fastapi import FastAPI, HTTPException

from core.models.ingestion import IngestRequest, IngestResponse
from core.services.ingestion import IngestionService

app = FastAPI(title="Web Ingestion + MCP Spawner", version="1.0.0")


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest):
    base_cfg = AppSettings.from_yaml("core/config/config-embedding-model.yaml")
    service = IngestionService(base_cfg)
    try:
        return await service.ingest(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Optional local runner for development: uvicorn main:app --reload --port 8000
