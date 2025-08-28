from __future__ import annotations

from dotenv import load_dotenv

# Load environment from .env early so modules that import config pick up values
load_dotenv()

from core.config.settings import AppSettings
from core import configure_logging, info

# Configure logging
settings = AppSettings()
configure_logging(settings)

info("Starting web ingestion application")

from fastapi import FastAPI, HTTPException

from core.models.ingestion import IngestRequest, IngestResponse
from core.services.ingestion import IngestionService

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


# Optional local runner for development: uvicorn main:app --reload --port 8000
