from core.models.ingestion import EmbeddingResult

__all__ = ["EmbeddingResult", "EmbeddingManager"]


class EmbeddingManager:  # minimal stub to satisfy type checking where imported
    def __init__(self, *args, **kwargs):  # pragma: no cover - used in alt server module
        pass

    async def query_llamaindex(self, *args, **kwargs):  # pragma: no cover
        return {"answer": "", "sources": []}
