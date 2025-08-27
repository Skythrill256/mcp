from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional

from core.config.settings import AppSettings
from core.models.ingestion import EmbeddingResult


class VectorStoreProvider(ABC):
    def __init__(self, config: AppSettings):
        self.config = config

    @abstractmethod
    async def create_collection(self, recreate: bool = False) -> bool:
        pass

    @abstractmethod
    async def store_embeddings(self, embeddings: List[EmbeddingResult]) -> int:
        pass

    @abstractmethod
    async def search_similar(
        self,
        query_embedding: List[float],
        limit: int = 10,
        score_threshold: float = 0.7,
        filters: Optional[dict[str, Any]] = None,
    ) -> List[dict[str, Any]]:
        pass

    @abstractmethod
    async def get_collection_info(self) -> dict[str, Any]:
        pass

    @abstractmethod
    async def delete_collection(self) -> bool:
        pass

    @abstractmethod
    async def count_points(self, filters: Optional[dict[str, Any]] = None) -> int:
        pass
