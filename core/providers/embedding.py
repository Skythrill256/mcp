from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List

from core.config.settings import AppSettings
from core.embeddings.embedding import EmbeddingResult


class EmbeddingProvider(ABC):
    def __init__(self, config: AppSettings):
        self.config = config

    @abstractmethod
    async def generate_embeddings(
        self, scraped_data: List[dict[str, Any]]
    ) -> List[EmbeddingResult]:
        pass

    @abstractmethod
    async def generate_query_embedding(self, query: str) -> List[float]:
        pass
