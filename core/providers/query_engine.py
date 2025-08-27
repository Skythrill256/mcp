from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional

from core.config.settings import AppSettings


class QueryEngineProvider(ABC):
    def __init__(self, config: AppSettings):
        self.config = config

    @abstractmethod
    async def query(
        self,
        question: str,
        top_k: int = 5,
        similarity_threshold: float = 0.7,
        include_sources: bool = True,
        filters: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        pass

    @abstractmethod
    async def similarity_search(
        self,
        query: str,
        top_k: int = 10,
        similarity_threshold: float = 0.7,
        filters: Optional[dict[str, Any]] = None,
    ) -> List[dict[str, Any]]:
        pass
