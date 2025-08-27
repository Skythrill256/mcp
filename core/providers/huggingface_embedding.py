
"""
Embedding generation using Hugging Face models.
"""

import asyncio
import logging
import os
from typing import Any, List

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError(
        "sentence-transformers is required. Install with: pip install sentence-transformers"
    ) from None

from core.config.settings import AppSettings
from core.errors.exceptions import EmbeddingError
from core.providers.embedding import EmbeddingProvider, EmbeddingResult

logger = logging.getLogger(__name__)

class HuggingFaceEmbeddingProvider(EmbeddingProvider):
    """Manages embedding generation using Hugging Face models."""

    def __init__(self, config: AppSettings):
        """Initialize the embedding manager."""
        super().__init__(config)

        if self.config.huggingface_api_key and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
            os.environ["HUGGING_FACE_HUB_TOKEN"] = self.config.huggingface_api_key

        try:
            self.embed_model = SentenceTransformer(self.config.embedding_model)
        except Exception as e:
            raise EmbeddingError(f"Failed to load Hugging Face model: {e}") from e

    async def generate_embeddings(self, scraped_data: List[dict[str, Any]]) -> List[EmbeddingResult]:
        """
        Generate embeddings for scraped content using Hugging Face.
        """
        try:
            texts = [item['content'] for item in scraped_data]
            embeddings = await asyncio.to_thread(self.embed_model.encode, texts)

            results = []
            for i, item in enumerate(scraped_data):
                results.append(
                    EmbeddingResult(
                        text=item['content'],
                        embedding=embeddings[i].tolist(),
                        token_count=len(item['content'].split()), # a rough estimate
                        chunk_id=str(i),
                        metadata=item.get('metadata', {})
                    )
                )
            return results
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise EmbeddingError(f"Failed to generate embeddings: {str(e)}") from e

    async def generate_query_embedding(self, query: str) -> List[float]:
        """
        Generate embedding for a query string using Hugging Face.
        """
        try:
            embedding = await asyncio.to_thread(self.embed_model.encode, query)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating query embedding: {str(e)}")
            raise EmbeddingError(f"Failed to generate query embedding: {str(e)}") from e
