"""
Cohere reranking service for improving search result relevance.
"""

import logging
import os
from typing import List, Dict, Any

import cohere
from cohere.errors import BadRequestError, UnauthorizedError

from core.config.settings import AppSettings
from core.errors.exceptions import QueryError

logger = logging.getLogger(__name__)


class CohereReranker:
    """Reranking service using Cohere's rerank API."""

    def __init__(self, config: AppSettings):
        """Initialize the Cohere reranker.

        Args:
            config: Application configuration containing API keys
        """
        self.config = config
        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize the Cohere client with API key."""
        cohere_api_key = self.config.cohere_api_key or os.getenv("COHERE_API_KEY")
        if not cohere_api_key:
            raise ValueError(
                "Cohere API key not found. Please set COHERE_API_KEY in environment variables "
                "or in the configuration."
            )
        self.client = cohere.Client(api_key=cohere_api_key)

    def rerank_documents(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_n: int = 5,
        model: str = "rerank-v3.5",
        return_documents: bool = True,
    ) -> List[Dict[str, Any]]:
        """Rerank documents based on their relevance to the query.

        Args:
            query: The search query
            documents: List of documents to rerank. Can be strings or dictionaries with content
            top_n: Number of top results to return
            model: Cohere rerank model to use
            return_documents: Whether to return the document content in results

        Returns:
            List of reranked documents with relevance scores

        Raises:
            QueryError: If reranking fails
        """
        if not self.client:
            raise QueryError("Cohere client not initialized")

        if not documents:
            return []

        try:
            # Prepare documents for reranking
            # If documents are dictionaries, convert to strings
            prepared_docs = []
            for doc in documents:
                if isinstance(doc, dict):
                    # Try to get content from common fields
                    content = (
                        doc.get("text")
                        or doc.get("content")
                        or doc.get("body")
                        or str(doc)
                    )
                    prepared_docs.append(content)
                else:
                    prepared_docs.append(str(doc))

            # Call Cohere rerank API
            response = self.client.rerank(
                model=model,
                query=query,
                documents=prepared_docs,
                top_n=top_n,
                return_documents=return_documents,
            )

            # Process results
            reranked_results = []
            for result in response.results:
                doc_index = result.index
                original_doc = documents[doc_index]

                reranked_doc = {
                    "index": doc_index,
                    "relevance_score": result.relevance_score,
                    "document": original_doc,
                }

                # If return_documents is True, add the document content
                if return_documents and hasattr(result, "document") and result.document:
                    reranked_doc["content"] = result.document.text

                reranked_results.append(reranked_doc)

            return reranked_results

        except (BadRequestError, UnauthorizedError) as e:
            logger.error(f"Cohere API error during reranking: {str(e)}")
            raise QueryError(f"Failed to rerank documents: {str(e)}") from e
        except Exception as e:
            logger.error(f"Unexpected error during reranking: {str(e)}")
            raise QueryError(f"Failed to rerank documents: {str(e)}") from e

    async def arerank_documents(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_n: int = 5,
        model: str = "rerank-v3.5",
        return_documents: bool = True,
    ) -> List[Dict[str, Any]]:
        """Async version of document reranking.

        Args:
            query: The search query
            documents: List of documents to rerank
            top_n: Number of top results to return
            model: Cohere rerank model to use
            return_documents: Whether to return the document content in results

        Returns:
            List of reranked documents with relevance scores
        """
        # For now, we'll use the synchronous version
        # In the future, we could implement a truly async version
        return self.rerank_documents(query, documents, top_n, model, return_documents)
