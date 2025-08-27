class ScrapingError(Exception):
    """Raised when scraping fails."""


class EmbeddingError(Exception):
    """Raised when embedding operations fail."""


class StorageError(Exception):
    """Raised when vector storage operations fail."""


class QueryError(Exception):
    """Raised when query execution or query-engine initialization fails."""
