"""Tests for the IngestionService."""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from core.services.ingestion import IngestionService
from core.config.settings import AppSettings
from core.models.ingestion import IngestRequest
from core.errors.exceptions import ScrapingError, EmbeddingError, StorageError


@pytest.fixture
def mock_config():
    """Create a mock AppSettings for testing."""
    config = Mock(spec=AppSettings)
    config.embedding_provider = "openai"
    config.collection_name = "test_collection"
    config.for_site = Mock(return_value=config)
    config.qdrant_url = "http://localhost:6333"
    return config


@pytest.fixture
def ingest_request():
    """Create a sample IngestRequest for testing."""
    return IngestRequest(
        url="https://example.com",
        recreate=False,
        collection_prefix="test"
    )


@pytest.mark.asyncio
async def test_ingestion_service_init(mock_config):
    """Test that the IngestionService is initialized correctly."""
    service = IngestionService(mock_config)
    assert service.base_config == mock_config


@pytest.mark.asyncio
async def test_ingestion_service_ingest_success(mock_config, ingest_request):
    """Test successful ingestion flow."""
    service = IngestionService(mock_config)
    
    # Mock all the dependencies
    with (
        patch('core.services.ingestion.WebScraper') as mock_scraper_class,
        patch('core.services.ingestion.load_db_settings') as mock_load_db,
        patch('core.services.ingestion.QdrantVectorStore') as mock_qdrant_class,
        patch('core.services.ingestion.OpenAIEmbeddingProvider') as mock_openai_class,
        patch('core.services.ingestion.spawn_site_mcp') as mock_spawn_mcp
    ):
        # Mock the scraper
        mock_scraper_instance = AsyncMock()
        mock_scraper_instance.scrape_website.return_value = [
            {"url": "https://example.com", "content": "test content"}
        ]
        mock_scraper_class.return_value.__aenter__.return_value = mock_scraper_instance
        
        # Mock database settings
        mock_db_settings = Mock()
        mock_db_settings.db_type = "qdrant"
        mock_load_db.return_value = mock_db_settings
        
        # Mock the vector store
        mock_qdrant_instance = AsyncMock()
        mock_qdrant_instance.create_collection = AsyncMock()
        mock_qdrant_instance.store_embeddings = AsyncMock(return_value={"count": 1})
        mock_qdrant_class.return_value = mock_qdrant_instance
        
        # Mock the embedding provider
        mock_openai_instance = AsyncMock()
        mock_openai_instance.generate_embeddings = AsyncMock(return_value=[])
        mock_openai_class.return_value = mock_openai_instance
        
        # Mock the MCP spawner
        mock_spawn_mcp.return_value = {
            "host": "localhost",
            "port": 8000,
            "path": "/mcp"
        }
        
        # Call the method
        result = await service.ingest(ingest_request)
        
        # Verify the result
        assert result.site == "https://example.com/"
        assert result.collection == mock_config.collection_name
        assert "stored" in result.ingestion
        assert "http_url" in result.mcp
        
        # Verify that the methods were called
        mock_scraper_instance.scrape_website.assert_called_once_with("https://example.com")
        mock_qdrant_instance.create_collection.assert_called_once_with(recreate=False)
        mock_openai_instance.generate_embeddings.assert_called_once()
        mock_spawn_mcp.assert_called_once()


@pytest.mark.asyncio
async def test_ingestion_service_scraping_error(mock_config, ingest_request):
    """Test that scraping errors are properly handled."""
    service = IngestionService(mock_config)
    
    with patch('core.services.ingestion.WebScraper') as mock_scraper_class:
        # Mock the scraper to raise an exception
        mock_scraper_instance = AsyncMock()
        mock_scraper_instance.scrape_website.side_effect = ScrapingError("Test error")
        mock_scraper_class.return_value.__aenter__.return_value = mock_scraper_instance
        
        # Verify that the exception is propagated
        with pytest.raises(ScrapingError, match="Test error"):
            await service.ingest(ingest_request)


@pytest.mark.asyncio
async def test_ingestion_service_embedding_error(mock_config, ingest_request):
    """Test that embedding errors are properly handled."""
    service = IngestionService(mock_config)
    
    with (
        patch('core.services.ingestion.WebScraper') as mock_scraper_class,
        patch('core.services.ingestion.load_db_settings') as mock_load_db
    ):
        # Mock successful scraping
        mock_scraper_instance = AsyncMock()
        mock_scraper_instance.scrape_website.return_value = [
            {"url": "https://example.com", "content": "test content"}
        ]
        mock_scraper_class.return_value.__aenter__.return_value = mock_scraper_instance
        
        # Mock database settings
        mock_db_settings = Mock()
        mock_db_settings.db_type = "qdrant"
        mock_load_db.return_value = mock_db_settings
        
        # Mock the vector store to raise an exception
        with patch('core.services.ingestion.QdrantVectorStore') as mock_qdrant_class:
            mock_qdrant_class.side_effect = EmbeddingError("Test error")
            
            # Verify that the exception is propagated
            with pytest.raises(EmbeddingError, match="Test error"):
                await service.ingest(ingest_request)


@pytest.mark.asyncio
async def test_ingestion_service_unsupported_db_type(mock_config, ingest_request):
    """Test that unsupported database types raise an error."""
    service = IngestionService(mock_config)
    
    with (
        patch('core.services.ingestion.WebScraper') as mock_scraper_class,
        patch('core.services.ingestion.load_db_settings') as mock_load_db
    ):
        # Mock successful scraping
        mock_scraper_instance = AsyncMock()
        mock_scraper_instance.scrape_website.return_value = [
            {"url": "https://example.com", "content": "test content"}
        ]
        mock_scraper_class.return_value.__aenter__.return_value = mock_scraper_instance
        
        # Mock database settings with unsupported type
        mock_db_settings = Mock()
        mock_db_settings.db_type = "unsupported"
        mock_load_db.return_value = mock_db_settings
        
        # Verify that the exception is raised
        with pytest.raises(StorageError, match="Unsupported db_type: unsupported"):
            await service.ingest(ingest_request)


@pytest.mark.asyncio
async def test_ingestion_service_unsupported_embedding_provider(mock_config, ingest_request):
    """Test that unsupported embedding providers raise an error."""
    # Modify the mock config to use an unsupported provider
    mock_config.embedding_provider = "unsupported"
    
    service = IngestionService(mock_config)
    
    with (
        patch('core.services.ingestion.WebScraper') as mock_scraper_class,
        patch('core.services.ingestion.load_db_settings') as mock_load_db
    ):
        # Mock successful scraping
        mock_scraper_instance = AsyncMock()
        mock_scraper_instance.scrape_website.return_value = [
            {"url": "https://example.com", "content": "test content"}
        ]
        mock_scraper_class.return_value.__aenter__.return_value = mock_scraper_instance
        
        # Mock database settings
        mock_db_settings = Mock()
        mock_db_settings.db_type = "qdrant"
        mock_load_db.return_value = mock_db_settings
        
        # Add the missing qdrant_url attribute to the mock config
        mock_config.qdrant_url = "http://localhost:6333"
        
        # Verify that the exception is raised
        with pytest.raises(EmbeddingError, match="Unsupported embedding provider: unsupported"):
            await service.ingest(ingest_request)