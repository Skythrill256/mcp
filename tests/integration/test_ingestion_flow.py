"""Integration tests for the web ingestion service."""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from core.services.ingestion import IngestionService
from core.config.settings import AppSettings
from core.models.ingestion import IngestRequest


@pytest.mark.asyncio
async def test_full_ingestion_flow():
    """Test the full ingestion flow with mocked dependencies."""
    # Create a real AppSettings instance
    config = AppSettings(
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        embedding_dimensions=1536,
        collection_name="test_collection"
    )
    
    # Create the service
    service = IngestionService(config)
    
    # Create a request
    request = IngestRequest(
        url="https://example.com",
        recreate=False,
        collection_prefix="test"
    )
    
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
        result = await service.ingest(request)
        
        # Verify the result structure
        assert result.site == "https://example.com"
        assert result.collection.startswith("test_")  # Should be namespaced
        assert "ingestion" in result.dict()
        assert "mcp" in result.dict()
        assert "http_url" in result.mcp


def test_app_settings_for_site_namespacing():
    """Test that the for_site method correctly namespaces collection names."""
    config = AppSettings()
    
    # Test various URL formats
    test_cases = [
        ("https://example.com", "site_example_com"),
        ("https://subdomain.example.com", "site_subdomain_example_com"),
        ("https://example.com:8080", "site_example_com_8080"),
        ("http://localhost:3000", "site_localhost_3000"),
    ]
    
    for url, expected_collection in test_cases:
        site_config = config.for_site(url, "site")
        assert site_config.collection_name == expected_collection