"""Tests for the vector store providers."""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from core.config.settings import AppSettings
from core.providers.qdrant import QdrantVectorStore
from core.providers.pgvector import PgVectorStoreProvider


@pytest.fixture
def mock_config():
    """Create a mock AppSettings for testing."""
    config = Mock(spec=AppSettings)
    config.collection_name = "test_collection"
    config.embedding_dimensions = 1536
    config.qdrant_url = "http://localhost:6333"
    config.postgres_connection_string = "postgresql://user:pass@localhost/db"
    return config


def test_qdrant_vector_store_init(mock_config):
    """Test that the QdrantVectorStore is initialized correctly."""
    with (
        patch('core.providers.qdrant.AsyncQdrantClient') as mock_client_class,
        patch('core.providers.qdrant.load_db_settings') as mock_load_db
    ):
        # Mock the Qdrant client
        mock_client_instance = Mock()
        mock_client_class.return_value = mock_client_instance
        
        # Mock database settings
        mock_db_settings = Mock()
        mock_db_settings.db_type = "qdrant"
        mock_db_settings.index_name = "default_collection"
        mock_load_db.return_value = mock_db_settings
        
        # Create the vector store
        vector_store = QdrantVectorStore(mock_config)
        
        # Verify that the client was initialized
        mock_client_class.assert_called_once_with(url=mock_config.qdrant_url)
        assert vector_store.config == mock_config


@pytest.mark.asyncio
async def test_qdrant_vector_store_create_collection(mock_config):
    """Test that a collection is created correctly."""
    with (
        patch('core.providers.qdrant.AsyncQdrantClient') as mock_client_class,
        patch('core.providers.qdrant.load_db_settings') as mock_load_db
    ):
        # Mock the Qdrant client
        mock_client_instance = Mock()
        mock_collections = Mock()
        mock_collections.collections = []
        mock_client_instance.get_collections = AsyncMock(return_value=mock_collections)
        mock_client_instance.create_collection = AsyncMock()
        mock_client_class.return_value = mock_client_instance
        
        # Mock database settings
        mock_db_settings = Mock()
        mock_db_settings.db_type = "qdrant"
        mock_db_settings.index_name = "default_collection"
        mock_load_db.return_value = mock_db_settings
        
        # Mock VectorParams and Distance
        with (
            patch('core.providers.qdrant.VectorParams') as mock_vector_params,
            patch('core.providers.qdrant.Distance') as mock_distance
        ):
            mock_vector_params_instance = Mock()
            mock_vector_params.return_value = mock_vector_params_instance
            mock_distance.COSINE = "Cosine"
            
            # Create the vector store
            vector_store = QdrantVectorStore(mock_config)
            
            # Call create_collection
            result = await vector_store.create_collection(recreate=False)
            
            # Verify that the collection was created
            mock_client_instance.create_collection.assert_called_once_with(
                collection_name=mock_config.collection_name,
                vectors_config=mock_vector_params_instance
            )
            assert result is True


@pytest.mark.asyncio
async def test_pgvector_store_provider_init(mock_config):
    """Test that the PgVectorStoreProvider is initialized correctly."""
    with (
        patch('core.providers.pgvector.load_db_settings') as mock_load_db,
        patch('core.providers.pgvector.PGVectorStore') as mock_pgvector_class
    ):
        # Mock database settings
        mock_db_settings = Mock()
        mock_db_settings.db_type = "pgvector"
        mock_db_settings.connection = "postgresql://user:pass@localhost/db"
        mock_db_settings.index_name = "default_table"
        mock_load_db.return_value = mock_db_settings
        
        # Mock PGVectorStore
        mock_pgvector_instance = Mock()
        mock_pgvector_class.return_value = mock_pgvector_instance
        
        # Create the vector store
        vector_store = PgVectorStoreProvider(mock_config)
        
        # Verify that PGVectorStore was initialized
        # Note: We can't directly test the vector_store property because it creates the PGVectorStore instance
        assert vector_store.config == mock_config
        assert vector_store.connection_string == mock_config.postgres_connection_string
        assert vector_store.table_name == mock_config.collection_name