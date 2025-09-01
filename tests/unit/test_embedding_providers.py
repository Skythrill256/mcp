"""Tests for the embedding providers."""

import pytest
from unittest.mock import Mock, patch
from core.config.settings import AppSettings
from core.providers.openai_embedding import OpenAIEmbeddingProvider
from core.providers.huggingface_embedding import HuggingFaceEmbeddingProvider


@pytest.fixture
def mock_config():
    """Create a mock AppSettings for testing."""
    config = Mock(spec=AppSettings)
    config.openai_api_key = "test-openai-key"
    config.huggingface_api_key = "test-hf-key"
    config.embedding_model = "text-embedding-3-small"
    config.embedding_dimensions = 1536
    config.batch_size = 32
    config.chunk_size = 1200
    config.chunk_overlap = 100
    config.collection_name = "test_collection"
    config.qdrant_url = "http://localhost:6333"
    return config


def test_openai_embedding_provider_init(mock_config):
    """Test that the OpenAIEmbeddingProvider is initialized correctly."""
    with patch("core.providers.openai_embedding.OpenAIEmbedding") as mock_openai_class:
        mock_openai_instance = Mock()
        mock_openai_class.return_value = mock_openai_instance

        provider = OpenAIEmbeddingProvider(mock_config)

        # Verify that the OpenAIEmbedding was initialized with correct parameters
        mock_openai_class.assert_called_once_with(
            model=mock_config.embedding_model,
            dimensions=mock_config.embedding_dimensions,
            embed_batch_size=mock_config.batch_size,
        )
        assert provider.config == mock_config


@pytest.mark.asyncio
async def test_openai_embedding_provider_generate_embeddings(mock_config):
    """Test that embeddings are generated correctly."""
    with (
        patch("core.providers.openai_embedding.OpenAIEmbedding") as mock_openai_class,
        patch("core.providers.openai_embedding.load_db_settings") as mock_load_db,
        patch("core.providers.openai_embedding.QdrantClient") as mock_qdrant_client,
    ):
        # Mock the OpenAI embedding model
        mock_openai_instance = Mock()
        mock_openai_instance.get_text_embedding.return_value = [0.1, 0.2, 0.3]
        mock_openai_class.return_value = mock_openai_instance

        # Mock database settings
        mock_db_settings = Mock()
        mock_db_settings.db_type = "qdrant"
        mock_db_settings.index_name = "default_collection"
        mock_load_db.return_value = mock_db_settings

        # Mock Qdrant client
        mock_qdrant_instance = Mock()
        mock_qdrant_client.return_value = mock_qdrant_instance

        provider = OpenAIEmbeddingProvider(mock_config)

        # Test data
        scraped_data = [
            {"content": "Test content 1", "metadata": {"url": "https://example.com/1"}},
            {"content": "Test content 2", "metadata": {"url": "https://example.com/2"}},
        ]

        # Mock the SentenceSplitter
        with patch(
            "core.providers.openai_embedding.SentenceSplitter"
        ) as mock_splitter_class:
            mock_splitter_instance = Mock()
            mock_node1 = Mock()
            mock_node1.get_text.return_value = "Test content 1"
            mock_node1.metadata = {"url": "https://example.com/1"}
            mock_node1.node_id = "node_1"

            mock_node2 = Mock()
            mock_node2.get_text.return_value = "Test content 2"
            mock_node2.metadata = {"url": "https://example.com/2"}
            mock_node2.node_id = "node_2"

            mock_splitter_instance.get_nodes_from_documents.return_value = [
                mock_node1,
                mock_node2,
            ]
            mock_splitter_class.return_value = mock_splitter_instance

            # Mock the LlamaIndex Settings
            with patch("core.providers.openai_embedding.Settings"):
                # Call the method
                result = await provider.generate_embeddings(scraped_data)

                # Verify the result
                assert len(result) == 2
                assert result[0].text == "Test content 1"
                assert result[0].embedding == [0.1, 0.2, 0.3]
                assert result[0].chunk_id == "node_1"
                assert result[1].text == "Test content 2"
                assert result[1].embedding == [0.1, 0.2, 0.3]
                assert result[1].chunk_id == "node_2"


@pytest.mark.asyncio
async def test_huggingface_embedding_provider_init(mock_config):
    """Test that the HuggingFaceEmbeddingProvider is initialized correctly."""
    with patch(
        "core.providers.huggingface_embedding.SentenceTransformer"
    ) as mock_sentence_transformer_class:
        mock_sentence_transformer_instance = Mock()
        mock_sentence_transformer_class.return_value = (
            mock_sentence_transformer_instance
        )

        # Change config for HuggingFace
        mock_config.embedding_model = "all-MiniLM-L6-v2"
        mock_config.embedding_dimensions = 384
        test_key = "test-hf-key"  # nosec B105
        mock_config.huggingface_api_key = test_key

        provider = HuggingFaceEmbeddingProvider(mock_config)

        # Verify that the SentenceTransformer was initialized with correct parameters
        mock_sentence_transformer_class.assert_called_once_with(
            mock_config.embedding_model, use_auth_token=test_key
        )
        assert provider.config == mock_config
