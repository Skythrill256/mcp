"""Tests for the AppSettings configuration."""

from core.config.settings import AppSettings


def test_app_settings_defaults():
    """Test that default values are correctly applied."""
    settings = AppSettings()

    # Test database defaults
    # Note: The actual value may be different due to environment variables
    assert "postgresql://" in settings.postgres_connection_string
    # qdrant_url can be set via environment variable, so just check it's a string or None
    assert settings.qdrant_url is None or isinstance(settings.qdrant_url, str)
    assert settings.collection_name == "web_vectors"

    # Test embedding defaults
    assert settings.embedding_provider == "openai"
    assert settings.embedding_model == "text-embedding-3-small"
    assert settings.embedding_dimensions == 1536
    assert settings.batch_size == 32
    assert settings.chunk_size == 1200
    assert settings.chunk_overlap == 100

    # Test scraping defaults
    assert settings.keywords == []
    assert settings.url_patterns == []
    assert settings.include_external is False
    assert settings.max_pages == 50
    assert settings.max_depth == 2

    # Test logging defaults
    assert settings.log_level == "INFO"
    assert settings.log_file is None


def test_app_settings_from_yaml_huggingface():
    """Test loading settings from YAML for HuggingFace provider."""
    settings = AppSettings.from_yaml("core/config/config-embedding-model.yaml")

    assert settings.embedding_provider == "huggingFace"
    assert settings.embedding_model == "all-MiniLM-L6-v2"
    assert settings.embedding_dimensions == 384


def test_app_settings_for_site():
    """Test the for_site method creates correctly namespaced settings."""
    settings = AppSettings()
    site_settings = settings.for_site("https://example.com/path", "test")

    # Should have the same settings except for collection name
    assert site_settings.embedding_provider == settings.embedding_provider
    assert site_settings.embedding_model == settings.embedding_model
    assert site_settings.collection_name == "test_example_com"

    # Test with different prefix
    site_settings2 = settings.for_site("https://subdomain.example.com", "web")
    assert site_settings2.collection_name == "web_subdomain_example_com"

    # Explicit collection_name should take precedence over prefix-derived
    site_settings3 = settings.for_site(
        "https://example.com",
        collection_prefix="ignored",
        collection_name="explicit_name",
    )
    assert site_settings3.collection_name == "explicit_name"


def test_app_settings_env_vars(monkeypatch):
    """Test that environment variables are correctly used."""
    # Set some environment variables
    monkeypatch.setenv(
        "POSTGRES_CONNECTION_STRING", "postgresql://test:test@localhost:5432/testdb"
    )
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("HF_API_KEY", "test-hf-key")

    settings = AppSettings()

    # Note: The actual value may be different due to environment variables
    assert (
        "postgresql://test:test@localhost:5432/testdb"
        in settings.postgres_connection_string
    )
    assert settings.qdrant_url == "http://localhost:6333"
    assert settings.openai_api_key == "test-openai-key"
    assert settings.huggingface_api_key == "test-hf-key"


def test_app_settings_from_yaml_model_dims(tmp_path, monkeypatch):
    """When dimensions are not provided, infer based on model name."""
    yml = tmp_path / "cfg.yaml"
    yml.write_text(
        """
preferred_provider: huggingFace
providers:
    huggingFace:
        model: jina-embeddings-v4
"""
    )
    settings = AppSettings.from_yaml(str(yml))
    assert settings.embedding_dimensions == 2048
