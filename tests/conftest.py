"""Shared test configurations and fixtures."""

import pytest
from core.config.settings import AppSettings


@pytest.fixture
def app_settings():
    """Create a default AppSettings instance for testing."""
    return AppSettings()


@pytest.fixture
def sample_ingest_request():
    """Create a sample IngestRequest for testing."""
    from core.models.ingestion import IngestRequest

    return IngestRequest(
        url="https://example.com",
        recreate=False,
        collection_prefix="test",
        max_pages=5,
        max_depth=2,
    )
