"""Tests for the main FastAPI application."""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from fastapi.testclient import TestClient
from main import app
from core.models.ingestion import IngestResponse
from core.config.settings import AppSettings


@pytest.fixture
def client():
    """Create a TestClient for the FastAPI app."""
    return TestClient(app)


def test_healthz_endpoint(client):
    """Test the healthz endpoint."""
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ingest_endpoint_success(client):
    """Test the ingest endpoint with a successful request."""
    # Mock the AppSettings.from_yaml and IngestionService
    with (
        patch("main.AppSettings.from_yaml") as mock_from_yaml,
        patch("main.IngestionService") as mock_service_class,
    ):
        # Mock the config
        mock_config = Mock(spec=AppSettings)
        mock_from_yaml.return_value = mock_config

        # Mock the service
        mock_service_instance = AsyncMock()
        mock_response = IngestResponse(
            site="https://example.com",
            collection="test_collection",
            ingestion={"stored": {"count": 5}},
            mcp={
                "host": "localhost",
                "port": 8000,
                "path": "/mcp",
                "http_url": "http://localhost:8000/mcp",
            },
        )
        mock_service_instance.ingest.return_value = mock_response
        mock_service_class.return_value = mock_service_instance

        # Make the request
        request_data = {
            "url": "https://example.com",
            "recreate": False,
            "collection_prefix": "test",
            "max_pages": 5,
            "max_depth": 2,
        }

        response = client.post("/ingest", json=request_data)

        # Verify the response
        assert response.status_code == 200
        assert response.json() == {
            "site": "https://example.com",
            "collection": "test_collection",
            "ingestion": {"stored": {"count": 5}},
            "mcp": {
                "host": "localhost",
                "port": 8000,
                "path": "/mcp",
                "http_url": "http://localhost:8000/mcp",
            },
        }

        # Verify that the service methods were called
        mock_from_yaml.assert_called_once_with(
            "core/config/config-embedding-model.yaml"
        )
        mock_service_instance.ingest.assert_called_once()


def test_ingest_endpoint_validation_error(client):
    """Test the ingest endpoint with invalid data."""
    request_data = {"url": "not-a-url", "recreate": False}

    response = client.post("/ingest", json=request_data)

    assert response.status_code == 422  # Validation error


def test_ingest_endpoint_server_error(client):
    """Test the ingest endpoint when the service raises an exception."""
    with (
        patch("main.AppSettings.from_yaml") as mock_from_yaml,
        patch("main.IngestionService") as mock_service_class,
    ):
        # Mock the config
        mock_config = Mock(spec=AppSettings)
        mock_from_yaml.return_value = mock_config

        # Mock the service to raise an exception
        mock_service_instance = AsyncMock()
        mock_service_instance.ingest.side_effect = Exception("Test error")
        mock_service_class.return_value = mock_service_instance

        # Make the request
        request_data = {"url": "https://example.com", "recreate": False}

        response = client.post("/ingest", json=request_data)

        # Verify the response
        assert response.status_code == 500
        assert "Test error" in response.json()["detail"]
