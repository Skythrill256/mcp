# Tests

This directory contains tests for the web ingestion service.

## Structure

- `unit/` - Unit tests for individual components
- `integration/` - Integration tests for combined components

## Running Tests

To run all tests:

```bash
pytest
```

To run only unit tests:

```bash
pytest tests/unit
```

To run only integration tests:

```bash
pytest tests/integration
```

To run tests with coverage:

```bash
pytest --cov=core --cov-report=html
```

## Test Organization

Tests are organized to mirror the structure of the core application:

- `test_models.py` - Tests for Pydantic models
- `test_settings.py` - Tests for configuration
- `test_scraper.py` - Tests for the web scraper service
- `test_ingestion_service.py` - Tests for the main ingestion service
- `test_embedding_providers.py` - Tests for embedding providers
- `test_vector_stores.py` - Tests for vector store providers
- `test_main.py` - Tests for the FastAPI application
