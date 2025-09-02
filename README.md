# Pebblify

A comprehensive web scraping and vector database package that ingests web content and stores it in vector databases for AI applications.

## Features

- Web scraping using Crawl4AI
- Vector storage with Qdrant and PostgreSQL
- OpenAI and HuggingFace embeddings
- FastAPI REST API
- Configurable ingestion pipeline

## Prerequisites

- Python 3.10 - 3.15
- uv
- OpenAi api key


## Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd <repository-url>
```



### 3. Install Dependencies
Using uv (recommended for faster installs):
```bash
pip install uv  # If not already installed
uv sync .
```
### 4. Environment Configuration

Create a `.env` file in the project root with the required environment variables:

```env
# OpenAI
OPENAI_API_KEY=your_openai_api_key

# Database (PostgreSQL)
POSTGRES_CONNECTION_STRING=postgresql://user:password@localhost:5432/database_name

# Qdrant
QDRANT_URL=http://localhost:6333




### 5. Configuration Files

The project uses YAML configuration files located in `core/config/`:

- `config-embedding-model.yaml`: Embedding model configuration
- `config-database.yaml`: Database configuration

Review and modify these files according to your needs.

## Usage

### Running the API Server

```bash
uvicorn main:app --reload --port 8000
```

Or using the Makefile:
```bash
make run
```

The API will be available at `http://localhost:8000`.

### API Endpoints

- `GET /healthz`: Health check endpoint
- `POST /ingest`: Ingest web content

Example ingest request:
```json
{
  "url": "https://example.com",
  "depth": 1,
  "max_pages": 10
}
```

### Using the Ingestion Service Directly

```python
from core.config.settings import AppSettings
from core.services.ingestion import IngestionService
from core.models.ingestion import IngestRequest

# Load configuration
base_cfg = AppSettings.from_yaml("core/config/config-embedding-model.yaml")

# Create service
service = IngestionService(base_cfg)

# Create request
req = IngestRequest(
    url="https://example.com",
    depth=1,
    max_pages=10
)

# Ingest content
result = await service.ingest(req)
```

## Development

### Running Tests

Unit tests:
```bash
pytest tests/unit
```

Integration tests:
```bash
pytest tests/integration
```

All tests:
```bash
pytest
```

### Code Quality

The project uses several tools to ensure code quality:

- Ruff for linting and formatting
- MyPy for type checking
- Pydocstyle for docstring conventions
- Bandit for security issues

Run all quality checks:
```bash
make lint
```

Or run individual tools:
```bash
make ruff
make mypy
make pydocstyle
make bandit
```

## Project Structure

```
pebblify/
├── core/                 # Core application logic
│   ├── config/           # Configuration files and settings
│   ├── embeddings/       # Embedding model integrations
│   ├── errors/           # Custom error definitions
│   ├── mcp/              # MCP (Model Context Protocol) related code
│   ├── models/           # Data models and schemas
│   ├── providers/        # External service providers
│   ├── services/         # Business logic services
│   ├── utils/            # Utility functions
│   ├── logging.py        # Logging configuration
│   └── __init__.py
├── tests/                # Test suite
│   ├── unit/             # Unit tests
│   ├── integration/      # Integration tests
│   └── conftest.py       # Test configuration
├── main.py               # Application entry point
├── pyproject.toml        # Project metadata and dependencies
├── Makefile              # Development helpers
└── README.md             # This file
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Commit your changes
6. Push to the branch
7. Create a pull request

