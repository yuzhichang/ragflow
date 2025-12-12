# RAGFlow Project Instructions for GitHub Copilot

This file provides context, build instructions, and coding standards for the RAGFlow project.
It is structured to follow GitHub Copilot's [customization guidelines](https://docs.github.com/en/copilot/concepts/prompting/response-customization).

## 1. Project Overview
RAGFlow is an open-source RAG (Retrieval-Augmented Generation) engine based on deep document understanding. It is a full-stack application with a Python backend and a React/TypeScript frontend.

- **Backend**: Python 3.10+ (Flask/Quart)
- **Frontend**: TypeScript, React, UmiJS
- **Architecture**: Microservices based on Docker.
  - `api/`: Backend API server.
  - `rag/`: Core RAG logic (indexing, retrieval).
  - `deepdoc/`: Document parsing and OCR.
  - `web/`: Frontend application.

## 2. Directory Structure
- `api/`: Backend API server (Flask/Quart).
  - `apps/`: API Blueprints (Knowledge Base, Chat, etc.).
  - `db/`: Database models and services.
- `rag/`: Core RAG logic.
  - `llm/`: LLM, Embedding, and Rerank model abstractions.
- `deepdoc/`: Document parsing and OCR modules.
- `agent/`: Agentic reasoning components.
- `web/`: Frontend application (React + UmiJS).
- `docker/`: Docker deployment configurations.
- `sdk/`: Python SDK.
- `test/`: Backend tests.

## 3. Build Instructions

### Backend (Python)
The project uses **uv** for dependency management.

1. **Setup Environment**:
   ```bash
   uv sync --python 3.12 --all-extras
   uv run download_deps.py
   ```

2. **Run Server**:
   - **Pre-requisite**: Start dependent services (MySQL, ES/Infinity, Redis, MinIO).
     ```bash
     docker compose -f docker/docker-compose-base.yml up -d
     ```
   - **Launch**:
     ```bash
     source .venv/bin/activate
     export PYTHONPATH=$(pwd)
     bash docker/launch_backend_service.sh
     ```

### Frontend (TypeScript/React)
Located in `web/`.

1. **Install Dependencies**:
   ```bash
   cd web
   npm install
   ```

2. **Run Dev Server**:
   ```bash
   npm run dev
   ```
   Runs on port 8000 by default.

### Docker Deployment
To run the full stack using Docker:
```bash
cd docker
docker compose -f docker-compose.yml up -d

## 4. Testing Instructions

### Backend Tests
- **Testing RAGFlow deployed on kubernetes**:
  To test RAGFlow deployed on kubernetes, first set the environment variable `HOST_ADDRESS=http://ragflow.local`.
- **Install test dependenciess**:
  ```bash
  uv sync --python 3.12 --only-group test --no-default-groups --frozen
  uv pip install sdk/python --group test
  ```
- **Run All Tests**:
  ```bash
  source .venv/bin/activate
  python run_tests.py
  pytest -s --tb=short test/testcases/test_sdk_api
  pytest -s --tb=short test/testcases/test_http_api
  pytest -s --tb=short sdk/python/test/test_frontend_api/get_email.py sdk/python/test/test_frontend_api/test_dataset.py
  ```
- **Run Specific Test**:
  ```bash
  source .venv/bin/activate
  pytest -s --tb=short test/testcases/test_sdk_api/test_chunk_management_within_dataset/test_retrieval_chunks.py -k test_page[payload3-0-]
  ```

### Frontend Tests
- **Run Tests**:
  ```bash
  cd web
  npm run test
  ```

## 5. Coding Standards & Guidelines
- **Python Formatting**: Use `ruff` for linting and formatting.
  ```bash
  ruff check
  ruff format
  ```
- **Frontend Linting**:
  ```bash
  cd web
  npm run lint
  ```
- **kubectl Timeout**: For kubectl operations that may hang (e.g., delete, apply, wait), ALWAYS set `--timeout=30s` to avoid indefinite waiting. If timed out, investigate the cause instead of waiting.
- **Pulumi Timeout**: For Pulumi operations that may hang (e.g., up, destroy), ALWAYS use `timeout 30s` to avoid indefinite waiting. If timed out, investigate the cause instead of waiting.
- **Pre-commit**: Ensure pre-commit hooks are installed.
  ```bash
  pre-commit install
  pre-commit run --all-files
  ```

