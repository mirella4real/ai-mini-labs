# AI Mini Labs

A series of hands-on mini labs exploring practical AI/ML engineering — from local inference and embeddings to RAG pipelines, structured output, evals, agents, and API design.

Built to demonstrate applied AI skills across the full development lifecycle: environment setup, clean code, testing, and reproducible delivery.

## Labs

| # | Lab | Skills | Status |
|---|-----|--------|--------|
| 01 | Local LLM CLI | HF Transformers, local inference, CLI design | 🔜 |
| 02 | Embeddings Similarity | Sentence transformers, cosine similarity, semantic search | 🔜 |
| 03 | LangChain RAG | Document ingestion, chunking, retrieval, citations | 🔜 |
| 04 | Structured Output | Pydantic, schema extraction, validation, retry logic | 🔜 |
| 05 | Mini Evals | Evaluation harness, golden test cases, regression testing | 🔜 |
| 06 | Tool Use Agent | Agent loops, function calling, safety controls | 🔜 |
| 07 | FastAPI Microservice | API design, request/response schemas, observability | 🔜 |

## Stack

- **Python 3.11** (Apple Silicon / ARM64)
- **uv** for dependency management
- **ruff + black** for linting and formatting
- **pre-commit** for code quality enforcement

## Setup
```bash
git clone https://github.com/mirella4real/ai-mini-labs.git
cd ai-mini-labs
uv venv --python 3.11
source .venv/bin/activate
uv sync
```

Each lab has its own `README.md`, `requirements.txt`, and `src/` + `tests/` directories.
