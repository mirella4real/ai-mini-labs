# AI Mini Labs — Applied AI Systems Engineering

This repository is a structured exploration of modern AI system design.

The goal is not to build toy scripts, but to explore production-minded AI engineering patterns including:

- Local model inference
- Embeddings and retrieval
- LangChain orchestration
- Structured outputs and validation
- Evaluation and regression testing
- Tool use and agent patterns
- API wrapping and service design

Each mini-lab is designed to be:
- Small (<1 hour to implement)
- Architecturally meaningful
- Reproducible
- Professionally structured
- Focused on reliability and engineering discipline

---
## Repository Structure

```yaml
ai-mini-labs:
  00-setup: "Shared standards and tooling"
  01-local-llm-cli: 
  shared: "Shared utilities (config, logging)"
  ```
---
  ## Engineering Standards

This repo enforces:

- Python 3.11 runtime standardization
- uv-based deterministic dependency management
- Ruff + Black formatting
- Pre-commit local quality gates
- Environment configuration pattern via `.env`

See `00-setup/` for details.

---
## Why This Repo Exists

This lab demonstrates architectural thinking around:

- Tradeoffs (local vs hosted models)
- Evaluation discipline
- Guardrails and validation
- Observability mindset
- Developer workflow automation

It is structured to resemble a small AI platform initiative rather than disconnected experiments.

---

## Getting Started

```bash
brew install uv
uv venv
uv sync
uv run pre-commit install
```