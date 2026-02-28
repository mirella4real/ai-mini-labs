# Lab 01 — Local LLM CLI

Run a small language model entirely on your local machine. No API keys, no internet connection, no cost per token.

## What This Demonstrates

- Loading and running a HuggingFace model locally
- Device-aware inference (Apple Silicon MPS, NVIDIA CUDA, or CPU)
- Clean separation of model logic from CLI interface
- Reproducible environment with pinned dependencies

## Architecture
```
src/
  model.py  — model loading, device detection, inference logic
  cli.py    — argument parsing, user-facing output
tests/
  test_model.py — regression tests for model pipeline
```

The separation of `model.py` and `cli.py` is intentional: the model layer can be reused in an API or agent without touching the interface layer.

## How to Run
```bash
# From the repo root
cd 01-local-llm-cli
uv pip install -r requirements.txt

python -m src.cli --prompt "Software engineers improve reliability by" --max-new-tokens 120
```

## Model Choice and Tradeoffs

| Model | Size | Speed (CPU) | Output Quality |
|-------|------|-------------|----------------|
| distilgpt2 | ~300MB | Fast | Low — good for testing |
| TinyLlama-1.1B-Chat | ~2GB | Slow | Much better — instruction-aware |

This lab uses `distilgpt2` for speed. To use TinyLlama, pass `--model TinyLlama/TinyLlama-1.1B-Chat-v1.0`.

## Prompting This Model

distilgpt2 is a *completion* model, not an *instruction* model. It was trained on raw internet text, not on instruction-following pairs. This means:

❌ **Instruction-style** (poor results): `"Write 3 bullet points about reliability"`
✅ **Completion-style** (better results): `"Software engineers improve reliability by"`

This is a fundamental distinction in LLM engineering — the gap between base models and instruction-tuned models.

## How to Run on CPU vs GPU

- **Apple Silicon (M1/M2/M3):** MPS backend is used automatically — no configuration needed
- **NVIDIA GPU:** CUDA is used automatically if available
- **CPU only:** Falls back automatically — expect slower generation

## Common Failure Modes

- **Bus error on Apple Silicon:** Caused by running on CPU with float32. Fixed by using the MPS backend (already handled in model.py)
- **Out of memory:** Try a smaller `--max-new-tokens` value or use distilgpt2 instead of larger models
- **Slow generation:** Expected on CPU — distilgpt2 on MPS generates in under a second

## Running Tests
```bash
cd 01-local-llm-cli
pip install pytest
python -m pytest tests/ -v
```
