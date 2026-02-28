"""
test_model.py — basic regression tests for model.py

Tests are intentionally lightweight: we verify the model loads,
runs inference, and returns output of the expected type and shape.
We do not test output quality (non-deterministic) but we do test
that the pipeline runs without errors.
"""

import torch
from src.model import load_model, generate, get_device


def test_get_device_returns_torch_device():
    """get_device() should always return a valid torch.device."""
    device = get_device()
    assert isinstance(device, torch.device)


def test_model_loads():
    """Model and tokenizer should load without errors."""
    tokenizer, model, device = load_model()
    assert tokenizer is not None
    assert model is not None
    assert isinstance(device, torch.device)


def test_generate_returns_string():
    """generate() should return a non-empty string."""
    tokenizer, model, device = load_model()
    result = generate(
        prompt="The key to good software is",
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_new_tokens=20,
    )
    assert isinstance(result, str)
    assert len(result) > 0


def test_generate_respects_max_new_tokens():
    """Output token count should not exceed max_new_tokens."""
    tokenizer, model, device = load_model()
    result = generate(
        prompt="Once upon a time",
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_new_tokens=10,
    )
    token_count = len(tokenizer.encode(result))
    assert token_count <= 10
