import time
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class GenParams:
    model_id: str = "distilgpt2"
    max_new_tokens: int = 120
    temperature: float = 0.7
    top_p: float = 0.95
    seed: int = 42


def generate(prompt: str, params: GenParams) -> dict[str, Any]:
    if not prompt.strip():
        raise ValueError("prompt must be non-empty")

    torch.manual_seed(params.seed)

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(params.model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(params.model_id)
    model.eval()

    inputs = tok(prompt, return_tensors="pt")

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=params.max_new_tokens,
            do_sample=True,
            temperature=params.temperature,
            top_p=params.top_p,
            pad_token_id=tok.eos_token_id,
        )

    text = tok.decode(out_ids[0], skip_special_tokens=True)
    completion = text[len(prompt) :] if text.startswith(prompt) else text

    return {
        "model_id": params.model_id,
        "prompt": prompt,
        "completion": completion.strip(),
        "timing_s": round(time.time() - t0, 4),
        "params": {
            "max_new_tokens": params.max_new_tokens,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "seed": params.seed,
        },
    }
