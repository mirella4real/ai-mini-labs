"""
model.py — loads a local HuggingFace model and runs text generation inference.

Kept separate from cli.py so the model logic can be reused or swapped
without touching the interface layer.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


MODEL_NAME = "distilgpt2"


def get_device() -> torch.device:
    """
    Detect the best available device.
    - Apple Silicon (M1/M2/M3): uses MPS (Metal Performance Shaders)
    - NVIDIA GPU: uses CUDA
    - Fallback: CPU
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def load_model(model_name: str = MODEL_NAME):
    """
    Load tokenizer and model from HuggingFace (cached locally after first download).
    Returns (tokenizer, model, device) tuple.
    """
    device = get_device()
    print(f"Loading model: {model_name} on device: {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model = model.to(device)
    model.eval()
    print("Model loaded.")
    return tokenizer, model, device


def generate(
    prompt: str,
    tokenizer,
    model,
    device: torch.device,
    max_new_tokens: int = 120,
) -> str:
    """
    Run inference on the given prompt.
    Returns the generated text (excluding the original prompt).
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0][input_length:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True)
