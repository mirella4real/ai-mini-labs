"""
cli.py — command-line interface for local LLM inference.

Parses arguments and delegates to model.py for inference.
Keeping CLI logic separate from model logic means either layer
can be changed independently.
"""

import argparse
import sys
from src.model import load_model, generate


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run local LLM inference using a HuggingFace model."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="The prompt to send to the model.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=120,
        help="Maximum number of new tokens to generate (default: 120).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="distilgpt2",
        help="HuggingFace model name to use (default: distilgpt2).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"\nPrompt: {args.prompt}\n")
    print("-" * 40)

    tokenizer, model, device = load_model(args.model)

    result = generate(
        prompt=args.prompt,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_new_tokens=args.max_new_tokens,
    )

    print(f"\nOutput:\n{result}\n")
    print("-" * 40)
    print(
        f"Model: {args.model} | Device: {device} | Max new tokens: {args.max_new_tokens}"
    )


if __name__ == "__main__":
    sys.exit(main())
