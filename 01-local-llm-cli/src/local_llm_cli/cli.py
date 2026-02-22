import json
from datetime import datetime
from pathlib import Path

import typer

from local_llm_cli.infer import GenParams, generate

THIS_FILE = Path(__file__).resolve()
LAB_ROOT = THIS_FILE.parents[2]
DEFAULT_ARTIFACTS_DIR = LAB_ROOT / "artifacts"
PROMPT_OPT = typer.Option(..., "--prompt", "-p")
MODEL_OPT = typer.Option("distilgpt2", "--model")
MAX_NEW_TOKENS = typer.Option(120, "--max-new-tokens")
ARTIFACTS_DIR_OPT = typer.Option(DEFAULT_ARTIFACTS_DIR, "--artifacts-dir")

app = typer.Typer(add_completion=False)


def write_artifacts(artifacts_dir: Path, payload: dict) -> Path:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = artifacts_dir / f"run_{stamp}.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


@app.command()
def main(
    prompt: str = PROMPT_OPT,
    model: str = MODEL_OPT,
    max_new_tokens: int = MAX_NEW_TOKENS,
    artifacts_dir: Path = ARTIFACTS_DIR_OPT,
):
    result = generate(
        prompt,
        GenParams(model_id=model, max_new_tokens=max_new_tokens),
    )
    out = write_artifacts(artifacts_dir, result)
    print(result["completion"])
    print(f"Wrote: {out}")


if __name__ == "__main__":
    app()
