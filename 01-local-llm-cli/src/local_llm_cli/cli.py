import json
from datetime import datetime
from pathlib import Path

import typer

THIS_FILE = Path(__file__).resolve()
LAB_ROOT = THIS_FILE.parents[2]
DEFAULT_ARTIFACTS_DIR = LAB_ROOT / "artifacts"
PROMPT_OPT = typer.Option(..., "--prompt", "-p")
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
    artifacts_dir: Path = ARTIFACTS_DIR_OPT,
):
    payload = {
        "prompt": prompt,
        "timestamp": datetime.now().isoformat(),
    }
    out = write_artifacts(artifacts_dir, payload)
    print(f"Wrote: {out}")


if __name__ == "__main__":
    app()
