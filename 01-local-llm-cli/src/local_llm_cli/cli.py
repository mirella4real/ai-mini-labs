import typer

app = typer.Typer(add_completion=False)


@app.command()
def main(name: str = "world"):
    print(f"hello, {name}")


if __name__ == "__main__":
    app()
