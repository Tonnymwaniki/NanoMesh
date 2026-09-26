import typer
from rich.console import Console
from rich.text import Text

app = typer.Typer()
console = Console()


@app.command()
def hello():
    console.print(Text("NanoMesh is working!", style="green"))


if __name__ == "__main__":
    app()
