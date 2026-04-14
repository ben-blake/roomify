"""Roomify CLI — thin typer shell over the service layer.

Usage:
    python -m roomify.cli --help
    python -m roomify.cli generate --spec configs/examples/bedroom_01.yaml
    python -m roomify.cli sweep --config configs/experiments/core.yaml
    python -m roomify.cli evaluate --run outputs/<runId>
    python -m roomify.cli report --run outputs/<runId>
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    name="roomify",
    help="Data-driven interior design image generator using Stable Diffusion.",
    add_completion=False,
)


@app.command()
def generate(
    spec: Path = typer.Option(
        ...,
        "--spec",
        help="Path to a RoomSpec YAML file.",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    strategy: str = typer.Option(
        "descriptive",
        "--strategy",
        help="Prompt strategy: minimal | descriptive | styleAnchored",
    ),
    control: Optional[str] = typer.Option(
        None,
        "--control",
        help="ControlNet signal: depth | canny | none",
    ),
    seed: int = typer.Option(42, "--seed", help="RNG seed for reproducibility."),
    steps: int = typer.Option(30, "--steps", help="Number of diffusion steps."),
    guidance: float = typer.Option(7.5, "--guidance", help="Classifier-free guidance scale."),
) -> None:
    """Generate a single interior design image from a room spec YAML."""
    typer.echo(f"[roomify generate] spec={spec} strategy={strategy} control={control} seed={seed}")
    typer.echo("Phase 3/4 implementation pending.")
    raise typer.Exit(code=0)


@app.command()
def sweep(
    config: Path = typer.Option(
        ...,
        "--config",
        help="Path to an experiment YAML file.",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
) -> None:
    """Run a full experiment sweep defined in an experiment YAML."""
    typer.echo(f"[roomify sweep] config={config}")
    typer.echo("Phase 5 implementation pending.")
    raise typer.Exit(code=0)


@app.command()
def evaluate(
    run: Path = typer.Option(
        ...,
        "--run",
        help="Path to an output run directory.",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
) -> None:
    """Compute evaluation metrics over a run directory."""
    typer.echo(f"[roomify evaluate] run={run}")
    typer.echo("Phase 7 implementation pending.")
    raise typer.Exit(code=0)


@app.command()
def report(
    run: Path = typer.Option(
        ...,
        "--run",
        help="Path to an output run directory.",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
) -> None:
    """Render a contact sheet and metrics markdown table for a run."""
    typer.echo(f"[roomify report] run={run}")
    typer.echo("Phase 7 implementation pending.")
    raise typer.Exit(code=0)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
