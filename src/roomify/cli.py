"""Roomify CLI — thin typer shell over the service layer.

Usage:
    python -m roomify.cli --help
    python -m roomify.cli generate --spec configs/examples/bedroom_01.yaml
    python -m roomify.cli sweep --config configs/experiments/core.yaml
    python -m roomify.cli evaluate --run outputs/<runId>
    python -m roomify.cli report --run outputs/<runId>
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer
import yaml

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
    ref_image: Optional[str] = typer.Option(
        None,
        "--ref-image",
        help="SUN RGB-D record ID to use as ControlNet conditioning source.",
    ),
    seed: int = typer.Option(42, "--seed", help="RNG seed for reproducibility."),
    steps: int = typer.Option(30, "--steps", help="Number of diffusion steps."),
    guidance: float = typer.Option(7.5, "--guidance", help="Classifier-free guidance scale."),
) -> None:
    """Generate a single interior design image from a room spec YAML."""
    from roomify.paths import getOutputDir
    from roomify.pipeline import getPipeline
    from roomify.promptBuilder import RoomSpec, buildPrompt

    # Load spec YAML and construct RoomSpec
    raw: dict = yaml.safe_load(spec.read_text())
    valid_fields = set(RoomSpec.__dataclass_fields__.keys())
    room_spec = RoomSpec(**{k: v for k, v in raw.items() if k in valid_fields})

    # Build prompts
    positive, negative = buildPrompt(room_spec, strategy)

    # Normalize control type: "none" string → None
    control_type = control if control in ("depth", "canny") else None

    # Extract control conditioning image from SUN RGB-D record if provided
    control_image = None
    if control_type and ref_image:
        from PIL import Image as PILImage
        from roomify.controlSignals import extractCanny, extractDepth
        from roomify.dataset import getRecord, loadManifest
        from roomify.paths import getDataDir
        _manifest = loadManifest(getDataDir() / "sunrgbd_subset" / "manifest.csv")
        record = getRecord(_manifest, ref_image)
        if control_type == "depth":
            ref_pil = PILImage.open(record.depthPath)
            control_image = extractDepth(ref_pil)
        else:
            ref_pil = PILImage.open(record.rgbPath)
            control_image = extractCanny(ref_pil)

    # Prepare output directory  <runId = timestamp_specId>
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    run_id = f"{ts}_{room_spec.id}"
    out_dir = getOutputDir() / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load pipeline and generate
    ctrl_label = control_type or "none"
    typer.echo(f"Loading pipeline (control={ctrl_label})...")
    pipeline = getPipeline()
    pipeline.load(controlType=control_type)

    typer.echo(f"Generating image (seed={seed}, steps={steps}, guidance={guidance})...")
    t0 = time.monotonic()
    image = pipeline.generate(
        positive, negative, seed=seed, steps=steps, guidance=guidance,
        control=control_image,
    )
    elapsed = round(time.monotonic() - t0, 2)

    # Write image
    img_path = out_dir / "img_0.png"
    image.save(str(img_path))

    # Resolve git SHA (best-effort)
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        git_sha = "unknown"

    # Write run.json
    from roomify.pipeline import SD_MODEL_ID

    from roomify.pipeline import CONTROLNET_CANNY_ID, CONTROLNET_DEPTH_ID

    controlnet_id = None
    if control_type == "depth":
        controlnet_id = CONTROLNET_DEPTH_ID
    elif control_type == "canny":
        controlnet_id = CONTROLNET_CANNY_ID

    run_json = {
        "runId": run_id,
        "spec": dataclasses.asdict(room_spec),
        "strategy": strategy,
        "controlled": control_type is not None,
        "controlType": control_type,
        "refImageId": ref_image,
        "model": SD_MODEL_ID,
        "controlnet": controlnet_id,
        "seed": seed,
        "steps": steps,
        "guidanceScale": guidance,
        "prompt": positive,
        "negativePrompt": negative,
        "imagePath": str(img_path),
        "gitSha": git_sha,
        "timings": {"generateSec": elapsed},
    }
    (out_dir / "run.json").write_text(json.dumps(run_json, indent=2))

    typer.echo(f"Done → {out_dir}")


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
    from roomify.orchestrator import runExperiment

    typer.echo(f"[roomify sweep] config={config}")

    total_shown: list = []

    def progress(done: int, total: int) -> None:
        if not total_shown:
            total_shown.append(total)
            typer.echo(f"Running {total} images...")
        typer.echo(f"  [{done}/{total}] done")

    out_dir = runExperiment(config, progressCb=progress)
    typer.echo(f"Done → {out_dir}")


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
