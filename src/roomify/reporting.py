"""Contact sheet and metrics table generation — Phase 7."""

from __future__ import annotations

import json
import math
from pathlib import Path

from PIL import Image


def contactSheet(runDir: Path, thumbSize: int = 256) -> Image.Image:
    """Build a contact-sheet grid from all images in *runDir*.

    Returns a PIL Image arranged in the smallest square grid that fits.
    Raises ValueError if no images are found.
    """
    imgs = []
    for run_json in sorted(Path(runDir).rglob("run.json")):
        img_path = run_json.parent / "img_0.png"
        if img_path.exists():
            imgs.append(Image.open(img_path).resize((thumbSize, thumbSize)))

    if not imgs:
        raise ValueError(f"No images found in {runDir}")

    cols = math.ceil(math.sqrt(len(imgs)))
    rows = math.ceil(len(imgs) / cols)

    sheet = Image.new("RGB", (cols * thumbSize, rows * thumbSize), color=(240, 240, 240))
    for idx, img in enumerate(imgs):
        r, c = divmod(idx, cols)
        sheet.paste(img, (c * thumbSize, r * thumbSize))

    return sheet


def metricsTable(runDir: Path) -> str:
    """Return a markdown metrics table for all runs under *runDir*.

    Raises ValueError if no run.json files are found.
    """
    rows = []
    for run_json in sorted(Path(runDir).rglob("run.json")):
        data = json.loads(run_json.read_text())
        rows.append({
            "runId": data.get("runId", ""),
            "strategy": data.get("strategy", ""),
            "seed": data.get("seed", ""),
            "steps": data.get("steps", ""),
            "controlled": data.get("controlled", False),
        })

    if not rows:
        raise ValueError(f"No run.json files found in {runDir}")

    headers = ["runId", "strategy", "seed", "steps", "controlled"]
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")

    return "\n".join(lines)
