"""Experiment orchestrator — runs the sweep matrix and persists outputs.

runExperiment(configPath, progressCb) iterates the cross-product of
  specs × strategies × controlled × seeds
and writes one image + run.json per cell into outputs/<runId>/.

progressCb(done, total) is called after each image if provided.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import yaml


def runExperiment(
    configPath: Path,
    progressCb: Optional[Callable[[int, int], None]] = None,
) -> Path:
    """Run the sweep defined in *configPath*.

    Returns the run output directory (outputs/<runId>/).
    """
    from roomify.paths import getOutputDir
    from roomify.pipeline import (
        CONTROLNET_CANNY_ID,
        CONTROLNET_DEPTH_ID,
        SD_MODEL_ID,
        getPipeline,
    )
    from roomify.promptBuilder import RoomSpec, buildPrompt

    config = yaml.safe_load(Path(configPath).read_text())

    run_id_prefix = config.get("runId", "sweep")
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    run_id = f"{ts}_{run_id_prefix}"

    out_dir = getOutputDir() / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    specs_raw: list = config.get("specs", [])
    strategies: list = config.get("strategies", ["descriptive"])
    controlled_flags: list = config.get("controlled", [False])
    seeds: list = config.get("seeds", [42])

    valid_fields = set(RoomSpec.__dataclass_fields__.keys())

    # Build the sweep matrix
    matrix = [
        (spec_raw, strategy, is_controlled, seed)
        for spec_raw in specs_raw
        for strategy in strategies
        for is_controlled in controlled_flags
        for seed in seeds
    ]

    total = len(matrix)

    # Resolve git SHA once
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        git_sha = "unknown"

    pipeline = getPipeline()

    for idx, (spec_raw, strategy, is_controlled, seed) in enumerate(matrix, start=1):
        room_spec = RoomSpec(**{k: v for k, v in spec_raw.items() if k in valid_fields})
        positive, negative = buildPrompt(room_spec, strategy)

        # Controlled sweeps require a referenceImageId on the spec
        control_type: Optional[str] = None
        control_image = None
        ref_image_id: Optional[str] = None

        if is_controlled and spec_raw.get("referenceImageId"):
            ref_image_id = spec_raw["referenceImageId"]
            control_type = "depth"
            from PIL import Image as PILImage
            from roomify.controlSignals import extractDepth
            from roomify.dataset import getRecord, loadManifest
            from roomify.paths import getDataDir
            _manifest = loadManifest(getDataDir() / "sunrgbd_subset" / "manifest.csv")
            record = getRecord(_manifest, ref_image_id)
            control_image = extractDepth(PILImage.open(record.depthPath))

        pipeline.load(controlType=control_type)

        t0 = time.monotonic()
        image = pipeline.generate(
            positive, negative,
            seed=seed,
            steps=30,
            guidance=7.5,
            control=control_image,
            conditioningScale=config.get("conditioningScale", 0.6),
        )
        elapsed = round(time.monotonic() - t0, 2)

        # Write image into its own sub-directory per sweep cell
        cell_id = f"{room_spec.id}_{strategy}_{'ctrl' if is_controlled else 'unctrl'}_s{seed}"
        cell_dir = out_dir / cell_id
        cell_dir.mkdir(parents=True, exist_ok=True)

        img_path = cell_dir / "img_0.png"
        image.save(str(img_path))

        controlnet_id: Optional[str] = None
        if control_type == "depth":
            controlnet_id = CONTROLNET_DEPTH_ID
        elif control_type == "canny":
            controlnet_id = CONTROLNET_CANNY_ID

        run_json = {
            "runId": f"{run_id}/{cell_id}",
            "spec": dataclasses.asdict(room_spec),
            "strategy": strategy,
            "controlled": is_controlled,
            "controlType": control_type,
            "refImageId": ref_image_id,
            "model": SD_MODEL_ID,
            "controlnet": controlnet_id,
            "conditioningScale": config.get("conditioningScale", 0.6) if is_controlled else None,
            "seed": seed,
            "steps": 30,
            "guidanceScale": 7.5,
            "prompt": positive,
            "negativePrompt": negative,
            "imagePath": str(img_path),
            "gitSha": git_sha,
            "timings": {"generateSec": elapsed},
        }
        (cell_dir / "run.json").write_text(json.dumps(run_json, indent=2))

        if progressCb is not None:
            progressCb(idx, total)

    return out_dir
