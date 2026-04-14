"""Build a curated ~200-sample subset of SUN RGB-D and emit manifest.csv.

Usage (run in Colab after dataset is downloaded to Google Drive):

    python scripts/buildSubset.py \
        --sunrgbd-root /content/drive/MyDrive/roomify/data/SUNRGBD \
        --output-dir  /content/drive/MyDrive/roomify/data/sunrgbd_subset \
        --samples-per-scene 40

Output
------
<output-dir>/manifest.csv
    id, sceneType, rgbPath, depthPath, objectLabels

<output-dir>/<sceneType>/<id>/rgb.jpg   (symlink or copy)
<output-dir>/<sceneType>/<id>/depth.png (symlink or copy)

Scene types curated
-------------------
bedroom, living_room, kitchen, office, bathroom — 40 samples each = 200 total.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

SCENE_TYPES = ["bedroom", "living_room", "kitchen", "office", "bathroom"]

# Keyword mapping from SUN RGB-D scene labels to our 5 canonical types.
# SUN RGB-D uses names like "bedroom", "living_room", "kitchen", "office", "bathroom" directly,
# but also includes variants such as "dining_room", "study", etc. We map greedily on the
# substring so "home_office" → "office" and "master_bedroom" → "bedroom".
SCENE_KEYWORD_MAP: Dict[str, str] = {
    "bedroom": "bedroom",
    "living_room": "living_room",
    "living room": "living_room",
    "kitchen": "kitchen",
    "office": "office",
    "bathroom": "bathroom",
    "bath": "bathroom",
}


def canonicalSceneType(rawLabel: str) -> Optional[str]:
    """Map a raw SUN RGB-D scene label to one of our 5 canonical types, or None."""
    lower = rawLabel.lower().strip()
    for keyword, canonical in SCENE_KEYWORD_MAP.items():
        if keyword in lower:
            return canonical
    return None


def discoverSamples(sunrgbdRoot: Path) -> Dict[str, List[Path]]:
    """Walk the SUN RGB-D directory tree and bin scene dirs by canonical type.

    SUN RGB-D layout (kv1 / kv2 / realsense / xtion sub-splits):

        SUNRGBD/<split>/<scene_label>/<instance>/
            image/  *.jpg
            depth/  *.png
            annotation/  scene.txt  (contains the scene label)

    Returns a dict mapping canonical scene type → list of instance dirs.
    """
    buckets: Dict[str, List[Path]] = {s: [] for s in SCENE_TYPES}

    for instanceDir in sorted(sunrgbdRoot.rglob("image")):
        instanceDir = instanceDir.parent  # step up from image/ to instance dir

        # Scene label lives either in the grandparent dir name or in scene.txt
        sceneLabel = _readSceneLabel(instanceDir)
        canonical = canonicalSceneType(sceneLabel)
        if canonical is None:
            continue

        rgbFiles = list((instanceDir / "image").glob("*.jpg"))
        depthFiles = list((instanceDir / "depth").glob("*.png"))
        if not rgbFiles or not depthFiles:
            continue

        buckets[canonical].append(instanceDir)

    for sceneType, dirs in buckets.items():
        log.info("%s: %d candidate samples", sceneType, len(dirs))

    return buckets


def _readSceneLabel(instanceDir: Path) -> str:
    """Return the scene label for an instance dir (best-effort)."""
    # 1. Try scene.txt next to image/
    sceneTxt = instanceDir / "scene.txt"
    if sceneTxt.exists():
        return sceneTxt.read_text().strip()

    # 2. Try annotation/scene.txt
    annotationScene = instanceDir / "annotation" / "scene.txt"
    if annotationScene.exists():
        return annotationScene.read_text().strip()

    # 3. Fall back to the parent directory name (often the scene category)
    return instanceDir.parent.name


def _readObjectLabels(instanceDir: Path) -> List[str]:
    """Extract a flat list of object labels from SUN RGB-D annotation JSON if present."""
    for jsonPath in (instanceDir / "annotation").glob("*.json") if (instanceDir / "annotation").exists() else []:
        try:
            data = json.loads(jsonPath.read_text())
            names = []
            for obj in data.get("objects", []):
                name = obj.get("name") or obj.get("label") or ""
                if name:
                    names.append(name.strip())
            return names
        except Exception:
            pass
    return []


def buildSubset(
    sunrgbdRoot: Path,
    outputDir: Path,
    samplesPerScene: int = 40,
    seed: int = 42,
    copyFiles: bool = False,
) -> Path:
    """Curate the subset and write manifest.csv.

    Parameters
    ----------
    sunrgbdRoot:     Root of the downloaded SUN RGB-D dataset.
    outputDir:       Where to write subset structure + manifest.csv.
    samplesPerScene: How many samples to keep per scene type.
    seed:            RNG seed for reproducible sampling.
    copyFiles:       If True, copy files; if False, use symlinks (faster, saves space).

    Returns
    -------
    Path to the written manifest.csv.
    """
    random.seed(seed)
    outputDir.mkdir(parents=True, exist_ok=True)

    buckets = discoverSamples(sunrgbdRoot)

    rows: List[Dict] = []
    globalIndex = 0

    for sceneType in SCENE_TYPES:
        candidates = buckets[sceneType]
        if not candidates:
            log.warning("No candidates found for scene type: %s", sceneType)
            continue

        if len(candidates) < samplesPerScene:
            log.warning(
                "%s: only %d candidates available (wanted %d)",
                sceneType,
                len(candidates),
                samplesPerScene,
            )
        selected = random.sample(candidates, min(samplesPerScene, len(candidates)))

        for instanceDir in selected:
            globalIndex += 1
            recordId = f"sunrgbd_{globalIndex:05d}"

            destDir = outputDir / sceneType / recordId
            destDir.mkdir(parents=True, exist_ok=True)

            rgbSrc = sorted((instanceDir / "image").glob("*.jpg"))[0]
            depthSrc = sorted((instanceDir / "depth").glob("*.png"))[0]

            rgbDest = destDir / "rgb.jpg"
            depthDest = destDir / "depth.png"

            _link_or_copy(rgbSrc, rgbDest, copyFiles)
            _link_or_copy(depthSrc, depthDest, copyFiles)

            objectLabels = _readObjectLabels(instanceDir)

            rows.append(
                {
                    "id": recordId,
                    "sceneType": sceneType,
                    "rgbPath": str(rgbDest),
                    "depthPath": str(depthDest),
                    "objectLabels": ", ".join(objectLabels),
                }
            )

        log.info("%s: %d samples written", sceneType, len(selected))

    manifestPath = outputDir / "manifest.csv"
    fieldnames = ["id", "sceneType", "rgbPath", "depthPath", "objectLabels"]
    with manifestPath.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    log.info("Manifest written: %s (%d rows)", manifestPath, len(rows))
    return manifestPath


def _link_or_copy(src: Path, dest: Path, copyFiles: bool) -> None:
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    if copyFiles:
        shutil.copy2(src, dest)
    else:
        dest.symlink_to(src.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Curate a SUN RGB-D subset and emit manifest.csv"
    )
    parser.add_argument(
        "--sunrgbd-root",
        required=True,
        type=Path,
        help="Root directory of the downloaded SUN RGB-D dataset",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory to write the curated subset and manifest.csv",
    )
    parser.add_argument(
        "--samples-per-scene",
        type=int,
        default=40,
        help="Samples per scene type (default: 40, total: 200)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42)",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of symlinking (use in Colab where symlinks may not persist)",
    )
    args = parser.parse_args()

    if not args.sunrgbd_root.exists():
        log.error("SUN RGB-D root not found: %s", args.sunrgbd_root)
        sys.exit(1)

    manifestPath = buildSubset(
        sunrgbdRoot=args.sunrgbd_root,
        outputDir=args.output_dir,
        samplesPerScene=args.samples_per_scene,
        seed=args.seed,
        copyFiles=args.copy,
    )
    print(f"Done. Manifest: {manifestPath}")


if __name__ == "__main__":
    main()
