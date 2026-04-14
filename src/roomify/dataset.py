"""SUN RGB-D manifest loader and record schema.

Implements Phase 1.  This stub satisfies Phase 0 import requirements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pandas as pd


VALID_SCENE_TYPES = frozenset(
    ["bedroom", "living_room", "kitchen", "office", "bathroom"]
)


@dataclass
class Record:
    id: str
    sceneType: str
    rgbPath: Path
    depthPath: Path
    objectLabels: List[str] = field(default_factory=list)


def loadManifest(manifestPath: Path) -> pd.DataFrame:
    """Load manifest.csv and return a DataFrame.

    Raises FileNotFoundError if the path does not exist.
    """
    if not manifestPath.exists():
        raise FileNotFoundError(f"Manifest not found: {manifestPath}")
    return pd.read_csv(manifestPath)


def getRecord(df: pd.DataFrame, recordId: str) -> Record:
    """Return the Record for *recordId* from *df*.

    Raises KeyError if the id is not found.
    """
    row = df[df["id"] == recordId]
    if row.empty:
        raise KeyError(f"Record not found: {recordId}")
    r = row.iloc[0]
    labels = (
        r["objectLabels"].split(",") if isinstance(r.get("objectLabels"), str) else []
    )
    return Record(
        id=r["id"],
        sceneType=r["sceneType"],
        rgbPath=Path(r["rgbPath"]),
        depthPath=Path(r["depthPath"]),
        objectLabels=[lbl.strip() for lbl in labels if lbl.strip()],
    )


def listByScene(df: pd.DataFrame, sceneType: str) -> pd.DataFrame:
    """Return subset of *df* matching *sceneType*."""
    if sceneType not in VALID_SCENE_TYPES:
        raise ValueError(f"Unknown scene type '{sceneType}'. Valid: {VALID_SCENE_TYPES}")
    return df[df["sceneType"] == sceneType].copy()
