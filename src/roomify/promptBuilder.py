"""Prompt builder — maps RoomSpec + strategy to (positive, negative) prompts.

Implements Phase 2.  This stub satisfies Phase 0 import requirements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal

import yaml

STRATEGIES = Literal["minimal", "descriptive", "styleAnchored"]


@dataclass
class RoomSpec:
    id: str
    roomType: str
    size: str = ""
    style: str = ""
    furniture: List[str] = field(default_factory=list)
    lighting: str = ""
    mood: str = ""
    referenceImageId: str = ""


def buildPrompt(spec: RoomSpec, strategy: str) -> tuple[str, str]:
    """Return *(positive, negative)* prompt strings for *spec* and *strategy*.

    Raises ValueError for unknown strategy names.
    Full implementation in Phase 2 — loads templates from configs/prompts.yaml.
    """
    raise NotImplementedError("Phase 2: implement prompt strategies")
