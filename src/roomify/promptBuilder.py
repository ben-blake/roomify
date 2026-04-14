"""Prompt builder — maps RoomSpec + strategy to (positive, negative) prompts.

Phase 2 implementation: loads strategy templates from configs/prompts.yaml
and renders them against a RoomSpec.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal

import yaml

STRATEGIES = Literal["minimal", "descriptive", "styleAnchored"]

# Locate prompts.yaml relative to this file so it works in both Colab and local dev.
_PROMPTS_YAML = Path(__file__).parent.parent.parent / "configs" / "prompts.yaml"


def _loadConfig() -> dict:
    with open(_PROMPTS_YAML, "r") as fh:
        return yaml.safe_load(fh)


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
    Loads templates from configs/prompts.yaml at call time (no module-level
    singleton — keeps tests isolated and avoids Drive path issues on import).
    """
    config = _loadConfig()
    strategies: dict = config.get("strategies", {})

    if strategy not in strategies:
        raise ValueError(f"unknown strategy '{strategy}'; valid: {sorted(strategies)}")

    template: str = strategies[strategy]["positive"]
    furnitureStr = ", ".join(spec.furniture) if spec.furniture else ""

    # Build substitution dict; missing optional fields become empty string.
    subs = {
        "roomType": spec.roomType,
        "style": spec.style,
        "size": spec.size,
        "furniture": furnitureStr,
        "lighting": spec.lighting,
        "mood": spec.mood,
    }

    positive = _renderTemplate(template, subs)
    negative: str = config.get("negative", "")

    return positive.strip(), negative.strip()


def _renderTemplate(template: str, subs: dict) -> str:
    """Replace {key} placeholders; drop surrounding punctuation for missing values."""
    result = template

    for key, value in subs.items():
        placeholder = "{" + key + "}"
        if value:
            result = result.replace(placeholder, value)
        else:
            # Remove the placeholder plus any immediately preceding separator
            # (", " or ", \n" or just leading/trailing comma-space sequences).
            result = result.replace(", " + placeholder, "")
            result = result.replace(placeholder + ", ", "")
            result = result.replace(placeholder, "")

    # Collapse multiple commas and tidy up whitespace artifacts.
    result = re.sub(r",\s*,", ",", result)
    result = re.sub(r",\s*$", "", result)
    result = re.sub(r"^\s*,", "", result)
    result = re.sub(r"\s{2,}", " ", result)

    return result.strip()
