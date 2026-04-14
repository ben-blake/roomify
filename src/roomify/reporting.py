"""Contact sheet and metrics table generation.

Implements Phase 7.  This stub satisfies Phase 0 import requirements.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image


def contactSheet(runDir: Path) -> Image.Image:
    """Build a contact-sheet grid from all images in *runDir*.

    Returns a PIL Image.  Full implementation in Phase 7.
    """
    raise NotImplementedError("Phase 7: implement contact sheet generation")


def metricsTable(runDir: Path) -> str:
    """Return a markdown metrics table for the run in *runDir*.

    Full implementation in Phase 7.
    """
    raise NotImplementedError("Phase 7: implement metrics table rendering")
