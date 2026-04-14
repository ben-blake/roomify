"""Reusable Streamlit UI components.

Implements Phase 6.  This stub satisfies Phase 0 import requirements.
All functions here are pure-Python helpers testable without a Streamlit runtime.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional


def specForm() -> Optional[Dict[str, Any]]:
    """Render the room-spec input form and return the spec dict on submit.

    Returns None if the form has not been submitted yet.
    Full implementation in Phase 6 — imports streamlit internally.
    """
    raise NotImplementedError("Phase 6: implement spec form component")


def imageCard(runJson: Dict[str, Any]) -> None:
    """Render a single image card with its run.json metadata.

    Full implementation in Phase 6.
    """
    raise NotImplementedError("Phase 6: implement image card component")


def metricsTable(df: Any) -> None:
    """Render a metrics DataFrame as a styled Streamlit table.

    Full implementation in Phase 6.
    """
    raise NotImplementedError("Phase 6: implement metrics table component")


def controlPreview(imagePath: Path) -> None:
    """Render the control signal preview (depth or Canny map).

    Full implementation in Phase 6.
    """
    raise NotImplementedError("Phase 6: implement control preview component")
