"""Experiment orchestrator — runs the sweep matrix and persists outputs.

Implements Phase 5.  This stub satisfies Phase 0 import requirements.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional


def runExperiment(
    configPath: Path,
    progressCb: Optional[Callable[[int, int], None]] = None,
) -> Path:
    """Run the sweep defined in *configPath*.

    Writes outputs to outputs/<runId>/ and calls progressCb(done, total)
    after each image if provided.

    Returns the run output directory.  Full implementation in Phase 5.
    """
    raise NotImplementedError("Phase 5: implement experiment orchestration")
