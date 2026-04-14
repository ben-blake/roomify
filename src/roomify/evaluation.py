"""Evaluation metrics for generated images.

Implements Phase 7.  This stub satisfies Phase 0 import requirements.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def clipAlignment(runDir: Path) -> pd.DataFrame:
    """Compute CLIP text-image cosine similarity for each image in *runDir*.

    Returns a DataFrame with columns: imagePath, prompt, clipScore.
    Full implementation in Phase 7.
    """
    raise NotImplementedError("Phase 7: implement CLIP alignment metric")


def lpipsDiversity(runDir: Path) -> float:
    """Compute mean pairwise LPIPS distance across same-spec images in *runDir*.

    Higher = more diverse.  Full implementation in Phase 7.
    """
    raise NotImplementedError("Phase 7: implement LPIPS diversity metric")


def styleConsistency(runDir: Path) -> float:
    """Compute mean pairwise CLIP image-image similarity within *runDir*.

    Higher = more consistent.  Full implementation in Phase 7.
    """
    raise NotImplementedError("Phase 7: implement style consistency metric")
