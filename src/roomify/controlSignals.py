"""Control signal extraction for ControlNet conditioning.

Implements Phase 4.  This stub satisfies Phase 0 import requirements.
"""

from __future__ import annotations

from PIL import Image


def extractDepth(image: Image.Image) -> Image.Image:
    """Return a depth-conditioned image suitable for ControlNet.

    Accepts either an RGB image (MiDaS depth estimation) or a pre-existing
    depth map.  Full implementation in Phase 4.
    """
    raise NotImplementedError("Phase 4: implement depth extraction")


def extractCanny(image: Image.Image, lo: int = 100, hi: int = 200) -> Image.Image:
    """Return a Canny edge map from *image* for ControlNet conditioning.

    *lo* and *hi* are the lower and upper Canny thresholds.
    Full implementation in Phase 4.
    """
    raise NotImplementedError("Phase 4: implement Canny extraction")
