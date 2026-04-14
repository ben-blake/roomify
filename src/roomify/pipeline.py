"""Stable Diffusion + ControlNet pipeline wrapper.

Process-level singleton accessible via getPipeline().
Streamlit callers wrap getPipeline() in @st.cache_resource.

Implements Phase 3 (baseline) and Phase 4 (ControlNet).
This stub satisfies Phase 0 import requirements.
"""

from __future__ import annotations

from typing import Optional

from PIL import Image

_PIPELINE_INSTANCE: Optional["Pipeline"] = None

SD_MODEL_ID = "runwayml/stable-diffusion-v1-5"
CONTROLNET_DEPTH_ID = "lllyasviel/sd-controlnet-depth"
CONTROLNET_CANNY_ID = "lllyasviel/sd-controlnet-canny"


class Pipeline:
    """Lazy-loading SD + ControlNet wrapper.

    Full implementation in Phase 3/4.
    """

    def __init__(self) -> None:
        self._sd = None
        self._controlnet = None

    def load(self, controlType: Optional[str] = None) -> None:
        """Load model weights.  controlType: 'depth' | 'canny' | None."""
        raise NotImplementedError("Phase 3: implement pipeline loading")

    def generate(
        self,
        positive: str,
        negative: str,
        seed: int = 42,
        steps: int = 30,
        guidance: float = 7.5,
        control: Optional[Image.Image] = None,
    ) -> Image.Image:
        """Generate one image.  Returns a PIL Image."""
        raise NotImplementedError("Phase 3: implement generation")


def getPipeline() -> Pipeline:
    """Return the process-level Pipeline singleton, creating it if needed."""
    global _PIPELINE_INSTANCE
    if _PIPELINE_INSTANCE is None:
        _PIPELINE_INSTANCE = Pipeline()
    return _PIPELINE_INSTANCE
