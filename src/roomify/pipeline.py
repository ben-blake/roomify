"""Stable Diffusion + ControlNet pipeline wrapper.

Process-level singleton accessible via getPipeline().
Streamlit callers wrap getPipeline() in @st.cache_resource.

torch and diffusers are imported lazily inside load() so that this module
can be imported in a no-GPU environment (local dev, unit tests).
"""

from __future__ import annotations

from typing import Optional

from PIL import Image

_PIPELINE_INSTANCE: Optional["Pipeline"] = None

SD_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
CONTROLNET_DEPTH_ID = "lllyasviel/sd-controlnet-depth"
CONTROLNET_CANNY_ID = "lllyasviel/sd-controlnet-canny"


class Pipeline:
    """Lazy-loading SD 1.5 wrapper.

    Call load() once before generate(). Streamlit wraps getPipeline() in
    @st.cache_resource so load() is called at most once per process.
    """

    def __init__(self) -> None:
        self._sd = None
        self._loaded = False
        self._controlType: Optional[str] = None

    def load(self, controlType: Optional[str] = None) -> None:
        """Load SD 1.5 weights with fp16 and attention slicing.

        controlType: 'depth' | 'canny' | None
          - 'depth' / 'canny' → loads StableDiffusionControlNetPipeline with the
            corresponding ControlNet checkpoint.
          - None → loads plain StableDiffusionPipeline.

        No-op if already loaded with the same controlType.
        When switching types, frees VRAM before loading the new pipeline.
        """
        if self._loaded and self._controlType == controlType:
            return

        import torch  # type: ignore[import-not-found]

        # Release previous pipeline to free VRAM before loading a new variant.
        if self._sd is not None:
            del self._sd
            self._sd = None
            self._loaded = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if controlType in ("depth", "canny"):
            from diffusers import (  # type: ignore[import-not-found]
                ControlNetModel,
                StableDiffusionControlNetPipeline,
            )

            controlnet_id = (
                CONTROLNET_DEPTH_ID if controlType == "depth" else CONTROLNET_CANNY_ID
            )
            controlnet = ControlNetModel.from_pretrained(
                controlnet_id,
                torch_dtype=torch.float16,
            )
            pipe = StableDiffusionControlNetPipeline.from_pretrained(
                SD_MODEL_ID,
                controlnet=controlnet,
                torch_dtype=torch.float16,
                safety_checker=None,
            )
        else:
            from diffusers import StableDiffusionPipeline  # type: ignore[import-not-found]

            pipe = StableDiffusionPipeline.from_pretrained(
                SD_MODEL_ID,
                torch_dtype=torch.float16,
                safety_checker=None,
            )

        pipe.enable_attention_slicing()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe.to(device)

        self._sd = pipe
        self._loaded = True
        self._controlType = controlType

    def generate(
        self,
        positive: str,
        negative: str,
        seed: int = 42,
        steps: int = 30,
        guidance: float = 7.5,
        control: Optional[Image.Image] = None,
    ) -> Image.Image:
        """Generate one interior design image.

        Returns a PIL Image. control image is reserved for Phase 4 ControlNet.
        Raises RuntimeError if load() has not been called.
        """
        if not self._loaded or self._sd is None:
            raise RuntimeError(
                "Pipeline not loaded — call load() before generate()"
            )

        import torch  # type: ignore[import-not-found]

        generator = torch.Generator().manual_seed(seed)

        kwargs: dict = dict(
            negative_prompt=negative,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
        )
        if control is not None:
            kwargs["image"] = control

        result = self._sd(positive, **kwargs)
        return result.images[0]


def getPipeline() -> Pipeline:
    """Return the process-level Pipeline singleton, creating it if needed."""
    global _PIPELINE_INSTANCE
    if _PIPELINE_INSTANCE is None:
        _PIPELINE_INSTANCE = Pipeline()
    return _PIPELINE_INSTANCE


def _resetPipeline() -> None:
    """Reset the singleton.  For testing only — do not call in production."""
    global _PIPELINE_INSTANCE
    _PIPELINE_INSTANCE = None
