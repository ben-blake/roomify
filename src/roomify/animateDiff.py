"""AnimateDiff GIF generation — Phase 8 multimodal extension.

Wraps AnimateDiffPipeline from diffusers to animate a room design prompt
into a short looping GIF. torch and diffusers are imported lazily inside
load() so the module is safe to import in a no-GPU environment.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from PIL import Image

_ANIMATE_INSTANCE: Optional["AnimateDiffGenerator"] = None

SD_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
MOTION_ADAPTER_ID = "guoyww/animatediff-motion-adapter-v1-5-2"

DEFAULT_NUM_FRAMES = 16
DEFAULT_FPS = 8


class AnimateDiffGenerator:
    """Lazy-loading AnimateDiff wrapper.

    Call load() once before generate(). Streamlit callers should wrap
    getAnimateDiffGenerator() in @st.cache_resource.
    """

    def __init__(self) -> None:
        self._pipe = None
        self._loaded = False

    def load(self) -> None:
        """Load MotionAdapter + AnimateDiffPipeline with fp16 and attention slicing.

        No-op if already loaded.
        """
        if self._loaded:
            return

        import torch  # type: ignore[import-not-found]
        from diffusers import AnimateDiffPipeline, MotionAdapter  # type: ignore[import-not-found]

        adapter = MotionAdapter.from_pretrained(MOTION_ADAPTER_ID, torch_dtype=torch.float16)
        pipe = AnimateDiffPipeline.from_pretrained(
            SD_MODEL_ID,
            motion_adapter=adapter,
            torch_dtype=torch.float16,
        )
        pipe.enable_attention_slicing()
        if torch.cuda.is_available():
            pipe.to("cuda")

        self._pipe = pipe
        self._loaded = True

    def generate(
        self,
        positive: str,
        negative: str,
        seed: int = 42,
        steps: int = 25,
        guidance: float = 7.5,
        numFrames: int = DEFAULT_NUM_FRAMES,
    ) -> List[Image.Image]:
        """Run AnimateDiff and return a list of PIL frames.

        Args:
            positive: Positive prompt text.
            negative: Negative prompt text.
            seed: RNG seed for reproducibility.
            steps: Number of denoising steps (25 is enough for animation).
            guidance: Classifier-free guidance scale.
            numFrames: Number of frames to generate.

        Returns:
            List of PIL Image frames.

        Raises:
            RuntimeError: If load() has not been called.
        """
        if not self._loaded or self._pipe is None:
            raise RuntimeError("AnimateDiffGenerator.load() must be called before generate()")

        import torch  # type: ignore[import-not-found]

        generator = torch.Generator()
        generator.manual_seed(seed)

        output = self._pipe(
            prompt=positive,
            negative_prompt=negative,
            num_frames=numFrames,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
        )
        return output.frames[0]


def framesToGif(frames: List[Image.Image], outPath: Path, fps: int = DEFAULT_FPS) -> Path:
    """Save a list of PIL Images as an animated GIF.

    Args:
        frames: Non-empty list of PIL Image frames.
        outPath: Destination file path (should end in .gif).
        fps: Frames per second — controls the per-frame display duration.

    Returns:
        The outPath, for chaining.

    Raises:
        ValueError: If frames is empty.
    """
    if not frames:
        raise ValueError("frames list must not be empty")

    duration_ms = round(1000 / fps)
    frames[0].save(
        str(outPath),
        save_all=True,
        append_images=frames[1:],
        loop=0,
        duration=duration_ms,
    )
    return outPath


def getAnimateDiffGenerator() -> AnimateDiffGenerator:
    """Return the process-level AnimateDiffGenerator singleton."""
    global _ANIMATE_INSTANCE
    if _ANIMATE_INSTANCE is None:
        _ANIMATE_INSTANCE = AnimateDiffGenerator()
    return _ANIMATE_INSTANCE


def _resetAnimateDiffGenerator() -> None:
    """Reset the singleton (test isolation only)."""
    global _ANIMATE_INSTANCE
    _ANIMATE_INSTANCE = None
