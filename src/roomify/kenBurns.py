"""Ken Burns pan/zoom effect — Phase 8 animation via post-processing.

Takes a static PIL Image and produces a list of frames that simulate a
camera zooming in/out or panning across the scene. Pure Pillow — no GPU.
"""

from __future__ import annotations

from typing import List

from PIL import Image

MOTION_TYPES = ("zoom_in", "zoom_out", "pan_right", "pan_left", "pan_up", "pan_down")


def _easeInOut(t: float) -> float:
    """Smoothstep: smooth acceleration and deceleration."""
    return t * t * (3.0 - 2.0 * t)


def applyKenBurns(
    image: Image.Image,
    frames: int = 24,
    motion: str = "zoom_in",
    intensity: float = 0.2,
) -> List[Image.Image]:
    """Apply a Ken Burns pan/zoom effect to a static image.

    Args:
        image: Source PIL Image (any size).
        frames: Number of output frames. Must be >= 2.
        motion: One of zoom_in, zoom_out, pan_right, pan_left, pan_up, pan_down.
        intensity: How much to zoom or pan as a fraction of the image dimension.
                   Clamped to [0.0, 0.5] to avoid over-cropping.

    Returns:
        List of PIL Images, all the same size as the input.

    Raises:
        ValueError: If frames < 2 or motion is unrecognised.
    """
    if frames < 2:
        raise ValueError(f"frames must be >= 2, got {frames}")
    if motion not in MOTION_TYPES:
        raise ValueError(f"unknown motion {motion!r}; choose from {MOTION_TYPES}")

    intensity = max(0.0, min(0.5, intensity))
    W, H = image.size
    result: List[Image.Image] = []

    for i in range(frames):
        t = _easeInOut(i / (frames - 1))

        if motion == "zoom_in":
            scale = 1.0 + intensity * t
            crop_w = round(W / scale)
            crop_h = round(H / scale)
            left = (W - crop_w) // 2
            top = (H - crop_h) // 2

        elif motion == "zoom_out":
            scale = 1.0 + intensity * (1.0 - t)
            crop_w = round(W / scale)
            crop_h = round(H / scale)
            left = (W - crop_w) // 2
            top = (H - crop_h) // 2

        elif motion == "pan_right":
            scale = 1.0 + intensity
            crop_w = round(W / scale)
            crop_h = round(H / scale)
            left = round((W - crop_w) * t)
            top = (H - crop_h) // 2

        elif motion == "pan_left":
            scale = 1.0 + intensity
            crop_w = round(W / scale)
            crop_h = round(H / scale)
            left = round((W - crop_w) * (1.0 - t))
            top = (H - crop_h) // 2

        elif motion == "pan_up":
            scale = 1.0 + intensity
            crop_w = round(W / scale)
            crop_h = round(H / scale)
            left = (W - crop_w) // 2
            top = round((H - crop_h) * (1.0 - t))

        else:  # pan_down
            scale = 1.0 + intensity
            crop_w = round(W / scale)
            crop_h = round(H / scale)
            left = (W - crop_w) // 2
            top = round((H - crop_h) * t)

        cropped = image.crop((left, top, left + crop_w, top + crop_h))
        result.append(cropped.resize((W, H), Image.LANCZOS))

    return result
