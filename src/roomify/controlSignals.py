"""Control signal extraction for ControlNet conditioning.

Both functions return a 3-channel RGB PIL Image suitable for passing
to StableDiffusionControlNetPipeline as the conditioning image.

cv2 (opencv-python) and numpy are imported lazily so this module can be
imported in environments where cv2 is unavailable (though cv2 must be
present when extractCanny is *called*).
"""

from __future__ import annotations

from PIL import Image


def extractDepth(image: Image.Image) -> Image.Image:
    """Return a depth-conditioned RGB image for ControlNet.

    Accepts a grayscale depth map (from SUN RGB-D) or an RGB image.
    For grayscale input the pixel values are normalized to 0-255 and
    tiled into 3 channels.  For RGB input luminance is used as a depth
    proxy — this is a lightweight fallback when no real depth channel is
    available.
    """
    import numpy as np

    arr = np.array(image)

    if arr.ndim == 2:
        # Grayscale / raw depth channel — normalize to [0, 255]
        arr_min = float(arr.min())
        arr_max = float(arr.max())
        if arr_max > arr_min:
            normalized = (
                (arr.astype(np.float32) - arr_min) / (arr_max - arr_min) * 255
            ).astype(np.uint8)
        else:
            normalized = np.zeros_like(arr, dtype=np.uint8)
        rgb = np.stack([normalized, normalized, normalized], axis=-1)
    else:
        # RGB — use mean luminance as a depth proxy
        gray = np.mean(arr[:, :, :3].astype(np.float32), axis=-1).astype(np.uint8)
        rgb = np.stack([gray, gray, gray], axis=-1)

    return Image.fromarray(rgb)


def extractCanny(image: Image.Image, lo: int = 100, hi: int = 200) -> Image.Image:
    """Return a Canny edge map from *image* for ControlNet conditioning.

    *lo* and *hi* are the lower and upper Canny thresholds passed to
    cv2.Canny.  The result is a 3-channel RGB image (edges are white,
    background is black) matching ControlNet's expected input format.
    """
    import cv2  # type: ignore[import-not-found]
    import numpy as np

    arr = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, lo, hi)
    rgb_edges = np.stack([edges, edges, edges], axis=-1)
    return Image.fromarray(rgb_edges)
