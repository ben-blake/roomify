"""Tests for src/roomify/controlSignals.py — Phase 4.

TDD: written before implementation. Run with:  pytest tests/testControlSignals.py -v

cv2 is Colab-only (not in requirements-dev.txt); mocked at module level.
numpy is available locally and used as-is for extractDepth tests.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image as PILImage

# ---------------------------------------------------------------------------
# Mock cv2 — not in requirements-dev.txt
# ---------------------------------------------------------------------------

_FAKE_SIZE_HW = (64, 64)  # (height, width)

_cv2_mock = MagicMock()
_cv2_mock.COLOR_RGB2GRAY = 7
_cv2_mock.cvtColor.return_value = np.zeros(_FAKE_SIZE_HW, dtype=np.uint8)
_cv2_mock.Canny.return_value = np.zeros(_FAKE_SIZE_HW, dtype=np.uint8)

sys.modules.setdefault("cv2", _cv2_mock)

# Safe to import once mocks are in place
from roomify.controlSignals import extractCanny, extractDepth  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rgb_image(hw: tuple[int, int] = (64, 64)) -> PILImage.Image:
    h, w = hw
    return PILImage.fromarray(
        np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    )


def _gray_image(hw: tuple[int, int] = (64, 64)) -> PILImage.Image:
    h, w = hw
    arr = np.random.randint(0, 255, (h, w), dtype=np.uint8)
    return PILImage.fromarray(arr, mode="L")


@pytest.fixture(autouse=True)
def resetCv2():
    """Reset cv2 mock before every test."""
    _cv2_mock.reset_mock()
    _cv2_mock.COLOR_RGB2GRAY = 7
    _cv2_mock.cvtColor.return_value = np.zeros(_FAKE_SIZE_HW, dtype=np.uint8)
    _cv2_mock.Canny.return_value = np.zeros(_FAKE_SIZE_HW, dtype=np.uint8)
    yield


# ---------------------------------------------------------------------------
# extractDepth — uses numpy only, no cv2 mock needed
# ---------------------------------------------------------------------------


def test_extractDepth_returns_pil_image():
    result = extractDepth(_gray_image())
    assert isinstance(result, PILImage.Image)


def test_extractDepth_output_is_rgb():
    result = extractDepth(_gray_image())
    assert result.mode == "RGB"


def test_extractDepth_preserves_size_grayscale():
    img = _gray_image((100, 80))  # 100 tall, 80 wide
    result = extractDepth(img)
    assert result.size == (80, 100)  # PIL .size is (width, height)


def test_extractDepth_accepts_rgb_input():
    result = extractDepth(_rgb_image())
    assert isinstance(result, PILImage.Image)


def test_extractDepth_rgb_input_preserves_size():
    img = _rgb_image((100, 80))
    result = extractDepth(img)
    assert result.size == (80, 100)


# ---------------------------------------------------------------------------
# extractCanny — uses cv2 (mocked)
# ---------------------------------------------------------------------------


def test_extractCanny_returns_pil_image():
    result = extractCanny(_rgb_image())
    assert isinstance(result, PILImage.Image)


def test_extractCanny_output_is_rgb():
    result = extractCanny(_rgb_image())
    assert result.mode == "RGB"


def test_extractCanny_preserves_size():
    h, w = 100, 80
    _cv2_mock.cvtColor.return_value = np.zeros((h, w), dtype=np.uint8)
    _cv2_mock.Canny.return_value = np.zeros((h, w), dtype=np.uint8)
    result = extractCanny(_rgb_image((h, w)))
    assert result.size == (w, h)


def test_extractCanny_passes_thresholds_to_canny():
    extractCanny(_rgb_image(), lo=50, hi=150)
    call_args = _cv2_mock.Canny.call_args[0]
    assert call_args[1] == 50
    assert call_args[2] == 150


def test_extractCanny_default_thresholds():
    extractCanny(_rgb_image())
    call_args = _cv2_mock.Canny.call_args[0]
    assert call_args[1] == 100
    assert call_args[2] == 200
