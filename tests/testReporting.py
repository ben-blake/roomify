"""Tests for src/roomify/reporting.py — Phase 7.

TDD: written before implementation. Run with:  pytest tests/testReporting.py -v

No GPU or external ML packages required — pure Pillow + JSON.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image as PILImage

from roomify.reporting import contactSheet, metricsTable

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _makeRunDir(tmp_path: Path, n: int = 1) -> Path:
    for i in range(n):
        sub = tmp_path / f"run_{i:03d}"
        sub.mkdir(parents=True)
        PILImage.new("RGB", (64, 64), color=(i * 30, 100, 200)).save(str(sub / "img_0.png"))
        (sub / "run.json").write_text(json.dumps({
            "runId": f"run_{i:03d}",
            "spec": {"id": "bedroom_01", "sceneType": "bedroom"},
            "strategy": "descriptive",
            "controlled": i % 2 == 0,
            "controlType": "depth" if i % 2 == 0 else None,
            "seed": i * 10,
            "steps": 30,
            "guidanceScale": 7.5,
            "prompt": f"a cozy bedroom run {i}",
            "negativePrompt": "blurry, dark",
            "imagePath": str(sub / "img_0.png"),
            "gitSha": "abc123",
            "timings": {"generateSec": 1.5},
        }))
    return tmp_path


# ---------------------------------------------------------------------------
# contactSheet
# ---------------------------------------------------------------------------


def test_contactSheet_returns_pil_image(tmp_path):
    _makeRunDir(tmp_path, n=1)
    result = contactSheet(tmp_path)
    assert isinstance(result, PILImage.Image)


def test_contactSheet_rgb_mode(tmp_path):
    _makeRunDir(tmp_path, n=1)
    result = contactSheet(tmp_path)
    assert result.mode == "RGB"


def test_contactSheet_single_image_size(tmp_path):
    _makeRunDir(tmp_path, n=1)
    result = contactSheet(tmp_path, thumbSize=256)
    assert result.size == (256, 256)


def test_contactSheet_four_images_two_by_two(tmp_path):
    _makeRunDir(tmp_path, n=4)
    result = contactSheet(tmp_path, thumbSize=256)
    assert result.size == (512, 512)


def test_contactSheet_nine_images_three_by_three(tmp_path):
    _makeRunDir(tmp_path, n=9)
    result = contactSheet(tmp_path, thumbSize=256)
    assert result.size == (768, 768)


def test_contactSheet_raises_on_empty_dir(tmp_path):
    with pytest.raises(ValueError):
        contactSheet(tmp_path)


def test_contactSheet_custom_thumb_size(tmp_path):
    _makeRunDir(tmp_path, n=1)
    result = contactSheet(tmp_path, thumbSize=128)
    assert result.size == (128, 128)


# ---------------------------------------------------------------------------
# metricsTable
# ---------------------------------------------------------------------------


def test_metricsTable_returns_string(tmp_path):
    _makeRunDir(tmp_path, n=1)
    result = metricsTable(tmp_path)
    assert isinstance(result, str)


def test_metricsTable_is_markdown(tmp_path):
    _makeRunDir(tmp_path, n=1)
    result = metricsTable(tmp_path)
    assert result.startswith("|")


def test_metricsTable_has_runId_column(tmp_path):
    _makeRunDir(tmp_path, n=1)
    result = metricsTable(tmp_path)
    assert "runId" in result


def test_metricsTable_has_strategy_column(tmp_path):
    _makeRunDir(tmp_path, n=1)
    result = metricsTable(tmp_path)
    assert "strategy" in result


def test_metricsTable_has_controlled_column(tmp_path):
    _makeRunDir(tmp_path, n=1)
    result = metricsTable(tmp_path)
    assert "controlled" in result


def test_metricsTable_correct_row_count(tmp_path):
    _makeRunDir(tmp_path, n=3)
    result = metricsTable(tmp_path)
    lines = [ln for ln in result.splitlines() if ln.strip()]
    # header + separator + 3 data rows = 5 lines
    assert len(lines) == 5


def test_metricsTable_includes_seed_value(tmp_path):
    _makeRunDir(tmp_path, n=1)
    result = metricsTable(tmp_path)
    assert "0" in result  # seed=0 for run_000


def test_metricsTable_raises_on_empty_dir(tmp_path):
    with pytest.raises(ValueError):
        metricsTable(tmp_path)
