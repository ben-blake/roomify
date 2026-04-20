"""Tests for src/roomify/evaluation.py — Phase 7.

TDD: written before implementation. Run with:  pytest tests/testEvaluation.py -v

torch, open_clip, and lpips are mocked via patch.dict so they do not pollute
sys.modules between test files (avoids conflicts with testPipeline.py's torch mock).
evaluation.py uses lazy imports inside functions, so importing it here is safe.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image as PILImage

# ---------------------------------------------------------------------------
# Import module under test — safe because evaluation.py has no top-level GPU imports
# ---------------------------------------------------------------------------

from roomify.evaluation import (
    clipAlignment,
    loadRatings,
    lpipsDiversity,
    saveRating,
    styleConsistency,
)

# ---------------------------------------------------------------------------
# Mock objects (built once, reset per test)
# ---------------------------------------------------------------------------

_mock_tensor = MagicMock()
_mock_tensor.item.return_value = 0.5
_mock_tensor.__truediv__.return_value = _mock_tensor
_mock_tensor.__matmul__.return_value = _mock_tensor
_mock_tensor.norm.return_value = _mock_tensor
_mock_tensor.T = _mock_tensor
_mock_tensor.unsqueeze.return_value = _mock_tensor

_torch_mock = MagicMock()
_torch_mock.tensor.return_value = _mock_tensor
_torch_mock.float32 = "float32"

_open_clip_mock = MagicMock()

_mock_lpips_result = MagicMock()
_mock_lpips_result.item.return_value = 0.3
_mock_lpips_instance = MagicMock(return_value=_mock_lpips_result)
_lpips_mock = MagicMock()
_lpips_mock.LPIPS.return_value = _mock_lpips_instance

_MOCK_CLIP_MODEL = MagicMock()
_MOCK_CLIP_MODEL.encode_image.return_value = _mock_tensor
_MOCK_CLIP_MODEL.encode_text.return_value = _mock_tensor
_MOCK_PREPROCESS = MagicMock(return_value=_mock_tensor)
_MOCK_TOKENIZER = MagicMock(return_value=_mock_tensor)


def _fakeLoadClipModel():
    return _MOCK_CLIP_MODEL, _MOCK_PREPROCESS, _MOCK_TOKENIZER


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def gpuMocks():
    """Inject GPU-only package mocks for the duration of each test only."""
    _mock_tensor.item.return_value = 0.5
    _MOCK_CLIP_MODEL.encode_image.return_value = _mock_tensor
    _MOCK_CLIP_MODEL.encode_text.return_value = _mock_tensor
    _mock_lpips_result.item.return_value = 0.3
    with patch.dict(sys.modules, {
        "torch": _torch_mock,
        "open_clip": _open_clip_mock,
        "lpips": _lpips_mock,
    }):
        yield


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
            "controlled": False,
            "controlType": None,
            "seed": i,
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
# clipAlignment
# ---------------------------------------------------------------------------


def test_clipAlignment_returns_dataframe(tmp_path):
    import pandas as pd
    _makeRunDir(tmp_path, n=1)
    with patch("roomify.evaluation._loadClipModel", side_effect=_fakeLoadClipModel):
        result = clipAlignment(tmp_path)
    assert isinstance(result, pd.DataFrame)


def test_clipAlignment_has_required_columns(tmp_path):
    _makeRunDir(tmp_path, n=1)
    with patch("roomify.evaluation._loadClipModel", side_effect=_fakeLoadClipModel):
        df = clipAlignment(tmp_path)
    for col in ("imagePath", "prompt", "clipScore"):
        assert col in df.columns, f"missing column: {col}"


def test_clipAlignment_one_run_one_row(tmp_path):
    _makeRunDir(tmp_path, n=1)
    with patch("roomify.evaluation._loadClipModel", side_effect=_fakeLoadClipModel):
        df = clipAlignment(tmp_path)
    assert len(df) == 1


def test_clipAlignment_two_runs_two_rows(tmp_path):
    _makeRunDir(tmp_path, n=2)
    with patch("roomify.evaluation._loadClipModel", side_effect=_fakeLoadClipModel):
        df = clipAlignment(tmp_path)
    assert len(df) == 2


def test_clipAlignment_clip_score_is_float(tmp_path):
    _makeRunDir(tmp_path, n=1)
    with patch("roomify.evaluation._loadClipModel", side_effect=_fakeLoadClipModel):
        df = clipAlignment(tmp_path)
    assert isinstance(df["clipScore"].iloc[0], float)


def test_clipAlignment_skips_missing_image(tmp_path):
    _makeRunDir(tmp_path, n=2)
    (tmp_path / "run_001" / "img_0.png").unlink()
    with patch("roomify.evaluation._loadClipModel", side_effect=_fakeLoadClipModel):
        df = clipAlignment(tmp_path)
    assert len(df) == 1


# ---------------------------------------------------------------------------
# lpipsDiversity
# ---------------------------------------------------------------------------


def test_lpipsDiversity_returns_float(tmp_path):
    _makeRunDir(tmp_path, n=2)
    result = lpipsDiversity(tmp_path)
    assert isinstance(result, float)


def test_lpipsDiversity_single_image_returns_zero(tmp_path):
    _makeRunDir(tmp_path, n=1)
    result = lpipsDiversity(tmp_path)
    assert result == 0.0


def test_lpipsDiversity_two_images_returns_positive(tmp_path):
    _makeRunDir(tmp_path, n=2)
    result = lpipsDiversity(tmp_path)
    assert result > 0.0


# ---------------------------------------------------------------------------
# styleConsistency
# ---------------------------------------------------------------------------


def test_styleConsistency_returns_float(tmp_path):
    _makeRunDir(tmp_path, n=1)
    with patch("roomify.evaluation._loadClipModel", side_effect=_fakeLoadClipModel):
        result = styleConsistency(tmp_path)
    assert isinstance(result, float)


def test_styleConsistency_single_image_returns_one(tmp_path):
    _makeRunDir(tmp_path, n=1)
    with patch("roomify.evaluation._loadClipModel", side_effect=_fakeLoadClipModel):
        result = styleConsistency(tmp_path)
    assert result == 1.0


def test_styleConsistency_two_images_returns_float(tmp_path):
    _makeRunDir(tmp_path, n=2)
    with patch("roomify.evaluation._loadClipModel", side_effect=_fakeLoadClipModel):
        result = styleConsistency(tmp_path)
    assert isinstance(result, float)


# ---------------------------------------------------------------------------
# saveRating
# ---------------------------------------------------------------------------


def test_saveRating_creates_csv(tmp_path):
    saveRating(tmp_path, "run_000", 4)
    assert (tmp_path / "ratings.csv").exists()


def test_saveRating_writes_correct_data(tmp_path):
    import pandas as pd
    saveRating(tmp_path, "run_000", 3, notes="looks good")
    df = pd.read_csv(tmp_path / "ratings.csv")
    assert df.iloc[0]["runId"] == "run_000"
    assert int(df.iloc[0]["rating"]) == 3


def test_saveRating_raises_on_invalid_rating(tmp_path):
    with pytest.raises(ValueError):
        saveRating(tmp_path, "run_000", 6)


def test_saveRating_upserts_existing(tmp_path):
    import pandas as pd
    saveRating(tmp_path, "run_000", 3)
    saveRating(tmp_path, "run_000", 5, notes="updated")
    df = pd.read_csv(tmp_path / "ratings.csv")
    assert len(df) == 1
    assert int(df.iloc[0]["rating"]) == 5


# ---------------------------------------------------------------------------
# loadRatings
# ---------------------------------------------------------------------------


def test_loadRatings_returns_empty_df_when_no_file(tmp_path):
    import pandas as pd
    df = loadRatings(tmp_path)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


def test_loadRatings_returns_saved_ratings(tmp_path):
    saveRating(tmp_path, "run_000", 4)
    saveRating(tmp_path, "run_001", 2)
    df = loadRatings(tmp_path)
    assert len(df) == 2
