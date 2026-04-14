"""Tests for pure-Python helpers in src/roomify/ui/components.py — Phase 6.

Streamlit runtime is NOT required.  Only the pure-Python helper functions
are exercised here:
  - parseRunJson(runDir)
  - listGalleryRuns(outputDir, ...)
  - buildMetricsDf(runs)
  - formatSpec(specDict)

Run with:  pytest tests/testUiComponents.py -v
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import pytest

from roomify.ui.components import (
    buildMetricsDf,
    formatSpec,
    listGalleryRuns,
    parseRunJson,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_REQUIRED_RUN_JSON_KEYS = {
    "runId", "spec", "strategy", "controlled", "controlType",
    "refImageId", "model", "controlnet", "seed", "steps",
    "guidanceScale", "prompt", "negativePrompt", "imagePath",
    "gitSha", "timings",
}


def _makeRunJson(
    run_id: str = "test_run_01",
    room_type: str = "bedroom",
    strategy: str = "minimal",
    controlled: bool = False,
) -> Dict[str, Any]:
    return {
        "runId": run_id,
        "spec": {
            "id": run_id,
            "roomType": room_type,
            "size": "10x12 ft",
            "style": "scandinavian",
            "furniture": ["queen bed", "nightstand"],
            "lighting": "natural light",
            "mood": "cozy",
        },
        "strategy": strategy,
        "controlled": controlled,
        "controlType": None,
        "refImageId": None,
        "model": "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "controlnet": None,
        "seed": 42,
        "steps": 30,
        "guidanceScale": 7.5,
        "prompt": "A scandinavian bedroom, natural light",
        "negativePrompt": "blurry, distorted",
        "imagePath": f"/content/outputs/{run_id}/img_0.png",
        "gitSha": "abc1234",
        "timings": {"generateSec": 10.5},
    }


def _writeRunDir(
    parent: Path,
    run_id: str,
    **kwargs: Any,
) -> Path:
    """Create a run directory with a run.json under *parent*."""
    run_dir = parent / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(json.dumps(_makeRunJson(run_id=run_id, **kwargs)))
    return run_dir


# ---------------------------------------------------------------------------
# parseRunJson
# ---------------------------------------------------------------------------


def test_parseRunJson_returns_dict(tmp_path):
    run_dir = _writeRunDir(tmp_path, "run_01")
    result = parseRunJson(run_dir)
    assert isinstance(result, dict)


def test_parseRunJson_has_required_keys(tmp_path):
    run_dir = _writeRunDir(tmp_path, "run_01")
    result = parseRunJson(run_dir)
    missing = _REQUIRED_RUN_JSON_KEYS - result.keys()
    assert not missing, f"parseRunJson result missing keys: {missing}"


def test_parseRunJson_raises_on_missing_file(tmp_path):
    empty_dir = tmp_path / "no_json_here"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        parseRunJson(empty_dir)


# ---------------------------------------------------------------------------
# listGalleryRuns
# ---------------------------------------------------------------------------


def test_listGalleryRuns_empty_dir_returns_empty(tmp_path):
    result = listGalleryRuns(tmp_path)
    assert result == []


def test_listGalleryRuns_returns_all_runs(tmp_path):
    _writeRunDir(tmp_path, "run_01")
    _writeRunDir(tmp_path, "run_02")
    result = listGalleryRuns(tmp_path)
    assert len(result) == 2


def test_listGalleryRuns_filter_by_scene_type(tmp_path):
    _writeRunDir(tmp_path, "bedroom_run", room_type="bedroom")
    _writeRunDir(tmp_path, "kitchen_run", room_type="kitchen")
    result = listGalleryRuns(tmp_path, sceneType="bedroom")
    assert len(result) == 1
    assert result[0]["spec"]["roomType"] == "bedroom"


def test_listGalleryRuns_filter_by_strategy(tmp_path):
    _writeRunDir(tmp_path, "run_minimal", strategy="minimal")
    _writeRunDir(tmp_path, "run_desc", strategy="descriptive")
    result = listGalleryRuns(tmp_path, strategy="minimal")
    assert len(result) == 1
    assert result[0]["strategy"] == "minimal"


def test_listGalleryRuns_filter_controlled_true(tmp_path):
    _writeRunDir(tmp_path, "ctrl_run", controlled=True)
    _writeRunDir(tmp_path, "unctrl_run", controlled=False)
    result = listGalleryRuns(tmp_path, controlled=True)
    assert len(result) == 1
    assert result[0]["controlled"] is True


def test_listGalleryRuns_filter_controlled_false(tmp_path):
    _writeRunDir(tmp_path, "ctrl_run", controlled=True)
    _writeRunDir(tmp_path, "unctrl_run", controlled=False)
    result = listGalleryRuns(tmp_path, controlled=False)
    assert len(result) == 1
    assert result[0]["controlled"] is False


def test_listGalleryRuns_skips_dirs_without_run_json(tmp_path):
    _writeRunDir(tmp_path, "valid_run")
    empty_dir = tmp_path / "no_json_dir"
    empty_dir.mkdir()
    result = listGalleryRuns(tmp_path)
    assert len(result) == 1


def test_listGalleryRuns_finds_nested_runs(tmp_path):
    """Sweep output: outputDir/<sweepId>/<cellId>/run.json — two levels deep."""
    sweep_dir = tmp_path / "2026-04-14T12-00-00_core_sweep"
    sweep_dir.mkdir()
    _writeRunDir(sweep_dir, "bedroom_minimal_unctrl_s42", room_type="bedroom")
    _writeRunDir(sweep_dir, "kitchen_descriptive_unctrl_s99", room_type="kitchen")
    result = listGalleryRuns(tmp_path)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# buildMetricsDf
# ---------------------------------------------------------------------------


def test_buildMetricsDf_returns_dataframe():
    runs = [_makeRunJson(run_id="r1"), _makeRunJson(run_id="r2")]
    result = buildMetricsDf(runs)
    assert isinstance(result, pd.DataFrame)


def test_buildMetricsDf_row_count_matches_input():
    runs = [_makeRunJson(run_id=f"r{i}") for i in range(3)]
    result = buildMetricsDf(runs)
    assert len(result) == 3


def test_buildMetricsDf_has_required_columns():
    result = buildMetricsDf([_makeRunJson()])
    for col in ("runId", "sceneType", "strategy", "controlled", "seed"):
        assert col in result.columns, f"buildMetricsDf missing column: {col}"


def test_buildMetricsDf_empty_input_returns_empty_dataframe():
    result = buildMetricsDf([])
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


# ---------------------------------------------------------------------------
# formatSpec
# ---------------------------------------------------------------------------


def test_formatSpec_returns_non_empty_string():
    result = formatSpec(_makeRunJson()["spec"])
    assert isinstance(result, str)
    assert len(result) > 0


def test_formatSpec_includes_room_type():
    spec = _makeRunJson(room_type="kitchen")["spec"]
    result = formatSpec(spec)
    assert "kitchen" in result


def test_formatSpec_includes_style():
    spec = {
        "roomType": "office", "style": "minimalist",
        "size": "10x10 ft", "furniture": [], "lighting": "LED", "mood": "focused",
    }
    result = formatSpec(spec)
    assert "minimalist" in result


def test_formatSpec_handles_empty_spec():
    result = formatSpec({})
    assert isinstance(result, str)
