"""Tests for src/roomify/dataset.py — Phase 1.

All tests use local fixture data (tmp_path) — no GPU or Google Drive required.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from roomify.dataset import (
    VALID_SCENE_TYPES,
    Record,
    getRecord,
    listByScene,
    loadManifest,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SCENE_TYPES = ["bedroom", "living_room", "kitchen", "office", "bathroom"]

MANIFEST_ROWS = [
    {
        "id": "sunrgbd_00001",
        "sceneType": "bedroom",
        "rgbPath": "data/sunrgbd_subset/bedroom/00001/rgb.jpg",
        "depthPath": "data/sunrgbd_subset/bedroom/00001/depth.png",
        "objectLabels": "bed, nightstand, lamp",
    },
    {
        "id": "sunrgbd_00002",
        "sceneType": "living_room",
        "rgbPath": "data/sunrgbd_subset/living_room/00002/rgb.jpg",
        "depthPath": "data/sunrgbd_subset/living_room/00002/depth.png",
        "objectLabels": "sofa, coffee table",
    },
    {
        "id": "sunrgbd_00003",
        "sceneType": "bedroom",
        "rgbPath": "data/sunrgbd_subset/bedroom/00003/rgb.jpg",
        "depthPath": "data/sunrgbd_subset/bedroom/00003/depth.png",
        "objectLabels": "",
    },
    {
        "id": "sunrgbd_00004",
        "sceneType": "kitchen",
        "rgbPath": "data/sunrgbd_subset/kitchen/00004/rgb.jpg",
        "depthPath": "data/sunrgbd_subset/kitchen/00004/depth.png",
        "objectLabels": "counter, refrigerator",
    },
]


@pytest.fixture()
def manifest_csv(tmp_path: Path) -> Path:
    """Write a small manifest CSV to a temp dir and return its path."""
    csv_path = tmp_path / "manifest.csv"
    pd.DataFrame(MANIFEST_ROWS).to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture()
def manifest_df(manifest_csv: Path) -> pd.DataFrame:
    return loadManifest(manifest_csv)


# ---------------------------------------------------------------------------
# VALID_SCENE_TYPES
# ---------------------------------------------------------------------------


def test_valid_scene_types_contains_all_five():
    assert VALID_SCENE_TYPES == frozenset(SCENE_TYPES)


# ---------------------------------------------------------------------------
# loadManifest
# ---------------------------------------------------------------------------


def test_loadManifest_returns_dataframe(manifest_csv: Path):
    df = loadManifest(manifest_csv)
    assert isinstance(df, pd.DataFrame)


def test_loadManifest_has_expected_columns(manifest_csv: Path):
    df = loadManifest(manifest_csv)
    for col in ("id", "sceneType", "rgbPath", "depthPath", "objectLabels"):
        assert col in df.columns, f"Missing column: {col}"


def test_loadManifest_row_count(manifest_csv: Path):
    df = loadManifest(manifest_csv)
    assert len(df) == len(MANIFEST_ROWS)


def test_loadManifest_raises_when_file_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        loadManifest(tmp_path / "does_not_exist.csv")


def test_loadManifest_scene_types_are_valid(manifest_csv: Path):
    df = loadManifest(manifest_csv)
    for scene in df["sceneType"]:
        assert scene in VALID_SCENE_TYPES


# ---------------------------------------------------------------------------
# getRecord
# ---------------------------------------------------------------------------


def test_getRecord_returns_record_instance(manifest_df: pd.DataFrame):
    rec = getRecord(manifest_df, "sunrgbd_00001")
    assert isinstance(rec, Record)


def test_getRecord_fields_match_csv(manifest_df: pd.DataFrame):
    rec = getRecord(manifest_df, "sunrgbd_00001")
    assert rec.id == "sunrgbd_00001"
    assert rec.sceneType == "bedroom"
    assert rec.rgbPath == Path("data/sunrgbd_subset/bedroom/00001/rgb.jpg")
    assert rec.depthPath == Path("data/sunrgbd_subset/bedroom/00001/depth.png")


def test_getRecord_parses_object_labels(manifest_df: pd.DataFrame):
    rec = getRecord(manifest_df, "sunrgbd_00001")
    assert rec.objectLabels == ["bed", "nightstand", "lamp"]


def test_getRecord_empty_labels_returns_empty_list(manifest_df: pd.DataFrame):
    rec = getRecord(manifest_df, "sunrgbd_00003")
    assert rec.objectLabels == []


def test_getRecord_raises_on_missing_id(manifest_df: pd.DataFrame):
    with pytest.raises(KeyError):
        getRecord(manifest_df, "nonexistent_id")


# ---------------------------------------------------------------------------
# listByScene
# ---------------------------------------------------------------------------


def test_listByScene_returns_dataframe(manifest_df: pd.DataFrame):
    result = listByScene(manifest_df, "bedroom")
    assert isinstance(result, pd.DataFrame)


def test_listByScene_filters_correctly(manifest_df: pd.DataFrame):
    result = listByScene(manifest_df, "bedroom")
    assert len(result) == 2
    assert all(result["sceneType"] == "bedroom")


def test_listByScene_single_match(manifest_df: pd.DataFrame):
    result = listByScene(manifest_df, "kitchen")
    assert len(result) == 1
    assert result.iloc[0]["id"] == "sunrgbd_00004"


def test_listByScene_no_match_returns_empty(manifest_df: pd.DataFrame):
    result = listByScene(manifest_df, "bathroom")
    assert len(result) == 0


def test_listByScene_raises_on_invalid_scene(manifest_df: pd.DataFrame):
    with pytest.raises(ValueError):
        listByScene(manifest_df, "garage")


def test_listByScene_result_is_independent_copy(manifest_df: pd.DataFrame):
    result = listByScene(manifest_df, "bedroom")
    result["sceneType"] = "mutated"
    # Original df must not be affected
    assert all(manifest_df[manifest_df["sceneType"] == "bedroom"]["sceneType"] == "bedroom")
