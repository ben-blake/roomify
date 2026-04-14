"""Tests for src/roomify/orchestrator.py — Phase 5.

TDD: written before implementation. Run with:  pytest tests/testOrchestrator.py -v

The orchestrator calls buildPrompt (pure Python, no GPU) and Pipeline.generate
(GPU). We mock getPipeline() at the roomify.pipeline level so these tests run
locally without a GPU and without polluting the diffusers/torch mock state used
by testPipeline.py.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image as PILImage

# roomify.pipeline only imports PIL at the top level; torch and diffusers are
# lazy-imported inside Pipeline.load(). Because we patch getPipeline() in every
# test, load() is never called and no GPU mocks are needed here.
# NOT registering torch/diffusers mocks at all avoids polluting sys.modules for
# testPipeline.py (which registers its own carefully-configured mocks).
from roomify.orchestrator import runExperiment  # noqa: E402

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_FAKE_IMAGE = PILImage.new("RGB", (64, 64), color=(100, 149, 237))

_MINIMAL_CONFIG = """\
runId: test_sweep
specs:
  - id: bedroom_01
    roomType: bedroom
    size: "10x12 ft"
    style: scandinavian
    furniture: [queen bed, nightstand]
    lighting: "natural light"
    mood: "cozy"
strategies: [minimal]
controlled: [false]
seeds: [42]
"""

_TWO_STRATEGY_CONFIG = """\
runId: two_strat
specs:
  - id: living_01
    roomType: living_room
    size: "12x14 ft"
    style: minimalist
    furniture: [sofa, coffee table]
    lighting: "overhead"
    mood: "calm"
strategies: [minimal, descriptive]
controlled: [false]
seeds: [1]
"""

_TWO_SEED_CONFIG = """\
runId: two_seed
specs:
  - id: kitchen_01
    roomType: kitchen
    size: "8x10 ft"
    style: industrial
    furniture: [island, barstools]
    lighting: "pendant lights"
    mood: "modern"
strategies: [minimal]
controlled: [false]
seeds: [7, 99]
"""

_CONTROLLED_CONFIG = """\
runId: ctrl_sweep
specs:
  - id: office_01
    roomType: office
    size: "10x10 ft"
    style: mid_century
    furniture: [desk, chair]
    lighting: "task lamp"
    mood: "focused"
strategies: [minimal]
controlled: [true, false]
seeds: [42]
"""


def _write_config(content: str) -> Path:
    """Write a YAML config to a temp file and return its Path."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, prefix="test_sweep_"
    )
    tmp.write(content)
    tmp.flush()
    return Path(tmp.name)


def _makeMockPipeline() -> MagicMock:
    """Return a mock Pipeline whose generate() returns our fake image."""
    mock_pl = MagicMock()
    mock_pl.generate.return_value = _FAKE_IMAGE
    return mock_pl


@pytest.fixture(autouse=True)
def mockPipeline():
    """Patch getPipeline to return a mock Pipeline for every test.

    This avoids touching sys.modules["diffusers"] / sys.modules["torch"] and
    prevents cross-module mock pollution with testPipeline.py.
    """
    mock_pl = _makeMockPipeline()
    with patch("roomify.pipeline.getPipeline", return_value=mock_pl):
        yield mock_pl


# ---------------------------------------------------------------------------
# Test: return value
# ---------------------------------------------------------------------------


def test_returns_path_object(tmp_path):
    cfg = _write_config(_MINIMAL_CONFIG)
    with patch("roomify.paths.getOutputDir", return_value=tmp_path):
        result = runExperiment(cfg)
    assert isinstance(result, Path)


def test_returns_existing_directory(tmp_path):
    cfg = _write_config(_MINIMAL_CONFIG)
    with patch("roomify.paths.getOutputDir", return_value=tmp_path):
        result = runExperiment(cfg)
    assert result.is_dir()


# ---------------------------------------------------------------------------
# Test: output file structure for a single-image sweep
# ---------------------------------------------------------------------------


def test_single_image_creates_png(tmp_path):
    cfg = _write_config(_MINIMAL_CONFIG)
    with patch("roomify.paths.getOutputDir", return_value=tmp_path):
        out_dir = runExperiment(cfg)
    pngs = list(out_dir.glob("**/*.png"))
    assert len(pngs) == 1


def test_single_image_creates_run_json(tmp_path):
    cfg = _write_config(_MINIMAL_CONFIG)
    with patch("roomify.paths.getOutputDir", return_value=tmp_path):
        out_dir = runExperiment(cfg)
    jsons = list(out_dir.glob("**/*.json"))
    assert len(jsons) == 1


# ---------------------------------------------------------------------------
# Test: run.json schema keys
# ---------------------------------------------------------------------------


def _get_run_json(out_dir: Path) -> dict:
    jsons = list(out_dir.glob("**/*.json"))
    assert jsons, "No run.json found"
    return json.loads(jsons[0].read_text())


def test_run_json_has_required_keys(tmp_path):
    cfg = _write_config(_MINIMAL_CONFIG)
    with patch("roomify.paths.getOutputDir", return_value=tmp_path):
        out_dir = runExperiment(cfg)
    data = _get_run_json(out_dir)
    required = {
        "runId", "spec", "strategy", "controlled", "controlType",
        "refImageId", "model", "controlnet", "seed", "steps",
        "guidanceScale", "prompt", "negativePrompt", "imagePath",
        "gitSha", "timings",
    }
    missing = required - data.keys()
    assert not missing, f"run.json missing keys: {missing}"


def test_run_json_strategy_matches_config(tmp_path):
    cfg = _write_config(_MINIMAL_CONFIG)
    with patch("roomify.paths.getOutputDir", return_value=tmp_path):
        out_dir = runExperiment(cfg)
    data = _get_run_json(out_dir)
    assert data["strategy"] == "minimal"


def test_run_json_seed_matches_config(tmp_path):
    cfg = _write_config(_MINIMAL_CONFIG)
    with patch("roomify.paths.getOutputDir", return_value=tmp_path):
        out_dir = runExperiment(cfg)
    data = _get_run_json(out_dir)
    assert data["seed"] == 42


def test_run_json_controlled_false_for_uncontrolled(tmp_path):
    cfg = _write_config(_MINIMAL_CONFIG)
    with patch("roomify.paths.getOutputDir", return_value=tmp_path):
        out_dir = runExperiment(cfg)
    data = _get_run_json(out_dir)
    assert data["controlled"] is False
    assert data["controlType"] is None


def test_run_json_image_path_exists(tmp_path):
    cfg = _write_config(_MINIMAL_CONFIG)
    with patch("roomify.paths.getOutputDir", return_value=tmp_path):
        out_dir = runExperiment(cfg)
    data = _get_run_json(out_dir)
    assert Path(data["imagePath"]).exists()


# ---------------------------------------------------------------------------
# Test: sweep matrix cardinality
# ---------------------------------------------------------------------------


def test_two_strategies_produce_two_images(tmp_path):
    cfg = _write_config(_TWO_STRATEGY_CONFIG)
    with patch("roomify.paths.getOutputDir", return_value=tmp_path):
        out_dir = runExperiment(cfg)
    pngs = list(out_dir.glob("**/*.png"))
    assert len(pngs) == 2


def test_two_seeds_produce_two_images(tmp_path):
    cfg = _write_config(_TWO_SEED_CONFIG)
    with patch("roomify.paths.getOutputDir", return_value=tmp_path):
        out_dir = runExperiment(cfg)
    pngs = list(out_dir.glob("**/*.png"))
    assert len(pngs) == 2


def test_controlled_and_uncontrolled_produce_two_images(tmp_path):
    cfg = _write_config(_CONTROLLED_CONFIG)
    with patch("roomify.paths.getOutputDir", return_value=tmp_path):
        out_dir = runExperiment(cfg)
    pngs = list(out_dir.glob("**/*.png"))
    assert len(pngs) == 2


# ---------------------------------------------------------------------------
# Test: progressCb is invoked
# ---------------------------------------------------------------------------


def test_progress_cb_called_once_for_single_image(tmp_path):
    cfg = _write_config(_MINIMAL_CONFIG)
    calls: List[tuple] = []

    def cb(done: int, total: int) -> None:
        calls.append((done, total))

    with patch("roomify.paths.getOutputDir", return_value=tmp_path):
        runExperiment(cfg, progressCb=cb)

    assert len(calls) == 1


def test_progress_cb_total_equals_matrix_size(tmp_path):
    cfg = _write_config(_TWO_STRATEGY_CONFIG)
    calls: List[tuple] = []

    def cb(done: int, total: int) -> None:
        calls.append((done, total))

    with patch("roomify.paths.getOutputDir", return_value=tmp_path):
        runExperiment(cfg, progressCb=cb)

    totals = {t for _, t in calls}
    assert totals == {2}


def test_progress_cb_done_increments(tmp_path):
    cfg = _write_config(_TWO_SEED_CONFIG)
    calls: List[tuple] = []

    def cb(done: int, total: int) -> None:
        calls.append((done, total))

    with patch("roomify.paths.getOutputDir", return_value=tmp_path):
        runExperiment(cfg, progressCb=cb)

    done_values = [d for d, _ in calls]
    assert done_values == [1, 2]


def test_progress_cb_none_does_not_raise(tmp_path):
    cfg = _write_config(_MINIMAL_CONFIG)
    with patch("roomify.paths.getOutputDir", return_value=tmp_path):
        runExperiment(cfg, progressCb=None)  # should not raise


# ---------------------------------------------------------------------------
# Test: CLI sweep command
# ---------------------------------------------------------------------------


def test_cli_sweep_exits_zero(tmp_path):
    from typer.testing import CliRunner
    from roomify.cli import app

    cfg = _write_config(_MINIMAL_CONFIG)
    runner = CliRunner()
    with patch("roomify.paths.getOutputDir", return_value=tmp_path):
        result = runner.invoke(app, ["sweep", "--config", str(cfg)])
    assert result.exit_code == 0, result.output


def test_cli_sweep_prints_output_dir(tmp_path):
    from typer.testing import CliRunner
    from roomify.cli import app

    cfg = _write_config(_MINIMAL_CONFIG)
    runner = CliRunner()
    with patch("roomify.paths.getOutputDir", return_value=tmp_path):
        result = runner.invoke(app, ["sweep", "--config", str(cfg)])
    assert str(tmp_path) in result.output or "Done" in result.output
