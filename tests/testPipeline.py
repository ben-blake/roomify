"""Tests for src/roomify/pipeline.py and CLI generate command — Phase 3.

TDD: written before implementation. Run with:  pytest tests/testPipeline.py -v

torch and diffusers are mocked at module level so these tests run locally
without a GPU. The mock setup must happen before any roomify.pipeline import.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image as PILImage

# ---------------------------------------------------------------------------
# Mock GPU-only packages — must happen before importing roomify.pipeline
# ---------------------------------------------------------------------------

_mock_generator = MagicMock()
_mock_generator.manual_seed.return_value = _mock_generator

_torch_mock = MagicMock()
_torch_mock.float16 = "float16"
_torch_mock.cuda.is_available.return_value = False
_torch_mock.Generator.return_value = _mock_generator

_mock_sd_class = MagicMock()
_diffusers_mock = MagicMock()
_diffusers_mock.StableDiffusionPipeline = _mock_sd_class

sys.modules.setdefault("torch", _torch_mock)
sys.modules.setdefault("diffusers", _diffusers_mock)

# Now safe to import
from roomify.pipeline import Pipeline, getPipeline, _resetPipeline, SD_MODEL_ID  # noqa: E402

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_FAKE_IMAGE = PILImage.new("RGB", (64, 64), color=(100, 149, 237))

_SPEC_YAML = str(Path(__file__).parent.parent / "configs" / "examples" / "bedroom_01.yaml")


def _makeMockPipeInstance() -> MagicMock:
    """Return a mock pipe instance whose __call__ returns our fake image."""
    inst = MagicMock()
    inst.return_value.images = [_FAKE_IMAGE]
    return inst


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def resetState():
    """Reset singleton and mocks before each test."""
    _resetPipeline()
    _mock_sd_class.reset_mock()
    _torch_mock.reset_mock()
    _torch_mock.float16 = "float16"
    _torch_mock.cuda.is_available.return_value = False
    _torch_mock.Generator.return_value = _mock_generator
    _mock_generator.reset_mock()
    _mock_generator.manual_seed.return_value = _mock_generator
    yield
    _resetPipeline()


@pytest.fixture
def loadedPipeline() -> Pipeline:
    """Return a Pipeline with load() already called (diffusers mocked)."""
    mock_pipe = _makeMockPipeInstance()
    _mock_sd_class.from_pretrained.return_value = mock_pipe
    p = Pipeline()
    p.load()
    return p


# ---------------------------------------------------------------------------
# getPipeline — singleton contract
# ---------------------------------------------------------------------------


def test_getPipeline_returns_pipeline_instance():
    p = getPipeline()
    assert isinstance(p, Pipeline)


def test_getPipeline_singleton():
    p1 = getPipeline()
    p2 = getPipeline()
    assert p1 is p2


def test_getPipeline_reset_yields_new_instance():
    p1 = getPipeline()
    _resetPipeline()
    p2 = getPipeline()
    assert p1 is not p2


# ---------------------------------------------------------------------------
# Pipeline.load() — diffusers integration (mocked)
# ---------------------------------------------------------------------------


def test_load_calls_from_pretrained_with_correct_model_id():
    _mock_sd_class.from_pretrained.return_value = _makeMockPipeInstance()
    p = Pipeline()
    p.load()
    args = _mock_sd_class.from_pretrained.call_args[0]
    assert args[0] == SD_MODEL_ID


def test_load_passes_fp16_dtype():
    _mock_sd_class.from_pretrained.return_value = _makeMockPipeInstance()
    p = Pipeline()
    p.load()
    kwargs = _mock_sd_class.from_pretrained.call_args[1]
    assert "torch_dtype" in kwargs


def test_load_enables_attention_slicing():
    mock_pipe = _makeMockPipeInstance()
    _mock_sd_class.from_pretrained.return_value = mock_pipe
    p = Pipeline()
    p.load()
    mock_pipe.enable_attention_slicing.assert_called_once()


def test_load_sets_loaded_flag():
    _mock_sd_class.from_pretrained.return_value = _makeMockPipeInstance()
    p = Pipeline()
    assert not p._loaded
    p.load()
    assert p._loaded


# ---------------------------------------------------------------------------
# Pipeline.generate()
# ---------------------------------------------------------------------------


def test_generate_raises_if_not_loaded():
    p = Pipeline()
    with pytest.raises(RuntimeError, match="load\\(\\)"):
        p.generate("positive", "negative")


def test_generate_returns_pil_image(loadedPipeline):
    image = loadedPipeline.generate("a cozy bedroom", "blurry, dark")
    assert isinstance(image, PILImage.Image)


def test_generate_passes_steps(loadedPipeline):
    loadedPipeline.generate("positive", "negative", steps=20)
    kwargs = loadedPipeline._sd.call_args[1]
    assert kwargs.get("num_inference_steps") == 20


def test_generate_passes_guidance(loadedPipeline):
    loadedPipeline.generate("positive", "negative", guidance=9.0)
    kwargs = loadedPipeline._sd.call_args[1]
    assert kwargs.get("guidance_scale") == 9.0


def test_generate_seeds_the_generator(loadedPipeline):
    loadedPipeline.generate("positive", "negative", seed=777)
    _mock_generator.manual_seed.assert_called_with(777)


# ---------------------------------------------------------------------------
# CLI generate command — integration via typer CliRunner
# ---------------------------------------------------------------------------


def _makeMockPipelineForCli() -> MagicMock:
    """Mock pipeline whose .generate() returns a real PIL image."""
    mock_pl = MagicMock()
    mock_pl.generate.return_value = PILImage.new("RGB", (64, 64))
    return mock_pl


def test_cli_generate_exits_zero(tmp_path):
    from typer.testing import CliRunner
    from roomify.cli import app

    runner = CliRunner()
    mock_pl = _makeMockPipelineForCli()

    with patch("roomify.pipeline.getPipeline", return_value=mock_pl), \
         patch("roomify.paths.getOutputDir", return_value=tmp_path):
        result = runner.invoke(app, [
            "generate", "--spec", _SPEC_YAML,
            "--seed", "42", "--steps", "5",
        ])

    assert result.exit_code == 0, result.output


def test_cli_generate_creates_run_directory(tmp_path):
    from typer.testing import CliRunner
    from roomify.cli import app

    runner = CliRunner()
    mock_pl = _makeMockPipelineForCli()

    with patch("roomify.pipeline.getPipeline", return_value=mock_pl), \
         patch("roomify.paths.getOutputDir", return_value=tmp_path):
        runner.invoke(app, ["generate", "--spec", _SPEC_YAML])

    run_dirs = list(tmp_path.iterdir())
    assert len(run_dirs) == 1
    assert run_dirs[0].is_dir()


def test_cli_generate_writes_png(tmp_path):
    from typer.testing import CliRunner
    from roomify.cli import app

    runner = CliRunner()
    mock_pl = _makeMockPipelineForCli()

    with patch("roomify.pipeline.getPipeline", return_value=mock_pl), \
         patch("roomify.paths.getOutputDir", return_value=tmp_path):
        runner.invoke(app, ["generate", "--spec", _SPEC_YAML])

    run_dir = next(tmp_path.iterdir())
    assert (run_dir / "img_0.png").exists()


def test_cli_generate_writes_run_json(tmp_path):
    from typer.testing import CliRunner
    from roomify.cli import app

    runner = CliRunner()
    mock_pl = _makeMockPipelineForCli()

    with patch("roomify.pipeline.getPipeline", return_value=mock_pl), \
         patch("roomify.paths.getOutputDir", return_value=tmp_path):
        runner.invoke(app, ["generate", "--spec", _SPEC_YAML])

    run_dir = next(tmp_path.iterdir())
    run_json = run_dir / "run.json"
    assert run_json.exists()

    data = json.loads(run_json.read_text())
    for key in ("runId", "spec", "strategy", "seed", "steps", "guidanceScale",
                "prompt", "negativePrompt", "imagePath", "gitSha", "timings"):
        assert key in data, f"run.json missing key '{key}'"


def test_cli_generate_run_json_seed(tmp_path):
    from typer.testing import CliRunner
    from roomify.cli import app

    runner = CliRunner()
    mock_pl = _makeMockPipelineForCli()

    with patch("roomify.pipeline.getPipeline", return_value=mock_pl), \
         patch("roomify.paths.getOutputDir", return_value=tmp_path):
        runner.invoke(app, ["generate", "--spec", _SPEC_YAML, "--seed", "99"])

    run_dir = next(tmp_path.iterdir())
    data = json.loads((run_dir / "run.json").read_text())
    assert data["seed"] == 99


def test_cli_generate_run_json_contains_prompts(tmp_path):
    from typer.testing import CliRunner
    from roomify.cli import app

    runner = CliRunner()
    mock_pl = _makeMockPipelineForCli()

    with patch("roomify.pipeline.getPipeline", return_value=mock_pl), \
         patch("roomify.paths.getOutputDir", return_value=tmp_path):
        runner.invoke(app, [
            "generate", "--spec", _SPEC_YAML, "--strategy", "descriptive",
        ])

    run_dir = next(tmp_path.iterdir())
    data = json.loads((run_dir / "run.json").read_text())
    assert data["prompt"].strip()
    assert data["negativePrompt"].strip()
    assert data["strategy"] == "descriptive"
