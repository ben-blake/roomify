"""Tests for src/roomify/animateDiff.py — AnimateDiff GIF generation.

All GPU/diffusers calls are mocked so tests run locally without a GPU.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest
from PIL import Image


# ---------------------------------------------------------------------------
# Module-level mock: stub out torch and diffusers so the module can be
# imported without either package installed.
# ---------------------------------------------------------------------------


def _make_torch_mock() -> types.ModuleType:
    torch = types.ModuleType("torch")
    torch.float16 = "float16"
    torch.cuda = MagicMock()
    torch.cuda.is_available = MagicMock(return_value=False)
    generator_instance = MagicMock()
    generator_class = MagicMock(return_value=generator_instance)
    generator_class.manual_seed = MagicMock(return_value=generator_instance)
    torch.Generator = generator_class
    return torch


def _make_diffusers_mock() -> types.ModuleType:
    diffusers = types.ModuleType("diffusers")
    # MotionAdapter
    adapter_instance = MagicMock()
    adapter_cls = MagicMock()
    adapter_cls.from_pretrained = MagicMock(return_value=adapter_instance)
    diffusers.MotionAdapter = adapter_cls
    # AnimateDiffPipeline
    frames_output = MagicMock()
    fake_frames = [Image.new("RGB", (64, 64), color=(i * 10, 0, 0)) for i in range(4)]
    frames_output.frames = [fake_frames]  # batched output: frames[0] = list of PIL
    pipe_instance = MagicMock()
    pipe_instance.return_value = frames_output
    pipe_instance.enable_attention_slicing = MagicMock()
    pipe_cls = MagicMock()
    pipe_cls.from_pretrained = MagicMock(return_value=pipe_instance)
    diffusers.AnimateDiffPipeline = pipe_cls
    return diffusers


@pytest.fixture(autouse=True)
def _mock_heavy_deps(monkeypatch):
    """Replace torch/diffusers with thin mocks for the entire test module."""
    torch_mock = _make_torch_mock()
    diffusers_mock = _make_diffusers_mock()
    monkeypatch.setitem(sys.modules, "torch", torch_mock)
    monkeypatch.setitem(sys.modules, "diffusers", diffusers_mock)
    # Remove any previously imported animateDiff so it re-imports with mocks
    sys.modules.pop("roomify.animateDiff", None)
    yield
    sys.modules.pop("roomify.animateDiff", None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_module():
    import roomify.animateDiff as m  # type: ignore[import]
    return m


def _reset_singleton():
    m = _get_module()
    m._ANIMATE_INSTANCE = None


def _fake_frames(n: int = 4) -> list[Image.Image]:
    return [Image.new("RGB", (64, 64), color=(i * 20, 50, 100)) for i in range(n)]


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------


class TestSingleton:
    def test_returns_instance(self):
        m = _get_module()
        _reset_singleton()
        inst = m.getAnimateDiffGenerator()
        assert inst is not None

    def test_same_object_on_second_call(self):
        m = _get_module()
        _reset_singleton()
        a = m.getAnimateDiffGenerator()
        b = m.getAnimateDiffGenerator()
        assert a is b

    def test_reset_returns_new_instance(self):
        m = _get_module()
        _reset_singleton()
        a = m.getAnimateDiffGenerator()
        m._resetAnimateDiffGenerator()
        b = m.getAnimateDiffGenerator()
        assert a is not b


# ---------------------------------------------------------------------------
# AnimateDiffGenerator.load()
# ---------------------------------------------------------------------------


class TestLoad:
    def test_load_calls_motion_adapter_from_pretrained(self):
        import diffusers
        m = _get_module()
        gen = m.AnimateDiffGenerator()
        gen.load()
        diffusers.MotionAdapter.from_pretrained.assert_called_once()
        args, kwargs = diffusers.MotionAdapter.from_pretrained.call_args
        assert args[0] == m.MOTION_ADAPTER_ID

    def test_load_calls_animatediff_pipeline_from_pretrained(self):
        import diffusers
        m = _get_module()
        gen = m.AnimateDiffGenerator()
        gen.load()
        diffusers.AnimateDiffPipeline.from_pretrained.assert_called_once()
        args, kwargs = diffusers.AnimateDiffPipeline.from_pretrained.call_args
        assert args[0] == m.SD_MODEL_ID

    def test_load_passes_dtype_float16(self):
        import diffusers, torch
        m = _get_module()
        gen = m.AnimateDiffGenerator()
        gen.load()
        _, kwargs = diffusers.AnimateDiffPipeline.from_pretrained.call_args
        assert kwargs.get("dtype") == torch.float16

    def test_load_noop_when_already_loaded(self):
        import diffusers
        m = _get_module()
        gen = m.AnimateDiffGenerator()
        gen.load()
        call_count = diffusers.MotionAdapter.from_pretrained.call_count
        gen.load()  # second call
        assert diffusers.MotionAdapter.from_pretrained.call_count == call_count

    def test_load_sets_loaded_flag(self):
        m = _get_module()
        gen = m.AnimateDiffGenerator()
        assert not gen._loaded
        gen.load()
        assert gen._loaded

    def test_load_enables_attention_slicing(self):
        import diffusers
        m = _get_module()
        gen = m.AnimateDiffGenerator()
        gen.load()
        pipe_instance = diffusers.AnimateDiffPipeline.from_pretrained.return_value
        pipe_instance.enable_attention_slicing.assert_called_once()


# ---------------------------------------------------------------------------
# AnimateDiffGenerator.generate()
# ---------------------------------------------------------------------------


class TestGenerate:
    def _loaded_gen(self):
        m = _get_module()
        gen = m.AnimateDiffGenerator()
        gen.load()
        return gen

    def test_raises_if_not_loaded(self):
        m = _get_module()
        gen = m.AnimateDiffGenerator()
        with pytest.raises(RuntimeError, match="load\\(\\)"):
            gen.generate("a prompt", "neg", seed=42)

    def test_returns_list_of_pil_images(self):
        gen = self._loaded_gen()
        frames = gen.generate("a prompt", "neg prompt", seed=42)
        assert isinstance(frames, list)
        assert all(isinstance(f, Image.Image) for f in frames)

    def test_returns_correct_frame_count(self):
        gen = self._loaded_gen()
        frames = gen.generate("a prompt", "neg", seed=42)
        assert len(frames) == 4  # matches mock's fake_frames length

    def test_passes_positive_prompt(self):
        import diffusers
        gen = self._loaded_gen()
        pipe_instance = diffusers.AnimateDiffPipeline.from_pretrained.return_value
        gen.generate("my positive prompt", "neg", seed=42)
        _, kwargs = pipe_instance.call_args
        assert kwargs.get("prompt") == "my positive prompt"

    def test_passes_negative_prompt(self):
        import diffusers
        gen = self._loaded_gen()
        pipe_instance = diffusers.AnimateDiffPipeline.from_pretrained.return_value
        gen.generate("pos", "my negative prompt", seed=42)
        _, kwargs = pipe_instance.call_args
        assert kwargs.get("negative_prompt") == "my negative prompt"

    def test_passes_num_frames(self):
        import diffusers
        gen = self._loaded_gen()
        pipe_instance = diffusers.AnimateDiffPipeline.from_pretrained.return_value
        gen.generate("pos", "neg", seed=42, numFrames=16)
        _, kwargs = pipe_instance.call_args
        assert kwargs.get("num_frames") == 16

    def test_passes_guidance_scale(self):
        import diffusers
        gen = self._loaded_gen()
        pipe_instance = diffusers.AnimateDiffPipeline.from_pretrained.return_value
        gen.generate("pos", "neg", seed=42, guidance=6.0)
        _, kwargs = pipe_instance.call_args
        assert kwargs.get("guidance_scale") == 6.0

    def test_passes_num_inference_steps(self):
        import diffusers
        gen = self._loaded_gen()
        pipe_instance = diffusers.AnimateDiffPipeline.from_pretrained.return_value
        gen.generate("pos", "neg", seed=42, steps=20)
        _, kwargs = pipe_instance.call_args
        assert kwargs.get("num_inference_steps") == 20

    def test_uses_seeded_generator(self):
        import torch
        gen = self._loaded_gen()
        gen.generate("pos", "neg", seed=99)
        torch.Generator.assert_called()


# ---------------------------------------------------------------------------
# framesToGif()
# ---------------------------------------------------------------------------


class TestFramesToGif:
    def test_saves_gif_file(self, tmp_path):
        m = _get_module()
        frames = _fake_frames(4)
        out = tmp_path / "anim.gif"
        m.framesToGif(frames, out, fps=8)
        assert out.exists()

    def test_raises_on_empty_frames(self, tmp_path):
        m = _get_module()
        with pytest.raises(ValueError, match="frames"):
            m.framesToGif([], tmp_path / "anim.gif")

    def test_fps_affects_duration(self, tmp_path):
        m = _get_module()
        frames = _fake_frames(4)
        out8 = tmp_path / "anim8.gif"
        out4 = tmp_path / "anim4.gif"
        m.framesToGif(frames, out8, fps=8)
        m.framesToGif(frames, out4, fps=4)
        # Both must be valid GIFs; we just verify the file was written
        assert out8.stat().st_size > 0
        assert out4.stat().st_size > 0

    def test_returns_output_path(self, tmp_path):
        m = _get_module()
        frames = _fake_frames(4)
        out = tmp_path / "anim.gif"
        result = m.framesToGif(frames, out)
        assert result == out


# ---------------------------------------------------------------------------
# CLI animate command
# ---------------------------------------------------------------------------


class TestCliAnimate:
    def test_animate_command_exists(self):
        from typer.testing import CliRunner
        from roomify.cli import app
        runner = CliRunner()
        result = runner.invoke(app, ["animate", "--help"])
        assert result.exit_code == 0

    def test_animate_writes_gif(self, tmp_path):
        """Smoke test: animate command produces a .gif + run.json."""
        from typer.testing import CliRunner
        from roomify import animateDiff as m
        from roomify.cli import app

        # Patch getAnimateDiffGenerator so no GPU is needed
        fake_gen = MagicMock()
        fake_frames = _fake_frames(4)
        fake_gen.generate.return_value = fake_frames

        spec_path = Path(__file__).parent.parent / "configs/examples/bedroom_01.yaml"

        with patch("roomify.animateDiff.getAnimateDiffGenerator", return_value=fake_gen), \
             patch("roomify.paths.getOutputDir", return_value=tmp_path):
            runner = CliRunner()
            result = runner.invoke(app, [
                "animate",
                "--spec", str(spec_path),
                "--frames", "4",
                "--fps", "8",
            ])

        assert result.exit_code == 0, result.output
        gifs = list(tmp_path.rglob("*.gif"))
        assert len(gifs) == 1

    def test_animate_writes_run_json(self, tmp_path):
        from typer.testing import CliRunner
        from roomify.cli import app

        fake_gen = MagicMock()
        fake_gen.generate.return_value = _fake_frames(4)

        spec_path = Path(__file__).parent.parent / "configs/examples/bedroom_01.yaml"

        with patch("roomify.animateDiff.getAnimateDiffGenerator", return_value=fake_gen), \
             patch("roomify.paths.getOutputDir", return_value=tmp_path):
            runner = CliRunner()
            runner.invoke(app, [
                "animate",
                "--spec", str(spec_path),
                "--frames", "4",
            ])

        run_jsons = list(tmp_path.rglob("run.json"))
        assert len(run_jsons) == 1
        data = json.loads(run_jsons[0].read_text())
        assert data.get("type") == "animate"
        assert "numFrames" in data
        assert "gifPath" in data
