"""Tests for src/roomify/kenBurns.py — Ken Burns pan/zoom effect."""

from __future__ import annotations

import pytest
from PIL import Image


def _img(w: int = 512, h: int = 512) -> Image.Image:
    """Gradient image so pan/zoom crops produce visibly different pixels."""
    img = Image.new("RGB", (w, h))
    pixels = [(int(x * 255 / w), int(y * 255 / h), 100) for y in range(h) for x in range(w)]
    img.putdata(pixels)
    return img


# ---------------------------------------------------------------------------
# applyKenBurns — return type and dimensions
# ---------------------------------------------------------------------------

class TestReturnType:
    def test_returns_list(self):
        from roomify.kenBurns import applyKenBurns
        result = applyKenBurns(_img(), frames=8)
        assert isinstance(result, list)

    def test_returns_pil_images(self):
        from roomify.kenBurns import applyKenBurns
        result = applyKenBurns(_img(), frames=8)
        assert all(isinstance(f, Image.Image) for f in result)

    def test_frame_count_matches(self):
        from roomify.kenBurns import applyKenBurns
        for n in (4, 8, 16, 24):
            assert len(applyKenBurns(_img(), frames=n)) == n

    def test_frames_preserve_input_size(self):
        from roomify.kenBurns import applyKenBurns
        img = _img(480, 320)
        for frame in applyKenBurns(img, frames=8):
            assert frame.size == (480, 320)

    def test_non_square_image(self):
        from roomify.kenBurns import applyKenBurns
        img = _img(768, 512)
        frames = applyKenBurns(img, frames=8)
        assert all(f.size == (768, 512) for f in frames)


# ---------------------------------------------------------------------------
# Motion types
# ---------------------------------------------------------------------------

class TestMotionTypes:
    def _first_last_differ(self, motion: str) -> bool:
        from roomify.kenBurns import applyKenBurns
        img = _img()
        frames = applyKenBurns(img, frames=8, motion=motion, intensity=0.25)
        return list(frames[0].getdata()) != list(frames[-1].getdata())

    def test_zoom_in_changes_frames(self):
        assert self._first_last_differ("zoom_in")

    def test_zoom_out_changes_frames(self):
        assert self._first_last_differ("zoom_out")

    def test_pan_right_changes_frames(self):
        assert self._first_last_differ("pan_right")

    def test_pan_left_changes_frames(self):
        assert self._first_last_differ("pan_left")

    def test_pan_up_changes_frames(self):
        assert self._first_last_differ("pan_up")

    def test_pan_down_changes_frames(self):
        assert self._first_last_differ("pan_down")

    def test_unknown_motion_raises(self):
        from roomify.kenBurns import applyKenBurns
        with pytest.raises(ValueError, match="motion"):
            applyKenBurns(_img(), motion="spin")

    def test_zero_intensity_produces_identical_frames(self):
        from roomify.kenBurns import applyKenBurns
        img = _img()
        frames = applyKenBurns(img, frames=8, motion="zoom_in", intensity=0.0)
        first = list(frames[0].getdata())
        assert all(list(f.getdata()) == first for f in frames[1:])


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_raises_if_frames_less_than_2(self):
        from roomify.kenBurns import applyKenBurns
        with pytest.raises(ValueError, match="frames"):
            applyKenBurns(_img(), frames=1)

    def test_exactly_2_frames(self):
        from roomify.kenBurns import applyKenBurns
        result = applyKenBurns(_img(), frames=2)
        assert len(result) == 2

    def test_intensity_clipped_at_0_5(self):
        from roomify.kenBurns import applyKenBurns
        # Should not raise even with extreme intensity
        result = applyKenBurns(_img(), frames=8, intensity=0.9)
        assert len(result) == 8
        assert all(f.size == (512, 512) for f in result)


# ---------------------------------------------------------------------------
# CLI kenburns command
# ---------------------------------------------------------------------------

class TestCliKenBurns:
    def test_command_exists(self):
        from typer.testing import CliRunner
        from roomify.cli import app
        runner = CliRunner()
        result = runner.invoke(app, ["kenburns", "--help"])
        assert result.exit_code == 0

    def test_writes_gif(self, tmp_path):
        from typer.testing import CliRunner
        from roomify.cli import app

        img = _img(64, 64)
        img_path = tmp_path / "test.png"
        img.save(str(img_path))
        out_path = tmp_path / "out.gif"

        runner = CliRunner()
        result = runner.invoke(app, [
            "kenburns",
            "--image", str(img_path),
            "--output", str(out_path),
            "--frames", "8",
            "--motion", "zoom_in",
        ])
        assert result.exit_code == 0, result.output
        assert out_path.exists()
        assert out_path.stat().st_size > 0
