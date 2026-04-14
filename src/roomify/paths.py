"""Path resolution helper — returns correct directories for Colab, Colab+Drive, and local dev.

Never hardcode /content/drive/... or /Users/... elsewhere; always use these helpers.
"""

from __future__ import annotations

from pathlib import Path

# Colab Drive mount point
_DRIVE_ROOT = Path("/content/drive/MyDrive/roomify")
# Colab without Drive
_COLAB_ROOT = Path("/content")
# Repo root (works locally and in Colab after git clone)
_REPO_ROOT = Path(__file__).parent.parent.parent


def _isDriveMounted() -> bool:
    return _DRIVE_ROOT.exists()


def _isColab() -> bool:
    return _COLAB_ROOT.exists()


def getOutputDir() -> Path:
    """Return the outputs directory, creating it if necessary."""
    if _isDriveMounted():
        path = _DRIVE_ROOT / "outputs"
    elif _isColab():
        path = _COLAB_ROOT / "outputs"
    else:
        path = _REPO_ROOT / "outputs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def getDataDir() -> Path:
    """Return the data directory."""
    if _isDriveMounted():
        return _DRIVE_ROOT / "data"
    if _isColab():
        return _COLAB_ROOT / "data"
    return _REPO_ROOT / "data"
