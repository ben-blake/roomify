"""Phase 0 smoke tests — verify module scaffold and CLI entry point.

These tests require NO GPU and NO Streamlit runtime.
They run anywhere: local dev machine, Colab, or CI.
"""

import importlib
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Module import smoke tests
# ---------------------------------------------------------------------------

EXPECTED_MODULES = [
    "roomify",
    "roomify.dataset",
    "roomify.promptBuilder",
    "roomify.controlSignals",
    "roomify.pipeline",
    "roomify.orchestrator",
    "roomify.evaluation",
    "roomify.reporting",
    "roomify.cli",
    "roomify.ui",
    "roomify.ui.components",
    "roomify.ui.pageGenerate",
    "roomify.ui.pageExperiments",
    "roomify.ui.pageGallery",
]


def test_all_modules_importable():
    """Every service-layer module must be importable without errors."""
    src_path = str(REPO_ROOT / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    failed = []
    for name in EXPECTED_MODULES:
        try:
            importlib.import_module(name)
        except ImportError as exc:
            failed.append(f"{name}: {exc}")

    assert not failed, "Import failures:\n" + "\n".join(failed)


# ---------------------------------------------------------------------------
# File-system structure tests
# ---------------------------------------------------------------------------

EXPECTED_FILES = [
    "app.py",
    "requirements.txt",
    ".gitignore",
    "README.md",
    "CLAUDE.md",
    "configs/prompts.yaml",
    "configs/examples/bedroom_01.yaml",
    "src/roomify/__init__.py",
    "src/roomify/dataset.py",
    "src/roomify/promptBuilder.py",
    "src/roomify/controlSignals.py",
    "src/roomify/pipeline.py",
    "src/roomify/orchestrator.py",
    "src/roomify/evaluation.py",
    "src/roomify/reporting.py",
    "src/roomify/cli.py",
    "src/roomify/ui/__init__.py",
    "src/roomify/ui/components.py",
    "src/roomify/ui/pageGenerate.py",
    "src/roomify/ui/pageExperiments.py",
    "src/roomify/ui/pageGallery.py",
    "notebooks/00_launchColab.ipynb",
    "docs/AI_TOOLS.md",
    "docs/PRD.md",
    "docs/ARCHITECTURE.md",
    "docs/TASKS.md",
    "tests/testPhase0.py",
]


def test_required_files_exist():
    """All scaffolded files must exist at their expected paths."""
    missing = [f for f in EXPECTED_FILES if not (REPO_ROOT / f).exists()]
    assert not missing, "Missing files:\n" + "\n".join(missing)


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

def test_cli_help_runs():
    """python -m roomify.cli --help must exit 0 with usage text."""
    result = subprocess.run(
        [sys.executable, "-m", "roomify.cli", "--help"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT / "src"),
    )
    assert result.returncode == 0, (
        f"CLI --help exited {result.returncode}\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "roomify" in result.stdout.lower() or "usage" in result.stdout.lower(), (
        f"Expected usage text, got: {result.stdout}"
    )


# ---------------------------------------------------------------------------
# requirements.txt sanity check
# ---------------------------------------------------------------------------

# Packages that must appear in requirements.txt.
# Colab 2026.01 pre-installs torch, torchvision, numpy, pandas, matplotlib,
# transformers 5.0, accelerate, diffusers 0.37.1, opencv-python-headless,
# Pillow, and huggingface-hub — so those are documented in comments, not
# re-installed. Only packages absent from Colab belong in this list.
REQUIRED_PACKAGES = [
    "controlnet-aux",
    "open_clip_torch",
    "lpips",
    "typer",
    "streamlit",
    "pyngrok",
    "pytest",
    "PyYAML",
]


def test_requirements_txt_covers_required_packages():
    """requirements.txt must list all required packages (case-insensitive)."""
    req_file = REPO_ROOT / "requirements.txt"
    assert req_file.exists(), "requirements.txt is missing"
    content = req_file.read_text().lower()
    missing = [pkg for pkg in REQUIRED_PACKAGES if pkg.lower() not in content]
    assert not missing, "Missing from requirements.txt:\n" + "\n".join(missing)
