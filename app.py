"""Roomify — Streamlit entrypoint.

This file is a thin shell.  All business logic lives in src/roomify/.

Launch:
    streamlit run app.py --server.port 8501 --server.headless true
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure src/ is on the path when running with `streamlit run app.py`
_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Point HF_HOME at Drive-backed cache when available (same as Cell 3 in launcher).
_DRIVE_HF_CACHE = Path("/content/drive/MyDrive/roomify/hf_cache")
if _DRIVE_HF_CACHE.exists() and "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = str(_DRIVE_HF_CACHE)
    os.environ["TRANSFORMERS_CACHE"] = str(_DRIVE_HF_CACHE / "transformers")

import streamlit as st

from roomify.ui import pageExperiments, pageGallery, pageGenerate

st.set_page_config(
    page_title="Roomify",
    page_icon="R",
    layout="wide",
)

# ── Pre-warm pipeline on first load ────────────────────────────────────────
# Runs once per Streamlit process.  A tiny 64×64 generation primes the model
# so the first user click is fast.

@st.cache_resource(show_spinner=False)
def _prewarm():
    """Load the SD pipeline and run a single 64x64 inference to prime CUDA."""
    try:
        from roomify.pipeline import getPipeline
        pipeline = getPipeline()
        pipeline.load(controlType=None)
        pipeline.generate(
            "a room",
            "blurry",
            seed=0,
            steps=1,
            guidance=1.0,
        )
    except Exception:
        # Non-fatal: pre-warm fails if there's no GPU (e.g., local dev).
        pass
    return True


with st.spinner("Pre-warming pipeline... (first load only)"):
    _prewarm()

# ── Page routing ────────────────────────────────────────────────────────────
PAGES = {
    "Generate": pageGenerate,
    "Experiments": pageExperiments,
    "Gallery": pageGallery,
}

page = st.sidebar.radio("Navigation", list(PAGES.keys()))
PAGES[page].render()
