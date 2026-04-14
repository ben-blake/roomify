"""Roomify — Streamlit entrypoint.

This file is a thin shell.  All business logic lives in src/roomify/.

Launch:
    streamlit run app.py --server.port 8501 --server.headless true
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is on the path when running with `streamlit run app.py`
_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import streamlit as st

from roomify.ui import pageGenerate, pageExperiments, pageGallery

st.set_page_config(
    page_title="Roomify",
    page_icon=":house:",
    layout="wide",
)

PAGES = {
    "Generate": pageGenerate,
    "Experiments": pageExperiments,
    "Gallery": pageGallery,
}

page = st.sidebar.radio("Navigation", list(PAGES.keys()))
PAGES[page].render()
