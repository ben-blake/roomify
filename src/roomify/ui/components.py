"""Reusable Streamlit UI components and pure-Python data helpers.

Pure-Python helpers (no Streamlit import):
  - parseRunJson(runDir)       load run.json from a run directory
  - listGalleryRuns(...)       scan + filter output runs
  - buildMetricsDf(runs)       build a pandas DataFrame from run dicts
  - formatSpec(specDict)       format a spec dict as a readable string

Streamlit components (import st lazily inside each function):
  - specForm()                 room-spec input form
  - imageCard(runJson)         image + metadata card
  - metricsTable(df)           styled metrics table
  - controlPreview(imagePath)  depth / canny map preview
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Pure-Python helpers (testable without a Streamlit runtime)
# ---------------------------------------------------------------------------


def parseRunJson(runDir: Path) -> Dict[str, Any]:
    """Load and return run.json from *runDir*.

    Raises FileNotFoundError if run.json is absent.
    """
    run_json_path = Path(runDir) / "run.json"
    if not run_json_path.exists():
        raise FileNotFoundError(f"run.json not found in {runDir}")
    return json.loads(run_json_path.read_text())


def listGalleryRuns(
    outputDir: Path,
    sceneType: Optional[str] = None,
    strategy: Optional[str] = None,
    controlled: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    """Scan *outputDir* recursively for run.json files and return filtered results.

    Handles both CLI output (one level deep) and sweep output (two levels deep):
      outputDir/<runId>/run.json
      outputDir/<sweepId>/<cellId>/run.json
    """
    output_path = Path(outputDir)
    runs: List[Dict[str, Any]] = []
    for json_path in sorted(output_path.rglob("run.json")):
        data = json.loads(json_path.read_text())
        runs.append(data)

    if sceneType is not None:
        runs = [r for r in runs if r.get("spec", {}).get("roomType") == sceneType]
    if strategy is not None:
        runs = [r for r in runs if r.get("strategy") == strategy]
    if controlled is not None:
        runs = [r for r in runs if r.get("controlled") == controlled]

    return runs


def buildMetricsDf(runs: List[Dict[str, Any]]) -> "Any":
    """Build a pandas DataFrame summary from a list of run.json dicts."""
    import pandas as pd

    if not runs:
        return pd.DataFrame(columns=["runId", "sceneType", "strategy", "controlled", "seed"])

    rows = [
        {
            "runId": r.get("runId", ""),
            "sceneType": r.get("spec", {}).get("roomType", ""),
            "strategy": r.get("strategy", ""),
            "controlled": r.get("controlled", False),
            "seed": r.get("seed", 0),
            "steps": r.get("steps", 0),
            "guidanceScale": r.get("guidanceScale", 0.0),
            "generateSec": r.get("timings", {}).get("generateSec", 0.0),
        }
        for r in runs
    ]
    return pd.DataFrame(rows)


def formatSpec(specDict: Dict[str, Any]) -> str:
    """Return a compact human-readable summary of a spec dict."""
    parts: List[str] = []
    if specDict.get("roomType"):
        parts.append(f"Room: {specDict['roomType']}")
    if specDict.get("size"):
        parts.append(f"Size: {specDict['size']}")
    if specDict.get("style"):
        parts.append(f"Style: {specDict['style']}")
    if specDict.get("furniture"):
        parts.append(f"Furniture: {', '.join(specDict['furniture'])}")
    if specDict.get("lighting"):
        parts.append(f"Lighting: {specDict['lighting']}")
    if specDict.get("mood"):
        parts.append(f"Mood: {specDict['mood']}")
    return " | ".join(parts) if parts else "(empty spec)"


# ---------------------------------------------------------------------------
# Streamlit components (streamlit imported lazily — not available locally)
# ---------------------------------------------------------------------------

_ROOM_TYPES = ["bedroom", "living_room", "kitchen", "office", "bathroom"]
_STYLES = [
    "scandinavian", "minimalist", "industrial", "mid_century",
    "bohemian", "contemporary", "traditional",
]
_STRATEGIES = ["minimal", "descriptive", "styleAnchored"]


def specForm() -> Optional[Dict[str, Any]]:
    """Render the room-spec input form and return the spec dict on submit.

    Returns None if the form has not yet been submitted.
    """
    import streamlit as st

    with st.form("spec_form"):
        col1, col2 = st.columns(2)
        with col1:
            room_type = st.selectbox("Room type", _ROOM_TYPES)
            style = st.selectbox("Style", _STYLES)
            size = st.text_input("Size (e.g. 10x12 ft)", value="10x12 ft")
        with col2:
            furniture = st.multiselect(
                "Furniture",
                ["bed", "sofa", "desk", "chair", "table", "dresser",
                 "nightstand", "bookshelf", "coffee table", "island"],
            )
            lighting = st.text_input("Lighting", value="natural light")
            mood = st.text_input("Mood / atmosphere", value="cozy")

        submitted = st.form_submit_button("Apply spec")

    if not submitted:
        return None

    import time
    return {
        "id": f"custom_{int(time.time())}",
        "roomType": room_type,
        "size": size,
        "style": style,
        "furniture": furniture,
        "lighting": lighting,
        "mood": mood,
    }


def imageCard(runJson: Dict[str, Any]) -> None:
    """Render a single image card with its run.json metadata."""
    import streamlit as st

    img_path = Path(runJson.get("imagePath", ""))
    if img_path.exists():
        st.image(str(img_path), use_column_width=True)
    else:
        st.warning(f"Image not found: {img_path}")

    caption = (
        f"{runJson.get('strategy', '')} | "
        f"seed={runJson.get('seed', '')} | "
        f"controlled={runJson.get('controlled', False)}"
    )
    st.caption(caption)

    with st.expander("Run metadata"):
        spec = runJson.get("spec", {})
        st.write(f"**Spec:** {formatSpec(spec)}")
        st.write(f"**Prompt:** {runJson.get('prompt', '')}")
        st.write(f"**Steps:** {runJson.get('steps', '')} | "
                 f"**Guidance:** {runJson.get('guidanceScale', '')}")
        st.write(f"**Model:** {runJson.get('model', '')}")
        if runJson.get("controlnet"):
            st.write(f"**ControlNet:** {runJson['controlnet']}")
        st.write(f"**Git SHA:** {runJson.get('gitSha', '')}")
        timing = runJson.get("timings", {}).get("generateSec", "?")
        st.write(f"**Generate time:** {timing}s")


def metricsTable(df: Any) -> None:
    """Render a metrics DataFrame as a styled Streamlit table."""
    import streamlit as st

    if df is None or len(df) == 0:
        st.info("No runs to display.")
        return
    st.dataframe(df, use_container_width=True)


def controlPreview(imagePath: Path) -> None:
    """Render a control signal preview (depth map or Canny edge image)."""
    import streamlit as st

    img_path = Path(imagePath)
    if not img_path.exists():
        st.warning(f"Control image not found: {img_path}")
        return
    st.image(str(img_path), caption="Control signal", use_column_width=True)
