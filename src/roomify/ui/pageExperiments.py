"""Experiments page — batch sweep runner + metrics visualizations.

Architecture notes:
  - All st.* calls are inside render() so streamlit is imported lazily.
  - runExperiment is called in a background thread so the progress bar
    updates live via a callback that writes to a shared list.
"""

from __future__ import annotations


def render() -> None:
    """Render the Experiments page."""
    import threading
    from pathlib import Path

    import streamlit as st

    from roomify.ui.components import buildMetricsDf, listGalleryRuns, metricsTable

    st.title("Experiment Sweep")
    st.markdown(
        "Pick an experiment YAML, click **Run sweep**, and watch progress live.  "
        "Results appear below when the sweep finishes."
    )

    # ── Config file picker ──────────────────────────────────────────────────
    config_path_str = st.text_input(
        "Experiment config path",
        value="configs/experiments/core.yaml",
        help="Path relative to the repo root, or an absolute path.",
    )
    config_path = Path(config_path_str)

    if not config_path.exists():
        st.warning(f"Config file not found: {config_path}")
        return

    run_btn = st.button("Run sweep", type="primary")

    if run_btn:
        from roomify.orchestrator import runExperiment

        progress_bar = st.progress(0, text="Starting sweep...")
        status_text = st.empty()

        done_list: list = []
        total_list: list = []
        error_holder: list = []
        result_holder: list = []

        def _progressCb(done: int, total: int) -> None:
            done_list.clear()
            done_list.append(done)
            total_list.clear()
            total_list.append(total)

        def _runInThread():
            try:
                out_dir = runExperiment(config_path, progressCb=_progressCb)
                result_holder.append(out_dir)
            except Exception as exc:
                error_holder.append(str(exc))

        thread = threading.Thread(target=_runInThread, daemon=True)
        thread.start()

        import time
        while thread.is_alive():
            time.sleep(0.5)
            if total_list and done_list:
                frac = done_list[0] / total_list[0]
                progress_bar.progress(frac, text=f"{done_list[0]}/{total_list[0]} images generated")
                status_text.text(f"Running... [{done_list[0]}/{total_list[0]}]")

        thread.join()

        if error_holder:
            st.error(f"Sweep failed: {error_holder[0]}")
            return

        progress_bar.progress(1.0, text="Sweep complete!")
        out_dir = result_holder[0]
        st.success(f"Sweep complete. Results saved to: {out_dir}")

        # ── Results section ─────────────────────────────────────────────────
        _renderResults(out_dir)

    # ── Show results for last completed sweep (from output dir picker) ──────
    st.divider()
    st.subheader("Browse past sweep results")

    from roomify.paths import getOutputDir
    output_dir = getOutputDir()
    sweep_dirs = sorted(
        [d for d in output_dir.iterdir() if d.is_dir()],
        reverse=True,
    )

    if not sweep_dirs:
        st.info("No completed sweeps found in the output directory.")
        return

    selected = st.selectbox(
        "Select a sweep run",
        options=[d.name for d in sweep_dirs],
    )
    if selected:
        _renderResults(output_dir / selected)


def _renderResults(sweep_dir: "Path") -> None:  # noqa: F821
    """Render metrics table + contact sheet for a completed sweep directory."""
    import streamlit as st
    from pathlib import Path

    from roomify.ui.components import buildMetricsDf, listGalleryRuns, metricsTable

    runs = listGalleryRuns(sweep_dir)
    if not runs:
        st.info("No run.json files found in this sweep directory.")
        return

    st.subheader("Metrics table")
    df = buildMetricsDf(runs)
    metricsTable(df)

    st.subheader("Contact sheet")
    cols = st.columns(4)
    col_cycle = cols * (len(runs) // len(cols) + 1)
    for col, run in zip(col_cycle, runs):
        with col:
            img_path = Path(run.get("imagePath", ""))
            if img_path.exists():
                caption = f"{run.get('strategy', '')} | s={run.get('seed', '')}"
                st.image(str(img_path), caption=caption, use_column_width=True)
            else:
                st.caption(f"Image missing: {img_path.name}")
