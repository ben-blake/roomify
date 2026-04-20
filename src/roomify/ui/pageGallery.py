"""Gallery page — browse past outputs with filters.

Architecture notes:
  - All st.* calls are inside render() so streamlit is imported lazily.
  - Manifest is cached with @st.cache_data.
"""

from __future__ import annotations


def render() -> None:
    """Render the Gallery page."""
    import streamlit as st

    from roomify.paths import getOutputDir
    from roomify.ui.components import buildMetricsDf, imageCard, listGalleryRuns, metricsTable

    st.title("Gallery")
    st.markdown("Browse all generated images.  Use the filters to narrow results.")

    output_dir = getOutputDir()

    # ── Sidebar-style filter row ────────────────────────────────────────────
    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        scene_options = ["All", "bedroom", "living_room", "kitchen", "office", "bathroom"]
        scene_filter = st.selectbox("Scene type", scene_options)
    with f_col2:
        strategy_options = ["All", "minimal", "descriptive", "styleAnchored"]
        strategy_filter = st.selectbox("Strategy", strategy_options)
    with f_col3:
        controlled_options = {"All": None, "Controlled": True, "Uncontrolled": False}
        controlled_label = st.selectbox("ControlNet", list(controlled_options.keys()))

    scene_arg = None if scene_filter == "All" else scene_filter
    strategy_arg = None if strategy_filter == "All" else strategy_filter
    controlled_arg = controlled_options[controlled_label]

    runs = listGalleryRuns(
        output_dir,
        sceneType=scene_arg,
        strategy=strategy_arg,
        controlled=controlled_arg,
    )

    if not runs:
        st.info("No images found matching the current filters.")
        return

    st.caption(f"{len(runs)} image(s) found")

    # ── Metrics summary ─────────────────────────────────────────────────────
    with st.expander("Metrics summary"):
        df = buildMetricsDf(runs)
        metricsTable(df)

    # ── Image grid (3 columns) ───────────────────────────────────────────────
    from roomify.evaluation import loadRatings, saveRating
    ratings_df = loadRatings(output_dir)
    rated_map = {} if ratings_df.empty else dict(
        zip(ratings_df["runId"], ratings_df["rating"].astype(int))
    )

    n_cols = 3
    cols = st.columns(n_cols)
    for idx, run in enumerate(runs):
        with cols[idx % n_cols]:
            imageCard(run)
            run_id = run.get("runId", "")
            current = int(rated_map.get(run_id, 0))
            new_val = st.select_slider(
                "Rating",
                options=[0, 1, 2, 3, 4, 5],
                value=current,
                format_func=lambda v: "unrated" if v == 0 else f"{'★' * v}{'☆' * (5 - v)}",
                key=f"rating_{run_id}",
            )
            if new_val != current and new_val > 0:
                saveRating(output_dir, run_id, new_val)
                st.rerun()
