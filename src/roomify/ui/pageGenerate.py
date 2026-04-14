"""Generate page — spec form + single-image generation.

Architecture notes:
  - All st.* calls are inside render() so streamlit is imported lazily.
  - Pipeline is cached with @st.cache_resource (one load per process).
  - Variant images accumulate in st.session_state["variants"].
"""

from __future__ import annotations


def render() -> None:
    """Render the Generate page."""
    import random
    import time
    from pathlib import Path

    import streamlit as st

    from roomify.paths import getOutputDir
    from roomify.promptBuilder import RoomSpec, buildPrompt
    from roomify.ui.components import (
        controlPreview,
        imageCard,
        specForm,
        formatSpec,
    )

    # ── Cached pipeline loader ──────────────────────────────────────────────
    @st.cache_resource(show_spinner="Loading Stable Diffusion pipeline...")
    def _loadedPipeline(control_type):
        from roomify.pipeline import getPipeline
        pipeline = getPipeline()
        pipeline.load(controlType=control_type)
        return pipeline

    st.title("Generate Room Design")
    st.markdown(
        "Fill in the spec form, choose a strategy, then click **Generate**."
    )

    # ── Left column: form + settings; Right column: output ─────────────────
    left, right = st.columns([1, 1])

    with left:
        st.subheader("Room Spec")
        spec_dict = specForm()

        st.subheader("Generation Settings")
        strategy = st.selectbox(
            "Prompt strategy",
            ["minimal", "descriptive", "styleAnchored"],
            index=1,
        )

        controlled = st.toggle("Use ControlNet conditioning")
        control_type = None
        ref_image_id = None

        if controlled:
            control_type = st.radio("Control signal", ["depth", "canny"], horizontal=True)
            ref_image_id = st.text_input(
                "Reference image ID (SUN RGB-D record)",
                placeholder="e.g. sunrgbd_00142",
            )

        seed_mode = st.radio("Seed", ["Random", "Fixed"], horizontal=True)
        seed = (
            st.number_input("Seed value", value=42, step=1)
            if seed_mode == "Fixed"
            else random.randint(0, 2**31 - 1)
        )

        with st.expander("Advanced"):
            steps = st.slider("Diffusion steps", 10, 50, 30)
            guidance = st.slider("Guidance scale", 1.0, 20.0, 7.5, step=0.5)

        generate_btn = st.button("Generate", type="primary", use_container_width=True)
        variant_btn = st.button("Generate variant", use_container_width=True)

    with right:
        st.subheader("Output")

        # Initialise session state for variants
        if "variants" not in st.session_state:
            st.session_state["variants"] = []

        def _doGenerate(spec_d, strat, ctrl_type, ref_id, sd, n_steps, n_guidance):
            """Run generation and append result to session_state variants."""
            import dataclasses
            import json
            import subprocess
            from datetime import datetime, timezone

            from PIL import Image as PILImage

            valid_fields = set(RoomSpec.__dataclass_fields__.keys())
            room_spec = RoomSpec(**{k: v for k, v in spec_d.items() if k in valid_fields})
            positive, negative = buildPrompt(room_spec, strat)

            control_image = None
            if ctrl_type and ref_id:
                from roomify.controlSignals import extractCanny, extractDepth
                from roomify.dataset import getRecord
                record = getRecord(ref_id)
                if ctrl_type == "depth":
                    control_image = extractDepth(PILImage.open(record.depthPath))
                else:
                    control_image = extractCanny(PILImage.open(record.rgbPath))

            pipeline = _loadedPipeline(ctrl_type)
            t0 = time.monotonic()
            image = pipeline.generate(
                positive, negative,
                seed=sd, steps=n_steps, guidance=n_guidance,
                control=control_image,
            )
            elapsed = round(time.monotonic() - t0, 2)

            ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
            run_id = f"{ts}_{room_spec.id}"
            out_dir = getOutputDir() / run_id
            out_dir.mkdir(parents=True, exist_ok=True)
            img_path = out_dir / "img_0.png"
            image.save(str(img_path))

            try:
                git_sha = subprocess.check_output(
                    ["git", "rev-parse", "--short", "HEAD"],
                    text=True, stderr=subprocess.DEVNULL,
                ).strip()
            except Exception:
                git_sha = "unknown"

            from roomify.pipeline import (
                CONTROLNET_CANNY_ID,
                CONTROLNET_DEPTH_ID,
                SD_MODEL_ID,
            )
            controlnet_id = (
                CONTROLNET_DEPTH_ID if ctrl_type == "depth"
                else CONTROLNET_CANNY_ID if ctrl_type == "canny"
                else None
            )

            run_json = {
                "runId": run_id,
                "spec": dataclasses.asdict(room_spec),
                "strategy": strat,
                "controlled": ctrl_type is not None,
                "controlType": ctrl_type,
                "refImageId": ref_id,
                "model": SD_MODEL_ID,
                "controlnet": controlnet_id,
                "seed": sd,
                "steps": n_steps,
                "guidanceScale": n_guidance,
                "prompt": positive,
                "negativePrompt": negative,
                "imagePath": str(img_path),
                "gitSha": git_sha,
                "timings": {"generateSec": elapsed},
            }
            (out_dir / "run.json").write_text(json.dumps(run_json, indent=2))
            st.session_state["variants"].append(run_json)

        if (generate_btn or variant_btn) and spec_dict:
            variant_seed = seed if generate_btn else random.randint(0, 2**31 - 1)
            with st.spinner(f"Generating (seed={variant_seed})..."):
                try:
                    _doGenerate(
                        spec_dict, strategy, control_type, ref_image_id,
                        variant_seed, steps, guidance,
                    )
                except Exception as exc:
                    st.error(f"Generation failed: {exc}")

        elif (generate_btn or variant_btn) and not spec_dict:
            st.info("Fill in the spec form on the left and click **Apply spec** first.")

        if st.session_state.get("variants"):
            if st.button("Clear variants"):
                st.session_state["variants"] = []
                st.rerun()

            cols = st.columns(min(len(st.session_state["variants"]), 3))
            for col, run_json in zip(
                cols * (len(st.session_state["variants"]) // len(cols) + 1),
                st.session_state["variants"],
            ):
                with col:
                    imageCard(run_json)

            if controlled and ref_image_id:
                st.subheader("Control signal preview")
                try:
                    from PIL import Image as PILImage
                    from roomify.controlSignals import extractCanny, extractDepth
                    from roomify.dataset import getRecord
                    record = getRecord(ref_image_id)
                    if control_type == "depth":
                        ctrl_img = extractDepth(PILImage.open(record.depthPath))
                    else:
                        ctrl_img = extractCanny(PILImage.open(record.rgbPath))
                    import tempfile, os
                    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                    ctrl_img.save(tmp.name)
                    controlPreview(Path(tmp.name))
                    os.unlink(tmp.name)
                except Exception:
                    pass
