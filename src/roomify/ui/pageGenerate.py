"""Generate page — spec form + single-image generation + AnimateDiff GIF.

Architecture notes:
  - All st.* calls are inside render() so streamlit is imported lazily.
  - SD pipeline is cached with @st.cache_resource (one load per process).
  - AnimateDiff generator is cached separately with @st.cache_resource.
  - Variant images accumulate in st.session_state["variants"].
  - Animate results accumulate in st.session_state["anim_results"].
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
        formatSpec,
        imageCard,
        specForm,
    )

    # ── Cached pipeline loaders ─────────────────────────────────────────────
    @st.cache_resource(show_spinner="Loading Stable Diffusion pipeline...")
    def _loadedPipeline(control_type):
        from roomify.pipeline import getPipeline
        pipeline = getPipeline()
        pipeline.load(controlType=control_type)
        return pipeline

    @st.cache_resource(show_spinner="Loading AnimateDiff pipeline...")
    def _loadedAnimator():
        from roomify.animateDiff import getAnimateDiffGenerator
        gen = getAnimateDiffGenerator()
        gen.load()
        return gen

    st.title("Generate Room Design")

    # ── Shared spec form (lives above the tabs so spec persists across them) ─
    st.subheader("Room Spec")
    submitted = specForm()
    if submitted is not None:
        st.session_state["spec_dict"] = submitted
    spec_dict = st.session_state.get("spec_dict")
    if spec_dict:
        st.success(f"Spec loaded: {spec_dict.get('roomType')} / {spec_dict.get('style')}")
    else:
        st.info("Fill in the spec form above and click **Apply spec** before generating.")

    st.divider()

    # ── Tabs ────────────────────────────────────────────────────────────────
    tab_gen, tab_anim = st.tabs(["Generate Image", "Animate (GIF)"])

    # ════════════════════════════════════════════════════════════════════════
    # Tab 1 — Single image generation
    # ════════════════════════════════════════════════════════════════════════
    with tab_gen:
        left, right = st.columns([1, 1])

        with left:
            st.subheader("Generation Settings")
            strategy = st.selectbox(
                "Prompt strategy",
                ["minimal", "descriptive", "styleAnchored"],
                index=1,
                key="gen_strategy",
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

            seed_mode = st.radio("Seed", ["Random", "Fixed"], horizontal=True, key="gen_seed_mode")
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

            if "variants" not in st.session_state:
                st.session_state["variants"] = []

            def _doGenerate(spec_d, strat, ctrl_type, ref_id, sd, n_steps, n_guidance):
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
                    from roomify.dataset import getRecord, loadManifest
                    from roomify.paths import getDataDir
                    _manifest = loadManifest(getDataDir() / "sunrgbd_subset" / "manifest.csv")
                    record = getRecord(_manifest, ref_id)
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
                st.info("Fill in the spec form above and click **Apply spec** first.")

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
                    import os
                    import tempfile

                    from PIL import Image as PILImage

                    from roomify.controlSignals import extractCanny, extractDepth
                    from roomify.dataset import getRecord, loadManifest
                    from roomify.paths import getDataDir
                    _manifest = loadManifest(getDataDir() / "sunrgbd_subset" / "manifest.csv")
                    record = getRecord(_manifest, ref_image_id)
                    if control_type == "depth":
                        ctrl_img = extractDepth(PILImage.open(record.depthPath))
                    else:
                        ctrl_img = extractCanny(PILImage.open(record.rgbPath))
                    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                    ctrl_img.save(tmp.name)
                    controlPreview(Path(tmp.name))
                    os.unlink(tmp.name)
                except Exception:
                    pass

    # ════════════════════════════════════════════════════════════════════════
    # Tab 2 — AnimateDiff GIF generation
    # ════════════════════════════════════════════════════════════════════════
    with tab_anim:
        left_a, right_a = st.columns([1, 1])

        with left_a:
            st.subheader("Animation Settings")

            anim_mode = st.radio(
                "Method",
                ["Ken Burns (instant)", "AnimateDiff (GPU, ~30 s)"],
                horizontal=True,
                key="anim_mode",
            )

            if anim_mode == "Ken Burns (instant)":
                st.caption("Pan or zoom a generated image. Pick an image from the Generate tab first, or paste a path below.")
                kb_image_path = st.text_input(
                    "Source image path",
                    placeholder="/content/drive/MyDrive/roomify/outputs/.../img_0.png",
                    key="kb_image_path",
                )
                kb_motion = st.selectbox(
                    "Motion",
                    ["zoom_in", "zoom_out", "pan_right", "pan_left", "pan_up", "pan_down"],
                    key="kb_motion",
                )
                kb_frames = st.slider("Frames", 12, 48, 24, step=4, key="kb_frames")
                kb_fps = st.slider("FPS", 8, 24, 12, key="kb_fps")
                kb_intensity = st.slider("Intensity", 0.05, 0.4, 0.2, step=0.05, key="kb_intensity")
                animate_btn = st.button("Animate", type="primary", use_container_width=True, key="animate_btn")
            else:
                anim_strategy = st.selectbox(
                    "Prompt strategy",
                    ["minimal", "descriptive", "styleAnchored"],
                    index=1,
                    key="anim_strategy",
                )
                anim_seed_mode = st.radio("Seed", ["Random", "Fixed"], horizontal=True, key="anim_seed_mode")
                anim_seed = (
                    st.number_input("Seed value", value=42, step=1, key="anim_seed_val")
                    if anim_seed_mode == "Fixed"
                    else random.randint(0, 2**31 - 1)
                )
                with st.expander("Advanced"):
                    anim_steps = st.slider("Diffusion steps", 10, 40, 25, key="anim_steps")
                    anim_guidance = st.slider("Guidance scale", 1.0, 15.0, 7.5, step=0.5, key="anim_guidance")
                    anim_frames = st.slider("Frames", 8, 32, 16, step=4, key="anim_frames")
                    anim_fps = st.slider("FPS", 4, 16, 8, key="anim_fps")
                animate_btn = st.button("Animate", type="primary", use_container_width=True, key="animate_btn")

        with right_a:
            st.subheader("Output")

            if "anim_results" not in st.session_state:
                st.session_state["anim_results"] = []

            def _doKenBurns(img_path_str, motion, n_frames, fps, intensity):
                import json
                from datetime import datetime, timezone

                from PIL import Image as PILImage

                from roomify.animateDiff import framesToGif
                from roomify.kenBurns import applyKenBurns

                img = PILImage.open(img_path_str).convert("RGB")
                t0 = time.monotonic()
                frames = applyKenBurns(img, frames=n_frames, motion=motion, intensity=intensity)
                elapsed = round(time.monotonic() - t0, 2)

                ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
                run_id = f"{ts}_kenburns_{motion}"
                out_dir = getOutputDir() / run_id
                out_dir.mkdir(parents=True, exist_ok=True)
                gif_path = out_dir / "anim.gif"
                framesToGif(frames, gif_path, fps=fps)

                run_json = {
                    "runId": run_id,
                    "type": "kenburns",
                    "sourceImage": img_path_str,
                    "motion": motion,
                    "numFrames": n_frames,
                    "fps": fps,
                    "intensity": intensity,
                    "gifPath": str(gif_path),
                    "timings": {"generateSec": elapsed},
                }
                (out_dir / "run.json").write_text(json.dumps(run_json, indent=2))
                st.session_state["anim_results"].append(run_json)

            def _doAnimateDiff(spec_d, strat, sd, n_steps, n_guidance, n_frames, fps):
                import dataclasses
                import json
                import subprocess
                from datetime import datetime, timezone

                from roomify.animateDiff import MOTION_ADAPTER_ID
                from roomify.animateDiff import SD_MODEL_ID as ANIM_SD_MODEL_ID
                from roomify.animateDiff import framesToGif

                valid_fields = set(RoomSpec.__dataclass_fields__.keys())
                room_spec = RoomSpec(**{k: v for k, v in spec_d.items() if k in valid_fields})
                positive, negative = buildPrompt(room_spec, strat)

                gen = _loadedAnimator()
                t0 = time.monotonic()
                frames = gen.generate(
                    positive, negative,
                    seed=sd, steps=n_steps, guidance=n_guidance, numFrames=n_frames,
                )
                elapsed = round(time.monotonic() - t0, 2)

                ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
                run_id = f"{ts}_{room_spec.id}_anim"
                out_dir = getOutputDir() / run_id
                out_dir.mkdir(parents=True, exist_ok=True)
                gif_path = out_dir / "anim.gif"
                framesToGif(frames, gif_path, fps=fps)

                try:
                    git_sha = subprocess.check_output(
                        ["git", "rev-parse", "--short", "HEAD"],
                        text=True, stderr=subprocess.DEVNULL,
                    ).strip()
                except Exception:
                    git_sha = "unknown"

                run_json = {
                    "runId": run_id, "type": "animate",
                    "spec": dataclasses.asdict(room_spec),
                    "strategy": strat, "model": ANIM_SD_MODEL_ID,
                    "motionAdapter": MOTION_ADAPTER_ID,
                    "seed": sd, "steps": n_steps, "guidanceScale": n_guidance,
                    "numFrames": n_frames, "fps": fps,
                    "prompt": positive, "negativePrompt": negative,
                    "gifPath": str(gif_path), "gitSha": git_sha,
                    "timings": {"generateSec": elapsed},
                }
                (out_dir / "run.json").write_text(json.dumps(run_json, indent=2))
                st.session_state["anim_results"].append(run_json)

            if animate_btn:
                if anim_mode == "Ken Burns (instant)":
                    if not kb_image_path:
                        st.info("Paste an image path from the Generate tab above.")
                    elif not Path(kb_image_path).exists():
                        st.error(f"File not found: {kb_image_path}")
                    else:
                        with st.spinner("Applying Ken Burns effect..."):
                            try:
                                _doKenBurns(kb_image_path, kb_motion, kb_frames, kb_fps, kb_intensity)
                            except Exception as exc:
                                st.error(f"Ken Burns failed: {exc}")
                else:
                    if not spec_dict:
                        st.info("Fill in the spec form above and click **Apply spec** first.")
                    else:
                        with st.spinner(f"Animating {anim_frames} frames (seed={anim_seed})..."):
                            try:
                                _doAnimateDiff(
                                    spec_dict, anim_strategy, anim_seed,
                                    anim_steps, anim_guidance, anim_frames, anim_fps,
                                )
                            except Exception as exc:
                                st.error(f"AnimateDiff failed: {exc}")

            if st.session_state.get("anim_results"):
                if st.button("Clear animations", key="clear_anim"):
                    st.session_state["anim_results"] = []
                    st.rerun()

                for result in reversed(st.session_state["anim_results"]):
                    gif_path = Path(result.get("gifPath", ""))
                    if gif_path.exists():
                        st.image(str(gif_path), use_container_width=True)
                    else:
                        st.warning(f"GIF not found: {gif_path}")

                    if result.get("type") == "kenburns":
                        caption = (
                            f"{result.get('motion', '')} | "
                            f"{result.get('numFrames', '')} frames @ {result.get('fps', '')} fps | "
                            f"intensity={result.get('intensity', '')}"
                        )
                    else:
                        caption = (
                            f"{result.get('strategy', '')} | "
                            f"seed={result.get('seed', '')} | "
                            f"{result.get('numFrames', '')} frames @ {result.get('fps', '')} fps"
                        )
                    st.caption(caption)

                    with st.expander("Run metadata"):
                        if result.get("type") == "kenburns":
                            st.write(f"**Source:** {result.get('sourceImage', '')}")
                            st.write(f"**Motion:** {result.get('motion', '')} | **Intensity:** {result.get('intensity', '')}")
                        else:
                            st.write(f"**Spec:** {formatSpec(result.get('spec', {}))}")
                            st.write(f"**Prompt:** {result.get('prompt', '')}")
                            st.write(f"**Motion adapter:** {result.get('motionAdapter', '')}")
                        timing = result.get("timings", {}).get("generateSec", "?")
                        st.write(f"**Generate time:** {timing}s")
