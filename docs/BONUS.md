# Phase 8 — Bonus: Multimodal Extension

**Course:** UMKC CS 5542 — Quiz Challenge-1
**Phase:** 8 (optional bonus)
**Status:** Complete

---

## What was built

Two complementary animation approaches were implemented, both producing looping GIFs from Roomify's room design outputs. They are surfaced together in the Streamlit **Animate** tab and via the CLI.

---

## Approach 1 — Ken Burns pan/zoom effect

**Module:** `src/roomify/kenBurns.py`
**CLI:** `roomify kenburns --image <path> --output <gif> --motion zoom_in`
**Runtime:** ~0.1 s on CPU — no GPU required

Takes any static PNG from the Generate pipeline and applies a smooth camera motion using PIL crop + resize. Six motion types are supported:

| Motion | Description |
|--------|-------------|
| `zoom_in` | Camera slowly pushes into the center of the room |
| `zoom_out` | Camera pulls back to reveal the full scene |
| `pan_right` | Camera travels left-to-right across the room |
| `pan_left` | Camera travels right-to-left |
| `pan_up` | Camera tilts up from floor to ceiling |
| `pan_down` | Camera tilts down from ceiling to floor |

All motions use a smoothstep ease-in-out curve so acceleration and deceleration look natural. Intensity (default 0.2 = 20% crop margin) and frame count are configurable.

**Why this works well:** because Roomify's static images are high-quality 512×512 renders with fine detail, a slow zoom reveals texture and depth that is invisible at full scale. The effect is deterministic and produces professional-looking results from any generation.

### CLI example

```bash
python -m roomify.cli kenburns \
  --image outputs/2026-04-20T20-11-11_core_comparison/kitchen_01_styleAnchored_unctrl_s123/img_0.png \
  --output examples/phase8/kitchen_01_styleAnchored_s123_kenburns.gif \
  --motion zoom_in --frames 24 --fps 12 --intensity 0.2
```

---

## Approach 2 — AnimateDiff text-to-video

**Module:** `src/roomify/animateDiff.py`
**CLI:** `roomify animate --spec configs/examples/kitchen_01.yaml --strategy styleAnchored --seed 123`
**Runtime:** ~10–30 s on A100 / ~2–5 min on T4 (GPU required)

Wraps `AnimateDiffPipeline` from HuggingFace diffusers with the `guoyww/animatediff-motion-adapter-v1-5-2` motion adapter. The motion adapter adds temporal attention layers on top of SD 1.5 so the denoising process produces a sequence of frames that share a consistent scene while varying over time.

**Key configuration decisions:**

| Parameter | Value | Reason |
|-----------|-------|--------|
| Scheduler | `DDIMScheduler` | Required for temporal coherence — default PNDM treats frames independently, causing flickering |
| `clip_sample` | `False` | Prevents aggressive clipping that washes out color |
| `beta_schedule` | `linear` | Matches the motion adapter's training distribution |
| `dtype` | `float16` | Halves VRAM; sufficient quality for 512×512 |
| `torch_dtype` | `float16` | Required by `MotionAdapter.from_pretrained` (different API than the pipeline) |
| Frames | 16 | Balances motion visibility with generation time |
| Steps | 25 | Sufficient for animation; more steps don't meaningfully improve motion |
| Conditioning scale | — | No ControlNet used for animation (spatial constraint degrades motion) |

### CLI example

```bash
python -m roomify.cli animate \
  --spec configs/examples/kitchen_01.yaml \
  --strategy styleAnchored --seed 123 \
  --frames 16 --fps 8 --steps 25
```

---

## Phase 8 outputs

Applied to the top 3 CLIP-scoring runs from the Phase 7 `core_comparison` sweep:

| Run | CLIP Score | Ken Burns motion | GIF |
|-----|-----------|-----------------|-----|
| kitchen_01 · styleAnchored · seed 123 | 0.3529 | zoom_in | `examples/phase8/kitchen_01_styleAnchored_unctrl_s123_kenburns.gif` |
| bathroom_01 · descriptive · seed 7 | 0.3500 | pan_right | `examples/phase8/bathroom_01_descriptive_unctrl_s7_kenburns.gif` |
| kitchen_01 · descriptive · seed 123 | 0.3448 | zoom_out | `examples/phase8/kitchen_01_descriptive_unctrl_s123_kenburns.gif` |

AnimateDiff GIFs for the same three specs are in `examples/phase8/*_animatediff.gif`.

---

## Streamlit integration

The **Animate** tab on the Generate page offers both methods side by side:

- **Ken Burns (instant):** paste any image path from the Generate tab, select motion type, click Animate. Result appears in under a second.
- **AnimateDiff (GPU, ~30 s):** uses the current spec form inputs, runs the full diffusion pipeline, displays the GIF inline via `st.image()`.

Both methods stack results in `st.session_state["anim_results"]` with metadata expanders.

---

## Implementation notes

### Ken Burns — smoothstep easing

```python
def _easeInOut(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)
```

Applied to the interpolation parameter `t ∈ [0, 1]` so motion accelerates out of the start frame and decelerates into the end frame, avoiding the mechanical look of linear interpolation.

### AnimateDiff — dtype mismatch across diffusers classes

`MotionAdapter.from_pretrained` requires `torch_dtype=torch.float16` while `AnimateDiffPipeline.from_pretrained` requires `dtype=torch.float16` — the two classes are on different deprecation schedules in diffusers. Using the wrong keyword for either raises a `TypeError` at runtime.

### AnimateDiff — CUDA placement

The pipeline must be explicitly moved to GPU with `pipe.to("cuda")` after loading. Without this call, inference runs on CPU (~100× slower — observed as a 30+ minute hang vs. 10–30 s on A100).

---

## Test coverage

| File | Tests | What is covered |
|------|-------|-----------------|
| `tests/testKenBurns.py` | 18 | Return type, frame count, size preservation, all 6 motion types, zero intensity, unknown motion error, frames < 2 error, CLI command |
| `tests/testAnimateDiff.py` | 27 | Singleton, load (MotionAdapter + AnimateDiffPipeline calls, dtype params, DDIMScheduler, attention slicing, VAE slicing, no-op on reload), generate (RuntimeError guard, PIL frames, all kwargs), framesToGif, CLI animate command |
