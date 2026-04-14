# Roomify — Product Requirements Document

**Course:** UMKC CS 5542 — Quiz Challenge-1 (GenAI: Stable Diffusion)
**Theme:** Real-World Controlled Image Generation
**Date:** 2026-04-14
**Status:** Draft v0.1 (pre-implementation)

---

## 1. Problem Statement

Interior design visualization is expensive and slow: designers hand-render rooms or rely on stock photography that rarely matches a client's constraints (size, style, furniture list, lighting). Companies like IKEA are adopting generative AI to preview spaces before committing to a layout.

**Roomify** is a data-driven image generation system that takes structured room descriptions and constraints and produces interior design renders using Stable Diffusion with explicit control mechanisms (prompt templates, negative prompts, ControlNet conditioning).

## 2. Goals

| # | Goal | Measurable Outcome |
|---|------|--------------------|
| G1 | Generate coherent interior design images from structured inputs | ≥80% of baseline generations render a recognizable room matching the requested type |
| G2 | Apply explicit control to improve quality over an uncontrolled baseline | Improved pipeline beats baseline on CLIP alignment and subjective rating |
| G3 | Maintain style and layout consistency across variants of the same room | Multiple seeds of one input share palette/layout per qualitative review |
| G4 | Compare prompt strategies and controlled vs uncontrolled generation | Side-by-side grid across ≥3 strategies with metrics table |
| G5 | Ship the full class deliverable | 10+ slide deck, 1-2 min demo, public GitHub repo with README, sample outputs, AI-tools disclosure |

## 3. Non-Goals

- No user authentication or multi-user accounts — single-user local app is enough.
- No sub-second "true real-time" generation — interactive on-demand (5-20 s/image) is the target.
- No training or fine-tuning of Stable Diffusion from scratch — use pretrained checkpoints only.
- No production deployment, scaling, or monetization concerns.
- No dataset expansion beyond a curated SUN RGB-D subset.

## 4. Users & Scenarios

**Primary user:** Course instructor and graders evaluating the submission.

**Secondary users (simulated):** Interior designers and homeowners who would input a room brief ("small bedroom, 10x12 ft, Scandinavian style, queen bed, natural light from east wall") and receive multiple generated options.

**Core scenario (web app):**
1. User opens the Streamlit app, fills in a room spec form (type, dimensions, style, furniture, lighting, mood) or picks a preset.
2. User optionally selects a SUN RGB-D reference image to enable ControlNet conditioning.
3. User picks a prompt strategy (minimal | descriptive | styleAnchored) and toggles controlled vs uncontrolled.
4. On submit, Roomify builds the prompt, runs SD, and displays the image inline within ~5-20 s with a live progress indicator.
5. User can generate additional variants (new seeds) side-by-side and flag favorites.
6. A separate "Experiments" page runs batch sweeps and shows the evaluation metrics + contact sheets.

A CLI wrapper provides the same functionality for reproducible batch runs and grading.

## 5. Functional Requirements

- **FR1 — Data loader:** Load a curated subset of SUN RGB-D (RGB + depth + scene label + object annotations) and expose a uniform record schema.
- **FR2 — Structured prompt builder:** Map a room spec dict → positive prompt + negative prompt using Jinja-style templates. Support at least 3 prompt strategies (minimal, descriptive, style-anchored).
- **FR3 — Baseline SD pipeline:** Text-to-image generation using a pretrained SD / SDXL checkpoint from Hugging Face Diffusers.
- **FR4 — Controlled pipeline:** Add ControlNet conditioning (depth and/or Canny) using SUN RGB-D depth maps or edge maps as layout guides.
- **FR5 — Batch orchestrator:** Run the same input across (a) prompt strategies and (b) controlled vs uncontrolled, saving all outputs plus metadata JSON.
- **FR6 — Evaluation:** Compute CLIP text-image alignment, LPIPS diversity across seeds, and a simple style-consistency score (CLIP image-image similarity within a group). Collect qualitative ratings in a CSV.
- **FR7 — Reporting:** Generate a contact-sheet grid (matplotlib) and a metrics table (pandas → markdown) suitable for slide inclusion.
- **FR8 — Streamlit web app:** Single-page interactive UI for spec input + on-demand generation, plus an "Experiments" page for batch sweeps and metrics visualization. Shows a progress bar/spinner during generation; displays image + metadata inline.
- **FR9 — Deliverables:** README with setup + CLI + web-app usage, sample outputs, dataset notes, AI-tools disclosure, and a link to the demo video.

## 6. Non-Functional Requirements

- **Runtime:** Google Colab Pro (T4 / L4 / A100 depending on availability). The repo is designed to clone-and-run from a Colab notebook. No local GPU required. The Streamlit UI runs inside the Colab VM and is exposed to the laptop browser through a tunnel (Cloudflare quick tunnel preferred; `pyngrok` as fallback).
- **Latency target:** Single generation returns within 15 s on T4 (SD 1.5, 30 steps, fp16); <8 s on L4/A100. The UI shows a progress indicator for anything >1 s.
- **Reproducibility:** All runs seeded; config stored as YAML alongside outputs; the UI exposes the seed and writes the same `run.json` as the CLI.
- **Simplicity:** Streamlit for the UI (one file, no build step); the demo must be showable in 1-2 minutes.
- **Transparency:** AI tool usage documented in `README.md` per course policy.

## 7. Success Criteria (Grading Alignment)

| Rubric (weight) | How Roomify earns it |
|-----------------|---------------------|
| Completion of system (30%) | End-to-end pipeline from spec → image, demo-able on CLI |
| Quality of outputs (20%) | Curated gallery of best generations, controlled beats baseline |
| Evaluation & analysis (25%) | Metrics table + failure-case section + trade-off discussion |
| Technical design (15%) | Clean module boundaries, prompt strategies documented, ControlNet integration explained |
| Presentation & demo (10%) | 10+ slide deck + concise video + README |
| Bonus | Optional multimodal extension (image → video via AnimateDiff, or narration via ElevenLabs) |

## 8. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| SUN RGB-D dataset is heavy (~20 GB) | High setup friction | Curate a ≤200-image subset and cache it to Google Drive so Colab sessions don't re-download |
| ControlNet models increase VRAM pressure | Pipeline OOMs on smaller Colab GPUs | Use SD 1.5 + controlnet-sd15; enable fp16 and attention slicing by default |
| Colab session disconnects mid-demo | Lost UI state, dead tunnel | Keep session alive with periodic activity; checkpoint pipeline into a module so re-run takes <30 s; document the launcher notebook's "reconnect" cell |
| Tunnel URL changes every Colab session | Demo prep churn | Use Cloudflare quick tunnels (no auth, stable for the session); print the URL prominently in the launcher cell; pin it in the demo script |
| Colab Pro GPU type is non-deterministic (T4 vs L4 vs A100) | Inconsistent timing in demo | Log GPU type into `run.json`; pick batch sizes and steps that are safe on T4; note actual GPU used in slides |
| Limited Colab GPU time for large sweeps | Incomplete eval | Cap at ~5 room specs × 3 strategies × 2 (controlled/uncontrolled) × 3 seeds = 90 images; save outputs to Drive so a sweep can resume across sessions |
| Subjective quality hard to measure | Weak analysis section | Combine CLIP/LPIPS metrics with a short human-rating CSV (self-rated 1-5) |
| Over-scoping the bonus | Misses core deadline | Bonus is strictly post-core; pick one (image→video via AnimateDiff) |
| Demo video runs over 2 min | Loses presentation points | Script 90s: 20s intro, 40s web-app walkthrough, 20s results, 10s conclusion |
| Streamlit reloads the SD pipeline on every rerun | 30+ s cold start on each interaction | Cache the pipeline with `@st.cache_resource`; keep it hot across reruns |
| First generation is slow (model download + warmup) | Bad demo experience | Pre-warm the model on app start; ship a small "warmup" button |

## 9. Out-of-Scope (explicit)

- Training / LoRA fine-tuning.
- Multi-user accounts, authentication, or cloud deployment.
- Sub-second "true real-time" generation (would require Turbo/LCM and is a deliberate follow-up, not core).
- 3D scene reconstruction from outputs.
