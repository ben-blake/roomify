# Roomify — Phased Task List

Estimates are rough solo-developer hours. Runtime is **Google Colab Pro** — no local GPU — with Streamlit hosted in the Colab VM and exposed via a Cloudflare quick tunnel. Each phase ends with a verifiable artifact before the next begins. The Streamlit web app is built on top of the service layer in a dedicated phase (§6) after the core pipeline is working from the CLI.

---

## Phase 0 — Repo & Colab Bring-Up (2-3 h)

- [ ] Initialize git repo (GitHub, public) with `.gitignore` (Python, `outputs/`, `data/sunrgbd_subset/`, `*.ckpt`, `.ipynb_checkpoints/`)
- [ ] Create `requirements.txt` with pinned versions (torch, diffusers, transformers, accelerate, controlnet-aux, opencv-python, open_clip_torch, lpips, pandas, matplotlib, typer, streamlit, pillow, pyngrok, pytest, jupyter)
- [ ] Scaffold module layout per `ARCHITECTURE.md` (§2) including `src/roomify/ui/`
- [ ] Add `README.md` with a prominent "Open in Colab" badge pointing at `notebooks/00_launchColab.ipynb`
- [ ] Author `notebooks/00_launchColab.ipynb` skeleton with labelled cells:
  - Cell 1: mount Google Drive and `mkdir -p /content/drive/MyDrive/roomify/{data,outputs,hf_cache}`; symlink into `/content/roomify/`
  - Cell 2: `git clone` the repo and `pip install -r requirements.txt`
  - Cell 3: set `HF_HOME` to the Drive-backed cache so weights persist across sessions
  - Cell 4: `nvidia-smi` + write the detected GPU name into a session-level log
  - Cell 5: smoke-test `diffusers` SD 1.5 text-to-image (produce one PNG)
  - Cell 6 (stub for Phase 6): placeholder for `streamlit run app.py &` + `cloudflared` tunnel
  - Cell 7: "reconnect" helper — remount Drive and restart the tunnel without redownloading weights
- [ ] Add AI-tools disclosure placeholder in `README.md`

**Exit criterion:** The launcher notebook runs end-to-end in a fresh Colab Pro session, produces a baseline SD PNG, and `python -m roomify.cli --help` runs from a shell cell.

---

## Phase 1 — Data Pipeline (2-3 h)

- [ ] Download SUN RGB-D **into Google Drive** (document the exact URL in README); the launcher notebook reads from Drive so this is a one-time cost per Drive
- [ ] Write `scripts/buildSubset.py` to curate ~200 samples across 5 scene types (bedroom, living_room, kitchen, office, bathroom) — even distribution; output lives at `/content/drive/MyDrive/roomify/data/sunrgbd_subset/`
- [ ] Emit `data/manifest.csv` (on Drive) with columns `id, sceneType, rgbPath, depthPath, objectLabels`
- [ ] Implement `src/roomify/dataset.py`: `loadManifest()`, `getRecord(id)`, `listByScene(sceneType)`
- [ ] `tests/testDataset.py`: manifest loads, depth file exists for each record, scene types valid
- [ ] `notebooks/01_explore_sunrgbd.ipynb`: sanity-check visualizations (RGB + depth pairs)

**Exit criterion:** `pytest tests/testDataset.py` passes and notebook renders RGB+depth pairs.

---

## Phase 2 — Prompt Builder (1-2 h)

- [ ] Author `configs/prompts.yaml` with three strategies (`minimal`, `descriptive`, `styleAnchored`) and a shared negative prompt
- [ ] Implement `src/roomify/promptBuilder.py`: `buildPrompt(spec, strategy) -> (positive, negative)`
- [ ] Define `RoomSpec` dataclass and schema validation
- [ ] `tests/testPromptBuilder.py`: each strategy produces non-empty, strategy-distinct prompts for a fixed spec

**Exit criterion:** unit tests green; manual eyeball of generated prompts reads cleanly.

---

## Phase 3 — Baseline SD Pipeline (2-3 h)

- [ ] Implement `src/roomify/pipeline.py::Pipeline` wrapping `StableDiffusionPipeline` (SD 1.5)
- [ ] Provide a process-level singleton accessor `getPipeline()` suitable for `@st.cache_resource`
- [ ] Enable fp16, attention slicing, deterministic seeds
- [ ] Support `.generate(positive, negative, seed, steps, guidance)` → PIL image
- [ ] Add `src/roomify/cli.py generate` command that takes a spec YAML and writes an image + `run.json`
- [ ] Generate one sample per scene type using the `descriptive` strategy

**Exit criterion:** `roomify generate --spec configs/examples/bedroom_01.yaml` produces an image + metadata.

---

## Phase 4 — ControlNet Integration (2-3 h)

- [ ] Add `src/roomify/controlSignals.py`: `extractDepth(rgbOrDepth)`, `extractCanny(rgb, lo, hi)`
- [ ] Extend `Pipeline` to load `StableDiffusionControlNetPipeline` with `lllyasviel/sd-controlnet-depth` and `sd-controlnet-canny`
- [ ] Wire CLI flag `--control depth|canny|none` and `--ref-image <id>`
- [ ] Confirm VRAM headroom; fall back to attention slicing / CPU offload if OOM
- [ ] Generate controlled vs uncontrolled pair for at least one room spec and eyeball layout adherence

**Exit criterion:** depth-conditioned generation visibly follows the reference layout.

---

## Phase 5 — Experiment Orchestrator (1-2 h)

- [ ] Define experiment YAML schema (specs × strategies × controlled/uncontrolled × seeds)
- [ ] Implement `src/roomify/orchestrator.py::runExperiment(configPath, progressCb=None)` — iterates and writes `outputs/<runId>/...` with per-image `run.json`; `progressCb(done, total)` so the UI can drive a progress bar
- [ ] Author `configs/experiments/core.yaml`: 5 specs × 3 strategies × 2 (controlled/uncontrolled) × 3 seeds = 90 images
- [ ] Add `roomify sweep` CLI command

**Exit criterion:** full sweep completes end-to-end; all outputs have metadata.

---

## Phase 6 — Streamlit Web App + Colab Tunnel (4-6 h)

- [ ] `app.py` entrypoint: `streamlit run app.py` launches a multi-page app
- [ ] Fill in Cell 6 of `00_launchColab.ipynb`:
  - Install `cloudflared` (binary download) on session start if missing
  - Launch `streamlit run app.py --server.port 8501 --server.headless true &` with logs tailed to a notebook cell
  - Launch `cloudflared tunnel --url http://localhost:8501` and parse the printed `trycloudflare.com` URL into a clickable output
  - Provide a `pyngrok` fallback cell commented out
- [ ] Resolve `outputs/` path to `/content/drive/MyDrive/roomify/outputs/` when Drive is mounted, else `/content/outputs/`
- [ ] Make `getPipeline()` load weights from the Drive-backed HF cache so a Colab re-launch is <30 s
- [ ] `src/roomify/ui/components.py`: reusable `specForm()`, `imageCard(runJson)`, `metricsTable(df)`, `controlPreview()`
- [ ] **Generate page** (`pageGenerate.py`):
  - Spec form (room type, size, style enum, furniture multi-select, lighting, mood)
  - Reference-image picker — thumbnails from SUN RGB-D subset, optional
  - Strategy selector (`minimal | descriptive | styleAnchored`)
  - Controlled toggle + control-type selector (`depth | canny`)
  - Seed input ("random" default), steps/guidance sliders (collapsible "advanced")
  - `Generate` button → `st.spinner` → image renders inline with its `run.json` metadata in an expander
  - `Generate variant` button stacks additional seeds side-by-side in the session
- [ ] **Experiments page** (`pageExperiments.py`):
  - Pick an experiment YAML, click `Run sweep`, show `st.progress` bar wired to `progressCb`
  - On completion, render metrics table + contact sheet + baseline-vs-improved comparison
- [ ] **Gallery page** (`pageGallery.py`): browse `outputs/`, filter by scene type / strategy / controlled, click opens image + JSON
- [ ] Cache the pipeline with `@st.cache_resource`, manifest with `@st.cache_data`; pre-warm on app start (tiny 64x64 generation)
- [ ] Ship a tiny `tests/testUiComponents.py` that validates pure-python helpers in `components.py` (no Streamlit runtime)
- [ ] Update `README.md` with the Colab-first flow (Open in Colab → run launcher → click tunnel URL) and a screenshot

**Exit criterion:** from a fresh Colab Pro session, the launcher notebook produces a tunnel URL in <3 min, and a spec-form submission returns an image in ≤15 s on T4.

---

## Phase 7 — Evaluation & Reporting (2-3 h)

- [ ] Implement `src/roomify/evaluation.py`: `clipAlignment()`, `lpipsDiversity()`, `styleConsistency()`
- [ ] Add manual rating workflow: UI control on the Gallery page to write/update `ratings.csv`; CLI mirror `roomify rate <runDir>`
- [ ] Implement `src/roomify/reporting.py`: `contactSheet(runDir)`, `metricsTable(runDir)`
- [ ] `notebooks/03_evaluation.ipynb`: render all comparisons (baseline vs improved, controlled vs uncontrolled, cross-strategy) and a failure-case section
- [ ] Export top comparisons as PNGs ready for slide inclusion

**Exit criterion:** one markdown metrics table + one contact-sheet PNG demonstrating improved beats baseline.

---

## Phase 8 — Bonus: Multimodal Extension (optional, 2-4 h)

Pick **one**:
- [ ] **Image → video:** AnimateDiff on top 3 generations, or Pika/Runway API for pan-zoom clips
- [ ] **Narration:** ElevenLabs TTS reading room spec + short caption, muxed into demo video
- [ ] **Text → video:** Pika/Runway text-to-video from the same prompt for comparison

Document rationale, examples, and evaluation in `docs/BONUS.md`. Surface it as a tab on the Generate page if straightforward.

---

## Phase 9 — Deliverables (3-4 h)

- [ ] **README.md:** Open-in-Colab badge, launcher-notebook walkthrough, CLI usage, dataset description (SUN RGB-D + subset process, Drive layout), tools/libraries list, runtime notes (Colab Pro + Cloudflare tunnel), AI-tools disclosure (Claude Code used for planning + scaffolding; list any others), sample outputs with links, screenshots of the web app
- [ ] **Sample outputs:** commit a curated `examples/` folder (≤20 small PNGs) generated on Colab
- [ ] **Slides (10+):** scenario, dataset, methodology, SD pipeline, prompt design, control strategy, tools (call out Colab Pro + Streamlit + Diffusers + ControlNet + Cloudflare tunnel), results (images), demo + repo URLs, evaluation, findings, limitations, AI disclosure; note the specific GPU used
- [ ] **Demo video (90 s):** pre-warm the Colab session before recording; script: intro (20s), launcher notebook + tunnel URL (15s), web app walkthrough with a live generation (35s), results/comparisons (20s), conclusion (10s); upload (YouTube unlisted or GitHub release). Have a backup pre-recorded generation in case of tunnel hiccup.
- [ ] **GitHub repo:** push public, verify the launcher notebook runs end-to-end in a fresh Colab session (incognito/no Drive cache) and document the first-run time budget

**Exit criterion:** rubric checklist in §7 of PRD green; submit.

---

## Totals (core, excluding bonus)

- **Core (Phases 0-7, 9):** ~18-27 h across 9 phases (Phase 0 +1 h and Phase 6 +1 h for Colab/tunnel wiring)
- **With bonus:** add 2-4 h
- **Buffer for Colab disconnects / tunnel flakes / debugging:** +25%

## Cross-Cutting

- **Testing:** each new module gets a `tests/testXxx.py`; target ≥60% coverage for pure-Python modules (pipeline + Streamlit runtime excluded since they need GPU / a live server)
- **Commits:** conventional prefixes (`feat:`, `fix:`, `docs:`, `test:`, `chore:`, `refactor:`)
- **Reproducibility:** every experiment config is committed; all runs log seed + git SHA + detected GPU into `run.json`; the web app writes the same `run.json` as the CLI
- **Persistence:** the launcher notebook caches HF weights, SUN RGB-D subset, and outputs under `/content/drive/MyDrive/roomify/` so Colab disconnects don't cost work
- **AI disclosure:** maintain `docs/AI_TOOLS.md` as work progresses
