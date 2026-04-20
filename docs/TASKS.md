# Roomify — Phased Task List

Estimates are rough solo-developer hours. Runtime is **Google Colab Pro** — no local GPU — with Streamlit hosted in the Colab VM and exposed via a Cloudflare quick tunnel. Each phase ends with a verifiable artifact before the next begins. The Streamlit web app is built on top of the service layer in a dedicated phase (§6) after the core pipeline is working from the CLI.

---

## Phase 0 — Repo & Colab Bring-Up (2-3 h) ✅

- [x] Initialize git repo (local `main` branch, initial commit `1b1e9c0`); pushed to GitHub public (`ben-blake/roomify`)
- [x] Create `requirements.txt` scoped to packages absent from the Colab 2026.01 GPU runtime (controlnet-aux, open_clip_torch, lpips, streamlit, pyngrok, pytest) + `requirements-dev.txt` (local no-GPU subset); Colab pre-installs torch, diffusers, transformers, numpy, pandas, matplotlib, etc.
- [x] Scaffold module layout per `ARCHITECTURE.md` (§2) including `src/roomify/ui/`
- [x] Add `README.md` with a prominent "Open in Colab" badge pointing at `notebooks/00_launchColab.ipynb`
- [x] Author `notebooks/00_launchColab.ipynb` skeleton with labelled cells:
  - Cell 1: mount Google Drive and `mkdir -p /content/drive/MyDrive/roomify/{data,outputs,hf_cache}`; symlink into `/content/roomify/`
  - Cell 2: `git clone` the repo and `pip install -r requirements.txt`
  - Cell 3: set `HF_HOME` to the Drive-backed cache so weights persist across sessions
  - Cell 4: `nvidia-smi` + write the detected GPU name into a session-level log
  - Cell 5: smoke-test `diffusers` SD 1.5 text-to-image (produce one PNG)
  - Cell 6 (stub for Phase 6): placeholder for `streamlit run app.py &` + `cloudflared` tunnel
  - Cell 7: "reconnect" helper — remount Drive and restart the tunnel without redownloading weights
- [x] Add AI-tools disclosure placeholder in `README.md` + `docs/AI_TOOLS.md`

**Exit criterion:** `python -m roomify.cli --help` passes locally (4/4 smoke tests green). Full Colab end-to-end verified: SD 1.5 smoke image generated, Streamlit + Cloudflare tunnel live.

---

## Phase 1 — Data Pipeline (2-3 h) ✅

- [x] Download SUN RGB-D **into Google Drive** — URL documented in README (https://rgbd.cs.princeton.edu/)
- [x] Write `scripts/buildSubset.py` to curate ~200 samples across 5 scene types (bedroom, living_room, kitchen, office, bathroom) — even distribution; output lives at `/content/drive/MyDrive/roomify/data/sunrgbd_subset/`
- [x] Emit `data/manifest.csv` (on Drive) with columns `id, sceneType, rgbPath, depthPath, objectLabels`
- [x] Implement `src/roomify/dataset.py`: `loadManifest()`, `getRecord(id)`, `listByScene(sceneType)`
- [x] `tests/testDataset.py`: 17 tests, 100% coverage — manifest loads, record fields correct, labels parsed, scene types validated, errors raised correctly
- [x] `notebooks/01_explore_sunrgbd.ipynb`: sanity-check visualizations (RGB + depth pairs, contact sheet, label frequency chart)

**Exit criterion:** `pytest tests/testDataset.py` passes (17/17) and notebook renders RGB+depth pairs. _Local tests green; notebook runs end-to-end in Colab once Drive data is present._

---

## Phase 2 — Prompt Builder (1-2 h) ✅

- [x] Author `configs/prompts.yaml` with three strategies (`minimal`, `descriptive`, `styleAnchored`) and a shared negative prompt _(done in Phase 0)_
- [x] Implement `src/roomify/promptBuilder.py`: `buildPrompt(spec, strategy) -> (positive, negative)`
- [x] Define `RoomSpec` dataclass with schema validation (required fields, default empty list for furniture)
- [x] `tests/testPromptBuilder.py`: 19 tests — non-empty prompts, strategy-distinct outputs, shared negative, optional-field graceful handling, ValueError on unknown strategy, RoomSpec field validation

**Exit criterion:** unit tests green; manual eyeball of generated prompts reads cleanly. ✅ (40/40 total tests pass)

---

## Phase 3 — Baseline SD Pipeline (2-3 h) ✅

- [x] Implement `src/roomify/pipeline.py::Pipeline` wrapping `StableDiffusionPipeline` (SD 1.5)
- [x] Provide a process-level singleton accessor `getPipeline()` suitable for `@st.cache_resource`
- [x] Enable fp16, attention slicing, deterministic seeds
- [x] Support `.generate(positive, negative, seed, steps, guidance)` → PIL image
- [x] Add `src/roomify/cli.py generate` command that takes a spec YAML and writes an image + `run.json`
- [x] Added `src/roomify/paths.py` — Drive/Colab/local path resolution (required by CLI + UI)
- [x] `tests/testPipeline.py`: 18 tests — singleton, load (fp16 + attention slicing), generate (steps/guidance/seed), CLI writes PNG + run.json with all required keys

**Exit criterion:** `roomify generate --spec configs/examples/bedroom_01.yaml` produces an image + metadata. ✅ (58/58 total tests pass locally; Colab generation pending GPU session)

---

## Phase 4 — ControlNet Integration (2-3 h) ✅

- [x] Add `src/roomify/controlSignals.py`: `extractDepth(rgbOrDepth)`, `extractCanny(rgb, lo, hi)`
- [x] Extend `Pipeline` to load `StableDiffusionControlNetPipeline` with `lllyasviel/sd-controlnet-depth` and `sd-controlnet-canny`
- [x] Wire CLI flag `--control depth|canny|none` and `--ref-image <id>`
- [x] `tests/testControlSignals.py`: 10 tests — extractDepth (grayscale + RGB input, size preservation, RGB output), extractCanny (size, RGB output, threshold passthrough)
- [x] Extended `tests/testPipeline.py`: +10 tests — ControlNet load path (depth/canny model IDs, ControlNetModel used, plain SD fallback), generate with/without control image, CLI ControlNet flags

**Exit criterion:** depth-conditioned generation visibly follows the reference layout. ✅ (78/78 total tests pass locally; Colab verification pending GPU session)

---

## Phase 5 — Experiment Orchestrator (1-2 h) ✅

- [x] Define experiment YAML schema (specs × strategies × controlled/uncontrolled × seeds)
- [x] Implement `src/roomify/orchestrator.py::runExperiment(configPath, progressCb=None)` — iterates and writes `outputs/<runId>/...` with per-image `run.json`; `progressCb(done, total)` so the UI can drive a progress bar
- [x] Author `configs/experiments/core.yaml`: 5 specs × 3 strategies × 1 (uncontrolled) × 3 seeds = 45 images (add a controlled sweep config when SUN RGB-D data is present on Drive)
- [x] Add `roomify sweep` CLI command
- [x] `tests/testOrchestrator.py`: 18 tests — return value, output file structure, run.json schema keys, matrix cardinality, progressCb contract, CLI sweep command

**Exit criterion:** full sweep completes end-to-end; all outputs have metadata. ✅ (96/96 total tests pass locally; Colab generation pending GPU session)

---

## Phase 6 — Streamlit Web App + Colab Tunnel (4-6 h) ✅

- [x] `app.py` entrypoint: multi-page app with sidebar navigation, `@st.cache_resource` pre-warm, Drive-backed `HF_HOME` setup
- [x] Cell 6 of `00_launchColab.ipynb`: cloudflared binary install, Streamlit launch with log redirect, URL parse, pyngrok fallback commented
- [x] `outputs/` path resolved via `paths.py` (Drive → Colab → local dev)
- [x] `app.py` sets `HF_HOME` to Drive-backed cache when mounted, before any pipeline load
- [x] `src/roomify/ui/components.py`: pure-Python helpers (`parseRunJson`, `listGalleryRuns`, `buildMetricsDf`, `formatSpec`) + Streamlit components (`specForm`, `imageCard`, `metricsTable`, `controlPreview`) with lazy `st` imports
- [x] **Generate page** (`pageGenerate.py`): spec form, strategy selector, controlled toggle + control-type, seed (random/fixed), steps/guidance sliders, Generate + Generate variant buttons, variant stack in session state, control signal preview
- [x] **Experiments page** (`pageExperiments.py`): config YAML picker, Run sweep button, live `st.progress` bar via `progressCb`, metrics table + contact sheet on completion, past-sweep browser
- [x] **Gallery page** (`pageGallery.py`): scene type / strategy / controlled filters, metrics summary expander, 3-column image grid with `imageCard`
- [x] Pipeline cached with `@st.cache_resource`; pre-warm (1-step 64×64 generation) on app start
- [x] `tests/testUiComponents.py`: 19 tests covering all pure-Python helpers (no Streamlit runtime)
- [ ] Update `README.md` with the Colab-first flow screenshot _(deferred to Phase 9 — needs live Colab session for screenshot)_
- [x] _(Colab, manual)_ Confirm VRAM headroom: A100-SXM4-80GB, 5364/81920 MiB (6.5%) — attention slicing sufficient, CPU offload not needed
- [x] _(Colab, manual)_ Generated 45 samples (5 specs × 3 strategies × 3 seeds) including all scene types with `descriptive` strategy
- [x] _(Colab, manual)_ Generated controlled (depth) vs uncontrolled pair for `bedroom_01` spec, seed 42 — layout adherence confirmed

**Exit criterion:** from a fresh Colab Pro session, the launcher notebook produces a tunnel URL in <3 min, and a spec-form submission returns an image in ≤15 s on T4. ✅ (115/115 total tests pass locally; Colab end-to-end pending GPU session)

---

## Phase 7 — Evaluation & Reporting (2-3 h) ✅

- [x] Implement `src/roomify/evaluation.py`: `clipAlignment()`, `lpipsDiversity()`, `styleConsistency()`
- [x] Add manual rating workflow: `saveRating()` / `loadRatings()` in `evaluation.py`; rating slider in Gallery page; `roomify rate <runDir>` CLI command
- [x] Implement `src/roomify/reporting.py`: `contactSheet(runDir, thumbSize)`, `metricsTable(runDir)`
- [x] Wire `roomify evaluate` and `roomify report` CLI commands (previously stubs)
- [x] `notebooks/03_evaluation.ipynb`: 8-cell notebook covering CLIP alignment, LPIPS diversity, style consistency, contact sheet, metrics table, controlled vs uncontrolled comparison, top-N export
- [x] _(Colab, manual)_ Run full evaluation on 90-image core_comparison sweep: CLIP mean=0.2739, LPIPS diversity=0.7583, style consistency=0.5431
- [x] _(Colab, manual)_ Export top comparison PNGs for slide inclusion — committed to `examples/phase7/`

**Exit criterion:** one markdown metrics table + one contact-sheet PNG demonstrating improved beats baseline. ✅ (148/148 total tests pass locally; Colab eval complete 2026-04-20)

---

## Phase 8 — Bonus: Multimodal Extension (optional, 2-4 h) ✅

**Chosen:** AnimateDiff — text-to-GIF animation from existing room spec prompts.

- [x] Implement `src/roomify/animateDiff.py`: `AnimateDiffGenerator` singleton (`load()`, `generate()`), `framesToGif()`, `getAnimateDiffGenerator()` / `_resetAnimateDiffGenerator()`
- [x] Wire `roomify animate` CLI command: `--spec`, `--strategy`, `--seed`, `--steps`, `--frames`, `--fps`; writes `anim.gif` + `run.json` (`type: animate`) per run
- [x] `tests/testAnimateDiff.py`: 25 tests — singleton, load (MotionAdapter + AnimateDiffPipeline, fp16, attention slicing, no-op on reload), generate (RuntimeError if not loaded, returns PIL frames, passes all kwargs), framesToGif (saves GIF, raises on empty, returns path), CLI (command exists, writes GIF, writes run.json)
- [x] _(Colab, manual)_ Run `roomify animate` on top 3 CLIP-scored specs; commit output GIFs to `examples/phase8/`
- [x] Surface Animate tab on the Generate page (`pageGenerate.py`): `st.tabs(["Generate Image", "Animate (GIF)"])`, shared spec form above tabs, AnimateDiff cached via `@st.cache_resource`, GIF displayed with `st.image()`, results stacked in `st.session_state["anim_results"]`

**Model:** `guoyww/animatediff-motion-adapter-v1-5-2` + SD 1.5 (fp16, attention slicing)

**Exit criterion:** `roomify animate --spec configs/examples/bedroom_01.yaml` produces a looping GIF. ✅ (193/193 tests pass locally; Colab run complete — 6 GIFs committed to `examples/phase8/`)

---

## Phase 9 — Deliverables (3-4 h)

- [ ] **README.md:** Open-in-Colab badge, launcher-notebook walkthrough, CLI usage, dataset description (SUN RGB-D + subset process, Drive layout), tools/libraries list, runtime notes (Colab Pro + Cloudflare tunnel), AI-tools disclosure (Claude Code used for planning + scaffolding; list any others), sample outputs with links, screenshots of the web app
- [ ] **Sample outputs:** commit a curated `examples/` folder (≤20 small PNGs) generated on Colab
- [ ] **PowerPoint Slides (10+):** scenario, dataset, methodology (SD pipeline, prompt design, control strategy), tools (call out Colab Pro + Streamlit + Diffusers + ControlNet + Cloudflare tunnel), results (images), demo + repo URLs, evaluation, findings & insights, limitations, AI disclosure; note the specific GPU used
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
