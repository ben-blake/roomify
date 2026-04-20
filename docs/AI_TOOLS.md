# AI Tools Disclosure

**Course:** UMKC CS 5542 — Quiz Challenge-1
**Project:** Roomify

Per course policy, all AI tool usage must be disclosed. This file is updated
throughout the project as new tools are used.

---

## Tools used

| Tool | Purpose | Scope |
|------|---------|-------|
| Claude Code (Anthropic) | Planning (PRD, architecture, task list), project scaffolding (module stubs, CLI, launcher notebook, configs), CLAUDE.md authoring | All phases — planning + scaffolding |

---

## Session log

### 2026-04-14 — Phase 0 scaffolding

**Tool:** Claude Code (claude-sonnet-4-6)

**Used for:**
- Authored `docs/PRD.md`, `docs/ARCHITECTURE.md`, `docs/TASKS.md` — full planning documents
- Authored `CLAUDE.md` — operating notes for future Claude Code sessions
- Scaffolded module layout: all stub files under `src/roomify/` and `src/roomify/ui/`
- Created `app.py` (Streamlit entrypoint), `src/roomify/cli.py` (typer CLI shell)
- Created `configs/prompts.yaml`, `configs/examples/bedroom_01.yaml`
- Created `notebooks/00_launchColab.ipynb` with 7 labelled cells
- Created `requirements.txt`, `.gitignore`, `README.md`
- Created `tests/testPhase0.py` (smoke tests for Phase 0 exit criterion)

**Hand-written / not AI-generated (this session):**
- None yet — Phase 0 is pure scaffolding

---

### 2026-04-14 — Phase 0 Colab verification + dependency fixes

**Tool:** Claude Code (claude-sonnet-4-6)

**Used for:**
- Diagnosed and fixed series of Colab runtime dependency conflicts (diffusers `cached_download` removal, numpy ABI mismatch, matplotlib numpy<2 constraint)
- Rewrote `requirements.txt` to only install packages absent from the Colab 2026.01 GPU runtime (sourced from `github.com/googlecolab/backend-info`)
- Updated SD 1.5 model ID from `runwayml/stable-diffusion-v1-5` to `stable-diffusion-v1-5/stable-diffusion-v1-5`
- Pushed repo to GitHub (`ben-blake/roomify`), replaced placeholder URLs

**Hand-written / not AI-generated (this session):**
- Colab execution and verification (running cells, observing outputs)
- HF token creation and model license acceptance

---

### 2026-04-14 — Phase 1 Data Pipeline

**Tool:** Claude Code (claude-sonnet-4-6)

**Used for:**
- Wrote `tests/testDataset.py` (17 tests, 100% coverage on `dataset.py`)
- Wrote `scripts/buildSubset.py` — curates ~200 samples from SUN RGB-D across 5 scene types, emits `manifest.csv`
- Wrote `notebooks/01_explore_sunrgbd.ipynb` — RGB+depth pair grid, contact sheet, label frequency chart
- Added `pytest.ini` and `conftest.py` so `pytest` works from repo root
- Updated README Dataset section with SUN RGB-D download URL and one-time setup command
- Updated `docs/TASKS.md` to mark Phase 1 complete

**Hand-written / not AI-generated (this session):**
- Dataset download (done manually in Google Drive)
- Colab notebook execution and verification

---

### 2026-04-14 — Phase 1 completion + notebook fixes

**Tool:** Claude Code (claude-sonnet-4-6)

**Used for:**
- Diagnosed and fixed series of Colab notebook issues: port 8501 conflict, cloudflared pipe-buffer deadlock, isolated runtime `ModuleNotFoundError`, FUSE persistence (subset files lost after session end)
- Fixed `notebooks/00_launchColab.ipynb`: Cell 6 now kills stale processes, redirects cloudflared output to file, verifies Streamlit started before launching tunnel
- Fixed `notebooks/01_explore_sunrgbd.ipynb`: added self-contained setup cell (Drive mount, clone, data symlink) so it runs independently in its own Colab runtime
- Added widget manager cells to launcher notebook
- Moved Phase 1 EDA outputs to `examples/phase1/`
- Updated README, CLAUDE.md, docs/AI_TOOLS.md to reflect Phase 1 completion and new Colab gotchas

**Hand-written / not AI-generated (this session):**
- Running all cells in Colab and verifying outputs
- Downloading EDA PNGs from Colab and adding to repo

---

### 2026-04-14 — Phase 2 Prompt Builder

**Tool:** Claude Code (claude-sonnet-4-6)

**Used for:**
- Wrote `tests/testPromptBuilder.py` (19 tests) before implementation — TDD RED phase
- Implemented `src/roomify/promptBuilder.py`: `buildPrompt(spec, strategy) -> (positive, negative)`, `_renderTemplate()` with graceful handling of missing optional fields
- Completed `RoomSpec` dataclass (required fields, `field(default_factory=list)` for furniture)
- Updated `docs/TASKS.md` to mark Phase 2 complete

**Hand-written / not AI-generated (this session):**
- None — Phase 2 is pure Python module implementation

---

---

### 2026-04-14 — Phase 3 Baseline SD Pipeline

**Tool:** Claude Code (claude-sonnet-4-6)

**Used for:**
- Wrote `tests/testPipeline.py` (18 tests) before implementation — TDD RED phase
- Implemented `src/roomify/pipeline.py`: `Pipeline.load()` (fp16, attention slicing, lazy imports), `Pipeline.generate()` (seeded generator, steps/guidance pass-through), `getPipeline()` singleton, `_resetPipeline()` for test isolation
- Implemented `src/roomify/paths.py`: `getOutputDir()` / `getDataDir()` resolving Drive → Colab → local dev path
- Wired up `src/roomify/cli.py` `generate` command: loads spec YAML → `buildPrompt` → `pipeline.generate` → writes `img_0.png` + `run.json` (all schema keys per ARCHITECTURE.md §4)
- Updated `docs/TASKS.md` to mark Phase 3 complete

**Hand-written / not AI-generated (this session):**
- None — Phase 3 is pure Python module implementation

---

---

### 2026-04-14 — Phase 4 ControlNet Integration

**Tool:** Claude Code (claude-sonnet-4-6)

**Used for:**
- Wrote `tests/testControlSignals.py` (10 tests) before implementation — TDD RED phase; mocked cv2 at module level (not in dev deps)
- Extended `tests/testPipeline.py` with 10 new ControlNet tests — TDD RED phase
- Implemented `src/roomify/controlSignals.py`: `extractDepth()` (grayscale normalization + RGB luminance proxy), `extractCanny()` (cv2.cvtColor + cv2.Canny)
- Extended `src/roomify/pipeline.py`: ControlNet load path branching on `controlType`, `generate()` passes `image=control` kwarg when control is provided
- Updated `src/roomify/cli.py`: `--ref-image <id>` flag, normalizes `"none"` → `None`, lazy-loads record + extracts control signal, passes `controlType` to `load()`, adds `controlnet` and `refImageId` keys to `run.json`

**Hand-written / not AI-generated (this session):**
- None — Phase 4 is pure Python module implementation

---

### 2026-04-14 — Phase 5 Experiment Orchestrator

**Tool:** Claude Code (claude-sonnet-4-6)

**Used for:**
- Wrote `tests/testOrchestrator.py` (18 tests) before implementation — TDD RED phase; patched `getPipeline()` directly to avoid sys.modules diffusers mock pollution with testPipeline.py
- Authored `configs/experiments/core.yaml` — 5 specs × 3 strategies × 1 controlled flag × 3 seeds
- Implemented `src/roomify/orchestrator.py::runExperiment(configPath, progressCb=None)` — iterates sweep matrix, writes per-cell subdirectory with `img_0.png` + `run.json`, calls `progressCb(done, total)` after each image
- Wired `roomify sweep` CLI command (replaced "Phase 5 implementation pending" stub in `cli.py`)
- Updated `docs/TASKS.md` to mark Phase 5 complete

**Hand-written / not AI-generated (this session):**
- None — Phase 5 is pure Python module implementation

---

### 2026-04-14 — Phase 6 Streamlit Web App + Colab Tunnel

**Tool:** Claude Code (claude-sonnet-4-6)

**Used for:**
- Wrote `tests/testUiComponents.py` (19 tests) before implementation — TDD RED phase; tests pure-Python helpers only (no Streamlit runtime required)
- Implemented pure-Python helpers in `src/roomify/ui/components.py`: `parseRunJson`, `listGalleryRuns` (recursive, with scene/strategy/controlled filters), `buildMetricsDf`, `formatSpec`
- Implemented Streamlit components in `components.py` with lazy `import streamlit` (not available locally): `specForm`, `imageCard`, `metricsTable`, `controlPreview`
- Implemented `src/roomify/ui/pageGenerate.py`: spec form, strategy/seed/steps/guidance controls, Generate + Generate variant buttons, variant stack in `st.session_state`, control signal preview
- Implemented `src/roomify/ui/pageExperiments.py`: config YAML picker, live `st.progress` via background thread + `progressCb`, metrics table + contact sheet, past-sweep browser
- Implemented `src/roomify/ui/pageGallery.py`: filters (scene/strategy/controlled), metrics expander, 3-column image grid
- Updated `app.py`: Drive-backed `HF_HOME` setup, `@st.cache_resource` pre-warm (1-step 64×64 generation on startup)
- Updated `docs/TASKS.md` to mark Phase 6 complete

**Hand-written / not AI-generated (this session):**
- None — Phase 6 is pure Python module implementation

---

### 2026-04-15 — Phase 6 Colab live testing + bug fixes

**Tool:** Claude Code (claude-sonnet-4-6)

**Used for:**
- Diagnosed and fixed `st.form` session state bug: form submit only valid during one Streamlit rerun; fixed by saving `specForm()` output to `st.session_state["spec_dict"]` so it persists across Generate button clicks
- Fixed `use_column_width` deprecation warnings in `components.py` and `pageExperiments.py` → `use_container_width=True`
- Fixed `getRecord()` missing-argument bug at 4 call sites (`cli.py`, `pageGenerate.py` ×2, `orchestrator.py`): function requires manifest df as first arg; all callers now call `loadManifest()` and pass result
- Fixed Cell 8/9 silent subprocess output: added `capture_output=True, text=True` + explicit `print(result.stdout/stderr)`
- Fixed Cell 9 `ModuleNotFoundError` for CLI subprocesses: injected `PYTHONPATH=str(REPO_DIR / 'src')` into subprocess env
- Fixed `torch_dtype` deprecation in `pipeline.py` (3 places) → `dtype=torch.float16`; updated corresponding test assertion in `testPipeline.py`
- Removed `TRANSFORMERS_CACHE` env var from `app.py` and Cell 3; `HF_HOME` alone is sufficient in current diffusers/transformers
- Added notebook Cells 8–10 to `00_launchColab.ipynb`: experiment sweep (Cell 8), controlled/uncontrolled pair generation (Cell 9), VRAM headroom check (Cell 10)
- Added "Clear all outputs" button to `app.py` sidebar (Danger zone expander)
- Updated `docs/TASKS.md`: marked all 3 deferred Phase 6 manual tasks complete with actual VRAM/GPU data

**Confirmed during live Colab session (A100-SXM4-80GB):**
- 45-image sweep (5 specs × 3 strategies × 3 seeds) completed successfully via Streamlit Experiments page
- Controlled (depth) vs uncontrolled pair for `bedroom_01`, seed 42 — layout adherence confirmed
- VRAM usage: 5364/81920 MiB (6.5%) — attention slicing sufficient, CPU offload not needed

**Hand-written / not AI-generated (this session):**
- All Colab cell execution and output verification
- Visual inspection of generated images

---

### 2026-04-20 — Phase 7 Evaluation & Reporting

**Tool:** Claude Code (claude-sonnet-4-6)

**Used for:**
- Wrote `tests/testEvaluation.py` (18 tests) before implementation — TDD RED phase; used `patch.dict(sys.modules)` fixture to avoid polluting testPipeline.py's torch mock
- Wrote `tests/testReporting.py` (15 tests) before implementation — TDD RED phase; no GPU mocking needed (pure Pillow + JSON)
- Implemented `src/roomify/evaluation.py`: `clipAlignment()` (CLIP ViT-B-32 text-image cosine similarity), `lpipsDiversity()` (mean pairwise LPIPS distance), `styleConsistency()` (mean pairwise CLIP image-image similarity), `saveRating()` / `loadRatings()` (ratings.csv upsert workflow)
- Implemented `src/roomify/reporting.py`: `contactSheet()` (smallest-square PIL grid, configurable thumbSize), `metricsTable()` (markdown table from run.json files)
- Wired `roomify evaluate` CLI command: calls all three metrics, prints DataFrame + means
- Wired `roomify report` CLI command: saves contact_sheet.png + prints markdown table
- Added `roomify rate` CLI command: interactive 1-5 star rating loop over a run/sweep directory
- Added rating slider widget to Gallery page (`pageGallery.py`): select_slider 0-5 with star display, auto-saves on change via `saveRating()` + `st.rerun()`
- Created `notebooks/03_evaluation.ipynb`: 8-cell notebook (setup, sweep picker, CLIP alignment, LPIPS/consistency, contact sheet, metrics table, controlled vs uncontrolled comparison, top-N export)
- Updated `docs/TASKS.md` to mark Phase 7 complete

**Hand-written / not AI-generated (this session):**
- None — Phase 7 is pure Python module implementation

---

### 2026-04-20 — Phase 7 Colab live testing + bug fixes

**Tool:** Claude Code (claude-sonnet-4-6)

**Used for:**
- Diagnosed and fixed `pipeline.load()` crash when switching from uncontrolled to controlled mid-sweep: added early-return when `controlType` is unchanged, and `del self._sd` + `torch.cuda.empty_cache()` before loading a different pipeline variant
- Fixed `dtype` vs `torch_dtype` mismatch: all three `from_pretrained` calls updated to `torch_dtype=torch.float16` (diffusers validates kwargs; `ControlNetModel` raised a hard error on unrecognized `dtype=`)
- Added `conditioningScale` parameter (default 0.6) to `Pipeline.generate()` and threaded through `orchestrator.py` + `run.json` to reduce ControlNet over-constraint
- Fixed `top_exports` cell in `03_evaluation.ipynb`: used `src.parent.name` as filename so all 6 exports are uniquely named instead of all overwriting `img_0.png`
- Created `examples/phase7/METRICS.md` summarizing aggregate metrics, top-10 CLIP table, and key findings
- Committed all Phase 7 evaluation outputs to `examples/phase7/` (contact sheet, 15 ctrl/unctrl comparison PNGs, 3 strategy comparison PNGs, 6 top-export PNGs)
- Marked both remaining manual Phase 7 tasks complete in `docs/TASKS.md`

**Confirmed during live Colab session (A100-SXM4-80GB):**
- 90-image `core_comparison` sweep completed (5 specs × 3 strategies × 2 ctrl/unctrl × 3 seeds)
- Mean CLIP score: 0.2739; LPIPS diversity: 0.7583; style consistency: 0.5431
- Top CLIP scores are uncontrolled images — confirms ControlNet trades prompt alignment for spatial structure

**Hand-written / not AI-generated (this session):**
- All Colab cell execution and output verification
- Visual inspection of controlled vs uncontrolled image pairs
- Selection and download of output PNGs for the repo

---

### 2026-04-20 — Phase 8 AnimateDiff

**Tool:** Claude Code (claude-sonnet-4-6)

**Used for:**
- Wrote `tests/testAnimateDiff.py` (25 tests) before implementation — TDD RED phase; mocked torch and diffusers at module level, patched singleton reset
- Implemented `src/roomify/animateDiff.py`: `AnimateDiffGenerator.load()` (MotionAdapter + AnimateDiffPipeline, fp16, attention slicing, no-op on reload), `AnimateDiffGenerator.generate()` (seeded generator, frames → list of PIL Images), `framesToGif()` (PIL animated GIF with fps-derived duration), `getAnimateDiffGenerator()` / `_resetAnimateDiffGenerator()` singleton helpers
- Wired `roomify animate` CLI command (`cli.py`): `--spec`, `--strategy`, `--seed`, `--steps`, `--frames`, `--fps`; writes `anim.gif` + `run.json` with `type: animate`
- Updated `docs/TASKS.md` to mark Phase 8 complete

**Hand-written / not AI-generated (this session):**
- None — Phase 8 is pure Python module implementation

---

*Append a new entry to the session log for each session that uses AI assistance.*
