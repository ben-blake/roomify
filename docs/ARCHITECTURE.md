# Roomify — Architecture Overview

## 1. System Diagram (logical)

```
 ┌────────────────────────────────────────────────────────────────────┐
 │                     Streamlit Web App (app.py)                     │
 │  ┌─────────────────┐  ┌──────────────────┐  ┌────────────────────┐ │
 │  │ Generate page   │  │ Experiments page │  │ Gallery page       │ │
 │  │ • spec form     │  │ • config picker  │  │ • browse outputs/  │ │
 │  │ • ref image     │  │ • run sweep btn  │  │ • contact sheets   │ │
 │  │ • strategy sel. │  │ • metrics table  │  │ • per-image JSON   │ │
 │  │ • [Generate]    │  │ • plots          │  │                    │ │
 │  └────────┬────────┘  └────────┬─────────┘  └────────────────────┘ │
 └───────────┼────────────────────┼───────────────────────────────────┘
             │ user submit        │ kick off sweep
             ▼                    ▼
 ┌────────────────────────────────────────────────────────────────────┐
 │            Service Layer (src/roomify/*.py, importable)            │
 │                                                                    │
 │   promptBuilder → controlSignals → Pipeline (cached singleton) →   │
 │                              orchestrator → evaluation → reporting │
 └────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
 ┌────────────────────────────────────────────────────────────────────┐
 │   Outputs + Metadata:  outputs/<runId>/img_*.png + run.json        │
 │   Metrics + contact sheets:  outputs/<runId>/metrics.csv, grid.png │
 └────────────────────────────────────────────────────────────────────┘

                          ▲
                          │ same service layer
                          │
 ┌────────────────────────┴───────────────────────────────────────────┐
 │   CLI (src/roomify/cli.py)  — reproducible batch + grading path    │
 └────────────────────────────────────────────────────────────────────┘
```

Both the Streamlit UI and the CLI are **thin shells over the same service layer**. No business logic lives in `app.py` or `cli.py`.

## 2. Module Layout

```
roomify/
├── README.md                      # setup, usage (web app + CLI), AI-tools disclosure
├── requirements.txt               # pinned deps
├── app.py                         # Streamlit entrypoint (thin)
├── configs/
│   ├── prompts.yaml               # strategy templates + negatives
│   └── experiments/               # one YAML per sweep
├── data/
│   ├── sunrgbd_subset/            # curated ~200 samples (not committed)
│   └── manifest.csv               # id, scene_type, rgb_path, depth_path
├── src/roomify/
│   ├── __init__.py
│   ├── dataset.py                 # SUN RGB-D loader + record schema
│   ├── promptBuilder.py           # spec dict → (positive, negative)
│   ├── pipeline.py                # SD + ControlNet wrapper (singleton-friendly)
│   ├── paths.py                   # Drive/Colab/local path resolution
│   ├── controlSignals.py          # depth / canny extraction
│   ├── orchestrator.py            # runs the sweep matrix
│   ├── evaluation.py              # CLIP, LPIPS, consistency metrics
│   ├── reporting.py               # contact sheets, markdown tables
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── pageGenerate.py        # the spec form + single-image generate page
│   │   ├── pageExperiments.py     # batch sweep + metrics visualizations
│   │   ├── pageGallery.py         # browse past outputs
│   │   └── components.py          # reusable specForm, imageCard, metricsTable
│   └── cli.py                     # `python -m roomify.cli generate|sweep|evaluate`
├── notebooks/
│   ├── 00_launchColab.ipynb        # clone → install → streamlit + tunnel (primary entry)
│   ├── 01_explore_sunrgbd.ipynb
│   ├── 02_prompt_strategies.ipynb
│   └── 03_evaluation.ipynb
├── outputs/                        # generated images + metadata (gitignored)
├── tests/
│   ├── testDataset.py
│   ├── testPromptBuilder.py
│   ├── testPipeline.py
│   ├── testControlSignals.py
│   └── testEvaluation.py
└── docs/
    ├── PRD.md
    ├── ARCHITECTURE.md
    └── TASKS.md
```

## 3. Component Responsibilities

| Module | Responsibility | Key APIs |
|--------|----------------|----------|
| `dataset.py` | Load SUN RGB-D subset, expose `Record(id, scene_type, rgb, depth, objects)` | `loadManifest()`, `getRecord(id)` |
| `promptBuilder.py` | Map `RoomSpec` → `(positive: str, negative: str)` per strategy | `buildPrompt(spec, strategy)` |
| `controlSignals.py` | Return conditioning image for ControlNet | `extractDepth(img)`, `extractCanny(img, lo, hi)` |
| `pipeline.py` | Own the Diffusers pipeline; lazy-load; unload to save VRAM | `Pipeline.load()`, `Pipeline.generate(positive, negative, seed, steps, guidance)`, `getPipeline()` |
| `paths.py` | Resolve output/data dirs across Drive, Colab, local dev | `getOutputDir()`, `getDataDir()` |
| `orchestrator.py` | Run the sweep matrix, persist outputs + `run.json` metadata | `runExperiment(configPath)` |
| `evaluation.py` | Compute metrics over an output directory | `clipAlignment()`, `lpipsDiversity()`, `styleConsistency()` |
| `reporting.py` | Render contact sheets + markdown metric tables | `contactSheet(runDir)`, `metricsTable(runDir)` |
| `cli.py` | Thin typer CLI wrapping the service layer | `roomify generate`, `roomify evaluate`, `roomify report` |
| `app.py` + `ui/` | Streamlit UI, imports the service layer, owns no business logic | `streamlit run app.py` |

## 4. Data Contracts

### `RoomSpec` (input)
```yaml
id: bedroom_01
roomType: bedroom            # bedroom | living_room | kitchen | office | bathroom
size: "10x12 ft"
style: scandinavian          # enum: minimalist | scandinavian | industrial | mid_century | boho
furniture: [queen bed, nightstand, reading chair]
lighting: "natural light from east window"
mood: "cozy, airy"
referenceImageId: sunrgbd_00142   # optional — enables ControlNet
```

### `run.json` (output metadata, one per generation)
```json
{
  "runId": "2026-04-14T12-30-01_bedroom_01",
  "spec": { "...": "as above" },
  "strategy": "descriptive",
  "controlled": true,
  "controlType": "depth",
  "refImageId": "sunrgbd_00142",
  "model": "stable-diffusion-v1-5/stable-diffusion-v1-5",
  "controlnet": "lllyasviel/sd-controlnet-depth",
  "seed": 42,
  "steps": 30,
  "guidanceScale": 7.5,
  "prompt": "...",
  "negativePrompt": "...",
  "imagePath": "outputs/.../img_0.png",
  "gitSha": "e82fbe1",
  "timings": { "generateSec": 6.4 }
}
```

## 4a. Runtime: Colab-Hosted Streamlit

The system runs entirely inside a Google Colab Pro VM. A launcher notebook (`notebooks/00_launchColab.ipynb`) performs the full bring-up; no local installation is required.

**Launcher notebook sequence:**
1. Mount Google Drive (for persistent `data/sunrgbd_subset/` and `outputs/`).
2. `git clone` the repo, `cd` in, `pip install -r requirements.txt`.
3. If `data/sunrgbd_subset/` is missing, run `scripts/buildSubset.py` (one-time per Drive).
4. Pre-download SD 1.5 + ControlNet weights into the HF cache (one-time warmup).
5. Start Streamlit in the background: `streamlit run app.py --server.port 8501 &`.
6. Start a Cloudflare quick tunnel: `cloudflared tunnel --url http://localhost:8501` (no auth). Fallback: `pyngrok` with the user's authtoken.
7. Print the public URL prominently; that URL is what the laptop browser + demo video use.

**Why this shape:**
- The Streamlit app code is identical whether it runs locally or in Colab — the tunnel is an ops concern, not a code concern.
- Google Drive persistence means SUN RGB-D and generated outputs survive Colab disconnects.
- Cloudflare quick tunnels require no auth and no rate-limit signup, which matters for a one-shot demo.

**Session hygiene:**
- Outputs write to `/content/drive/MyDrive/roomify/outputs/` when Drive is mounted, else `/content/outputs/`.
- `run.json` logs the detected GPU type (`nvidia-smi --query-gpu=name`) for slide transparency.
- The launcher notebook has a "reconnect" cell that re-mounts Drive and re-starts the tunnel after a Colab disconnect without redownloading weights.

## 4b. Streamlit App Structure

Streamlit multi-page app rooted at `app.py`:

- **Generate page** (default): spec form (room type, size, style, furniture chips, lighting text, mood text), optional reference-image picker from the SUN RGB-D subset, strategy selector, controlled/uncontrolled toggle, seed input (or "random"), and a `Generate` button. On submit: `st.spinner` shows progress, the image renders inline with its metadata, and a `Generate variant` button runs again with a new seed so variants stack side-by-side.
- **Experiments page**: pick an `experiment.yaml`, run the sweep with live progress, render the metrics table and the contact sheet. Shows baseline-vs-improved comparisons.
- **Gallery page**: browse `outputs/` with thumbnails; click opens the full image + `run.json`.

**Streamlit performance rules (critical):**
- Cache the SD pipeline with `@st.cache_resource` — one load per process, never per interaction.
- Cache the SUN RGB-D manifest with `@st.cache_data`.
- Pre-warm the pipeline at app start (render a tiny 64x64 "warmup" on first load behind a spinner).
- Never reload model weights inside a rerun; never put heavy work in page render paths outside a button handler.
- Use `st.status` / `st.progress` for visible feedback during the 5-20 s generation.

## 5. Key Technical Choices & Trade-offs

| Choice | Alternative | Why |
|--------|-------------|-----|
| **SD 1.5 + ControlNet** | SDXL + ControlNet | SD 1.5 fits 8 GB VRAM, mature ControlNet ecosystem, faster iteration — trade-off: lower photorealism than SDXL |
| **Diffusers library** | CompVis original, InvokeAI | Cleanest Python API for scripting, matches course reference materials |
| **SUN RGB-D subset** | Full dataset, HM3D, Hypersim | Full dataset is ~20 GB; a curated subset is enough to demonstrate data-to-prompt mapping and ControlNet conditioning |
| **Depth + Canny ControlNet** | Segmentation ControlNet | SUN RGB-D has native depth maps, making depth conditioning free; Canny is cheap and adds a second comparison axis |
| **CLIP + LPIPS metrics** | FID, IS | FID needs a large reference distribution; CLIP+LPIPS are per-image and fit a class scale |
| **Streamlit web app + CLI** | CLI + notebooks only | Interactive web UI makes the demo video more compelling and shows the data-to-prompt flow live; Streamlit is one file with no build step so the cost is low. CLI preserved for reproducible grading runs. |

## 6. Control Strategy

**Prompt templates (examples, finalized in `configs/prompts.yaml`):**

- `minimal`: `"{roomType}, {style} style"`
- `descriptive`: `"A {style} {roomType}, {size}, featuring {furniture}, {lighting}, {mood}, interior design photography, 4k"`
- `styleAnchored`: descriptive + reference phrases for style (e.g., "inspired by IKEA catalog photography, muted palette, natural materials")

**Shared negative prompt:**
`"low quality, blurry, distorted proportions, extra furniture, cluttered, warped walls, text, watermark, people, cartoon"`

**ControlNet signals:**
- `depth` — from SUN RGB-D depth channel (preferred when reference image is provided)
- `canny` — from OpenCV Canny on the reference RGB (secondary comparison)

## 7. Evaluation Plan

| Metric | How | Signal |
|--------|-----|--------|
| Prompt alignment | CLIP score: cosine sim of CLIP text embedding vs image embedding | higher = better |
| Diversity | Mean pairwise LPIPS across same-spec, different-seed outputs | higher = more diverse |
| Consistency | Mean pairwise CLIP image-image sim within same spec | higher = more consistent |
| Quality (qualitative) | 1-5 self-rating CSV per image | higher = better |

**Comparisons to produce:**
1. Baseline (minimal prompt, no ControlNet) vs Improved (style-anchored prompt + depth ControlNet).
2. Controlled vs uncontrolled, same prompt.
3. Cross-strategy comparison on fixed room spec.
4. Failure-case gallery with notes.

## 8. Bonus (choose one, after core is done)

- **Image → short clip:** AnimateDiff or RunwayML Gen-3 API on top 3 generations.
- **Audio narration:** ElevenLabs TTS reading the room spec + caption.
- **Image → video:** Pika for a pan/zoom clip of the rendered room.

## 9. Tech Stack Summary

- **Language:** Python 3.11
- **Core:** PyTorch, diffusers, transformers, accelerate, controlnet-aux, opencv-python
- **Eval:** open_clip_torch, lpips, pandas, matplotlib
- **Web app:** streamlit, pillow
- **Runtime / hosting:** Google Colab Pro, `cloudflared` (quick tunnel), `pyngrok` (fallback), Google Drive (persistent storage)
- **CLI:** typer
- **Testing:** pytest
- **Notebooks:** Jupyter / Colab
- **AI tools disclosed:** Claude Code (planning + scaffolding), plus any others used during implementation
