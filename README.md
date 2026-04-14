# Roomify

**Data-driven interior design image generator** — UMKC CS 5542 Quiz Challenge-1

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ben-blake/roomify/blob/main/notebooks/00_launchColab.ipynb)

---

## What it does

Roomify takes a structured room description (type, dimensions, style, furniture list, lighting, mood) and generates interior design renders using **Stable Diffusion 1.5** with explicit control via **ControlNet** (depth + Canny edge maps from the SUN RGB-D dataset).

Three prompt strategies — `minimal`, `descriptive`, `styleAnchored` — are compared against a shared baseline, and results are evaluated with CLIP alignment, LPIPS diversity, and qualitative ratings.

---

## Quick start (Google Colab — recommended)

1. Click the **Open in Colab** badge above.
2. Run all cells in `notebooks/00_launchColab.ipynb` top to bottom.
   - Enable widget cell enables third-party widget support.
   - Cell 1 mounts your Google Drive and creates the folder structure.
   - Cell 2 clones this repo and installs dependencies (~3 min on first run).
   - Cell 2b **(first run only)** unzips SUN RGB-D and builds the 200-sample subset.
   - Cell 3 sets `HF_HOME` so model weights persist across Colab sessions.
   - Cell 4 verifies your GPU with `nvidia-smi`.
   - Cell 5 loads SD 1.5 and runs a smoke-test generation (skips if already done).
   - Cell 6 starts the Streamlit web app and opens a Cloudflare tunnel.
   - Cell 7 is a reconnect helper for after a Colab disconnect.
3. Click the `trycloudflare.com` URL printed by Cell 6 to open the web app.

**First-run time budget:** ~5–8 min (model download + warmup). Subsequent sessions re-use the Drive-cached weights and start in < 1 min.

> **Note:** Each notebook in this repo has its own self-contained setup cell — Colab runtimes are isolated per notebook tab.

---

## Local development

```bash
git clone https://github.com/ben-blake/roomify.git
cd roomify
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run the pure-Python tests (no GPU required)
pytest tests/ -v

# Launch the CLI help
python -m roomify.cli --help
```

> The Streamlit UI and SD pipeline require a GPU (or Colab). Local dev is for
> pure-Python modules only: dataset loader, prompt builder, evaluation metrics.

---

## CLI reference

```bash
# Generate one image from a room spec
python -m roomify.cli generate --spec configs/examples/bedroom_01.yaml

# Run a full experiment sweep
python -m roomify.cli sweep --config configs/experiments/core.yaml

# Compute CLIP + LPIPS metrics over a run directory
python -m roomify.cli evaluate --run outputs/<runId>

# Render contact sheet + markdown metrics table
python -m roomify.cli report --run outputs/<runId>
```

---

## Dataset

**SUN RGB-D** — a large-scale dataset of indoor RGB-D images with scene labels and object annotations.

- **Download URL:** https://rgbd.cs.princeton.edu/ — click "SUNRGBD V1" to get the full ~2.6 GB zip
- Full dataset: ~10,000 images (not committed; too large for GitHub)
- Curated subset: ~200 images across 5 scene types (bedroom, living_room, kitchen, office, bathroom), even distribution
- Location in Colab: `/content/drive/MyDrive/roomify/data/SUNRGBD` (raw) and `/content/drive/MyDrive/roomify/data/sunrgbd_subset` (curated)
- Manifest: `data/sunrgbd_subset/manifest.csv` (on Drive; not committed)

**One-time setup** (run once per Google Drive account):

```bash
# In Colab, after mounting Drive:
python scripts/buildSubset.py \
    --sunrgbd-root /content/drive/MyDrive/roomify/data/SUNRGBD \
    --output-dir   /content/drive/MyDrive/roomify/data/sunrgbd_subset \
    --samples-per-scene 40 \
    --copy
```

The subset is cached to Google Drive. Subsequent Colab sessions load directly from Drive — no re-download needed.

---

## Runtime notes

- **Platform:** Google Colab Pro (T4 / L4 / A100 depending on availability)
- **Public URL:** Cloudflare quick tunnel (`trycloudflare.com`) — no auth, no signup. URL changes every Colab session.
- **Persistence:** All generated outputs and HF model weights live under `/content/drive/MyDrive/roomify/` and survive Colab disconnects.
- **GPU logged:** every `run.json` records the actual GPU detected by `nvidia-smi`.

---

## Project structure

```
app.py                       # Streamlit entrypoint (thin shell)
src/roomify/
  dataset.py                 # SUN RGB-D manifest + Record schema
  promptBuilder.py           # RoomSpec + strategy → (positive, negative)
  controlSignals.py          # depth / Canny extraction for ControlNet
  pipeline.py                # SD 1.5 + ControlNet wrapper, cached singleton
  orchestrator.py            # sweep runner, writes run.json per image
  evaluation.py              # CLIP alignment, LPIPS diversity, consistency
  reporting.py               # contact sheets, markdown metric tables
  ui/                        # Streamlit pages + reusable components
  cli.py                     # typer CLI
notebooks/
  00_launchColab.ipynb       # PRIMARY: clone → install → streamlit + tunnel
configs/
  prompts.yaml               # strategy templates + shared negative prompt
  examples/                  # sample RoomSpec YAMLs
  experiments/               # sweep configuration YAMLs
tests/                        # pytest (pure-Python modules only)
docs/
  PRD.md                     # Product Requirements Document
  ARCHITECTURE.md            # System architecture + data contracts
  TASKS.md                   # Phased task list with exit criteria
  AI_TOOLS.md                # AI tool usage disclosure (course requirement)
```

---

## AI tools disclosure

This project uses AI tools. See [docs/AI_TOOLS.md](docs/AI_TOOLS.md) for the full disclosure required by course policy.

---

## License

For academic use — UMKC CS 5542.
