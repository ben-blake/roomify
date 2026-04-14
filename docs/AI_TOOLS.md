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

*Append a new entry to the session log for each session that uses AI assistance.*
