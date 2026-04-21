# Demo Video Script

**Duration:** 90 seconds  
**Format:** Screen recording with voiceover  
**Platform:** Google Colab Pro (record on a live session)

---

## Pre-recording checklist

- [ ] Colab session running, all cells already executed
- [ ] Streamlit app loaded and pre-warmed (tunnel URL live in Cell 6 output)
- [ ] Browser open on the Generate page
- [ ] Spec form pre-filled: `bedroom_01 · descriptive · seed 42` (so first generation is instant)
- [ ] `examples/phase7/strategy-compare/strategy_compare_bedroom_01_s42.png` open in a second tab as backup
- [ ] `examples/phase7/controlled-vs-uncontrolled/comparison_bedroom_01_s42.png` open in a third tab as backup

---

## Script

### [0:00–0:20] Intro (20 s)

**Screen:** GitHub README open in browser, showing the Open-in-Colab badge

> "Roomify is a data-driven interior design image generator built for UMKC CS 5542.
> You give it a structured room description — room type, dimensions, style, furniture,
> lighting — and it generates photorealistic renders using Stable Diffusion 1.5
> with ControlNet depth conditioning.
> Everything runs on Google Colab Pro. Here's how it works."

---

### [0:20–0:35] Launcher notebook → tunnel URL (15 s)

**Screen:** Colab tab showing `00_launchColab.ipynb` with all cells already run. Scroll to Cell 6 output showing the `trycloudflare.com` URL.

> "The launcher notebook mounts Google Drive, clones the repo, installs dependencies,
> and starts the Streamlit app. Cell 6 opens a Cloudflare tunnel and prints the public URL.
> Click it — and the app is live."

**Action:** Click the `trycloudflare.com` URL. Browser opens the Streamlit Generate page.

---

### [0:35–1:10] Web app walkthrough + live generation (35 s)

**Screen:** Streamlit Generate page. Spec form already filled with `bedroom_01 · descriptive · seed 42`.

> "The Generate page has a spec form — room type, size, style, furniture, lighting, mood.
> I'll use a Scandinavian bedroom spec with the descriptive prompt strategy."

**Action:** Click **Generate**. Spinner appears.

> "The pipeline is pre-warmed, so generation takes about 8 seconds on A100."

*(wait for image)*

> "There's the render. I can hit Generate Variant to get a new seed side-by-side —"

**Action:** Click **Generate Variant**. Second image appears.

> "— and the variants stack so you can compare them directly."

**Action:** Briefly show the **Experiments** tab — YAML picker and Run Sweep button visible. Do not click.

> "The Experiments page runs full sweeps across all specs, strategies, and control modes,
> with a live progress bar."

---

### [1:10–1:30] Results + evaluation (20 s)

**Screen:** `examples/phase7/strategy-compare/strategy_compare_bedroom_01_s42.png` (open in browser tab or shown in Gallery).

> "Here are the three prompt strategies on the same spec and seed.
> Minimal on the left — descriptive in the middle — style-anchored on the right.
> Richer prompts produce more composed, coherent renders."

**Action:** Switch to `examples/phase7/controlled-vs-uncontrolled/comparison_bedroom_01_s42.png`.

> "And here's ControlNet depth conditioning — the depth map from SUN RGB-D constrains
> the room layout. ControlNet improves spatial fidelity but lowers CLIP text-image alignment.
> Across 90 images the mean CLIP score was 0.274 and LPIPS diversity 0.758."

---

### [1:30–1:40] Conclusion (10 s)

**Screen:** GitHub README, showing the Open-in-Colab badge and repo URL.

> "The full pipeline — dataset, prompt builder, SD 1.5, ControlNet, evaluation,
> and Streamlit UI — is open source at github.com/ben-blake/roomify.
> Click Open in Colab to run it yourself."

---

## Recording notes

- **Resolution:** 1920×1080, Colab in default browser zoom
- **Cursor:** move deliberately to what you are narrating; avoid hovering over unrelated UI
- **Pacing:** the generation section has the most slack — if inference takes longer than 8 s on the day, let the spinner run silently while you pause the voiceover
- **Tunnel hiccup fallback:** if the Cloudflare URL is unreachable, cut to the pre-opened comparison PNG for the generation section — the narration still holds
- **Upload:** YouTube unlisted; paste the URL into `README.md` and mark the Phase 9 demo task complete in `docs/TASKS.md`
