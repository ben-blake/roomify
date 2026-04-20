# Phase 7 Evaluation Metrics

**Sweep:** `core_comparison` — 90 images (5 specs × 3 strategies × 2 ctrl/unctrl × 3 seeds)  
**Run date:** 2026-04-20  
**GPU:** A100-SXM4-80GB  
**ControlNet conditioning scale:** 1.0

---

## Aggregate Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| Mean CLIP alignment score | 0.2739 | Text-image cosine similarity (CLIP ViT-B-32) |
| LPIPS diversity | 0.7583 | Mean pairwise perceptual distance — higher = more variety |
| Style consistency | 0.5431 | Mean pairwise CLIP image-image similarity — higher = more coherent |

---

## Top 10 by CLIP Score

| Rank | Run | Strategy | Controlled | CLIP Score |
|------|-----|----------|------------|------------|
| 1 | kitchen_01_styleAnchored_unctrl_s123 | styleAnchored | No | 0.3529 |
| 2 | bathroom_01_descriptive_unctrl_s7 | descriptive | No | 0.3500 |
| 3 | kitchen_01_descriptive_unctrl_s123 | descriptive | No | 0.3448 |
| 4 | bathroom_01_styleAnchored_unctrl_s123 | styleAnchored | No | 0.3445 |
| 5 | bedroom_01_styleAnchored_unctrl_s123 | styleAnchored | No | 0.3438 |
| 6 | office_01_minimal_unctrl_s42 | minimal | No | 0.3435 |
| 7 | bathroom_01_styleAnchored_unctrl_s7 | styleAnchored | No | 0.3416 |
| 8 | bathroom_01_styleAnchored_ctrl_s123 | styleAnchored | Yes | 0.3402 |
| 9 | bathroom_01_descriptive_unctrl_s123 | descriptive | No | 0.3357 |
| 10 | living_room_01_descriptive_unctrl_s123 | descriptive | No | 0.3354 |

---

## Key Findings

- **Uncontrolled images dominate the top CLIP scores** — 9 of the top 10 are uncontrolled. Depth ControlNet conditioning trades prompt-image alignment for spatial structure, which is expected.
- **styleAnchored and descriptive strategies outperform minimal** — richer prompts produce higher CLIP alignment, confirming prompt engineering has measurable impact.
- **High LPIPS diversity (0.7583)** — the system generates visually distinct outputs across seeds and strategies, demonstrating meaningful variation rather than mode collapse.
- **Moderate style consistency (0.5431)** — reasonable coherence within a sweep; style varies more across room types and strategies than across seeds.

---

## Output Files

| File | Description |
|------|-------------|
| `contact_sheet.png` | Full 90-image sweep grid (thumbnail overview) |
| `controlled-vs-uncontrolled/comparison_*.png` | Side-by-side ctrl vs unctrl for all 5 specs × 3 seeds |
| `strategy-compare/strategy_compare_*_s42.png` | minimal vs descriptive vs styleAnchored for bedroom, bathroom, kitchen |
| `top-exports/*.png` | Top 6 images by CLIP score (slide-ready) |
