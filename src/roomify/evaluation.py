"""Evaluation metrics for generated images — Phase 7."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

_RATINGS_FILE = "ratings.csv"
_RATING_COLUMNS = ["runId", "imagePath", "rating", "notes"]


def _loadClipModel():
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval()
    return model, preprocess, tokenizer


def clipAlignment(runDir: Path) -> pd.DataFrame:
    """Compute CLIP text-image cosine similarity for each image in *runDir*.

    Returns a DataFrame with columns: imagePath, prompt, clipScore.
    """
    import torch
    from PIL import Image as PILImage

    model, preprocess, tokenizer = _loadClipModel()

    records = []
    for run_json in sorted(Path(runDir).rglob("run.json")):
        data = json.loads(run_json.read_text())
        img_path = run_json.parent / "img_0.png"
        if not img_path.exists():
            continue

        img = preprocess(PILImage.open(img_path)).unsqueeze(0)
        tokens = tokenizer([data["prompt"]])

        with torch.no_grad():
            img_feat = model.encode_image(img)
            txt_feat = model.encode_text(tokens)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
            score = float((img_feat @ txt_feat.T).item())

        records.append({
            "imagePath": str(img_path),
            "prompt": data["prompt"],
            "clipScore": score,
        })

    return pd.DataFrame(records, columns=["imagePath", "prompt", "clipScore"])


def lpipsDiversity(runDir: Path) -> float:
    """Compute mean pairwise LPIPS distance across images in *runDir*.

    Higher = more diverse.  Returns 0.0 when fewer than 2 images are found.
    """
    import numpy as np
    import torch
    import lpips
    from PIL import Image as PILImage

    loss_fn = lpips.LPIPS(net="alex")

    tensors = []
    for run_json in sorted(Path(runDir).rglob("run.json")):
        img_path = run_json.parent / "img_0.png"
        if not img_path.exists():
            continue
        arr = np.array(
            PILImage.open(img_path).resize((256, 256)).convert("RGB"),
            dtype=np.float32,
        ) / 127.5 - 1.0
        arr = arr.transpose(2, 0, 1)[None]  # HWC → BCHW
        tensors.append(torch.tensor(arr))

    if len(tensors) < 2:
        return 0.0

    dists = []
    with torch.no_grad():
        for i in range(len(tensors)):
            for j in range(i + 1, len(tensors)):
                d = loss_fn(tensors[i], tensors[j]).item()
                dists.append(d)

    return float(sum(dists) / len(dists))


def styleConsistency(runDir: Path) -> float:
    """Compute mean pairwise CLIP image-image similarity within *runDir*.

    Higher = more consistent.  Returns 1.0 when fewer than 2 images are found.
    """
    import torch
    from PIL import Image as PILImage

    model, preprocess, _ = _loadClipModel()

    feats = []
    for run_json in sorted(Path(runDir).rglob("run.json")):
        img_path = run_json.parent / "img_0.png"
        if not img_path.exists():
            continue
        img = preprocess(PILImage.open(img_path)).unsqueeze(0)
        with torch.no_grad():
            feat = model.encode_image(img)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            feats.append(feat)

    if len(feats) < 2:
        return 1.0

    sims = []
    for i in range(len(feats)):
        for j in range(i + 1, len(feats)):
            s = float((feats[i] @ feats[j].T).item())
            sims.append(s)

    return float(sum(sims) / len(sims))


def saveRating(runDir: Path, runId: str, rating: int, notes: str = "") -> None:
    """Save or update a 1-5 star rating for *runId* under *runDir*/ratings.csv."""
    if not 1 <= rating <= 5:
        raise ValueError(f"rating must be 1-5, got {rating}")

    csv_path = Path(runDir) / _RATINGS_FILE
    if csv_path.exists():
        df = pd.read_csv(csv_path, dtype={"notes": str})
        df["notes"] = df["notes"].fillna("")
    else:
        df = pd.DataFrame(columns=_RATING_COLUMNS)

    mask = df["runId"] == runId
    img_path = str(Path(runDir) / runId / "img_0.png")
    if mask.any():
        df.loc[mask, "rating"] = rating
        df.loc[mask, "notes"] = notes
    else:
        new_row = pd.DataFrame([{
            "runId": runId,
            "imagePath": img_path,
            "rating": rating,
            "notes": notes,
        }])
        df = pd.concat([df, new_row], ignore_index=True)

    df.to_csv(csv_path, index=False)


def loadRatings(runDir: Path) -> pd.DataFrame:
    """Load ratings.csv from *runDir*.  Returns empty DataFrame if absent."""
    csv_path = Path(runDir) / _RATINGS_FILE
    if not csv_path.exists():
        return pd.DataFrame(columns=_RATING_COLUMNS)
    return pd.read_csv(csv_path)
