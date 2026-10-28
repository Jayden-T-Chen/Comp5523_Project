"""Shared CLIP retrieval core.

Loads the gallery from data/metadata.csv, embeds it with CLIP ViT-B/32
(open_clip, "openai" weights), caches features, and ranks images for a
text query. Cross-platform: macOS (mps) / Windows (cuda/cpu).
"""
from __future__ import annotations

import csv
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
META_PATH = ROOT / "data" / "metadata.csv"
CACHE_PATH = ROOT / "data" / "image_features.pt"
MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"
BATCH = 32

# Product thumbnails are catalogue shots on a light background, but the queries
# are terse shopper phrases. Averaging a few phrasings pulls the text anchor
# towards the caption style CLIP learned to pair with such images.
PROMPT_TEMPLATES = (
    "{q}",
    "a photo of {q}",
    "a product photo of {q}",
    "a pair of {q}",
    "a {q} on a white background",
    "a close-up photo of {q}",
)


def get_device() -> str:
    if torch.backends.mps.is_available():      # Apple Silicon
        return "mps"
    if torch.cuda.is_available():              # NVIDIA on Windows
        return "cuda"
    return "cpu"


def load_rows() -> tuple[list[dict], list[Path]]:
    if not META_PATH.is_file():
        raise SystemExit(f"Missing {META_PATH}. Run scripts/download_footwear.py first.")
    rows = list(csv.DictReader(META_PATH.open(encoding="utf-8")))
    paths = [ROOT / r["path"] for r in rows]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise SystemExit(f"Missing {len(missing)} images (first: {missing[0]})")
    return rows, paths


def load_model(device: str):
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED
    )
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    return model.to(device).eval(), preprocess, tokenizer


def build_image_features(model, preprocess, paths, device) -> torch.Tensor:
    from PIL import Image

    feats = []
    with torch.no_grad():
        for i in range(0, len(paths), BATCH):
            batch = [preprocess(Image.open(p).convert("RGB")) for p in paths[i : i + BATCH]]
            x = torch.stack(batch).to(device)
            f = model.encode_image(x)
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.cpu())
    return torch.cat(feats, dim=0)


def get_image_features(model, preprocess, paths, device) -> torch.Tensor:
    """Return cached features if present and newer than metadata, else rebuild."""
    if CACHE_PATH.is_file() and CACHE_PATH.stat().st_mtime >= META_PATH.stat().st_mtime:
        blob = torch.load(CACHE_PATH, map_location="cpu")
        if blob.get("n") == len(paths):
            return blob["feats"]
    feats = build_image_features(model, preprocess, paths, device)
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"n": len(paths), "feats": feats}, CACHE_PATH)
    return feats


def embed_query(model, tokenizer, query: str, device) -> torch.Tensor:
    with torch.no_grad():
        tok = tokenizer([query]).to(device)
        t = model.encode_text(tok)
        t = t / t.norm(dim=-1, keepdim=True)
    return t.cpu()


def embed_query_ensemble(model, tokenizer, query: str, device,
                         templates=PROMPT_TEMPLATES) -> torch.Tensor:
    """Mean of several prompt phrasings, re-normalised, shape [1, d]."""
    texts = [t.format(q=query) for t in templates]
    with torch.no_grad():
        tok = tokenizer(texts).to(device)
        t = model.encode_text(tok)
        t = t / t.norm(dim=-1, keepdim=True)
        t = t.mean(dim=0, keepdim=True)
        t = t / t.norm(dim=-1, keepdim=True)
    return t.cpu()


def search(model, tokenizer, img_feats, rows, query, topk, device) -> list[dict]:
    q = embed_query(model, tokenizer, query, device)
    scores = (img_feats @ q.T).squeeze(1)
    k = min(topk, len(rows))
    vals, idxs = torch.topk(scores, k)
    out = []
    for rank, (idx, score) in enumerate(zip(idxs.tolist(), vals.tolist()), 1):
        r = rows[idx]
        out.append(
            {
                "rank": rank,
                "id": r["id"],
                "score": score,
                "path": str(ROOT / r["path"]),
                "title": r.get("display_name", ""),
                "type": r.get("articleType", ""),
                "color": r.get("color", ""),
            }
        )
    return out
