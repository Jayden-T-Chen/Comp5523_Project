#!/usr/bin/env python3
"""CLI: rank gallery images for text queries with CLIP.

Usage:
  python retrieval/searchCli.py
  python retrieval/searchCli.py --query "black leather shoes" --top 8
"""
from __future__ import annotations

import argparse

from clipCore import (
    get_device,
    get_image_features,
    load_model,
    load_rows,
    search,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="CLIP text->image ranking")
    ap.add_argument("--query", action="append", dest="queries")
    ap.add_argument("--top", type=int, default=8)
    args = ap.parse_args()

    queries = args.queries or [
        "black leather shoes",
        "white sneakers",
        "red sandals",
        "blue flip flops",
        "brown heels",
    ]

    rows, paths = load_rows()
    device = get_device()
    print(f"device: {device}  gallery: {len(rows)} images")

    model, preprocess, tokenizer = load_model(device)
    img_feats = get_image_features(model, preprocess, paths, device)

    for q in queries:
        print(f"\n=== Query: {q!r} ===")
        for r in search(model, tokenizer, img_feats, rows, q, args.top, device):
            print(
                f"  {r['rank']}. {r['score']:.3f}  "
                f"[{r['type']:<14}] {r['color']:<10} {r['title']}"
            )


if __name__ == "__main__":
    main()
