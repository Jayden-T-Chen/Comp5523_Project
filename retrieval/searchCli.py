"""Command-line ranking over the footwear gallery.

Reached through `python main.py search`; see main.py for the flags.
"""
from __future__ import annotations

from . import rerank
from .clipCore import (
    get_device,
    get_image_features,
    load_model,
    load_rows,
    search,
)

SAMPLE_QUERIES = [
    "black leather shoes",
    "white sneakers",
    "red sandals",
    "blue flip flops",
    "brown heels",
]


def add_args(sub) -> None:
    sub.add_argument("--query", action="append", dest="queries",
                     help="repeatable; five built-in samples when omitted")
    sub.add_argument("--top", type=int, default=8)
    sub.add_argument("--method", choices=["baseline", "rerank"],
                     default="baseline")


def main(args) -> None:
    queries = args.queries or SAMPLE_QUERIES

    rows, paths = load_rows()
    device = get_device()
    print(f"device: {device}  gallery: {len(rows)} images  "
          f"method: {args.method}")

    model, preprocess, tokenizer = load_model(device)
    img_feats = get_image_features(model, preprocess, paths, device)

    for q in queries:
        print(f"\n=== Query: {q!r} ===")
        if args.method == "rerank":
            hits = rerank.rerank_detail(model, tokenizer, img_feats, rows,
                                        q, max(args.top, rerank.RERANK_DEPTH),
                                        device)[:args.top]
            for r in hits:
                agree = "  -   " if r["agree"] is None else f"{r['agree']:.2f}"
                print(
                    f"  {r['rank']}. cos {r['sim']:.3f}  attr {agree}  "
                    f"[{r['type']:<14}] {r['color']:<10} {r['title']}"
                )
            continue
        for r in search(model, tokenizer, img_feats, rows, q, args.top, device):
            print(
                f"  {r['rank']}. {r['score']:.3f}  "
                f"[{r['type']:<14}] {r['color']:<10} {r['title']}"
            )
