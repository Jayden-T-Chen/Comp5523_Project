#!/usr/bin/env python3
"""Render the top-K hits of one or more queries into a PNG for by-eye judging.

Each cell shows the image plus rank, gallery id, metadata colour and
articleType, so a judgment can be written as pure relevance: does this image
actually match the query?  Several queries stack vertically in one file.

Usage:
  python evaluation/renderTopHits.py --qid color_black
  python evaluation/renderTopHits.py --qid color_black color_white --out /tmp/pair.png
  python evaluation/renderTopHits.py --all-strict
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "retrieval"))

META_PATH = ROOT / "data" / "metadata.csv"
QUERIES_PATH = ROOT / "evaluation" / "queries.json"
OUT_DIR = ROOT / "evaluation" / "montages"

COLS = 5
CELL_W, CELL_H = 200, 240
IMG_W, IMG_H = 150, 200
PAD = 6
TITLE_H = 34


def load_meta() -> dict[str, dict]:
    with META_PATH.open(encoding="utf-8", newline="") as f:
        return {r["id"]: r for r in csv.DictReader(f)}


def font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:                      # Pillow < 10.1
        return ImageFont.load_default()


def render(qid_query: list[tuple[str, str]], topk: int) -> Path:
    import clipCore

    meta = load_meta()
    rows, paths = clipCore.load_rows()
    device = clipCore.get_device()
    model, preprocess, tokenizer = clipCore.load_model(device)
    feats = clipCore.get_image_features(model, preprocess, paths, device)

    per_q = len(qid_query)
    rows_h = (CELL_H + PAD) * 2 + TITLE_H
    W = COLS * CELL_W + PAD * (COLS + 1)
    H = rows_h * per_q
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    f = font(13)
    f_title = font(18)

    for qi, (qid, query) in enumerate(qid_query):
        base_y = qi * rows_h
        draw.text((PAD, base_y + 6), f"{qid}   “{query}”", fill=(0, 0, 0), font=f_title)
        hits = clipCore.search(model, tokenizer, feats, rows, query, topk, device)
        for i, hit in enumerate(hits[:COLS * 2]):
            r, c = divmod(i, COLS)
            x = PAD + c * (CELL_W + PAD)
            y = base_y + TITLE_H + r * (CELL_H + PAD)
            m = meta.get(hit["id"], {})
            img = Image.open(hit["path"]).convert("RGB").resize((IMG_W, IMG_H))
            canvas.paste(img, (x, y))
            draw.rectangle([x - 1, y - 1, x + IMG_W, y + IMG_H], outline=(190, 190, 190))
            draw.text((x, y + IMG_H + 3), f"#{hit['rank']}  id {hit['id']}",
                      fill=(0, 0, 0), font=f)
            draw.text((x, y + IMG_H + 19),
                      f"{m.get('color', '?')} / {m.get('articleType', '?')[:13]}",
                      fill=(60, 60, 60), font=f)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{'-'.join(q for q, _ in qid_query)}.png"
    canvas.save(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qid", nargs="+", default=None)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--all-strict", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    data = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))
    by_qid = {q["qid"]: q for q in data["queries"]}

    if args.all_strict:
        picks = [(q["qid"], q["query"]) for q in data["queries"] if q["strict"]]
        for qid, query in picks:
            print(render([(qid, query)], args.top))
        return

    if not args.qid:
        raise SystemExit("give --qid (or --all-strict)")
    unknown = [q for q in args.qid if q not in by_qid]
    if unknown:
        raise SystemExit(f"unknown qid: {unknown}")
    pairs = [(q, by_qid[q]["query"]) for q in args.qid]
    out = render(pairs, args.top)
    if args.out:
        out.replace(args.out)
        out = args.out
    print(out)


if __name__ == "__main__":
    main()
