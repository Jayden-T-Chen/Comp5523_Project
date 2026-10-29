#!/usr/bin/env python3
"""Render the failure-analysis figure: baseline vs attribute rerank, top-3.

One column per case, two rows: frozen-CLIP cosine on top, re-ranked list below.
Captions show rank, cosine similarity and (for rerank) the attribute agreement,
because the interesting failures sit where agreement and rank disagree: high
agreement on top of a bad image, or low agreement under a good one.

The five cases are fixed on purpose: each illustrates a failure mode the
report discusses.

The gallery ships as 480x640 JPEGs; the figure downsamples them to 200x267 per
thumb with LANCZOS and saves at 300 dpi - a larger canvas for slides without
blowing up the file size.

Usage:
  python evaluation/tools/renderFailureCases.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "retrieval"))

META_PATH = ROOT / "data" / "metadata.csv"
RESULTS_DIR = ROOT / "results"
OUT_PATH = ROOT / "evaluation" / "montages" / "failureCases.png"

# (qid, query text, one-line reading of the case)
CASES = [
    ("type_sports_sandals", "sports sandals", "taxonomy: sandal vs sports sandal"),
    ("color_grey", "grey shoes", "silver still nudges grey"),
    ("combo_beige_heels", "beige heels", "tan / khaki family, now with heels"),
    ("combo_bronze_heels", "bronze heels", "metallic sheen vs brown"),
    ("type_casual_shoes", "casual shoes", "control - rerank wins"),
]

TH_W, TH_H = 200, 267
PAD, CAP, GAP = 8, 62, 30
COL_W = 3 * TH_W + 2 * PAD
ROW_LABEL_W = 132
TOP = 128
QH = 62                                    # query text + failure-mode tag
ROW_H = TH_H + CAP
BOTTOM = 132
HEIGHT = TOP + QH + 24 + ROW_H + 40 + ROW_H + BOTTOM
WIDTH = 32 + ROW_LABEL_W + len(CASES) * COL_W + (len(CASES) - 1) * GAP + 32

FONT_CANDIDATES = {
    False: [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ],
    True: [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ],
}


def font(size: int, bold: bool = False):
    """A real TrueType face when the system has one, else Pillow's default."""
    for path in FONT_CANDIDATES[bold]:
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:                      # Pillow < 10.1
        return ImageFont.load_default()


def load_results() -> tuple[dict, dict]:
    def per_query(name: str) -> dict:
        path = RESULTS_DIR / name
        data = json.loads(path.read_text(encoding="utf-8"))
        return {row["qid"]: row for row in data["per_query"]}

    return per_query("baseline_auto.json"), per_query("rerank_auto.json")


def thumb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB").resize((TH_W, TH_H), Image.LANCZOS)


def render() -> Path:
    import clipCore
    import rerank

    base_res, rr_res = load_results()
    rows, paths = clipCore.load_rows()
    device = clipCore.get_device()
    model, preprocess, tokenizer = clipCore.load_model(device)
    feats = clipCore.get_image_features(model, preprocess, paths, device)
    by_id = {r["id"]: r for r in rows}

    canvas = Image.new("RGB", (WIDTH, HEIGHT), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    f_title, f_sub = font(40, True), font(22)
    f_q, f_tag = font(28, True), font(19)
    f_row, f_cap, f_meta = font(24, True), font(19), font(17)
    f_note = font(17)

    draw.text((32, 24),
              "Failure analysis: baseline (top) vs attribute rerank (bottom), top-3",
              fill=(17, 17, 17), font=f_title)
    draw.text((32, 78),
              "caption: rank, cosine, attribute agreement   |   "
              "AP is per query on auto labels, rerank depth 100",
              fill=(105, 105, 105), font=f_sub)

    band_y = TOP + QH + 12
    draw.rectangle([32, band_y, WIDTH - 32, band_y + ROW_H + 20],
                   fill=(246, 247, 249))
    draw.line([32, band_y, WIDTH - 32, band_y], fill=(225, 225, 228), width=1)
    draw.line([32, band_y + ROW_H + 20, WIDTH - 32, band_y + ROW_H + 20],
              fill=(225, 225, 228), width=1)

    for ci, (qid, label, tag) in enumerate(CASES):
        x0 = 32 + ROW_LABEL_W + ci * (COL_W + GAP)
        draw.text((x0, TOP), label, fill=(17, 17, 17), font=f_q)
        draw.text((x0, TOP + 34), tag, fill=(120, 120, 120), font=f_tag)

        hits = clipCore.search(model, tokenizer, feats, rows, label, 3, device)
        base = [(h["id"], h["score"], None) for h in hits]
        rr = [(d["id"], d["sim"], d["agree"])
              for d in rerank.rerank_detail(model, tokenizer, feats, rows,
                                            label, rerank.RERANK_DEPTH,
                                            device)[:3]]

        for ri, (row_label, items) in enumerate([("baseline", base), ("rerank", rr)]):
            y0 = TOP + QH + 24 + ri * (ROW_H + 40)
            mid = y0 + TH_H // 2
            draw.text((32, mid - 40), row_label, fill=(34, 34, 34), font=f_row)
            note = "cosine" if row_label == "baseline" else "cos + attr"
            draw.text((32, mid - 8), note, fill=(120, 120, 120), font=f_meta)
            if row_label == "rerank":
                draw.text((32, mid + 18), "depth 100", fill=(120, 120, 120),
                          font=f_meta)
            for k, (iid, score, agree) in enumerate(items):
                meta = by_id[iid]
                x = x0 + k * (TH_W + PAD)
                canvas.paste(thumb(ROOT / meta["path"]), (x, y0))
                draw.rectangle([x, y0, x + TH_W - 1, y0 + TH_H - 1],
                               outline=(198, 198, 202))
                cap = f"{k + 1}   cos {score:.2f}"
                if agree is not None:
                    cap += f"   a {agree:.2f}"
                draw.text((x + 2, y0 + TH_H + 5), cap, fill=(17, 17, 17),
                          font=f_cap)
                draw.text((x + 2, y0 + TH_H + 30),
                          f"{meta['color']} {meta['articleType']}"[:26],
                          fill=(105, 105, 105), font=f_meta)

        delta = rr_res[qid]["ap50"] - base_res[qid]["ap50"]
        colour = (176, 0, 32) if delta < 0 else (11, 110, 46)
        draw.text((x0, HEIGHT - 104),
                  f"AP {base_res[qid]['ap50']:.2f} -> {rr_res[qid]['ap50']:.2f}"
                  f"  ({delta:+.2f})", fill=colour, font=font(22, True))
        draw.text((x0, HEIGHT - 70),
                  f"P@10 {base_res[qid]['p10']:.2f} -> {rr_res[qid]['p10']:.2f}",
                  fill=(105, 105, 105), font=f_cap)

    draw.text((32, HEIGHT - 30),
              "AP delta in red = rerank lowers AP, green = rerank raises it.  "
              "Auto labels are metadata-derived, so case 1 mixes a label "
              "problem with the ranking problem.",
              fill=(140, 140, 140), font=f_note)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT_PATH, dpi=(300, 300))
    return OUT_PATH


if __name__ == "__main__":
    print(render())
