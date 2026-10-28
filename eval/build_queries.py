#!/usr/bin/env python3
"""Build eval/queries.json: 60 auto-labelled queries + 20 marked for strict judging.

Labels come from the metadata attributes (color / articleType / gender), so they
are noisy: a product's `color` is the colour in its marketing name, not what the
thumbnail looks like. That noise is measured, not hidden -- the `strict` flag
marks queries whose top-10 will be judged by eye and reported as a separate
column.

Groups (60 total):
  color       20  "black shoes"                  -> every Black image
  type        18  "heels" / "casual shoes for men"
  combo       22  "white sports shoes"           -> color x articleType

Deterministic: counts + names decide everything, no RNG.

Usage:
  python eval/build_queries.py
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
META_PATH = ROOT / "data" / "metadata.csv"
OUT_PATH = ROOT / "eval" / "queries.json"

MIN_REL_COMBO = 12      # a color x type query needs this many images to be worth asking
MIN_REL_GENDER = 20     # type+"for men/women" needs this many
N_COLOR, N_TYPE, N_COMBO = 20, 18, 22

COLOR_TEXT = {"Multi": "multi colored"}
TYPE_TEXT = {
    "Casual Shoes": "casual shoes",
    "Sports Shoes": "sports shoes",
    "Formal Shoes": "formal shoes",
    "Flip Flops": "flip flops",
    "Sports Sandals": "sports sandals",
    "Heels": "heels",
    "Flats": "flats",
    "Sandals": "sandals",
}
STRICT_IDS = [
    # covered while judging top-10 by eye (spread across groups)
    "color_black", "color_white", "color_red", "color_blue",
    "color_brown", "color_silver",
    "type_heels", "type_sandals", "type_casual_shoes", "type_formal_shoes",
    "type_sports_shoes_women", "type_flip_flops_men",
    "combo_white_sports_shoes", "combo_silver_sports_shoes",
    "combo_olive_casual_shoes", "combo_khaki_casual_shoes",
    "combo_grey_sports_shoes", "combo_beige_casual_shoes",
    "combo_bronze_heels", "combo_red_casual_shoes",
]


def load_rows() -> list[dict]:
    with META_PATH.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def slug(*parts: str) -> str:
    """'White' + 'Sports Shoes' -> 'white_sports_shoes'."""
    return "_".join(p.lower().replace(" ", "_").replace("-", "_") for p in parts)


def make(qid: str, query: str, group: str, attrs: dict, rows: list[dict]) -> dict:
    rel = sorted(
        r["id"] for r in rows if all(r[k] == v for k, v in attrs.items())
    )
    return {
        "qid": qid,
        "query": query,
        "group": group,
        "attrs": attrs,
        "n_relevant": len(rel),
        "relevant": rel,
        "strict": qid in STRICT_IDS,
    }


def main() -> None:
    rows = load_rows()
    queries: list[dict] = []

    # --- color only -------------------------------------------------------
    colors = sorted(Counter(r["color"] for r in rows).items(),
                    key=lambda kv: (-kv[1], kv[0]))[:N_COLOR]
    for color, _n in colors:
        text = COLOR_TEXT.get(color, color.lower())
        queries.append(
            make(f"color_{slug(color)}", f"{text} shoes", "color",
                 {"color": color}, rows)
        )

    # --- articleType only, plus gender-restricted variants ----------------
    types = sorted(Counter(r["articleType"] for r in rows).items(),
                   key=lambda kv: (-kv[1], kv[0]))
    for atype, _n in types:
        text = TYPE_TEXT.get(atype, atype.lower())
        queries.append(
            make(f"type_{slug(atype)}", text, "type", {"articleType": atype}, rows)
        )

    gender_types = sorted(
        ((g, t), n) for (g, t), n in Counter(
            (r["gender"], r["articleType"]) for r in rows).items()
        if n >= MIN_REL_GENDER
    )
    gender_types.sort(key=lambda gt: (-gt[1], gt[0]))
    for (gender, atype), _n in gender_types:
        text = TYPE_TEXT.get(atype, atype.lower())
        who = gender.lower()
        queries.append(
            make(f"type_{slug(atype)}_{who}", f"{text} for {who}", "type",
                 {"articleType": atype, "gender": gender}, rows)
        )

    # --- color x articleType ---------------------------------------------
    combos = sorted(
        ((c, t), n) for (c, t), n in Counter(
            (r["color"], r["articleType"]) for r in rows).items()
        if n >= MIN_REL_COMBO
    )
    combos.sort(key=lambda ct: (-ct[1], ct[0]))
    for (color, atype), _n in combos[:N_COMBO]:
        ctext = COLOR_TEXT.get(color, color.lower())
        ttext = TYPE_TEXT.get(atype, atype.lower())
        queries.append(
            make(f"combo_{slug(color, atype)}", f"{ctext} {ttext}", "combo",
                 {"color": color, "articleType": atype}, rows)
        )

    known = {q["qid"] for q in queries}
    unknown = [s for s in STRICT_IDS if s not in known]
    if unknown:
        raise SystemExit(f"STRICT_IDS not generated: {unknown}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(
            {
                "note": "auto labels from data/metadata.csv attributes; "
                        "strict=True marks queries judged by eye (top-10)",
                "queries": queries,
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    by_group = Counter(q["group"] for q in queries)
    print(f"wrote {len(queries)} queries -> {OUT_PATH}")
    print("groups      :", dict(by_group))
    print("strict      :", sum(q["strict"] for q in queries))
    print("min rel/img :", min(q["n_relevant"] for q in queries),
          "max:", max(q["n_relevant"] for q in queries))
    for q in queries:
        if q["strict"]:
            print(f"  [strict] {q['qid']:<32} {q['query']:<34} rel={q['n_relevant']}")


if __name__ == "__main__":
    main()
