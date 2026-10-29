#!/usr/bin/env python3
"""Ablate the parser on the 60 auto queries (docs/projectReport.md 6.3).

Four configurations, one process, one gallery:
  legacy       no phrase masking, every alias value at full weight
  mask         phrase masking only
  color 0.5    masking + colour-family values at half weight (shipped)
  color 0.25   masking + colour-family values at quarter weight

Type aliases stay at weight 1.0 throughout: they are true synonyms, not
approximate colours. Beta and the candidate depth are held at the shipped
values so the only variable is the parser.

Usage:
  python evaluation/tools/ablateParser.py
  python evaluation/tools/ablateParser.py --out results/parser_ablation.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "retrieval"))
sys.path.insert(0, str(ROOT / "evaluation" / "tools"))

import queryParse                                  # noqa: E402
import rerank                                      # noqa: E402
import runEvaluation as ev                         # noqa: E402

QUERIES_PATH = ROOT / "evaluation" / "sets" / "queries.json"
RESULTS_DIR = ROOT / "results"
CONFIGS = [
    ("legacy", False, 1.0),
    ("mask", True, 1.0),
    ("color 0.5", True, 0.5),
    ("color 0.25", True, 0.25),
]
CASES = ["color_grey", "color_beige", "color_olive", "color_orange",
         "combo_beige_heels", "type_sports_sandals"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Parser ablation")
    ap.add_argument("--out", type=Path,
                    default=RESULTS_DIR / "parser_ablation.json")
    args = ap.parse_args()

    data = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))
    queries = [q for q in data["queries"] if q["relevant"]]
    ev.load_context()

    records = []
    header = f"{'config':<12}{'mAP@50':>8}{'P@10':>7}{'R@10':>7}"
    header += "".join(f"{q.split('_', 1)[1][:11]:>12}" for q in CASES)
    print(header)
    for name, mask, color_w in CONFIGS:
        queryParse.MASK_SHORT_MATCHES = mask
        queryParse.COLOR_FAMILY_WEIGHT = color_w
        queryParse.TYPE_FAMILY_WEIGHT = 1.0
        rows = ev.eval_auto(queries, ev.rank_rerank)
        summary = ev.summarize(rows, ("ap50", "p10", "r10"))
        cases = {r["qid"]: r["ap50"] for r in rows}
        print(f"{name:<12}{summary['overall']['ap50']:>8.4f}"
              f"{summary['overall']['p10']:>7.4f}"
              f"{summary['overall']['r10']:>7.4f}"
              + "".join(f"{cases[q]:>12.3f}" for q in CASES))
        records.append(
            {
                "name": name,
                "mask_short_matches": mask,
                "color_family_weight": color_w,
                "type_family_weight": 1.0,
                "overall": summary["overall"],
                "by_group": summary["by_group"],
                "cases": {q: cases[q] for q in CASES},
            }
        )

    payload = {
        "note": "Parser ablation; see docs/projectReport.md 6.3",
        "beta": rerank.BETA,
        "rerank_depth": rerank.RERANK_DEPTH,
        "n_queries": len(queries),
        "configs": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
