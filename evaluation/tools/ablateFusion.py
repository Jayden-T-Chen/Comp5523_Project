#!/usr/bin/env python3
"""Sweep the two fusion knobs: cosine candidate pool and the weight beta.

One process, one gallery, one set of attribute tables. The shipped point is
pool 100 with beta 0.1; every other row re-scores the same 60 auto queries with
one knob moved, so the columns are directly comparable and `compareResults.py`
can take any two of them.

  pool  how many cosine candidates re-ranking is allowed to reorder. The metric
        cutoff stays at 50 (runEvaluation.DEPTH), so pool 50 is the case where
        re-ranking can only permute the list it was given.
  beta  weight of the attribute term on top of cosine.

Pool recall is reported next to each pool size: the share of a query's relevant
images that the pool contains at all. It bounds what re-ranking can do, which is
why deepening the pool and strengthening the attribute term are different levers.

Usage:
  python evaluation/tools/ablateFusion.py
  python evaluation/tools/ablateFusion.py --out results/fusion_ablation.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "retrieval"))
sys.path.insert(0, str(ROOT / "evaluation" / "tools"))

import clipCore        # noqa: E402
import queryParse      # noqa: E402
import rerank          # noqa: E402
import runEvaluation as ev   # noqa: E402

QUERIES_PATH = ROOT / "evaluation" / "sets" / "queries.json"
RESULTS_DIR = ROOT / "results"
POOLS = (50, 100, 200)
BETAS = (0.2, 0.4, 0.8, 1.5)
CASES = ["color_grey", "color_beige", "color_olive", "combo_beige_heels"]


def rank_in_pool(query_pool: int, beta: float):
    """Re-rank the cosine top-`query_pool` with `beta`, return metric depth."""

    def rank(query: str, depth: int) -> list[str]:
        ctx = ev._CTX
        return rerank.rerank(
            ctx["model"], ctx["tokenizer"], ctx["feats"], ctx["rows"],
            query, max(depth, query_pool), ctx["device"], beta)[:depth]

    return rank


def pool_recall(queries: list[dict], pool: int) -> float:
    """Macro share of relevant images inside the cosine top-`pool`."""
    ctx = ev._CTX
    import torch

    shares = []
    for q in queries:
        vec = clipCore.embed_query(ctx["model"], ctx["tokenizer"],
                                   q["query"], ctx["device"])
        sims = (ctx["feats"] @ vec.T).squeeze(1)
        k = min(pool, len(ctx["rows"]))
        _v, idxs = sims.topk(k)
        in_pool = {ctx["rows"][i]["id"] for i in idxs.tolist()}
        rel = set(q["relevant"])
        shares.append(len(rel & in_pool) / len(rel))
    return float(torch.tensor(shares).mean())


def run(queries: list[dict], name: str, pool: int, beta: float) -> dict:
    rows = ev.eval_auto(queries, rank_in_pool(pool, beta))
    summary = ev.summarize(rows, ("ap50", "p10", "r10"))
    cases = {r["qid"]: r["ap50"] for r in rows}
    recall = pool_recall(queries, pool)
    print(f"{name:<12}{pool:>6}{beta:>6.1f}{summary['overall']['ap50']:>9.4f}"
          f"{summary['overall']['p10']:>8.4f}{recall:>9.3f}"
          + "".join(f"{cases[q]:>10.3f}" for q in CASES))
    return {
        "name": name,
        "rerank_depth": pool,
        "beta": beta,
        "pool_recall": round(recall, 4),
        "overall": summary["overall"],
        "by_group": summary["by_group"],
        "cases": {q: cases[q] for q in CASES},
        "per_query": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Fusion ablation (pool, beta)")
    ap.add_argument("--out", type=Path,
                    default=RESULTS_DIR / "fusion_ablation.json")
    args = ap.parse_args()

    queries = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))["queries"]
    ev.load_context()

    header = (f"{'config':<12}{'pool':>6}{'beta':>6}{'mAP@50':>9}{'P@10':>8}"
              f"{'pool rec':>9}")
    header += "".join(f"{q.split('_', 1)[1][:8]:>10}" for q in CASES)
    print(header)

    records = [run(queries, f"pool {p}", p, rerank.BETA) for p in POOLS]
    records += [run(queries, f"beta {b}", rerank.RERANK_DEPTH, b)
                for b in BETAS]

    payload = {
        "note": "Fusion ablation (candidate pool, beta); "
                "see docs/projectReport.md 6.4",
        "shipped": {"rerank_depth": rerank.RERANK_DEPTH, "beta": rerank.BETA,
                    "metric_depth": ev.DEPTH, "topk": ev.TOPK},
        "parser": {"mask_short_matches": queryParse.MASK_SHORT_MATCHES,
                   "color_family_weight": queryParse.COLOR_FAMILY_WEIGHT,
                   "type_family_weight": queryParse.TYPE_FAMILY_WEIGHT},
        "n_queries": len(queries),
        "configs": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
