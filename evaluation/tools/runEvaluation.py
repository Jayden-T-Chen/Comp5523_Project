#!/usr/bin/env python3
"""Score a retrieval method against evaluation/sets/queries.json.

Metrics (all averaged over queries):
  mAP@50   mean average precision, relevant set capped at the ranking depth
  P@10     precision in the top 10
  R@10     recall in the top 10 (hits / all relevant images in the gallery)

--labels auto    uses the metadata-derived `relevant` list (all 60 queries)
--labels strict  uses the by-eye judgments in evaluation/sets/judgements.json
                 (only the queries that have been judged)

Reported numbers use `auto`, because its labels cover the whole gallery and are
therefore comparable across methods. The strict pool was judged on the baseline
top-10 only, so a method that surfaces new images is scored 0 on them until they
are judged too. Use strict to calibrate absolute quality; cross-method ranking
rests on auto.

Methods live in METHODS; add "prompt" / "rerank" there as they are built, so
every number in the report comes from the same script and the same queries.

Usage:
  python evaluation/tools/runEvaluation.py --method baseline
  python evaluation/tools/runEvaluation.py --method baseline --labels strict
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "retrieval"))

QUERIES_PATH = ROOT / "evaluation" / "sets" / "queries.json"
JUDGEMENTS_PATH = ROOT / "evaluation" / "sets" / "judgements.json"
RESULTS_DIR = ROOT / "results"
DEPTH = 50          # ranking depth used for mAP
TOPK = 10           # depth used for P@10 / R@10


def average_precision(ranked: list[str], relevant: set[str], depth: int) -> float:
    if not relevant:
        return 0.0
    hits = 0
    total = 0.0
    for i, iid in enumerate(ranked[:depth], 1):
        if iid in relevant:
            hits += 1
            total += hits / i
    return total / min(len(relevant), depth)


def rank_baseline(query: str, depth: int) -> list[str]:
    """Frozen CLIP: cosine similarity over cached image features."""
    import clipCore

    ctx = _CTX
    q = clipCore.embed_query(ctx["model"], ctx["tokenizer"], query, ctx["device"])
    return _topk(q, depth)


def rank_prompt(query: str, depth: int) -> list[str]:
    """Same ranking, but the query is the mean of several prompt phrasings."""
    import clipCore

    ctx = _CTX
    q = clipCore.embed_query_ensemble(
        ctx["model"], ctx["tokenizer"], query, ctx["device"])
    return _topk(q, depth)


def _topk(q, depth: int) -> list[str]:
    ctx = _CTX
    scores = (ctx["feats"] @ q.T).squeeze(1)
    k = min(depth, len(ctx["rows"]))
    _vals, idxs = scores.topk(k)
    return [ctx["rows"][i]["id"] for i in idxs.tolist()]


def rank_rerank(query: str, depth: int) -> list[str]:
    """Similarity + zero-shot attribute agreement over the coarse top-k.

    The candidate pool is RERANK_DEPTH, independent of the metric cutoffs:
    re-ranking 100 candidates and returning the best 50 (DEPTH) is the shipped
    configuration; `depth` only bounds what the caller keeps.
    """
    import rerank

    ctx = _CTX
    beta = ctx.get("beta", rerank.BETA)
    return rerank.rerank(
        ctx["model"], ctx["tokenizer"], ctx["feats"], ctx["rows"],
        query, max(depth, rerank.RERANK_DEPTH), ctx["device"], beta)


METHODS = {"baseline": rank_baseline, "prompt": rank_prompt,
           "rerank": rank_rerank}

_CTX: dict = {}


def load_context() -> None:
    import clipCore

    rows, paths = clipCore.load_rows()
    device = clipCore.get_device()
    model, preprocess, tokenizer = clipCore.load_model(device)
    feats = clipCore.get_image_features(model, preprocess, paths, device)
    _CTX.update(rows=rows, paths=paths, device=device,
                model=model, tokenizer=tokenizer, feats=feats)
    print(f"device: {device}  gallery: {len(rows)} images")


def eval_auto(queries: list[dict], rank) -> list[dict]:
    out = []
    for q in queries:
        ranked = rank(q["query"], DEPTH)
        rel = set(q["relevant"])
        hits10 = len(rel & set(ranked[:TOPK]))
        row = {
            "qid": q["qid"],
            "query": q["query"],
            "group": q["group"],
            "n_relevant": len(rel),
            "ap50": average_precision(ranked, rel, DEPTH),
            "p10": hits10 / TOPK,
            "r10": hits10 / len(rel) if rel else 0.0,
        }
        if "kind" in q:
            row["kind"] = q["kind"]
        out.append(row)
    return out


def eval_strict(queries: list[dict], rank, judged_map: dict) -> list[dict]:
    """Judged top-10 pool: p10 = judged-relevant hits / 10; unjudged ids count 0."""
    out = []
    for q in queries:
        judged = judged_map.get(q["qid"])
        if not judged:
            continue
        ranked = rank(q["query"], TOPK)
        hits = sum(int(judged.get(iid, 0)) for iid in ranked[:TOPK])
        out.append(
            {
                "qid": q["qid"],
                "query": q["query"],
                "group": q["group"],
                "pool_judged": len(judged),
                "hits": hits,
                "p10": hits / TOPK,
            }
        )
    return out


def summarize(rows: list[dict], keys: tuple[str, ...]) -> dict:
    def mean(vals):
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    overall = {k: mean([r[k] for r in rows]) for k in keys}
    by_group: dict[str, dict] = {}
    for g in sorted({r["group"] for r in rows}):
        sub = [r for r in rows if r["group"] == g]
        by_group[g] = {"n": len(sub), **{k: mean([r[k] for r in sub]) for k in keys}}
    return {"n_queries": len(rows), "overall": overall, "by_group": by_group}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=sorted(METHODS), default="baseline")
    ap.add_argument("--labels", choices=["auto", "strict"], default="auto")
    ap.add_argument("--beta", type=float, default=None,
                    help="rerank fusion weight (default: rerank.BETA); "
                         "non-default values get a _b<value> output suffix")
    ap.add_argument("--queries", type=Path, default=QUERIES_PATH,
                    help="query file; non-default files add a tag to the "
                         "default output name "
                         "(default: evaluation/sets/queries.json)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if args.beta is not None:
        _CTX["beta"] = args.beta

    data = json.loads(args.queries.read_text(encoding="utf-8"))
    queries = data["queries"]
    judged_map: dict = {}
    if args.labels == "auto":
        queries = [q for q in queries if q["relevant"]]
    else:
        if not JUDGEMENTS_PATH.is_file():
            raise SystemExit(f"Missing {JUDGEMENTS_PATH}.")
        judged_map = json.loads(
            JUDGEMENTS_PATH.read_text(encoding="utf-8"))["judged"]
        queries = [q for q in queries if q["qid"] in judged_map]
        if args.method != "baseline":
            print(f"note: strict pool was judged on baseline top-10; "
                  f"{args.method} ids outside it count 0 until judged")
        if not queries:
            raise SystemExit(
                "No judged queries overlap the query file; check qids against "
                "evaluation/sets/judgements.json."
            )

    load_context()
    rank = METHODS[args.method]
    rows = (eval_auto(queries, rank) if args.labels == "auto"
            else eval_strict(queries, rank, judged_map))
    keys = ("ap50", "p10", "r10") if args.labels == "auto" else ("p10",)
    summary = summarize(rows, keys)

    print(f"\n== {args.method} / {args.labels} labels ==")
    print(f"queries: {summary['n_queries']}")
    print("  overall :", summary["overall"])
    for g, s in summary["by_group"].items():
        print(f"  {g:<6}: n={s['n']:<3} " +
              "  ".join(f"{k}={s[k]:.4f}" for k in keys))

    suffix = ""
    if args.queries.resolve() != QUERIES_PATH.resolve():
        suffix = "_" + args.queries.stem.removeprefix("queries_")
    if args.out is not None:
        out = args.out
    elif args.beta is not None:
        out = RESULTS_DIR / f"{args.method}_{args.labels}{suffix}_b{args.beta:g}.json"
    else:
        out = RESULTS_DIR / f"{args.method}_{args.labels}{suffix}.json"
    try:
        queries_file = args.queries.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        queries_file = str(args.queries)
    extra = {"queries_file": queries_file}
    if args.method == "rerank":
        import queryParse
        import rerank

        extra["rerank_depth"] = rerank.RERANK_DEPTH
        extra["beta"] = args.beta if args.beta is not None else rerank.BETA
        extra["parser"] = {
            "mask_short_matches": queryParse.MASK_SHORT_MATCHES,
            "color_family_weight": queryParse.COLOR_FAMILY_WEIGHT,
            "type_family_weight": queryParse.TYPE_FAMILY_WEIGHT,
        }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"method": args.method, "labels": args.labels,
             "depth": DEPTH, "topk": TOPK, **extra, **summary, "per_query": rows},
            ensure_ascii=False, indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
