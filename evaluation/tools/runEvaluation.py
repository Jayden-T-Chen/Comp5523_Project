#!/usr/bin/env python3
"""Score a retrieval method against an evaluation set.

Metrics (all averaged over queries):
  mAP@50   mean average precision, relevant set capped at the ranking depth
  P@10     precision in the top 10
  R@10     recall in the top 10 (hits / all relevant images in the gallery)

--labels auto    uses the metadata-derived `relevant` list (all 60 queries)
--labels strict  uses the by-eye judgments in evaluation/sets/judgements.json
                 (only the queries that have been judged)

Five methods live in METHODS. Two of them are diagnostics rather than candidate
systems, and the report says so wherever they appear:

  baseline    frozen CLIP cosine
  prompt      the same cosine over a mean of six prompt phrasings
  rerank      cosine plus the zero-shot attribute term (the shipped system)
  attr_only   the attribute term on its own, cosine breaking ties only. It
              answers how much of the rerank gain the visual score carries.
  metadata    a catalogue lookup on the query's ground-truth attributes. It
              reads the labels, so it is an upper bound on the auto oracle and
              never a method we would ship.

Reached through `python main.py eval`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
QUERIES_PATH = ROOT / "evaluation" / "sets" / "queries.json"
JUDGEMENTS_PATH = ROOT / "evaluation" / "sets" / "judgements.json"
RESULTS_DIR = ROOT / "results"
DEPTH = 50          # ranking depth used for mAP
TOPK = 10           # depth used for P@10 / R@10

_CTX: dict = {}


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


def _pool(depth: int) -> int:
    """Candidate pool for the re-ranking arms: never narrower than the metric."""
    from retrieval import rerank

    return max(depth, rerank.RERANK_DEPTH)


def rank_baseline(q: dict, depth: int) -> list[str]:
    """Frozen CLIP: cosine similarity over cached image features."""
    from retrieval import clipCore

    ctx = _CTX
    vec = clipCore.embed_query(ctx["model"], ctx["tokenizer"], q["query"],
                               ctx["device"])
    return _topk(vec, depth)


def rank_prompt(q: dict, depth: int) -> list[str]:
    """Same ranking, but the query is the mean of several prompt phrasings."""
    from retrieval import clipCore

    ctx = _CTX
    vec = clipCore.embed_query_ensemble(ctx["model"], ctx["tokenizer"],
                                        q["query"], ctx["device"])
    return _topk(vec, depth)


def _topk(vec, depth: int) -> list[str]:
    ctx = _CTX
    scores = (ctx["feats"] @ vec.T).squeeze(1)
    k = min(depth, len(ctx["rows"]))
    _vals, idxs = scores.topk(k)
    return [ctx["rows"][i]["id"] for i in idxs.tolist()]


def rank_rerank(q: dict, depth: int) -> list[str]:
    """Similarity plus zero-shot attribute agreement over the coarse top-k."""
    from retrieval import rerank

    ctx = _CTX
    beta = ctx.get("beta", rerank.BETA)
    return rerank.rerank(ctx["model"], ctx["tokenizer"], ctx["feats"],
                         ctx["rows"], q["query"], _pool(depth),
                         ctx["device"], beta)


def rank_attr_only(q: dict, depth: int) -> list[str]:
    """The attribute term alone over the same pool; cosine only breaks ties."""
    from retrieval import rerank

    ctx = _CTX
    return rerank.rerank(ctx["model"], ctx["tokenizer"], ctx["feats"],
                         ctx["rows"], q["query"], _pool(depth),
                         ctx["device"], mode="attr")


def rank_metadata(q: dict, depth: int) -> list[str]:
    """Upper bound on the auto oracle: look the answer up in the catalogue.

    Uses the ground-truth `attrs` of the query, so it sees the labels. Images
    that match every named attribute come first, ordered among themselves by
    cosine so the head is not arbitrary; the rest follow.
    """
    from retrieval import clipCore

    ctx = _CTX
    attrs = q.get("attrs") or {}
    rows = ctx["rows"]
    vec = clipCore.embed_query(ctx["model"], ctx["tokenizer"], q["query"],
                               ctx["device"])
    sims = (ctx["feats"] @ vec.T).squeeze(1)
    matched = [i for i, r in enumerate(rows)
               if all(r.get(field) == value for field, value in attrs.items())]
    rest = [i for i in range(len(rows)) if i not in set(matched)]

    def by_sim(idxs):
        return sorted(idxs, key=lambda i: -float(sims[i]))

    return [rows[i]["id"] for i in (by_sim(matched) + by_sim(rest))[:depth]]


METHODS = {
    "baseline": rank_baseline,
    "prompt": rank_prompt,
    "rerank": rank_rerank,
    "attr_only": rank_attr_only,
    "metadata": rank_metadata,
}


def load_context() -> None:
    from retrieval import clipCore

    rows, paths = clipCore.load_rows()
    device = clipCore.get_device()
    model, preprocess, tokenizer = clipCore.load_model(device)
    feats = clipCore.get_image_features(model, preprocess, paths, device)
    _CTX.update(rows=rows, paths=paths, device=device,
                model=model, preprocess=preprocess, tokenizer=tokenizer, feats=feats)
    print(f"device: {device}  gallery: {len(rows)} images")


def eval_auto(queries: list[dict], rank) -> list[dict]:
    out = []
    for q in queries:
        ranked = rank(q, DEPTH)
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
        ranked = rank(q, TOPK)
        hits = sum(int(judged.get(iid, 0)) for iid in ranked[:TOPK])
        out.append(
            {
                "qid": q["qid"],
                "query": q["query"],
                "group": q["group"],
                "pool_judged": len(judged),
                "hits": hits,
                "p10": hits / TOPK,
                "unjudged": sum(1 for iid in ranked[:TOPK] if iid not in judged),
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


def add_args(sub) -> None:
    sub.add_argument("--method", choices=sorted(METHODS), default="baseline")
    sub.add_argument("--labels", choices=["auto", "strict"], default="auto")
    sub.add_argument("--beta", type=float, default=None,
                     help="rerank fusion weight (default: rerank.BETA); "
                          "non-default values get a _b<value> output suffix")
    sub.add_argument("--parser", choices=["shipped", "legacy"], default="shipped",
                     help="parser preset; 'legacy' regenerates the archived "
                          "pre-alignment run without editing source")
    sub.add_argument("--queries", type=Path, default=QUERIES_PATH,
                     help="query file; non-default files add a tag to the "
                          "default output name "
                          "(default: evaluation/sets/queries.json)")
    sub.add_argument("--out", type=Path, default=None)


def main(args) -> None:
    if args.beta is not None:
        _CTX["beta"] = args.beta

    from retrieval import queryParse

    if args.parser != "shipped":
        queryParse.apply_preset(args.parser)

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
                  f"{args.method} ids outside it count 0 until they are judged "
                  "(python main.py label-sheet)")
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
    if args.labels == "strict":
        open_cells = sum(r.get("unjudged", 0) for r in rows)
        if open_cells:
            print(f"  unjudged cells scored as 0: {open_cells} "
                  f"(python main.py label-sheet)")
    for g, s in summary["by_group"].items():
        print(f"  {g:<6}: n={s['n']:<3} " +
              "  ".join(f"{k}={s[k]:.4f}" for k in keys))

    suffix = ""
    if args.queries.resolve() != QUERIES_PATH.resolve():
        suffix = "_" + args.queries.stem.removeprefix("queries_")
    parser_tag = "" if args.parser == "shipped" else f"_{args.parser}"
    if args.out is not None:
        out = args.out
    elif args.beta is not None:
        out = RESULTS_DIR / (f"{args.method}{parser_tag}_{args.labels}"
                             f"{suffix}_b{args.beta:g}.json")
    else:
        out = RESULTS_DIR / f"{args.method}{parser_tag}_{args.labels}{suffix}.json"
    try:
        queries_file = args.queries.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        queries_file = str(args.queries)
    extra: dict = {"queries_file": queries_file}
    if args.method in ("rerank", "attr_only"):
        from retrieval import rerank

        extra["rerank_depth"] = rerank.RERANK_DEPTH
        extra["parser"] = queryParse.current_config()
        if args.method == "rerank":
            extra["beta"] = args.beta if args.beta is not None else rerank.BETA
    if args.labels == "strict":
        extra["unjudged_cells_scored_zero"] = sum(r.get("unjudged", 0)
                                                  for r in rows)
    if args.method == "metadata":
        extra["reads_labels"] = True
        extra["note"] = ("upper bound on the auto oracle, not a candidate "
                         "system: it matches the query's ground-truth "
                         "attributes against the catalogue")
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
