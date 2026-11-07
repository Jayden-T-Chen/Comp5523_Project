#!/usr/bin/env python3
"""Measure the two things the ranking story rests on but never measured.

1. How accurate is the zero-shot attribute table? For each slot the table gives
   a probability over the catalogue values of that slot; we compare its argmax
   with the catalogue field of the same image. This says whether the re-ranker
   is reading colour and type off the pixels or just reshuffling the pool.

2. What does a query cost? Per-query latency for the baseline and the shipped
   re-ranking, the one-off cost of building the tables, the cost of encoding
   the gallery once more, and the cold start.

3. How large are the two fused terms inside the candidate pool? The cosine
   spread and the peak of the attribute term, per query group, which is what
   makes a choice of β arguable.

4. How often does a marketing title name an attribute the pixels cannot show?
   Counts over `display_name`, which is the evidence against deeper attribute
   keys.

Writes results/attribute_probe.json. Reached through `python main.py probe`. The latency fields depend on
what else the machine is doing: the shipped file holds an 8-round run with a warm cache, and the numbers
quoted in docs/projectReport.md 4.4 are that run, so a re-timing means `--rounds 8` and then refreshing
that section.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from . import runEvaluation as ev

ROOT = Path(__file__).resolve().parents[2]
QUERIES_PATH = ROOT / "evaluation" / "sets" / "queries.json"
OUT_PATH = ROOT / "results" / "attribute_probe.json"
SLOTS = (("color", "color"), ("type", "articleType"), ("gender", "gender"))
TITLE_TERMS = {
    "material": ["leather", "suede", "canvas", "mesh", "knit", "patent", "nubuck",
                 "synthetic", "rubber", "eva", "textile", "fabric", "jute", "cork",
                 "denim", "velvet", "pu "],
    "toe": ["toe"],
    "lace": ["lace"],
    "closure": ["buckle", "zip", "velcro", "clasp", "slip-on", "strap"],
}


def title_terms(rows: list[dict]) -> dict:
    """Title words that name an attribute no slot carries."""
    titles = [r["display_name"].lower() for r in rows]
    hits = {kind: sum(1 for t in titles if any(word in t for word in words))
            for kind, words in TITLE_TERMS.items()}
    any_hit = sum(1 for t in titles
                  if any(word in t for words in TITLE_TERMS.values() for word in words))
    return {"n_titles": len(titles), "with_any_term": any_hit, "by_kind": hits,
            "terms": TITLE_TERMS}


def table_accuracy(tables: dict, rows: list[dict]) -> dict:
    out = {}
    for slot, field in SLOTS:
        values = tables["values"][slot]
        probs = tables[slot]
        top1 = probs.argmax(dim=1).tolist()
        top2 = probs.topk(min(2, len(values)), dim=1).indices.tolist()
        out[slot] = {
            "field": field,
            "n_values": len(values),
            "top1": round(sum(values[i] == r[field]
                              for i, r in zip(top1, rows)) / len(rows), 4),
            "top2": round(sum(r[field] in [values[i] for i in idxs]
                              for idxs, r in zip(top2, rows)) / len(rows), 4),
        }
    return out


def latency(queries: list[dict], rounds: int = 4) -> dict:
    """Milliseconds per query, after one warm-up call, with each round kept
    separate so a reader can see the spread instead of a single averaged number."""
    timed = {}
    for name in ("baseline", "rerank"):
        rank = ev.METHODS[name]
        rank(queries[0], ev.DEPTH)
        per_round = []
        for _ in range(rounds):
            start = time.perf_counter()
            for q in queries:
                rank(q, ev.DEPTH)
            per_round.append(round((time.perf_counter() - start)
                                   / len(queries) * 1000, 1))
        timed[f"{name}_per_query_ms"] = round(sum(per_round) / len(per_round), 1)
        timed[f"{name}_per_round_ms"] = per_round
    return timed


def score_scales(queries: list[dict]) -> dict:
    """How big the two fused terms actually are inside the candidate pool.

    β only means something relative to the cosine spread it has to resolve, so
    for every query we take the shipped top-100 pool and record the spread of
    the cosine term and the peak of the attribute term, per query group.
    """
    from statistics import median
    from retrieval import rerank

    per_group: dict[str, dict] = {}
    for q in queries:
        detail = rerank.rerank_detail(
            ev._CTX["model"], ev._CTX["tokenizer"], ev._CTX["feats"],
            ev._CTX["rows"], q["query"], rerank.RERANK_DEPTH, ev._CTX["device"])
        sims = [d["sim"] for d in detail]
        agree = [d["agree"] for d in detail if d["agree"] is not None]
        row = per_group.setdefault(q["group"], {"spreads": [], "peaks": []})
        row["spreads"].append(max(sims) - min(sims))
        row["peaks"].append(max(agree) if agree else 0.0)

    out = {"rerank_depth": rerank.RERANK_DEPTH, "beta": rerank.BETA,
           "device_note": "top-100 candidate pool, 60 main queries"}
    for group, row in sorted(per_group.items()):
        out[group] = {
            "n": len(row["spreads"]),
            "cosine_spread_median": round(median(row["spreads"]), 4),
            "agreement_peak_median": round(median(row["peaks"]), 3),
            "agreement_peak_min": round(min(row["peaks"]), 3),
            "agreement_peak_max": round(max(row["peaks"]), 3),
            "bonus_at_beta_median": round(rerank.BETA * median(row["peaks"]), 4),
        }
    return out


def gallery_encode(ctx: dict) -> float:
    """Seconds to embed the whole gallery from the JPEGs, in memory only.

    The cache is read but never written, so this says what the offline pass
    costs without putting the run's features at risk.
    """
    from retrieval import clipCore

    start = time.perf_counter()
    feats = clipCore.build_image_features(ctx["model"], ctx["preprocess"],
                                          ctx["paths"], ctx["device"])
    spent = time.perf_counter() - start
    del feats
    return round(spent, 1)


def add_args(sub) -> None:
    sub.add_argument("--out", type=Path, default=OUT_PATH)
    sub.add_argument("--rounds", type=int, default=4,
                     help="timing passes over the query set (default: 4; the shipped file is an "
                          "8-round run)")


def main(args) -> None:
    from retrieval import clipCore, rerank

    cached = clipCore.CACHE_PATH.is_file()
    t0 = time.perf_counter()
    ev.load_context()
    t_context = time.perf_counter() - t0
    ctx = ev._CTX

    t0 = time.perf_counter()
    tables = rerank.attribute_tables(ctx["model"], ctx["tokenizer"],
                                     ctx["feats"], ctx["rows"], ctx["device"])
    t_tables = time.perf_counter() - t0

    queries = [q for q in json.loads(QUERIES_PATH.read_text(encoding="utf-8"))
               ["queries"] if q["relevant"]]
    accuracy = table_accuracy(tables, ctx["rows"])
    timing = latency(queries, args.rounds)
    scales = score_scales(queries)
    terms = title_terms(ctx["rows"])
    encode_s = gallery_encode(ctx)

    for slot, row in accuracy.items():
        print(f"zero-shot {slot:<7} top1 {row['top1']:.3f}  top2 {row['top2']:.3f}"
              f"   ({row['n_values']} catalogue values, field={row['field']})")
    for group, row in scales.items():
        if isinstance(row, dict) and "n" in row:
            print(f"scale {group:<7} cosine spread {row['cosine_spread_median']:.4f}"
                  f"   agreement peak median {row['agreement_peak_median']:.3f}"
                  f"  [{row['agreement_peak_min']:.3f}, {row['agreement_peak_max']:.3f}]"
                  f"   bonus at beta {row['bonus_at_beta_median']:.4f}")
    for key, value in timing.items():
        print(f"{key:<24} {value}")
    print(f"title terms             {terms['with_any_term']} of "
          f"{terms['n_titles']} titles name a deeper attribute "
          f"({', '.join(f'{k} {v}' for k, v in terms['by_kind'].items())})")
    print(f"attribute tables once   {t_tables * 1000:.0f} ms")
    print(f"gallery re-encode once  {encode_s:.1f} s (cache left untouched)")
    print(f"cold start to first query {t_context:.1f} s "
          f"(device {ctx['device']}, feature cache "
          f"{'hit' if cached else 'rebuilt'})")

    payload = {
        "note": "Zero-shot attribute table accuracy against the catalogue "
                "fields, the size of the two fused terms, and query latency; "
                "see docs/projectReport.md 4.3, 4.4 and 6.4",
        "device": ctx["device"],
        "gallery": len(ctx["rows"]),
        "n_queries_timed": len(queries),
        "timing_rounds": args.rounds,
        "attribute_table": accuracy,
        "score_scales": scales,
        "title_terms": terms,
        "latency": {**timing,
                    "attribute_tables_once_ms": round(t_tables * 1000, 1),
                    "gallery_reencode_once_s": encode_s,
                    "cold_start_s": round(t_context, 1),
                    "feature_cache_hit": cached},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(f"\n-> {args.out}")
