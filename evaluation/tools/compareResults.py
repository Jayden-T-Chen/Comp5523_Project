#!/usr/bin/env python3
"""Paired comparison of two results/*.json files.

Both runs use the same queries, so the comparison is paired: it resamples the
query deltas together, keeping the two scores of each query in the same draw.
The reported interval is a percentile bootstrap over queries, 10,000 resamples,
fixed seed. The breakdown at the end uses `--by` (default: the query group).

Reached through `python main.py compare A B`; `--out` archives the interval as
JSON under `results/comparisons/` so the report can cite a file for every
confidence bound it prints.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

METRICS = ("ap50", "p10", "r10")   # only those present in both files are used
LABELS = {"ap50": "mAP@50", "p10": "P@10", "r10": "R@10"}
EPS = 1e-9


def load(path: Path) -> dict[str, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {row["qid"]: row for row in data["per_query"]}


def bootstrap_ci(deltas: list[float], n: int, rng: random.Random) -> tuple[float, float]:
    size = len(deltas)
    means = []
    for _ in range(n):
        total = sum(deltas[rng.randrange(size)] for _ in range(size))
        means.append(total / size)
    means.sort()
    return means[int(0.025 * n)], means[int(0.975 * n)]


def compare(ra: dict[str, dict], rb: dict[str, dict], qids: list[str],
            keys: list[str], by: str, n: int, seed: int) -> dict:
    """Paired bootstrap of B against A over the shared queries."""
    rng = random.Random(seed)
    metrics = {}
    for key in keys:
        deltas = [rb[q][key] - ra[q][key] for q in qids]
        lo, hi = bootstrap_ci(deltas, n, rng)
        wins = sum(1 for d in deltas if d > EPS)
        losses = sum(1 for d in deltas if d < -EPS)
        metrics[key] = {
            "a": sum(ra[q][key] for q in qids) / len(qids),
            "b": sum(rb[q][key] for q in qids) / len(qids),
            "delta": sum(deltas) / len(deltas),
            "ci95": [lo, hi],
            "better": wins,
            "worse": losses,
            "tied": len(deltas) - wins - losses,
        }

    breakdown = {}
    for g in sorted({ra[q][by] for q in qids}):
        rows = [q for q in qids if ra[q][by] == g]
        breakdown[str(g)] = {
            "n_queries": len(rows),
            **{key: sum(rb[q][key] - ra[q][key] for q in rows) / len(rows)
               for key in keys},
        }

    return {
        "n_queries": len(qids),
        "by": by,
        "resamples": n,
        "seed": seed,
        "metrics": metrics,
        "breakdown": breakdown,
    }


def add_args(sub) -> None:
    sub.add_argument("a", type=Path, help="reference results file")
    sub.add_argument("b", type=Path, help="candidate results file")
    sub.add_argument("--n", type=int, default=10000, help="bootstrap resamples")
    sub.add_argument("--seed", type=int, default=0)
    sub.add_argument("--by", default="group",
                    help="per-query field for the breakdown (default: group)")
    sub.add_argument("--out", type=Path,
                     help="archive the interval as JSON so a report can cite it")


def main(args) -> None:
    ra, rb = load(args.a), load(args.b)
    qids = sorted(set(ra) & set(rb))
    if not qids:
        raise SystemExit("The two files share no query ids.")
    if set(ra) != set(rb):
        print(f"note: comparing the {len(qids)} queries present in both files")

    keys = [k for k in METRICS
            if all(k in ra[q] for q in qids) and all(k in rb[q] for q in qids)]
    if not keys:
        raise SystemExit("The two files share no metric column.")

    missing = [q for q in qids if args.by not in ra[q]]
    if missing:
        raise SystemExit(f"{args.by!r} missing on {len(missing)} rows "
                         f"(e.g. {missing[0]})")

    summary = compare(ra, rb, qids, keys, args.by, args.n, args.seed)

    print(f"A: {args.a}  ({len(qids)} queries)")
    print(f"B: {args.b}")
    print()
    print(f"{'metric':<8}{'A':>8}{'B':>8}{'delta':>9}   {'95% CI':<22}{'W/L/T':>9}")
    for key, m in summary["metrics"].items():
        lo, hi = m["ci95"]
        sig = ("significant" if lo > 0
               else "significant (negative)" if hi < 0 else "not significant")
        wlt = f"{m['better']}/{m['worse']}/{m['tied']}"
        print(f"{LABELS[key]:<8}{m['a']:>8.4f}{m['b']:>8.4f}"
              f"{m['delta']:>+9.4f}   "
              f"[{lo:+.4f}, {hi:+.4f}]".ljust(30)
              + f"{wlt:>9}"
              f"  {sig}")

    print()
    print(f"per {args.by} (delta " + " / delta ".join(LABELS[k] for k in keys) + "):")
    for g, row in summary["breakdown"].items():
        parts = [f"{row[key]:+.4f}" for key in keys]
        print(f"  {g:<10} {'/'.join(parts)}  (n={row['n_queries']})")

    print()
    print(f"{args.n:,} paired bootstrap resamples over queries (seed {args.seed}).")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "a": str(args.a),
            "b": str(args.b),
            **summary,
        }
        args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
