#!/usr/bin/env python3
"""Paired comparison of two results/*.json files.

Both runs use the same queries, so the comparison is paired: resample the 60
query deltas, not the two score vectors independently. The reported interval is
a percentile bootstrap over queries, 10,000 resamples, fixed seed.

Usage:
  python evaluation/compareResults.py results/baseline_auto.json results/rerank_auto.json
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

METRICS = ("ap50", "p10")
LABELS = {"ap50": "mAP@50", "p10": "P@10"}
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Paired bootstrap comparison")
    ap.add_argument("a", type=Path, help="reference results file")
    ap.add_argument("b", type=Path, help="candidate results file")
    ap.add_argument("--n", type=int, default=10000, help="bootstrap resamples")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ra, rb = load(args.a), load(args.b)
    qids = sorted(set(ra) & set(rb))
    if not qids:
        raise SystemExit("The two files share no query ids.")
    if set(ra) != set(rb):
        print(f"note: comparing the {len(qids)} queries present in both files")

    rng = random.Random(args.seed)
    print(f"A: {args.a}  ({len(qids)} queries)")
    print(f"B: {args.b}")
    print()
    print(f"{'metric':<8}{'A':>8}{'B':>8}{'delta':>9}   {'95% CI':<22}{'W/L/T':>9}")
    for key in METRICS:
        da = [ra[q][key] for q in qids]
        db = [rb[q][key] for q in qids]
        deltas = [b - a for a, b in zip(da, db)]
        lo, hi = bootstrap_ci(deltas, args.n, rng)
        wins = sum(1 for d in deltas if d > EPS)
        losses = sum(1 for d in deltas if d < -EPS)
        ties = len(deltas) - wins - losses
        mean_a = sum(da) / len(da)
        mean_b = sum(db) / len(db)
        sig = "significant" if lo > 0 else ("significant (negative)" if hi < 0 else "not significant")
        print(f"{LABELS[key]:<8}{mean_a:>8.4f}{mean_b:>8.4f}{sum(deltas)/len(deltas):>+9.4f}   "
              f"[{lo:+.4f}, {hi:+.4f}]".ljust(30) + f"{f'{wins}/{losses}/{ties}':>9}"
              f"  {sig}")

    print()
    print("per group (delta mAP@50 / delta P@10):")
    groups = sorted({ra[q]["group"] for q in qids})
    for g in groups:
        sub = [q for q in qids if ra[q]["group"] == g]
        parts = []
        for key in METRICS:
            d = sum(rb[q][key] - ra[q][key] for q in sub) / len(sub)
            parts.append(f"{d:+.4f}")
        print(f"  {g:<6} {'/'.join(parts)}")

    print()
    print(f"{args.n:,} paired bootstrap resamples over queries (seed {args.seed}).")


if __name__ == "__main__":
    main()
