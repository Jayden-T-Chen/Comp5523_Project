#!/usr/bin/env python3
"""Quantify the gap between the shipped encoder config and OpenAI's own.

`open_clip` builds ViT-B-32 from its default model config, where the text tower
uses ordinary GELU, while the `openai` weights were trained with QuickGELU, so
the library warns on every load. The gallery features in `data/` are cached
under a signature of the images alone, which means the shipped numbers and the
cached file belong to the same GELU configuration.

This command scores the 60 main queries under both activations with the real
ranking functions and writes `results/encoderCheck.json`, so the report can say
how far the deviation reaches instead of leaving the warning unmeasured. The
feature cache is never touched: the second configuration is encoded into memory.

Reached through `python main.py encoder-check`.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import runEvaluation as ev

ROOT = Path(__file__).resolve().parents[2]
QUERIES_PATH = ROOT / "evaluation" / "sets" / "queries.json"
JUDGEMENTS_PATH = ROOT / "evaluation" / "sets" / "judgements.json"
OUT_PATH = ROOT / "results" / "encoderCheck.json"
CONFIGS = ("gelu", "quick_gelu")


def build(clipCore, device: str, quick: bool):
    """Model, preprocess and tokenizer with the text activation chosen here."""
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        clipCore.MODEL_NAME, pretrained=clipCore.PRETRAINED,
        force_quick_gelu=quick)
    tokenizer = open_clip.get_tokenizer(clipCore.MODEL_NAME)
    return model.to(device).eval(), preprocess, tokenizer


def score(queries: list[dict], name: str) -> dict:
    rows = ev.eval_auto(queries, ev.METHODS[name])
    return ev.summarize(rows, ("ap50", "p10", "r10"))["overall"]


def pool_churn(queries: list[dict], judged: dict, name: str) -> dict:
    """How many top-10 cells a ranking puts on images nobody has judged."""
    still, unseen = 0, 0
    for q in queries:
        if q["qid"] not in judged:
            continue
        ranked = ev.METHODS[name](q, ev.TOPK)[:ev.TOPK]
        pool = judged[q["qid"]]
        still += sum(1 for iid in ranked if iid in pool)
        unseen += sum(1 for iid in ranked if iid not in pool)
    return {"judged": still, "unjudged": unseen}


def add_args(sub) -> None:
    sub.add_argument("--out", type=Path, default=OUT_PATH)


def main(args) -> None:
    from retrieval import clipCore, rerank

    queries = [q for q in json.loads(QUERIES_PATH.read_text(encoding="utf-8"))
               ["queries"] if q["relevant"]]
    judged = json.loads(JUDGEMENTS_PATH.read_text(encoding="utf-8"))["judged"]
    rows, paths = clipCore.load_rows()
    device = clipCore.get_device()

    payload = {
        "note": "Encoder-configuration check; see docs/projectReport.md 7.3. "
                "The shipped configuration is 'gelu', which is also what "
                "results/*.json and data/image_features.pt were produced with.",
        "device": device,
        "gallery": len(rows),
        "n_queries": len(queries),
        "configs": {},
    }

    for quick in CONFIGS:
        model, preprocess, tokenizer = build(clipCore, device,
                                             quick == "quick_gelu")
        feats = clipCore.build_image_features(model, preprocess, paths, device)
        rerank._TABLES.clear()       # the table cache keys on shape, not weights
        ev._CTX.update(rows=rows, paths=paths, device=device,
                       model=model, tokenizer=tokenizer, feats=feats)
        payload["configs"][quick] = {
            "force_quick_gelu": quick == "quick_gelu",
            "baseline": score(queries, "baseline"),
            "rerank": score(queries, "rerank"),
            "attr_only": score(queries, "attr_only"),
            "prompt": score(queries, "prompt"),
            "top10_against_the_judged_pool": {
                name: pool_churn(queries, judged, name)
                for name in ("baseline", "rerank")},
        }
        print(f"{quick:<11} baseline {payload['configs'][quick]['baseline']['ap50']:.4f}"
              f"  rerank {payload['configs'][quick]['rerank']['ap50']:.4f}"
              f"  P@10 {payload['configs'][quick]['baseline']['p10']:.4f} ->"
              f" {payload['configs'][quick]['rerank']['p10']:.4f}")

    a = payload["configs"]["gelu"]
    b = payload["configs"]["quick_gelu"]
    payload["delta_quick_gelu_minus_gelu"] = {
        key: round(b["baseline"][key] - a["baseline"][key], 4)
        for key in ("ap50", "p10", "r10")}
    payload["gain_under_each_config"] = {
        quick: {key: round(cfg["rerank"][key] - cfg["baseline"][key], 4)
                for key in ("ap50", "p10", "r10")}
        for quick, cfg in payload["configs"].items()}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(f"\n-> {args.out}")
