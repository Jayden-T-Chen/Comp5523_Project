"""Attribute-verified re-ranking.

CLIP cosine similarity alone cannot tell "silver" from "bronze" or a lifestyle
sneaker from a casual shoe, because those words sit close together in its text
space. So after the coarse top-k we score every image against the attribute
values the query actually names (zero-shot, one prompt per value), and add that
agreement to the similarity before sorting again:

    score = cos(query, image) + beta * attribute_agreement

where agreement multiplies the per-attribute probabilities, because a query
naming a colour and a type wants both at once (AND, not OR).

    beta = 0 means no re-ranking (identical to the baseline).
"""
from __future__ import annotations

import torch

from clipCore import ROOT, embed_query
from queryParse import parse_query

BETA = 0.1
RERANK_DEPTH = 100           # cosine candidates re-scored after the coarse pass
LOGIT_SCALE = 100.0          # CLIP's own temperature

_TABLES: dict = {}


def _probs(model, tokenizer, feats, values, template, device) -> torch.Tensor:
    """[N, A] softmax over the candidate values of one attribute."""
    texts = [template.format(v.lower()) for v in values]
    with torch.no_grad():
        tok = tokenizer(texts).to(device)
        t = model.encode_text(tok)
        t = t / t.norm(dim=-1, keepdim=True)
        logits = feats @ t.cpu().T * LOGIT_SCALE
        return logits.softmax(dim=-1)


def attribute_tables(model, tokenizer, feats, rows, device) -> dict:
    """Zero-shot attribute distributions, built once per process."""
    colors = sorted({r["color"] for r in rows})
    types = sorted({r["articleType"] for r in rows})
    genders = sorted({r["gender"] for r in rows})
    key = (len(feats), tuple(colors), tuple(types), tuple(genders))
    if _TABLES.get("key") != key:
        _TABLES.clear()
        _TABLES.update(
            key=key,
            color=_probs(model, tokenizer, feats, colors,
                         "a photo of a {} shoe", device),
            type=_probs(model, tokenizer, feats, types,
                        "a photo of {}", device),
            gender=_probs(model, tokenizer, feats, genders,
                          "a photo of a shoe for {}", device),
        )
        _TABLES.update(values={"color": colors, "type": types,
                               "gender": genders})
    return _TABLES


def attribute_agreement(attrs: dict, tables: dict) -> torch.Tensor | None:
    """[N] product of the probabilities for each named attribute value.

    Values the parser marked as family (weight < 1) contribute proportionally
    less, so "grey" nudges Silver but a Grey image still wins.
    """
    parts = []
    weights = attrs.get("weights", {})
    for name in ("color", "type", "gender"):
        wanted = attrs.get(name)
        if not wanted:
            continue
        values = tables["values"][name]
        wanted = [wanted] if isinstance(wanted, str) else wanted
        pairs = [(values.index(v), weights.get(name, {}).get(v, 1.0))
                 for v in wanted if v in values]
        if pairs:
            cols = [i for i, _ in pairs]
            w = torch.tensor([x for _, x in pairs], dtype=tables[name].dtype)
            parts.append((tables[name][:, cols] * w).sum(dim=1))
    if not parts:
        return None
    agreement = parts[0]
    for p in parts[1:]:
        agreement = agreement * p
    return agreement


def rerank_detail(model, tokenizer, feats, rows, query, depth, device,
                  beta: float = BETA) -> list[dict]:
    """Same ordering as rerank(), with both score components kept for display."""
    q = embed_query(model, tokenizer, query, device)
    sims = (feats @ q.T).squeeze(1)
    k = min(depth, len(rows))
    _vals, idxs = sims.topk(k)

    attrs = parse_query(query)
    bonus = attribute_agreement(attrs, attribute_tables(
        model, tokenizer, feats, rows, device))
    if bonus is None:
        mixed, agree, order = sims[idxs], None, torch.arange(len(idxs))
    else:
        mixed = sims[idxs] + beta * bonus[idxs]
        agree = bonus[idxs]
        order = mixed.argsort(descending=True)

    out = []
    for pos, j in enumerate(order.tolist()):
        i = int(idxs[j])
        row = rows[i]
        out.append(
            {
                "rank": pos + 1,
                "id": row["id"],
                "sim": float(sims[i]),
                "agree": None if agree is None else float(agree[j]),
                "path": str(ROOT / row["path"]),
                "title": row.get("display_name", ""),
                "type": row.get("articleType", ""),
                "color": row.get("color", ""),
            }
        )
    return out


def rerank(model, tokenizer, feats, rows, query, depth, device,
           beta: float = BETA) -> list[str]:
    """Ordered ids from rerank_detail()."""
    detail = rerank_detail(model, tokenizer, feats, rows, query, depth,
                           device, beta)
    return [d["id"] for d in detail]
