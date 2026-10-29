#!/usr/bin/env python3
"""Gradio UI: text -> image ranking over the footwear gallery with CLIP.

Two methods share the same gallery and the same image features, so the demo can
switch between them and show what attribute re-ranking changes.
"""
from __future__ import annotations

import gradio as gr

import rerank
from clipCore import (
    get_device,
    get_image_features,
    load_model,
    load_rows,
    search,
)
from queryParse import parse_query

ROWS, PATHS = load_rows()
DEVICE = get_device()
MODEL, PREPROCESS, TOKENIZER = load_model(DEVICE)
IMG_FEATS = get_image_features(MODEL, PREPROCESS, PATHS, DEVICE)

METHODS = ["baseline (raw CLIP)", "rerank (attributes)"]


def run(query: str, k: int, method: str):
    query = (query or "").strip()
    if not query:
        return [], "Type a query, e.g. black leather shoes"
    k = int(k or 8)
    head = [f"Query: {query}", f"Gallery: {len(ROWS)} images", ""]

    if method == METHODS[1]:
        parsed = parse_query(query)
        hits = rerank.rerank_detail(MODEL, TOKENIZER, IMG_FEATS, ROWS,
                                    query, max(k, rerank.RERANK_DEPTH),
                                    DEVICE)[:k]
        gallery = [
            (h["path"],
             f"{h['rank']}. cos {h['sim']:.3f} · attr {h['agree'] or 0:.2f}")
            for h in hits
        ]
        head.append(
            "parsed: "
            + ", ".join(f"{name}={parsed[name] or '-'}"
                        for name in ("color", "type", "gender"))
        )
        head.append("")
        head += [
            f"{h['rank']}. cos {h['sim']:.3f}  attr {h['agree'] or 0:.2f}  "
            f"[{h['type']}] {h['color']}  {h['title']}"
            for h in hits
        ]
        return gallery, "\n".join(head)

    res = search(MODEL, TOKENIZER, IMG_FEATS, ROWS, query, k, DEVICE)
    gallery = [(r["path"], f"{r['rank']}. {r['score']:.3f} · {r['type']}")
               for r in res]
    head += [
        f"{r['rank']}. {r['score']:.3f}  [{r['type']}] {r['color']}  {r['title']}"
        for r in res
    ]
    return gallery, "\n".join(head)


with gr.Blocks(title="COMP5523 Shoes Text-to-Image") as demo:
    gr.Markdown(
        "# COMP5523 — Shoes text→image (CLIP)\n"
        f"Fashion Product Images · footwear subset · **{len(ROWS)}** images · CLIP ViT-B/32"
    )
    with gr.Row():
        q = gr.Textbox(label="Text query", value="silver sports shoes", scale=4)
        k = gr.Slider(1, 20, value=8, step=1, label="Top-K", scale=1)
        method = gr.Radio(METHODS, value=METHODS[0], label="Method", scale=2)
        btn = gr.Button("Search", variant="primary", scale=1)

    out_gal = gr.Gallery(label="Top matches", columns=5, height=420, object_fit="contain")
    out_txt = gr.Textbox(label="Ranking", lines=14)
    outputs = [out_gal, out_txt]

    btn.click(run, inputs=[q, k, method], outputs=outputs)
    q.submit(run, inputs=[q, k, method], outputs=outputs)
    method.change(run, inputs=[q, k, method], outputs=outputs)
    gr.Examples(
        examples=[
            ["black leather shoes", 8, METHODS[0]],
            ["silver sports shoes", 8, METHODS[1]],
            ["beige casual shoes", 8, METHODS[1]],
            ["bronze heels", 8, METHODS[1]],
        ],
        inputs=[q, k, method],
    )
    demo.load(lambda: run("silver sports shoes", 8, METHODS[1]), outputs=outputs)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, show_error=True)
