#!/usr/bin/env python3
"""Gradio UI: text -> image ranking over the footwear gallery with CLIP."""
from __future__ import annotations

import gradio as gr

from clipCore import (
    get_device,
    get_image_features,
    load_model,
    load_rows,
    search,
)

ROWS, PATHS = load_rows()
DEVICE = get_device()
MODEL, PREPROCESS, TOKENIZER = load_model(DEVICE)
IMG_FEATS = get_image_features(MODEL, PREPROCESS, PATHS, DEVICE)


def run(query: str, k: int):
    query = (query or "").strip()
    if not query:
        return [], "Type a query, e.g. black leather shoes"
    res = search(MODEL, TOKENIZER, IMG_FEATS, ROWS, query, int(k or 8), DEVICE)
    gallery = [(r["path"], f"{r['rank']}. {r['score']:.3f} · {r['type']}") for r in res]
    lines = [f"Query: {query}", f"Gallery: {len(ROWS)} images", ""]
    lines += [
        f"{r['rank']}. {r['score']:.3f}  [{r['type']}] {r['color']}  {r['title']}"
        for r in res
    ]
    return gallery, "\n".join(lines)


with gr.Blocks(title="COMP5523 Shoes Text-to-Image") as demo:
    gr.Markdown(
        "# COMP5523 — Shoes text→image (CLIP)\n"
        f"Fashion Product Images (Small) · footwear subset · **{len(ROWS)}** images · CLIP ViT-B/32"
    )
    with gr.Row():
        q = gr.Textbox(label="Text query", value="black leather shoes", scale=4)
        k = gr.Slider(1, 20, value=8, step=1, label="Top-K", scale=1)
        btn = gr.Button("Search", variant="primary", scale=1)

    out_gal = gr.Gallery(label="Top matches", columns=5, height=420, object_fit="contain")
    out_txt = gr.Textbox(label="Ranking", lines=14)
    outputs = [out_gal, out_txt]

    btn.click(run, inputs=[q, k], outputs=outputs)
    q.submit(run, inputs=[q, k], outputs=outputs)
    gr.Examples(
        examples=[
            ["black leather shoes"],
            ["white sneakers"],
            ["red sandals"],
            ["blue flip flops"],
            ["brown heels"],
        ],
        inputs=q,
    )
    demo.load(lambda: run("black leather shoes", 8), outputs=outputs)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, show_error=True)
