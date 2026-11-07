"""Gradio page: one text box, one ranked list, a switch between the two methods.

Reached through `python main.py web`. The model and the gallery features are
loaded inside build_demo(), so importing this module costs nothing; the page
warms itself with one query on load, which matters because the demonstration is
timed from the moment the screen is shared.
"""
from __future__ import annotations

from . import rerank
from .clipCore import (
    get_device,
    get_image_features,
    load_model,
    load_rows,
    search,
)
from .queryParse import parse_query

METHODS = ["baseline (raw CLIP)", "rerank (attributes)"]


def add_args(sub) -> None:
    sub.add_argument("--port", type=int, default=7860)
    sub.add_argument("--host", default="127.0.0.1")
    sub.add_argument("--share", action="store_true",
                     help="expose the page through a gradio tunnel")


def build_demo():
    import gradio as gr

    rows, paths = load_rows()
    device = get_device()
    model, preprocess, tokenizer = load_model(device)
    img_feats = get_image_features(model, preprocess, paths, device)
    print(f"device: {device}  gallery: {len(rows)} images  page ready")

    def run(query: str, k: int, method: str):
        query = (query or "").strip()
        if not query:
            return [], "Type a query, e.g. black leather shoes"
        k = int(k or 8)
        head = [f"Query: {query}", f"Gallery: {len(rows)} images", ""]

        if method == METHODS[1]:
            parsed = parse_query(query)
            hits = rerank.rerank_detail(model, tokenizer, img_feats, rows,
                                        query, max(k, rerank.RERANK_DEPTH),
                                        device)[:k]
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

        res = search(model, tokenizer, img_feats, rows, query, k, device)
        gallery = [(r["path"], f"{r['rank']}. {r['score']:.3f} · {r['type']}")
                   for r in res]
        head += [
            f"{r['rank']}. {r['score']:.3f}  [{r['type']}] {r['color']}  "
            f"{r['title']}"
            for r in res
        ]
        return gallery, "\n".join(head)

    with gr.Blocks(title="COMP5523 Shoes Text-to-Image") as demo:
        gr.Markdown(
            "# COMP5523 — Shoes text→image (CLIP)\n"
            f"Fashion Product Images · footwear subset · **{len(rows)}** images "
            "· CLIP ViT-B/32"
        )
        with gr.Row():
            q = gr.Textbox(label="Text query", value="silver sports shoes",
                           scale=4)
            k = gr.Slider(1, 20, value=8, step=1, label="Top-K", scale=1)
            method = gr.Radio(METHODS, value=METHODS[0], label="Method", scale=2)
            btn = gr.Button("Search", variant="primary", scale=1)

        out_gal = gr.Gallery(label="Top matches", columns=5, height=420,
                             object_fit="contain")
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
        demo.load(lambda: run("silver sports shoes", 8, METHODS[1]),
                  outputs=outputs)

    return demo


def main(args) -> None:
    build_demo().launch(server_name=args.host, server_port=args.port,
                        share=args.share, show_error=True)
