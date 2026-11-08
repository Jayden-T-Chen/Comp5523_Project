"""Gradio page: one query, one ranked list, and the evidence behind each card.

Reached through `python main.py web`. The model and the gallery features are
loaded inside build_demo(), so importing this module costs nothing; the same
function also builds the first page state, so the page paints a ranked list
before the browser asks anything, which matters because the demonstration is
timed from the moment the screen is shared.

The list is the shipped score: cosine plus beta times the attribute agreement,
over the cosine top-100 pool. Each card prints the catalogue values the score
read, marks the ones the query named, and two checkboxes add the cosine and
agreement numbers and the rank cosine alone gave. rerank_detail() returns the
pool and the fused ordering in one call, so nothing is recomputed on the way
and the numbers on the page are the numbers `main.py search --method rerank`
prints.
"""
from __future__ import annotations

import time

from . import rerank
from . import webStyle as style
from .clipCore import (
    CACHE_PATH,
    ROOT,
    get_device,
    get_image_features,
    load_model,
    load_rows,
)
from .queryParse import parse_query

GALLERY_DIR = ROOT / "data" / "gallery"
PANE_NAME = "Ranked list"
FUSED_NOTE = f"cosine + {rerank.BETA} x attribute agreement over the cosine" \
             f" top {rerank.RERANK_DEPTH}"
# The five chips an audience can click. Each one was measured on the shipped
# catalogue: all eight images in its fused head carry a colour and a type the
# agreement term accepts, the eight titles are eight different shoes, and the
# thumbnails were compared with the words one by one, because the catalogue's
# own colour field is sometimes wrong. Between them they cover a colour with a
# type word that maps to two catalogue types, a colour whose family does the
# work, a material word the parser does not know (`leather`), a colour alone,
# and a type with a gender and no colour.
SAMPLES = ["green sneakers", "grey sports shoes", "brown leather loafers",
           "maroon shoes", "flip flops for men"]


def light_only(theme):
    """Pin every dark-mode theme variable to its light value.

    Gradio adds a `dark` class to the page when the browser asks for a dark
    scheme, and its theme then swaps 221 CSS variables, which would repaint the
    cards and the controls in the opposite palette. The design brief settled on
    one light neutral palette, so the swap is switched off variable by variable
    before launch; the stylesheet in webStyle.CSS is left to decide the colours.
    """
    for attr, value in list(vars(theme).items()):
        if attr.startswith("_") or attr.endswith("_dark"):
            continue
        setattr(theme, f"{attr}_dark", value)
    return theme


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

    def run(query: str, k: int, raw: bool, ranks: bool):
        query = (query or "").strip()
        k = max(1, min(int(k or 8), 20))
        blank = style.slots({"color": [], "type": [], "gender": None})
        if not query:
            note = ("<div class='pane'>Nothing to rank yet. Type a query, or"
                    " click one of the sample chips below the box.</div>")
            return (blank, note,
                    style.status("", 0.0, device, CACHE_PATH.is_file(), False))

        started = time.perf_counter()
        parsed = parse_query(query)
        detail = rerank.rerank_detail(model, tokenizer, img_feats, rows, query,
                                      max(k, rerank.RERANK_DEPTH), device)
        cards = style.ranking(detail, k)
        ms = (time.perf_counter() - started) * 1000
        named = bool(parsed["color"] or parsed["type"] or parsed["gender"])
        return (style.slots(parsed) + style.tag_legend(ranks),
                style.pane(cards, PANE_NAME, FUSED_NOTE, parsed, raw, ranks),
                style.status(query, ms, device, CACHE_PATH.is_file(), named,
                             ranks))

    # Twice, so the timings the page prints are warm ones: the first pass
    # touches the text tower and the torch caches, the second is what an
    # audience sees. Running it here rather than on page load means the first
    # paint already carries a ranked list, which matters because the
    # demonstration is timed from the moment the screen is shared.
    run(SAMPLES[0], 8, False, False)
    first = run(SAMPLES[0], 8, False, False)

    with gr.Blocks(title="COMP5523 footwear retrieval") as demo:
        gr.HTML(
            "<div id='head'><h1>Footwear text-to-image retrieval</h1></div>"
            f"<div id='meta'>{len(rows):,} gallery images &middot; CLIP ViT-B/32"
            " frozen &middot; re-ranking the cosine top"
            f" {rerank.RERANK_DEPTH} with &beta; = {rerank.BETA}</div>"
        )
        with gr.Row():
            q = gr.Textbox(label="Query", value=SAMPLES[0], scale=6,
                           placeholder=SAMPLES[1])
            k = gr.Slider(1, 20, value=8, step=1, label="Head size", scale=2)
            btn = gr.Button("Search", variant="primary", scale=1)
        # Their own row, because next to the slider each label wrapped onto two
        # lines and a wrapped label reads like a different control.
        with gr.Row():
            show = gr.Checkbox(label="Show cos and attr", value=False)
            ranks = gr.Checkbox(label="Show cosine rank", value=False)
        inputs = [q, k, show, ranks]
        gr.Examples(examples=[[s] for s in SAMPLES], inputs=[q],
                    label="Sample queries")
        out_slots = gr.HTML(value=first[0])
        out_pane = gr.HTML(value=first[1])
        out_status = gr.HTML(value=first[2])

        outputs = [out_slots, out_pane, out_status]
        btn.click(run, inputs, outputs)
        q.submit(run, inputs, outputs)
        k.release(run, inputs, outputs)
        show.change(run, inputs, outputs)
        ranks.change(run, inputs, outputs)

    return demo


def main(args) -> None:
    import gradio as gr

    # Gradio 6 takes the theme and the stylesheet at launch, not on Blocks.
    # The palette is white with one green accent, so the controls are pulled to
    # the same green the matched tags use and the component labels come back to
    # grey.
    theme = light_only(
        gr.themes.Soft(primary_hue="green", neutral_hue="slate",
                       radius_size="md", text_size="md")
        .set(block_label_background_fill="#f1f2f4",
             block_label_text_color="#4b5563",
             block_title_text_color="#4b5563",
             button_primary_background_fill=style.ACCENT,
             button_primary_background_fill_hover="#126a33",
             button_primary_border_color=style.ACCENT,
             input_border_color_focus=style.ACCENT_LINE,
             slider_color=style.ACCENT))
    build_demo().launch(server_name=args.host, server_port=args.port,
                        share=args.share, show_error=True,
                        allowed_paths=[str(GALLERY_DIR)],
                        theme=theme, css=style.CSS, footer_links=[])
