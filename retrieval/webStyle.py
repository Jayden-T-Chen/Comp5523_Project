"""Presentation layer for `python main.py web`: styles and card markup.

The page answers one question per query: what the words were read as, and why
each image sits where it does. The parser's three slots are printed above the
list, every card carries its catalogue values with the matched ones marked, and
the cosine and agreement numbers behind the order sit behind one checkbox. The
order is the shipped score, so every number here is a number
`main.py search --method rerank` prints.

This module owns no model and no scoring. It takes the detail rows the
retrieval side already returns and renders HTML. rerank.rerank_detail() hands
over the cosine top-100 pool along with the fused ordering of that same pool,
which is where the optional cosine rank comes from.
"""
from __future__ import annotations

from html import escape

# Gradio 6 serves files from the API prefix, and launch(allowed_paths=[...])
# decides which of them it will actually hand out.
FILE_ROUTE = "/gradio_api/file="

ACCENT = "#15803d"
ACCENT_SOFT = "#eaf5ee"
ACCENT_LINE = "#b7dcc4"

CSS = """
:root {
  --page: #ffffff;
  --ink: #1a1a1a;
  --muted: #6b7280;
  --line: #e4e7ec;
  --edge: #d6dbe3;
  --accent: #15803d;
  --accent-soft: #eaf5ee;
  --accent-line: #b7dcc4;
  --gap: 16px;
}
/* Native form controls follow the OS palette, so pin them to light. */
:root, :root.dark, :root .dark { color-scheme: light; }
body, .gradio-container { background: var(--page) !important; }
.gradio-container { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
#head { display: flex; align-items: baseline; gap: 12px; padding: 4px 0 0; }
#head h1 { font-size: 28px; line-height: 1.1; margin: 0; color: var(--ink);
           font-weight: 650; letter-spacing: -0.2px; }
#meta { color: var(--muted); font-size: 13px; padding: 2px 0 12px;
        border-bottom: 1px solid var(--line); }
#slots { display: flex; gap: 8px; flex-wrap: wrap; padding: 12px 0 4px; }
.chip { border: 1px solid var(--line); border-radius: 8px; padding: 3px 10px;
        font-size: 13px; background: #fff; color: var(--muted); }
.chip b { color: var(--ink); font-weight: 600; }
.chip.hit { border-color: var(--accent-line); background: var(--accent-soft);
            color: var(--accent); }
.chip.hit b { color: var(--accent); }
#legend { font-size: 12px; color: var(--muted); padding: 6px 0 2px; }
#legend span { margin-right: 16px; white-space: nowrap; }
#legend b { color: var(--ink); font-weight: 600; }
.pane { background: #fff; border: 1px solid var(--edge); border-radius: 8px;
        padding: 16px; }
.pane h2 { font-size: 16px; margin: 0 0 2px; color: var(--ink); font-weight: 600; }
.pane p.sub { font-size: 13px; color: var(--muted); margin: 0 0 12px; }
.grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: var(--gap); }
@media (max-width: 900px) { .grid { grid-template-columns: repeat(2,
        minmax(0, 1fr)); } }
.card { border: 1px solid var(--edge); border-radius: 8px; background: #fff;
        overflow: hidden; }
.shot { position: relative; background: #fff; }
.shot img { display: block; width: 100%; aspect-ratio: 3 / 4; object-fit: contain;
            background: #fff; }
.rank { position: absolute; left: 6px; top: 6px; background: rgba(26,26,26,.82);
        color: #fff; border-radius: 8px; font-size: 12px; padding: 1px 7px;
        font-variant-numeric: tabular-nums; }
.cosrank { position: absolute; right: 6px; top: 6px; background: #fff;
           border: 1px solid var(--edge); color: var(--muted); border-radius: 8px;
           font-size: 12px; padding: 1px 7px; font-variant-numeric: tabular-nums; }
.text { padding: 8px 10px 10px; border-top: 1px solid var(--line); }
.title { font-size: 13px; line-height: 1.25; color: var(--ink);
         display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;
         overflow: hidden; min-height: 33px; }
.tags { display: flex; gap: 6px; margin-top: 6px; flex-wrap: wrap; }
.tag { font-size: 12px; color: var(--muted); background: #f6f7f9;
       border: 1px solid transparent; border-radius: 8px; padding: 1px 7px; }
.tag.named { color: var(--accent); background: var(--accent-soft);
             border-color: var(--accent-line); }
.tag.family { color: var(--accent); background: #fff;
              border: 1px dashed var(--accent-line); }
.raw { display: none; margin-top: 6px; font-size: 12px;
       color: var(--muted); font-variant-numeric: tabular-nums; }
.showraw .raw { display: block; }
input[type="checkbox"] { accent-color: var(--accent);
                         border: 1px solid #c3c8d1 !important; background: #fff; }
input[type="checkbox"]:checked { border-color: var(--accent) !important; }
.mono textarea { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas,
                 monospace; font-size: 12px; line-height: 1.35; }
#status { font-size: 13px; color: var(--muted); padding-top: 8px; }
@media (prefers-reduced-motion: no-preference) {
  .card { transition: box-shadow 120ms ease; }
  .card:hover { box-shadow: 0 1px 4px rgba(26, 26, 26, .12); }
}
"""


def ranks_by_cosine(detail: list[dict]) -> dict:
    """Cosine rank of every candidate in the pool, by image id.

    The pool is the cosine top-100, so sorting it by `sim` recovers each
    image's rank against the whole gallery, which is the number the optional
    badge prints.
    """
    return {d["id"]: pos + 1
            for pos, d in enumerate(sorted(detail, key=lambda d: -d["sim"]))}


def ranking(detail: list[dict], k: int) -> list[dict]:
    """The shipped head of k, each card knowing where cosine alone put it."""
    cosine = ranks_by_cosine(detail)
    return [dict(d, cos_rank=cosine[d["id"]]) for d in detail[:k]]


def image_url(path: str) -> str:
    return FILE_ROUTE + path


def tag(item: dict, slot: str, parsed: dict) -> str:
    """One catalogue value, shaded by how the query treated it.

    The three states are the whole explanation of a card's position: a value
    the query names, a family value the agreement term still accepts at partial
    weight, and a value the query said nothing about.
    """
    value = item.get(slot) or ""
    if not value:
        return ""
    named, family = split_values(parsed, slot)
    cls = "tag named" if value in named else \
        "tag family" if value in family else "tag"
    return f"<span class='{cls}'>{escape(value)}</span>"


def card(item: dict, parsed: dict = None, raw: bool = False,
         show_rank: bool = False) -> str:
    """One thumbnail with its rank, its catalogue values and its score parts.

    The numbers are always in the markup; the grid class decides whether they
    are visible, so expanding them costs no re-query. Only cosine and
    agreement are printed, because those are the two numbers the command line
    prints; the order they explain is the sum of them.
    """
    parsed = parsed or {}
    badge = ""
    if show_rank:
        badge = f"<span class='cosrank'>cos #{item['cos_rank']}</span>"
    scores = f"cos {item['sim']:.3f}"
    if item.get("agree") is not None:
        scores += f" · attr {item['agree']:.2f}"
    tags = "".join(tag(item, slot, parsed)
                   for slot in ("type", "color", "gender"))
    return (
        "<div class='card'>"
        f"<div class='shot'><img loading='lazy' src='{image_url(item['path'])}'>"
        f"<span class='rank'>{item['rank']}</span>{badge}</div>"
        "<div class='text'>"
        f"<div class='title' title='{escape(item['title'])}'>"
        f"{escape(item['title'])}</div>"
        f"<div class='tags'>{tags}</div>"
        f"<div class='raw'>{scores}</div>"
        "</div></div>"
    )


def pane(cards: list[dict], name: str, note: str, parsed: dict = None,
         raw: bool = False, show_rank: bool = False) -> str:
    grid = "".join(card(c, parsed, raw, show_rank) for c in cards)
    return (f"<div class='pane'><h2>{escape(name)}</h2>"
            f"<p class='sub'>{escape(note)}</p>"
            f"<div class='grid{' showraw' if raw else ''}'>{grid}</div></div>")


def split_values(parsed: dict, name: str) -> tuple[list, list]:
    """The value the query names outright, and the family values beside it.

    parse_query() returns both in one list; the agreement weights say which is
    which, since a family value scores below 1.0.
    """
    weights = (parsed.get("weights") or {}).get(name) or {}
    values = parsed.get(name) or []
    if isinstance(values, str):
        values = [values]
    named = [v for v in values if weights.get(v, 1.0) >= 1.0]
    return named, [v for v in values if weights.get(v, 1.0) < 1.0]


def slots(parsed: dict) -> str:
    """What the dictionary parser found, so a miss is visible as a miss.

    Each chip carries only the value the query named; the family values the
    agreement term also accepts are listed apart, because they are the reason a
    "silver" query can surface a metallic shoe.
    """
    chips = []
    soft = []
    for name in ("color", "type", "gender"):
        named, family = split_values(parsed, name)
        soft += family
        shown = ", ".join(named) if named else "—"
        chips.append(f"<span class='chip{' hit' if named else ''}'>"
                     f"{name} <b>{escape(shown)}</b></span>")
    if soft:
        chips.append("<span class='chip'>also accepted <b>"
                     f"{escape(', '.join(sorted(set(soft))))}</b></span>")
    return "<div id='slots'>" + "".join(chips) + "</div>"


def tag_legend(show_rank: bool = False) -> str:
    """What the three tag shades mean, spelled out once above the list."""
    parts = ["<span><b>green</b> the value the query names</span>",
             "<span><b>dashed</b> family value, accepted at partial weight"
             "</span>",
             "<span><b>grey</b> not named by the query</span>"]
    if show_rank:
        parts.append("<span><b>cos #n</b> where cosine alone ranked it</span>")
    return "<div id='legend'>" + "".join(parts) + "</div>"


def accepted(parsed: dict) -> dict:
    """Every catalogue value the agreement term can score, per slot."""
    out = {}
    for name in ("color", "type"):
        values = list(parsed.get(name) or [])
        values += sorted((parsed.get("weights") or {}).get(name, {}))
        out[name] = set(values)
    return out


def status(query: str, ms: float, device: str, cached: bool, named: bool,
           show_rank: bool = False) -> str:
    """One line of what ran and how it was scored, for the demo and markers."""
    if not query:
        return ("<div id='status'>Type a query, or click one of the sample"
                " chips, and the ranked list appears here.</div>")
    cache = "feature cache hit" if cached else "feature cache rebuilt"
    parts = [f"<b>{escape(query)}</b>", f"{ms:.0f} ms", device, cache]
    if not named:
        parts.append("no attribute named, so the order is cosine alone")
    elif show_rank:
        parts.append("cosine ranks shown")
    return "<div id='status'>" + " · ".join(parts) + "</div>"
