#!/usr/bin/env python3
"""Click-through sheet for the human relevance judgements that are still open.

The 200 cells in evaluation/sets/judgements.json were judged on the baseline
top-10, which is enough to say how strict the automatic labels are but not
enough to compare two methods: an image the re-ranker promotes into the head
scores 0 until somebody looks at it. This tool lists exactly those cells.

  python main.py label-sheet                 writes the sheet and reports the count
  python main.py label-sheet --merge FILE    folds an exported judgement file back in

The sheet is a local HTML page (gitignored). Open it in a browser, press 1 for
relevant and 0 for not on each highlighted image, then click Export to download
the decisions and merge them.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
QUERIES_PATH = ROOT / "evaluation" / "sets" / "queries.json"
JUDGEMENTS_PATH = ROOT / "evaluation" / "sets" / "judgements.json"
SHEET_PATH = ROOT / "evaluation" / "sets" / "labelSheet.html"
TOPK = 10

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Open judgements — COMP5523 footwear</title>
<style>
 body{font:14px/1.5 -apple-system,Segoe UI,Arial,sans-serif;margin:24px;color:#111}
 h1{font-size:20px} .rule{background:#f6f7f9;border-left:3px solid #ccc;
   padding:10px 14px;white-space:pre-wrap;max-width:900px}
 .bar{position:sticky;top:0;background:#fff;padding:10px 0;border-bottom:1px solid #ddd;
   margin-bottom:16px;z-index:5}
 button{font:inherit;padding:6px 14px;margin-right:8px}
 .q{margin:22px 0 8px;font-weight:600}
 .row{display:flex;gap:8px;flex-wrap:wrap}
 .cell{width:150px;border:1px solid #ddd;padding:4px;text-align:center;background:#fff}
 .cell.open{border-color:#e0a800;background:#fffbef}
 .cell.yes{border-color:#0b6e2e;background:#f0f9f2}
 .cell.no{border-color:#b00020;background:#fdf0f2}
 .cell img{width:140px;height:187px;object-fit:contain;display:block;margin:auto}
 .cell .meta{font-size:11px;color:#555}
 .cell .rank{font-size:11px;color:#888}
 .cell.sel{outline:3px solid #1a73e8}
 kbd{background:#eee;border:1px solid #ccc;border-radius:3px;padding:0 4px}
</style></head><body>
<h1>Open judgements: __COUNT__ cells over __NQ__ queries</h1>
<div class="rule">__RULE__</div>
<div class="bar">
  <span id="progress"></span>
  <button id="export">Export decisions</button>
  <span style="color:#666">select an image, then press <kbd>1</kbd> relevant /
  <kbd>0</kbd> not relevant / <kbd>u</kbd> undo; <kbd>&rarr;</kbd> <kbd>&larr;</kbd> move</span>
</div>
<div id="content"></div>
<script>
const DATA = __DATA__;
const decisions = {};
let order = [], pos = 0;

function render(){
  const root = document.getElementById('content');
  root.innerHTML = '';
  for (const q of DATA){
    const h = document.createElement('div');
    h.className = 'q';
    h.textContent = q.qid + '  —  "' + q.query + '"  (' + q.open + ' open of ' + q.items.length + ')';
    root.appendChild(h);
    const row = document.createElement('div');
    row.className = 'row';
    q.items.forEach((it, i) => {
      const c = document.createElement('div');
      const key = q.qid + '|' + it.id;
      const decided = decisions[key];
      const state = decided === undefined ? (it.judged === null ? 'open' : (it.judged ? 'yes' : 'no'))
                                          : (decided ? 'yes' : 'no');
      c.className = 'cell ' + state + (it.open ? ' clickable' : '');
      c.innerHTML = '<div class="rank">#' + it.rank + (it.open ? ' · judge me' : ' · already judged') + '</div>'
        + '<img src="../../' + it.path + '" loading="lazy">'
        + '<div class="meta">' + it.color + ' · ' + it.type + '</div>';
      if (it.open){
        order.push(key);
        c.dataset.key = key;
        c.onclick = () => { pos = order.indexOf(key); mark(); };
      }
      row.appendChild(c);
    });
    root.appendChild(row);
  }
  update();
}

function mark(){
  document.querySelectorAll('.cell.sel').forEach(e => e.classList.remove('sel'));
  const el = document.querySelector('[data-key="' + CSS.escape(order[pos]) + '"]');
  if (el){ el.classList.add('sel'); el.scrollIntoView({block:'center', behavior:'smooth'}); }
  update();
}

function update(){
  const done = Object.keys(decisions).length;
  document.getElementById('progress').textContent =
    done + ' / ' + order.length + ' decided';
}

function set(value){
  const key = order[pos];
  if (!key) return;
  if (value === null) delete decisions[key]; else decisions[key] = value;
  paint();
  if (value !== null && pos < order.length - 1) pos++;
  mark();
}

function paint(){
  document.querySelectorAll('[data-key]').forEach(el => {
    const d = decisions[el.dataset.key];
    el.classList.remove('yes','no','open');
    el.classList.add(d === undefined ? 'open' : (d ? 'yes' : 'no'));
  });
  update();
}

document.addEventListener('keydown', e => {
  if (e.key === '1') set(1);
  else if (e.key === '0') set(0);
  else if (e.key === 'u') set(null);
  else if (e.key === 'ArrowRight') { pos = Math.min(pos + 1, order.length - 1); mark(); }
  else if (e.key === 'ArrowLeft') { pos = Math.max(pos - 1, 0); mark(); }
});

document.getElementById('export').onclick = () => {
  const out = {};
  for (const [key, value] of Object.entries(decisions)){
    const [qid, id] = key.split('|');
    (out[qid] = out[qid] || {})[id] = value;
  }
  const blob = new Blob([JSON.stringify(out, null, 2)], {type:'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'judgements_new.json';
  a.click();
};

render();
mark();
</script></body></html>
"""


def add_args(sub) -> None:
    sub.add_argument("--merge", type=Path, default=None,
                     help="fold an exported judgements_new.json into "
                          "evaluation/sets/judgements.json")
    sub.add_argument("--out", type=Path, default=SHEET_PATH,
                     help="where to write the sheet (default: "
                          "evaluation/sets/labelSheet.html)")
    sub.add_argument("--method", choices=["rerank", "attr_only"], default="rerank",
                     help="which ranking to judge the head of (default: rerank)")


def open_cells() -> tuple[list[dict], dict]:
    """The judged queries, with the head cells nobody has looked at flagged."""
    from . import runEvaluation as ev

    judgements = json.loads(JUDGEMENTS_PATH.read_text(encoding="utf-8"))
    judged = judgements["judged"]
    queries = {q["qid"]: q
               for q in json.loads(QUERIES_PATH.read_text(encoding="utf-8"))
               ["queries"]}
    ev.load_context()
    rank = ev.METHODS["rerank"]
    rows = {r["id"]: r for r in ev._CTX["rows"]}

    sheets = []
    for qid, cells in judged.items():
        q = queries[qid]
        ranked = rank(q, TOPK)[:TOPK]
        items = []
        for position, iid in enumerate(ranked, 1):
            row = rows[iid]
            items.append({
                "id": iid,
                "rank": position,
                "path": row["path"],
                "color": row["color"],
                "type": row["articleType"],
                "judged": cells.get(iid),
                "open": iid not in cells,
            })
        sheets.append({"qid": qid, "query": q["query"], "items": items,
                       "open": sum(1 for i in items if i["open"])})
    return sheets, judgements


def build(args) -> None:
    sheets, judgements = open_cells()
    total = sum(s["open"] for s in sheets)
    page = (PAGE.replace("__DATA__", json.dumps(sheets, ensure_ascii=False))
                .replace("__COUNT__", str(total))
                .replace("__NQ__", str(len(sheets)))
                .replace("__RULE__", judgements["rule"]))
    args.out.write_text(page, encoding="utf-8")
    print(f"{total} open cells over {len(sheets)} queries "
          f"(head judged: {args.method} top-{TOPK})")
    for s in sheets:
        if s["open"]:
            print(f"  {s['qid']:<28} {s['open']} open   '{s['query']}'")
    print(f"\n-> {args.out}\n   open it in a browser, then merge the export with:\n"
          f"   python main.py label-sheet --merge ~/Downloads/judgements_new.json")


def merge(args) -> None:
    new = json.loads(args.merge.read_text(encoding="utf-8"))
    judgements = json.loads(JUDGEMENTS_PATH.read_text(encoding="utf-8"))
    judged = judgements["judged"]
    added = 0
    for qid, cells in new.items():
        if qid not in judged:
            raise SystemExit(f"{qid} is not one of the judged queries")
        for iid, value in cells.items():
            if int(value) not in (0, 1):
                raise SystemExit(f"{qid}/{iid}: value must be 0 or 1")
            judged[qid][str(iid)] = int(value)
            added += 1
    today = dt.date.today().isoformat()
    judgements["note"] += (f" Extended {today}: {added} cells added over the "
                           f"{args.method} top-{TOPK} so the head of both methods "
                           "is judged on the same queries.")
    judgements.setdefault("extension", []).append(
        {"date": today, "cells": added, "method": args.method,
         "source": str(args.merge)})
    JUDGEMENTS_PATH.write_text(
        json.dumps(judgements, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    total = sum(len(v) for v in judged.values())
    print(f"merged {added} cells; judgements.json now holds {total} cells "
          f"over {len(judged)} queries")
    print("re-run the strict scoring for both methods:")
    print("  python main.py eval --method baseline --labels strict")
    print("  python main.py eval --method rerank --labels strict")
    print("  python main.py compare results/baseline_strict.json "
          "results/rerank_strict.json")


def main(args) -> None:
    if args.merge is not None:
        merge(args)
    else:
        build(args)
