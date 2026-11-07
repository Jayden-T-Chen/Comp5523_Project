# COMP5523 group project: text-to-image retrieval for footwear

A frozen CLIP ViT-B/32 ranks 1,000 catalogue photographs of shoes for a shopper
query such as "silver sports shoes". On top of the cosine ranking we parse the
query into catalogue attributes (colour, article type, gender) and add a
zero-shot attribute agreement term, which lifts mAP@50 from 0.348 to 0.408 on
our 60-query set. Nothing is trained; the gallery, the query sets and every
result file ship with the repository.

The reasoning behind each number, the failure analysis and the draft of the
8-page report live in `docs/projectReport.md`. This file only says what is in
the repository and how to run it.

## Layout

| Directory | What it holds |
| --- | --- |
| `data/` | the 1,000-image gallery and its attribute table; `image_features.pt` is a rebuildable cache, not in git |
| `dataset/` | how the 1,000 were sampled from Kaggle *Fashion Product Images*, and the licence |
| `retrieval/` | CLIP core, query parser, attribute re-ranking, command-line and Gradio interfaces |
| `evaluation/sets/` | four frozen query sets and the 200 human judgements |
| `evaluation/tools/` | scoring, paired comparison, ablations, the attribute and encoder probes, the figure |
| `evaluation/montages/` | the failure-case figure used by the report and the slides |
| `results/` | one json per method and label set; `results/ablations/` holds the two sweeps and `results/comparisons/` the archived confidence intervals |
| `docs/` | the report draft, including the internal presentation and formatting plans |
| `tests/` | unit tests for the parser, the metric, the fusion arithmetic and the cache signature |

`retrieval/` never reads the evaluation sets, and `evaluation/tools/` imports the
ranking code from `retrieval/` instead of reimplementing it, so a number in
`results/` always comes from the code that ships.

## Setup

```bash
git clone https://github.com/Jayden-T-Chen/Comp5523_Project.git comp5523project
cd comp5523project
python3.12 -m venv .venv && source .venv/bin/activate
python -m pip install -U pip
python -m pip install torch==2.14.1 torchvision==0.29.1 open-clip-torch==3.3.0 \
                     gradio==6.31.0 Pillow==12.3.0 numpy==2.5.4
```

On Windows, create and activate the environment with `py -3.12 -m venv .venv` and
`.venv\Scripts\Activate.ps1`; the install line and every command below are the same.
Python 3.11 or 3.12. The first run downloads the CLIP weights (a few hundred MB)
into `~/.cache`; the gallery itself needs no download. On an M1 with the feature
cache present, cold start to the first result takes about 30 s.

Loading the model prints one warning, `QuickGELU mismatch between final model
config and pretrained tag 'openai'`: open_clip builds the text tower with
ordinary GELU while OpenAI trained these weights with QuickGELU. The repository
keeps the default configuration, because the 200 human judgements were collected
on the ranking it produces, and `python main.py encoder-check` measures how far
the two differ (`results/encoderCheck.json`, report §7.3).

## Commands

Everything goes through one entry point.

| Command | Writes | Reproduces a number in the report |
| --- | --- | --- |
| `python main.py search --method rerank --top 5` | prints the ranking with both score components | no |
| `python main.py web` | the Gradio page on 127.0.0.1:7860 | no |
| `python main.py eval --method baseline` | `results/baseline_auto.json` | yes, §6.1 |
| `python main.py eval --method prompt` | `results/prompt_auto.json` | yes, §6.8 |
| `python main.py eval --method rerank` | `results/rerank_auto.json` | yes, §6.1 |
| `python main.py eval --method attr_only` | `results/attr_only_auto.json` | yes, §6.1 |
| `python main.py eval --method metadata` | `results/metadata_auto.json` | yes, §6.1 (upper bound) |
| `python main.py eval --method baseline --labels strict` | `results/baseline_strict.json` | yes, §6.5 |
| `python main.py eval --method rerank --labels strict` | `results/rerank_strict.json` | yes, §6.5 |
| `python main.py eval --method rerank --parser legacy` | `results/rerank_legacy_auto.json` | yes, §6.3 |
| `python main.py eval --method rerank --queries evaluation/sets/queries_family.json` | `results/rerank_auto_family.json` | yes, §6.5 |
| `python main.py eval --method rerank --queries evaluation/sets/queries_heldout.json` | `results/rerank_auto_heldout.json` | yes, §6.6 |
| `python main.py compare results/baseline_auto.json results/rerank_auto.json` | prints Δ, 95% CI, win/loss/tie; `--out results/comparisons/auto.json` archives them | yes, §6.1 |
| `python main.py ablate-parser` | `results/ablations/parser_ablation.json` | yes, §6.3 |
| `python main.py ablate-fusion` | `results/ablations/fusion_ablation.json` | yes, §6.4 |
| `python main.py probe` | `results/attribute_probe.json` | yes, §3.3, §4.4, §4.5 and §6.1 |
| `python main.py encoder-check` | `results/encoderCheck.json` | yes, §7.3 |
| `python main.py figure` | `evaluation/montages/failureCases.png` | yes, §6.7 |
| `python main.py label-sheet` | a local click-through sheet for the open judgements | no |
| `python main.py test` | runs the unit tests | no |

The family, sandal and held-out rows repeat for `baseline` as well; the report
quotes both methods on each set, and Appendix A lists the six `compare --out`
calls behind `results/comparisons/`. `python main.py` with no arguments lists the
commands, and every command takes `--help`.

## The headline numbers

All on the 60 main queries with automatic labels, depth 50, top 10.

| Arm | mAP@50 | P@10 | What it is |
| --- | --- | --- | --- |
| `baseline` | 0.3481 | 0.5817 | frozen CLIP cosine |
| `prompt` | 0.3302 | 0.5633 | cosine over a mean of six prompt phrasings |
| `rerank` | 0.4078 | 0.6283 | cosine plus the attribute term, pool 100 |
| `attr_only` | 0.4086 | 0.6133 | the attribute term alone, cosine breaking ties |
| `metadata` | 1.0000 | 1.0000 | a catalogue lookup on the ground-truth attributes |

The last row is an upper bound on the automatic oracle, not a system we would
ship: it reads the labels. It is in the table because it says what the oracle
can and cannot measure, and §6.1 of the report discusses it. `attr_only` next to
`rerank` says the same thing about the fusion: the attribute term carries the
deep ranking, the cosine term carries the head.

Paired bootstrap over queries, 10,000 resamples, seed 0: `rerank` against
`baseline` gives ΔmAP@50 +0.0597 (95% CI +0.0413 to +0.0816, 49 wins 11 losses)
and ΔP@10 +0.0467 (CI +0.0150 to +0.0800). On the 20 queries with human
judgements the baseline head scores P@10 0.925 while the same head scores 0.625
under automatic labels; 54 cells of the re-ranked head still wait for a human
look, and because those cells count as zero the re-ranked head reads 0.675
against 0.925 — of the 146 cells a person did judge, 135 are relevant, the same
0.925 rate as the baseline. Until the 54 are filled the strict comparison stays
incomplete (§6.5).

Per query both arms take about 37 ms on an M1 laptop, with the two arms'
round-to-round spreads on top of each other, so re-ranking adds no measurable
latency; the one-off costs are 1.8 s for the attribute tables and 41 s to
encode the gallery with an empty cache. The zero-shot colour
table agrees with the catalogue colour on 37% of images (top-2: 51%), the type
table on 40% (64%), gender on 86%; `results/attribute_probe.json` holds the
full table.

## Status

Done: the gallery and its provenance, four query sets, the 200 human judgements,
all five scoring arms, the two ablations, the two probes, the archived
confidence intervals, the failure figure, the unit tests and the report draft.
Open: the 54 remaining human judgements, the cut from the measured 10.2 laid-out
pages to the 8-page limit that §11.1 itemises, the PDF and its two figures, the
slides and the 90-second recording, the AI-use statement and the member
contribution table in §11, and the peer review after the demonstration. The
demonstration is on 18 November 2026 at 18:30 in TU101, seven minutes including
setup; the report is due 25 November 2026.

The images come from Kaggle *Fashion Product Images* (MIT) and are used for this
course project only; see `dataset/NOTICE`.
