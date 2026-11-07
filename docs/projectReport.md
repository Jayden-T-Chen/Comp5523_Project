# Domain-Specific Text-to-Image Retrieval for Footwear

**COMP5523 Computer Vision — Group Project, 2026/27 Semester 1**
Demonstration: 18 November 2026, TU101 · Report deadline: 25 November 2026

> Working document. Sections 1–8 plus the references are the text of the 8-page report;
> sections 0 and 9–11 are internal working material and stay out of the submitted PDF.
> Every number below is archived in `results/` and can be regenerated with the commands in
> Appendix A.

---

## 0. Alignment with the brief (internal)

| Requirement in the brief | Where it is addressed | What we point to |
| --- | --- | --- |
| Build a text-to-image retrieval system for a domain of your choice | §1, §3–§4 | 1,000-image footwear gallery; one text box, one ranked list |
| Input: a natural-language query; output: a ranked list of relevant images | §4.1, §4.6 | CLI and Gradio; images with cosine and attribute scores |
| Define relevance and explain how the system measures it | §5.1–§5.3 | Metadata labels, 200 human judgements, three metrics, and the limit of each (§6.1, §6.5) |
| Explain who will use it and why | §1.1 | Shoppers who know appearance but not SKUs; catalogue operators |
| Cross-modal representation | §4.1 | Frozen CLIP ViT-B/32, 512-d, L2-normalised |
| Image descriptions | §2, §4.1 | Direct image encoding; captioning adds a lossy layer |
| Similarity search and indexing | §4.5 | Cosine over 1,000 cached embeddings |
| Domain adaptation | §4.2–§4.4 | Query parsing and zero-shot attribute re-ranking |
| Re-ranking | §4.4 | `cos + β·Π(weighted P)` over the top-100 |
| Query understanding | §4.2 | Three slots, alias tables, longest-phrase masking, family weights |
| User interface | §4.6 | Gradio page with a baseline / re-rank switch |
| Custom test set, human annotation, metrics, protocol, failure analysis | §5–§6 | 60 queries, 200 labelled cells (54 still open), mAP@50 / P@10 / R@10, paired bootstrap, five cases |
| Efficiency evaluation (optional) | §4.5, §7.2 | Measured: 37 ms per query for both arms, tables 1.8 s once, gallery 41 s once (`results/attribute_probe.json`) |
| 7-minute demonstration and 8-page report | §10–§11 | Timing plan, page budget, APA, disclosure, contributions |

The six marking dimensions (Appropriateness, Soundness, Excitement, Presentation, Writing,
Peer Review, 2.5% each) are mapped to evidence in §9.

---

## Abstract

We study text-to-image retrieval in one product domain, footwear. A frozen CLIP ViT-B/32 encodes
a 1,000-image catalogue and ranks images by cosine similarity to a shopper's phrase. On top of
that baseline we add a query parser and a zero-shot attribute re-ranker: the parser reads a phrase
into colour, type and gender constraints from the catalogue vocabulary, CLIP text prompts turn the
same attributes into per-image probability tables, and a weighted product of those probabilities is
added to the cosine score. Over 60 queries mAP@50 rises from 0.348 to 0.408 (+17.2%, paired
bootstrap 95% CI +0.041 to +0.082) and P@10 from 0.582 to 0.628 (+0.047, CI +0.015 to +0.080). Two
diagnostic arms bound that gain: the attribute term alone reaches the same mAP, and a lookup on the
ground-truth labels reaches 1.000, so the automatic oracle is saturated by construction. A
six-template prompt ensemble loses 5.1%. We calibrate the automatic labels against 200 human
judgements, measure query latency, and diagnose five failures.

---

## 1. Introduction

### 1.1 Task and users

Text-to-image retrieval returns images that match a free-form description. In a product catalogue
the task is practical: a shopper can describe appearance — grey casual shoes, silver sports shoes —
but rarely knows a model name or a stock code, and keyword search over catalogue fields depends on
naming that varies between brands and leaves many visual properties untagged.

Footwear suits a course-scale study because its attributes are visually meaningful and comparable
across products: eight article types, forty colour values, and a colour space where neighbours such
as grey, silver and charcoal are easy for a person to separate and hard for a text encoder. Two
groups use the system. Shoppers search by appearance when they cannot name a product; catalogue
operators assemble or check a collection. Both look at the top of the list first, which is why we
report P@10 alongside mAP@50.

### 1.2 What the system does

Given a query, the system returns a ranked list of gallery images, each with its cosine
similarity and, in the re-ranked list, an attribute agreement score. Every arm we compare shares
the gallery, the image features and the truncation. The Gradio page switches between baseline and
re-rank on the same query, so the difference is visible during the demonstration.

### 1.3 Contributions

1. A reproducible benchmark around a 1,000-image footwear gallery: 60 queries in three difficulty
   groups, 200 human relevance judgements, two widened label variants and a 30-query frozen held-out
   set, all shipped in the repository.
2. A training-free adaptation method — query parsing plus zero-shot attribute re-ranking — with each
   design choice isolated by an ablation.
3. A decomposition of the +17.2% mAP@50 gain into candidate depth, attribute ordering and cosine
   tie-breaking, with two diagnostic arms that bound what the automatic labels measure.
4. A negative result for prompt ensembling, four diagnosed failures with visual evidence, a
   calibration of automatic labels against human judgement, and measured latency.

---

## 2. Related work

CLIP (Radford et al., 2021) learns a shared image-text space by contrastive pretraining on 400
million pairs and transfers through prompt text. We use the open-source reimplementation open_clip
(Ilharco et al., 2021) with the original OpenAI weights and keep the encoders frozen: a thousand
images give no domain pairs to fine-tune on, and a frozen encoder makes the adaptation layer's
contribution measurable.

Two-stage retrieval is standard — a cheap first pass proposes candidates and a costlier second pass
reorders them (Nogueira & Cho, 2019). Our second stage is a zero-shot attribute classifier, so it
needs no labelled pairs. Evaluation follows standard practice (Manning et al., 2008) with confidence
intervals from a paired bootstrap (Efron & Tibshirani, 1993). The gallery comes from the Fashion
Product Images dataset (Aggarwal, 2019), whose product-shot style keeps the visual domain narrow.

---

## 3. Data

### 3.1 Source and sampling

The publisher describes 44k product images upstream; `dataset/NOTICE` records that 9,222 of them
carry `masterCategory = Footwear`. We sample 500 men's and 500 women's items, rotating through
`baseColour` within each gender so that common colours contribute about fifty images each and rare
colours stay as distractors. The gallery covers eight article types: Casual Shoes (283), Heels (237),
Sports Shoes (162), Flip Flops (126), Flats (101), Sandals (60), Formal Shoes (21), Sports Sandals (10).

Source images are 1080×1440, resized with Lanczos to 480×640 at JPEG quality 90, 22.1 MiB in total.
That is more than twice CLIP's 224-pixel input, so the metallic highlights that separate gold, bronze
and brown survive. The gallery ships with the repository, so a reviewer encodes exactly the pixels
behind every number here.

### 3.2 Attribute schema

| Field | Example | Role |
| --- | --- | --- |
| `id` | 26658 | File name and ranking output |
| `path` | `data/gallery/26658.jpg` | Relative to the repository root |
| `articleType` | Casual Shoes | Type slot and type labels (8 values) |
| `color` | Black | Colour slot and colour labels (40 values) |
| `gender` | Men | Gender slot and labels (500 / 500) |
| `category` | Shoes | Sanity check only (Shoes 804 / Flip Flops 126 / Sandal 70) |
| `usage`, `season`, `year` | Casual, Summer, 2012 | Display only |
| `display_name` | "Puma Men Basics Black Casual Shoes" | Interface caption only |

`dataset/manifest.csv` records the 1,000 sampled ids and their attributes; `dataset/NOTICE`
records provenance, licence (MIT) and the resize rule. The same fields supply the attribute
vocabulary, the automatic relevance labels and the interface captions, a coincidence we
return to in §5.2 and §6.1.

### 3.3 Caching

Image features are cached in `data/image_features.pt` under a signature of image count, newest
modification time and total bytes, so replacing any image invalidates the cache and cached features
always match the gallery that produced them. A full re-encode takes 41 s on the 8 GB M1 laptop used
for every number here, which `main.py probe` measures once per run.

### 3.4 Gallery size

Each query has between 10 and 283 relevant images (mean 54), enough for a stable P@10 and a
meaningful mAP@50, and the size keeps the study runnable on a laptop. All claims are scoped to this
gallery.

---

## 4. Method

### 4.1 Baseline: frozen CLIP cosine

Images are encoded once with ViT-B/32 (224-pixel centre crop), L2-normalised to 512 dimensions and
cached. A query is encoded with the same text encoder, as written, and scored by the dot product,
which is cosine similarity for unit vectors. The baseline returns the top 50; §6.8 quantifies what
happens when templates are added to the query text. The model is built by open_clip from its default
`ViT-B-32` config with the `openai` weights, in which the text tower uses ordinary GELU, while
OpenAI trained that checkpoint with QuickGELU; the library prints a mismatch warning on every load,
and §7.3 measures how far the deviation reaches.

### 4.2 Query understanding

The parser fills three slots from vocabulary derived from the metadata: 36 colour words covering all
40 colour values, 30 type phrases for the eight article types and their synonyms, and 9 gender words.
Each alias maps to a list of catalogue values.

| Query | Colour (weight) | Type (weight) | Gender |
| --- | --- | --- | --- |
| `grey casual shoes` | Grey 1.0, Charcoal 0.5, Silver 0.5 | Casual Shoes 1.0 | — |
| `beige heels` | Beige 1.0, Mushroom Brown 0.5, Off White 0.5, Cream 0.5, Khaki 0.5, Tan 0.5 | Heels 1.0 | — |
| `sneakers` | — | Sports Shoes 1.0, Casual Shoes 1.0 | — |
| `sports sandals for women` | — | Sports Sandals 1.0 | Women |

Three rules turn a parse into a score. Matches are sorted by length and an alias contained in an
accepted longer phrase is dropped, so "sports sandals" parses as Sports Sandals and no longer fires
Sports Shoes and Sandals as well. Within a colour alias the first value is the word's own catalogue
value at weight 1.0 and the rest are the family it is confused with at 0.5: a catalogue gives one
colour per image while CLIP places silver, grey and charcoal close together, so full weight pulls
wrong-coloured images up and zero weight loses images a person would accept for "grey". Type synonyms
keep weight 1.0, since "sneakers" names two catalogue types; gender stays unconstrained when the query
names neither or both.

The parser never rewrites the query text: the encoder still sees the shopper's words and the parse
only supplies weights to the second stage.

### 4.3 Zero-shot attribute tables

For each slot we encode one prompt per catalogue value — "a photo of a {colour} shoe", "a photo of
{type}", "a photo of a shoe for {gender}" — and apply a softmax over that slot's values with CLIP's
logit scale of 100. The result is a distribution per image and slot (40 colours, 8 types, 2 genders),
built once per process from 50 text encodes. These tables are weak classifiers: their argmax agrees
with the catalogue field for 36.9% of images on colour, 39.5% on type and 85.8% on gender
(`results/attribute_probe.json`). The fusion never asks for an argmax; it reads the probability of
the values a query named, which acts as a soft vote over the candidate list (§7.1).

### 4.4 Fusion scoring

For a query parsed into slots *a* with values *v* and weights *w*:

```
score(q, x) = cos(q, x) + β · Π_a ( Σ_{v∈a} w_v · P_a(v | x) )
```

The product across slots gives AND semantics: a query that names a colour and a type wants both, and
a low probability in either slot suppresses the bonus. The sum inside a slot lets a family value
contribute in proportion to its weight.

The two terms have to be on comparable scales for β to mean anything, so `main.py probe` records both
inside the shipped top-100 pool, per query group. Cosine spans 0.0249 to 0.0445 across a pool
depending on the group, while the agreement term peaks at 0.814 on type queries, 0.639 on colour
queries and 0.315 when a query names both slots, the product across slots being what pulls the
two-slot peak down; over the 60 queries that peak runs from 0.059 to 0.978. At β = 0.1 the median
peak bonus is therefore 0.0315 on a two-slot query and 0.0814 on a type-only one, the same size as the
whole cosine range of the pool it re-sorts, so a candidate outside the cosine top-50 can be pulled into
the head — which is what the pool-depth rows of §6.4 measure. The candidate pool is separate from the
metric truncation: under pure cosine, retrieving 100 and cutting to 50 gives the same list as
retrieving 50, so re-ranking changes only the order.

### 4.5 Algorithm and cost

Offline, encode the 1,000 gallery images, normalise, cache, and build the three attribute tables once.
Per query, encode the text once, take one matrix-vector product over the cached embeddings, keep the
top 100, weight and multiply the named values' probabilities across slots, add β times the result to
the cosine, and return 50. The coarse pass is linear in gallery size, the re-rank pass touches 100
candidates and at most three slots, and image features occupy about 2 MB.

`main.py probe` times all of it on an 8 GB Apple M1 laptop with the feature cache warm and the
60 queries run eight times over: 37.0 ms per query for the baseline and 37.9 ms for the re-ranked arm,
with the individual rounds of the two arms spread over 24.9–46.1 ms and 29.1–44.5 ms. The two
distributions sit on top of each other, so re-ranking adds no measurable latency at query time. What
it costs is offline: 1.8 s to build the three tables and 41.0 s to encode the gallery when the cache
is empty, and a cold start that answers the first query 31.2 s after the process begins
(`results/attribute_probe.json`). All four numbers are from one machine, and the round-to-round spread
above is the load of an ordinary laptop, not a property of the method.

### 4.6 Interfaces

`main.py search` prints a ranking with cosine and attribute scores for inspecting individual cases.
`main.py web` opens a Gradio page with a query box, a Top-K slider and a method switch, showing the
parsed attributes above the image grid and both scores under each image. Both call the same functions
as the evaluation, so the demonstration and the reported numbers share one code path.

---

## 5. Evaluation design

### 5.1 Queries and relevance

| Set | Items | Labels | Role |
| --- | --- | --- | --- |
| `queries.json` | 60: colour 20, type 18, combo 22 | Exact metadata match on the named attributes | Main comparison |
| `judgements.json` | 20 queries × 10 images = 200 cells | Human 0/1 on the baseline top-10 | Calibrate the auto labels |
| `queries_family.json` | Same 60 queries | Colour families widened (e.g. Brown←Tan/Beige) | Sensitivity to the colour definition |
| `queries_sandal.json` | Same 60 queries | Sandals and Sports Sandals merged | Sensitivity to taxonomy |
| `queries_heldout.json` | 30: 20 paraphrases, 10 typos | Exact metadata match, frozen | Generalisation check |

The 60 main queries separate three difficulties. The twenty colour queries use the twenty most
frequent colours, one each; the eighteen type queries cover every article type and add ten
type-and-gender queries with at least twenty relevant images; the twenty-two combo queries pair a
colour with a type, require at least twelve relevant images, and exercise the AND semantics the
product implements. Relevant sets total 3,220 images and range from 10 to 283. Because R@10 divides
by the relevant-set size its ceiling here is 0.368, so we report it as a diagnostic and use mAP@50 as
the primary metric. The human subset draws six colour, six type and eight combo queries, a size one
person can judge in a sitting. The held-out set was frozen once the method was fixed: twenty
paraphrases ("kicks", "trainers", "ballerinas", "slippers", "oxfords") test vocabulary generalisation
and ten misspellings ("blak", "silvr", "charcole") test parser robustness.

### 5.2 Relevance definitions

**Auto.** An image is relevant when its metadata matches every attribute the query names: `color`,
`articleType`, and `gender` when the query names one. It covers all 60 queries and is fully
reproducible.

**Strict.** A person judged the baseline top-10 for 20 queries by eye under a written rule: dominant
visible colour, same-hue neighbours counted together, type by visible design, gender rejected only
when clearly opposite. Re-checking the twelve least certain cells changed one label and filled six
blanks.

**Family and Sandal.** The same queries with the colour definition widened to families, or the two
sandal types merged; these change what counts as correct.

**Held-out.** Exact metadata labels, frozen and never used for tuning.

The scoring functions never read these labels; they compare vectors and consult the metadata
vocabulary. The auto labels are still not independent of the method's inputs: the alias tables and the
attribute tables come from `articleType`, `color` and `gender`, the same three fields the auto labels
use, so an auto score measures agreement with the catalogue rather than with a shopper, and it can be
saturated by a lookup on those fields, which the `metadata` arm in §6.1 quantifies. The human
judgements, the family variant and the sandal variant are the definitions that do not share the
parser's source, which is why §6.5 carries as much weight as §6.1.

### 5.3 Metrics

mAP@50 averages, over queries, the average precision of the top-50 list against the relevant set; it
reflects the whole ranking and is primary. P@10 is the fraction of the top ten that is relevant and
reflects what a user sees first. R@10 is the fraction of all relevant images in the top ten; it
depends on relevant-set size and is reported for completeness.

### 5.4 Statistics

Queries share one gallery and are correlated, so we compare arms with a paired bootstrap over
queries: 10,000 resamples at a fixed seed, drawing each query's two scores together, reporting the
2.5th and 97.5th percentiles of the mean difference, and counting how many queries improve, worsen or
stay unchanged. An interval that excludes zero is read as significant. Every interval quoted in §6 is
archived as JSON by `main.py compare --out` under `results/comparisons/`.

### 5.5 Controls

All comparisons use one gallery, one set of image features, one query file and one truncation, and
vary one factor at a time: the text side (prompt ensemble), the parser (four configurations), the
candidate pool (50, 100, 200), the fusion weight (0.1 and 0.2–1.5), the fusion form (with and
without cosine), the relevance definition (auto, family, sandal, strict) and, in §7.3, the encoder
activation. The held-out set is evaluated once the method is fixed, and every result file records the
depth, β and parser constants that produced it.

---

## 6. Results

### 6.1 Five arms and the quantity each one isolates

| Arm | mAP@50 | P@10 | R@10 | What it isolates |
| --- | --- | --- | --- | --- |
| `baseline` — frozen CLIP cosine | 0.3481 | 0.5817 | 0.1896 | Text-image similarity alone |
| `prompt` — six-template ensemble | 0.3302 | 0.5633 | 0.1855 | Editing the query text (−5.1%) |
| `attr_only` — agreement ordering, cosine tie-break | 0.4086 | 0.6133 | 0.1949 | The attribute term by itself |
| `rerank` — cosine + 0.1·agreement over top-100 | **0.4078** | **0.6283** | **0.2024** | The shipped fusion |
| `metadata` — lookup on ground-truth attributes | 1.0000 | 1.0000 | 0.3681 | Oracle; reads the labels |

Re-ranking improves mAP@50 over the baseline by 0.0597 (95% CI +0.0413 to +0.0816; 49 queries better,
11 worse) and P@10 by 0.0467 (CI +0.0150 to +0.0800; 28 better, 15 worse, 17 tied). Both intervals
exclude zero, and the comparison files behind them are archived (`results/comparisons/auto.json`).

The two added arms change how we read that gain. `attr_only` orders the same 100 candidates by
agreement alone and reaches 0.4086 mAP@50, level with the shipped fusion, while its P@10 of 0.6133
sits below 0.6283: under automatic labels essentially all of the mAP gain comes from the attribute
ordering, and cosine contributes at the head. The intervals agree with that split — dropping the
cosine term leaves a P@10 gain of 0.0317 whose CI covers zero (−0.0083 to +0.0717,
`results/comparisons/attr_only.json`), whereas keeping it gives the significant +0.0467 above. The
`metadata` arm scores 1.000 by construction, because matching the query's ground-truth attributes
against the catalogue fields is exactly the auto relevance rule; it is an upper bound on the oracle
rather than a system, and it says that auto mAP measures how closely a method reproduces the
catalogue. Together they leave an honest reading: the fusion reaches 0.408 where a catalogue lookup
reaches 1.000, and §6.5 and §6.6 are where the ordering has to prove it is more than agreement with
the catalogue.

### 6.2 Group analysis

| Group (queries) | Baseline mAP / P@10 | `attr_only` mAP / P@10 | Re-ranking mAP / P@10 | Δ mAP / Δ P@10 |
| --- | --- | --- | --- | --- |
| Colour (20) | 0.256 / 0.580 | 0.329 / 0.650 | 0.311 / 0.655 | +0.0546 / +0.0750 |
| Type (18) | 0.443 / 0.678 | 0.534 / 0.733 | 0.538 / 0.739 | +0.0949 / +0.0611 |
| Combo (22) | 0.354 / 0.505 | 0.379 / 0.482 | 0.390 / 0.514 | +0.0356 / +0.0091 |

Type queries gain most in mAP and colour queries most at the head. Combo queries improve deeper in the
list: the product over two slots penalises a candidate that is uncertain in either, so the head moves
by 0.0091 and the combo column carries the smallest gain of the three groups. `attr_only` follows the
same shape except on colour, where keeping cosine in the score costs 0.018 mAP against the
attribute-only ordering and returns 0.005 P@10 for it.

### 6.3 Parser ablation

| Parser configuration | mAP@50 | P@10 | Grey AP | Beige AP | Olive AP |
| --- | --- | --- | --- | --- | --- |
| No masking, all alias values at 1.0 | 0.3946 | 0.5933 | 0.147 | 0.131 | 0.316 |
| + longest-phrase masking | 0.3944 | 0.5950 | 0.147 | 0.131 | 0.316 |
| + colour family at 0.5 (shipped) | **0.4078** | **0.6283** | 0.196 | 0.182 | 0.299 |
| + colour family at 0.25 | 0.4131 | 0.6300 | 0.218 | 0.207 | 0.294 |

Against the unweighted parser the shipped configuration gains 0.0132 mAP (CI +0.0049 to +0.0224; 31
better, 10 worse, 19 tied) and 0.0350 P@10 (CI +0.0117 to +0.0617;
`results/comparisons/parser.json`). Masking alone is neutral in
aggregate (−0.0002); its value is internal consistency, since "sports sandals" stops firing two extra
type values and that lets a family weight act on the colour slot without noise from the type slot.
The 0.25 alternative gains another 0.0053 mAP while pushing olive and orange below their 0.5 values,
so we keep 0.5.

### 6.4 Candidate depth and fusion weight

From `results/ablations/fusion_ablation.json`: one process, one gallery, the shipped parser, one knob
moved per row. The shipped point is pool 100, β 0.1.

| Pool | β | mAP@50 | P@10 | Pool recall | Grey AP | Beige AP |
| --- | --- | --- | --- | --- | --- | --- |
| 50 | 0.1 | 0.3747 | 0.6250 | 0.477 | 0.182 | 0.183 |
| **100** | **0.1** | **0.4078** | **0.6283** | **0.655** | 0.196 | 0.182 |
| 200 | 0.1 | 0.4113 | 0.6283 | 0.818 | 0.214 | 0.182 |
| 100 | 0.2 | 0.4134 | 0.6350 | 0.655 | 0.188 | 0.176 |
| 100 | 0.4 | 0.4142 | 0.6333 | 0.655 | 0.182 | 0.172 |
| 100 | 0.8 | 0.4136 | 0.6283 | 0.655 | 0.188 | 0.166 |
| 100 | 1.5 | 0.4132 | 0.6250 | 0.655 | 0.184 | 0.165 |

The 0.0597 total splits into +0.0266 from re-ranking the same top-50 pool (0.3481 → 0.3747) and
+0.0331 from deepening the pool to 100 (0.3747 → 0.4078). P@10 barely moves across depths, 0.625 to
0.628, which places the head gain in the re-ordering; a pool of 200 lifts the recall ceiling to 0.818
and adds 0.0035 mAP, inside the noise, at twice the scoring cost. Stronger fusion is the same trade in
reverse: β from 0.2 to 1.5 buys at most 0.0064 mAP while grey loses up to 0.014 AP, beige up to 0.017
and `combo_beige_heels` falls from 0.106 to 0.096 at β = 1.5.

### 6.5 Human calibration and label sensitivity

| Relevance definition | Baseline mAP | Re-ranking mAP | Δ mAP (95% CI) |
| --- | --- | --- | --- |
| Auto, 60 queries | 0.3481 | 0.4078 | +0.0597 (+0.0413, +0.0816) |
| Colour family labels | 0.3997 | 0.4754 | +0.0757 (+0.0565, +0.0980) |
| Sandal-merged labels | 0.3498 | 0.4138 | +0.0640 (+0.0442, +0.0867) |

On the 20 human-judged queries the baseline's P@10 is 0.925 (185 of 200 cells) against 0.625 when the
same twenty lists are scored with automatic labels, which is `results/baseline_auto.json` restricted to
the judged qids. The three deltas in the table are archived with their intervals in
`results/comparisons/auto.json`, `family.json` and `sandal.json`. The gap measures label tolerance:
people count tan as brown, treat silver and grey
as one family, and accept shoes whose gender is not visually obvious. By group, human P@10 is 0.967
for colour, 0.983 for type and 0.850 for combo. Under the family and sandal definitions the gain is
larger than under auto labels and still significant, so the result does not rest on exact colour names
or a fine sandal split.

The human cells also expose a hole in our protocol: they cover the baseline top-10 only, and
re-ranking brings new images into those twenty lists. Scoring the unjudged cells as zero gives the
re-ranked arm a strict P@10 of 0.675 against the baseline's 0.925 (`results/rerank_strict.json`, 54 of
200 cells unjudged), an artefact of missing labels — of the 146 cells a person had judged, the
re-ranked list has 135 relevant, a rate of 0.925, identical to the baseline's. The 54 open cells are
queued for judgement with `main.py label-sheet`; until they are filled we report strict P@10 for the
baseline alone.

### 6.6 Held-out generalisation

On the 30 held-out queries mAP rises from 0.2579 to 0.3167 (+0.0588, CI +0.0050 to +0.1184) and P@10
from 0.4433 to 0.4767 (+0.0333, CI −0.0267 to +0.0967; `results/comparisons/heldout.json`), so the mAP
interval excludes zero and the
P@10 interval does not. Split by kind, the twenty paraphrases gain 0.0980 mAP and 0.0600 P@10 while
the ten misspellings lose 0.0198 and 0.0200: in eight of them the parser misses at least one intended
attribute, usually the colour, and the query falls back towards plain cosine. A dictionary parser
cannot match a word it does not contain.

### 6.7 Failure analysis

| Case | Baseline AP / P@10 | Re-ranking AP / P@10 | Diagnosis |
| --- | --- | --- | --- |
| `sports sandals` | 0.003 / 0.0 | 0.034 / 0.1 | Taxonomy: ten Sports Sandals images against sixty Sandals, visually close and inconsistently labelled |
| `grey shoes` | 0.161 / 0.4 | 0.196 / 0.6 | Exact labels split silver and grey; under family labels 0.289 → 0.425 |
| `beige heels` | 0.119 / 0.3 | 0.106 / 0.4 | Tan and khaki family combined with heels; head improves, deep ranking stays confused |
| `bronze heels` | 0.235 / 0.4 | 0.238 / 0.4 | Metallic colour depends on highlights that text cannot separate |
| `casual shoes` (control) | 0.450 / 0.7 | 0.867 / 0.9 | Strong type signal; re-ranking recovers it |

`main.py figure` renders the five cases side by side, baseline above and re-ranked below, into
`evaluation/montages/failureCases.png`. Two failures sit on the label side, `sports sandals` and grey's
exact colour split; two sit on the visual side, beige heels and bronze. Reporting them keeps the
aggregate from hiding the queries where the method is not usable.

### 6.8 Negative results and rejected variants

**Prompt ensembling.** Averaging six templates loses 5.1% mAP, 0.3302 against 0.3481, and 0.018 P@10
(`results/prompt_auto.json`), and the loss falls mainly on type queries, −0.0451 mAP against −0.006 on
colour and combo: template words dilute the colour and type terms inside one pooled vector, where the
object class is itself the cue.

**Rejected fusion settings.** β at 0.2–1.5 (§6.4) and a colour family weight of 0.25 (§6.3) buy a few
thousandths of mAP and widen the visible colour regressions. Two further settings were tried in one
exploratory run each and have no archived sweep, so they are stated as observations rather than
numbers a clone can reproduce: splitting β per slot moved mAP by less than a thousandth, inside the
bootstrap noise, and a penalty against confusable colours lowered the bronze case. The shipped score
keeps one β and the positive attribute term.

**Deeper attribute keys.** Forty of the 1,000 catalogue titles mention a lace, closure, toe or
material word at all — 31 a material, 8 a toe, 1 a lace and none a closure (`dataset/manifest.csv`) —
and none of those words is reliably visible at 480×640, so grounding them would need detection or a
vision-language model.

**Fine-tuning and captioning.** The only text paired with an image is its marketing title and one base
colour, already the fields the auto labels use, so fine-tuning would train on the test signal and a
captioning branch adds a lossy intermediate text.

---

## 7. Discussion

### 7.1 Where the gain comes from

The arms in §6.1 support a narrower account than "the attribute classifier is accurate". Its top-1
agreement with the catalogue is 0.369 on colour and 0.395 on type, so as a classifier it is unreliable.
What the fusion needs is weaker: among 100 candidates that cosine already considered plausible, the
probability of the named values has to be higher for images a person would accept. Type queries gain
most in mAP because eight coarse categories support that comparison even when the argmax is often
wrong; gender, at 0.858 top-1, is the one slot that behaves like a classifier. Colour queries gain most
at the head because the parser was the limiting factor there (§6.3), and combo queries expose the cost
of AND semantics: when either slot is uncertain the product is small and the head barely moves.

The auto-label ceiling matters for how far this generalises. A lookup on the ground-truth attributes
scores 1.000, so on that definition our 0.408 measures catalogue agreement. The evidence that the
ordering is more than an artefact of it sits elsewhere: the gain grows under family labels (+0.0757)
and the merged sandal taxonomy (+0.0640), holds on paraphrases outside the development set (+0.0980),
and the 146 human-judged cells the re-ranked list reached are relevant at the same 0.925 rate as the
baseline's.

### 7.2 Limitations

**Gallery.** One thousand catalogue photographs on light backgrounds; user photographs, cluttered
scenes or a much larger catalogue may behave differently. One exploratory pass over the full
9,222-row Footwear pool, at a lower resolution than the shipped gallery, diluted the relative gain; it
has no archived sweep, so it stands as a scale caveat and not as a result. One frozen backbone at
480×640: a larger model or a different resolution may move the colour confusions that motivate the
method.

**Labels.** Automatic labels inherit catalogue naming and are stricter than human judgement, 0.625
against 0.925 P@10 on the same twenty queries. All arms share them, so comparisons are fair, but an
absolute auto score is not user-facing accuracy and §6.1 shows the definition is saturable. The 200
human cells cover the baseline top-10 only and 54 of them are still open (§6.5), so until they are
judged the human definition gives a lower bound for re-ranking.

**Queries.** Authored by the group in the phrasing of the catalogue; the held-out set limits
overfitting but does not remove authorship bias.

**Configuration and cost.** Latency was measured on one 8 GB laptop over eight passes of the query set
(§4.5), with no multi-threaded or larger-gallery numbers, and the encoder runs open_clip's default text
activation rather than QuickGELU (§7.3).

### 7.3 Encoder configuration

open_clip builds `ViT-B-32` from its default model config, where the text tower uses ordinary GELU,
while the `openai` weights were trained with QuickGELU, so the library warns on every load.
`python main.py encoder-check` re-encodes the gallery under both activations and re-runs the four
arms on the 60 main queries (`results/encoderCheck.json`). QuickGELU lifts the baseline from 0.3481 to
0.3676 mAP@50, so the shipped configuration sits 0.0195 below the activation the weights were trained
with. The comparison between arms, which is what the report claims, survives the change: re-ranking
gains +0.0597 mAP under GELU and +0.0580 under QuickGELU, 0.047 P@10 against 0.043, and prompt
ensembling stays a loss at −5.1% against −4.7%. One reading does tilt: the attribute-only ordering
reaches 0.4086 against the fusion's 0.4078 here, but 0.4136 against 0.4256 under QuickGELU, so the
share of the mAP gain carried by cosine depends on the configuration. We keep the shipped one because
the 200 human judgements were collected on its baseline top-10; under QuickGELU, 44 of those cells and
66 of the re-ranked ones would name images nobody has looked at.

---

## 8. Conclusion

A frozen CLIP model ranks footwear images tolerably, but its text space confuses colours a shopper
separates. A parser that knows the catalogue's colour families, combined with zero-shot attribute
tables and a small fusion weight, raises mAP@50 by 17.2% and P@10 by 0.047 on a 1,000-image gallery,
with both intervals excluding zero, and the gain survives a wider colour definition, a merged sandal
taxonomy and held-out paraphrases. The method needs no training, no domain image-text pairs and no
extra query-time budget, and its failures are diagnosable: sandal taxonomy, metallic colours,
multi-attribute heads and misspelled queries remain hard. Prompt ensembling, the usual first attempt
at adaptation, is a net loss here, and the diagnostic arms read the automatic metric plainly — a label
lookup closes it, the attribute term carries most of the mAP gain and cosine works at the head. Next
come the 54 open human cells, a larger and messier gallery, and a parser that is not a dictionary.

---

## 9. Marking dimensions (internal)

| Dimension | What we did | Evidence |
| --- | --- | --- |
| Appropriateness (2.5%) | Chose a domain with visually meaningful attributes; named two user groups; defined relevance four ways and stated where each one binds | §1, §3, §5.1–5.2 |
| Soundness (2.5%) | Frozen encoder; paired bootstrap with archived intervals; one-factor ablations; two diagnostic arms that bound the oracle; held-out set; latency measured; encoder activation measured; shipped gallery and scripts | §4.5, §5.4–5.5, §6.1, §7.3, `results/` |
| Excitement (2.5%) | Negative result for prompt ensembling; the oracle and attribute-only arms reported against our own method; four diagnosed failures plus a control; live baseline/re-rank switch | §6.1, §6.7–6.8, §10 |
| Presentation (2.5%) | 7-minute structure with one query shown under both methods; five tables; one failure figure | §10 |
| Writing (2.5%) | 8-page plan, APA references, explicit limitations, AI disclosure, contributions | §11 |
| Peer Review (2.5%) | Review another group after all demonstrations on the course form; goes inside the report after the 8 pages | §11.6 |

---

## 10. Presentation plan (internal)

### 10.1 Constraints

Seven minutes, no Q&A, shared over Zoom, and setup counts towards the time. The server is
started before the session and the browser is already on the page, so the demo begins with a
query. The cold start is about 32 s (§4.5); do not start it live.

### 10.2 Timing

| Time | Slide | Content |
| --- | --- | --- |
| 0:00–0:35 | 1 | Task, users, why footwear |
| 0:35–1:20 | 2 | Baseline and one visible failure |
| 1:20–2:40 | 3 | Parser, attribute tables, fusion formula, weights |
| 2:40–4:10 | 4 | Live demo: one query, both methods |
| 4:10–5:30 | 5 | Main results, the five arms, confidence intervals |
| 5:30–6:25 | 6 | Ablations, negative results, failure figure |
| 6:25–7:00 | 7 | Limitations and conclusion |

### 10.3 Slide script

**Slide 1 — Problem and users.**
"A shopper can describe a shoe — grey casual shoes, silver sports shoes — but cannot name its
stock code. Catalogue search needs keywords; we search pixels. Our users are shoppers browsing
by appearance and catalogue teams checking a collection."

**Slide 2 — Baseline.**
"We start from a frozen CLIP model and cosine similarity. It works, and it confuses colours
that people separate: ask for grey and silver shoes come back, ask for brown and tan shoes are
missed. In CLIP's text space those colour words are neighbours."

**Slide 3 — Method.**
"We add two things without training. A parser turns the query into colour, type and gender,
with a literal weight of one and half weight for colour-family neighbours. CLIP text prompts
turn the same attributes into probability tables per image. We add the weighted product of
those probabilities to the cosine score — a product, because a query that names a colour and a
type wants both. Candidate pool 100, returned list 50."

**Slide 4 — Demo.**
"Same query, two methods. Baseline on the left, re-ranked on the right. [Type `silver sports
shoes`.] The baseline returns grey and white shoes high; the re-ranked list puts silver first.
The line under each image is cosine and attribute agreement, so you can see which term moved
it. [Switch to `sports sandals`.] This one is a failure we report: the catalogue's sandal
types are visually close and neither method fixes it."

**Slide 5 — Results.**
"Across 60 queries mAP@50 goes from 0.348 to 0.408, plus 17.2%, with a paired bootstrap
interval that excludes zero; P@10 goes from 0.582 to 0.628. Two extra arms keep us honest.
Ordering by the attribute term alone reaches the same mAP, so cosine is doing the head work.
And a lookup on the ground-truth labels scores 1.000, which tells you the automatic metric
measures catalogue agreement — that is why we also report human judgements and two widened
label sets, where the gain is bigger."

**Slide 6 — What did not work.**
"Prompt ensembling, the usual free trick, loses five percent: template words dilute the colour
and type terms. Raising the fusion weight buys total score with visible colour failures, so we
keep it small. Our failure figure shows four cases and one control: taxonomy, silver versus
grey, beige heels, metallic bronze."

**Slide 7 — Limits and close.**
"One thousand catalogue images, automatic labels stricter than people, and a human check of 200
cells with 54 of them still to judge on the re-ranked lists. Within that scope the method is
training-free, adds no query-time cost, and its failures are visible. The repository ships the
gallery, the labels and every script."

### 10.4 Recommended demonstration queries

Numbers are read from `results/baseline_auto.json` and `results/rerank_auto.json` (auto labels,
60 queries, β = 0.1, candidate depth 100), and the score components come from
`python main.py search --method rerank --top 10 --query "…"`, which prints cosine and agreement for
every item it returns.

| Query | Role in the talk | Baseline AP / P@10 | Re-rank AP / P@10 | What to point at |
| --- | --- | --- | --- | --- |
| `casual shoes` | Strongest single switch | 0.450 / 0.70 | **0.867 / 0.90** | Ranks 1–4 are all Casual Shoes with agreement 0.54–0.64; the baseline head mixes colours and a Sports shoe |
| `silver sports shoes` | Depth, not the head | 0.254 / 0.50 | 0.317 / 0.50 | Both lists open with a silver shoe and a white one at rank 3; the re-ranked head adds a grey shoe at rank 5 and 22 of its top 50 are relevant against the baseline's 14, so point at the list below row ten, not at P@10, which is unchanged |
| `beige casual shoes` | Visually clean backup | 0.394 / 0.60 | 0.386 / 0.50 | Top-5 all Beige/Khaki Casual; AP dips because tan and khaki count as non-relevant under metadata labels, so use it for the label caveat |
| `black casual shoes` | Honest limitation | 0.153 / 0.30 | 0.215 / 0.40 | Two black men's formal shoes survive at ranks 7 and 10: the Casual/Formal zero-shot gap is small, their agreement is 0.03–0.04 against 0.05–0.10 for the casual items, so β moves the score by 0.003–0.010 across a cosine range of 0.011 in the head |
| `sports sandals` | Diagnosed failure (slide 4) | 0.003 / 0.00 | 0.034 / 0.10 | Sandal subtypes are visually close and the metadata split is finer than the pixels |
| `bronze heels` | Backup for the metallic case | 0.235 / 0.40 | 0.238 / 0.40 | Bronze reads as brown and the attribute term is a near no-op, which matches `evaluation/montages/failureCases.png` |

Slide 4 uses `silver sports shoes` then `sports sandals`. `casual shoes` is the replacement if
the audience needs one obvious win, and `black casual shoes` is the query to volunteer when
someone asks what still fails.

### 10.5 Demo checklist

- Run `python main.py web` before the session and leave the tab open; with the cached features
  the page is up in a few seconds and no model download happens.
- Pre-type the §10.4 queries in the order you plan to use them; keep `bronze heels` as the
  spare.
- Check that the method switch re-runs the query, and show both lists for the same query.
- Record a 90-second backup video in case the network or projector fails.
- Share the application window in Zoom, not the desktop; font size at least 18 pt.

---

## 11. Report plan (internal)

### 11.1 Page budget, measured by laying the draft out

The submission range — the abstract through the end of §8, plus the AI-use statement and the
contribution table — lays out at 10.2 pages. The number comes from a render, not from arithmetic: the
markdown is converted with the brief's own settings (A4 portrait, 1-inch margins, 12 pt Times New
Roman, single line spacing, tables at full text width, Figure 2 printed 6.5 in wide, no space before
or after paragraphs) and printed to PDF by headless Chrome
(`Google Chrome --headless=new --no-pdf-header-footer --print-to-pdf=page.pdf page.html`), and the
text ends 103 words into page 11.
Word's default 8 pt paragraph spacing adds 0.4 pages; adding the Figure 1 method diagram adds 0.35,
which puts the two-figure version at 10.9. A laid-out page carries about 490 words of mixed prose and
table text, so the 600 words per page assumed by the first version of this section was too generous
and the excess is larger than it recorded.

Inside that range the draft holds 4,279 words of prose over 296 lines, 44 data rows under nine header
rows, 38 headings, one displayed formula and one figure. Four levers pay the excess back, cheapest
first:

- Paragraph spacing set to 0 pt instead of Word's default 8 pt: 0.4 pages, and the brief constrains
  the line spacing only.
- The nine tables merged into the four of §11.2: 0.1 pages, five header rows.
- The 38 headings reduced to 28 — §4 to four, §5 to three, §6 to four, §3 to three, §7 to two: 0.2
  pages, and one level of navigation.
- Prose, for the remainder.

Reaching 8 pages therefore costs 1,100 words of prose with Figure 2 alone and 1,270 if Figure 1 stays.
The allowance per section, ordered by how much each can lose without dropping a claim, with the words
it measures now: §4.4 80 of 227, §6.8 80 of 239, §4.5 80 of 217, §5.2 70 of 222, §6.5 70 of 224, §7.1
70 of 217, §6.1 60 of 218, §7.2 55 of 214, §4.2 55 of 191, §7.3 45 of 179, §8 40 of 160, §5.1 35 of
149, §1.1 35 of 144, §3.1 35 of 126, Abstract 35 of 162, §2 30 of 136, §4.3 30 of 120, §1.3 25 of 106,
§6.4 25 of 126, §4.1 20 of 104, §6.6 20 of 94, §5.5 15 of 88, §6.3 15 of 86, §6.2 10 of 78, §5.4 10 of
73, §4.6 10 of 64, §5.3 10 of 63, §3.3 10 of 59, §1.2 10 of 59, §6.7 10 of 55, §3.2 5 of 42, §3.4 5 of
37. The list sums to 1,105 words and leaves 3,174 of prose; no single allowance passes 37% of its
section, so every table row, every metric and every claim above survives the cut. Figure 1 is worth
its 0.35 page only if the method diagram earns more with the markers than the two paragraphs it
displaces; the slides can carry the same diagram as a standalone image.

### 11.2 Figures and tables

- Figure 1: method diagram (query → parser → three tables → fusion → ranking). Draw once for
  the report and the slides. Still to do.
- Figure 2: `evaluation/montages/failureCases.png`, five cases, baseline above and re-ranked
  below. Ships in the repository; regenerate with `python main.py figure`.
- Table 1: dataset and query sets (merge §3.2 and §5.1).
- Table 2: the five arms plus the group breakdown (merge §6.1 and §6.2).
- Table 3: parser ablation and fusion sweep (merge §6.3 and §6.4).
- Table 4: label sensitivity, held-out and failures (merge §6.5–6.7; keep failures as a short
  list if space is tight).

### 11.3 Formatting checklist

The rules below are the course deck's own words (slide 16, *Report Format*, and slide 17, *Peer
Review*); the page count in §11.1 is measured against them.

- 8 A4 portrait pages maximum for the body; 12 pt Times New Roman; 1-inch margins; single line
  spacing; one column; no cover page or table of contents.
- Paragraph spacing set to 0 pt before and after, which the brief leaves open and §11.1 prices at 0.4
  pages.
- APA 7th edition for references and in-text citations; references and the peer review sit outside the
  8 pages on unlimited pages.
- Every number in a table must match a file in `results/`, with the command that produced it
  named in the caption or the text. Observations with no archived sweep stay in prose (§6.8).
- Include the AI-use disclosure and the member-contribution table (templates below).
- Give the code location on page 1: `https://github.com/Jayden-T-Chen/Comp5523_Project`. The
  repository is public, so the link alone lets a marker clone the gallery, the labels and the
  evaluation suite.

### 11.4 AI-use disclosure (draft; confirm against actual use)

> AI-assisted tools were used to help optimise code, write a few convenience scripts outside
> the core pipeline, review code, and polish the final report text. The authors set the
> research questions, selected and curated the data, produced the human relevance labels,
> reviewed all generated code and text, and verified every reported number against the
> repository. The authors take responsibility for the content.

### 11.5 Member contributions

| Member | Contribution |
| --- | --- |
| [Name] |  |

Fill before submission. The brief asks for delineated roles, so name what each person owned:
gallery curation, query authoring, the 200 human judgements, parser and fusion code, evaluation
scripts, report sections, demo and recording.

### 11.6 Peer review

Peer review is assigned after all demonstrations and is submitted **inside this report**, after
the 8 pages, where it does not count towards the limit. Our group reviews another group's
presentation on the course form, focusing on whether the relevance definition is stated clearly
and whether the demonstration ran within the time limit. Placeholder for the review text:
[to be added after 18 November 2026].

### 11.7 Open items

| Item | Owner | Blocks |
| --- | --- | --- |
| Judge the 54 open strict cells (`python main.py label-sheet`, then `python main.py label-sheet --merge <export>`) | group | §6.5 re-run and the strict row of Table 4 |
| Re-run strict for baseline and re-rank after the merge, refresh §6.5 and §7.2 | — | Table 4 |
| Cut the 1,105 words listed in §11.1, merge the nine tables into four and the 38 headings into 28 | — | 8-page limit |
| Draw Figure 1 (method diagram) | — | Report layout |
| Export the report to an 8-page PDF | — | Submission |
| Slides and the 90-second backup recording | — | 18 November demo |
| Fill §11.5 contributions and confirm §11.4 against actual use | group | Submission |
| Peer review text after the demo session | group | Submission |

Decided on 5 November 2026: the encoder keeps open_clip's default text activation, with §7.3 and
`results/encoderCheck.json` recording the measured deviation, so the 200 existing human judgements
stay valid and no re-labelling follows from that choice.

---

## References

Aggarwal, P. (2019). *Fashion product images dataset* [Data set]. Kaggle.
https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset

Efron, B., & Tibshirani, R. J. (1993). *An introduction to the bootstrap*. Chapman & Hall.

Ilharco, G., Wortsman, M., Wightman, R., Gordon, C., Carlini, N., Taori, R., Dave, A.,
Shankar, V., Namkoong, H., Miller, J., Hajishirzi, H., Farhadi, A., & Schmidt, L. (2021).
*OpenCLIP* [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.5143773

Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to information retrieval*.
Cambridge University Press.

Nogueira, R., & Cho, K. (2019). *Passage re-ranking with BERT*. arXiv.
https://arxiv.org/abs/1901.04085

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell,
A., Mishkin, P., Clark, J., Krueger, G., & Sutskever, I. (2021). Learning transferable visual
models from natural language supervision. In *Proceedings of the 38th International Conference
on Machine Learning* (pp. 8748–8763). PMLR.

---

## Appendix A. Reproduction commands (not for submission)

```bash
# The five arms on the main 60 queries
python main.py eval --method baseline
python main.py eval --method prompt
python main.py eval --method attr_only
python main.py eval --method rerank
python main.py eval --method metadata          # oracle; reads the labels

# Human labels and the widened definitions
python main.py eval --method baseline --labels strict
python main.py eval --method rerank   --labels strict
python main.py label-sheet                     # judge the 54 open cells, then --merge the export
python main.py eval --method baseline --queries evaluation/sets/queries_family.json
python main.py eval --method rerank   --queries evaluation/sets/queries_family.json
python main.py eval --method baseline --queries evaluation/sets/queries_sandal.json
python main.py eval --method rerank   --queries evaluation/sets/queries_sandal.json

# Held-out set and the intervals quoted in section 6
python main.py eval --method baseline --queries evaluation/sets/queries_heldout.json
python main.py eval --method rerank   --queries evaluation/sets/queries_heldout.json
python main.py compare results/baseline_auto.json results/rerank_auto.json --out results/comparisons/auto.json
python main.py compare results/baseline_auto.json results/attr_only_auto.json --out results/comparisons/attr_only.json
python main.py compare results/rerank_legacy_auto.json results/rerank_auto.json --out results/comparisons/parser.json
python main.py compare results/baseline_auto_family.json results/rerank_auto_family.json --out results/comparisons/family.json
python main.py compare results/baseline_auto_sandal.json results/rerank_auto_sandal.json --out results/comparisons/sandal.json
python main.py compare results/baseline_auto_heldout.json results/rerank_auto_heldout.json --by kind --out results/comparisons/heldout.json
python main.py eval --method rerank --parser legacy   # rerank_legacy_auto.json, §6.3

# Ablations, probes, figure, tests
python main.py ablate-parser      # results/ablations/parser_ablation.json
python main.py ablate-fusion      # results/ablations/fusion_ablation.json
python main.py probe              # results/attribute_probe.json
python main.py encoder-check      # results/encoderCheck.json, §7.3: two gallery encodes
python main.py figure             # evaluation/montages/failureCases.png
python main.py test
```
