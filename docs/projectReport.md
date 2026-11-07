# Domain-Specific Text-to-Image Retrieval for Footwear

**COMP5523 Computer Vision — Group Project, 2026/27 Semester 1**
Demonstration: 18 November 2026, TU101 · Report deadline: 25 November 2026

> Working document. Sections 1–8, the AI-use statement and the contribution table are the submitted
> text; the references join them under the rule §11.1 measures. Sections 0 and 9–11 are internal
> working material and stay out of the submitted PDF.
> Every number below is archived in `results/` and can be regenerated with the commands in
> Appendix A.

---

## 0. Alignment with the brief (internal)

| Requirement in the brief | Where it is addressed | What we point to |
| --- | --- | --- |
| Build a text-to-image retrieval system for a domain of your choice | §1, §3–§4 | 1,000-image footwear gallery; one text box, one ranked list |
| Input: a natural-language query; output: a ranked list of relevant images | §4.1, §4.4 | CLI and Gradio; images with cosine and attribute scores |
| Define relevance and explain how the system measures it | §5.1–§5.2 | Metadata labels, 254 human judgements, three metrics, and the limit of each (§6.1, §6.3) |
| Explain who will use it and why | §1.1 | Shoppers who know appearance but not SKUs; catalogue operators |
| Cross-modal representation | §4.1 | Frozen CLIP ViT-B/32, 512-d, L2-normalised |
| Image descriptions | §2, §4.1 | Direct image encoding; captioning adds a lossy layer |
| Similarity search and indexing | §4.4 | Cosine over 1,000 cached embeddings |
| Domain adaptation | §4.2–§4.3 | Query parsing and zero-shot attribute re-ranking |
| Re-ranking | §4.3 | `cos + β·Π(weighted P)` over the top-100 |
| Query understanding | §4.2 | Three slots, alias tables, longest-phrase masking, family weights |
| User interface | §4.4 | Gradio page with a baseline / re-rank switch |
| Custom test set, human annotation, metrics, protocol, failure analysis | §5–§6 | 60 queries, 200 labelled cells (54 still open), mAP@50 / P@10 / R@10, paired bootstrap, five cases |
| Efficiency evaluation (optional) | §4.4, §7.2 | Measured: 37 ms per query for both arms, tables 1.8 s once, gallery 41 s once (`results/attribute_probe.json`) |
| 7-minute demonstration and 8-page report | §10–§11 | Timing plan, page budget, APA, disclosure, contributions |

The six marking dimensions (Appropriateness, Soundness, Excitement, Presentation, Writing,
Peer Review, 2.5% each) are mapped to evidence in §9.

---

## Abstract

In footwear text-to-image retrieval, a frozen CLIP ViT-B/32 encodes a 1,000-image catalogue and ranks
images by cosine similarity to a shopper's phrase. A parser reads the phrase into colour, type and
gender constraints from the catalogue vocabulary, CLIP prompts turn those attributes into per-image
probabilities, and a weighted product of them is added to the cosine score. Over 60 queries mAP@50
rises from 0.348 to 0.408 (+17.2%, paired bootstrap 95% CI +0.041 to +0.082) and P@10 from 0.582 to
0.628 (+0.047, CI +0.015 to +0.080). Two diagnostic arms bound the gain: the attribute term alone
reaches the same mAP, and a ground-truth label lookup reaches 1.000 by construction. A six-template
prompt ensemble loses 5.1%, 254 human judgements calibrate the automatic labels, latency is measured and
five failures are diagnosed.

---

## 1. Introduction

### 1.1 Task and users

Text-to-image retrieval returns images that match a free-form description. In a catalogue the task
is practical: a shopper can describe appearance — grey casual shoes, silver sports shoes — but
rarely knows a model name or stock code, and keyword search depends on naming that varies between
brands and leaves visual properties untagged.

Footwear suits a course-scale study: its attributes are visually meaningful and comparable — eight
article types, forty colour values, and neighbours such as grey, silver and charcoal that a person
separates easily and a text encoder does not. Shoppers search by appearance when they cannot name a
product and catalogue operators assemble or check a collection; both read the top of the list first, so
P@10 accompanies mAP@50. Given a query the system returns a ranked list in which every image carries its
cosine similarity and, under re-ranking, an attribute agreement score.

### 1.2 Contributions

1. A reproducible footwear benchmark: 1,000 images, 60 queries in three groups, 254 human
   judgements, two widened label variants and a frozen 30-query held-out set, all shipped.
2. A training-free adaptation — query parsing plus zero-shot attribute re-ranking — with every
   design choice isolated by an ablation.
3. A decomposition of the +17.2% gain into candidate depth, attribute ordering and cosine
   tie-breaking, with two arms that bound what the automatic labels measure.
4. A negative result for prompt ensembling, four diagnosed failures, label calibration and
   measured latency.

---

## 2. Related work

CLIP (Radford et al., 2021) learns a shared image-text space by contrastive pretraining on 400 million
pairs. We use the reimplementation open_clip (Ilharco et al., 2021) with the original weights, frozen: a
thousand images give no domain pairs to fine-tune on, and freezing keeps the adaptation layer's
contribution measurable.

Two-stage retrieval is standard — a cheap pass proposes candidates, a costlier one reorders them
(Nogueira & Cho, 2019) — and our second stage is a zero-shot attribute classifier that needs no labelled
pairs. Evaluation follows standard practice (Manning et al., 2008) with paired-bootstrap intervals
(Efron & Tibshirani, 1993). The gallery is the Fashion Product Images dataset (Aggarwal, 2019), whose
product-shot style keeps the visual domain narrow.

---

## 3. Data

### 3.1 Source and sampling

The publisher describes 44k product images upstream; `dataset/NOTICE` records that 9,222 carry
`masterCategory = Footwear`. We sample 500 men's and 500 women's items, rotating through `baseColour`
within each gender so common colours contribute about fifty images each and rare colours stay as
distractors. The gallery covers eight article types: Casual Shoes (283), Heels (237), Sports Shoes
(162), Flip Flops (126), Flats (101), Sandals (60), Formal Shoes (21), Sports Sandals (10).

Source images are 1080×1440, resized with Lanczos to 480×640 at JPEG quality 90, 22.1 MiB in total:
more than twice CLIP's 224-pixel input, so the metallic highlights that separate gold, bronze and
brown survive. Each query has 10 to 283 relevant images (mean 54), enough for a stable P@10 and a
meaningful mAP@50 at a size a laptop can hold.

### 3.2 Attribute schema and caching

`data/metadata.csv` carries one row per image: `id` (26658) and `path` (`data/gallery/26658.jpg`),
then `articleType`, `color` and `gender` — the three slots, with 8, 40 and 500/500 values — followed by
`category` as a sanity check (Shoes 804, Flip Flops 126, Sandal 70) and the display-only fields
`usage`, `season`, `year` and `display_name`. `dataset/NOTICE` records provenance, licence (MIT) and the
resize rule. The same fields supply the attribute vocabulary, the automatic labels and the interface
captions, a coincidence §5.1 and §6.1 return to.

Image features are cached in `data/image_features.pt` under a signature of image count, newest
modification time and total bytes, so replacing any image invalidates the cache and cached features
always match the gallery that produced them.

---

## 4. Method

### 4.1 Baseline: frozen CLIP cosine

Images are encoded once with ViT-B/32 (224-pixel centre crop), L2-normalised to 512 dimensions and
cached. A query is encoded with the same text encoder, as written, and scored by the dot product —
cosine similarity for unit vectors. The baseline returns the top 50. The model is open_clip's default
`ViT-B-32` config with the `openai` weights, whose text tower uses ordinary GELU where OpenAI trained
QuickGELU; the library warns on every load and §7.2 measures the deviation.

### 4.2 Query understanding

The parser fills three slots from vocabulary derived from the metadata — 36 colour words covering all
40 colour values, 30 type phrases for the eight article types and their synonyms, and 9 gender words,
each alias mapping to a list of catalogue values. The parse never rewrites the query text: the encoder
still sees the shopper's words.

| Query | Colour (weight) | Type (weight) | Gender |
| --- | --- | --- | --- |
| `grey casual shoes` | Grey 1.0, Charcoal 0.5, Silver 0.5 | Casual Shoes 1.0 | — |
| `beige heels` | Beige 1.0, Mushroom Brown 0.5, Off White 0.5, Cream 0.5, Khaki 0.5, Tan 0.5 | Heels 1.0 | — |
| `sports sandals for women` | — | Sports Sandals 1.0 | Women |

Three rules set the weights. Matches are sorted by length and an alias inside an accepted longer
phrase is dropped, so "sports sandals" parses as Sports Sandals alone. Within a colour alias the first
value is the word's own catalogue value at 1.0 and the rest are its confusable family at 0.5: the
catalogue gives one colour per image while CLIP places silver, grey and charcoal close together, so
full weight promotes wrong colours and no weight loses images a person accepts for "grey". Type
synonyms keep 1.0, since "sneakers" names two catalogue types; gender stays unconstrained when a query
names neither or both.

### 4.3 Zero-shot attribute tables and fusion scoring

For each slot we encode one prompt per catalogue value — "a photo of a {colour} shoe", "a photo of
{type}", "a photo of a shoe for {gender}" — and softmax over that slot's values at CLIP's logit scale
of 100, giving a distribution per image and slot (40 colours, 8 types, 2 genders) built once per
process from 50 text encodes. These are weak classifiers: argmax matches the catalogue for 36.9% of
images on colour, 39.5% on type and 85.8% on gender (`results/attribute_probe.json`). The fusion asks
no argmax; it reads the probability of the values a query named.

For a query parsed into slots *a* with values *v* and weights *w*:

```
score(q, x) = cos(q, x) + β · Π_a ( Σ_{v∈a} w_v · P_a(v | x) )
```

The product across slots gives AND semantics, a low probability in any slot suppressing the bonus; the
sum inside a slot lets a family value contribute in proportion to its weight.

For β to mean anything both terms must be comparable, so `main.py probe` records them inside the top-100
pool: cosine spans 0.0249 to 0.0445, while agreement peaks at 0.814 on type queries, 0.639 on colour and
0.315 on two-slot queries, where the product pulls it down, ranging from 0.059 to 0.978 across the 60
queries. At β = 0.1 the median peak bonus is 0.0315 on a two-slot query and 0.0814 on a type-only one —
the size of the whole cosine range it re-sorts, so a candidate outside the cosine top-50 can reach the
head (§6.2).

### 4.4 Algorithm, cost and interfaces

The coarse pass is linear in gallery size, the re-rank touches 100 candidates and at most three slots,
and the cached features occupy about 2 MB. `main.py probe` times the rest on an 8 GB Apple M1 laptop,
cache warm, over eight passes of the 60 queries: 37.0 ms per query baseline against 37.9 ms re-ranked,
with overlapping round ranges of 24.9–46.1 and 29.1–44.5 ms, so re-ranking adds no measurable
query-time latency. The cost is offline: 1.8 s to build the tables, 41.0 s to encode an empty cache, and
a cold start that answers the first query 31.2 s after the process begins
(`results/attribute_probe.json`).

`main.py search` prints a ranking with both scores and `main.py web` opens a Gradio page with a query
box, a Top-K slider and a switch between the arms; both call the functions the evaluation uses, so the
demonstration and the numbers share one code path.

---

## 5. Evaluation design

### 5.1 Query sets and relevance definitions

| Set | Items | Labels | Role |
| --- | --- | --- | --- |
| `queries.json` | 60: colour 20, type 18, combo 22 | Exact metadata match on named attributes | Main comparison |
| `judgements.json` | 20 queries × both heads = 254 cells | Human 0/1 on the baseline and re-ranked top-10 | Calibrates the auto labels |
| `queries_family.json` | Same 60 queries | Colour families widened (Brown←Tan/Beige) | Colour-definition sensitivity |
| `queries_sandal.json` | Same 60 queries | Sandals and Sports Sandals merged | Taxonomy sensitivity |
| `queries_heldout.json` | 30: 20 paraphrases, 10 typos | Exact metadata match, frozen | Generalisation check |

The sixty main queries form three groups. Twenty colour queries use the twenty most frequent colours,
one each; eighteen type queries cover every article type plus ten type-and-gender queries with at least
twenty relevant images; twenty-two combo queries pair a colour with a type and need at least twelve
relevant. Relevant sets total 3,220 images, and the human subset draws six colour, six type and eight
combo queries. Held-out queries, frozen once the method was fixed and never used for tuning, test
vocabulary with twenty paraphrases ("kicks", "trainers", "ballerinas") and robustness with ten
misspellings ("blak", "silvr", "charcole").

**Auto.** An image is relevant when its metadata matches every attribute the query names: `color`,
`articleType`, and `gender` when named. It covers all 60 queries and is fully reproducible.

**Strict.** A person judged the baseline top-10 of 20 queries by eye under a written rule: dominant
visible colour, same-hue neighbours counted together, type by visible design, gender rejected only
when clearly opposite. Re-checking the twelve least certain cells changed one label and filled six
blanks.

**Family and Sandal.** The same queries with colour widened to families, or the two sandal types
merged.

The scoring functions never read these labels, but the auto rule consults the same three fields as the
alias and attribute tables, so an auto score measures agreement with the catalogue and is saturable by a
lookup (§6.1). The human, family and sandal definitions do not share that source, which is why §6.3
carries as much weight as §6.1.

### 5.2 Metrics, statistics and controls

mAP@50 averages, over queries, the average precision of the top-50 list against the relevant set; it
reflects the whole ranking and is primary. P@10 is the fraction of the top ten that is relevant — what
a user sees first. R@10, the fraction of all relevant images in the top ten, is a diagnostic here: it
divides by the relevant-set size and ceilings at 0.368 on this gallery.

Queries share one gallery and are correlated, so arms are compared by a paired bootstrap over queries:
10,000 resamples at a fixed seed, drawing each query's two scores together and reporting the 2.5th and
97.5th percentiles of the mean difference, alongside the counts of queries that improve, worsen or tie.
An interval excluding zero is significant, and every interval in §6 is archived by
`main.py compare --out` under `results/comparisons/`.

Every comparison fixes one gallery, one set of image features, one query file and one truncation and
varies a single factor: the text side, the parser, the candidate pool, the fusion weight and form, the
relevance definition and, in §7.2, the encoder activation. Each result file records the depth, β and
parser constants that produced it.

---

## 6. Results

### 6.1 Five arms and the gain by query group

| Arm | mAP@50 | P@10 | R@10 | What it isolates |
| --- | --- | --- | --- | --- |
| `baseline` — frozen CLIP cosine | 0.3481 | 0.5817 | 0.1896 | Similarity alone |
| `prompt` — six-template ensemble | 0.3302 | 0.5633 | 0.1855 | Editing the query text (−5.1%) |
| `attr_only` — agreement ordering, cosine tie-break | 0.4086 | 0.6133 | 0.1949 | The attribute term alone |
| `rerank` — cosine + 0.1·agreement over top-100 | **0.4078** | **0.6283** | **0.2024** | The shipped fusion |
| `metadata` — lookup on ground-truth attributes | 1.0000 | 1.0000 | 0.3681 | Oracle; reads the labels |

Re-ranking improves mAP@50 by 0.0597 (95% CI +0.0413 to +0.0816; 49 queries better, 11 worse) and
P@10 by 0.0467 (CI +0.0150 to +0.0800; 28 better, 15 worse, 17 tied). Both intervals exclude zero
(`results/comparisons/auto.json`).

`attr_only`, which orders the same 100 candidates by agreement alone, reaches 0.4086 mAP@50, level with
the shipped fusion, while its P@10 of 0.6133 sits below 0.6283: under automatic labels essentially all of
the mAP gain comes from the attribute ordering and cosine contributes at the head. The intervals agree — dropping cosine
leaves a P@10 gain of 0.0317 whose CI covers zero (−0.0083 to +0.0717,
`results/comparisons/attr_only.json`), where keeping it gives the significant +0.0467. The `metadata`
arm scores 1.000 by construction: matching ground-truth attributes against the catalogue fields is the
auto relevance rule itself, so it bounds the oracle rather than a system, and §6.3 is where the ordering
must show it is more than catalogue agreement.

| Group (queries) | Baseline mAP / P@10 | `attr_only` mAP / P@10 | Re-ranking mAP / P@10 | Δ mAP / Δ P@10 |
| --- | --- | --- | --- | --- |
| Colour (20) | 0.256 / 0.580 | 0.329 / 0.650 | 0.311 / 0.655 | +0.0546 / +0.0750 |
| Type (18) | 0.443 / 0.678 | 0.534 / 0.733 | 0.538 / 0.739 | +0.0949 / +0.0611 |
| Combo (22) | 0.354 / 0.505 | 0.379 / 0.482 | 0.390 / 0.514 | +0.0356 / +0.0091 |

Type queries gain most in mAP, colour queries most at the head, and combo queries improve deeper in the
list: the product over two slots penalises a candidate uncertain in either, so combo's head moves by
0.0091. `attr_only` follows the same shape except on colour, where keeping cosine costs 0.018 mAP and
returns 0.005 P@10.

### 6.2 Parser and fusion ablations

| Parser configuration | mAP@50 | P@10 | Grey AP | Beige AP | Olive AP |
| --- | --- | --- | --- | --- | --- |
| All alias values at 1.0, no masking | 0.3946 | 0.5933 | 0.147 | 0.131 | 0.316 |
| + longest-phrase masking | 0.3944 | 0.5950 | 0.147 | 0.131 | 0.316 |
| + colour family at 0.5 (shipped) | **0.4078** | **0.6283** | 0.196 | 0.182 | 0.299 |
| + colour family at 0.25 | 0.4131 | 0.6300 | 0.218 | 0.207 | 0.294 |

Against the unweighted parser the shipped configuration gains 0.0132 mAP (CI +0.0049 to +0.0224; 31
better, 10 worse, 19 tied) and 0.0350 P@10 (CI +0.0117 to +0.0617; `results/comparisons/parser.json`).
Masking alone is neutral (−0.0002) and its value is consistency: "sports sandals" stops firing two extra
type values, leaving the family weight on the colour slot. The 0.25 weight gains 0.0053
mAP more but pushes olive and orange below their 0.5 values, so 0.5 is kept.

From `results/ablations/fusion_ablation.json`: one process, one gallery, shipped parser, one knob per
row.

| Pool | β | mAP@50 | P@10 | Pool recall | Grey AP | Beige AP |
| --- | --- | --- | --- | --- | --- | --- |
| 50 | 0.1 | 0.3747 | 0.6250 | 0.477 | 0.182 | 0.183 |
| **100** | **0.1** | **0.4078** | **0.6283** | **0.655** | 0.196 | 0.182 |
| 200 | 0.1 | 0.4113 | 0.6283 | 0.818 | 0.214 | 0.182 |
| 100 | 0.2 | 0.4134 | 0.6350 | 0.655 | 0.188 | 0.176 |
| 100 | 0.4 | 0.4142 | 0.6333 | 0.655 | 0.182 | 0.172 |
| 100 | 0.8 | 0.4136 | 0.6283 | 0.655 | 0.188 | 0.166 |
| 100 | 1.5 | 0.4132 | 0.6250 | 0.655 | 0.184 | 0.165 |

The 0.0597 total splits into +0.0266 from re-ranking the top-50 pool (0.3481 → 0.3747) and +0.0331 from
deepening it to 100 (0.3747 → 0.4078). P@10 moves from 0.625 to 0.628 across depths, placing the head
gain in the re-ordering; a pool of 200 lifts recall to 0.818 for 0.0035 mAP, inside the noise, at twice
the scoring cost. Stronger fusion trades in reverse: β from 0.2 to 1.5 buys at most 0.0064 mAP while
grey loses up to 0.014 AP, beige up to 0.017, and `combo_beige_heels` from 0.106 to 0.096.

### 6.3 Human calibration, label sensitivity and held-out queries

| Relevance definition | Baseline mAP | Re-ranking mAP | Δ mAP (95% CI) |
| --- | --- | --- | --- |
| Auto, 60 queries | 0.3481 | 0.4078 | +0.0597 (+0.0413, +0.0816) |
| Colour family labels | 0.3997 | 0.4754 | +0.0757 (+0.0565, +0.0980) |
| Sandal-merged labels | 0.3498 | 0.4138 | +0.0640 (+0.0442, +0.0867) |

On the 20 human-judged queries the baseline's P@10 is 0.925 (185 of 200 cells) against 0.625 under
automatic labels (`results/baseline_auto.json`, judged qids only): people count tan as brown, treat
silver and grey as one family, and accept shoes of unclear gender. Human P@10 by group is 0.967 colour,
0.983 type and 0.850 combo. Under family and sandal labels the gain is larger than under auto and still
significant, so the result does not rest on exact colour names or a fine sandal split.

The first human pass covered the baseline top-10 alone; `main.py label-sheet` queued the 54 images
re-ranking adds to those lists, they were judged on 7 November 2026, and 238 of the 254 labelled cells are
relevant. The re-ranked head has 188 of its 200 cells relevant, P@10 0.940 against 0.925, Δ +0.0150 with a
95% interval of −0.0100 to +0.0400 over 3 wins, 1 loss and 16 ties (`results/comparisons/strict.json`).
The head gain §6.1 measures under automatic labels falls to that margin under the human one, which already
accepts most of what cosine placed there, so the gain people judge sits below row ten.
`combo_khaki_casual_shoes` is the weakest list in both arms, at 0.500, and `combo_bronze_heels` moves from
0.500 to 0.700.

On the 30 held-out queries mAP rises from 0.2579 to 0.3167 (+0.0588, CI +0.0050 to +0.1184) and P@10
from 0.4433 to 0.4767 (+0.0333, CI −0.0267 to +0.0967; `results/comparisons/heldout.json`) — the mAP
interval excludes zero, the P@10 interval does not. By kind, twenty paraphrases gain 0.0980 mAP and
0.0600 P@10 while ten misspellings lose 0.0198 and 0.0200: in eight the parser misses an intended
attribute, usually the colour, and the query falls back towards cosine, since a dictionary parser cannot
match a word it does not contain.

### 6.4 Failures and rejected variants

| Case | Baseline AP / P@10 | Re-ranking AP / P@10 | Diagnosis |
| --- | --- | --- | --- |
| `sports sandals` | 0.003 / 0.0 | 0.034 / 0.1 | Taxonomy: 10 Sports Sandals against 60 close-looking, inconsistently named Sandals |
| `grey shoes` | 0.161 / 0.4 | 0.196 / 0.6 | Exact labels split silver and grey; family labels 0.289 → 0.425 |
| `beige heels` | 0.119 / 0.3 | 0.106 / 0.4 | Tan and khaki family: head up, deep ranking confused |
| `bronze heels` | 0.235 / 0.4 | 0.238 / 0.4 | Metallic colour rests on highlights text cannot separate |
| `casual shoes` (control) | 0.450 / 0.7 | 0.867 / 0.9 | Strong type signal, recovered by re-ranking |

`main.py figure` renders the five cases, baseline above re-ranking, into
`evaluation/montages/failureCases.png`. Two failures sit on the label side and two on the visual side;
reporting them keeps the aggregate from hiding the queries where the method is unusable.

**Prompt ensembling.** Averaging six templates loses 5.1% mAP, 0.3302 against 0.3481, and 0.018 P@10
(`results/prompt_auto.json`), mostly on type queries (−0.0451, against −0.006 on colour and combo):
template words dilute the colour and type terms inside one pooled vector.

**Rejected fusion settings.** β at 0.2–1.5 and a family weight of 0.25 (§6.2) buy a few thousandths of
mAP and widen the colour regressions. Two settings tried once, with no archived sweep, stand as
observations: splitting β per slot moved mAP by less than a thousandth, and a penalty against confusable
colours lowered the bronze case.

**Deeper attribute keys, fine-tuning and captioning.** Of the 1,000 titles, 48 mention a lace, closure,
toe or material word — 33 material, 8 toe, 1 lace, 7 closure (`results/attribute_probe.json`,
`title_terms`) — and none is reliably visible at 480×640, so grounding them would need detection. The only
text paired with an image is its marketing title and one base colour, already the fields the auto labels
use: fine-tuning would train on the test signal, and captioning adds a lossy intermediate.

---

## 7. Discussion

### 7.1 Where the gain comes from

§6.1 supports a narrower account than "the attribute classifier is accurate": at §4.3's top-1 rates it is
unreliable. What the fusion needs is weaker — among 100 candidates cosine already finds plausible, the
probability of the named values has to be higher for images a person would accept. Eight coarse type
categories support that comparison even when the argmax is often wrong, and gender, at 0.858, is the one
slot that behaves like a classifier. Combo queries show the cost of AND semantics: an uncertain slot
makes the product small and the head barely moves.

The auto-label ceiling sets how far this generalises: a ground-truth lookup scores 1.000, so on that
definition 0.408 measures catalogue agreement. The evidence that the ordering is more than an artefact
of it sits elsewhere — the gain grows under family labels (+0.0757) and the merged sandal taxonomy
(+0.0640), holds on held-out paraphrases (+0.0980), and survives the human definition, where both heads
score above 0.92 P@10 (§6.3).

### 7.2 Limitations and encoder configuration

**Gallery.** One thousand catalogue photographs on light backgrounds; user photographs, cluttered
scenes or a larger catalogue may behave differently. One exploratory pass over the full 9,222-row
Footwear pool, at lower resolution, diluted the relative gain; with no archived sweep it is a scale
caveat, not a result. One frozen backbone at 480×640: a larger model or a different resolution may
move the colour confusions the method is built around.

**Labels.** Automatic labels inherit catalogue naming and are stricter than human judgement (§6.3). All
arms share them, so comparisons are fair, but an absolute auto score is not user-facing accuracy and the
definition is saturable. The human definition rests on 20 queries and 254 cells, its head interval covers
zero, and it corroborates the deep ranking of §6.1.

**Queries.** Authored by the group in the catalogue's own phrasing; the held-out set limits
overfitting but not authorship bias.

**Configuration and cost.** Latency comes from one 8 GB laptop over eight passes (§4.4), with no
multi-threaded or larger-gallery numbers; the encoder runs open_clip's default text activation.

**Encoder configuration.** The activation difference §4.1 records is measurable: `main.py encoder-check`
re-encodes the gallery under both activations and re-runs the four arms on the 60 main queries
(`results/encoderCheck.json`). QuickGELU lifts the baseline from 0.3481 to 0.3676 mAP@50, 0.0195 above
the shipped configuration, and the comparisons between arms survive: re-ranking gains +0.0597 mAP under
GELU and +0.0580 under QuickGELU, 0.047 P@10 against 0.043, and prompt ensembling stays a loss at −5.1%
against −4.7%. One reading tilts — attribute-only reaches 0.4086 against the fusion's 0.4078 under GELU
but 0.4136 against 0.4256 under QuickGELU, so the share carried by cosine depends on the configuration.
The shipped one is kept because the 254 judgements were collected on its two heads; under QuickGELU 33 of
the baseline cells and 31 of the re-ranked ones would name images nobody has seen.

---

## 8. Conclusion

A frozen CLIP model ranks footwear images tolerably, but its text space confuses colours a shopper
separates. A parser that knows the catalogue's colour families, with zero-shot attribute tables and a
small fusion weight, raises mAP@50 by 17.2% and P@10 by 0.047 on a 1,000-image gallery, both intervals
excluding zero, and the gain survives a wider colour definition, a merged sandal taxonomy and held-out
paraphrases. It needs no training, no image-text pairs and no extra query-time budget; its failures are
diagnosable — sandal taxonomy, metallic colours, multi-attribute heads and misspellings. A label lookup
closes the automatic metric, the attribute term carries most of the mAP gain, cosine works at the head,
and prompt ensembling is a net loss. Next come a wider human query set, a larger and messier gallery,
and a parser that is not a dictionary.

---

## 9. Marking dimensions (internal)

| Dimension | What we did | Evidence |
| --- | --- | --- |
| Appropriateness (2.5%) | Chose a domain with visually meaningful attributes; named two user groups; defined relevance four ways and stated where each one binds | §1, §3, §5.1–5.2 |
| Soundness (2.5%) | Frozen encoder; paired bootstrap with archived intervals; one-factor ablations; two diagnostic arms that bound the oracle; held-out set; latency measured; encoder activation measured; shipped gallery and scripts | §4.4, §5.2, §6.1, §7.2, `results/` |
| Excitement (2.5%) | Negative result for prompt ensembling; the oracle and attribute-only arms reported against our own method; four diagnosed failures plus a control; live baseline/re-rank switch | §6.1, §6.4, §10 |
| Presentation (2.5%) | 7-minute structure with one query shown under both methods; five tables; one failure figure | §10 |
| Writing (2.5%) | 8-page plan, APA references, explicit limitations, AI disclosure, contributions | §11 |
| Peer Review (2.5%) | Review another group after all demonstrations on the course form; goes inside the report after the 8 pages | §11.6 |

---

## 10. Presentation plan (internal)

### 10.1 Constraints

Seven minutes, no Q&A, shared over Zoom, and setup counts towards the time. The server is
started before the session and the browser is already on the page, so the demo begins with a
query. The cold start is about 32 s (§4.4); do not start it live.

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
"One thousand catalogue images, automatic labels stricter than people, and a human check of 254 cells
covering the top-10 of both methods on 20 queries. Within that scope the method is
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
- Record a backup video of one query on both arms, 60–90 s long, in case the network or the projector
  fails. The brief sets no length, so keep it small enough to send as a file.
- Share the application window in Zoom, not the desktop; font size at least 18 pt.

---

## 11. Report plan (internal)

### 11.1 Page budget, measured by laying the draft out

The submission range — the abstract through the end of §8, plus the AI-use statement and the
contribution table — lays out at 8 pages. The number comes from a render, not from arithmetic: the
markdown is converted with the brief's settings (A4 portrait, 1-inch margins, 12 pt Times New Roman,
single line spacing, tables at full text width, Figure 2 printed 5.2 in wide inside §6.4, no space
before or after paragraphs) and printed to PDF by headless Chrome
(`Google Chrome --headless=new --no-pdf-header-footer --print-to-pdf=page.pdf page.html`), which counts
the pages and the words on each. The `---` rules between sections are markdown scaffolding and stay out
of the typeset version; the contribution table is measured with three member rows. The eighth page ends
0.54 in above the bottom margin.

That range now holds 3,244 words of prose — paragraph text between the abstract and §9 with the markdown
markup stripped and headings, table cells and the displayed formula left out — plus 25 headings, eight
tables and 35 data rows, one displayed formula and one figure. Pages carry 433 to 576 words depending on
where a table can break, so the 600 words per page assumed by the first version of this section was too
generous: it had measured 10.2 pages for 4,279 words of prose, 38 headings and nine tables. Four levers
closed the gap, cheapest first, and all four are now in the file:

- Paragraph spacing set to 0 pt instead of Word's default 8 pt, which the brief leaves open and which
  measures at exactly one page: the same document at 8 pt runs to 9. Whoever types the submission sets
  this before anything else.
- The §3.2 attribute table rewritten as prose, since a list of field names cost more than the
  sentences that replaced it, and §3.3's caching paragraph folded in beside it.
- 38 headings reduced to 25: §1 from three to two, §3 from four to two, §4 from six to four, §5 from
  five to two, §6 from eight to four and §7 from three to two, the removed sections folded into their
  neighbours with every cross-reference rewritten.
- Prose in the range cut from 4,279 words to 3,244, a paragraph at a time with every metric, table row
  and claim left in place and re-checked against `results/` after each pass. The completed human
  comparison then refilled §6.3 with 119 of them, and the 0.54 in of slack still holds.

Further table merging was tried and dropped: the eight tables share no columns, so joining them adds a
header row and a page break without saving a line. Figure 1 is the remaining claim on the page budget,
and it does not fit — a 150 pt diagram with caption takes the document to 9 pages. It goes on a slide
instead, where it can be drawn large.

The reference list sits outside the range measured above. Typeset with it, the same document runs to 9
pages, 135 words of them on the ninth when the references follow the two statements and 91 when they
precede them. An inch of this setting carries about 53 words, so the 0.54 in left on page eight absorbs
roughly 30 of them, and bringing the list inside the limit needs some 60 further words of prose removed.
Whether that cut follows depends on the brief's rule about whether the 8 pages include the references
(§11.7).

### 11.2 Figures and tables

- Figure 1: method diagram (query → parser → three tables → fusion → ranking). Slides only; the report
  has no room for it (§11.1). Still to draw.
- Figure 2: `evaluation/montages/failureCases.png`, five cases, baseline above and re-ranked below,
  printed 5.2 in wide inside §6.4. Ships in the repository; regenerate with `python main.py figure`.
- Table, §4.2: three parsed queries with their slot values and weights.
- Table, §5.1: the five query and judgement files, their sizes and what each is for.
- Tables, §6.1: the five arms, then the same comparison split by query group.
- Tables, §6.2: the four parser configurations, then the pool and β sweep.
- Table, §6.3: the gain under each relevance definition.
- Table, §6.4: the five failure cases with both arms' AP and P@10 and a one-line diagnosis.

### 11.3 Formatting checklist

The rules below are the course deck's own words (slide 16, *Report Format*, and slide 17, *Peer
Review*); the page count in §11.1 is measured against them.

- 8 A4 portrait pages maximum for the body; 12 pt Times New Roman; 1-inch margins; single line
  spacing; one column; no cover page or table of contents.
- Paragraph spacing set to 0 pt before and after, which the brief leaves open and §11.1 measures at one
  page of the eight.
- APA 7th edition for references and in-text citations; references and the peer review sit outside the
  8 pages on unlimited pages.
- Every number in a table must match a file in `results/`, with the command that produced it
  named in the caption or the text. Observations with no archived sweep stay in prose (§6.4).
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
| [Name] |  |
| [Name] |  |

Fill before submission with delineated roles: gallery curation and the sampling rule, query authoring
for the 60 main and 30 held-out sets, the 254 human judgements over both heads,
parser and fusion code, the evaluation and probe scripts, the report sections and the figures, the
slides and the demonstration, and the peer review.

### 11.6 Peer review

Peer review is assigned after all demonstrations and is submitted **inside this report**, after
the 8 pages, where it does not count towards the limit. Our group reviews another group's
presentation on the course form, focusing on whether the relevance definition is stated clearly
and whether the demonstration ran within the time limit. Placeholder for the review text:
[to be added after 18 November 2026].

### 11.7 Open items

| Item | Owner | Blocks |
| --- | --- | --- |
| Check the brief on whether the 8 pages include the reference list (§11.1: with it the document runs to 9) | — | PDF export |
| Draw Figure 1 (method diagram) for the slides | — | 18 November demo |
| Export the report to an 8-page PDF at 0 pt paragraph spacing | — | Submission |
| Slides and the backup recording | — | 18 November demo |
| Fill §11.5 contributions and confirm §11.4 against actual use | group | Submission |
| Peer review text after the demo session | group | Submission |

Completed on 7 November 2026: the 54 re-ranked head cells were judged through `label-sheet` and merged
into `evaluation/sets/judgements.json`, which now holds 254 cells over 20 queries; the strict arms were
re-run, `results/comparisons/strict.json` archived, `results/encoderCheck.json` recomputed against the
wider pool, and §6.3, §7.1, §7.2, §8 and the README updated to the complete comparison. The same day the
repository was swept for dead code: every module under `retrieval/` and `evaluation/tools/` is reachable
from one of the eleven `main.py` commands and every shipped data and results file is cited, which left the
duplicate `dataset/manifest.csv` — the same 1,000 rows as `data/metadata.csv` apart from a derivable path
column — for removal, with §3.2 re-cited to `data/metadata.csv` and the title counts of §6.4 moved into
the new `title_terms` block of `results/attribute_probe.json`.

Decided on 5 November 2026: the encoder keeps open_clip's default text activation, with §7.2 and
`results/encoderCheck.json` recording the measured deviation, so the 254 human judgements
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
python main.py compare results/baseline_strict.json results/rerank_strict.json --out results/comparisons/strict.json
python main.py label-sheet                     # the sheet for the human cells; --merge folds an export back in
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
python main.py eval --method rerank --parser legacy   # rerank_legacy_auto.json, §6.2

# Ablations, probes, figure, tests
python main.py ablate-parser      # results/ablations/parser_ablation.json
python main.py ablate-fusion      # results/ablations/fusion_ablation.json
python main.py probe              # results/attribute_probe.json
python main.py encoder-check      # results/encoderCheck.json, §7.2: two gallery encodes
python main.py figure             # evaluation/montages/failureCases.png
python main.py test
```
