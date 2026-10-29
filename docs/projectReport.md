# Domain-Specific Text-to-Image Retrieval for Footwear

**COMP5523 Computer Vision — Group Project, 2026/27 Semester 1**
Demonstration: 18 November 2026, TU101 · Report deadline: 25 November 2026

> Working document. It holds the full prose for the 8-page report and the 7-minute
> presentation, aligned with the two course documents (*Group Project Specification*
> and *Project Introduction*, slides 3–17). Figures are produced by this repository;
> every number can be regenerated from `results/*.json`. Sections 0 and 9–11 are
> internal working material and do not belong in the submitted PDF.

---

## 0. Alignment with the brief (internal)

| Requirement in the brief | Where it is addressed | What we point to |
| --- | --- | --- |
| Build a text-to-image retrieval system for a domain of your choice | §1, §3–§4 | 1,000-image footwear gallery; one text box, one ranked list |
| Input: a natural-language query; output: a ranked list of relevant images | §4.1, §4.6 | CLI and Gradio; images with cosine and attribute scores |
| Define relevance and explain how the system measures it | §5.1–§5.3 | Metadata oracle, 200 human judgements, three metrics |
| Explain who will use it and why | §1.1 | Shoppers who know appearance but not SKUs; catalogue operators |
| Cross-modal representation | §4.1 | Frozen CLIP ViT-B/32, 512-d, L2-normalised |
| Image descriptions | §2, §4.1 | Direct image encoding; captioning adds a lossy layer |
| Similarity search and indexing | §4.5 | Cosine over 1,000 cached embeddings |
| Domain adaptation | §4.2–§4.4 | Query parsing and zero-shot attribute re-ranking |
| Re-ranking | §4.4 | `cos + β·Π(weighted P)` over the top-100 |
| Query understanding | §4.2 | Three slots, alias tables, longest-phrase masking, family weights |
| User interface | §4.6 | Gradio page with a baseline / rerank switch |
| Custom test set, human annotation, metrics, protocol, failure analysis | §5–§6 | 60 queries, 200 labelled cells, mAP@50 / P@10 / R@10, paired bootstrap, five cases |
| Efficiency evaluation (optional) | §7.2 | Not benchmarked; stated as a limitation |
| 7-minute demonstration and 8-page report | §10–§11 | Timing plan, page budget, APA, disclosure, contributions |

The six marking dimensions (Appropriateness, Soundness, Excitement, Presentation,
Writing, Peer Review, 2.5% each) are mapped to evidence in §9.

---

## Abstract

We study text-to-image retrieval in a single product domain: footwear. A frozen CLIP
ViT-B/32 model encodes a 1,000-image catalogue and ranks images by cosine similarity to
a natural-language query. On top of this baseline we add a query parser and a zero-shot
attribute re-ranker: the parser turns a shopper phrase into colour, type and gender
constraints; CLIP text prompts turn the same attributes into per-image probability
tables; and a weighted product of those probabilities is added to the cosine score. On
60 queries, mAP@50 rises from 0.348 to 0.408 (+17.2%, paired bootstrap 95% CI +0.041 to
+0.082) and P@10 from 0.582 to 0.628 (+0.047, CI +0.015 to +0.080). A six-template prompt
ensemble, often treated as a free improvement, loses 5.1% on the same data. We calibrate
automatic labels against 200 human judgements and analyse five failure modes.

---

## 1. Introduction

### 1.1 Task and users

Text-to-image retrieval asks for images that match a free-form description. In product
catalogues the task is practical: shoppers can describe appearance ("grey casual shoes",
"silver sports shoes") but rarely know model names or stock codes. Metadata search is
the usual alternative, yet catalogue naming is inconsistent across colours and styles,
and many visual properties are never tagged. Retrieval that works directly on pixels
complements metadata search and supports visual browsing.

We choose footwear because the domain has attributes that are visually meaningful and
comparable across products: eight article types, forty colour values, and a colour space
in which neighbours (grey, silver, charcoal; tan, beige, khaki) are easy for a person to
tell apart and hard for a text encoder. That gap is where a domain-specific method can
demonstrate value.

Two groups use the system. Shoppers search by appearance when they cannot name a product;
catalogue operators use attribute search to assemble or check collections. Both need the
head of the ranking to be right, which is why we report P@10 alongside mAP@50.

### 1.2 What the system does

Given a query, the system returns a ranked list of gallery images, each with its cosine
similarity and, for the re-ranked list, an attribute agreement score. Two methods share
the gallery, the image features and the truncation: a frozen-CLIP cosine baseline, and
attribute-verified re-ranking. A Gradio page switches between them on the same query, so
the contribution is visible in the demonstration rather than asserted in text.

### 1.3 Contributions

1. A reproducible benchmark for a 1,000-image footwear gallery: 60 queries across three
   difficulty groups, 200 human relevance judgements, two widened label variants and a
   30-query frozen held-out set.
2. A training-free domain adaptation method: query parsing plus zero-shot attribute
   re-ranking, with each design choice isolated by an ablation.
3. A statistically tested +17.2% mAP@50 gain, a negative result for prompt ensembling,
   and a decomposition showing how much of the gain comes from a deeper candidate pool.
4. A failure taxonomy with visual evidence, and a calibration of automatic labels against
   human judgement that separates label noise from model error.

---

## 2. Related work

CLIP (Radford et al., 2021) learns a shared image-text space by contrastive pretraining on
400 million pairs and transfers to new tasks through prompt text. We use the open-source
reimplementation open_clip (Ilharco et al., 2021) with the original OpenAI weights and keep
the encoders frozen: with one thousand images there are no domain image-text pairs to
fine-tune on, and a frozen encoder makes the contribution of the adaptation layer
measurable.

Re-ranking is standard in retrieval: a cheap first stage proposes candidates and a more
expensive second stage reorders them (Nogueira & Cho, 2019). Our second stage is a
zero-shot attribute classifier rather than a trained cross-encoder, so it needs no
labelled pairs. Evaluation follows standard information-retrieval practice (Manning et
al., 2008), with confidence intervals from a paired bootstrap (Efron & Tibshirani, 1993).

The gallery comes from the Fashion Product Images dataset (Aggarwal, 2019), which pairs
catalogue photographs with article type, colour and gender. Its product-shot style keeps
the visual domain narrow, which suits a course-scale study: differences between methods
come from the method, not from background clutter.

---

## 3. Data

### 3.1 Source and sampling

The full dataset contains 44,441 images; 9,222 carry `masterCategory = Footwear`. We sample
500 men's and 500 women's items, rotating through `baseColour` within each gender so that
common colours contribute about fifty images each and rare colours remain as distractors.
The gallery covers eight article types: Casual Shoes (283), Heels (237), Sports Shoes
(162), Flip Flops (126), Flats (101), Sandals (60), Formal Shoes (21), Sports Sandals (10).

Source images are 1080×1440 (a few at 1800×2400), resized with Lanczos to
**480×640, JPEG quality 90**, 22.1 MiB in total. This is more than twice CLIP's 224-pixel
input, so the metallic highlights that separate gold, bronze and brown survive the resize.
The gallery ships with the repository, so reviewers encode exactly the pixels used in every
number below.

### 3.2 Attribute schema

| Field | Example | Role |
| --- | --- | --- |
| `id` | 15970 | File name and ranking output |
| `path` | `data/gallery/15970.jpg` | Relative to the repository root |
| `articleType` | Casual Shoes | Type slot and type labels |
| `color` | Black | Colour slot and colour labels (40 values) |
| `gender` | Men | Gender slot and labels (500 / 500) |
| `category` | Footwear | Sanity check only |
| `usage`, `season`, `year` | Casual, Summer, 2011 | Display only |
| `display_name` | "Puma Men Black Sneakers" | Interface caption only |

`dataset/manifest.csv` records the 1,000 sampled ids and their attributes; `dataset/NOTICE`
records provenance, licence (MIT) and the resize rule. The metadata is the source of the
attribute vocabulary, the automatic relevance labels and the interface captions.

### 3.3 Caching

Image features are cached in `data/image_features.pt` under a signature of image count,
newest modification time and total bytes. Replacing any image invalidates the cache, so
cached features always match the gallery that produced them.

### 3.4 Gallery size

The 1,000-image gallery gives each query between 10 and 283 relevant images (mean 54),
enough for a stable P@10 and a meaningful mAP@50. It also keeps a full re-encode near 15
seconds on an Apple M1, so the gallery and every number in this report can be rebuilt on a
laptop. All claims are scoped to this gallery.

---

## 4. Method

### 4.1 Baseline: frozen CLIP cosine

Images are encoded once with ViT-B/32 (224-pixel centre crop), L2-normalised to 512
dimensions, and cached. A query is encoded with the same text encoder, without templates,
and scored by the dot product, which equals cosine similarity for unit vectors. The
baseline returns the top 50 images. Queries are encoded as written, with no template; §6.8
quantifies how templates affect the score.

### 4.2 Query understanding

The parser fills three slots from vocabulary derived from the metadata: 36 colour words
covering all 40 colour values, 30 type phrases covering the eight article types and their
synonyms, and 9 gender words. Each alias maps to a list of catalogue values:

| Query | Colour (weight) | Type (weight) | Gender |
| --- | --- | --- | --- |
| `grey casual shoes` | Grey 1.0, Charcoal 0.5, Silver 0.5 | Casual Shoes 1.0 | — |
| `beige heels` | Beige 1.0, Mushroom Brown 0.5, Off White 0.5, Cream 0.5, Khaki 0.5, Tan 0.5 | Heels 1.0 | — |
| `sneakers` | — | Sports Shoes 1.0, Casual Shoes 1.0 | — |
| `sports sandals for women` | — | Sports Sandals 1.0 | Women |

Three design rules matter for scoring:

- **Longest phrase wins.** Matches are sorted by length; any alias contained in an accepted
  longer phrase is dropped. "Sports sandals" therefore parses as Sports Sandals and no
  longer fires "sports" (Sports Shoes) and "sandals" (Sandals).
- **Colour weights separate literal and family values.** The first value of an alias list
  is the word's own catalogue value and carries weight 1.0; the remaining values are the
  colour family it is often confused with and carry 0.5. A catalogue assigns one colour
  label per image, while CLIP places silver, grey and charcoal close together. Giving
  family values full weight pulls wrong-coloured images to the top; giving them zero weight
  discards images a person would accept for "grey". The half weight keeps the literal value
  dominant while allowing family credit. Section 6.3 shows the 0.25 alternative.
- **Type synonyms carry 1.0.** "Sneakers" is a genuine synonym for two catalogue types
  (Sports Shoes and Casual Shoes), and both should count fully; a fractional weight would
  penalise one arbitrarily. Gender is a single value, and is left unconstrained when the
  query names neither gender or names both.

The parser never rewrites the query text. The text encoder still sees the shopper's words;
the parse only supplies weights to the second stage, which avoids the text-side dilution
measured for prompt templates.

### 4.3 Zero-shot attribute tables

For each slot we encode one prompt per catalogue value — "a photo of a {colour} shoe",
"a photo of {type}", "a photo of a shoe for {gender}" — and apply a softmax over the values
of that slot with CLIP's logit scale of 100. The result is a distribution per image and
slot: 40 colours, 8 types, 2 genders. The tables are built once per process (50 text
encodes) and their vocabulary comes from the metadata value set, not from the evaluation
labels; the relevance labels never enter scoring.

### 4.4 Fusion scoring

For a query parsed into slots *a* with values *v* and weights *w*:

```
score(q, x) = cos(q, x) + β · Π_a ( Σ_{v∈a} w_v · P_a(v | x) )
```

The product across slots expresses AND semantics: a query naming a colour and a type wants
both at once, and a low probability in either slot suppresses the bonus. The sum inside a
slot lets a family value contribute in proportion to its weight.

The two weights interact with the scale of the two terms. Inside the top-100 candidates,
cosine spans roughly 0.04. The agreement term is a product of two or three probabilities;
on representative queries its maximum over the candidate list is 0.24–0.36, so with
β = 0.1 the bonus never exceeds 0.024–0.036 — the same order as the cosine differences it
has to resolve. This is why β = 0.1 can reorder near-ties without letting attribute
probabilities override visual similarity. Raising β to 0.2–1.5 lifts mAP to 0.413–0.416
but degrades grey and beige (Section 6.4); the aggregate gain comes from failures we can
see, which is a trade we decline.

The candidate pool is a separate constant from the metric truncation. With pure cosine,
retrieving 100 and truncating to 50 is identical to retrieving 50, so baseline and
re-ranking see the same candidate range and the same cutoff; re-ranking changes the order
and can promote images ranked 51–100. Section 6.4 reports pools of 50, 100 and 200.

### 4.5 Algorithm and cost

1. **Offline.** Encode the 1,000 gallery images (about 15 s on an Apple M1), L2-normalise,
   cache. Build the three attribute tables once per process.
2. **Query.** Encode the query text once.
3. **Coarse pass.** One matrix-vector product over 1,000 cached embeddings; keep the top 100.
4. **Attribute pass.** For each candidate, look up the probabilities of the named values,
   weight and multiply across slots; add β times the result to the cosine.
5. **Return.** Sort and return 50; the interface shows the first 10 with both scores.

The coarse pass is linear in gallery size (1,000); the rerank pass touches 100 candidates
and at most three slots. Image features occupy about 2 MB. Query latency is dominated by
the single text encode; we did not run a formal latency benchmark, which is listed as a
limitation in §7.2.

### 4.6 Interfaces

`searchCli.py` prints the ranking with cosine and attribute scores for quick inspection and
for reproducing individual cases. `searchWeb.py` is a Gradio page with a query box, a
Top-K slider and a method switch; it shows the parsed attributes above the image grid and
each image's cosine and attribute score. Both call the same functions as the evaluation, so
the demonstration and the reported numbers share one code path.

---

## 5. Evaluation design

### 5.1 Queries and relevance

| Set | Items | Labels | Role |
| --- | --- | --- | --- |
| `queries.json` | 60: colour 20, type 18, combo 22 | Exact metadata match on the named attributes | Main oracle for method comparison |
| `judgements.json` | 20 queries × 10 images = 200 cells | Human 0/1 on the baseline top-10 | Calibrate the absolute level of auto labels |
| `queries_family.json` | Same 60 queries | Colour families widened (e.g. Brown←Tan/Beige) | Test sensitivity to the colour definition |
| `queries_sandal.json` | Same 60 queries | Sandals and Sports Sandals merged | Test sensitivity to taxonomy |
| `queries_heldout.json` | 30: 20 paraphrases, 10 typos | Exact metadata match, frozen | Generalisation check outside development |

The 60 main queries are built to separate three difficulties. The twenty colour queries use
the twenty most frequent colours, one query each, covering both common colours and rare
distractors. The eighteen type queries cover every article type and add ten type-and-gender
queries that have at least twenty relevant images, which exercises both the type vocabulary
and the gender slot. The twenty-two combo queries pair a colour with a type and require at
least twelve relevant images; they reflect realistic shopper phrasing and test the AND
semantics that the product implements. A relevant set ranges from 10 to 283 images (3,220
in total). Because R@10 divides by the size of the relevant set, its ceiling on this
gallery is 0.368; we report it as a diagnostic and use mAP@50 as the primary metric.

The human subset draws six colour, six type and eight combo queries (200 cells), a size one
person can judge in a single sitting while covering all three groups. The held-out set was
frozen after the method was fixed: twenty paraphrases ("kicks", "trainers", "ballerinas",
"slippers", "oxfords") test vocabulary generalisation, and ten misspellings ("blak",
"silvr", "charcole") test parser robustness.

### 5.2 Relevance definitions

- **Auto.** An image is relevant if its metadata matches every attribute the query names:
  `color` equality, `articleType` equality, and `gender` equality when the query names one.
  This covers all 60 queries and is fully reproducible, at the cost of inheriting catalogue
  naming (a tan shoe may be labelled Beige and miss a "brown" query).
- **Strict.** A person judged the baseline top-10 for 20 queries, by eye, with a written
  rule: the dominant visible colour, same-hue neighbours counted together, type judged by
  visible design, gender rejected only when clearly opposite. Re-checking the twelve least
  certain cells changed one label and filled six that had been left blank.
- **Family and Sandal.** The same queries with the colour definition widened to families,
  or the two sandal types merged. These variants change what counts as correct; they test
  whether a conclusion depends on a strict label definition.
- **Held-out.** Exact metadata labels, frozen and never used for tuning.

The scoring function sees none of these labels: it compares image and text vectors and
metadata vocabulary. Labels are used only by the evaluation scripts, which keeps the test
set independent of the method.

### 5.3 Metrics

- **mAP@50** averages, over queries, the average precision of the top-50 list against the
  relevant set. It reflects the whole ranking and is the primary metric.
- **P@10** is the fraction of the top ten that is relevant. It reflects what a user sees
  first and is reported alongside mAP.
- **R@10** is the fraction of all relevant images that appear in the top ten. It depends on
  the size of the relevant set and is reported for completeness.

### 5.4 Statistics

Queries share one gallery and are correlated, so we compare methods with a paired bootstrap
over queries: 10,000 resamples with a fixed seed, drawing each query's two scores together
and reporting the 2.5th and 97.5th percentiles of the mean difference. We also report how
many queries improve, worsen or stay unchanged. A confidence interval that excludes zero is
read as a significant difference.

### 5.5 Controls and ablations

All comparisons use one gallery, one set of image features, one query file and one
truncation. We vary one factor at a time: the text side (prompt ensemble), the parser (four
configurations), the candidate pool (50, 100, 200), the fusion weight (0.1 and 0.2–1.5),
and the relevance definition (auto, family, sandal, strict). The held-out set is evaluated
once, after the method is fixed.

---

## 6. Results

### 6.1 Main comparison

| Method | mAP@50 | P@10 | R@10 | Relative mAP |
| --- | --- | --- | --- | --- |
| Baseline (frozen CLIP cosine) | 0.3481 | 0.5817 | 0.1896 | — |
| + prompt ensemble (6 templates) | 0.3302 | 0.5633 | 0.1855 | −5.1% |
| + query parsing + attribute re-ranking | **0.4078** | **0.6283** | **0.2024** | **+17.2%** |

Re-ranking improves mAP@50 by 0.0597 (95% CI +0.0413 to +0.0816; 49 queries better,
11 worse, none unchanged) and P@10 by 0.0467 (CI +0.0150 to +0.0800; 28 better, 15 worse,
17 tied). Both intervals exclude zero.

### 6.2 Group analysis

| Group (queries) | Baseline mAP / P@10 | Re-ranking mAP / P@10 | Δ mAP / Δ P@10 |
| --- | --- | --- | --- |
| Colour (20) | 0.256 / 0.580 | 0.311 / 0.655 | +0.0546 / +0.0750 |
| Type (18) | 0.443 / 0.678 | 0.538 / 0.739 | +0.0949 / +0.0611 |
| Combo (22) | 0.354 / 0.505 | 0.390 / 0.514 | +0.0356 / +0.0091 |

The largest mAP gain is on type queries, where eight visually distinct categories make the
zero-shot table reliable. The largest head gain is on colour queries, where parser alignment
matters most. Combo queries improve mainly deeper in the list: the product over two slots
penalises candidates that are uncertain in either slot, so the head changes little. We
report this rather than describing the method as solving multi-attribute queries.

### 6.3 Parser ablation

| Parser configuration | mAP@50 | P@10 | Grey AP | Beige AP | Olive AP |
| --- | --- | --- | --- | --- | --- |
| No masking, all alias values at 1.0 | 0.3946 | 0.5933 | 0.147 | 0.131 | 0.316 |
| + longest-phrase masking | 0.3944 | 0.5950 | 0.147 | 0.131 | 0.316 |
| + colour family at 0.5 (shipped) | **0.4078** | **0.6283** | 0.196 | 0.182 | 0.299 |
| + colour family at 0.25 | 0.4131 | 0.6300 | 0.218 | 0.207 | 0.294 |

Relative to the unweighted parser, the shipped configuration gains 0.0132 mAP (CI +0.0049 to
+0.0224; 31 better, 10 worse, 19 tied) and 0.0350 P@10 (CI +0.0117 to +0.0617). Masking
alone is neutral in aggregate (−0.0002) but makes the parse internally consistent, which is
what allows the family weight to act. The 0.25 alternative gains another 0.0053 mAP while
pushing olive and orange below their 0.5 values and widening the worst per-query regression,
so we keep 0.5.

### 6.4 Candidate depth and fusion weight

| Candidate pool | mAP@50 | P@10 | Pool recall |
| --- | --- | --- | --- |
| 50 | 0.3747 | 0.6250 | 0.477 |
| **100 (shipped)** | **0.4078** | **0.6283** | 0.655 |
| 200 | 0.4113 | 0.6283 | 0.818 |

The 0.0597 total decomposes into +0.027 from attribute re-ranking on the same top-50 pool
(0.3481 → 0.3747) and +0.033 from deepening the pool to 100 (0.3747 → 0.4078). P@10 barely
moves across depths (0.625 → 0.628), which places the head gain in the re-ranking itself
rather than in the deeper pool. A pool of 200 adds 0.0035, inside the noise, at twice the
scoring cost.

Raising β from 0.1 to 0.2–1.5 lifts mAP to 0.413–0.416, but grey falls from 0.196 to
0.184–0.191 and beige from 0.182 to 0.165–0.176. Higher β amplifies CLIP's colour
confusion, so the aggregate gain is bought with visible failures; we keep β = 0.1.

### 6.5 Human calibration and label sensitivity

| Relevance definition | Baseline mAP | Re-ranking mAP | Δ mAP (95% CI) |
| --- | --- | --- | --- |
| Auto, 60 queries | 0.3481 | 0.4078 | +0.0597 (+0.0413, +0.0816) |
| Colour family labels | 0.3997 | 0.4754 | +0.0757 (+0.0565, +0.0980) |
| Sandal-merged labels | 0.3498 | 0.4138 | +0.0640 (+0.0442, +0.0867) |

On the 20 human-judged queries, baseline P@10 is **0.925** (185 of 200 cells relevant),
against 0.625 on the same queries under automatic labels. The gap measures label tolerance,
not model quality: people count tan as brown, treat silver and grey as one family, and
accept shoes whose gender is not visually obvious. By group, human P@10 is 0.967 for colour
(58/60), 0.983 for type (59/60) and 0.850 for combo (68/80). The family and sandal variants
keep the re-ranking gain significant, so the conclusion does not rest on exact colour names.
The human pool covers the baseline top-10 only; re-ranking brings in unjudged images, which
count as zero, so strict is reported for the baseline alone.

### 6.6 Held-out generalisation

On the 30 held-out queries, mAP rises from 0.2579 to 0.3167 (+0.0588, CI +0.0050 to
+0.1184) and P@10 from 0.4433 to 0.4767 (+0.0333, CI −0.0267 to +0.0967). The mAP interval
excludes zero; the P@10 interval does not. Splitting by kind, the twenty paraphrases gain
0.0980 mAP and 0.0600 P@10, while the ten misspellings lose 0.0198 mAP and 0.0200 P@10. In
eight of the ten typos the parser misses at least one intended attribute, usually the
colour; the attribute term is then partial or absent and those queries fall back towards
plain cosine. The pattern is consistent with the design: the method helps when the
vocabulary is understood and does no harm when it is not.

### 6.7 Failure analysis

| Case | Baseline AP / P@10 | Re-ranking AP / P@10 | Diagnosis |
| --- | --- | --- | --- |
| `sports sandals` | 0.003 / 0.0 | 0.034 / 0.1 | Taxonomy: sandal and sports sandal are visually close and inconsistently labelled |
| `grey shoes` | 0.161 / 0.4 | 0.196 / 0.6 | Exact labels split silver and grey; under family labels 0.289 → 0.425 |
| `beige heels` | 0.119 / 0.3 | 0.106 / 0.4 | Tan/khaki family combined with heels; head improves, deep ranking still confused |
| `bronze heels` | 0.235 / 0.4 | 0.238 / 0.4 | Metallic colour depends on highlights that text cannot separate |
| `casual shoes` (control) | 0.450 / 0.7 | 0.867 / 0.9 | Strong type signal; re-ranking recovers it |

The five cases are rendered side by side, baseline above and re-ranked below, by
`evaluation/tools/renderFailureCases.py`. The figure is used in the report and the
demonstration. Two failures are label-side (sports sandals, grey) and two are visual-side
(beige heels, bronze); only the control is a clean success. Reporting all four prevents the
aggregate number from hiding the cases where the method is not usable.

### 6.8 Negative results and rejected variants

- **Prompt ensembling.** Six templates averaged lose 5.1% mAP; every template loses, and the
  loss grows with the number of attributes in the query. The worst template ("a {q} on a
  white background") drops combo P@10 from 0.505 to 0.386. A control that crops the image
  background widens the loss (−0.132 to −0.140), locating the cause on the text side: the
  template words dilute the colour and type terms inside one pooled vector.
- **Higher β.** See §6.4: aggregate gain, visible colour regressions.
- **Colour family at 0.25.** See §6.3: +0.0053 mAP, wider worst-case regression.
- **Per-slot β.** Separate colour and type weights improved mAP by less than 0.001, with a
  confidence interval crossing zero; one shared β is kept.
- **Contrastive colour term.** A penalty against confusable colours lowered bronze AP, so
  the score keeps the positive attribute term only.
- **Deeper attribute keys.** Laces, closures, toe shape and material appear in fewer than
  0.1% of `display_name` values and are not reliably visible at this resolution; they would
  require detection or a vision-language model, beyond the course scope.
- **Fine-tuning and captioning.** No domain image-text pairs exist for fine-tuning, and a
  captioning branch adds a lossy intermediate text; the method stays training-free.

---

## 7. Discussion

### 7.1 Why the gains appear where they do

Type queries gain most in mAP because eight categories are visually distinct and the
zero-shot table separates them well. Colour queries gain most at the head because the
parser, not the model, was the limiting factor: aligning the family weights removed the
systematic pull towards silver and khaki. Combo queries expose the cost of AND semantics:
when either slot is uncertain the product is small, so the head changes little even though
the deep ranking improves. Metallic colours and the sandal taxonomy remain unsolved; both
would need better labels or a model that reads material and construction.

### 7.2 Limitations and threats to validity

- **Gallery.** One thousand catalogue photographs on light backgrounds. Results may not
  transfer to user photographs, cluttered scenes or a much larger catalogue.
- **Labels.** Automatic labels inherit catalogue naming and are stricter than human
  judgement (0.625 vs 0.925 P@10 on the same 20 queries). Cross-method comparisons use the
  same labels, so the bias is shared, but absolute scores are not user-facing accuracy.
- **Human labels.** 200 cells on the baseline top-10 only. They calibrate the baseline and
  cannot rank a method that introduces new images.
- **Queries.** Authored by the group, in the phrasing of the catalogue. The held-out set
  mitigates overfitting but does not remove authorship bias.
- **Model and resolution.** One frozen backbone at 480×640. Larger models or higher
  resolution may change the colour confusions that motivate the method.
- **Cost.** Latency was not benchmarked; the efficiency item in the brief is optional and
  remains open.

---

## 8. Conclusion

A frozen CLIP model ranks footwear images reasonably, but its text space confuses colours
that shoppers distinguish. A parser that knows the catalogue's colour families, combined
with zero-shot attribute tables and a small fusion weight, raises mAP@50 by 17.2% and P@10
by 0.047 on a 1,000-image gallery, with both gains outside the bootstrap interval. The
method needs no training and no domain image-text pairs, and its failures are diagnosable:
taxonomy, metallic colours and multi-attribute queries remain hard. Prompt ensembling, the
usual first attempt at adaptation, is a net loss on this data. Future work would extend
human labels to the re-ranked pool, measure latency, and test the method on a larger and
messier gallery.

---

## 9. Marking dimensions (internal)

| Dimension | What we did | Evidence |
| --- | --- | --- |
| Appropriateness (2.5%) | Chose a domain with visually meaningful attributes; defined two user groups; defined relevance with metadata and human labels | §1, §3, §5.1–5.2 |
| Soundness (2.5%) | Frozen encoder; scoring independent of labels; paired bootstrap; one-factor ablations; held-out set; shipped gallery and scripts | §4, §5.4–5.5, §6, `results/` |
| Excitement (2.5%) | Negative results for prompt ensembling and rejected fusion variants; four diagnosed failures plus a control; live baseline/rerank switch | §6.7–6.8, §10 |
| Presentation (2.5%) | 7-minute structure with a live demo of one query under both methods; one results table; one failure figure | §10 |
| Writing (2.5%) | 8-page plan with tables, APA references, explicit limitations, AI disclosure and contributions | §11 |
| Peer Review (2.5%) | Review another group after all demonstrations, using the course form; submitted separately | §11.6 |

---

## 10. Presentation plan (internal)

### 10.1 Constraints

Seven minutes, no Q&A, shared over Zoom; setup counts towards the time. The server is
started before the session and the browser is already on the page, so the demo begins with
a query rather than an install.

### 10.2 Timing

| Time | Slide | Content |
| --- | --- | --- |
| 0:00–0:35 | 1 | Task, users, why footwear |
| 0:35–1:20 | 2 | Baseline and one visible failure |
| 1:20–2:40 | 3 | Parser, attribute tables, fusion formula, weights |
| 2:40–4:10 | 4 | Live demo: one query, both methods |
| 4:10–5:30 | 5 | Main results, confidence intervals, group caveat |
| 5:30–6:25 | 6 | Ablations, negative results, failure figure |
| 6:25–7:00 | 7 | Limitations and conclusion |

### 10.3 Slide script

**Slide 1 — Problem and users.**
"A shopper can describe a shoe — grey casual shoes, silver sports shoes — but cannot name
its stock code. Catalogue search needs keywords; we search pixels. Our users are shoppers
browsing by appearance and catalogue teams checking a collection."

**Slide 2 — Baseline.**
"We start from a frozen CLIP model and cosine similarity. It works, but it confuses colours
that people separate: ask for grey and silver shoes come back; ask for brown and tan shoes
are missed. The model sees colour words as neighbours in one text space."

**Slide 3 — Method.**
"We add two things without training. A parser turns the query into colour, type and gender,
with a literal weight of one and half weight for colour-family neighbours. CLIP text prompts
turn the same attributes into probability tables per image. We add the weighted product of
those probabilities to the cosine score — a product, because a query that names a colour
and a type wants both. The candidate pool is 100, the returned list 50."

**Slide 4 — Demo.**
"Same query, two methods. On the left, baseline; on the right, re-ranked. [Type `silver
sports shoes`.] The baseline returns grey and white shoes high; the re-ranked list puts
silver first. The line under each image is cosine and attribute agreement, so you can see
which term moved it. [Switch to `sports sandals`.] This one is a failure we report: the
catalogue's sandal types are visually close, and neither method fixes it."

**Slide 5 — Results.**
"Across 60 queries, mAP@50 goes from 0.348 to 0.408, plus 17.2%, with a paired bootstrap
interval that excludes zero; P@10 goes from 0.582 to 0.628. The gain is largest on type
queries and in the head of colour queries. Multi-attribute queries improve mostly deeper in
the list: their head P@10 moves only 0.009."

**Slide 6 — What did not work.**
"Prompt ensembling, the usual free trick, loses five percent: template words dilute the
colour and type terms. Raising the fusion weight buys total score with visible colour
failures, so we keep it small. Our failure figure shows four cases and one control:
taxonomy, silver versus grey, beige heels, metallic bronze."

**Slide 7 — Limits and close.**
"One thousand catalogue images, automatic labels that are stricter than people, and a human
check of 200 cells on the baseline only. Within that scope the method is training-free,
reproducible, and its failures are visible. The repository ships the gallery, the labels and
every script."

### 10.4 Recommended demonstration queries

Numbers below are read from `results/baseline_auto.json` and `results/rerank_auto.json`
(auto labels, 60 queries, β = 0.1, candidate depth 100), so each one is checkable in the
repository. Every query is typed into the demo with the method switch flipped between
baseline and re-rank.

| Query | Role in the talk | baseline AP / P@10 | re-rank AP / P@10 | What to point at |
| --- | --- | --- | --- | --- |
| `casual shoes` | Strongest single switch | 0.450 / 0.70 | **0.867 / 0.90** | Ranks 1–4 are all Casual Shoes with agreement 0.54–0.64; the baseline head mixes colours and a Sports shoe. |
| `silver sports shoes` | Colour separation (slide 4) | 0.254 / 0.50 | 0.317 / 0.50 | Silver first instead of white and grey. R@10 stays 0.185, so the gain is ordering inside the head — say so rather than over-claiming. |
| `beige casual shoes` | Visually clean backup | 0.394 / 0.60 | 0.386 / 0.50 | Top-5 all Beige/Khaki Casual. AP dips because tan and khaki count as non-relevant under metadata labels; use it to show the label caveat, not the gain. |
| `black casual shoes` | Honest limitation | 0.153 / 0.30 | 0.215 / 0.40 | Two black men's formal shoes survive in the head: the Casual/Formal zero-shot gap is small, and at β = 0.1 the attribute term moves the score by 0.003–0.006 against a cosine spread of the same size. |
| `sports sandals` | Diagnosed failure (slide 4) | 0.003 / 0.00 | 0.034 / 0.10 | Sandal subtypes are visually close and the metadata split is finer than the pixels; neither method fixes it. |
| `bronze heels` | Backup for the metallic case | 0.235 / 0.40 | 0.238 / 0.40 | Bronze reads as brown; the attribute term is a near no-op here, which matches `evaluation/montages/failureCases.png`. |

Slide 4 uses `silver sports shoes` then `sports sandals`. `casual shoes` is the best
replacement if the audience needs one obvious win, and `black casual shoes` is the query to
volunteer when someone asks what still fails.

### 10.5 Demo checklist

- Start `python retrieval/searchWeb.py` before the session; confirm the page loads with the
  cached features (a few seconds, no model download).
- Pre-type the queries from §10.4 in the order you plan to use them; keep `bronze heels` as
  the spare.
- Check that the method switch re-runs the query; show both lists for the same query.
- Record a 90-second backup video in case the network or projector fails.
- Zoom share the application window, not the desktop; font size at least 18 pt.

---

## 11. Report plan (internal)

### 11.1 Page budget (A4, 12 pt Times New Roman, 1-inch margins, single column)

| Section | Pages |
| --- | --- |
| Title, abstract, introduction | 0.9 |
| Related work | 0.4 |
| Data | 0.6 |
| Method | 1.6 |
| Evaluation design | 0.9 |
| Results | 1.9 |
| Discussion and conclusion | 0.6 |
| AI disclosure and contributions | 0.4 |
| References | 0.7 (not counted) |

The body (abstract to conclusion) currently runs to about 4,400 words. With two figures and
four tables, the 8-page limit needs roughly 3,700 words of body text. Planned cuts: the
group commentary in §6.2 to three sentences; the β paragraph in §6.4 to one sentence (the
table carries the numbers); the held-out paragraph in §6.6 to two sentences; §6.8 to four
bullets; §3.1 and §5.1 to their tables plus one sentence each. These cuts remove about 700
words without dropping a claim.

### 11.2 Figures and tables

- Figure 1: method diagram (query → parser → three tables → fusion → ranking). Draw once
  for the report and the slides.
- Figure 2: `evaluation/montages/failureCases.png`, five cases, baseline above, re-ranked
  below.
- Table 1: dataset and query sets (merge §3.2 and §5.1).
- Table 2: main results and group breakdown (merge §6.1 and §6.2).
- Table 3: parser ablation and candidate depth (merge §6.3 and §6.4).
- Table 4: label sensitivity, held-out and failures (merge §6.5–6.7, or keep failures as a
  short list if space is tight).

### 11.3 Formatting checklist

- 8 pages maximum; 12 pt Times New Roman; 1-inch margins; single spacing; no cover page or
  table of contents.
- APA 7th edition for references and in-text citations.
- Figures and tables count towards the page limit; references and peer review do not.
- Every number in the report must match a file in `results/`; cite the script that produced
  it in the caption or the text.
- Include the AI-use disclosure and the member-contribution table (templates below).

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

### 11.6 Peer review

Peer review is assigned after all demonstrations. Our group reviews another group's
presentation with the course form, focusing on whether the relevance definition is stated
clearly and whether the demonstration runs within the time limit. The review is submitted
separately and is not part of the 8 pages.

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
A., Mishkin, P., Clark, J., Krueger, G., & Sutskever, I. (2021). Learning transferable
visual models from natural language supervision. In *Proceedings of the 38th International
Conference on Machine Learning* (pp. 8748–8763). PMLR.

---

## Appendix A. Reproduction commands (not for submission)

```bash
# Main oracle
python evaluation/tools/runEvaluation.py --method baseline
python evaluation/tools/runEvaluation.py --method prompt
python evaluation/tools/runEvaluation.py --method rerank
python evaluation/tools/runEvaluation.py --method baseline --labels strict

# Widened labels
python evaluation/tools/runEvaluation.py --method rerank --queries evaluation/sets/queries_family.json
python evaluation/tools/runEvaluation.py --method rerank --queries evaluation/sets/queries_sandal.json

# Held-out and significance
python evaluation/tools/runEvaluation.py --method rerank --queries evaluation/sets/queries_heldout.json
python evaluation/tools/compareResults.py results/baseline_auto.json results/rerank_auto.json
python evaluation/tools/compareResults.py results/baseline_auto_heldout.json results/rerank_auto_heldout.json --by kind

# Ablation and figure
python evaluation/tools/ablateParser.py
python evaluation/tools/renderFailureCases.py
```
