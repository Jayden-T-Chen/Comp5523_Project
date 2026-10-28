# COMP5523 文生图检索 — 鞋子域 完整方案（v0.1）

## 0. 目标与约束

- **任务**：domain-specific text-to-image retrieval。输入任意自然语言 query（如 "red sneakers with white soles"），从图库返回相关图排序。
- **域**：鞋子（时尚域，spec 第一个示例的天然对位）。
- **约束**：MacBook Air M1（无 CUDA，可选 MPS 加速），~10 天完成核心，训练量近零。
- **交付**：11-18 课堂演示（7 min + 录屏 demo），11-25 报告（≤8 页 A4）。
- **评分**：Appropriateness / Soundness / Excitement / Presentation / Writing / Peer Review，各 2.5%。

---

## 1. 最小骨架 vs 完整链路（区别在哪）

"最小骨架"不是"另一个方案"，而是**完整链路的第 1 步**：先搭一条最短但端到端能出数的流水线，验证代码正确，再在每个环节逐个加"能带来可量化增益"的组件。

| 环节 | 最小骨架（第 1 步） | 完整链路（最终） |
|---|---|---|
| 图库 | ~100 张切片 smoke test | ~1000 张均衡抽样 |
| 跨模态表示 | 冻结 CLIP ViT-B/32 零样本，单模板 | 同左 + 多模板 ensemble + 可选 LLM 扩写 |
| 属性 schema | 手写 8–12 属性，关键词规则解析 | 属性词典（含同义词/否定词）+ 规则/LLM 解析 |
| 属性来源 | 只用数据集标签当 oracle | 数据集标签 oracle + CLIP 零样本打标（系统自身估计） |
| 重排 | 规则打分（属性命中计数），固定权重 | 学到的融合权重 / 小 MLP（冻结特征上，M1 分钟级） |
| 领域适应 | 无 | 提示工程（必做）+ 软提示 CoOp / 小 adapter（可选） |
| 评测 | 只算 P@10 | mAP + P@10 + Recall@10 + latency + 按属性拆 + 失败分析 |

**递进逻辑**：骨架证明"链路通、能出数"；完整链路把每个组件配一个消融对照，报告里就能写"去掉 X 掉多少点"，这是 Soundness + Excitement 的核心来源。

---

## 2. 数据集选取

### 主选：Fashion Product Images（Small），footwear 子集 —— 已下载

- Kaggle `paramaggarwal/fashion-product-images-small`，565MB，`styles.csv` + `images/`。
- **实际规模**：footwear 共 9222 行；**平衡采样 1000 张**（Men/Women 各 500），落 `data/gallery/` + `data/metadata.csv`。
- **采样策略**：性别硬平衡 500/500；颜色按 `baseColour` 在每个性别内轮转 → 主色（Black/White/Brown/Grey/Blue/Red/Green/Silver/Beige/Yellow/Navy/Tan）各约 48–50 张，稀有长尾色保留作干扰项。
- **articleType 分布**：Casual Shoes 283、Heels 237、Sports Shoes 162、Flip Flops 126、Flats 101、Sandals 60、Formal Shoes 21、Sports Sandals 10（后两者偏少，若要纳入评测可再加一档类型平衡）。
- **图片是 60×80 缩略图**：检索/演示够用但放大偏糊；要高清换 full 版（24.7GB）只取抽样 id。
- **类别用 `articleType` 不是 `subCategory`**：`subCategory` 只有 3 类，`articleType` 才有区分度。

### 备选：UT-Zap50K（属性更细，但有获取摩擦）

- 5 万鞋图，自带材质 / 鞋头 / 跟高 / 闭合等细粒度属性标签。
- 缺点：官方渠道常需填表申请，或找镜像；适合想要"细属性当 oracle"时再切换。

**选型结论**：只要能拿到「类别 + 颜色」两个可靠标签当 oracle 就够了。材质/跟高/鞋头/闭合这些**细属性交给系统的 CLIP 零样本打标**（作为"系统能力"展示，而不是 ground truth）——这样评测既诚实又可复现，还天然多了一个"tagging"组件可讲。

### 数据落盘与合规

- 图片文件**不入 git**（`.gitignore`）；入 git 的是下载脚本 + `dataset/manifest.csv`（锁定这 1000 个 id）+ `queries.json`；`metadata.csv` 与图片由脚本按 manifest 生成（逐步校验，上游变了会报错）。
- 预处理：去重（按文件名/哈希）、去损坏图、按许可用数据；不包含人脸/个人信息。

---

## 3. 属性 schema（鞋子的 relevance 定义）

Relevance = 类别匹配 + 颜色匹配 + 其余属性匹配。schema 分两类来源：

**来自数据集（oracle + metadata）**：

| 属性 | 取值示例 |
|---|---|
| category（用 `articleType`） | Flip Flops, Sandals, Sports Shoes, Casual Shoes, Heels, Flats, Formal Shoes, Sports Sandals |
| color | red, black, white, blue, brown, beige, grey, green, multi… |
| gender | men, women, unisex |
| usage | casual, formal, sports, party |

**来自 CLIP 零样本打标（系统自身估计）**：

| 属性 | 取值示例 |
|---|---|
| material | leather, canvas, suede, mesh, synthetic, rubber |
| heel height | flat, low, mid, high |
| toe style | round, pointed, open, square |
| closure | laces, slip-on, buckle, strap, zipper |

**颜色同义词映射**是关键的细活：`baseColour` 有几十个值，query 的 "red" 要映射到 `[Red, Maroon, Burgundy, Wine, Rust]` 这一组。这份映射表（`data/color_synonyms.py`）既是评测 oracle 的一部分，也是 query 解析的一部分。

---

## 4. 检索链路（数据流）

```
query 文本
   │
   ├─(1) query 解析 ──→ 结构化属性 {category, color, material, heel, …}
   │
   ├─(2) CLIP text 编码（模板 ensemble / LLM 扩写）──→ q_text_emb
   │
   └─(3) Stage-1 粗筛：cosine(q_text_emb, img_emb) ──→ top-K (K=50)
              │
              ▼
        (4) Stage-2 重排：对 top-K 逐张算"query 属性 vs 图属性"一致性
              score = w1·cos_sim + w2·attr_agree   （w 学到或固定）
              │
              ▼
        (5) 输出 top-10 + 每张的 relevance score（UI 展示）
```

图库侧（离线一次）：

```
图片 ──CLIP image 编码──→ img_emb.npy（L2 归一化，缓存）
图片 ──CLIP 零样本打标──→ img_attr.npy（每属性一个 one-hot/概率，缓存）
```

---

## 5. 指标与消融对照

**指标**（同一 eval set 上算，test 只碰一次）：

- **mAP**（主）：反映整条排序质量。
- **P@10**：前 10 张相关比例（对应用户只看首页）。
- **Recall@10**：相关图召回率。
- **query latency**（可选）：平均每条查询耗时，spec 点名的效率加分项。

**消融对照（对照组）**——对照的是"去组件版 vs 完整版"的**量化提升 + 定性变化**：

| 组 | 配置 | 要回答的问题 |
|---|---|---|
| 1. Baseline | 冻结 CLIP，单模板 `a photo of a {query}`，cosine top-K | 朴素底线是多少 |
| 2. +Prompt | CLIP + 多模板 ensemble + LLM 扩写 | 提示工程带来多少增益 |
| 3. 完整 B | +Prompt + 属性解析 + 属性验证重排（学融合权重） | 属性验证/重排是核心增益来源 |

每组报 mAP/P@10/Recall@10 看增幅；再**按属性维度拆 P@10**（颜色/类别/材质各一组）看系统强弱点；最后定性分析失败案例（哪类 query 靠哪个组件救回来）。

---

## 6. 代码架构

```
comp5523project/                        # ✅ 已有 / ⬜ 待补
├── README.md                           # 用法总入口（结构、启动、依赖版本、数据、评测、分工）
├── dataset/                            # ① 数据的定义与构建
│   ├── manifest.csv                    # ✅ 入 git：1000 个 id + 属性（唯一数据真源）
│   └── buildGallery.py                 # ✅ 下载 + 按 manifest 出 1000 张；--resample 重建划分
├── retrieval/                          # ② 检索主链路
│   ├── clipCore.py                     # ✅ 内核：设备/模型/嵌入缓存/打分
│   ├── searchCli.py                    # ✅ 命令行壳
│   ├── searchApp.py                    # ✅ Gradio 网页壳（演示/录屏）
│   ├── queryParse.py                   # ✅ query → {color, type, gender}，词表对齐 metadata
│   └── rerank.py                       # ✅ 属性校验重排（已注册成 eval 的 method：rerank）
├── evaluation/                         # ③ 评测
│   ├── buildQueries.py                 # ✅ 生成 60 条 query（确定性）
│   ├── queries.json                    # ✅ 入 git：query + 自动相关集 + strict 标记
│   ├── judgements.json                 # ✅ 入 git：人眼精标（20/20 条）
│   ├── renderTopHits.py                # ✅ top-10 拼图，供人眼判
│   ├── montages/                       # 生成，不入 git
│   └── runEvaluation.py                # ✅ mAP/P@10/R@10，按 group 拆，写 results/
├── results/                            # ④ <method>_<labels>.json，入 git
├── data/                               # ⑤ 生成产物，整体不入 git
│   ├── raw/                            #    Kaggle 原始快照（44k 图）
│   ├── gallery/                        #    抽出的 1000 张
│   ├── metadata.csv                    #    manifest + path
│   └── image_features.pt               #    嵌入缓存
└── docs/
    └── plan.md                         # 本文件
```

分层原则：`dataset/` 定义输入、`retrieval/` 只管排序、`evaluation/` 只管判分（import `retrieval/clipCore.py` 复用能力），`data/` 与 `results/` 是产物。模块名 camelCase。

**关键函数签名（伪代码）**：

```python
# features/embed.py
def load_clip(model_name="ViT-B/32"): -> (model, preprocess, tokenizer)
def embed_images(img_paths, batch=32) -> np.ndarray       # [N, d] L2-normalized
def embed_texts(texts) -> np.ndarray                      # [M, d]

# features/tag.py
def tag_images(img_paths, attr_schema) -> np.ndarray      # [N, A] 每属性 argmax/概率
def attr_prompt(attr, value) -> str                       # "a shoe with {value} {attr}"

# retrieval/queryParse.py
def parse_query(text, attr_schema, color_map) -> dict     # {category, color, material, ...}

# retrieval/rerank.py
def attr_agree(query_attrs, img_attrs, attr_weights) -> float
def rerank(q_emb, topk_emb, topk_attr, query_attrs) -> list[int]  # 重排后索引

# retrieval/searchPipeline.py
def search(query, gallery, topk=50, top=10) -> list[(img_id, score)]

# evaluation/runEvaluation.py
def ap(ranked_ids, relevant_ids) -> float
def precision_at_k(ranked_ids, relevant_ids, k) -> float
def recall_at_k(ranked_ids, relevant_ids, k) -> float
def mean_ap(all_ranked, all_relevant) -> float
```

**离线缓存约定**：`img_emb.npy`、`img_attr.npy` 只在图库变化时重算；query 端每次实时编码。这样 demo 的 query latency 只含"query 编码 + 点积 + 重排"，秒级可展示。

---

## 7. 10 天节奏 + 分工建议

| 天 | 内容 | 对应组件 |
|---|---|---|
| 1–2 | 定域 + 取数 + 清洗 + 抽样 + metadata | data/* |
| 2–3 | 图库嵌入 + baseline 检索（骨架跑通） | features/embed + search |
| 3–4 | 写评测 query + 借属性标签定 ground truth | evaluation/queries.json |
| 4–6 | 属性解析 + 零样本打标 + 重排（完整 B） | tag + query_parse + rerank |
| 6–7 | 轻量适应（学融合权重/软提示）+ 调参 | rerank 权重 |
| 7–8 | 全量评测 + 失败分析 | eval |
| 8–10 | Gradio UI + 录屏 demo + 幻灯片 | app |

分工（4–6 人）：数据/评测集 1–2 人，检索/表示 1–2 人，重排/属性 1 人，UI/demo/报告 1 人，Peer review 专人。

---

## 8. 风险与兜底

- **数据拿不到/许可受限**：换 Fashion Product Images 的其他子集，或自己拍/收集 500 张鞋图（spec 允许"组员自拍"）。
- **M1 装依赖摩擦**：兜底走 Colab 免费 GPU 做嵌入与训练，本地只跑 Gradio。
- **细属性打标不准**：评测 oracle 只用类别+颜色，细属性只作"系统能力"展示，不影响指标公正性。
- **时间紧**：软提示/adapter 标为"可选"，砍掉不影响主线完整度；方案 C（BLIP 描述分支）已按要求排除。

---

## 9. 参考实现与数据选型决策

参考仓库：`github.com/micycc/comp5523-sneakers-demo`（同为 COMP5523 的 CLIP + Gradio demo）。

**可复用**：`buildGallery.py`（Kaggle 下载 + 分层抽样 + 补足）、`searchApp.py`（Gradio Blocks + 嵌入缓存按 mtime 失效）、`searchCli.py`（CLI，当前只用 open_clip 后端）、`.gitignore` 约定。

**它缺的（正是评分点，必须自己补）**：无评测集/指标、无属性解析与重排、无 captioning 分支、无消融与失败分析。它等价于本文档的"组 1 Baseline"。

**数据选型（待决策）**：
- A. 沿用该仓库的 `Popular Sneakers Classification`（50 个 sneaker 型号、类别=品牌+型号，MIT）：骨架现成、好跑；但相关性偏"品牌+型号+颜色"识别，属性重排空间小。
- B. `Fashion Product Images`（Kaggle，footwear 子集，含 boots/sandals/heels/loafers）：属性故事强、失败分析好写；需自建数据管线（可借 A 的下载脚本骨架）。
- 结论：先用 A 跑通 baseline 验证骨架，再决定是否换 B 撑属性重排；最省事的折中是保留 sneaker、把 relevance 定义为 "brand + model + color"。

**使用纪律**：当作参考实现，不整包抄；借来的代码在报告/演示的 AI 使用与贡献说明里写清来源。
