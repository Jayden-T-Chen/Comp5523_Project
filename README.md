# COMP5523 · 鞋子文本检索（Domain-Specific Text-to-Image Retrieval）

课程项目代码仓库。做的是**一个检索系统 + 自建评测 + 消融实验**，不是"微调一个模型"：
骨架是冻结的 CLIP ViT-B/32，我们在它上面叠轻量的领域适配（prompt 模板、查询解析、属性校验重排），
用 mAP / P@10 这类指标量化"每一层适配加了多少分"。完整方案见 [`docs/plan.md`](docs/plan.md)。

当前进度：baseline（冻结 CLIP + 嵌入缓存 + CLI/Gradio）已跑通；60 条评测集与 mAP/P@10/R@10 打分已可用；
查询解析（长短语屏蔽 + 色族权重）+ 属性校验重排已接进评测和 CLI/Gradio（auto mAP 0.329 → 0.397，
mAP 与 P@10 的配对 bootstrap 都显著），人眼精标 20 条（strict P@10 0.875）。

---

## 0. 目录结构

按职责分层（数据 / 检索 / 评测与结果 / 文档），而不是按文件类型堆在一起：

```
comp5523project/
├── README.md                   # 结构、启动、依赖版本、数据、评测、分工
│
├── dataset/                    # ① 数据：怎么造出这 1000 张图
│   ├── manifest.csv            #    入 git，唯一的数据真源（1000 个 id + 属性）
│   └── buildGallery.py         #    下载 Kaggle → 按 manifest 复制 → data/gallery + metadata.csv
│
├── retrieval/                  # ② 检索主链路：query 进，排序出
│   ├── clipCore.py             #    内核：设备选择 / 模型加载 / 嵌入缓存 / 打分
│   ├── queryParse.py           #    query → {color, type, gender}，输出 metadata 词表
│   ├── rerank.py               #    属性校验重排：cos + β·Π 属性概率
│   ├── searchCli.py            #    命令行壳：快速看排序和分数
│   └── searchApp.py            #    网页壳：Gradio，演示与录屏
│
├── evaluation/                 # ③ 评测：分数从哪来
│   ├── buildQueries.py         #    生成 60 条 query（确定性）
│   ├── queries.json            #    入 git：query + 自动相关集 + strict 标记
│   ├── judgements.json         #    入 git：人眼精标（qid → {图片 id: 0/1}）
│   ├── renderTopHits.py        #    渲 top-10 拼图，供人眼判
│   ├── renderFailureCases.py   #    baseline / rerank 上下对照图（报告与演示复用）
│   ├── compareResults.py       #    两份 results 的配对 bootstrap（delta + 95% CI）
│   └── montages/               #    拼图输出（生成，不入 git）
│
├── results/                    # ④ 历史成绩：<method>_<labels>.json，入 git
├── data/                       # ⑤ 运行产物，全部生成、不入 git
│   ├── raw/                    #    原始 Kaggle 快照（44k 图）
│   ├── gallery/                #    抽出的 1000 张（检索真正用的图库）
│   ├── metadata.csv            #    manifest + 本地路径
│   └── image_features.pt       #    CLIP 嵌入缓存
└── docs/
    ├── plan.md                 # 方案与排期
    └── specAlignment.md        # 课程要求 → 当前实现对照（报告/答辩用）
```

三条边界，看名字就知道东西该放哪：

- `dataset/` 是**输入的定义**（要哪 1000 张、怎么拿），`data/` 是**跑出来的结果**——所以一个入 git、一个不入。
- `retrieval/` 只关心"给定图库和 query，怎么排序"，不关心评测集长什么样。
- `evaluation/` 只关心"怎么判对错、怎么算分"，通过 import `retrieval/clipCore.py` 复用检索能力。

模块名用 camelCase（`buildGallery.py`、`runEvaluation.py`），类名照常用 PascalCase。

---

## 1. 日常启动（环境已装好，直接跑）

```bash
cd ~/Documents/project/comp5523project
source .venv/bin/activate                                 # Windows: .venv\Scripts\Activate.ps1

python retrieval/searchCli.py --top 5                     # 命令行看排序
python retrieval/searchApp.py                             # 网页：http://127.0.0.1:7860，Ctrl+C 停止
```

看分数（不用起网页）：

```bash
python evaluation/runEvaluation.py --method baseline                  # auto 标签，60 条
python evaluation/runEvaluation.py --method baseline --labels strict  # 人工精标过的 query
```

> 嵌入缓存 `data/image_features.pt` 已存在时不会重算，起服务约几秒。

## 2. 首次安装（换机器走这里）

```bash
git clone <仓库地址> comp5523project
cd comp5523project

# 1) 虚拟环境（Python 3.11/3.12）
python3.12 -m venv .venv && source .venv/bin/activate      # Windows: py -3.12 -m venv .venv && .venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install torch==2.14.1 torchvision==0.29.1 open-clip-torch==3.3.0 \
                     gradio==6.31.0 Pillow==12.3.0 numpy==2.5.4 kaggle==2.2.5

# 2) Kaggle 凭据（只需一次）
mkdir -p ~/.kaggle && echo 'KGAT_你的token' > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token
#   Windows: 放到 C:\Users\<你>\.kaggle\access_token；也可用环境变量 KAGGLE_API_TOKEN=KGAT_...

# 3) 取数据：下载 Kaggle → 按 manifest 复制出 1000 张 → 写 metadata.csv
python dataset/buildGallery.py

# 4) 验证检索
python retrieval/searchCli.py --query "black leather shoes" --top 8
```

> 已在 MacBook Air M1（macOS 14.8 / Python 3.12 / torch 2.14.1）实测：`device: mps`，1000 张嵌入约 27 秒，CLI 与 Gradio（`127.0.0.1:7860` 返回 200）均正常。
> 首次运行会看到 `QuickGELU mismatch ... pretrained tag 'openai'` 的 UserWarning，是 open_clip 加载 openai 权重的已知提示，可忽略。

---

## 3. 文件说明

| 文件 | 职责 | 怎么用 |
| --- | --- | --- |
| `retrieval/clipCore.py` | 检索内核：读 `data/metadata.csv`、选设备（mps/cuda/cpu）、加载 CLIP ViT-B/32、批量编码并缓存到 `data/image_features.pt`、文本编码与 top-k | 被其它脚本 import，不直接运行 |
| `retrieval/searchCli.py` | 命令行检索，验证数据/模型/排序 | `python retrieval/searchCli.py --query "..." --top 8`（`--query` 可重复） |
| `retrieval/queryParse.py` | 把 query 拆成颜色/类型/性别；长短语屏蔽子串，颜色族给 0.5 权重、鞋型同义词给 1.0 | 被 `rerank.py` import，不直接运行 |
| `retrieval/rerank.py` | 属性校验重排：零样本属性表 + `cos + β·Π(加权P)` | 经 `runEvaluation.py --method rerank` 调用 |
| `retrieval/searchApp.py` | Gradio 网页：输入框 + Top-K + 图片墙 + 排行 | `python retrieval/searchApp.py` |
| `dataset/manifest.csv` | **唯一入 git 的数据**：1000 条样本的 id 与属性，"到底是哪 1000 张" | 脚本读取，不要手改 |
| `dataset/buildGallery.py` | 下载/解压 Kaggle、按 manifest 复制图片、写 `data/metadata.csv`；`--resample` 才重新抽样 | `python dataset/buildGallery.py` / `--resample`（仅维护者） |
| `evaluation/buildQueries.py` | 生成 `queries.json`：60 条 query（色 20 / 类型 18 / 组合 22） | `python evaluation/buildQueries.py` |
| `evaluation/queries.json` | 评测集本体：query 文本、属性、自动相关 id、`strict` 标记 | 由上一行生成 |
| `evaluation/renderTopHits.py` | 把某条 query 的 top-10 渲成带 id 的拼图 | `python evaluation/renderTopHits.py --qid color_black` |
| `evaluation/renderFailureCases.py` | 五个 case 的 baseline / rerank 上下对照图 | `python evaluation/renderFailureCases.py` → `evaluation/montages/failureCases.png` |
| `evaluation/judgements.json` | 人眼精标：`qid -> {图片 id: 0/1}`，只判 top-10 池 | 看拼图后手填 |
| `evaluation/runEvaluation.py` | 打分：mAP@50 / P@10 / R@10，按 group 拆，导出 `results/<method>_<labels>.json` | `python evaluation/runEvaluation.py --method baseline [--labels strict] [--beta 0.2]` |
| `evaluation/compareResults.py` | 两份 results 的配对 bootstrap：Δ 与 95% CI、胜/负/平、分组 Δ | `python evaluation/compareResults.py results/baseline_auto.json results/rerank_auto.json` |
| `results/` | 各方法 × 各标签口径的历史成绩 | 由 `runEvaluation.py` 写入，入 git 便于对比 |
| （无 requirements 文件） | 依赖版本直接写在 §2 的安装命令里，实测即锁定 | 照 §2 复制一行安装 |
| `docs/plan.md` | 方案：数据集、属性 schema、指标与消融、架构、排期 | 写报告/分工时对着看 |
| `docs/specAlignment.md` | 课程 spec（docx/pptx）逐条对应到当前实现：任务、组件、指标、结果、未完成项 | 写报告 §1/§2 与答辩提纲时对着看 |

---

## 4. 数据：只传脚本 + manifest，不传图片

**`raw` 与 `gallery` 的区别**：`data/raw/` 是 Kaggle 数据集的完整快照（44,441 张，全品类：衣服、包、鞋都有），只作原料；`data/gallery/` 是从中挑出的 1000 张鞋图，才是检索系统的图库，被编码的就是它。

**能保证每次下的都是同一批 1000 张吗？** 能，三层保险：

1. 数据集是固定快照，同一个 id 永远对应同一张图。
2. **id 清单入 git**（`dataset/manifest.csv`），脚本默认"复现模式"——不重新抽样，只按清单复制，抽样的随机性被完全绕开（`--seed 42` 只在 `--resample` 时用）。
3. 复制时逐条校验属性与存在性：`styles.csv` 变了就打印 WARNING/ERROR 并非 0 退出，不会静默换图。

已实测：两次运行 `data/gallery` 文件名集合 md5 一致；`--resample --seed 42` 生成的 manifest 与原文件逐字节相同。

```bash
python dataset/buildGallery.py                                   # 复现既定 1000 张（日常）
python dataset/buildGallery.py --resample --target 1000 --seed 42  # 维护者重建划分，覆盖 manifest
python dataset/buildGallery.py --force                           # 强制重新下载
```

抽样策略：`masterCategory == Footwear`（9222 行）→ Men/Women 各 500 → 性别内按 `baseColour` 轮转，主色各约 50 张，稀有长尾色保留作干扰项。类别看 `articleType`（8 类）而不是 `category`（3 类）。图是 **60×80 缩略图**：检索和演示够用，放大偏糊。

`data/raw`（632MB）在 gallery 生成后可删；删了就不能重跑 `--resample` 或做属性校验。

---

## 5. 评测：分数怎么来的

`evaluation/` 里四样东西分工很明确：`queries.json` 说"问什么、什么算对"，`judgements.json` 说"人眼认为什么算对"，`runEvaluation.py` 只负责算分并把结果写进 `results/`，`compareResults.py` 只负责两份结果的配对比较（不重新检索）。

两套标签口径，用途不同：

- **auto**（`queries.json` 的 `relevant`）：按 metadata 的 `color`/`articleType`/`gender` 自动构造，60 条全覆盖、零人工、可复现。缺点是**标签就是商品命名里的颜色**，不等于图里看起来的颜色，绝对分数会被系统性压低；适合做**方法之间的相对比较**（同一套标签，谁高谁低可信）。
- **strict**（`judgements.json`）：对 top-10 池逐张肉眼判相关。覆盖面小，但接近"人觉得对不对"，用来**校准 auto 的噪声**。

**关键限制：strict 不能用来给新方法排名。** 精标的池子是 baseline 的 top-10，新方法排进来的新图没判过，`eval_strict` 一律记 0——换个 prompt 就能把 0.875 打成 0.705，那是池子不完整，不是模型变差。所以：**跨方法比较只看 auto（标签覆盖全库），strict 只报 baseline 的绝对水平。** 想用 strict 排名，得把候选方法的池子合并后再判一遍。

### 5.1 baseline：冻结 CLIP，纯余弦

| 口径 | 范围 | mAP@50 | P@10 | R@10 |
| --- | --- | --- | --- | --- |
| auto | 60 条 | 0.329 | 0.538 | 0.180 |
| auto | 其中 20 条 strict 子集 | 0.330 | 0.565 | 0.164 |
| strict | 同样 20 条 | — | **0.875** | — |

strict 分组 P@10：color 0.850（6 条）/ type 0.900（6 条）/ combo 0.875（8 条）。**同一批 20 条 query，换标签口径从 0.565 跳到 0.875**——差值几乎全在 auto 的标签噪声上，而不是模型变好：

| 组 | auto P@10 | strict P@10 |
| --- | --- | --- |
| color | 0.533 | 0.850 |
| type | 0.717 | 0.900 |
| combo | 0.475 | 0.875 |

最典型的是 brown：auto 只有 0.2，人眼判 1.0——因为 Tan/Beige 的鞋肉眼就是棕色，标签却不算。

判得最勉强的 7 张（金属色、绿-卡其、低跟 mule 的边界）由本人复查过一遍，`judgements.json` 的 `review` 字段记录了改判的 3 张（运动鞋不算 casual、金色不算 bronze、芥末卡其算 beige 一族）和维持原判的 4 张；颜色族规则也写在同一文件的 `rule` 里，报告的方法学部分照抄即可。

### 5.2 消融：60 条 auto，三个方法同一脚本

| 方法 | mAP@50 | P@10 | R@10 | 相对 baseline |
| --- | --- | --- | --- | --- |
| baseline（裸 CLIP 余弦） | 0.329 | 0.538 | 0.180 | — |
| + prompt 集成（6 个模板取平均） | 0.326 | 0.538 | 0.175 | **−0.9%** |
| + 查询解析 + 属性校验重排 | **0.397** | **0.588** | 0.187 | **+20.4%** |

rerank 的分组（mAP@50 / P@10）：

| 组 | baseline | rerank（P0） |
| --- | --- | --- |
| color（20 条） | 0.249 / 0.550 | 0.315 / **0.625** |
| type（18 条） | 0.394 / 0.583 | 0.502 / 0.672 |
| combo（22 条） | 0.349 / 0.491 | 0.384 / **0.486** |

四个细节要如实写进报告：**combo 组的头部仍略降**——P@10 从 0.491 掉到 0.486，重排把相关图从第 11 名后提上来，也挤掉了少量原本在前 10 的图；**type 组 mAP 提升最大**（+0.109），因为鞋型在属性表里最容易零样本判对；**color 组头部提升明显**（P@10 +0.075），来自 5.5 的解析器对齐。mAP 的涨幅还有一部分来自候选池本身变深（见 5.4），不是纯粹的"打分更准"，必须分开写。

五个 case 的上下对照图由 `renderFailureCases.py` 生成（`evaluation/montages/failureCases.png`，不入 git），覆盖四种失败和一组对照：

| case | 现象 | 归因 |
| --- | --- | --- |
| sports sandals | AP 0.00 → 0.01，返回的全是 sandals | taxonomy 与视觉不可分，标签问题 |
| grey shoes | AP 0.24 → 0.20，银鞋仍被顶上来 | 银/灰在 CLIP 文本空间太近，色族权重只压到 0.5 |
| beige heels | AP 0.14 → 0.08，裸色/金色高跟混入 | tan/khaki 色族与高跟叠加，头部 P@10 0.4 → 0.3 |
| bronze heels | AP 0.20 → 0.22，棕色带光泽仍排前 | 金属色依赖高光，缩略图看不清 |
| casual shoes（对照） | AP 0.50 → 0.92 | 鞋型特征强，重排能救回来 |

**prompt 集成的负结果值得单写一段**：这是文献里最常被当作"免费提升"的技巧，在这套数据上不成立。单模板实测 P@10（60 条 auto，按组拆）：

| 模板 | color | type | combo |
| --- | --- | --- | --- |
| `{q}`（裸 query） | 0.550 | 0.583 | 0.491 |
| `a pair of {q}` | 0.535 | 0.622 | 0.468 |
| `a photo of {q}` | 0.565 | 0.561 | 0.432 |
| `a product photo of {q}` | 0.540 | 0.550 | 0.423 |
| `a close-up photo of {q}` | 0.545 | 0.550 | 0.418 |
| `a {q} on a white background` | 0.425 | 0.428 | 0.332 |

规律很清楚：**query 里要同时满足的属性越多，多加的词就越伤**。单属性（color、type）加词基本无损甚至小涨（`a pair of` 让 type +0.04），双属性（combo）从 0.491 一路掉到 0.332。机制在文本侧：CLIP 把整句池化成一个向量，模板词稀释了颜色/类型词的相对权重，两个属性同时存在时就开始互相让位。

"背景占画面比例大导致看背景"这个直觉假设被实验否掉了：把每张图裁掉边缘背景（各切 1/6 宽、1/8 高）后重算嵌入，`a {q} on a white background` 的损失只从 −0.125 收窄到 −0.100，几乎没变——**问题不在图像侧的背景，在文本侧的稀释**。**结论：这套数据上不用模板，直接用 query。**

rerank 的做法（`retrieval/rerank.py`）：先按余弦取 **top-100** 候选，再用零样本属性表给每张图算"颜色/类型/性别"三类概率，`score = cos + β·Π(加权 P)`，按新分数重排、返回前 50。权重来自解析器：query 的字面值记 1.0，颜色族里的其他值记 0.5（鞋型同义词仍记 1.0）。属性表是 **CLIP 文本编码出来的，不读 metadata**，所以不与评测标签共享信息。候选池深度是一个独立常量 `RERANK_DEPTH`，和指标截断（mAP@50 / P@10）分开，见 5.4。

β 仍是 0.1，P0 **没有**重新标定它：把 β 加到 0.2–1.5 能把 mAP 抬到 0.401–0.404，但 grey/beige 会从 0.202/0.233 掉到 0.191/0.176（高 β 放大了 CLIP 的颜色混淆）。用失败案例换总分不划算，分属性 β 留给 P1。

```bash
python evaluation/runEvaluation.py --method rerank     # 出 results/rerank_auto.json（当前口径）
python evaluation/runEvaluation.py --method rerank --beta 0.2   # 扫描用，输出文件名带 _b0.2
python evaluation/compareResults.py results/baseline_auto.json results/rerank_auto.json
python evaluation/compareResults.py results/rerank_legacy_auto.json results/rerank_auto.json   # P0 前后
```

`results/` 里每个文件都含 `per_query` 明细，`compareResults.py` 直接吃两份文件出 delta 和置信区间，方便挑失败案例、写报告。

### 5.3 统计检验：mAP 与 P@10 都显著

60 条 query 共用同一个图库、同一套构造规则，不能当独立样本做两样本检验；这里用**配对 bootstrap**——重采样 query（保留两方法的配对关系），10,000 次，固定种子 0，取 2.5/97.5 分位。

```bash
python evaluation/compareResults.py results/baseline_auto.json results/rerank_auto.json
```

| 指标 | baseline | rerank | Δ | 95% CI | 胜/负/平 | 结论 |
| --- | --- | --- | --- | --- | --- | --- |
| mAP@50 | 0.3292 | 0.3965 | +0.0672 | **[+0.0481, +0.0887]** | 50 / 10 / 0 | 显著 |
| P@10 | 0.5383 | 0.5883 | +0.0500 | **[+0.0217, +0.0817]** | 26 / 10 / 24 | **显著** |

分组 Δ（mAP@50 / P@10）：color +0.0664 / +0.0750，type +0.1086 / +0.0889，combo +0.0341 / −0.0045。

报告口径照这个写：**"查询解析 + 属性校验重排（检索 100、返回 50）把 mAP@50 相对提升 20.4%（配对 bootstrap 95% CI +0.048～+0.089，50/60 条 query 变好），P@10 提升 0.050（CI +0.022～+0.082，26 条变好、10 条变差、24 条持平）也显著"**。P@10 的显著提升主要来自 color 组（+0.075）；combo 组头部仍略降（−0.005），重排的收益在 combo 上仍偏排序深处。

注意 P0 自身的增量要单独说：相对 P0 之前的 0.386，P0 是 **mAP +0.0104（CI +0.0038～+0.0177，28/12/20，显著）、P@10 +0.0183（CI −0.0017～+0.0417，不显著）**——解析器修复在 mAP 上坐实了，头部还没。

### 5.4 候选池深度：为什么取 100

`RERANK_DEPTH` 和指标截断是两个概念。纯余弦时检索 100 再截前 50，和直接检索 50 完全等价；所以"检索 100、重排、返回 50"与 baseline 是公平对照，不是多看了答案。

| 候选池 | mAP@50 | P@10 | 说明 |
| --- | --- | --- | --- |
| 50（旧口径） | 0.3630 | 0.5867 | 只在准命中的前 50 里换位 |
| **100（现口径）** | **0.3965** | **0.5883** | 拐点：候选池 recall 0.540 → 0.668 |
| 200 | 0.3972 | 0.5883 | 再深无收益，只是多算 |

mAP 的这 0.034 涨幅来自"把第 51–100 名的相关图捞进前 50"，是检索流程的改动，不是打分函数变准；报告里要和属性重排的贡献分开写（属性重排本身在同一深度下是 0.363 → 基线 0.329 的那部分，+0.034）。两项加起来 ≈ 0.067，和总涨幅对得上。

### 5.5 P0：从失败分析回到解析器

5.2 里 grey/beige 的锅不只在 CLIP：`queryParse.py` 把 `grey` 映射成 Grey + Charcoal + **Silver**，把 `beige` 映射成 6 个色值，而 auto 标签是**精确匹配**——重排当然会把银鞋、卡其鞋顶上来。`sports sandals` 同理，`sports` 会额外命中 Sports Shoes。P0 只改解析器，不动模型、图库和 β：

- **长短语屏蔽子串**：`sports sandals` 只解析成 Sports Sandals，不再同时命中 `sports` 和 `sandals`；
- **颜色分「字面值 / 色族」**：字面值 1.0，族内其他色 0.5（灰→银、米→卡其等）；
- **鞋型别名保持 1.0**：`sneakers` = Sports Shoes / Casual Shoes 是真同义词，不是近似色；
- `rerank.py` 按权重加权求和（`Π` 里每项乘权重），其余打分逻辑不变。

| 配置 | mAP@50 | P@10 | grey AP | beige AP | olive AP |
| --- | --- | --- | --- | --- | --- |
| 旧解析器（`rerank_legacy_auto.json`） | 0.3861 | 0.5700 | 0.137 | 0.153 | 0.369 |
| 只做短语屏蔽 | 0.3858 | 0.5700 | 0.137 | 0.153 | 0.369 |
| **+ 色族 0.5（现口径）** | **0.3965** | **0.5883** | 0.202 | 0.233 | 0.359 |
| + 色族 0.25（更激进） | 0.3980 | 0.5900 | 0.222 | 0.255 | 0.292 |

这张表由 `evaluation/ablateParser.py` 一次跑出（同一进程、同一图库、β 与候选深度固定），结果存 `results/parser_ablation.json`：

```bash
python evaluation/ablateParser.py
```

选 0.5 而不是 0.25：总分只差 0.0015（在噪声内），但 0.25 把 olive 从 0.369 打到 0.292、orange 从 0.452 打到 0.401，最差单条回撤 −0.077；0.5 的最差单条只有 −0.030。短语屏蔽单独看是 −0.0003（sports sandals 0.029 → 0.014），它没有修好 taxonomy，只是让解析结果不再自相矛盾——那要等标签或层级分类。

P0 相对旧解析器：**mAP +0.0104（CI +0.0038～+0.0177，28/12/20，显著），P@10 +0.0183（CI −0.0017～+0.0417，不显著）**。它把 grey/beige 从"重排明显变差"拉回"接近 baseline"（0.202 vs 0.236；0.233 vs 0.238），但没有变成正收益；色族 × 鞋型叠加的 combo case（beige heels 0.14 → 0.08）仍然差。β 在 P0 后没有重调，原因见 5.2。

---

## 6. 跨平台注意（macOS / Windows 本地，无需虚拟机）

- 设备统一由 `clipCore.get_device()` 决定：M1 → `mps`，N 卡 → `cuda`，否则 `cpu`。别写死 `"cpu"`。
- **M1 个别算子报错**：`export PYTORCH_ENABLE_MPS_FALLBACK=1`。8GB Air 用 ViT-B/32 + batch 32 足够，别上 ViT-L/14。
- 首次跑联网下载 CLIP 权重（几百 MB），之后走本地缓存；嵌入缓存按 `metadata.csv` 的 mtime 失效。
- 路径全用 `pathlib`，CSV 显式 `encoding="utf-8"`（Windows 默认 cp1252），入口都有 `if __name__ == "__main__":`。
- Windows 上 Gradio 首次启动可能弹防火墙授权，允许本地访问；解压路径别太深。
- **token 不进仓库**：只放 `~/.kaggle/access_token`（`chmod 600`）或环境变量。

---

## 7. 分工建议（多人协作）

- **一份数据、一份 manifest、一份评测集**：只在 `main` 上改 `dataset/manifest.csv`、`evaluation/queries.json`、`evaluation/judgements.json`，其他人只 `git pull`。
- 只有一台机器跑 `--resample` 和补精标；其他人跑复现模式，保证大家算 mAP 用的是同一批图和同一批 query。
- 录 demo 的人只跑两行：`dataset/buildGallery.py` → `retrieval/searchApp.py`，不需要 GPU。
- 嵌入缓存不提交（不同设备算出的向量有极小差异），各自本地生成。

---

## 8. 待做（按方案 B 的顺序）

1. ~~补完 14 条 strict 精标~~ 已完成，20/20 并复查过 7 张边界。
2. ~~`retrieval/rerank.py`：查询解析 + 属性校验重排~~ 已完成（`--method rerank`，auto mAP 0.329→0.397）。
3. ~~把 rerank 接到 CLI/Gradio 上~~ 已完成：`searchCli.py --method rerank` 打 cos/attr 两列，Gradio 单选切换。
4. ~~P0：解析器对齐（长短语屏蔽 + 色族权重）~~ 已完成，见 5.5：相对旧解析器 mAP +0.0104（显著）。
5. 报告里的失败案例分析（颜色混淆、类别混淆、缩略图模糊、标签噪声）。
6. 报告骨架：方法 / 评测口径 / 主表 + 分组表 / 负结果 / 统计检验 / 局限。
7. P1：分属性 β（颜色 vs 鞋型）；P2：对比式颜色分（银/灰、tan/olive）。只接受 mAP 的 CI 下界 > 0 且 P@10 不倒退的改动。
6. 7 分钟演示脚本与录屏（含现场切 baseline / rerank 的同一 query 对照）。
