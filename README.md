# COMP5523 · 鞋子文本检索（Domain-Specific Text-to-Image Retrieval）

课程项目代码仓库。做的是**一个检索系统 + 自建评测 + 消融实验**，不是"微调一个模型"：
骨架是冻结的 CLIP ViT-B/32，我们在它上面叠轻量的领域适配（prompt 模板、查询解析、属性校验重排），
用 mAP / P@10 这类指标量化"每一层适配加了多少分"。完整方案见 [`docs/plan.md`](docs/plan.md)。

当前进度：baseline（冻结 CLIP + 嵌入缓存 + CLI/Gradio）已跑通；60 条评测集与 mAP/P@10/R@10 打分已可用；
查询解析与重排待做。

---

## 0. 目录结构

按"做什么"分三块，而不是按文件类型堆在一起：

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
│   ├── searchCli.py            #    命令行壳：快速看排序和分数
│   └── searchApp.py            #    网页壳：Gradio，演示与录屏
│
├── evaluation/                 # ③ 评测：分数从哪来
│   ├── buildQueries.py         #    生成 60 条 query（确定性）
│   ├── queries.json            #    入 git：query + 自动相关集 + strict 标记
│   ├── judgements.json         #    入 git：人眼精标（qid → {图片 id: 0/1}）
│   ├── renderTopHits.py        #    渲 top-10 拼图，供人眼判
│   └── montages/               #    拼图输出（生成，不入 git）
│
├── results/                    # ④ 历史成绩：<method>_<labels>.json，入 git
├── data/                       # ⑤ 运行产物，全部生成、不入 git
│   ├── raw/                    #    原始 Kaggle 快照（44k 图）
│   ├── gallery/                #    抽出的 1000 张（检索真正用的图库）
│   ├── metadata.csv            #    manifest + 本地路径
│   └── image_features.pt       #    CLIP 嵌入缓存
└── docs/plan.md                # 方案与排期
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
| `retrieval/searchApp.py` | Gradio 网页：输入框 + Top-K + 图片墙 + 排行 | `python retrieval/searchApp.py` |
| `dataset/manifest.csv` | **唯一入 git 的数据**：1000 条样本的 id 与属性，"到底是哪 1000 张" | 脚本读取，不要手改 |
| `dataset/buildGallery.py` | 下载/解压 Kaggle、按 manifest 复制图片、写 `data/metadata.csv`；`--resample` 才重新抽样 | `python dataset/buildGallery.py` / `--resample`（仅维护者） |
| `evaluation/buildQueries.py` | 生成 `queries.json`：60 条 query（色 20 / 类型 18 / 组合 22） | `python evaluation/buildQueries.py` |
| `evaluation/queries.json` | 评测集本体：query 文本、属性、自动相关 id、`strict` 标记 | 由上一行生成 |
| `evaluation/renderTopHits.py` | 把某条 query 的 top-10 渲成带 id 的拼图 | `python evaluation/renderTopHits.py --qid color_black` |
| `evaluation/judgements.json` | 人眼精标：`qid -> {图片 id: 0/1}`，只判 top-10 池 | 看拼图后手填 |
| `evaluation/runEvaluation.py` | 打分：mAP@50 / P@10 / R@10，按 group 拆，导出 `results/<method>_<labels>.json` | `python evaluation/runEvaluation.py --method baseline [--labels strict]` |
| `results/` | 各方法 × 各标签口径的历史成绩 | 由 `runEvaluation.py` 写入，入 git 便于对比 |
| （无 requirements 文件） | 依赖版本直接写在 §2 的安装命令里，实测即锁定 | 照 §2 复制一行安装 |
| `docs/plan.md` | 方案：数据集、属性 schema、指标与消融、架构、排期 | 写报告/分工时对着看 |

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

`evaluation/` 里三样东西分工很明确：`queries.json` 说"问什么、什么算对"，`judgements.json` 说"人眼认为什么算对"，`runEvaluation.py` 只负责算分并把结果写进 `results/`。

两套标签口径，用途不同：

- **auto**（`queries.json` 的 `relevant`）：按 metadata 的 `color`/`articleType`/`gender` 自动构造，60 条全覆盖、零人工、可复现。缺点是**标签就是商品命名里的颜色**，不等于图里看起来的颜色，绝对分数会被系统性压低；适合做**方法之间的相对比较**（同一套标签，谁高谁低可信）。
- **strict**（`judgements.json`）：对 top-10 池逐张肉眼判相关。覆盖面小，但接近"人觉得对不对"，用来报告真实质量并校准 auto 的噪声。

baseline（冻结 CLIP，纯余弦）实测：

| 口径 | 范围 | mAP@50 | P@10 | R@10 |
| --- | --- | --- | --- | --- |
| auto | 60 条 | 0.329 | 0.538 | 0.180 |
| auto | 其中 6 条颜色 query | — | 0.533 | — |
| strict | 同样 6 条颜色 query | — | 0.850 | — |

分组：color mAP 0.249 / type 0.394 / combo 0.349。auto 在 brown 一条上 P@10 只有 0.2，人眼判却是 1.0——因为 Tan/Beige 的鞋肉眼就是棕色，标签却不算。**这个差距是评测方法学的一部分**：auto 是下界，方法间提升仍有效；strict 报真实水平。strict 目前覆盖 6/20 条。

`results/` 里 `baseline_auto.json` / `baseline_strict.json` 都含 `per_query` 明细，方便挑失败案例写报告。

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

1. 补完 14 条 strict 精标（类型 6 条 + 组合 8 条）。
2. `retrieval/rerank.py`：查询解析（类别/颜色/性别）+ 模板集成 + 属性校验重排，注册成 `runEvaluation.py` 的新 method。
3. 消融：baseline → +prompt → +解析与重排，同一份 queries 出表。
4. 报告里的失败案例分析（颜色混淆、类别混淆、缩略图模糊、标签噪声）。
