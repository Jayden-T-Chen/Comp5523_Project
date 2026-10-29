# COMP5523 · 鞋子文本检索（Domain-Specific Text-to-Image Retrieval）

用冻结的 CLIP ViT-B/32 做单域图文检索：输入一句购物 query，返回图库中最相关的鞋图。
在 1000 张 480×640 图库、60 条 query 上，属性校验重排把 mAP@50 从 0.348 提到 0.408
（+17.2%，配对 bootstrap 95% CI +0.041～+0.082），P@10 从 0.582 提到 0.628
（+0.047，CI +0.015～+0.080），两项都显著。方法、评测口径、消融、失败分析，以及
演示与报告的全部文字见 [`docs/projectReport.md`](docs/projectReport.md)。

---

## 1. 快速开始

环境已装好时：

```bash
source .venv/bin/activate                                  # Windows: .venv\Scripts\Activate.ps1

python retrieval/searchCli.py --top 5                      # 命令行看排序
python retrieval/searchWeb.py                              # 网页 http://127.0.0.1:7860
python evaluation/tools/runEvaluation.py --method rerank   # 打分，写 results/rerank_auto.json
```

换机器首次安装（Python 3.11/3.12）：

```bash
git clone https://github.com/Jayden-T-Chen/Comp5523_Project.git comp5523project && cd comp5523project
python3.12 -m venv .venv && source .venv/bin/activate      # Windows: py -3.12 -m venv .venv && .venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install torch==2.14.1 torchvision==0.29.1 open-clip-torch==3.3.0 \
                     gradio==6.31.0 Pillow==12.3.0 numpy==2.5.4
python retrieval/searchCli.py --query "black leather shoes" --top 8
```

仓库是 private：只有被加为 collaborator 的 GitHub 账号能 clone，其他访问返回 404。
加人在网页 Settings → Collaborators and teams，或者：

```bash
gh api -X PUT repos/Jayden-T-Chen/Comp5523_Project/collaborators/<github-用户名> \
   -f permission=pull          # pull=只读，push=可提交
```

图库与 metadata 随仓库分发；首次运行会联网拉 CLIP 权重（几百 MB），之后走本地缓存。
M1 上 1000 张图编码约 15 秒，之后起服务只加载缓存。
首次运行出现 `QuickGELU mismatch ... pretrained tag 'openai'` 的 UserWarning 是 open_clip
加载 openai 权重的已知提示，可以忽略。

| 想做什么 | 命令 |
| --- | --- |
| 单条 query 看排序与分数 | `python retrieval/searchCli.py --query "silver sports shoes" --method rerank` |
| 演示 / 录屏 | `python retrieval/searchWeb.py`（页面内切 baseline / rerank） |
| 主评测（60 条） | `python evaluation/tools/runEvaluation.py --method rerank` |
| 人眼精标（20 条） | `python evaluation/tools/runEvaluation.py --method baseline --labels strict` |
| 比较两组结果 | `python evaluation/tools/compareResults.py results/baseline_auto.json results/rerank_auto.json` |
| 解析器消融 | `python evaluation/tools/ablateParser.py` |
| 候选池 / β 扫描 | `python evaluation/tools/ablateFusion.py` |
| 失败案例图 | `python evaluation/tools/renderFailureCases.py` |

### 1.1 推荐演示 query

AP / P@10 取自 `results/baseline_auto.json` 与 `results/rerank_auto.json`（auto 标签，
β=0.1，候选深度 100），每条都能在仓库里核对。

| query | 演示用途 | baseline AP / P@10 | rerank AP / P@10 |
| --- | --- | --- | --- |
| `casual shoes` | 切换效果最直接 | 0.450 / 0.70 | **0.867 / 0.90** |
| `silver sports shoes` | 颜色区分（银色顶到第一位） | 0.254 / 0.50 | 0.317 / 0.50 |
| `beige casual shoes` | 视觉干净的备用项 | 0.394 / 0.60 | 0.386 / 0.50 |
| `black casual shoes` | 主动讲的局限（正式鞋仍在头部） | 0.153 / 0.30 | 0.215 / 0.40 |
| `sports sandals` | 已诊断的失败（拖鞋子类分不开） | 0.003 / 0.00 | 0.034 / 0.10 |
| `bronze heels` | 金属色备用项 | 0.235 / 0.40 | 0.238 / 0.40 |

讲法与逐条解说见 [`docs/projectReport.md`](docs/projectReport.md) §10.4。

---

## 2. 目录结构

按职责分四层：数据定义 / 检索链路 / 评测 / 文档。

```
comp5523project/
├── README.md                   # 本文：结构导读、启动、数据、方法与结果摘要
├── docs/
│   └── projectReport.md        # 报告 + 演示的全部文字底稿（英文），含课程要求对照
│
├── dataset/                    # ① 数据定义：这 1000 张怎么来的
│   ├── manifest.csv            #    1000 个 id + 属性，抽样的审计记录
│   └── NOTICE                  #    来源、许可、抽样与缩放规则
│
├── data/                       # ② 检索的输入
│   ├── gallery/                #    入 git：1000 张 480×640 JPEG（22 MiB）
│   ├── metadata.csv            #    入 git：id + 相对路径 + 颜色 / 鞋型 / 性别等
│   └── image_features.pt       #    本机嵌入缓存（不入 git，按图库签名失效）
│
├── retrieval/                  # ③ 检索主链路：query 进，排序出
│   ├── clipCore.py             #    设备选择 / 模型加载 / 图像嵌入与缓存 / 余弦 top-k
│   ├── queryParse.py           #    query → {color, type, gender, weights}
│   ├── rerank.py               #    零样本属性表 + cos + β·Π(加权 P) 重排
│   ├── searchCli.py            #    命令行壳
│   └── searchWeb.py            #    Gradio 壳（演示 / 录屏）
│
├── evaluation/                 # ④ 评测：分数从哪来
│   ├── sets/                   #    冻结的评测定义（入 git）
│   │   ├── queries.json            #      60 条 query + auto 相关集 + strict 标记
│   │   ├── queries_family.json     #      色族口径
│   │   ├── queries_sandal.json     #      Sandal 合并口径
│   │   ├── queries_heldout.json    #      30 条冻结留出集（20 改写 + 10 拼错）
│   │   └── judgements.json         #      人眼精标（当前 baseline top-10，20×10 格）
│   ├── tools/                  #    五个入口脚本
│   │   ├── runEvaluation.py        #      mAP@50 / P@10 / R@10 → results/
│   │   ├── compareResults.py       #      配对 bootstrap（Δ + 95% CI）
│   │   ├── ablateParser.py         #      解析器四配置消融
│   │   ├── ablateFusion.py         #      候选池 × β 扫描
│   │   └── renderFailureCases.py   #      baseline / rerank 上下对照图
│   └── montages/               #    生成的图（不入 git）
│
├── results/                    # ⑤ 结果存档：<method>_<labels>.json + 两份消融，入 git
└── .venv/                      # 本地虚拟环境（不入 git）
```

三条边界：

- `dataset/` 记录**抽样定义**；`data/gallery/` 与 `data/metadata.csv` 是**随仓库分发的输入**；`image_features.pt` 是本机缓存。
- `retrieval/` 只管“给定图库和 query 怎么排序”，不依赖评测集。
- `evaluation/` 只管“怎么判对错、怎么算分”，通过 import `retrieval/clipCore.py` 复用检索能力。

模块名用 camelCase（`clipCore.py`、`runEvaluation.py`）。

---

## 3. 数据

- 来源：Kaggle *Fashion Product Images* 全量版，`masterCategory == Footwear` 共 9,222 行；男女各抽 500 张，性别内按 `baseColour` 轮转，主色各约 50 张，稀有长尾色保留作干扰项。
- 图库：原图 1080×1440（少数 1800×2400），Lanczos 缩到 **480×640 / JPEG q90**，共 22 MiB；480×640 是 CLIP 输入（224）的两倍多，金属高光与鞋型细节都保留。
- 属性：8 个 `articleType`（Casual 283 / Heels 237 / Sports 162 / Flip Flops 126 / Flats 101 / Sandals 60 / Formal 21 / Sports Sandals 10）、40 个颜色值、性别 500/500。
- 审计：`dataset/manifest.csv` 是“为什么是这 1000 张”的依据；来源、许可与缩放规则见 `dataset/NOTICE`。
- 缓存：`data/image_features.pt` 以 `(图片数, 最新 mtime, 总字节)` 为签名，图库一有变化就重算，缓存始终与图库一致。

---

## 4. 方法

两条线共用同一图库、同一图像特征、同一截断，只差最后一步打分。

**baseline**：冻结 CLIP ViT-B/32，文本与图像各得 512 维、L2 归一化向量，余弦 = 点积，取前 50。

**rerank**：余弦先取 100 个候选，再用零样本属性表给每张候选算颜色 / 鞋型 / 性别概率，按

```
score = cos(query, image) + β · Π_a ( Σ_{v∈a} w_v · P_a(v | image) )
```

重排并返回前 50。`β = 0.1`；连乘表达 AND 语义（两个属性要同时满足）；`w_v` 由查询解析给出：

| 情况 | 权重 | 例子 |
| --- | --- | --- |
| 字面值 | 1.0 | `grey` → Grey |
| 色族邻居 | 0.5 | `grey` → Charcoal 0.5、Silver 0.5 |
| 鞋型同义词 | 1.0 | `sneakers` → Sports Shoes、Casual Shoes |
| 长短语 | 屏蔽子串 | `sports sandals` 不再命中 `sports` / `sandals` |

关键常量集中在 `retrieval/rerank.py` 与 `queryParse.py`：`BETA = 0.1`、`RERANK_DEPTH = 100`、
`COLOR_FAMILY_WEIGHT = 0.5`、`TYPE_FAMILY_WEIGHT = 1.0`、`MASK_SHORT_MATCHES = True`。
权重与候选深度的取值依据都在 `results/`：色族 0.25 与 β 0.2–1.5 的对照分别见
`parser_ablation.json`、`fusion_ablation.json`，说明见
[`docs/projectReport.md`](docs/projectReport.md) 的 §4 与 §6。

---

## 5. 评测与结果

五套冻结定义，用途不同：

| 文件 | 内容 | 用途 |
| --- | --- | --- |
| `queries.json` | 60 条（颜色 20 / 鞋型 18 / 组合 22），metadata 精确匹配 | **主口径**，方法比较都用它 |
| `judgements.json` | 当前 baseline top-10 的 20×10 格人眼 0/1 | 校准 auto 的绝对水平 |
| `queries_family.json` | 同样 60 条，按人眼色族放宽颜色 | 标签敏感性 |
| `queries_sandal.json` | 同样 60 条，Sandals 与 Sports Sandals 合并 | taxonomy 敏感性 |
| `queries_heldout.json` | 另 30 条（20 改写 + 10 拼错），冻结 | 改动前后复核，不参与调参 |

指标 mAP@50（主）、P@10（头部）、R@10（召回，受相关集大小限制）；显著性用配对 bootstrap
（按 query 重采样 10,000 次，种子 0，2.5/97.5 分位）。`results/` 里每个方法 × 标签的结果都含
逐 query 明细（`parser_ablation.json` 只存四组配置的汇总）。

| 方法 | mAP@50 | P@10 | R@10 | 相对 baseline |
| --- | --- | --- | --- | --- |
| baseline（冻结 CLIP 余弦） | 0.3481 | 0.5817 | 0.1896 | — |
| + prompt 集成（6 模板，负结果） | 0.3302 | 0.5633 | 0.1855 | −5.1% |
| + 查询解析 + 属性重排 | **0.4078** | **0.6283** | **0.2024** | **+17.2%** |

mAP Δ+0.0597（95% CI +0.0413～+0.0816，49 胜 / 11 负），P@10 Δ+0.0467（CI +0.0150～+0.0800）。
分组看，颜色组头部 +0.075、鞋型组 mAP +0.095，组合组头部只有 +0.009；人眼 strict 下
baseline P@10 0.925，同 20 条 auto 0.625，差的是标签口径。完整表格、失败案例与局限见
[`docs/projectReport.md`](docs/projectReport.md) §5–§7。

---

## 6. 复现主要结果

```bash
# 主口径
python evaluation/tools/runEvaluation.py --method baseline                    # baseline_auto.json
python evaluation/tools/runEvaluation.py --method prompt                      # prompt_auto.json
python evaluation/tools/runEvaluation.py --method rerank                      # rerank_auto.json
python evaluation/tools/runEvaluation.py --method baseline --labels strict    # baseline_strict.json

# 两套放宽标签
python evaluation/tools/runEvaluation.py --method baseline --queries evaluation/sets/queries_family.json
python evaluation/tools/runEvaluation.py --method rerank   --queries evaluation/sets/queries_family.json
python evaluation/tools/runEvaluation.py --method baseline --queries evaluation/sets/queries_sandal.json
python evaluation/tools/runEvaluation.py --method rerank   --queries evaluation/sets/queries_sandal.json

# 留出集与比较（--by kind 拆改写 / 拼错）
python evaluation/tools/runEvaluation.py --method baseline --queries evaluation/sets/queries_heldout.json
python evaluation/tools/runEvaluation.py --method rerank   --queries evaluation/sets/queries_heldout.json
python evaluation/tools/compareResults.py results/baseline_auto_heldout.json results/rerank_auto_heldout.json --by kind

# 消融与配图
python evaluation/tools/ablateParser.py                                       # parser_ablation.json
python evaluation/tools/ablateFusion.py                                       # fusion_ablation.json
python evaluation/tools/renderFailureCases.py                                 # montages/failureCases.png
```

---

## 7. 跨平台注意

- 设备由 `clipCore.get_device()` 决定：M1 → `mps`，N 卡 → `cuda`，否则 `cpu`；不要写死 `"cpu"`。
- M1 个别算子报错时 `export PYTORCH_ENABLE_MPS_FALLBACK=1`；8 GB Air 用 ViT-B/32 + batch 32 足够。
- 路径全部 `pathlib`，CSV 显式 `encoding="utf-8"`（Windows 默认 cp1252），入口都有 `if __name__ == "__main__":`。
- Windows 首次启动 Gradio 可能弹防火墙授权，允许本地访问即可。

---

## 8. 文档与状态

- [`docs/projectReport.md`](docs/projectReport.md)：课程要求逐条对照、权重与算法设计、评测协议、全部结果与失败、评分维度对应、7 分钟演示文字、8 页报告排版清单与参考文献。
- 已完成：数据与图库、baseline、查询解析、属性重排、CLI/Gradio、五套评测定义、人眼精标、消融与统计检验。
- 待完成：现场演示与录屏（2026-11-18）、8 页报告（2026-11-25）、AI 使用声明与成员分工填写、peer review。
- 数据来自 Kaggle *Fashion Product Images*（MIT），仅用于课程作业，详见 `dataset/NOTICE`。
