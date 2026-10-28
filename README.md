# COMP5523 · 鞋子文本检索（Domain-Specific Text-to-Image Retrieval）

课程项目代码仓库。做的是**一个检索系统 + 自建评测 + 消融实验**，不是"微调一个模型"：
骨架是冻结的 CLIP ViT-B/32，我们在它上面叠轻量的领域适配（prompt 模板、查询解析、属性校验重排），
用 mAP / P@10 这类指标量化"每一层适配加了多少分"。完整方案见 [`docs/plan.md`](docs/plan.md)。

当前进度：baseline（冻结 CLIP + 图库嵌入缓存 + CLI/Gradio demo）已可跑；评测与重排待做。

---

## 1. 克隆后 5 步跑起来

```bash
git clone <这个仓库的地址> comp5523project
cd comp5523project

# 1) 虚拟环境（Python 3.11/3.12）
python3.12 -m venv .venv && source .venv/bin/activate      # Windows: py -3.12 -m venv .venv && .venv\Scripts\Activate.ps1
python -m pip install -U pip -r requirements.txt

# 2) Kaggle 凭据（只需一次）
mkdir -p ~/.kaggle && echo 'KGAT_你的token' > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token
#   Windows: 把 access_token 放到 C:\Users\<你>\.kaggle\access_token
#   也可以临时用环境变量 KAGGLE_API_TOKEN=KGAT_...

# 3) 取数据：自动下载 Kaggle 数据集 → 按 manifest 复制出那 1000 张 → 写 metadata.csv
python scripts/download_footwear.py

# 4) CLI 验证检索（首次会嵌入 1000 张并存缓存；M1 实测约 27 秒，含加载模型）
python demo_clip_rank.py --query "black leather shoes" --top 8

# 5) 网页 demo（本机浏览器打开 http://127.0.0.1:7860）
python demo_ui.py
```

> 已在 MacBook Air M1（macOS 14.8 / Python 3.12 / torch 2.14.1）实测：`device: mps`，1000 张嵌入约 27 秒，CLI 排序正常，Gradio 页面 `127.0.0.1:7860` 返回 200。
> 首次运行会看到 `QuickGELU mismatch ... pretrained tag 'openai'` 的 UserWarning，这是 open_clip 加载 openai 权重时的已知提示，可忽略。

---

## 2. 文件说明

| 文件 / 目录 | 作用 | 怎么用 |
| --- | --- | --- |
| `clip_core.py` | 全项目共用的检索内核：读 `data/metadata.csv`、选设备（mps/cuda/cpu）、加载 CLIP ViT-B/32（`openai` 权重）、批量编码图片并缓存到 `data/image_features.pt`、文本编码与 top-k 排序 | 被其它脚本 import，不直接运行 |
| `demo_clip_rank.py` | 命令行版检索，用来验证数据与模型是否正常、看分数 | `python demo_clip_rank.py --query "black leather shoes" --top 8`（`--query` 可重复） |
| `demo_ui.py` | Gradio Blocks 网页：输入框 + Top-K 滑条 + 图片墙 + 排行文本 | `python demo_ui.py`，默认 `127.0.0.1:7860` |
| `scripts/download_footwear.py` | 数据脚本：下载/解压 Kaggle 数据集、按 `data/manifest.csv` 复制图片、生成 `data/metadata.csv`；`--resample` 才重新抽样 | `python scripts/download_footwear.py`（复现）/ `--resample`（重建划分，仅维护者） |
| `data/manifest.csv` | **唯一提交进 git 的数据文件**：1000 条样本的 id 与属性，锁定"到底是哪 1000 张" | 由脚本读取；不要手工改 |
| `data/`（其余） | `raw/`（原始数据集）、`gallery/`（抽出的 1000 张）、`metadata.csv`（含图片路径）、`image_features.pt`（嵌入缓存） | 全部由脚本生成，已在 `.gitignore` |
| `requirements.txt` | 运行依赖（torch / open-clip-torch / gradio / kaggle …），写的是版本区间 | `pip install -r requirements.txt` |
| `requirements.lock.txt` | 本机实测通过的**精确版本**（torch 2.14.1 / open-clip-torch 3.3.0 / gradio 6.31.0，Python 3.12） | 想完全复现就用 `pip install -r requirements.lock.txt`；换环境后重装建议用这一份 |
| `docs/plan.md` | 项目方案：数据集、属性 schema、指标与消融、代码架构、10 天排期 | 写报告/分工时对着看 |
| `eval/build_queries.py` | 生成 `eval/queries.json`：60 条 query（色 20 / 类型 18 / 组合 22），按属性自动标相关集，完全确定性 | `python eval/build_queries.py` |
| `eval/queries.json` | 评测集本体（入 git）：query 文本、属性、相关 id 列表、`strict` 标记 | 由上一行生成，一般不改 |
| `eval/eval.py` | 打分：mAP@50 / P@10 / R@10，按 group 拆分，导出 `results/<method>_<labels>.json` | `python eval/eval.py --method baseline [--labels strict]` |
| `eval/make_montage.py` | 把某条 query 的 top-10 渲染成带 id 的拼图，供人工判相关 | `python eval/make_montage.py --qid color_black` |
| `eval/judgements.json` | 人工精标（入 git）：`qid -> {图片 id: 0/1}`，只判 top-10 池 | 看拼图后手填 |
| `results/` | 各方法 × 各标签口径的评测结果（入 git，方便横向对比和历史） | 由 `eval.py` 写入 |
| `.gitignore` | 排除 `data/`（除 manifest）、虚拟环境、嵌入缓存、Kaggle token | — |

---

## 3. 数据：只传脚本 + manifest，不传图片

**要传 `scripts/download_footwear.py` 吗？** 要。它是纯代码、不含任何密钥，别人 `git pull` 后一条命令就能拿到数据。
1000 张图（17MB）和 `data/raw`（632MB）都不进仓库——体积大、且完全可由脚本重建。

**能保证每次下的都是同一批 1000 张吗？** 能，靠三层保险：

1. **数据集是固定快照**：Kaggle `paramaggarwal/fashion-product-images-small` 的 id `NNNNN.jpg` 与 `styles.csv` 内容稳定，同一个 id 永远对应同一张图。
2. **id 清单提交在仓库里**：`data/manifest.csv` 存的就是这 1000 个 id（及属性）。脚本默认走**复现模式**——不重新抽样，只按 manifest 复制图片。所以抽样的随机性完全被绕开了（`--seed 42` 只是重建划分时才用到）。
3. **逐条校验**：复制时拿 manifest 的属性跟 `styles.csv` 对一遍，并且校验 1000 张 id 是否都找到。任何不一致会打印 WARNING/ERROR 并以非 0 退出，而不是悄悄给你另一批图。

已实测：连续两次运行，`data/gallery` 的文件名集合 md5 完全一致（`c809ca70…`）；用 `--seed 42` 重新抽样，生成的 manifest 与原文件逐字节相同。

> 唯一的残余风险是 Kaggle 上游把数据集**重新上传**了。这时第 3 层会报错，请停下来对比，而不是继续跑——重新生成划分只会让大家的评测结果不可比。

常用命令：

```bash
python scripts/download_footwear.py                                  # 复现既定 1000 张（默认，日常用这个）
python scripts/download_footwear.py --resample --target 1000 --seed 42  # 维护者重建划分，会覆盖 manifest.csv
python scripts/download_footwear.py --force                          # 强制重新下载（一般不需要）
```

抽样策略（`--resample` 时生效）：只取 `masterCategory == Footwear`（9222 行）→ **Men/Women 各 500** →
在每个性别内按 `baseColour` 轮转，主色各约 50 张，稀有颜色也能分到几张。类别看 `articleType`（8 类）而不是 `category`（只有 3 类）。
注意 Kaggle **small 版图是 60×80 缩略图**：检索和演示够用，投屏放大偏糊。

`data/raw`（632MB）在 `gallery` 生成后可以删掉省空间；但删了就没法重跑 `--resample` 或做上面的校验，要重新下载。

---

## 4. 跨平台注意（macOS / Windows 本地，无需虚拟机）

- 设备统一由 `clip_core.get_device()` 决定：M1 → `mps`，N 卡 → `cuda`，否则 `cpu`。别写死 `"cpu"`。
- **M1 个别算子报错**：`export PYTORCH_ENABLE_MPS_FALLBACK=1`。8GB Air 用 ViT-B/32 + batch 32 足够，别上 ViT-L/14。
- 首次跑会联网下载 CLIP 权重（几百 MB），之后走本地缓存；嵌入缓存在 `data/image_features.pt`，按 `metadata.csv` 的 mtime 失效。
- 路径全用 `pathlib`，CSV 读写显式 `encoding="utf-8"`（Windows 默认可能是 cp1252），可执行入口都有 `if __name__ == "__main__":`。
- Windows 上 Gradio 首次启动可能弹防火墙授权，允许本地访问即可；数据集解压路径别太深。
- **token 不进仓库**：只放 `~/.kaggle/access_token`（`chmod 600`）或用环境变量。若曾贴到聊天/公开处，用完去 Kaggle 轮换。

---

## 5. 分工建议（多人协作）

- **一份数据、一份 manifest、一份评测集**：只在 `main` 上改 `data/manifest.csv` 与之后的 `eval/queries.json`，其他人只 `git pull`。
- 只有一台机器负责 `--resample`（重建划分）和生成评测集；其他人跑复现模式，保证大家算 mAP 用的是同一批图和同一批 query。
- 录 demo 的人只跑最后两步（`download_footwear.py` → `demo_ui.py`），不需要 GPU。
- 嵌入缓存不要提交（不同机器/设备算出的向量可能有极小数值差异），各自本地生成。

---

## 6. 评测（已可用）

```bash
python eval/build_queries.py                                       # 重建 60 条 query（确定性，不覆盖人工标注）
python eval/eval.py --method baseline                              # auto 标签：60 条全跑
python eval/make_montage.py --qid color_black color_white          # 出拼图 → 人工判 → 写 judgements.json
python eval/eval.py --method baseline --labels strict              # 已判过的 query 上算 P@10
```

两套标签口径，用途不同：

- **auto**（`queries.json` 的 `relevant`）：按 metadata 的 `color`/`articleType`/`gender` 自动构造，60 条全覆盖、零人工、可复现。缺点是**标签本身就是商品命名里的颜色**，不是图里看起来的颜色，绝对分数会被系统性压低；它适合做**不同方法之间的相对比较**（同一套标签，谁高谁低是可信的）。
- **strict**（`judgements.json`）：对 top-10 池逐张肉眼判相关。覆盖面小，但接近"人觉得对不对"，用来报告系统真实质量和校准 auto 标签的噪声。

当前 baseline（冻结 CLIP ViT-B/32，纯余弦）实测：

| 口径 | 范围 | mAP@50 | P@10 | R@10 |
| --- | --- | --- | --- | --- |
| auto | 60 条 | 0.329 | 0.538 | 0.180 |
| auto | 其中 6 条颜色 query | — | 0.533 | — |
| strict | 同样 6 条颜色 query | — | 0.850 | — |

auto 在 color 组只有 0.249 mAP、brown 一条 P@10 仅 0.2，而人眼判是 1.0——因为 Tan/Beige 的鞋肉眼就是棕色，标签却写成别的词。**这条差距是评测方法学的一部分**，报告里要写清楚：auto 分数是下界，方法间的相对提升仍然有效。

`strict` 目前覆盖 6/20 条（6 个颜色），其余 14 条（类型/组合）还需要看拼图逐条判。

---

## 7. 待做（按方案 B 的顺序）

1. 补完 14 条 strict 精标（类型 6 条 + 组合 8 条）。
2. `retrieval/rerank.py`：查询解析（类别/颜色/性别）+ 模板集成 + 属性校验重排，注册成 `eval.py` 的新 method。
3. 消融：baseline → +prompt → +解析与重排，同一份 queries 出表。
4. 报告里的失败案例分析（颜色混淆、类别混淆、缩略图模糊、标签噪声）。
