# 课程要求 → 当前实现对照

依据：`COMP5523 Group Project Specification-2627.docx` 与 `COMP5523 Project Introduction-2627.pptx`（第 3–11 页任务与评分、第 12–17 页演示与报告）。
整理日期：2026-10-28。数字全部来自 `results/*.json`，复合口径见 README §5。

---

## 1. 任务与目的

| 课程原文要求 | 我们的设定 |
| --- | --- |
| Build a text-to-image retrieval system for a domain of your choice | 鞋子（shoes），1,000 张图库，8 个 articleType |
| Input: a natural-language query | 自然语言 query，如 `black heels`、`brown formal shoes`、`grey casual shoes` |
| Output: a ranked list of the most relevant images | 返回排序列表；界面显示图、余弦分、属性一致度 |
| Define relevance and explain how the system measures it | 相关 = `articleType` 一致 + 颜色落在同一色族；问性别时性别也要对上。系统用 CLIP 余弦 + 零样本属性概率度量，不读标签 |
| Explain who will use it and why | 想按外观找鞋、但不知道款号/关键词的用户；电商目录运营按属性筛图 |

这是**检索系统**，不是"训一个模型"：CLIP 全程冻结，领域知识放在查询解析与重排里。

## 2. 表示与检索

| 课程建议组件 | 我们的实现 | 位置 |
| --- | --- | --- |
| Cross-modal representation（如 CLIP） | 冻结 CLIP ViT-B/32（`open_clip`, `pretrained="openai"`），文本/图像各 512 维、L2 归一化 | `retrieval/clipCore.py` |
| Image descriptions（caption/tag 再匹配） | 未采用；直接编码图像像素，比"先写 caption 再匹配"少一层信息损失 | — |
| Similarity search and indexing | 余弦相似度，1,000 张全量扫描；图像嵌入缓存到 `data/image_features.pt`，启动只编码 query | `clipCore.py` |
| Domain adaptation | ① prompt 模板集成（6 个）→ 实测负收益，**不采用**；② 查询解析（属性抽取）替代 query rewriting | `clipCore.py`、`retrieval/queryParse.py` |
| Re-ranking | 零样本属性表（颜色/鞋型/性别）+ 属性连乘一致度，`score = cos + 0.1·Π(加权P)`（颜色族 0.5）；先取 100 再重排返回 50 | `retrieval/rerank.py` |
| Query understanding | 36 个颜色词（映射 40 个 metadata 色值）、27 个鞋型词、9 个性别词的别名表 → `{color, type, gender}`；长短语屏蔽子串 | `queryParse.py` |
| User interface | Gradio：方法切换（baseline / rerank）、解析出的属性、每张图的 cos 与 attr | `retrieval/searchApp.py` |

**向量是什么**：图像向量来自像素（不是 caption），文本向量来自 query 原文（不加模板），两者同一 512 维空间；余弦 = 归一化后的点积。
**baseline 是什么**：冻结 CLIP 的图像/文本向量直接算余弦、取前 50——就是课程第 7 页那个最小系统的"看图版"（那个 demo 匹配的是 Flickr30k 的 caption，模型不看像素）。

**额外做的优化（相对 baseline）**

1. **查询解析**：把 `grey casual shoes` 拆成 `{color: grey, type: casual shoes}`，只用于重排，不改写检索文本；长短语命中时屏蔽子串（`sports sandals` 不再命中 `sports`），颜色族给 0.5 权重。
2. **零样本属性表**：属性词表由 CLIP 文本编码得到，对每张图做 softmax，得到 P(颜色)、P(鞋型)、P(性别)；**不读 metadata**，所以不跟评测标签共享信息。
3. **属性校验重排**：query 要求的属性概率加权连乘（AND 语义），加到余弦上；β=0.1。P0 改了权重尺度后没有重调 β——加到 0.2–1.5 能把 mAP 抬到 0.401–0.404，但 grey/beige 会变差，留给分属性 β。
4. **候选池 100**：余弦先取 100、重排后返回 50；纯余弦时"取 100 截 50"与"直接取 50"等价，所以对照公平。
5. **prompt 模板集成试过但否掉**：6 模板平均 mAP 0.326（−0.9%）；combo 组 P@10 从 0.491 掉到 0.332（最差模板）。裁背景实验（−0.125 → −0.100）证明问题在文本侧稀释，不在图像背景。

## 3. 评测集与协议

| 课程要求 | 我们的做法 |
| --- | --- |
| 自建测试集：queries + ground-truth + 相关标准 | `evaluation/queries.json`：60 条 query（color 20 / type 18 / combo 22），相关集由 metadata 规则在全库生成（auto），相关标准写在 `note` 字段 |
| 人工标注（鼓励） | `evaluation/judgements.json`：20 条 query 的 top-10 池共 200 格，逐张判 0/1；颜色族判定规则见 `rule`；7 张边界复查（3 改判、4 维持）见 `review` |
| ≥1 个合适指标并说明理由 | mAP@50 为主（衡量整个排序，不只看头部）；P@10 看头部可用性；R@10 看召回，但因每条 query 的相关数 10–283 不等，1k 图库上 R@10 上限只有 0.368 |
| 说明评测协议 | 同一图库、同一 query 集、同一截断；auto 用于跨方法比较（标签覆盖全库），strict 只报 baseline 绝对水平（池子是 baseline top-10，不能给新方法排名） |
| 失败案例分析 | README §5.2 五个 case（`evaluation/montages/failureCases.png`）：taxonomy 不可分、银/灰、tan/khaki（beige heels）、金属色高光、以及一组 rerank 生效的对照；§5.5 记录了一次从失败回到代码的 P0 修复 |
| 可选效率评估 | 尚未测（见 §5） |

统计：60 条 query 共用图库，不能当独立样本；用**配对 bootstrap**（按 query 重采样 10,000 次，种子 0，2.5/97.5 分位），同时报胜/负/平。

## 4. 结果（auto 标签，60 条 query）

| 方法 | mAP@50 | P@10 | R@10 | 相对 baseline |
| --- | --- | --- | --- | --- |
| baseline（冻结 CLIP 余弦） | 0.3292 | 0.5383 | 0.1795 | — |
| + prompt 集成（6 模板） | 0.3259 | 0.5383 | 0.1751 | −0.9% |
| + 查询解析 + 属性重排（含 P0） | **0.3965** | **0.5883** | **0.1874** | **+20.4%** |

- mAP Δ +0.0672，95% CI **[+0.0481, +0.0887]**，50 胜 / 10 负 / 0 平 → 显著。
- P@10 Δ +0.0500，CI **[+0.0217, +0.0817]**，26 / 10 / 24 → **显著**；color 组贡献最大（+0.075），combo 组头部仍 −0.005。
- P0 解析器对齐自身（相对旧解析器）：mAP +0.0104（CI +0.0038～+0.0177，显著），P@10 +0.0183（不显著）。
- strict 子集（20 条）baseline P@10 **0.875**（同一批 query 的 auto 口径是 0.565），差距来自 auto 标签噪声。
- 深度消融：50 → 0.3630，100 → 0.3965，200 → 0.3972；+0.0672 里约 +0.034 来自重排打分、+0.034 来自候选池变深。
- 规模探测：图库 1k → 9.2k，相对增益 +7.8% → +3.5%，R@10 掉到 0.038；因此主口径保留 1k。

## 5. 与课程要求的差距（待补）

| 项 | 状态 | 说明 |
| --- | --- | --- |
| 现场演示 | 有 Gradio，未录屏 | 需录 1–2 分钟备用视频；演示机提前装依赖 |
| 效率/延迟 | 未测 | 可选；加一个 60 条 query 的平均耗时即可 |
| 报告 | 未写 | ≤8 页 A4、12pt Times New Roman、单倍行距、1 英寸边距、APA 参考文献 |
| AI 使用声明 | 未写 | 需说明哪些部分由成员完成、哪些由 AI 完成、如何核查 |
| 成员分工 | 未写 | 报告与演示都要 |
| Peer review | 演示后分配 | 不计入 8 页 |
| 微调 | 未做，且不打算做 | 没有领域图文对；冻结 CLIP + 零样本属性表已能隔离"领域适配"的贡献；prompt 集成的负结果本身就是证据 |
| strict 扩标 | 可选 | 若要 strict 给 rerank 排名，需把 rerank top-10 的新图再判一遍（约 200 格） |

## 6. 时间

- 演示：2026-11-18（周三）18:30–21:20，TU101，每组 7 分钟，无 Q&A，Zoom 共享屏幕，**调试时间计入 7 分钟**。
- 报告：2026-11-25（周三）23:59 前。
- 评分：Appropriateness / Soundness / Excitement / Presentation / Writing / Peer Review，各 2.5%。
