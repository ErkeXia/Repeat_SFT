# High / Low 实际训练名单中的人工改写候选

2026-11-19：从你提供的实际 high-100、low-100 原始行号名单中，各选出 5 条。不是从全集报告的两端直接取样。

这是一份供人工修改的候选包。2026-11-19 已按你选择的“轻度统一”完成第二轮清理：10 份可编辑 `.response.txt` 保留原主解法和必要推导，删除重复验算、额外解法与口头犹豫，每条最多保留一次检查。详见 [第二轮修改记录](CLEANUP_V2.md)。随后按你的要求调整最终答案样式：全部 10 条均在 Final Answer 后仅保留一个完整 boxed。其中原来有多个 boxed 的方程组与骑士题，已按你的决定改为数据集短答案。详见 [第四轮修改记录](CLEANUP_V4.md)。

原始数据及逐题审阅原文仍在；[第一轮文本备份](revisions/cleanup-v1/responses.jsonl)、[第二轮文本备份](revisions/cleanup-v2/responses.jsonl) 和 [第三轮文本备份](revisions/cleanup-v3/responses.jsonl) 均保存全部 10 条，方便比较或还原。已提供 [VM 实验入口：Base 似然度 + 10 条 × 128 轮 LoRA](../../../CURATED_TRAINING.md)，本地没有运行模型。

## 怎么使用

1. 点击下面的逐题审阅文件，先看“必须注意”和原文定位。
2. 编辑旁边的 `.response.txt`：正文是第二轮轻度清理版，全部最终答案已完成单 boxed 格式调整；`.md` 与 `originals.jsonl` 中仍保留完整原回答，便于对照。逐题审阅中的删改提示针对原文。
3. 保留 `originals.jsonl`、`selection.json` 和审阅 Markdown，便于追溯原文和对照改动。
4. 通过 [VM 实验入口](../../../CURATED_TRAINING.md) 按原聊天模板重新编码、核对监督范围和长度，并生成独立训练数据目录。原始样本身份只作追溯；入口会为改写内容生成新 sample_id。

## High：五条不同主线

| 原始行号 | 题目 / 审阅 | 来源标签 | 原始 tokens¹ | 修改文本 |
|---|---|---|---:|---|
| 6430 | [三条直线确定三角形，求外接圆](high/row-06430.md) | olympiads | 5,736 | [编辑](high/row-06430.response.txt) |
| 2988 | [二元二次方程组，求全部实数解](high/row-02988.md) | aops_forum | 6,600 | [编辑](high/row-02988.response.txt) |
| 4561 | [含 k、k²、k³ 的二项式系数求和](high/row-04561.md) | cn_contest | 6,124 | [编辑](high/row-04561.response.txt) |
| 14120 | [求 10^2022−9^2022 的末三位](high/row-14120.md) | olympiads | 8,829 | [编辑](high/row-14120.response.txt) |
| 1432 | [sin³α/cosα + cos³α/sinα 的最小值](high/row-01432.md) | olympiads | 6,236 | [编辑](high/row-01432.response.txt) |

本组原始 response_length 平均 6,705.0 tokens；不是清理后的长度。

## Low：五条不同主线

| 原始行号 | 题目 / 审阅 | 来源标签 | 原始 tokens¹ | 修改文本 |
|---|---|---|---:|---|
| 16190 | [10 米梯子滑动时，顶端下降速度](low/row-16190.md) | aops_forum | 6,053 | [编辑](low/row-16190.response.txt) |
| 15054 | [任选五项有相同绝对值，最多有多少不同实数](low/row-15054.md) | aops_forum | 5,079 | [编辑](low/row-15054.response.txt) |
| 17772 | [五个三人队围圆桌，队长坐两名队员之间](low/row-17772.md) | olympiads | 4,615 | [编辑](low/row-17772.response.txt) |
| 506 | [A 说‘至少一人说谎’，判断骑士和说谎者](low/row-00506.md) | olympiads | 2,825 | [编辑](low/row-00506.response.txt) |
| 10037 | [容量两块的锅，三块肉饼如何在 15 分钟煎完](low/row-10037.md) | olympiads | 2,584 | [编辑](low/row-10037.response.txt) |

本组原始 response_length 平均 4,231.2 tokens；不是清理后的长度。

¹ 此列使用原数据 `response_length` 字段，本地没有重新加载 tokenizer。已有似然报告覆盖其中 9 条，其监督 token 数与发布字段一致；row 16190 不在该报告内，似然值留空。

## 本轮保留与修正

- High 的 row 6430、2988、4561 保留多阶段计算，其中方程组的两个分支和全部四组解均保留。
- Low 的 row 16190、506 保留相关变化率和角色分类的完整论证，删除符号争论与重复证明。
- Low 的 row 17772 保留成块计数，删除后来把队长写在端点的错误例子。
- Low 的 row 10037 保留 15 分钟时间表及资源下界，删除“不能取出再放回”的无依据限制。
- Low 的 row 15054 保留上界，将原文后部已有的 ±1、±2、±3、±4 构造移到上界之后，说明如何补满 2019 项。

## 当前样式与尚未调整的部分

当前文本均为 `<think>` 内的主推导、至多一次验证，关闭 think 后仅保留一个 Final Answer。全部 10 条的最终答案只留 `\boxed{...}`，没有外围句子、单位和标点。必要的分类、可达性和最优性证明属于主论证，不当作重复验算删掉。没有强行加入相同的分步标题，也没有统一正文的数学排版。结束符由原聊天模板添加，文本副本中不手工补 `<|im_end|>`。

按你的决定，方程组和骑士题使用原数据集 `answer` 字段：方程组为 `(31.5, 10.5), (20, 22), (28, 7), (16.5, 18.5)`，骑士题为 `A`，各放在一个 boxed 中。骑士题的完整角色说明仍保留在推理正文里。

Low 中逻辑题和调度题本来较短，删循环后长度会明显下降。这里优先满足首条主要论证有效、组内题型不同；没有声称五条都需要很长推理，也未匹配两组的有效推理长度或难度。

## 保留实验解释所需的信息

High/Low 标签指修改前样本所在的历史训练组；改写后在 Base 上的似然可能改变。为研究改写的作用，后续最好对同样这 5 道题同时保留原解答组和改写组；否则题目选择、长度和风格会一起变化。这里只准备候选资料，没有创建或启动训练实验。

## 可追溯资料

- [selection.json](selection.json)：你提供的两个完整 100 条名单、所选 sample_id/question_id、原文校验值、来源和文件映射。
- [originals.jsonl](originals.jsonl)：10 条完整原始数据记录（含原问题、prompt/message、短答案和原回答），用于对照，勿作为改写工作区。
- [review_notes.json](review_notes.json)：选择理由、删改提示和部分排除例子。
- [verification.json](verification.json)：首条主要论证的核对及独立计算记录；不表示全部原文没有错误。
- [cleanup-v1.json](cleanup-v1.json) / [第一轮说明](CLEANUP_V1.md)：仅删尾版本的历史记录。
- [cleanup-v2.json](cleanup-v2.json) / [第二轮说明](CLEANUP_V2.md)：轻度清理阶段的修改、字符数、哈希和核对结果。
- [revisions/cleanup-v1/responses.jsonl](revisions/cleanup-v1/responses.jsonl)：第二轮修改前的全部 10 条文本。
- [cleanup-v3.json](cleanup-v3.json) / [第三轮说明](CLEANUP_V3.md)：单 boxed 格式调整及当时两条待决定题目的历史状态。
- [revisions/cleanup-v2/responses.jsonl](revisions/cleanup-v2/responses.jsonl)：第三轮修改前的全部 10 条文本。
- [cleanup-v4.json](cleanup-v4.json) / [第四轮说明](CLEANUP_V4.md)：两条多 boxed 回答采用数据集短答案，待决定项已解决。
- [revisions/cleanup-v3/responses.jsonl](revisions/cleanup-v3/responses.jsonl)：第四轮修改前的全部 10 条文本。

归属校验以你粘贴的实际训练名单为依据，本地没有读取 VM 的 train.jsonl。原始 Parquet SHA-256 已核对；9 条有报告的样本还逐字核对了题目、回答和 sample_id。未加载模型，未下载数据。
