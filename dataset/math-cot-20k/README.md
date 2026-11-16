# Math-CoT-20k

来源：jasonrqh/Math-CoT-20k。原始文件 20,480 条，包含问题、完整推理回答、参考答案和来源信息。

据[论文的数据构造说明](https://arxiv.org/html/2604.06628v1)，题目从 OpenR1-Math-220k 的 default 子集抽取；长解答由 **Qwen3-32B（thinking 开启）**生成，温度 0.6、top-p 0.95、top-k 20、min-p 0，上限 16,384 token。每题生成多条，用 math-verify 检查最终答案，保留一条正确解答；多条通过时随机选一条。这是最终答案校验，不等于逐步推理均经人工验证。

本地冻结 revision 为 `1435fb21d4fecc8ad4966a26f22a874cf2b527f1`，Parquet SHA256 为 `3166f342ec838ebe0979be43923abb33fa1c80f33043aa6acc04e5d82161142a`。`data_source` 实际分布：

| 来源字段 | 条数 |
| --- | ---: |
| olympiads | 14,238 |
| cn_contest | 3,720 |
| aops_forum | 1,851 |
| amc_aime | 357 |
| inequalities | 180 |
| olympiads_ref | 77 |
| number_theory | 57 |

发布文件的字段为 `data_source, question, answer, message, response, response_length, advantage`，**没有独立的官方题号/原始题目 URL**。本项目使用从 0 开始的原始 `row_index`、规范化题目哈希 `question_id` 和题目+解答哈希 `sample_id`；这些是本地生成的标识，不能当作上游竞赛题号。行号必须和数据 revision 配套使用。`response_length` 是发布者记录的解答 token 数，未必等于本项目按聊天模板计算的实际监督 token 数。

本地核对：20,480 行原始题目字符串互不相同；按项目现有规则进行 Unicode NFKC 规范化并移除空白后，有 20,461 个不同 `question_id`，而 `sample_id` 有 20,480 个。全集似然任务以原始行号逐条处理，保留这些规范化后相同的题目及各自解答，不因题目哈希重复而漏算。

- [原始数据](raw/Math-CoT-20k.parquet)：下载时的原始 Parquet，未修改内容。
- [解答最长的 100 条：阅读入口](exports/longest-100/README.md)。
- [原始顺序前 100 条：阅读入口](exports/first-100/README.md)。
- [前 100 条人工标注清单](exports/first-100/annotations.csv)。
- [最长 100 条人工标注清单](exports/longest-100/annotations.csv)。
- [选取规则、原始行号及校验值](exports/longest-100/selection.json)。
- [全集 Base 解答似然：运行、恢复和报告说明](../LIKELIHOOD.md)。

阅读导出按选择方式分别放在 exports/longest-100 和 exports/first-100；后续其他筛选使用新的目录。训练准备后的候选池、子集和划分若另行下载，可以放在 prepared 下，分别保留版本与 manifest；不要用原始文件替换已冻结的训练数据。
