# 本地数据目录

按数据集分目录；同一数据集再区分原始文件、阅读导出和人工标注。

目前已下载的数据集：

- [Math-CoT-20k](math-cot-20k/README.md)
- [直接阅读：解答最长的 100 条](math-cot-20k/exports/longest-100/README.md)
- [直接阅读：原始顺序前 100 条](math-cot-20k/exports/first-100/README.md)
- [在 VM 上计算 Math-CoT-20k 全集 Base 似然并生成阅读报告](LIKELIHOOD.md)
- [高/低似然各 100 条：一次启动两组 16 轮 LoRA 训练](LIKELIHOOD_TRAINING.md)
- [High/Low 四个 benchmark：一次启动 vLLM 评测、判分与报告](LIKELIHOOD_BENCHMARKS.md)
- [人工改写 10 条：Base 似然度后，从 Base 训练 128 轮 LoRA](CURATED_TRAINING.md)
- [从完整检查点续训：128 → 256 轮，保留原实验并支持中断恢复](CONTINUE_TRAINING.md)
- [改写数据检查点：按 epoch 倒序做四任务评测及统一报告](CURATED_BENCHMARKS.md)

## 目录约定

    dataset/
      math-cot-20k/
        README.md
        raw/
          Math-CoT-20k.parquet
        exports/
          longest-100/
            README.md          阅读目录
            annotations.csv    人工挑选、标签、备注
            selection.json     排序规则、原始行号、校验值
            samples/           每条完整题目和回答，一个 Markdown 文件
          first-100/           原始顺序前 100 条，包含相同的目录、逐题文件和标注表

其他数据集将来分别使用 math500、gpqa、bbh、livecodebench 等同级目录，采用相同的 raw、exports 结构；只有实际有数据时才创建目录。完整训练准备数据若下载到本地，放入对应数据集的 prepared/版本目录。

本次仅整理已经下载的本地原始数据及阅读副本。实验产生的 artifacts 和 runs 仍是实验快照与结果目录，训练、恢复和评测继续使用各自原来的路径。

raw 目录的原始文件不纳入 Git；目录说明和较小的阅读、标注文件可按需提交。
