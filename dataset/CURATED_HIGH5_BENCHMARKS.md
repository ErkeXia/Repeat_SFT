# High 5 条实验：epoch 16、32、48、64 的四任务评测

入口 `dataset/run_curated_high5_benchmarks.sh` 复用已有 `evaluate_curated.py` 和统一报告代码，默认读取 `runs/curated-high5-64-s42/train`。只评测 **64 → 48 → 32 → 16**，不包含 epoch 0。四个检查点都必须已完整保存。

## 在 VM 上执行

本地 push、VM `git pull` 后，在工程根目录执行：

```bash
mkdir -p /data/repeat_sft/logs

nohup bash dataset/run_curated_high5_benchmarks.sh \
  >> /data/repeat_sft/logs/curated-high5-four-benchmarks-v1.log 2>&1 < /dev/null &
echo "评测 PID=$!"
tail -n 30 -f /data/repeat_sft/logs/curated-high5-four-benchmarks-v1.log
```

脚本自动使用现有 `/data/repeat_sft/venvs/vllm-0.11.0/bin/python`，无需手动切换 `.venv`。沿用挂载在 `/data` 的 volume、已有固定评测面板和已经通过 self-test 的 `repeat-sft-lcb:py311-v1` Docker 镜像。当前用户需要能运行 `docker info`。

如果训练目录或面板路径不同，可显式覆盖默认值：

```bash
bash dataset/run_curated_high5_benchmarks.sh \
  --run runs/curated-high5-64-s42/train \
  --panel-dir artifacts/transfer-s2026 \
  --job-name four-benchmarks-e16plus-vllm-v1 \
  --prepare-only
```

`--prepare-only` 检查节点、输入重合与 Docker，冻结配置和代码，不做 GPU 生成。去掉该参数重跑即可开始评测。训练及 teacher-forcing 拟合结果保持原样，新生成和报告保存在本次 job 目录。

## 题目、顺序与配置

| Benchmark | 每个 epoch 的题数 |
|---|---:|
| MATH500 | 128 |
| GPQA | 64 |
| BBH | 100 |
| LiveCodeBench | 100 |

每个 epoch 共 **392** 题，四个节点共 **1,568 次生成**。保持原题、提示词和判分规则：vLLM 并发 **8**，每题一次采样，temperature **0.6**、top-p **0.95**、top-k **20**，生成上限 **8,192 新 token**。

执行次序：epoch 64 的四项任务全部生成、判分并发布该节点报告 → epoch 48 → epoch 32 → epoch 16。内部仍用两个面板各四组：MATH500/GPQA 每组 48 题，BBH/LiveCodeBench 每组 50 题。对外每个 epoch 是一份包含四项任务的统一报告。

## 报告位置与呈现

全部输出位于：

```text
/data/repeat_sft/runs/curated-high5-64-s42/train/postprocess/four-benchmarks-e16plus-vllm-v1/
  epochs/
    epoch-064/report/report.html
    epoch-048/report/report.html
    epoch-032/report/report.html
    epoch-016/report/report.html
  report/
    report.html                  总报告：四张表，每张表一个 benchmark
    summary.json
    curves.csv
    question_milestones.csv       逐题首次答对、退化和恢复的观测节点
  progress.json                  当前节点、面板及已完成节点
  complete.json                  四个节点均完成后的标记
```

每完成一个 epoch，就保存该 epoch 的报告，并更新总报告。单个 epoch 报告是一张四任务指标表，附全部 392 道题的题目、参考答案/判分依据、完整生成回答和判分详情。总报告按 **MATH500、GPQA、BBH、LiveCodeBench 分四张表**，每行一个 epoch，按 **16、32、48、64 升序**展示，方便比较；GPU 执行顺序仍为倒序。

指标仍包括正确数/题数、正确率、平均生成 token 数、EOS 结束率、Chat end 率、长度截断率、答案/代码提取率和判分异常数。每个 epoch 的 `report/` 同时保留 `summary.json`、`curves.csv`、`samples.jsonl` 和来源校验。`samples.jsonl` 含完整题目、回答、生成 token、提取及判分结果；原始生成和判分还分别保存在内部面板的组目录。

未完成的 epoch 明确显示“未完成”，不当作 0%。本次没有 epoch 0，逐题“首次答对”仅指这四个已评测节点内首次观察到正确，不代表训练前一定不会做。

## 暂停、续跑、重建总报告

用 `kill -TERM <评测PID>` 停止，等进程退出后重复最上面的 nohup 命令。已完整保存的逐题回答、已完成的组和 epoch 报告都保留并跳过，未完成请求会重跑；无需加 `--resume`。`tail -f` 中 Ctrl+C 只停止看日志。

仅重建已完成节点的总报告，不启动 GPU/Docker：

```bash
bash dataset/run_curated_high5_benchmarks.sh --report-only
```

每个 epoch 的 HTML 可单独下载阅读；总报告通过相对链接打开各节点详情，若需离线跳转，请保留 `report/` 和 `epochs/` 的目录层次。完整协议及恢复机制见 [CURATED_BENCHMARKS.md](CURATED_BENCHMARKS.md)。
