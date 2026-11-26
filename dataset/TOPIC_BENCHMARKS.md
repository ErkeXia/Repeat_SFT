# 五题型 50 条实验：按指定顺序评测全部 16 轮

入口：`dataset/run_topic_benchmarks.sh`。读取 `runs/topic-balanced-50x16-s42/train`，用 vLLM 评测 epoch **1–16** 的 LoRA 检查点。

执行顺序固定为：

```text
1 → 2 → 3 → 4 → 8 → 12 → 16 → 5 → 6 → 7 → 9 → 10 → 11 → 13 → 14 → 15
```

先完成优先的七个节点，再按训练轮次升序补齐其余九个。每个 epoch 都必须完成四个 benchmark 的全部生成、判分和该节点报告，才进入下一个 epoch。首次准备时冻结完整执行顺序；中断续跑跳过已经完成的节点和逐题结果。

## 在 VM 启动

本地 push、VM `git pull` 后，在工程根目录运行。需要训练的 epoch 1–16 都已完整保存，以及原 `artifacts/transfer-s2026` 面板。使用已有 vLLM 环境、LiveCodeBench Docker 镜像和当前用户的 Docker 权限。

```bash
mountpoint -q /data || { echo '/data 尚未挂载'; exit 1; }
mkdir -p /data/repeat_sft/logs

nohup bash dataset/run_topic_benchmarks.sh \
  >> /data/repeat_sft/logs/topic-balanced-four-benchmarks-v1.log 2>&1 < /dev/null &
echo "评测 PID=$!"
tail -n 30 -f /data/repeat_sft/logs/topic-balanced-four-benchmarks-v1.log
```

脚本自动使用 `/data/repeat_sft/venvs/vllm-0.11.0/bin/python`，不用切换训练 `.venv`。环境位置不同时，通过 `VLLM_PYTHON_BIN` 指定 Python。Docker 镜像仍为 `repeat-sft-lcb:py311-v1`，应已完成构建和 self-test；当前用户须能执行 `docker info`。

如果训练或面板不是默认位置：

```bash
bash dataset/run_topic_benchmarks.sh \
  --run runs/topic-balanced-50x16-s42/train \
  --panel-dir artifacts/transfer-s2026 \
  --job-name four-benchmarks-priority-vllm-v1 \
  --prepare-only
```

`--prepare-only` 检查全部节点、数据和 Docker，冻结评测输入与代码，不启动 GPU 生成。去掉这个参数重跑开始评测。没有准备阶段也可直接用 nohup 一次完成。相对的 `runs/` 和 `artifacts/` 都指向 `/data/repeat_sft`。

## 题目和参数

| Benchmark | 每个 epoch 题数 |
| --- | ---: |
| MATH500 | 128 |
| GPQA | 64 |
| BBH | 100 |
| LiveCodeBench | 100 |

每个 epoch **392 题**，全部 16 个 epoch 共 **6,272 次生成**。固定原面板、提示词、选项顺序和判分协议；检查与 50 道训练题的重合，有重合则停止，不替换题目。

vLLM 并发 **8**，每题一次采样；temperature **0.6**、top-p **0.95**、top-k **20**，上限 **8,192 新 token**。到上限停止并记录长度截断。LoRA 加载、最终答案抽取，以及 LiveCodeBench 的代码抽取和 Docker 执行测试沿用已有实现。

一个 epoch 内，先完成数学/GPQA 面板的四组（每组 32+16 题），再完成 BBH/LCB 面板的四组（每组 25+25 题），然后生成统一的四任务报告。这些内部面板目录用于逐题保存和恢复；对外报告按四个 benchmark 呈现。

## 报告与原始回答

默认输出根目录：

```text
/data/repeat_sft/runs/topic-balanced-50x16-s42/train/postprocess/four-benchmarks-priority-vllm-v1/
  suite/                         冻结顺序、节点身份、代码、配置和输入面板
  epochs/
    epoch-001/
      report/
        report.html              epoch 1 的四任务指标与全部 392 条完整回答
        summary.json
        curves.csv
        samples.jsonl            完整题目、参考答案、回答、token、提取和判分
        manifest.json
      math-gpqa/                 分组原始生成、独立判分与恢复信息
      bbh-livecodebench/
      complete.json              四项任务及该节点报告均完成的标记
    epoch-002/ ... epoch-016/
  report/
    report.html                  总报告：四张表，每张表一个 benchmark
    summary.json
    curves.csv
    question_milestones.csv       每题首次观察答对、退化、恢复的训练节点
  progress.json                  当前 epoch、面板、已完成节点
  complete.json                  全部 16 轮完成后的标记
```

**每个 epoch 的全部题目完成后立即发布该节点报告，同时更新总报告；最后生成完整总报告。** 节点报告包含一张四任务指标表，逐题详情按 MATH500、GPQA、BBH、LiveCodeBench 分四部分。总报告固定分四张表，每张表中按 epoch **1–16 升序**排列，方便比较；报告会另列实际 GPU 执行顺序。未完成节点显示“未完成”，不按 0% 统计。

指标包括正确数/题数、正确率、平均生成 token 数、EOS 结束率、Chat end 率、长度截断率、答案/代码提取率、判分异常数。完整回答和判分依据保存在节点 HTML 和 `samples.jsonl`；逐题 token IDs 和原生成记录也保留在内部组目录。

逐题首次答对/退化按**训练轮次**计算，不按 GPU 执行先后。优先节点完成后的报告只反映当时已测节点，补齐中间节点后会重新汇总；单次采样仍有随机波动。

## 停止、恢复和只刷新报告

停止用 `kill -TERM <评测PID>`；前台运行时也可 Ctrl+C。等进程退出后，重复同一条 nohup 命令即可恢复，**不需要 `--resume`**。`tail -f` 的 Ctrl+C 只退出日志查看。

已完整写入的逐题回答、判分、完成组和节点报告保留；最多 8 个未完成生成请求重新开始。恢复核验文件身份，完成节点跳过。对同一任务的重复启动会被文件锁阻止。任务完成后再执行会核验并退出，不重新生成 16 轮回答。

只根据已完成节点刷新总报告，不启动 GPU/Docker：

```bash
bash dataset/run_topic_benchmarks.sh --report-only
```

代码首次冻结后，git pull 不改变这个 job 的实际推理实现。改变节点、顺序或评测参数时，应使用新的 `--job-name`。底层入口 `evaluate_curated.py` 新增 `--epoch-order listed` 才按参数列出的顺序执行；不传此选项的原实验仍按降序执行。
