# 10 条改写训练：按 epoch 倒序完成四个 benchmark

入口：`dataset/run_curated_benchmarks.sh`。默认评测 `runs/curated-10x128-s42/train` 中全部已配置的节点，顺序为 **128 → 112 → 96 → 80 → 64 → 48 → 32 → 16 → 0**。

每个 epoch 必须完成全部 392 道题的生成和判分、发布该 epoch 的四任务统一报告后，才进入下一个 epoch。所有选择的检查点必须已经存在；不会默默跳过缺失节点。若暂时只完成部分训练，用 `--epochs 64,48,32,16,0` 等明确选择，并使用独立的 `--job-name`。

如需按给出的顺序执行，可在新任务显式添加 `--epoch-order listed --epochs ...`。默认仍为降序；顺序首次冻结后不可在同一 job 中改动。五题型 50 条实验的优先节点及补齐顺序已配置在 [专用入口](TOPIC_BENCHMARKS.md)。

## 在 VM 上启动

`git pull` 后，确认 `/data` 已挂载、原 vLLM 0.11.0 环境可用、LiveCodeBench Docker 镜像 `repeat-sft-lcb:py311-v1` 已构建并通过 self-test，当前用户可执行 `docker info`。

```bash
mkdir -p /data/repeat_sft/logs
nohup bash dataset/run_curated_benchmarks.sh \
  --run runs/curated-10x128-s42/train \
  --panel-dir artifacts/transfer-s2026 \
  --job-name four-benchmarks-epoch-first-v1 \
  >> /data/repeat_sft/logs/curated-four-benchmarks-v1.log 2>&1 < /dev/null &
echo "评测 PID=$!"
tail -n 30 -f /data/repeat_sft/logs/curated-four-benchmarks-v1.log
```

脚本自动使用 `/data/repeat_sft/venvs/vllm-0.11.0/bin/python`。环境路径不同可设置 `VLLM_PYTHON_BIN`。不需要往训练环境安装 vLLM，也不需要重新下载 benchmark。

若不想测 epoch 0，首次启动时显式加：

```bash
--epochs 128,112,96,80,64,48,32,16
```

可先用同一命令去掉 nohup/background、加 `--prepare-only`：检查模型节点、原题、重合情况和 Docker，冻结评测配置与代码，不生成答案。随后去掉该参数继续。准备阶段发现训练题与评测题重合会停止并保存审计记录，不替换评测题目。

## 保持不变的评测协议

| Benchmark | 题数 | 来源 |
|---|---:|---|
| MATH500 | 128 | 此次训练 run 的冻结 panel |
| GPQA | 64 | 同上，选项顺序和提示词不变 |
| BBH | 100 | 现有 `artifacts/transfer-s2026`，原四个逻辑子任务各 25 题 |
| LiveCodeBench | 100 | 同上，保留原题、starter code 和全部公开/隐藏测试 |

vLLM 并发 **8**，每题一次采样，temperature **0.6**、top-p **0.95**、top-k **20**，最多 **8,192 新 token**。LoRA 检查点加载方式、答案抽取及判分沿用已有代码，包括 LiveCodeBench 的 Python 代码块抽取和 Docker 执行；本次没有修改它们。

内部仍使用原来的分组以便保存和恢复：数学/GPQA 四组各 48 题，BBH/LCB 四组各 50 题，但每个子任务目录只包含**一个 epoch**。因此执行时不会出现“一个组先跑完所有 epoch，才做下一个组”的情况。也不会把两个面板作为对外的两个独立报告。

每个 epoch 392 次生成；默认 9 个节点，共 **3,528 次生成**。epoch 0 使用本次训练保存的零增量初始 LoRA，作为 Base 等价节点。

## 报告与数据位置

所有输出位于：

```text
/data/repeat_sft/runs/curated-10x128-s42/train/postprocess/four-benchmarks-epoch-first-v1/
  suite/                              冻结代码、配置、节点哈希、重合检查和转移任务面板
  epochs/
    epoch-128/
      math-gpqa/                      原生成/判分流程的工作目录（仅 epoch 128）
      bbh-livecodebench/              同上；每个目录内部仍有四组
      report/
        report.html                  该 epoch 的统一报告：一张四任务指标表，392 条逐题详情
        summary.json                 四任务准确率、输出行为、判分异常等
        curves.csv                   精确指标值
        samples.jsonl                完整问题、回答、生成 token、提取与判分结果
        manifest.json                报告来源及哈希
      complete.json                  全部四任务和报告完成后才写入
    epoch-112/ ... epoch-000/         相同结构，每个 epoch 独立保留
  report/
    report.html                      总报告：四张表，每张表对应一个 benchmark
    summary.json
    curves.csv
    question_milestones.csv           逐题首次答对、退化、恢复的观测节点
  progress.json                       当前 epoch、面板及已完成的 epoch
  complete.json                       全部所选 epoch 完成后才写入
```

**每完成一个 epoch，就立即产生它的 `report/report.html`，并更新总报告。** 每个 epoch 报告的汇总和完整问题/回答都在同一个 HTML 里，可单独下载阅读。总报告内部固定为 MATH500、GPQA、BBH、LiveCodeBench 四张表，每行一个 epoch，按训练轮次升序便于比较；执行顺序仍是倒序。总报告链接各节点详情，若需离线点击这些链接，请同时下载报告目录结构。

统计列包括：正确数/总题数、正确率、平均生成 tokens、EOS 结束率、Chat end 率、长度截断率、答案/代码提取率、判分异常数。EOS 包含聊天结束符和 Base EOS；Chat end 仅包含聊天结束符。没有把四个任务的分数混成一个总正确率。

总报告仅纳入四任务均完成的 epoch。正在执行的 epoch 即使已有部分结果，也明确显示“未完成”，不会计作 0% 或混用不同题数。原始结果即时保存在相应 `groups/group-*/generation/epoch-*/responses.jsonl`；判分独立保存在 `grades/`，以后可以用新输出目录重新判分，不必重新生成。

逐题节点分析按**训练 epoch 升序**，不按 GPU 实际执行顺序；“首次答对”只指已完成的观测节点，不能确定两个保存节点之间具体哪一轮发生变化。每题一次采样也不能据此断言能力已稳定提升或退化。

## 中断恢复与只看报告

用 `kill -TERM <评测PID>` 停止，等待进程退出，再重复同一条启动命令。前台 Ctrl+C 也可停止任务；`tail -f` 中 Ctrl+C 仅停止查看日志。

逐题生成沿用已有恢复机制：已完整写入的回答保留，尚未完成的请求重新生成。完成组会跳过；一个 epoch 的统一报告已提交后，重跑时会核对并跳过整个 epoch。不会覆盖旧实验的推理或 teacher-forcing 结果。首次启动冻结代码、提示词、生成参数和 Docker 镜像，后续 git pull 不改变该任务的实际执行代码。

已完成的总报告随时可以打开。如果只想从已提交的 epoch 重新汇总，不启动 GPU 或 Docker：

```bash
bash dataset/run_curated_benchmarks.sh \
  --run runs/curated-10x128-s42/train \
  --panel-dir artifacts/transfer-s2026 \
  --job-name four-benchmarks-epoch-first-v1 \
  --report-only
```

`--report-only` 仍会核对原始结果和文件身份。发现已提交结果被改动会报错，不会把混杂结果直接用于报告。若要改变节点列表、生成设置或判分规则，请建立新的 job。

## 验证

```bash
python -B -m unittest discover -s dataset/tests -p 'test_curated_benchmarks.py' -v
```

离线测试使用合成题目和回答，验证倒序执行屏障、同题同提示、逐题恢复、已完成 epoch 报告保持不变、四表总报告、未完成节点处理及来源校验。本地未下载模型或数据、未运行真实 GPU 生成或 Docker 判分；实际运行由 VM 完成。
