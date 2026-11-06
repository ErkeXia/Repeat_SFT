# 用新 checkpoint 做四任务 vLLM 评测

入口：`run_intermediate_vllm.sh`。默认读取补跑 run `runs/random-lora-s42-intermediate-v1`，使用它记录的全部新节点：**30 → 26 → 22 → 20 → 14 → 12 → 10 → 9 → 7 → 6**。不使用原 run 的旧节点，也不要求先做 teacher-forcing 拟合评测。

沿用已经准备的题目与提示词，不重新抽题、不重新打乱 GPQA 选项：

| 数据集 | 总题数 | 每组题数 | 判分 |
| --- | --- | --- | --- |
| MATH500 | 128 | 32 | 沿用最终答案抽取和 math-verify 符号验证 |
| GPQA | 64 | 16 | 沿用最终选项字母匹配 |
| BBH | 100 | 25 | 沿用四个逻辑子任务的答案匹配 |
| LiveCodeBench | 100 | 25 | 沿用 Docker 中的公开及隐藏测试 |
| 合计 | **392** | **两个面板分别分组** | 每题一次采样 |

沿用**两个面板，各自分四组**，保留原来的分组：

1. 先跑 MATH500/GPQA：四组各 **48 题**，每组 32 道 MATH500 + 16 道 GPQA。
2. 该面板全部完成后，再跑 BBH/LiveCodeBench：四组各 **50 题**，每组 25 道 BBH + 25 道 LiveCodeBench。

每组生成十个节点的全部回答、完成判分和报告并保存后，才进入该面板的下一组。两套面板分别保存，合计 **392 × 10 = 3,920 条回答**。

vLLM 并发 **8**；BF16 Base + BF16 LoRA adapter；temperature=0.6、top-p=0.95、top-k=20、最多 **8,192 个新 token**。每组只加载一次 Base，依次切换 adapter。每个面板自己的全部题目决定上下文预算，保证该面板各组/节点配置一致。到上限停止，按实际 EOS 和 token 数标记截断，不伪造结束符。

## VM 上运行

先推送本地修改，在 VM `git pull`。使用原有的独立 vLLM 环境 `/data/repeat_sft/venvs/vllm-0.11.0/bin/python`；`requirements-vllm.txt` 已包含 math-verify，无需为这个入口新增 Python 依赖。

需要以下现有材料：补跑 run 中的目标 checkpoint、`artifacts/transfer-s2026`、可访问的 Docker，以及之前通过自检的 `repeat-sft-lcb:py311-v1` 镜像。**这次没有修改容器内的 worker，已有通过自检的镜像可以直接复用**。只有镜像不存在或提示 worker 不匹配时，才按 `TRANSFER_EVALUATION.md` 重新构建。

在 GPU 空闲后启动：

```bash
cd ~/Repeat_SFT
mkdir -p /data/repeat_sft/logs
nohup bash run_intermediate_vllm.sh \
  --run runs/random-lora-s42-intermediate-v1 \
  --panel-dir artifacts/transfer-s2026 \
  --job-name intermediate-two-panels-vllm-v1 \
  >> /data/repeat_sft/logs/intermediate-two-panels-vllm-v1.log 2>&1 < /dev/null &
echo "新节点四任务评测 PID=$!"
tail -n 30 -f /data/repeat_sft/logs/intermediate-two-panels-vllm-v1.log
```

默认节点从该 replay run 的 `run.json` 读取。如果它只计划了另一组新节点，就使用那组节点。首次启动时，在开始任何 GPU 生成前，会先准备两套面板，并核验完整 checkpoint、原始输入、math-verify 和 Docker 自检；缺少节点会提前报错，不自动跳过。

如果暂时只有部分新节点完成，可以在**新任务名**中明确选择，例如：

```bash
nohup bash run_intermediate_vllm.sh \
  --run runs/random-lora-s42-intermediate-v1 \
  --panel-dir artifacts/transfer-s2026 \
  --epochs 6,7,9,10,12,14 \
  --job-name intermediate-first-six-vllm-v1 \
  >> /data/repeat_sft/logs/intermediate-first-six-vllm-v1.log 2>&1 < /dev/null &
```

实际仍按 epoch 降序生成。已启动任务的节点列表和协议不会变；以后新增节点用另一个 job-name。不要把 `--run` 指向原 `random-lora-s42`，新入口会拒绝普通 run，防止拿错模型。

## 停止与恢复

- `tail -f` 的 Ctrl+C 只退出日志查看。
- 停止评测用 `kill -TERM <评测PID>`，等待当前进程退出。优先使用 TERM，让子进程和 Docker 有机会清理。
- **恢复原样重跑同一条 nohup 命令**；相同 run、panel-dir、job-name 和可选 epochs，不加 reset。
- 每个完整生成回答立即保存；中断时正在生成的最多 8 题会重跑，已完成的题保留。
- 判分逐题保存；已判分题跳过，中断的代码测试重跑。完整组不会再生成或执行测试。
- 首次启动会冻结代码、题目、提示词、参数、模型节点身份和 Docker 镜像 ID；之后 `git pull` 仍调用该任务的冻结代码恢复。保留它使用的 Docker 镜像。
- 同一任务有进程锁，防止两次启动同时写入。

## 保存位置与怎么看结果

所有新输出在数据 volume 上：

```text
/data/repeat_sft/runs/random-lora-s42-intermediate-v1/
  postprocess/intermediate-two-panels-vllm-v1/
    suite/plan.json                         节点、两个面板、replay 来源、配置身份
    suite/code/                             冻结调度代码
    progress.json                           当前面板及其进度文件位置
    report.html                             两套报告的入口
    complete.json                           默认 3,920 条回答及判分已全部完成
    math-gpqa/                              192 题；四组各 48 题
    bbh-livecodebench/                      200 题；四组各 50 题
```

两个面板子目录各自使用下面的结构：

```text
setup/plan.json                             分组、参数、checkpoint 身份
setup/code/                                冻结生成与判分代码
setup/panel/panel.jsonl                     本面板全部固定题目和实际提示词
setup/panel/references/                     原题/参考解答或代码测试数据
progress.json                              当前组及 generation/grading/complete 阶段
groups/group-001/
  generation/epoch-030/responses.jsonl       原始回答、token IDs、长度、停止原因
  grades/epoch-030/graded.jsonl              抽取、判分、代码测试详情
  report/report.html                        本组十节点指标与逐题全文
  complete.json                             本组全部完成
generation/epoch-*/responses.jsonl           完成后合并的本面板原始回答
grades/epoch-*/graded.jsonl                  完成后合并的本面板判分；不重复执行测试
report/report.html                          本面板完整报告
report/curves.csv                            各任务/epoch 正确率及输出行为
report/question_milestones.csv              首次/持续答对、退化及恢复节点
report/questions/*.html                     全部已选节点的逐题完整回答
report/questions/*.json                     同题完整数据和 token IDs
complete.json                               本面板全部完成
```

生成和判分仍分离；以后修改判分规则，可对**对应面板子目录**中的 `generation/` 使用 `grade_transfer.py --panel-dir .../setup/panel --responses .../generation --out .../grades-v2`，然后用 `report_transfer.py` 生成新报告。数学/GPQA 判分不使用 Docker；BBH/LCB 判分需要原执行镜像。原回答无需重新生成。

逐题退化分析会自动识别这个入口下的两个面板，并分别处理各自实际完成的节点：

```bash
python -B analyze_regressions.py \
  --run runs/random-lora-s42-intermediate-v1 \
  --out runs/random-lora-s42-intermediate-v1/analysis-regressions-v1
```

已有原 run 的旧权重、拟合、MATH500/GPQA 和迁移结果保持原样；新任务不复制 Base 或 checkpoint。与原 epoch 8/16/32 等点一起解读时，应注明新节点来自 [独立补跑](REPLAY_CHECKPOINTS.md)，并核对题目、后端和生成设置。当前每题只采样一次，不能把一次正确率起伏直接当作确定规律。
