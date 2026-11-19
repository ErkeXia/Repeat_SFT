# 10 条人工改写数据：Base 似然度 + 128 轮 LoRA

一个命令依次完成两个阶段：先用未加载任何 adapter 的 `Qwen/Qwen3-4B-Base` 计算这 10 条**改写后**完整回答的似然度，再从同一固定版本的 Base 初始化全新的 LoRA，训练 128 轮。High 5 条和 Low 5 条合在一起训练，不是各训练一遍。

## 在 VM 上运行

先 `git pull`，确认 `/data` 已挂载，并使用原来的训练虚拟环境 `.venv`。现有的独立 vLLM 0.11.0 环境也需要保留；启动器会自动调用它完成评分，然后退出评分进程、释放 GPU，再启动训练。不需要手动切换环境，不要在训练环境里重新安装 vLLM。

输入：

- Git 中的 `dataset/math-cot-20k/manual-curation/high-low-5-v1/`：10 份 `.response.txt`、`selection.json`、`originals.jsonl`。这些新修改和文件都需要提交并推送；不需要上传完整 Parquet。
- `/data/repeat_sft/artifacts/data-s42/`：此前准备好的 Base 版本、tokenizer、holdout 和评测题。它只提供固定协议和评测数据，旧训练的 400 条不会混入此次训练。
- 现有训练环境和 `/data/repeat_sft/venvs/vllm-0.11.0/bin/python`。路径不同可加 `--vllm-python /实际路径/bin/python`，或设置 `VLLM_PYTHON_BIN`；恢复时保持同一配置。

在工程目录运行：

```bash
source .venv/bin/activate
mountpoint -q /data || { echo '/data 尚未挂载'; exit 1; }
mkdir -p /data/repeat_sft/logs

nohup python -B -u dataset/train_curated.py \
  --data artifacts/data-s42 \
  --out runs/curated-10x128-s42 \
  --seed 42 \
  >> /data/repeat_sft/logs/curated-10x128-s42.log 2>&1 < /dev/null &
echo "实验 PID=$!"
tail -n 30 -f /data/repeat_sft/logs/curated-10x128-s42.log
```

`--curation` 默认指向本次 Git checkout 中的候选目录。`--data`、`--out` 的相对路径指向 `/data/repeat_sft`；所有评分、日志、代码快照和检查点存到数据 volume。无需下载全量数据；VM 若没有缓存固定版本的 Base 权重，会由原加载流程缓存到 volume。本地没有执行模型评分或训练。

可先用同样参数运行以下任一种检查：

```bash
# 仅冻结/编码，检查监督范围与长度，不加载模型权重。
python -B -u dataset/train_curated.py --out runs/curated-10x128-s42 --prepare-only

# 完成 Base 评分后暂停，不开始训练。
python -B -u dataset/train_curated.py --out runs/curated-10x128-s42 --score-only

# 先评分，再训练到第 20 次更新保存暂停，可实际检验恢复。
python -B -u dataset/train_curated.py --out runs/curated-10x128-s42 --stop-after 20
```

随后执行上面的正式命令即可继续；恢复时去掉 `--prepare-only`、`--score-only` 或 `--stop-after`。如果用了非默认输入/seed/环境路径，前后保持一致。

## 实验配置

| 项目 | 配置 |
|---|---|
| 数据 | 人工改写的 high 5 + low 5，共 10 个不同问题 |
| 模型 | 原 `artifacts/data-s42/manifest.json` 固定的 Qwen3-4B-Base commit |
| 似然度 | vLLM 0.11.0、BF16、无 adapter、并发 8，复用原全集评分算法 |
| 监督 | 仅助手完整推理、Final Answer 和 `<\|im_end\|>`；prompt、padding 和结束后的换行不计入 |
| 长度 | 实际聊天模板总长 ≤ 10,240；超限报错，不截断或少训练某一条 |
| 训练 | BF16 Base + LoRA；默认 rank 16、alpha 32、dropout 0，7 个 attention/MLP 投影；adapter 参数 FP32 |
| Batch | 1，梯度累积 1；每题更新一次，每轮打乱 10 条各训练一次 |
| 总更新 | 128 × 10 = 1,280 |
| 优化器 | AdamW8bit，LR 1e-5，weight decay 0，clip 1.0，梯度检查点开启，SDPA |
| 学习率 | 保留原配置：前 **100 次更新 = 10 轮** warmup，然后恒定 |
| 永久检查点 | **0、16、32、48、64、80、96、112、128** 轮；每个含 adapter 和完整恢复状态 |
| 滚动恢复点 | 每 10 次更新（每轮），保留最近 2 个；永久节点不重复写 recovery |

默认继承 `configs/baseline.json`，只覆盖本次样本数、轮数、保存节点、恢复频率和 seed。实际配置保存在 `bundle/config.json`。epoch 0 是 Base 的等价初始 LoRA（零增量），没有从旧实验的 adapter 继续训练。检查点只保存 adapter，加载时仍需要固定的 Base 权重。

原题、原 user/system 消息保留；对编辑后的回答重新生成 sample_id。历史 high/low 标签及原样本身份放在 `curation` 字段，不沿用旧似然或旧 token 长度。检查与原 holdout、MATH500/GPQA 参考题的重复，不声称排除了语义污染。首次准备会打印两个样本的实际监督范围，文件中保存全部 10 条。

## 输出与查看

```text
/data/repeat_sft/runs/curated-10x128-s42/
  bundle/
    plan.json                    固定的模型版本、请求参数、所有输入/代码哈希
    config.json                  实际训练配置
    curation/                    原记录、选择清单和本次 10 份改写文本
    prepared/train/train.jsonl   新 sample_id、完整题目和改写回答
    prepared/train/inspection.jsonl   全部 10 条输入和监督范围
    index.jsonl                  实际 token 数与 token 身份
    code/                        本次运行使用的冻结代码
  base_likelihood/
    report.html                  10 条排序、汇总及可展开的完整题目/回答
    summary.json                 总体及历史 high/low 各 5 条的统计
    scores.csv                   逐题 NLL、平均 logprob、归一化似然等
    samples.jsonl                每题完整文本、来源及分数
    records/                     每题原子提交的评分结果
    tokens/                      完整输入 token 和助手逐 token logprob（含结束符）
    job.json                     评分环境、模型和算法身份
    complete.json                10 条评分完成的标记
  train/                         标准 LoRA run，兼容现有 evaluate_loss / 分析工具
    checkpoints/epoch-000 ... epoch-128
    recovery/
    logs/                        每次更新的 loss、样本、LR
    progress.json                步数、轮数和时间进度
  commands.jsonl                 每次阶段启动/恢复命令
  progress.json                  整体阶段：评分/训练/暂停/完成
  complete.json                  全部评分和 9 个节点完成后生成
```

这里的归一化似然是 `exp(mean_logprob)`，不是整条解答概率；同时保留 sum logprob、序列 log10 likelihood。汇总区分逐样本平均和按 token 加权的平均，避免回答长度变化带来解释混淆。High/Low 只是改写前标签，新分数不一定仍处于原来的高/低端。

不会自动运行四个 benchmark 的长生成。训练完成后可用 [按 epoch 倒序的四任务评测入口](CURATED_BENCHMARKS.md)，每个 epoch 全部完成后生成统一报告。若之后查看训练/holdout 拟合曲线，可以在训练环境运行：

```bash
python evaluate_loss.py \
  --run runs/curated-10x128-s42/train \
  --epochs 0,16,32,48,64,80,96,112,128

python analyze_run.py \
  --run runs/curated-10x128-s42/train \
  --epochs 0,16,32,48,64,80,96,112,128 \
  --allow-partial \
  --out runs/curated-10x128-s42/analysis-fit-v1
```

## 中断与恢复

用 `kill -TERM <实验PID>` 或前台 Ctrl+C 停止；`tail -f` 中 Ctrl+C 只停止看日志。

重复同一启动命令即可恢复。已提交的逐题评分不会重算；训练从最近的完整恢复点继续，保留优化器、scheduler、随机数和数据位置。正式节点永久保留。`kill -9`/断电时，最近一个尚未完整保存的更新区间可能需要重算，正常情况下最多约一轮（10 次更新）。

如果在 epoch 0 完整写出之前中断，会把该次未完成的训练目录保留到 `incomplete_initializations/`，再从 Base 初始化。未提交的 bundle 初始化目录也保留，可重新初始化。运行锁防止同一个实验重复启动；停止后应等进程退出再重跑。

首次冻结后，继续使用快照代码和文本；后来 `git pull` 或编辑候选文件不会改变已开始的实验。要训练另一版改写，请选择新的 `--out`。不要手工覆盖 bundle、评分记录或检查点。

## 本地验证边界

```bash
python -B -m unittest discover -s dataset/tests -p 'test_curated_training.py' -v
```

离线验证使用真实的 10 条文本、合成 tokenizer/logprob 和模拟训练进程，检查新旧身份、监督范围、结果保存、评分续跑和训练中断恢复。它不代表真实 GPU 的模型加载、显存、数值与速度已经验证；这些需要在 VM 上通过准备、评分和短暂停恢复步骤确认。
