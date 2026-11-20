# 原高似然组的 5 条人工数据：从 Base 训练 64 轮

入口：`dataset/train_curated_high5.py`。使用之前 10 条人工改写回答中，历史分组为 `high` 的 **5 条**，从固定版本的 `Qwen/Qwen3-4B-Base` 初始化新的 LoRA。

当前这五条的原始行号是 **6430、2988、4561、14120、1432**。读取的是原实验 `bundle/prepared/train/train.jsonl` 的冻结文本，保持相同 sample_id、问题和改写后的完整回答。High 表示原先从高似然 100 条中挑选出来的组，不按改写后或训练后似然重新排序；不会混入 Low 的 5 条。

## 正式训练

本地 push、VM `git pull` 后，在工程目录使用原训练 `.venv`：

```bash
source .venv/bin/activate
mountpoint -q /data || { echo '/data 尚未挂载'; exit 1; }
mkdir -p /data/repeat_sft/logs

nohup python -B -u dataset/train_curated_high5.py \
  --source-experiment runs/curated-10x128-s42 \
  --out runs/curated-high5-64-s42 \
  --seed 42 \
  >> /data/repeat_sft/logs/curated-high5-64-s42.log 2>&1 < /dev/null &
echo "训练 PID=$!"
tail -n 30 -f /data/repeat_sft/logs/curated-high5-64-s42.log
```

`--source-experiment` 不带 `/train`。新旧实验目录独立，原 10 条实验及结果保留。本次仅使用原实验的冻结数据、tokenizer、Base 版本和训练配置；不加载原 epoch 64/128 的 adapter，也不重新做 Base 似然评分。无需激活 vLLM。

| 项目 | 本次配置 |
|---|---|
| 数据 | 人工改写 High 5 条，完整推理和 boxed 最终答案 |
| 模型 | 同一固定 Base；BF16、LoRA rank 16 / alpha 32 / dropout 0（继承原实际配置） |
| 更新 | batch 1、梯度累积 1；每题一次，每轮打乱后训练 5 次 |
| 训练量 | 64 轮 × 5 条 = **320 次更新** |
| 检查点 | **0、16、32、48、64**；每个含 adapter 和完整恢复状态 |
| 学习率 | 沿用原配置，默认 `1e-5`；前 100 次更新线性预热，之后恒定 |
| 优化器 | 原 AdamW8bit、weight decay 0、clip 1.0、梯度检查点、SDPA |
| 监督与长度 | 仅助手回答及结束符；沿用模板和 10,240-token 上限，不截断 |
| 滚动恢复 | 每 5 次更新保存，保留最近 2 个；永久节点不删除 |

**100 次更新预热在这里等于 20 轮**。因此 epoch 16（第 80 次更新）保存时，最后一次更新学习率为 `8e-6`；第 20 轮最后一次更新达到 `1e-5`。保持原预热步数有利于保留原配置，但按 epoch 对比不同训练集大小时要注意这一差别。

先检查冻结配置和五条监督内容，可在同一命令参数后加 `--prepare-only`（不用 nohup 也可以）。准备阶段仅复制已验证的文本、tokenizer 和原 token 检查记录，不加载模型；训练入口仍会重新编码并检查长度与重合。

输出都在 `/data/repeat_sft/runs/curated-high5-64-s42/`：

```text
bundle/config.json                    实际配置
bundle/selection.json                 五条题目身份、历史分组及来源
bundle/prepared/train/train.jsonl     五条完整训练文本
bundle/prepared/train/inspection.jsonl  五条输入及监督范围
bundle/index.jsonl                    原 token 长度与身份
bundle/code/                         冻结的运行代码
train/checkpoints/epoch-000 ... 064   永久检查点和完整恢复状态
train/recovery/                      滚动恢复点
train/logs/                          每次更新的 loss、样本、学习率
train/progress.json、epoch_times.json  累计进度和逐轮时间
commands.jsonl、progress.json、complete.json  启动、恢复与完成记录
```

## 未完成时恢复

用 `kill -TERM <训练PID>` 或前台 Ctrl+C，程序完成当前更新后保存恢复点。进程退出后，**重复同一条训练命令**即可恢复。`tail -f` 中 Ctrl+C 仅停止看日志。源码更新后继续使用冻结代码；已保存的数据、节点和日志保留。

可以先加 `--stop-after 7`，在第 7 次更新（第 2 轮中间）保存暂停，再去掉该参数重跑，验证恢复。初始化在第 0 轮检查点保存之前中断时，旧的未完成尝试会归档，随后重新初始化 Base。硬中断从最近完整恢复点重算未提交的区间。

## 跑完 64 轮后继续增加轮数

每个节点都保存优化器、学习率状态、随机数和数据位置。已经接入现有 [续训入口](CONTINUE_TRAINING.md)，例如 **再训练 64 轮，到累计 128 轮**：

```bash
nohup python -B -u dataset/continue_training.py \
  --source-run runs/curated-high5-64-s42/train \
  --out runs/curated-high5-128-s42/train \
  --additional-epochs 64 \
  --checkpoint-every 16 \
  >> /data/repeat_sft/logs/curated-high5-128-s42.log 2>&1 < /dev/null &
echo "续训 PID=$!"
```

从 step 320 恢复到 step 640，新增节点 **80、96、112、128**；不重新预热、不重置优化器，继续用同一五条数据。新目录保留起点副本，原 0–64 的实验保持完整。续训中断后重复续训命令即可；也可改 `--additional-epochs`，但已启动的任务要保持参数不变。

训练和续训完成后，标准 `train/` 目录兼容现有 `evaluate_loss.py` 及 `dataset/run_curated_benchmarks.sh`。四任务脚本默认评测该 run 配置中的节点，仍是每个 epoch 的四项任务全部完成后再开始下一 epoch；新五条训练集与评测题会重新做重合检查。训练入口不会自动占用 GPU 启动评测。

离线验证：`python -B -m unittest discover -s dataset/tests -p test_curated_high5.py -v`。使用真实五条文本和模拟子进程检查选题、冻结、Base 初始化路径、中途恢复、完整状态及后续扩展计划；实际大模型 BF16/LoRA 训练在 VM 上执行。
