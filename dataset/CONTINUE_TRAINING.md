# 从 128 轮继续训练到 256 轮

`continue_training.py` 从原训练最后一轮的完整检查点续训到新的总轮数。对目前的 10 条人工改写数据，默认从 epoch 128 / step 1280 开始，再训练 128 轮、1,280 次更新，到 epoch 256 / step 2560。

沿用原实验冻结的训练代码、10 条完整数据、tokenizer、Base 版本和 LoRA 配置。恢复 adapter、AdamW8bit 优化器、学习率调度器、随机数状态和数据位置。轮数及步数累计；前 100 次更新的 warmup 已完成，继续保持原来的 `1e-5`。不会重新计算 Base 似然度。

## 在 VM 上运行

先 `git pull`，使用原训练 `.venv`；这一步不使用 vLLM 环境。以下命令在工程根目录执行：

```bash
source .venv/bin/activate
mountpoint -q /data || { echo '/data 尚未挂载'; exit 1; }
mkdir -p /data/repeat_sft/logs

nohup python -B -u dataset/continue_training.py \
  --source-run runs/curated-10x128-s42/train \
  --out runs/curated-10x256-s42/train \
  --additional-epochs 128 \
  --checkpoint-every 16 \
  >> /data/repeat_sft/logs/curated-10x256-s42.log 2>&1 < /dev/null &
echo "续训 PID=$!"
tail -n 30 -f /data/repeat_sft/logs/curated-10x256-s42.log
```

`--source-run` 是包含 `run.json` 的训练目录；`--out` 是独立的新训练目录。相对路径按已有约定指向 `/data/repeat_sft`。原实验的检查点、拟合结果、推理结果和报告继续留在原目录。

新目录保存 **144、160、176、192、208、224、240、256** 的完整检查点，另复制一份 epoch 128 作为续训起点。默认评测节点只包含八个新增节点；原 0–128 节点仍在原实验评测。除总轮数和检查点列表外，所有训练配置保持原值，包括每轮打乱、每条更新一次、恢复点频率和保留数量（这次原实验为每 10 步保存、保留最近 2 份）。

可先查看计划和按实际检查点体积估算的磁盘需求，不导入 torch、不加载模型，也不写实验目录：

```bash
python -B dataset/continue_training.py \
  --source-run runs/curated-10x128-s42/train \
  --out runs/curated-10x256-s42/train \
  --plan-only
```

加 `--prepare-only` 会在 CPU 上复制并验证完整训练状态，但不加载模型、不开始训练。正式执行时去掉此参数即可。空间估计包含新检查点、复制的起点、滚动恢复点、临时保存和余量，不包含缺失的 Base 缓存或后续评测输出。

## 暂停与恢复

使用 `kill -TERM <续训PID>` 或前台 `Ctrl+C`；程序完成当前更新并保存恢复点。`tail -f` 的 `Ctrl+C` 仅停止看日志。

**等待进程退出后，重复同一条 nohup 命令即可继续，不需要加 `--resume`。** 已保存节点和日志保留。程序从新目录中最近的完整检查点恢复；原训练目录不再参与恢复。Git 更新后继续使用续训目录冻结的代码。已完成 256 轮时再次运行会验证结果并退出，不会又训练 128 轮。

可先验证一次暂停恢复：首次启动时在同样参数后加 `--stop-after 1290`，训练到全局第 1,290 次更新（新增 10 次）保存暂停；然后去掉 `--stop-after` 重跑。此参数使用累计更新次数，不能写成 `10`。强制杀进程或断电时，从最近完整恢复点重算尚未提交的区间；原始日志保留。

初始化时若源 epoch 128 缺少完整状态，会报错退出。不能只提供 adapter 权重代替完整恢复点。不要手动修改旧 `run.json` 或检查点。

## 输出与评测

```text
runs/curated-10x256-s42/train/
  run.json                       延长后的配置、原实验身份、延长记录
  data/、tokenizer/              原实验的冻结输入
  checkpoints/epoch-128/         起点副本，保留原权重及训练状态
  checkpoints/epoch-144 ... 256  新增完整检查点
  recovery/                     滚动恢复点
  logs/、timing/、attempts.jsonl 新增更新的损失、逐轮时间、恢复记录
  progress.json                 当前累计步数、轮数及 ETA
  continuation/
    plan.json、manifest.json    原始/新增轮数、文件校验、空间估计
    source_run.json             原 run.json 副本
    source_checkpoint.json      原 epoch 128 的完成标记
    code/                      冻结的续训启动器
    commands.jsonl             每次训练的实际命令和起始步数
    environments.jsonl         启动环境
    progress.json              续训任务状态
    complete.json              八个新增节点均完成后生成
```

新目录只记录续训阶段的 loss 和时间；0–128 的历史仍查原目录。进度采用累计 2,560 步，因此启动时约为 50%；早期轮次时间在新目录中没有测量，标为缺失。

保存状态中的配置身份随新预算更新，其他训练状态不重置；`continuation/` 保存旧身份和源文件哈希以便追溯。再次恢复会检查冻结配置、代码、输入和起点副本，拒绝混用不同实验。

完成续训后，现有四任务脚本可直接使用新目录，按 **256 → 240 → 224 → 208 → 192 → 176 → 160 → 144** 执行，每个 epoch 完成全部四个 benchmark 后出报告：

```bash
nohup bash dataset/run_curated_benchmarks.sh \
  --run runs/curated-10x256-s42/train \
  --panel-dir artifacts/transfer-s2026 \
  --job-name four-benchmarks-epoch-first-v1 \
  >> /data/repeat_sft/logs/curated-256-four-benchmarks-v1.log 2>&1 < /dev/null &
```

仍为 MATH500 128、GPQA 64、BBH 100、LiveCodeBench 100 道，vLLM 并发 8、最大生成 8,192 token；两个内部面板各分四组。详细报告和断点恢复约定见 [CURATED_BENCHMARKS.md](CURATED_BENCHMARKS.md)。续训本身不会自动启动评测。

查看新增节点的训练集/holdout 拟合：

```bash
python evaluate_loss.py --run runs/curated-10x256-s42/train \
  --epochs 144,160,176,192,208,224,240,256
python analyze_run.py --run runs/curated-10x256-s42/train \
  --epochs 144,160,176,192,208,224,240,256 --allow-partial \
  --out runs/curated-10x256-s42/analysis-fit-v1
```

## 离线验证

`python -B -m unittest discover -s dataset/tests -p test_continue_training.py -v` 使用小型 CPU 模型，运行现有训练循环与真实 torch 优化器/保存恢复，检查暂停恢复后的权重、优化器、调度器、随机数和样本顺序与连续执行一致。还检查旧实验不变、错误状态拒绝、已完成任务不重训。GPU、BF16、bitsandbytes 和实际大模型训练需在 VM 上验证；本地不下载模型或数据。
