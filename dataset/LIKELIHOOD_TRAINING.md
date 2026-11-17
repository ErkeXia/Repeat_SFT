# 高 / 低似然各 100 条，独立 LoRA 训练 16 轮

入口：`dataset/train_likelihood_extremes.py`。一个命令先训练 high，再训练 low；两组都从同一个固定版本的 Base、同一个 seed 重新初始化 LoRA。不会把 high 的已训练权重接着用于 low。

## 选题和配置

- 必须先完成全部 20,480 条的 Base 似然评分；部分报告、smoke 或缺失的逐条分数不能启动本实验。
- **按用户选择，从原 `artifacts/data-s42/pool.jsonl` 合格候选池中排名**。继承原准备阶段的题目去重、答案检查、模板后 10,240-token 上限，以及对 holdout / 原 MATH500、GPQA 评测题的避重规则。原 holdout、原固定评测面板保持一致，两组训练都不使用它们。
- 主指标为 `mean_logprob`，越大越容易模仿；与 `exp(mean_logprob)` 排序相同。同分按原始行号。high 取前 100，low 取最后 100；两组不重合。它们未必就是全集 HTML 报告里的 top100/bottom100。
- 默认 BF16 Base、不量化；LoRA rank 16、alpha 32、dropout 0，7 类 attention/MLP 投影；adapter 为 FP32。
- 每题更新一次，batch 1 / accumulation 1；每轮随机打乱 100 条；每组 16 轮、1,600 次更新，两组合计 3,200 次。
- 学习率 1e-5，前 100 次更新预热、之后恒定；AdamW8bit，weight decay 0，梯度裁剪 1.0；梯度检查点、SDPA。
- 保存 **epoch 0、1、2、4、8、10、12、14、16**。你要求的 8 个节点全部保留；额外保留 epoch 0 作为起点。
- 每 100 步保存恢复点（与关键节点重合时不重复写），每组保留最近 2 个滚动恢复点。关键节点都含优化器、学习率、随机状态和数据位置。

这次是纯粹比较似然两端，**没有匹配来源或解答长度**；两组的来源分布、平均/最大 token 数和初始 NLL 会记录在 `bundle/plan.json`，分析结果时一起查看。当前避重继承原数据准备的 MATH500/GPQA 参考集；以后若评测 BBH/LiveCodeBench 等额外面板，仍需按相应评测入口检查重合。

## 空间

按当前 [Qwen3-4B-Base 配置](https://huggingface.co/Qwen/Qwen3-4B-Base/blob/main/config.json)计算，rank 16 的 7 类投影合计 **33,030,144 个 LoRA 参数**，FP32 adapter 权重约 **126 MiB/份**。Base 权重使用共同缓存，不在每个 checkpoint 重复保存。

两组共 18 个关键节点（含两个 epoch 0），另保留最多 4 个滚动恢复点。包含 8bit 优化器、tokenizer、日志及代码，通常新增约 **5–7 GiB**。脚本按冻结的实际模型配置估算，首次启动采用更保守的约 **21 GiB 空闲预算**（含临时保存、优化器较大状态、代码和 8 GiB 余量）；**建议至少留出 25 GiB 空闲**。

1000G 卷的总容量够用，但能否立即启动取决于当前剩余容量。脚本会检查目标文件系统；每次 checkpoint 保存也有空间检查。可先只读查看：

```bash
df -h /data
python -B dataset/train_likelihood_extremes.py --plan-only
```

## VM 一次启动两组

在仓库根目录，**切回原训练环境 `.venv`，不是 vLLM 环境**。vLLM 用于之前的似然评分，LoRA 训练仍用 Transformers / PEFT / PyTorch。已有训练环境无需重装依赖。

```bash
source .venv/bin/activate
mkdir -p /data/repeat_sft/logs

nohup python -B -u dataset/train_likelihood_extremes.py \
  --scores dataset/math-cot-20k/likelihood/base-v1 \
  --data artifacts/data-s42 \
  --out runs/likelihood-100x16-s42 \
  --seed 42 \
  >> /data/repeat_sft/logs/likelihood-100x16-s42.log 2>&1 < /dev/null &
echo "两组训练任务 PID=$!"
tail -n 30 -f /data/repeat_sft/logs/likelihood-100x16-s42.log
```

相对数据/结果路径落到 `/data/repeat_sft`；`/data` 必须已挂载。程序只读取已准备好的数据、似然文件和 tokenizer；无需重新准备数据。先停止占用同一 GPU 的 vLLM 进程，避免训练和评分同时占显存。

若先想看选题，将相同命令去掉 nohup/background 并增加 `--prepare-only`：只冻结两组数据、配置和代码，不加载模型训练；随后用原正式命令继续。

## 中断恢复

用 `kill -TERM <任务PID>` 或前台 Ctrl+C。信号会传给当前训练，完成本次更新并保存后停止，**不会继续启动第二组**。`tail -f` 中 Ctrl+C 只停止看日志。

**恢复直接重新执行同一条启动命令**：

- high 未完成：从 high 最近的完整恢复点继续；完成后再开始 low。
- high 完成、low 未完成：核验并跳过 high，恢复 low。
- 两组完成：核验所有要求的 checkpoint 后跳过训练。
- 在第一个完整恢复点之前崩溃：保留未完成目录到 `incomplete_initializations/`，该组重新从 Base 初始化。
- kill -9 / 断电：只能恢复到最近完整保存点，最多重算约 100 次更新；不要在旧进程尚未退出时启动第二份任务。

首次准备时会冻结训练器和启动器。之后的 git pull 不会改写本任务实际使用的冻结代码；恢复仍要求相同的数据路径、输出目录、配置文件路径和 seed。未完成的旧尝试、已有关键 checkpoint 和旧实验均不会被覆盖。

## 结果和后续评测

```text
/data/repeat_sft/runs/likelihood-100x16-s42/
  bundle/
    plan.json                  选题身份、原始行号、来源/长度统计、空间预算
    config.json                两组共用的实际配置
    prepared/high/train.jsonl  高似然 100 条全文及每条初始分数/候选池排名
    prepared/low/train.jsonl   低似然 100 条全文及每条初始分数/候选池排名
    code/                     本任务使用的冻结代码
  high/                       标准训练 run，可直接交给原评测脚本
    checkpoints/epoch-000 ... epoch-016
    recovery/                 最近 2 个非关键恢复点
    logs/                     每步 loss、样本 ID、学习率
    progress.json             当前步数、epoch、耗时/进度
  low/                        同样结构
  commands.jsonl              每次启动/恢复的实际命令
  progress.json               当前组和整体进度
  complete.json               两组和全部关键节点完成后才生成
```

训练完成后不会自动启动耗时的自由生成。拟合评测示例（同一训练环境）：

```bash
python evaluate_loss.py --run runs/likelihood-100x16-s42/high --epochs 0,1,2,4,8,10,12,14,16
python evaluate_loss.py --run runs/likelihood-100x16-s42/low  --epochs 0,1,2,4,8,10,12,14,16
```

各组为普通 LoRA run，可继续使用现有生成/判分/分析入口；评测时显式指定本次实际保存的 epoch。

两组一起做 MATH500、GPQA、BBH、LiveCodeBench 自由生成评测，使用新增的 [四任务 vLLM 入口](LIKELIHOOD_BENCHMARKS.md)。该入口不要求先做上述拟合评测，保留原题并重新核验新训练集的题目重合；一次启动依次完成 high、low，各面板分四组，可中断续跑。

## 验证范围

```bash
python -B -m unittest discover -s dataset/tests -p 'test_likelihood_training.py' -v
```

本地用合成分数和真实子进程模拟暂停、恢复、第二组独立启动、完成后跳过，以及检查点损坏检测；核对了选题排序和空间公式。本地没有下载模型或跑 GPU 训练，实际显存、速度与数值结果须在 VM 验证。
