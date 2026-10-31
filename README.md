# Repeated SFT：LoRA 训练曲线实验

研究固定数据反复训练时，拟合何时饱和，以及数学泛化、跨任务表现和输出行为如何变化。**当前所有新训练使用 LoRA；默认冻结不量化的 BF16 Base，只更新 LoRA adapter**。模型仍是 `Qwen/Qwen3-4B-Base`，目标 GPU 为 RTX A6000。

从 VM clone 后的完整操作见 **[TODO_VM.md](TODO_VM.md)**。模型下载、数据准备和 GPU 实验均在 VM 执行。默认使用挂载到 `/data` 的 1000G volume，存储根目录为 **`/data/repeat_sft`**。

## 能跑哪些实验

| 实验 / 入口 | 使用的数据 | configuration 与执行方式 | 保存的结果 |
| --- | --- | --- | --- |
| 数据准备 `prepare_data.py` | Math-CoT-20k、MATH500、GPQA Diamond | 随机 train 400、holdout 200；固定评测 128 道数学＋64 道 GPQA | 题目和原解答全文、候选池、版本、去重记录、监督范围检查 |
| 连通与恢复测试 `smoke_test.py` | train 最短 7 条＋最长 1 条；少量 holdout/评测题 | 同正式 LoRA 配置；8 条×4 轮；比较连续训练与第 13 步崩溃后从第 8 步恢复 | loss、adapter、恢复状态、逐位比较、生成与判分、`SUCCESS.json` |
| 随机基线 `train.py` | 固定 400 道 Math-CoT 数学题及其完整原解答 | 默认 BF16 LoRA，64 轮，每题一次更新 | 每步日志、每轮耗时/进度、8 个关键 adapter、滚动恢复点 |
| 初始 loss 分组 `score_pool.py` → `select_groups.py` → `train.py` | 同一候选池；low/mid/high 各 400 条 | 同精度冻结 Base 打分；按来源和解答长度分层匹配；各组从 Base 初始化 LoRA | 初始 loss、选题全文、匹配统计、各组独立训练曲线 |
| 速度对比 `benchmark_train.py` | 同一 400 条训练数据，每组一轮 | batch 1/4 × SDPA/FlashAttention 2，四组均使用 LoRA | 耗时、tokens/s、峰值显存、attention 证据、失败原因；可选 adapter |
| 拟合评测 `evaluate_loss.py` | 400 train＋200 holdout | 节点 0/1/2/4/8/16/32/64；给定原解答前文预测下一 token | 逐样本 loss、token 准确率、结束符指标、逐 token NPZ |
| 自由生成 `run_grouped_analysis.sh` | 固定 128 MATH500＋64 GPQA | 四组各 48 题；每组按 64→32→16→8→4→2→1 评测，再进入下一组 | 每题完整输出、token IDs、长度、停止原因、答案提取、判分和报告 |
| 汇总与复判 `grade.py` / `analyze_run.py` / `export_curves.py` | 已保存结果 | CPU；不重新训练或生成 | 饱和/首次答对/退步/迁移表、早晚排名、可追溯到题目与解答的 HTML |

**MATH500 和 GPQA 都不参与训练。** Holdout 看未见原解答的拟合，MATH500 看独立解题的数学泛化。初始 loss 低只表示 Base 容易模仿该解答，不预设它的短期或长期效果最好。

## 默认 configuration

| 项目 | 设置 |
| --- | --- |
| 配置文件 | `configs/baseline.json` |
| Base | Qwen3-4B-Base；准备数据时固定 Hub commit |
| 训练方式 | LoRA；Base 冻结为 BF16，不量化 |
| LoRA | rank 16、alpha 32、dropout 0；标准零增量初始化 |
| 注入位置 | 每层 `q/k/v/o_proj`、`gate/up/down_proj`；embedding、LM head 和 bias 不训练 |
| adapter 参数 | FP32；只有 LoRA A/B 参数进入优化器与梯度裁剪 |
| batch / 梯度累积 | 1 / 1，即每题更新一次 adapter |
| 预算 | 400×64＝25,600 次更新；每轮打乱 |
| 优化器 | AdamW8bit；betas=(0.9, 0.999)，eps=1e-8，weight decay=0 |
| 学习率 | 1e-5；前 100 次更新线性预热，之后恒定 |
| 显存控制 | 梯度检查点；SDPA；LM-head loss 按 32 个目标 token 分块 |
| 梯度裁剪 | 1.0 |
| 监督与长度 | 只监督助手完整推理、答案、结束符；模板后总长超过 10,240 的整条样本过滤，不截断 |
| 保存 | epoch 0/1/2/4/8/16/32/64；每 400 步滚动恢复，保留最近 2 份，节点重复时只存一份 |

学习率沿用原基线，**尚未证明是 LoRA 最优值**。更改 rank、学习率、长度规则或量化方式时另建 run，不能混入已有曲线。

`configs/lora_bf16.json` 与默认配置相同，保留为显式的 BF16 LoRA 入口。另有 **`configs/qlora_nf4.json`**：冻结 NF4 4-bit Base、double quantization、BF16 计算，其他配置相同，供需要更低显存时单独比较。训练、smoke、benchmark 通过 `--config` 选择；初始 loss 打分必须使用同一个配置，后续评测自动读取 run 的冻结配置。

[NVIDIA RTX A6000](https://www.nvidia.com/en-us/products/workstations/rtx-a6000/)有 48GB 显存。默认采用 BF16 LoRA；实际可用显存、长样本峰值及 batch 4 是否能运行，仍由 VM 上的 smoke 和 benchmark 验证。OOM 会报错并保留完成结果，不自动截断、删题或换 batch。可选 QLoRA 实现参考 [PEFT 量化文档](https://huggingface.co/docs/peft/v0.18.0/developer_guides/quantization)。

## 评测与结果怎么读

自由生成每题一次采样：temperature=0.6、top-p=0.95、top-k=20，**最多 8,192 个新 token**。到上限停止并标记截断，不伪造 EOS，不把截断算作正常结束。每组 32 道 MATH500＋16 道 GPQA；该组七个节点的生成、判分、报告保存后才进入下一组。生成和判分分离，可修改抽取规则后复判。

默认分组生成不包含 epoch 0，不能判断相对原始 Base 的自由解题提升；epoch 0 拟合结果仍保留。可用 `generate.py --epochs 0` 在新目录补测 Base，再与同协议七个节点一起重新判分/分析。也可用 `run_analysis.sh` 跑全部八个节点的完整面板；这个入口按节点执行，不采用四组顺序。

| 要观察什么 | 优先查看 |
| --- | --- |
| 训练是否正常、每轮耗时 | `logs/*.jsonl`、`progress.json`、`epoch_times.json` |
| 逐样本 loss / token 准确率何时稳定 | `fit_curves.csv`、`fit_samples.csv`、`fit_saturation.csv` |
| 数学泛化、GPQA 迁移及输出行为 | `downstream_curves.csv`、`task_milestones.csv`、`paired_changes.csv` |
| 某题首次答对、后续退步 | `question_milestones.csv`、`question_transitions.csv`、`questions/*.html` |
| 题目、原解答、各节点完整回答 | `analysis/report.html` 中的题目链接及随报告保留的 evidence |
| 数据组早晚排名 | `export_curves.py` 的 `early_late_rankings.csv` |

“首次答对”是首次**观测**答对的检查点，无法定位未评测的中间 epoch。饱和依赖容差；高 loss 稳定不等于记住。正常结束率与正确率共同提升是相关性，不直接代表因果。小面板、单次采样用于探索，排名交叉须换抽样/训练 seed 复现。详细定义见 [ANALYSIS.md](ANALYSIS.md)。

## 检查点、恢复和存储

新检查点包含 `adapter_model.safetensors`、`adapter_config.json`、`recipe_model.json`、tokenizer 和 `complete.json`；正式训练还保存 `training_state.pt`（优化器、学习率、随机状态、数据位置）。**adapter 不是独立完整模型**：推理需要记录的 Base commit 加该节点 adapter，项目评测自动处理。epoch 0 的零增量 adapter 表示同精度的原始 Base。

恢复核验权重哈希、代码、数据、tokenizer 和配置。Ctrl+C / SIGTERM 在当前更新后保存；断电或 kill -9 回到最近完整恢复点。恢复追加 attempt，关键节点不覆盖。git pull 后源代码变化，使用 run 内原始代码快照恢复，见 TODO。

相对的 `runs/...`、`artifacts/...` 指向 **`/data/repeat_sft`，不是 clone 目录**；缓存/临时文件默认也在 volume。1000G 通常足以容纳这批 LoRA 实验，旧全参权重、重复缓存和历史生成备份仍占空间。新节点不重复保存完整 Base，备份时保留 Base 缓存/固定版本信息与全部 adapter。

**旧全参 run 和拟合结果不会被改写或转换为 LoRA。** 新训练使用 `random-lora-s42` 等新目录，不能接着旧全参恢复点训练。旧任务恢复沿用它的代码快照及原环境。

若已经用此前默认值启动 QLoRA，继续使用该 run 的冻结配置与代码快照恢复；新的 BF16 LoRA 必须另建目录，不能与 QLoRA 结果合并为同一训练曲线。

操作链接：[VM TODO](TODO_VM.md) · [分组评测](GROUPED_EVALUATION.md) · [四种测速](BENCHMARK.md)。本地只验证 CPU 小随机模型与合成数据；真实 NF4 CUDA 训练、显存、速度、GPU 中断恢复仍须在 VM 验证。
