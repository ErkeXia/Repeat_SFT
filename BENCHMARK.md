# 400 条数据、一轮训练测速

`benchmark_train.py` 只测试以下四种组合，每种用一个独立子进程，串行使用一张 GPU：

| 名称 | Batch size | Attention 配置 | 更新次数 |
| --- | ---: | --- | ---: |
| `b1-sdpa` | 1 | `sdpa` | 400 |
| `b4-sdpa` | 4 | `sdpa` | 100 |
| `b1-flash` | 1 | `flash_attention_2` | 400 |
| `b4-flash` | 4 | `flash_attention_2` | 100 |

四组都重新加载数据准备阶段固定 commit 的 `Qwen/Qwen3-4B-Base`，使用同一份 400 条训练数据、完整文本和同一乱序；每条恰好更新一次。不按长度排序、不 packing、不截断、不跳过长样本。Batch 4 右侧填充到该批最长序列，填充和提示词不参与监督。助手的完整推理、最终答案和结束符继续参与监督。

使用与正式训练相同的 LoRA（默认冻结不量化的 BF16 Base，rank 16 / alpha 32 / dropout 0），梯度检查点、AdamW8bit、梯度裁剪和分块 LM-head loss。仅 adapter 参数参与更新，学习率、100 次更新预热等设置来自 `configs/baseline.json`；可用 `--config configs/qlora_nf4.json` 在独立目录测试 NF4 QLoRA。Batch 4 的预热覆盖整个 100 步 epoch；它与 batch 1 的训练轨迹不同。Loss 是每批所有助手目标 token 的平均值，因此批内长回答贡献更多 token。这个实验比较速度，不用于断言相同训练效果。目标 GPU 为 RTX A6000 48GB，实际 batch 4 显存峰值仍需测试；OOM 会记录失败，不改变 batch 或丢弃长题。

## 在 VM 上准备依赖

使用已安装本工程依赖的虚拟环境；先暂停同一张 GPU 上的正式训练和评测，避免争用显存及算力。脚本不修改 `train.py`、正式配置、正式运行或其恢复点。

```bash
cd /home/ubuntu/data_recipe/Repeat_SFT
source .venv/bin/activate
git pull
export REPEAT_SFT_ROOT=/data/repeat_sft
mountpoint -q /data
export HF_HOME="$REPEAT_SFT_ROOT/.cache/huggingface"
export TMPDIR="$REPEAT_SFT_ROOT/.cache/tmp"
export PIP_CACHE_DIR="$REPEAT_SFT_ROOT/.cache/pip"
mkdir -p "$HF_HOME" "$TMPDIR" "$PIP_CACHE_DIR"
export CUDA_VISIBLE_DEVICES=0
```

如果还没安装 Flash Attention 2，在 **VM** 安装可与现有 PyTorch / CUDA 配合的 `flash-attn`。不必在本地安装。下面采用 [FlashAttention 官方的安装方式](https://github.com/Dao-AILab/flash-attention#installation-and-features)，将版本限制为 2.x，并避免安装器更换已有 PyTorch：

```bash
python -m pip install packaging psutil ninja einops
MAX_JOBS=4 python -m pip install 'flash-attn>=2.6,<3' --no-build-isolation --no-deps
python -c 'import torch, flash_attn; print("torch", torch.__version__, "CUDA", torch.version.cuda, "flash_attn", flash_attn.__version__)'
```

若没有匹配的预编译 wheel，安装会编译 CUDA 扩展，需要 CUDA toolkit（`nvcc --version` 可检查）；只有显卡驱动不够。脚本会记录实际依赖版本，不会自动安装软件。Flash 两组明确请求 FA2，缺依赖或不兼容会记录失败，不会悄悄切回 SDPA。确定性设置沿用训练配置；在 `deterministic=true` 时，额外设置 `FLASH_ATTENTION_DETERMINISTIC=1`，与 [Transformers 4.57.1 的 FA2 接口](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/modeling_flash_attention_utils.py) 保持一致。

## 运行

```bash
python -u benchmark_train.py \
  --data artifacts/data-s42 \
  --out runs/benchmark-lora-s42 \
  --seed 42
```

结果默认写到 `/data/repeat_sft/runs/benchmark-lora-s42`。必须使用新目录；同名目录不会被覆盖。四组在屏幕上分别显示样本进度、更新次数、loss、已用时间和 ETA，完成后打印四行汇总。

后台运行：

```bash
mkdir -p /data/repeat_sft/logs
log_file="/data/repeat_sft/logs/benchmark-$(date +%Y%m%d-%H%M%S)-$$.log"
nohup python -u benchmark_train.py \
  --data artifacts/data-s42 \
  --out runs/benchmark-lora-s42 \
  --seed 42 \
  >> "$log_file" 2>&1 < /dev/null &
echo "PID=$! 日志=$log_file"
tail -f "$log_file"
```

默认保存测速结果而不保存 adapter。若需要之后加载这四个模型推理，在首次命令上添加 `--save-weights`；每组保存最终 LoRA adapter、Base commit/量化配置和 tokenizer，不重复保存完整 Base，保存耗时单独记录。这里不保存优化器恢复状态。推理须加载固定 Base 再加载 adapter，可使用 `recipe.modeling.load_model(权重目录)`。

## 查看结果

```bash
column -s, -t < /data/repeat_sft/runs/benchmark-lora-s42/summary.csv
```

也可以下载 `summary.csv` 用表格软件查看。详细定义：

| 字段 | 含义 |
| --- | --- |
| `status` | `complete` 才表示这组完成；失败原因在 `error` 和该次 `console.log` |
| `epoch_wall_seconds` | 全部 400 条训练一遍的墙钟时间，含逐步日志，不含加载、保存权重、内核探测 |
| `speedup_vs_b1_sdpa` | batch 1 + SDPA 的整轮耗时 ÷ 本组整轮耗时；2 表示这一轮约快一倍 |
| `nonpadding_tokens_per_second` | 排除相同前 16 条样本后，有效输入 token 数 ÷ 累计更新耗时；不含 padding 和日志写入时间 |
| `padding_fraction` | 一轮中填充位置占批量输入总位置的比例 |
| `peak_allocated_gib` | 训练期间 PyTorch 已分配显存峰值，含梯度和已初始化的优化器状态 |
| `peak_reserved_gib` | 训练期间 PyTorch 内存分配器保留显存峰值，含缓存；与 `nvidia-smi` 全设备用量不同 |
| `model_load_seconds` / `save_weights_seconds` | 单独统计的模型加载 / 权重保存耗时；未请求保存时后者为 0 |
| `probe_status` / `observed_backends` | 额外探测的状态和 attention 实现线索；不能仅凭 `sdpa` 这个配置名判断未使用 Flash |

每次更新前后同步 CUDA，更新计时包含 padding、CPU→GPU 传输、前向、反向、裁剪和优化器更新。整轮计时保留初始化开销，吞吐量指标默认排除相同前 16 条样本（batch 1 为 16 步，batch 4 为 4 步）。这不是额外训练预热：400 条中的每条仍只更新一次。可用 `--warmup-samples 32` 改变排除范围，必须是 4 的倍数且小于 400；它不改变学习率预热。

训练结束后，默认对第一批的形状额外做一次有 profiler 的前向和反向，不做优化器更新。这段时间和显存不计入训练指标；保留算子名称用于核对实际内核。**探测只代表该批形状**，其他序列长度可能使用不同内核。SDPA 允许 PyTorch 自动选内核，可能本身已经使用 Flash；FA2 组显式调用外部 `flash-attn`。Profiler 不可用或失败时保留有效训练结果并标明探测失败。可用 `--no-kernel-probe` 关闭额外探测。

`summary.json` 还保存所有更新的合计、排除前 16 条后的指标及各组详细状态。原始记录结构：

```text
benchmark-lora-s42/
  benchmark.json, base_config.json     # 实际协议、seed、版本和哈希
  data/, tokenizer/, order.json       # 400 条完整文本、模板及固定顺序
  code/                              # 当次代码与 Git 状态快照
  summary.csv, summary.json           # 每组完成/失败后更新的汇总
  b1-sdpa/attempt-.../
    config.json, environment.json, hardware.json, command.json
    inspection.jsonl                 # 实际输入与监督范围
    steps.jsonl, progress.json       # 每批样本 ID、loss、LR、长度、时间
    training_result.json             # 一轮结束立即保存，后续探测失败也保留
    attention_probe.json, console.log, result.json
    weights/                         # 仅 --save-weights
  b4-sdpa/attempt-.../
  b1-flash/attempt-.../
  b4-flash/attempt-.../
```

## 失败和中断后继续

某组 OOM 或依赖报错后，其余组仍继续测试；不会自动缩小 batch 或丢弃长题。只要有一组没完成，主命令返回非零退出码，并保留已完成结果。Ctrl+C 会停止当前子进程和整个测试队列。

修复依赖或重新启动后：

```bash
python -u benchmark_train.py --out runs/benchmark-lora-s42 --resume
```

它验证并使用本次测试冻结的代码、配置和数据，跳过已经完成的组合；未完成组合从同一个 Base 重跑这一轮，另建 `attempt-*` 保留旧日志。**测速没有轮内断点恢复**，避免把跨进程的加载和计时混成一轮。既有正式训练的完整断点恢复机制保持原样。

比较时优先看 `epoch_wall_seconds`、有效 token/s 和显存。当前四组各测一次，GPU 温度、频率和其他进程会造成波动；需要确认小幅差异时使用新的输出目录重复测量。改变 batch 会同时改变更新次数，不能由这些计时推导最终模型质量。
