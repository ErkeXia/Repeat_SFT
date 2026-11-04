# vLLM 自由生成：并发 8 道题

适用于已经完成的 **BF16 LoRA** run。复用固定版本的 Qwen3-4B-Base 和五个已保存 adapter，不重新训练、不合并成五份完整模型。本入口暂不支持 NF4 QLoRA 或旧全参 run；这些实验仍使用 Transformers 入口。

## 本次协议

| 项目 | 配置 |
| --- | --- |
| 后端与环境 | vLLM **0.11.0**，PyTorch **2.8.0**，Transformers **4.57.1**；独立虚拟环境 |
| 并发 | **最多 8 道题同时进行**；应用请求队列和引擎 `max_num_seqs` 双重限制 |
| 模型精度 | Base BF16、LoRA 计算 BF16；原 adapter 文件保持不变 |
| 数据 | 已冻结的 128 MATH500＋64 GPQA；200 holdout 仍只看拟合 |
| 分组与顺序 | 四组各 32 MATH500＋16 GPQA；每组按 **64→32→16→8→4**，共 960 条回答 |
| 模型加载 | 每个组内 Base 只初始化一次；切换 LoRA，每个 checkpoint 使用独立 adapter ID |
| 生成 | 每题一次；temperature=0.6、top-p=0.95、top-k=20、repetition penalty=1 |
| 长度 | 最多 **8192 新 token**；不截断题目；上下文预算由全部 192 题的最长 prompt＋8192 决定 |
| 停止 | `<|im_end|>`、Base EOS 或长度上限；不强行补 EOS |
| 显存 | `gpu_memory_utilization=0.85`，chunked prefill，`max_num_batched_tokens=4096` |
| 缓存 | 禁用 prefix caching；缓存和编译产物默认保存在 `/data/repeat_sft/.cache` |

并发是上限，最后剩少于 8 题时会自然降低。每题使用自己的固定 seed，提示词先由 run 内冻结 tokenizer 编成 token IDs，再交给 vLLM；不会重新套用别的聊天模板。组内完成顺序可能不同，但题目身份与最终汇总顺序固定。

现有 PEFT 生成使用 FP32 adapter 计算，vLLM 此处明确使用 BF16 adapter。因此推理数值实现不同；即便 seed 相同，也不保证与 Transformers 或不同调度下逐 token 一致。**五个节点统一使用同一 vLLM 协议，另建任务目录**。程序记录后端、版本、Base revision、adapter 哈希、引擎参数、输入和原始输出；分析会拒绝混合不同后端。已有 teacher-forcing 结果继续保留并复用。

## 1. 在 VM 安装独立环境

本地修改由你提交并 push，在 VM pull。不要把 vLLM 装进训练用的 `.venv`，也不要在 vLLM 环境安装训练用 `requirements.txt`。vLLM 预编译 CUDA/PyTorch 需要兼容的 NVIDIA 驱动；此版本依赖 torch 2.8.0，不能沿用原来 torch 2.6.0/cu124 的安装命令。驱动不兼容时先修复环境，不改成其他精度来绕过。

```bash
cd /home/ubuntu/Repeat_SFT
git pull
nvidia-smi
mountpoint -q /data || { echo '/data 未挂载'; exit 1; }
mkdir -p /data/repeat_sft/venvs /data/repeat_sft/.cache/pip /data/repeat_sft/.cache/tmp /data/repeat_sft/logs

export PIP_CACHE_DIR=/data/repeat_sft/.cache/pip
export TMPDIR=/data/repeat_sft/.cache/tmp
python3 -m venv /data/repeat_sft/venvs/vllm-0.11.0
/data/repeat_sft/venvs/vllm-0.11.0/bin/python -m pip install --upgrade pip
/data/repeat_sft/venvs/vllm-0.11.0/bin/python -m pip install -r requirements-vllm.txt
/data/repeat_sft/venvs/vllm-0.11.0/bin/python -m pip check
/data/repeat_sft/venvs/vllm-0.11.0/bin/python -B -c 'from generate_vllm import configure_vllm_runtime, runtime_identity; configure_vllm_runtime(); print(runtime_identity())'
```

最后一条检查包版本、CUDA、BF16 和 GPU，不加载模型。模型运行时复用 `/data/repeat_sft/.cache/huggingface` 中的 Base；若尚未缓存该固定 revision，下载只发生在 VM。没有本地模型或完整数据下载。

默认 launcher 使用上述 volume 中的解释器。自定义环境时可设置 `VLLM_PYTHON_BIN=/你的环境/bin/python`；整个生成、判分与报告流程使用同一解释器。

## 2. 后台启动

先完成八个节点的拟合评测。下面假设此前重复记录的修复已显示 `SUCCESS`，因此显式使用 `teacher_forcing-dedup-v1`；若你使用其他已校验目录，只替换这一参数。

```bash
nohup bash run_grouped_vllm.sh runs/random-lora-s42 grouped-e4plus-vllm-v1 \
  --epochs 4,8,16,32,64 \
  --teacher-forcing runs/random-lora-s42/evaluations/teacher_forcing-dedup-v1 \
  >> /data/repeat_sft/logs/random-lora-s42.vllm.log 2>&1 < /dev/null &
echo "$!" > /data/repeat_sft/logs/random-lora-s42.vllm.pid
tail -f /data/repeat_sft/logs/random-lora-s42.vllm.log
```

开始时会有模型加载、显存规划与 CUDA 图编译日志，可能暂时没有逐题输出。随后每道题完成会显示：

```text
group=1/4 epoch=64 1/48 tokens=1234 stop=chat_end backend=vllm concurrency=8
```

每组五个节点全部完成后，自动判分并保存该组报告，再进入下一组。不会重复计算拟合 loss，也不运行并发 8 对 16 的对比实验。

## 3. 停止与续跑

```bash
kill -TERM "$(cat /data/repeat_sft/logs/random-lora-s42.vllm.pid)"
```

等待任务退出后，重新执行同一条 `nohup` 和 PID 保存命令即可。**不用 `--resume`，不要加 `--reset-generation`**。

- 每道题完成立即追加到 JSONL 并 fsync，不等待其余 47 题。
- 已完成回答保留原样；最多 8 道正在进行而未保存完整结果的题，会重新生成。不保存单题内 KV cache，不保证重跑的未完成题逐 token 相同。
- 最后半行会先备份为 `.torn-*` 再恢复；完整记录保留。
- 同一任务和同一生成目录有写入锁，避免重复启动导致重复结果。只杀控制器的 `kill -9` 可能留下子进程；应优先用 SIGTERM，并确认旧任务退出后再启动。
- 错误/OOM 会停止当前阶段，保留已保存题，不自动降并发、截断、删题或把错误当答错。
- 恢复使用 `grouped_code` 冻结代码与 `vllm_config.json`；更换后端、引擎设置或已记录的运行环境会被拒绝，应另建任务。
- 判分或报告失败可以重试，已完成生成不重跑，报告可能落在 `*-retry-*` 目录。

## 4. 查看结果

```text
/data/repeat_sft/runs/random-lora-s42/postprocess/grouped-e4plus-vllm-v1/
  grouped_pipeline.json, vllm_config.json
  groups/group-001/
    generation/epoch-064/
      responses.jsonl          # 完整问题、prompt IDs、生成 IDs、原回答、seed、停止原因
      identity.json            # Base/adapter 身份、vLLM 版本、精度、并发与生成参数
      generation_attempts.jsonl # 本次节点执行的墙钟耗时、完成题数与生成 token 总数
      engine_startups.jsonl    # 共享引擎的初始化耗时
      complete.json
    grades/
    analysis/report.html
    complete.json              # 成功判分和报告的实际路径
  groups/group-002/...
  groups/group-003/...
  groups/group-004/...
  generation/                  # 合并后的 192 题 × 5 节点
  grades/
  analysis/report.html
  grouped_complete.json
```

问题级 `generation_seconds` 是请求提交至保存前的耗时，各请求时间有重叠，不能相加当整体耗时；节点整体吞吐看 `generation_attempts.jsonl`，初始化时间另列。这里是执行日志，没有增加测速对比实验。

下载整个成功的 `analysis` 目录，用浏览器打开 `report.html`。旧 Transformers 的生成目录仍可独立查看，但不能与 vLLM 混成同一训练曲线。仅从 epoch 4 开始观测，不能把首次观测答对解释成恰好第 4 轮才学会；没有 Base 生成对照时不报告相对 Base 的提升。

需要检查具体长回答时，可另运行 [report_examples.py](EXAMPLE_REPORT.md)，生成五个 epoch、四类截断/判分情况的案例报告；跨 epoch 尽量复用同一道题。它只读已保存判分，不影响正在生成的任务。

## 实现与验证范围

入口是 `generate_vllm.py`、`run_grouped_vllm.sh`；请求调度、停止校验和续跑在 `recipe/vllm_generation.py`。引擎配置是 `configs/vllm.json`，分组控制器通过 `--backend vllm` 选择。

离线测试用模拟引擎验证并发上限 8、逐题立即写盘、错误/取消、半行恢复、Base 复用、adapter ID、身份校验、分组与报告。它们不测真实 vLLM CUDA 内核或吞吐；实际显存与速度要以 VM 运行日志为准。

API 对照固定在 [vLLM v0.11.0 AsyncLLM 源码](https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/v1/engine/async_llm.py)、[LoRA 配置](https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/config/lora.py)和 [CUDA 依赖](https://github.com/vllm-project/vllm/blob/v0.11.0/requirements/cuda.txt)。不要直接升级正在运行任务的 vLLM 版本。
