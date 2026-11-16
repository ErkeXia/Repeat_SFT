# Math-CoT-20k 全集在 Base 上的解答似然

用 VM 的 GPU 对 **20,480 条原始解答**做 teacher forcing；模型为未加载 LoRA 的 `Qwen/Qwen3-4B-Base`，BF16，vLLM 0.11.0，默认并发 8 条。只前向计算，不训练，不要求模型重新生成解答。

## 计算什么

每个解答 token 都以题目和**原解答已有前文**为条件：

```text
sum_logprob            = Σ log p(y_t | prompt, y_<t)
mean_logprob           = sum_logprob / n_tokens
mean_nll               = -mean_logprob
normalized_likelihood  = exp(mean_logprob) = 1 / perplexity
```

主报告按 `mean_logprob` 从高到低排序，即按长度归一化似然从高到低；同分按原始 `row_index` 升序。**“前 100 / 后 100”是排名最高 / 最低各 100 条，不是原文件前后 100 行**。各组平均值为逐样本的 `normalized_likelihood` 算术平均，不是把所有 token 混在一起平均。额外保存平均 NLL、token 加权 NLL、整段 log-likelihood 和整段概率的 log10；整段概率均值也用 log10 保存，避免长解答全部数值下溢成 0。

- 聊天模板、提示词和监督边界复用现有 SFT 编码器；包含完整推理、最终答案和 `<|im_end|>`，排除用户提示词与结束后的换行。
- 使用数据的 `response` 长解答；`answer` 是短参考答案，只供阅读，不拿它替换长解答。
- 不应用训练的 10,240-token 过滤，也不应用自由生成的 8,192-token 上限。不去重、不抽样、不排除 holdout 或评测重复题。正式任务须为全部 20,480 行。
- 先检查所有样本能放入真实模型上下文，为 vLLM 的一个额外输出 token 留出位置；超长会明确报错，绝不截断或跳过。
- vLLM 的 `prompt_logprobs` 返回每个已有 token 的条件 log-probability。请求只额外生成 1 个 token 来结束，**这个 token 不参与评分、不作为解答保存**。禁用 prefix caching，开启 chunked prefill，每次最多 1,024 个 prefill/decode token，降低词表 logits 的显存峰值。
- 这个指标衡量模型模仿给定解答的容易程度，不能直接解释为解题正确率、数据质量或训练后的收益。

接口依据：[vLLM 0.11.0 sampling parameters](https://docs.vllm.ai/en/v0.11.0/api/vllm/sampling_params.html)、[该版本 prompt logprobs 实现](https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/v1/worker/gpu_model_runner.py)。

## VM 上运行

在仓库根目录执行。先 `git pull`，使用已有的独立 vLLM 环境；不要把这些依赖安装到训练环境。下面的环境路径如与你的 VM 不同，替换为实际路径。

```bash
VLLM_PYTHON=/data/repeat_sft/venvs/vllm-0.11.0/bin/python
"$VLLM_PYTHON" -m pip install -r dataset/requirements-likelihood.txt
"$VLLM_PYTHON" -m pip check
mkdir -p /data/repeat_sft/logs
```

相对数据和结果路径解析到 `/data/repeat_sft`，缓存也默认写到该 volume；`/data` 未挂载会停止。如果主动设置了 `HF_HOME` 或 `TMPDIR`，脚本尊重这些设置，请确认它们指向数据盘。

原始 Parquet 被 Git 忽略。`--download-data` 只在 VM 上文件不存在时下载**固定版本**的约 158 MiB 文件；也可 SFTP 上传本地 `dataset/math-cot-20k/raw/Math-CoT-20k.parquet` 到 `/data/repeat_sft/dataset/math-cot-20k/raw/Math-CoT-20k.parquet`，然后省略该参数。

先跑 4 条 smoke（独立目录）：

```bash
"$VLLM_PYTHON" -u dataset/score_likelihood.py \
  --base-run runs/random-lora-s42 \
  --download-data --limit 4 \
  --out dataset/math-cot-20k/likelihood/base-smoke-v1
```

`--base-run` 只复用旧实验的 **Base revision 和 tokenizer**，不会读取已训练的 LoRA 权重。这样与之前实验的初始模型保持一致。不传时使用 `Qwen/Qwen3-4B-Base` 的 main，并在第一次运行时解析、锁定具体 commit；恢复时不再重新解析 main。

成功后启动全集：

```bash
nohup "$VLLM_PYTHON" -u dataset/score_likelihood.py \
  --base-run runs/random-lora-s42 \
  --download-data \
  --out dataset/math-cot-20k/likelihood/base-v1 \
  >> /data/repeat_sft/logs/math-cot-base-likelihood-v1.log 2>&1 < /dev/null &
echo "似然评分 PID=$!"
tail -n 30 -f /data/repeat_sft/logs/math-cot-base-likelihood-v1.log
```

没有额外的训练或评测进程占 GPU 时运行。日志先显示 CPU 编码检查 `PREPARE .../20480`，再显示逐条评分 `SCORE .../20480`、原始行号、token 数、NLL、归一化似然和估算剩余时间。异步评分的行号不一定按顺序完成。

若只想先检查编码，在**同一条正式命令**中加 `--prepare-only`，检查结果目录中的 `inspection.jsonl`；随后移除该选项运行。该阶段只获取配置/tokenizer，不加载模型权重。

## 中断、恢复和部分报告

对评分进程 `kill -TERM <PID>`，或前台运行时 Ctrl+C。每条完成后都保存并 fsync；正在计算的条目在恢复时重做。`tail -f` 中 Ctrl+C 只结束查看日志。

**恢复时重新运行完全相同的评分命令**即可，不需要 `--resume`。已提交的逐题结果不会覆盖。代码、软件版本、模型/数据版本、模板或关键参数不同会拒绝混写，需使用原配置继续，或另选输出目录。每个任务有独占文件锁，不能在同一目录同时启动两个评分进程。

完成后自动生成报告。也可以在停止评分后，用 CPU 重建报告，无需加载模型：

```bash
"$VLLM_PYTHON" -B dataset/report_likelihood.py \
  --out dataset/math-cot-20k/likelihood/base-v1

# 中途查看：明确允许部分结果；报告会标明尚未覆盖全集。
"$VLLM_PYTHON" -B dataset/report_likelihood.py \
  --out dataset/math-cot-20k/likelihood/base-v1 --allow-partial
```

smoke 的 4 条会显示“部分数据 / smoke”；前后组不足 100 条时会重叠。正式全量报告必须 `complete_corpus: true`，`scored_rows: 20480`。最终报告会替换此前可重建的部分报告，但原始逐样本分数不会被替换。

## 保存在哪里、怎么看

默认结果在 `/data/repeat_sft/dataset/math-cot-20k/likelihood/base-v1/`：

| 文件 | 内容 |
| --- | --- |
| `report/report.html` | 单文件可离线阅读；全部平均、最高/最低 100 平均、最高/最低值，及 200 条完整题目、答案、提示词和原解答 |
| `report/summary.json` | 机器可读汇总，含是否完成全集 |
| `report/scores.csv`、`scores.jsonl` | 全部样本分数、排名、来源、原始行号、IDs、长度、token 文件路径 |
| `report/top100.jsonl`、`bottom100.jsonl` | 入选样本的完整题目、短参考答案、长推理回答和分数 |
| `source.parquet` | 本次全量原始文本快照，SHA256 校验；以后无需重新下载即可重做阅读导出 |
| `records/row-XXXXX.json` | 每个原始行的已提交分数，也是断点恢复的依据 |
| `tokens/row-XXXXX.npz` | 完整聊天输入 token IDs、助手起止位置，以及所有助手 token 的 log-probability |
| `job.json`、`environment.json`、`code/` | 实际配置、版本、精度、代码哈希与快照 |
| `tokenizer/`、`base_config.json` | 冻结的 tokenizer/模板和模型配置 |
| `index.jsonl`、`inspection.jsonl` | 全集编码描述和前 3 条实际监督范围检查 |
| `progress.jsonl`、`engine_startups.jsonl` | 逐条完成记录、引擎配置、GPU 和启动记录 |

只下载 `report.html` 就能阅读全部 200 条原文；需要表格/JSON 下载链接也可用时，下载整个 `report/` 文件夹。所有 20,480 条的全文在 `source.parquet`，分数通过 `row_index` 关联。不是只保存两端的 200 条。

逐 token 复查示例（CPU）：

```python
import numpy as np
from transformers import AutoTokenizer
from pathlib import Path
out = Path('/data/repeat_sft/dataset/math-cot-20k/likelihood/base-v1')
tok = AutoTokenizer.from_pretrained(out / 'tokenizer', local_files_only=True)
with np.load(out / 'tokens/row-00000.npz', allow_pickle=False) as x:
    start, end = int(x['assistant_start']), int(x['assistant_end'])
    for token, logp in zip(x['input_ids'][start:end+1], x['logprobs']):
        print(int(token), repr(tok.decode([int(token)], skip_special_tokens=False)), float(logp))
```

## 本地测试与边界

```bash
python -B -m unittest discover -s dataset/tests -v
```

测试使用合成 tokenizer、随机/固定 CPU logits 和小型 Parquet，覆盖因果 token 对齐与交叉熵对照、屏蔽范围、长解答保留、无效 logprobs、逐题落盘中断与恢复、重复题身份、数据完整性、205 条排序/均值与 200 条完整原文导出。不会下载模型或调用 GPU。

本地不具备 Linux vLLM/CUDA 环境，因此真实引擎、显存和速度需通过上面的 VM smoke 验证。本次没有在本地下载模型或运行全集 GPU 评分。
