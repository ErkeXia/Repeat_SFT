# Base 独立解答三组 Math-CoT-20k 训练题

入口：`run_training_base_vllm.sh` / `solve_training_base.py`。直接使用 **Qwen/Qwen3-4B-Base** 的固定 revision，BF16，**不加载任何 LoRA adapter**。输入只有训练时保存的原题及提示词；数据集参考解答、短答案仅供保存和事后判分，不输入模型。

这与 `evaluate_loss.py` / `dataset/score_likelihood.py` 的 teacher forcing 不同：本实验让 Base 从题目独立生成，再核对最终答案。测的是所选训练题的初始解题表现；不替代四个 benchmark 的泛化评测，也不代表一次判错就说明模型完全不会这题。

## 数据与设置

| 组 | 默认读取的冻结训练文件 | 题数 |
| --- | --- | ---: |
| random | `runs/random-lora-s42/data/train.jsonl` | 400 |
| high | `runs/likelihood-100x16-s42/high/data/train.jsonl` | 100 |
| low | `runs/likelihood-100x16-s42/low/data/train.jsonl` | 100 |

读取**实际用于训练的名单**，不重新抽样、重排似然度或下载整个数据集。不加入 holdout、人工改写的 10 条或 high 5 条。核对题数、样本身份、输入哈希、Base revision 和 tokenizer 一致性；三组间若有相同的标准化题目和完全相同的提示词，只生成一次，保存全部组归属，分别计入相应组的结果。因此共有 600 个组成员，实际生成可能少于 600 次。相同题目但提示词不同会分开生成。

默认沿用 `configs/generation.json` 和 `configs/vllm.json`：

- vLLM 0.11.0，BF16，并发 8，Base 只加载一次。
- 每题一次采样；temperature=0.6、top-p=0.95、top-k=20、repetition_penalty=1.0。
- 最多 8192 个新 token；到上限直接停止并标记截断，不补造结束符。
- generation seed=314159，再按题目及提示词身份生成逐题 seed。
- 原提示词按冻结 tokenizer 的聊天模板编码；不加参考解答，不截断题目。
- EOS 与 `<|im_end|>` 都可结束；分别记录 `ended_with_eos` 和 `normal_chat_end`。

所有新结果默认在 volume：`/data/repeat_sft/runs/math-cot-training-base-s42`。只保存输入快照、生成和报告，不复制模型权重或训练 checkpoint；共用现有 HF 模型缓存。`/data` 未挂载时会停止。

## VM 上运行

先 `git pull`，并使用已安装的独立 vLLM 环境（见 [VLLM_EVALUATION.md](VLLM_EVALUATION.md)）。包装脚本默认使用 `/data/repeat_sft/venvs/vllm-0.11.0/bin/python`；可用 `VLLM_PYTHON_BIN` 指定同版本环境，不必切换当前训练 venv。没有 Docker/LiveCodeBench 依赖。

可先核对数据，不加载 GPU 模型：

```bash
bash run_training_base_vllm.sh \
  --random-run runs/random-lora-s42 \
  --experiment runs/likelihood-100x16-s42 \
  --out runs/math-cot-training-base-s42 \
  --prepare-only
```

正式后台执行，生成结束、释放推理进程后自动判分并生成报告：

```bash
mkdir -p /data/repeat_sft/logs
nohup bash run_training_base_vllm.sh \
  --random-run runs/random-lora-s42 \
  --experiment runs/likelihood-100x16-s42 \
  --out runs/math-cot-training-base-s42 \
  >> /data/repeat_sft/logs/math-cot-training-base-s42.log 2>&1 < /dev/null &
echo "Base 解题 PID=$!"
tail -n 30 -f /data/repeat_sft/logs/math-cot-training-base-s42.log
```

`tail -f` 中按 Ctrl+C 只退出看日志。停止后台任务使用刚才打印的 PID：`kill -TERM <PID>`，等待进程退出后重跑**同一条启动命令**即可恢复。已有完整回答逐题写入并 fsync；最多 8 条正在生成但尚未完成的请求会重跑，不恢复单题 KV cache。判分也逐题恢复。锁会拒绝同一输出目录的第二个进程。

首次启动冻结配置、完整题目、参考解答、tokenizer、源 run 元数据和代码。恢复使用冻结代码；更新仓库不会改变这次生成设置。改变生成配置请使用新 `--out`，并可传 `--config` / `--engine-config`。已有训练、拟合、benchmark 结果均不改写。

## 查看结果

主要阅读文件：

```text
/data/repeat_sft/runs/math-cot-training-base-s42/report/report.html
```

包含三组汇总，以及按 random/high/low 分段的逐题完整内容：来源、原始行号、原训练 sample_id、组归属、题目、正确答案、提取答案、判分异常、可展开的原参考解答和最后呈现的 Base 完整回答。HTML 可单独下载阅读；CSV/JSONL 链接需一起下载对应文件。

```text
setup/plan.json                        数据/模型版本、组规模、重合数量及选题规则
setup/training/{random,high,low}/       实际训练名单及完整文本、源 run 元数据
setup/panel.jsonl                      合并后的题目与全部原始记录
setup/generation.json                  本次实际生成参数
setup/engine.json                      本次实际 vLLM 参数
setup/code/                            本次冻结代码和 Git 信息
generation/epoch-000/responses.jsonl    原始回答、输入/输出 token IDs、停止原因、seed、耗时
generation/epoch-000/progress.json      已完成题数 / 合并后总题数
grades/graded.jsonl                    合并后的逐题判分、完整文本及来源对应
report/summary.json                    三组正确率、长度、结束/截断/提取率、判分异常数
report/summary.csv                     可直接做图的三组指标
report/{random,high,low}.jsonl          各组的完整题目、解答、生成与判分记录
report/report.html                     汇总与逐题阅读报告
```

判分沿用现有数学规则：取结束的思考段之后最后一个完整 `boxed`，没有时尝试答案标记，再用 math-verify 比较短答案。未闭合 `<think>` 不提取。截断且有可提取答案仍可能判对；判分异常单独记录，并保留在正确率分母。一次采样的正确率不是 pass@k，也不能直接等同“掌握率”。

想在中途查看报告：先停止生成、等待退出，再执行以下命令（仅 CPU 判分，不加载模型）。未完成题明确列出，不记作错误；部分正确率仅针对已经生成的题。

```bash
bash run_training_base_vllm.sh \
  --random-run runs/random-lora-s42 \
  --experiment runs/likelihood-100x16-s42 \
  --out runs/math-cot-training-base-s42 \
  --stage grade --allow-partial
```

之后重跑原启动命令继续生成，最终报告会更新为完整 400/100/100 的结果，已保存的逐题回答和判分保留。生成与判分也可分别用 `--stage generate`、`--stage grade` 启动。

以后修改答案抽取规则，可直接用当前版本通用判分器读取保存的回答，输出到一个新目录，无需重新推理：

```bash
/data/repeat_sft/venvs/vllm-0.11.0/bin/python -B grade.py \
  --responses runs/math-cot-training-base-s42/generation/epoch-000/responses.jsonl \
  --out runs/math-cot-training-base-s42/regraded-v2
```

该通用复判命令保存新判分 JSONL，不覆盖原报告。固定任务内的 `--stage grade` 使用首次冻结的规则。

## 本地验证范围

离线测试覆盖输入一致性、三组重合、提示词不含参考解答、禁用 LoRA、并发与长度参数、模拟中断后续跑、残缺尾行恢复、部分/完整报告、HTML 转义和结果篡改拒绝。本地不下载模型、不运行真实 GPU 推理；实际速度和显存需在 VM 验证。
