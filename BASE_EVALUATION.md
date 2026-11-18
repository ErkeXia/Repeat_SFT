# 原始 Base：四个 benchmark 的 vLLM 基线

入口：`run_base_vllm.sh`。只评测 **Qwen/Qwen3-4B-Base、epoch 0**，使用原 run 固定的 Base revision 和 tokenizer，**vLLM 的 LoRA 开关关闭，每个请求不携带 adapter**。不重新训练，也不分别为 high/low 重复生成同一份 Base 对照。

默认 `--run runs/random-lora-s42` 仅提供模型版本、tokenizer、epoch 0 来源记录及固定面板。需要该 run 的 `run.json`、`data/`、`tokenizer/` 和完整 `checkpoints/epoch-000/`；不要求其他轮次的 checkpoint 或 teacher-forcing 结果。epoch 0 的 adapter 文件只做来源完整性校验，不加载到推理引擎。不要指向 intermediate replay run。

| 面板 | 题目 | 执行顺序 |
| --- | --- | --- |
| MATH500 / GPQA | 原固定 128 道 MATH500 + 64 道 GPQA | 先完成四组，每组 32 + 16 = 48 题 |
| BBH / LiveCodeBench | 原 `artifacts/transfer-s2026` 中各 100 道 | 再完成四组，每组 25 + 25 = 50 题 |

每组完成生成、判分和报告后才进入下一组，共 **392 条回答**。逐题固定 seed、提示词、GPQA 选项和分组沿用已有协议，不重新抽题。BF16；vLLM 0.11.0；并发 8；temperature=0.6、top-p=0.95、top-k=20、repetition penalty=1；最多 **8,192 个新 token**。遇到 EOS 或长度上限停止，保留真实停止原因。vLLM 调度和禁用 LoRA 的计算路径可能产生数值差异，不承诺与旧零增量 adapter 路径逐 token 一致。

## 在 VM 上启动

先推送本地改动，在 VM 工程根目录 `git pull`。沿用现有独立 vLLM 环境和已通过自检的 Docker 镜像，无新增依赖。等其他 GPU 训练/评测结束或正常暂停后再启动。

```bash
cd ~/Repeat_SFT
git pull
mountpoint -q /data
mkdir -p /data/repeat_sft/logs

nohup bash run_base_vllm.sh \
  --run runs/random-lora-s42 \
  --panel-dir artifacts/transfer-s2026 \
  --job-name base-four-benchmarks-vllm-v1 \
  >> /data/repeat_sft/logs/base-four-benchmarks-vllm-v1.log 2>&1 < /dev/null &
echo "Base 评测 PID=$!"
tail -n 30 -f /data/repeat_sft/logs/base-four-benchmarks-vllm-v1.log
```

脚本自动调用 `/data/repeat_sft/venvs/vllm-0.11.0/bin/python`，无需切换当前激活的 `.venv`。环境在其他位置时设置 `VLLM_PYTHON_BIN=/实际路径/bin/python`。

首次启动会先检查两个面板、math-verify 和 Docker 自检，再加载 GPU 模型。可在上面命令中追加 `--prepare-only` 只完成准备检查；之后原样去掉该选项即可开始生成。LiveCodeBench 仍需原来的 `repeat-sft-lcb:py311-v1` 镜像，容器没有修改，不必重新构建已通过自检的镜像。

## 暂停与恢复

`tail -f` 中 Ctrl+C 只退出日志查看。停止评测用 `kill -TERM <上面输出的评测PID>`，等子进程退出后再使用 GPU。恢复时**原样重跑同一条 nohup 命令**，使用相同 `--run`、`--panel-dir`、`--job-name`。

每个完整回答立即写盘；已完成回答和判分跳过，当前未完成的最多 8 题重新生成，不恢复单题 KV cache。完整组保留并校验文件哈希。同一任务有进程锁。首次启动冻结代码、输入、生成参数和容器身份，后续 git pull 不会把新代码混入既有任务。没有 reset/覆盖旧生成的默认操作；需要另一套协议时另用 job-name。

## 结果在哪里

```text
/data/repeat_sft/runs/random-lora-s42/postprocess/base-four-benchmarks-vllm-v1/
  report.html                 总入口、四任务指标表、已完成组报告链接
  curves.csv                  已完整完成面板的正确率、长度、结束/截断/提取率
  summary.json                Base 身份、完成面板和同一组汇总指标
  progress.json               当前面板及其详细进度文件位置
  complete.json               两个面板全部生成并判分完毕时才存在
  suite/                      固定计划、代码版本及判分配置
  math-gpqa/
  bbh-livecodebench/
```

每个面板内：

- `setup/panel/`：完整原题、提示词、参考答案；LCB 的测试数据也保留。
- `groups/group-001/` 至 `group-004/`：分组结果与分组报告。
- `groups/group-001/generation/epoch-000/responses.jsonl`：每题原始完整回答、生成 token IDs、长度、停止原因、seed。
- `groups/group-001/grades/epoch-000/graded.jsonl`：答案提取、正确与否、代码测试详情。
- 面板完成后，`generation/epoch-000/`、`grades/epoch-000/` 汇总该面板所有题，`report/report.html` 提供逐题答案和结果链接。

总入口的汇总只纳入完整面板；面板尚未全部完成时，可以读已完成组的报告。生成与判分分开保存，后续修正判分规则可复判，不必重新生成。

这份 Base 对照可用于原随机组以及使用同一模型版本、题目和生成协议的 high/low 组。比较前核对各任务冻结的 Base revision、panel 内容、采样参数和实际覆盖题数，不能把缺失题当成答错。

本地验证使用合成题、模拟 vLLM/Docker 的 CPU 测试，覆盖禁用 adapter、epoch 0 限制、固定分组、判分与中断续跑。真实 GPU 推理与 Docker 执行须在 VM 上运行。
