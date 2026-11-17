# High / Low：四个 benchmark 的 vLLM 评测

入口为 `dataset/run_likelihood_benchmarks.sh`，控制器为 `dataset/evaluate_likelihood_extremes.py`。一次启动依次评测两个 100 条训练实验，**不需要先运行 `evaluate_loss.py`**。这里是模型独立生成回答，再判分；不是给定参考解答的 teacher forcing。

## 固定协议

默认读取 `runs/likelihood-100x16-s42/high` 和 `low`。两组都评测 **16 → 14 → 12 → 10 → 8 → 4 → 2 → 1**，共 8 个节点。

| 面板 | 固定题目 | 分组 |
| --- | --- | --- |
| MATH500 / GPQA | 原 run 的 128 道 MATH500 + 64 道 GPQA | 四组，每组 32 + 16 = 48 道 |
| BBH / LiveCodeBench | 原 `artifacts/transfer-s2026` 的 100 道 BBH + 100 道 LCB | 四组，每组 25 + 25 = 50 道 |

顺序固定为：high 的数学/GPQA 全部四组 → high 的 BBH/LCB 全部四组 → low 的数学/GPQA 全部四组 → low 的 BBH/LCB 全部四组。**每一组题目完成全部指定 epoch 的生成、判分和组报告后，再进入下一组。**

- vLLM 0.11.0，默认独立环境 `/data/repeat_sft/venvs/vllm-0.11.0`。
- BF16 Base + BF16 LoRA 推理，最多 8 题并发。
- 每题一次采样；temperature 0.6、top-p 0.95、top-k 20，最多 8,192 个新 token。
- 每个题目使用原提示词、选项排列和固定采样 seed；同一面板的两组训练模型使用一致的设置。
- 保留真实 EOS/长度停止原因，不补结束符。保存回答原文、token IDs、抽取答案、判分及 LCB 测试详情。
- MATH500 用原数学判分器；GPQA 用最终选项匹配；BBH 沿用逻辑子任务判分；LCB 用已有隔离 Docker 镜像执行公开和隐藏测试。

默认共 **2 × 392 × 8 = 6,272 条回答**。不会自动加入 epoch 0；如需 Base 对照，在首次启动的新任务中显式加 `--epochs 0,1,2,4,8,10,12,14,16`，此时共 7,056 条回答。两组的节点列表始终相同，生成顺序自动降序。

## 首次核验

首次启动先核验所有选定 checkpoint、冻结的训练集与 tokenizer、两组共同的数学/GPQA 面板、math-verify 和 Docker 自检，再开始 GPU 生成。

旧 BBH/LCB 面板绑定的是旧 400 条训练集，不能直接修改其哈希冒充已核验。新入口会：

1. 对 high、low 各自的 100 条训练题与全部 392 条评测题，检查规范化精确重合和 char7 Jaccard ≥ 0.9 的近似重合。
2. 若发现重合，保存 `overlap-rejections-*.json`，在 GPU 生成前停止；不会删题、补题或重新抽样。
3. 若通过，为两组分别复制原 BBH/LCB 固定面板并保存新的训练集哈希、旧哈希、原面板身份和重合检查结果。题目、顺序、提示词、参考文件与代码测试数据保持一致。

该检查不能排除所有语义上的题目重合。原 `artifacts/transfer-s2026`、已有推理结果和拟合数据均不修改。

## VM 一次启动

在仓库根目录执行。先在 VM `git pull`，确认两组目标 checkpoint 已保存，以及原 Docker 镜像 `repeat-sft-lcb:py311-v1` 可用。使用原环境即可，没有新增 Python 依赖，也无需重建已经通过自检且 worker 匹配的镜像。

```bash
mkdir -p /data/repeat_sft/logs

nohup bash dataset/run_likelihood_benchmarks.sh \
  --experiment runs/likelihood-100x16-s42 \
  --panel-dir artifacts/transfer-s2026 \
  --job-name four-benchmarks-vllm-v1 \
  >> /data/repeat_sft/logs/likelihood-four-benchmarks-v1.log 2>&1 < /dev/null &

echo "四任务评测 PID=$!"
tail -n 30 -f /data/repeat_sft/logs/likelihood-four-benchmarks-v1.log
```

`.sh` 会直接调用指定 vLLM 环境，不依赖当前激活的 `.venv`。若你的 vLLM 环境在别处，启动前设置：

```bash
export VLLM_PYTHON_BIN=/你的vllm环境/bin/python
```

先停止同一 GPU 上的训练或其他推理任务。缓存、结果和临时文件继续默认使用 `/data/repeat_sft`，Base 与 checkpoint 不复制到评测目录。

可先前台检查全部依赖并冻结任务，暂不加载 GPU 模型：

```bash
bash dataset/run_likelihood_benchmarks.sh \
  --experiment runs/likelihood-100x16-s42 \
  --panel-dir artifacts/transfer-s2026 \
  --job-name four-benchmarks-vllm-v1 \
  --prepare-only
```

通过后运行上面的 nohup 命令。若只评测已完成的部分节点，首次启动用 `--epochs 1,2,4,8` 等明确列表，并另起任务名；后续扩充节点也使用新任务名。

## 停止与恢复

- 用 `kill -TERM <评测PID>` 停止，等待子进程退出；`tail -f` 的 Ctrl+C 只停止查看日志。
- **原样重跑同一条启动命令即可恢复**，不加 reset。已完成回答和判分保留，完整组直接跳过。
- 单题生成完成后立即保存；中断时未完成的最多 8 条回答会重跑，不保存单题内部 KV cache。
- 判分独立逐题落盘；中断的代码测试可重跑，已完成判分跳过。完整组与关键输入会校验哈希，出现错误不会记为答错。
- 套件及每个面板都冻结代码、配置、题目、checkpoint 身份和 Docker 镜像 ID。之后的 git pull 不会改变本任务实际运行的代码。
- 不要同时运行本套件和它的子任务目录中的其他写入进程。只杀父进程的 `kill -9` 可能遗留 GPU 子进程；重新启动前确认旧任务已退出。

## 输出与报告

总入口：

```text
/data/repeat_sft/runs/likelihood-100x16-s42/
  postprocess/four-benchmarks-vllm-v1/
    report.html                 high/low 四个面板的报告入口和已完成面板的指标表
    curves.csv                  合并已完成面板的曲线，增加 training_group 列
    summary.json                实际已完成面板、目标节点、指标和预计回答数
    progress.json               正在评测哪个训练组、哪个面板
    suite/plan.json             固定协议、路由、checkpoint 身份和代码版本
    suite/overlap_checks.json   两组各自对四个任务的重合核验记录
    suite/panels/high/          high 的 BBH/LCB 新核验副本
    suite/panels/low/           low 的 BBH/LCB 新核验副本
    complete.json              四个面板全部完成后才生成
```

各个面板存放在对应 run 的标准 `postprocess` 下，兼容已有退化分析入口：

```text
high/postprocess/four-benchmarks-vllm-v1-math-gpqa/
high/postprocess/four-benchmarks-vllm-v1-bbh-livecodebench/
low/postprocess/four-benchmarks-vllm-v1-math-gpqa/
low/postprocess/four-benchmarks-vllm-v1-bbh-livecodebench/
```

每个面板含以下文件：

```text
setup/panel/                               固定题目、参考答案或代码测试数据
groups/group-001/generation/epoch-016/responses.jsonl
groups/group-001/grades/epoch-016/graded.jsonl
groups/group-001/report/report.html         这一组的全部节点报告
groups/group-002/...                        后续组，同样结构
generation/epoch-*/responses.jsonl          面板完成后合并的原始回答
grades/epoch-*/graded.jsonl                 面板完成后合并的判分
report/report.html                         正确率、长度、结束/截断/答案提取率
report/curves.csv                          逐任务、逐 epoch 指标
report/question_milestones.csv             首次/持续答对、退化和恢复节点
report/questions/*.html                    每题在所有选定节点的完整回答
report/questions/*.json                    对应结构化数据与 token IDs
```

总报告在每个面板结束后更新；面板还没全部完成时，已经完成的各组报告仍可直接打开。只下载总 `report.html` 不会包含链接的逐题文件；要离线阅读完整回答，请连同对应面板的 `report/` 下载，需要原题和代码测试数据时同时带上 `setup/panel/` 并保持相对目录关系。

默认没有 Base 生成对照，首次观测是 epoch 1；不能从某个节点首次答对推断模型恰好在那一轮学会。每题仅采样一次，保留逐题结果供后续配对分析；LCB 是本项目的 Python 执行协议，不是官方完整排行榜分数。

## 本地验证

```bash
python -B -m unittest discover -s dataset/tests -p 'test_likelihood_benchmarks.py' -v
python -B -m unittest discover -s tests -p 'test_intermediate_evaluation.py' -v
python -B -m unittest discover -s tests -p 'test_transfer_evaluation.py' -v
```

使用合成 checkpoint/回答和模拟执行器，测试两组的相同面板、重新核验、缺失节点/重合拒绝、分组顺序、中断续跑、原始回答和判分保留、最终报告，以及旧入口回归。没有本地模型或数据下载，没有运行真实 vLLM/CUDA 推理；真实 VM 环境可先用 `--prepare-only` 检查。
