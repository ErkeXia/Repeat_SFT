# BBH / LiveCodeBench 迁移评测数据准备

`prepare_transfer_data.py` 在 VM 上准备一个**独立的新增评测面板**，默认 BBH 100 道 + LiveCodeBench 100 道，共 200 道。只需 Python 3.10+ 标准库，不用 GPU、不加载模型或 tokenizer、不安装新依赖。

**这个入口只完成数据准备。** 数据完成后，使用新增的 [BBH / LiveCodeBench 分组评测](TRANSFER_EVALUATION.md)：`run_transfer_vllm.sh` 复用原 epoch 64/32/16/8/4，四组各 50 题，并发 8；`grade_transfer.py` 单独判分，LiveCodeBench 在 Docker 中运行公开和隐藏测试。不要把新 `panel.jsonl` 替换进旧 run，也不要用原 MATH500 / GPQA 的 `grade.py` 判这两种任务。

## 容量

2026-11-04 只查询官方文件元数据得到：

| 数据源 | 固定版本下原始大小 |
| --- | ---: |
| 所选四类 BBH 的完整 JSON 文件 | 395,335 bytes，约 0.40 MB |
| LiveCodeBench Lite `release_v6` 六个 JSONL 文件（含测试） | 4,485,994,821 bytes，约 4.49 GB / 4.18 GiB |
| LiveCodeBench 仅新增 `v6`，不包含旧版本 | 134,303,240 bytes，约 134 MB |

1000G volume 足以容纳此次新增数据，前提是仍有空闲空间。脚本采用保守准备预算 `3 × 所选源文件大小 + 2 GiB`，默认约 **14.54 GiB**；建议至少预留 **20 GiB**，同时考虑旧全参 checkpoint 等已有占用。预算涵盖分文件抽样缓存、最终选中题目的原文和原子写入临时空间，不包括未来代码执行容器、解压后的隐藏测试或多个评测备份。

只取 50/100 道不代表只需读取 50/100 行：为使抽样覆盖整个固定版本，需要**流式扫描约 4.49 GB**。不创建完整 JSONL 下载缓存或 Arrow 副本；每个源文件最多保留 `--lcb-size` 条候选，最终再合并抽样。完成的候选缓存保留用于续跑、审计，实际大小取决于抽中的测试用例，记录在 `manifest.json`。

新增面板不需要额外模型权重。200 题 × 5 个 checkpoint × 8192 上限为最多 8,192,000 个生成 token；未来保存 token IDs、文本和判分通常是数十至数百 MB 量级，具体取决于重复副本与报告。执行测试的临时空间需在判分器接入时另行预算。

来源：[BBH 文件目录](https://github.com/suzgunmirac/BIG-Bench-Hard/tree/main/bbh)、[LiveCodeBench Lite](https://huggingface.co/datasets/livecodebench/code_generation_lite/tree/main)。LiveCodeBench 原始加载脚本会先下载全部文件；这里直接读取固定 commit 下的 JSONL，不执行远程加载脚本。

## 在 VM 上运行

在 repo 根目录运行；相对 `artifacts/...` 和 `runs/...` 都解析到 `/data/repeat_sft`。

```bash
export REPEAT_SFT_ROOT=/data/repeat_sft
mountpoint -q /data || { echo '/data 未挂载'; exit 1; }
df -h /data

# 可选：只查询文件元数据、计算预算和检查剩余空间，不读取数据文件正文。
python3 -B prepare_transfer_data.py \
  --train-jsonl runs/random-lora-s42/data/train.jsonl \
  --out artifacts/transfer-s2026 \
  --bbh-size 100 --lcb-size 100 --seed 2026 \
  --estimate-only

# 准备默认的 200 道新增评测题。
python3 -u -B prepare_transfer_data.py \
  --train-jsonl runs/random-lora-s42/data/train.jsonl \
  --out artifacts/transfer-s2026 \
  --bbh-size 100 --lcb-size 100 --seed 2026
```

`--train-jsonl` 指向**实际训练 run 保存的训练子集**，不要在训练使用其他子集时误传通用的 `artifacts/data-s42/train.jsonl`。

默认 **BBH 每类 25 道**；LiveCodeBench 在整个符合条件的候选池中随机抽 100 道。两个 size 参数可以修改；改变题数或抽样 seed 时使用新的输出目录。

需要后台运行时：

```bash
mkdir -p /data/repeat_sft/logs
nohup python3 -u -B prepare_transfer_data.py \
  --train-jsonl runs/random-lora-s42/data/train.jsonl \
  --out artifacts/transfer-s2026 \
  --bbh-size 100 --lcb-size 100 --seed 2026 \
  >> /data/repeat_sft/logs/transfer-s2026.prepare.log 2>&1 < /dev/null &
echo "数据准备 PID=$!"
```

不要同时运行前台和后台两条准备命令。每个输出目录有进程锁。

## 数据与抽样协议

- **BBH**：`logical_deduction_five_objects`、`tracking_shuffled_objects_five_objects`、`boolean_expressions`、`web_of_lies`。默认每类 25 道，类内按 seed 确定的稳定哈希随机排名抽样。使用零样本推理提示词，不提供带答案的示例；这是自定义子面板，不是官方完整 BBH 分数。
- **LiveCodeBench**：默认固定 `release_v6`，在通过日期和重复检查的全部候选中均匀伪随机抽样，不按文件头部、解答长短或难度挑题。保存实际平台、难度和日期分布。生成目标为 Python 3，保留题干和 starter code；标准输入输出题与函数题使用不同接口提示。
- 版本默认固定为 BBH commit `9ee07bd481feebf959a6b59d61ea57bdcf30964d`、LiveCodeBench commit `0fe84c3912ea0c4d4a78037083943e8f0c4dd505`。可通过 `--bbh-revision`、`--lcb-revision` 显式选择其他版本，实际 commit 和文件校验和都会记录；不支持含糊的 `release_latest`。
- `--lcb-version` 支持 `release_v1` 到 `release_v6`（累计版本），或 `v6`（只有第六批新增题），或 `v5_v6`（第五、六批）。不同范围必须另建面板，不能混在同一条曲线中。
- `--lcb-start-date YYYY-MM-DD` / `--lcb-end-date YYYY-MM-DD` 可指定包含两端日期的窗口。保存题目日期**不等于已经证明预训练无污染**；需要另行确认 Base 的知识截止日期。日期过滤仍需扫描所选源文件。
- 与实际训练问题执行规范化精确匹配、字符 7-gram Jaccard ≥ 0.9 的近重复检查，匹配题排除并记录。不能排除语义重复或未知预训练污染。
- 本步不调用 tokenizer，也不做 token 长度过滤或截断；`summary.json` 明确记录 `prompt_tokens_checked=false`。后续生成入口必须使用 run 的 tokenizer 检查完整提示词加生成预算是否超上下文。
- LiveCodeBench 不保证提供参考解答代码；正确性依据是测试用例。`gold_answer=null` 是有意的。隐藏测试保留为原始值，可能是压缩编码；本脚本不反序列化 pickle、不执行测试、不运行生成代码。

## 中断和恢复

每个上游文件完整读取并通过大小、SHA256/Git blob 校验后，保存候选及 `stages/.../complete.json`。中途停止后，使用**相同参数**加 `--resume`：

```bash
python3 -u -B prepare_transfer_data.py \
  --train-jsonl runs/random-lora-s42/data/train.jsonl \
  --out artifacts/transfer-s2026 \
  --bbh-size 100 --lcb-size 100 --seed 2026 \
  --resume
```

已完成文件不再联网扫描；中断所在文件会从头重读，不支持文件内部字节位置恢复。输出目录、源版本、训练清单、题数、seed、日期窗口必须保持一致。最终完成后重跑 `--resume` 只校验已有文件，不覆盖结果。文件哈希不符会报错，不自动修复或覆盖。

若 `git pull` 后代码版本变化，使用输出目录内保存的 `code/prepare_transfer_data.py` 加相同参数与 `--resume`。不要把旧评测任务的代码快照用于准备新的面板。

## 查看结果

```bash
python3 -B - <<'PY'
import json
from pathlib import Path
p = Path('/data/repeat_sft/artifacts/transfer-s2026')
summary = json.loads((p / 'summary.json').read_text())
print(json.dumps(summary, ensure_ascii=False, indent=2))
for line in (p / 'inspection.jsonl').read_text().splitlines():
    row = json.loads(line)
    print('\nTASK:', row['task'], '\nID:', row['sample_id'])
    print(row['messages'][0]['content'])
PY
```

| 文件 | 内容 |
| --- | --- |
| `panel.jsonl` | 固定题目、生成提示词、来源、BBH 标准答案、引用文件路径；不嵌入隐藏测试 |
| `references/*.json` | 每道选中题的完整上游记录；LCB 包括公开/隐藏测试及 metadata，BBH 包括原始 target |
| `selection.jsonl` | 题目身份、子任务/平台/难度/日期、源文件行号 |
| `inspection.jsonl` | BBH 和 LCB 各一条实际提示词，便于人工检查 |
| `summary.json` | 数量、分布、重复/日期/训练重合排除统计、当前功能边界 |
| `plan.json` / `manifest.json` | 实际配置、固定源版本、文件校验和、存储大小、完成状态 |
| `stages/` | 各源文件候选、排除记录、完成标记，供续跑与审计 |
| `code/` | 准备数据时的完整代码快照及 Git 信息 |

仅测试用的离线输入支持 `--bbh-dir` 和 `--lcb-jsonl 文件1 文件2 ...`，必须同时提供。它们读取已有原始格式文件，不联网；仍记录输入哈希，仍需要 `--train-jsonl`。
