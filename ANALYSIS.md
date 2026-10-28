# 训练曲线与逐题分析

现有 `evaluate_loss.py`、`generate.py`、`grade.py` 负责计算并保存评测结果；新增 `analyze_run.py` 将这些结果跨检查点关联起来。分析阶段只读已保存的结果，在 CPU 上运行，不下载数据或模型，不加载训练权重。

按照本次实验的约定：

| 数据 | 观察内容 |
| --- | --- |
| 400 条 train | 对固定原解答的逐样本损失、teacher-forcing token 准确率、记忆和饱和趋势 |
| 200 条 holdout | 对未见原解答的拟合；不额外自由生成 |
| 固定 128 道 MATH500 | 数学题自由生成的正确率，用于观察同领域泛化 |
| 固定 64 道 GPQA | 自由生成正确率，用于观察跨领域迁移 |

## 1. 完成各检查点评测

### 一条命令在后台完成整个流程

新增 `run_analysis.sh` 会依次运行损失评测、MATH500／GPQA 生成、判分和分析。默认使用训练配置中保存的全部检查点；请在训练结束、GPU 可供评测使用时启动。

在 VM 的工程目录拉取代码后执行：

```bash
cd /home/ubuntu/data_recipe/Repeat_SFT
git pull
mountpoint -q /data
mkdir -p /data/repeat_sft/logs

nohup bash run_analysis.sh runs/random-s42 full-v1 \
  >> /data/repeat_sft/logs/random-s42-full-v1.log 2>&1 < /dev/null &
echo "后台 PID：$!"
```

脚本自动使用工程的 `.venv/bin/python`，无需在 `nohup` 中再次激活环境。若要使用其他 Python，可在启动前设置 `export PYTHON_BIN=/完整路径/python`。输出日志持续追加，查看时执行：

```bash
tail -f /data/repeat_sft/logs/random-s42-full-v1.log
```

`tail` 中按 Ctrl+C 只会结束查看日志，后台流程继续运行。全部完成后，结果位于：

```text
/data/repeat_sft/runs/random-s42/postprocess/full-v1/
  pipeline.json, launcher.sh, code/   # 本次检查点列表、输入身份及冻结代码
  teacher_forcing/                    # train／holdout 拟合结果与逐 token NPZ
  generation/                         # 每题完整回答及 token
  grades/                             # 判分
  analysis/report.html                # 可检索的总结与逐题报告
  state/*.done                        # 完成标记
```

它使用本次 `postprocess/full-v1` 内的评测目录。若之前已手动完成全部评测，直接使用第 2 节的 `analyze_run.py` 即可，无需重新执行整套 GPU 评测。

中途失败会停止后续步骤。再次执行**相同的 `nohup` 命令**会使用保存的 Python 代码和配置，跳过已经完成的检查点；当前检查点内的已写入记录由原评测脚本继续处理。判分或分析若留下未完成目录，会另建 `grades-retry-*`／`analysis-retry-*`，保留原文件；实际输出路径同时记录在 `grades_path.txt`、`analysis_path.txt` 和日志末尾。若要主动暂停，使用启动时打印的 PID 执行 `kill -TERM PID`，之后可按上述方式继续。

只评测早期节点时用一个独立名称并显式指定列表：

```bash
nohup bash run_analysis.sh runs/random-s42 early-v1 0,1,2,4 \
  >> /data/repeat_sft/logs/random-s42-early-v1.log 2>&1 < /dev/null &
```

同一名称固定检查点列表，修改列表或生成／判分协议时应另选名称。脚本恢复会继续使用最初冻结的代码。若后续只修正判分规则，可对该目录的 `generation/` 单独重新判分和分析，继续复用已有回答。

### 分步骤执行

在 VM 的工程目录、已激活的虚拟环境中执行。所有相对数据和结果路径仍在 `/data/repeat_sft` 下。

```bash
python -u evaluate_loss.py \
  --run runs/random-s42 \
  --epochs 0,1,2,4,8,16,32,64

python -u generate.py \
  --run runs/random-s42 \
  --epochs 0,1,2,4,8,16,32,64 \
  --config configs/generation.json

python grade.py \
  --responses runs/random-s42/evaluations/generation \
  --out runs/random-s42/grades-v1
```

损失评测和生成使用 GPU，可以在 64 轮完成后依次加载保留的检查点执行。判分只用 CPU。若这些评测和判分已经完成，直接执行下一节即可；分析不会重新训练、生成或判分。

`grade.py` 的输出目录必须是新目录；如果已有 `grades-v1`，分析时可直接使用它。需要修改判分规则时另写 `grades-v2`，分析输出也另用新目录。当前分析不读取旧 `summary.jsonl` 的总分，而是从逐题记录重新汇总，避免遗漏具体题目。

若已有一部分评测由旧代码生成，先保留它们。评测入口校验完整代码指纹；用新代码继续写同一个 epoch 的旧评测目录可能被拒绝。可使用该评测目录的 `code/` 快照补完，或用新的 `--out` 评测目录重新评测并在分析时传 `--teacher-forcing`、`--grades`。分析脚本本身可以直接读取旧格式的完整结果，不要求重训。

## 2. 生成总结和逐题证据

```bash
python analyze_run.py \
  --run runs/random-s42 \
  --grades runs/random-s42/grades-v1 \
  --out runs/random-s42/analysis-v1
```

默认分析训练配置中的全部关键检查点，通常是 `0,1,2,4,8,16,32,64`。输出目录必须不存在，避免覆盖已经完成的分析。

生成结果位于：

```text
/data/repeat_sft/runs/random-s42/analysis-v1/
```

把**整个 `analysis-v1` 目录**下载到电脑，浏览器打开 `report.html`。报告支持搜索题目、sample ID、集合或任务；点击题目可查看所有已评测检查点的回答和具体指标。HTML、JSON、NPZ 链接均指向该目录内的文件，不依赖原 VM 的绝对路径或在线服务。

## 3. 关注什么、打开哪些文件

| 文件 | 用途 |
| --- | --- |
| `report.html` | 可检索的总览；点击题目或改善清单进入具体证据 |
| `summary.json` | 总结后的拟合、任务正确率、输出行为和任务改善节点，供后续程序读取 |
| `fit_curves.csv` | 各 epoch 的 train／holdout 损失和 token 准确率，含逐题等权、逐 token 加权、分位数、结束符准确率 |
| `fit_samples.csv` | 每个样本在每个 epoch 的损失、token 准确率及相邻节点变化；同时保留除以实际 epoch 间隔的变化量 |
| `fit_saturation.csv` | 每个样本的损失稳定、token 准确率稳定、二者共同稳定和达到 99% token 准确率的节点 |
| `fit_summary_milestones.csv` | train／holdout 逐题平均指标的稳定节点 |
| `downstream_curves.csv` | 每个任务、每个 epoch 的正确率及相对 Base 的变化；长度、正常结束率、EOS 率、长度上限／截断率、答案提取率及条件正确率 |
| `task_milestones.csv` | 每个任务首次高于 Base、此后持续高于 Base、最高正确率及对应 epoch |
| `question_milestones.csv` | 每道评测题首次观察到答对、持续答对、答对后退步、各节点正误清单 |
| `question_checkpoints.csv` | 每道评测题在每个检查点的判分、提取答案、长度、停止行为和完整回答路径 |
| `question_transitions.csv` | 相邻检查点每题由错变对、由对变错、持续正确、持续错误或未知 |
| `paired_changes.csv` | 相邻检查点及相对 Base 的配对改善／退步数，连到对应题目清单 |
| `analysis_manifest.json` | 实际分析参数、输入文件哈希、代码快照、方法定义及覆盖提示 |

每个 CSV 中的 `report_file`、`evidence_file`、`response_file` 或 `token_file` 都相对于分析输出根目录。Excel 中可以按 `first_observed_correct_epoch`、`joint_tail_stable_epoch`、`task` 等筛选，再通过对应 HTML 或 JSON 找回具体题目。

### 逐题证据保存内容

```text
analysis-v1/
  report.html, summary.json
  *.csv
  questions/*.html                 # 训练、holdout、评测题的独立页面
  changes/*.html                   # 每组比较的改善／退步题目清单
  evidence/
    fit/{train,holdout}/*.json     # 完整题目、原推理解答、参考答案及拟合轨迹
    tokens/epoch-*/{train,holdout}/*.npz
                                  # 复制的原始逐 token NLL、位置、预测 ID、目标 ID
    questions/*.json              # 评测题完整原始记录、参考解答（数据集有提供时）和判分轨迹
    responses/epoch-*/*.json      # 每道题的完整生成回答、全部生成／提示 token ID、提取答案和判分
    changes/*.json                # 改善／退步及行为变化的具体 sample ID 清单
  inputs/                         # 冻结题目、运行信息、评测身份、判分版本；有 tokenizer 时一并复制
  code/                           # 此次分析的代码与 Git 状态快照
  analysis_manifest.json, analysis_status.json
```

MATH500 原数据含参考解答时会保留并展示；GPQA 若只有正确选项，不会虚构参考推理。模型自身在各 epoch 的完整推理和答案均保留。逐 token NPZ 也复制到分析目录，因此该目录大小可能达到数百 MB 或更多；不复制模型权重。

## 4. 如何解释节点

### 损失／token 准确率“饱和”

默认采用回顾性的稳定区间判断：从候选检查点开始到最后一个请求的检查点，至少有 **3 个**完整观测，整个区间的 `max − min` 不超过阈值：

- 平均 token NLL 范围不超过 `0.01`（自然对数单位，nats/token）。
- Token 准确率范围不超过 `0.005`，即 **0.5 个百分点**。

分别保存损失稳定节点、准确率稳定节点和两者共同稳定节点。逐样本与整体平均都可查看。最终一个节点不会自动被认定为饱和；大量小变化累计超过阈值也不会被认定为稳定。缺失节点不能当作稳定证据。

平稳可能发生在高损失、低准确率处，所以“稳定”不等于“已记住”。另外保存首次以及此后持续达到 `99%` token 准确率的节点，配合最终损失和完整曲线判断。

阈值可调整；每次使用新输出目录，实际值会写入分析清单：

```bash
python analyze_run.py \
  --run runs/random-s42 \
  --grades runs/random-s42/grades-v1 \
  --out runs/random-s42/analysis-strict-v1 \
  --loss-tolerance 0.005 \
  --accuracy-tolerance 0.002 \
  --saturation-min-points 3 \
  --token-accuracy-target 0.99
```

### 一道题何时做对

`first_observed_correct_epoch` 是**首次评测答对**的检查点。Base 就答对的题记为 `0`。若第 2 轮错、第 4 轮对，保存左界 `2` 和右界 `4`；只观察到这两个检查点，无法确定第 3 轮是否已经答对。单次采样还存在生成波动，不能把这个区间解释为潜在能力首次出现的精确时间。

`sustained_correct_epoch` 要求从该节点开始直到最后一个请求节点全部答对，默认至少 **2 个**检查点。仅在第 64 轮答对时，首次答对可记 `64`，持续答对仍为空。`regression_intervals` 单独保存答对后又答错的区间，不会把所有题压缩成“首次学会以后就永久掌握”。

任务层面同样保存首次正确率高于 Base 和此后持续高于 Base 的节点。它们是这批固定题目、这次采样的描述性结果，用于定位后续复查和重复实验，不是迁移能力出现或显著性的证明。

### 输出行为与正确率

`paired_changes.csv` 同时比较相同题目：由错变对数、由对变错数，以及新增答对题中有多少题同时改善正常结束或答案提取。报告里可点开这几组题的原始回答。

结束改善与提取改善可以发生在同一道题上，因此两组计数可能重叠。报告还保留结束／提取状态均相同但由错变对的题。结合 `accuracy_given_normal_chat_end` 等条件正确率观察关联；这些数据本身不提供收益的因果分解。

## 5. 早期或不完整结果

只评测了早期节点时，明确选择这些节点：

```bash
python analyze_run.py \
  --run runs/random-s42 \
  --grades runs/random-s42/grades-early \
  --epochs 0,1,2,4 \
  --out runs/random-s42/analysis-early-v1
```

此时所有“稳定／持续”判断都只到第 4 轮为止，后面补做 64 轮分析可能得到不同结果。

默认要求所有请求节点、所有题目评测和判分完整，拒绝混入不同面板、生成参数、参考答案或训练运行的结果。若只想检查部分完成的记录，可增加 `--allow-partial`：

```bash
python analyze_run.py \
  --run runs/random-s42 \
  --grades runs/random-s42/grades-v1 \
  --out runs/random-s42/analysis-partial-v1 \
  --allow-partial
```

缺失或判分异常会标成未知，保留 `expected_n`、`observed_n` 和 `complete`。不完整任务的正式 `accuracy` 留空，另提供明确标注的 `observed_accuracy` 便于调试；不会缩小分母后冒充完整面板正确率。重新补齐评测后，输出到新的分析目录。

如果评测目录自定义过，可指定：

```bash
python analyze_run.py \
  --run runs/random-s42 \
  --teacher-forcing runs/random-s42/evaluations/teacher-forcing-v2 \
  --grades runs/random-s42/grades-v2 \
  --out runs/random-s42/analysis-v2
```

这些分析仅依赖已有的逐题数据，不需要重跑训练。修正答案抽取规则时先用 `grade.py` 重新判分，再重新分析，已有原始生成回答可继续使用。
