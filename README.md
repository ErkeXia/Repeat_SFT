# 重复 SFT 数据实验

研究对象是同一批解答在反复训练时的记忆、泛化和迁移曲线。默认实验为 `Qwen/Qwen3-4B-Base`、随机 400 题、公共留出 200 题、64 轮；正式实验从原始 Base 独立启动。这里提供代码，不包含预训练权重或原始数据。

本工程目录为 `D:\data_recipe_research`。上传整个工程到服务器后，在工程根目录执行下列命令。**数据准备、训练、生成仅在服务器运行。** 本地离线测试不联网、不加载任何预训练模型。

## 文件入口

| 文件 | 作用 |
| --- | --- |
| `prepare_data.py` | 固定 Hub commit、处理真实模板、验证解答、过滤长度、去重、划分数据、固定评测面板 |
| `train.py` | 全参数 BF16、AdamW8bit、逐样本更新、关键权重和完整恢复点 |
| `evaluate_loss.py` | 逐样本损失、teacher forcing token 准确率、逐 token 明细 |
| `generate.py` | 在固定下游面板生成；逐题保存原始文字和全部 token ID |
| `grade.py` | 独立 CPU 判分，允许对已有生成重新判分 |
| `score_pool.py` / `select_groups.py` | 原始 Base 对候选池打分，生成低／中／高损失匹配组 |
| `smoke_test.py` | 32 步连续训练与硬中断恢复对照，真实权重保存加载，少量生成判分 |
| `export_curves.py` | 导出训练日志、评测曲线和早期／后期描述性排名 CSV |
| `tests/test_offline.py` / `tests/test_storage_resume.py` | 无下载的 CPU 测试、存储路径与旧烟雾测试恢复检查 |

## 1. 服务器环境

建议 Linux、Python 3.11 或 3.12、支持 BF16 的 NVIDIA GPU。当前实现是**单 GPU、单进程**，不要用 `torchrun` 启动。请先按服务器驱动安装 CUDA 版 PyTorch 2.6–2.9，再安装其余依赖；[PyTorch 安装页](https://pytorch.org/get-started/locally/)提供 CUDA 对应命令。`requirements.txt` 固定 Transformers 4.57.1、bitsandbytes 0.48.1、datasets 4.3.0 和 math-verify 0.8.0，运行时另外保存完整实际包版本。

```bash
python -m venv .venv
source .venv/bin/activate
# 在此安装适合服务器驱动的 CUDA 版 torch，然后：
python -m pip install -r requirements.txt
export REPEAT_SFT_ROOT=/data/repeat_sft
# /data 必须已挂载；新服务器首次创建目录时可能需要 sudo：
mountpoint -q /data
sudo install -d -o "$(id -u)" -g "$(id -g)" "$REPEAT_SFT_ROOT"
export HF_HOME="$REPEAT_SFT_ROOT/.cache/huggingface"
export TMPDIR="$REPEAT_SFT_ROOT/.cache/tmp"
mkdir -p "$HF_HOME" "$TMPDIR"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES=0
python -B -m unittest discover -s tests -v
```

这不是 LoRA，也不是 8bit 权重量化：训练参数为 BF16，8bit 指优化器状态。长序列和全参数训练仍占显存；LM head 分块计算交叉熵，减轻完整词表 logits 的峰值，但没有承诺某张小显存卡一定可运行。先用服务器烟雾测试测实际用量。检查点含完整模型和训练状态，单组预计需要约 150–200 GB，加上缓存、烟雾测试与多组实验应额外预留空间；这是按约 4B 参数估算，具体以磁盘实测为准。

GPQA 使用 `Idavidrein/gpqa` 的 `gpqa_diamond` 配置，64 题选自其训练 split；它在这里仍是下游评测集。需要先在 Hugging Face 取得访问权限并通过 `hf auth login` 登录，或者传入已有合法访问的 `--gpqa-csv path/to/gpqa_diamond.csv`。脚本不绕过授权，也不自动改用不同数据源。

### 存储目录与现有 VM 的一次迁移

所有实验脚本将相对的运行、准备数据、分组和结果路径解释为 **`/data/repeat_sft` 下的路径**，因此后面的命令仍写 `runs/...`、`artifacts/...`。例如 `--run runs/random-s42` 实际写入 `/data/repeat_sft/runs/random-s42`，检查点、恢复状态、日志及默认评测结果都在该目录中。可用绝对路径环境变量 `REPEAT_SFT_ROOT` 更改根目录；显式传入的绝对路径仍按指定位置使用。`--config` 和准备阶段的原始输入文件参数仍相对于当前工程目录。

训练、数据准备和生成入口在导入 Hugging Face 库之前，将未设置的 `HF_HOME`、`TMPDIR` 默认设到这个根目录的 `.cache/` 下。已有环境变量会被尊重，所以迁移后请用下面的 `export` 明确指定。使用 `/data` 路径时会检查挂载；卷未挂载就报错，避免误写系统盘。

**已经有失败的 `runs/smoke-s42` 时，先迁移原文件，再续跑，不要重新 prepare 数据。** 停止正在运行的训练、评测和 smoke 进程。在 VM 的工程根目录 `git pull` 后执行以下一次迁移。代码与 `.venv` 留在工程目录；迁移 `runs`、`artifacts`、工程下的 `.cache`，并在旧位置建立符号链接，保留旧命令的路径兼容性。整个 smoke 目录都要保留，包括配置、日志、代码快照和恢复点。

```bash
# 在 VM 上执行；如果实际工程位置不同，只修改这一行。
cd /home/ubuntu/data_recipe/Repeat_SFT
git pull

bash <<'BASH'
set -euo pipefail
mountpoint -q /data || { echo '/data 尚未挂载，停止迁移' >&2; exit 1; }
target_root=/data/repeat_sft
sudo install -d -o "$(id -u)" -g "$(id -g)" "$target_root"
# 先检查全部目的地，避免覆盖已经存在的实验。
for name in runs artifacts .cache; do
    target="$target_root/$name"
    if [ -L "$name" ]; then
        [ "$(readlink -f -- "$name")" = "$target" ] || { echo "现有链接指向其他位置：$name" >&2; exit 1; }
    elif [ -e "$name" ]; then
        [ -d "$name" ] || { echo "$name 不是目录" >&2; exit 1; }
        if [ -e "$target" ] || [ -L "$target" ]; then
            echo "目标已存在，请先核对，未执行覆盖：$target" >&2; exit 1
        fi
    fi
done
for name in runs artifacts .cache; do
    target="$target_root/$name"
    [ ! -L "$name" ] || continue
    if [ -d "$name" ]; then mv -T -- "$name" "$target"; fi
    if [ -d "$target" ]; then ln -s -- "$target" "$name"; fi
done
BASH
```

约 61 GiB 的已有 smoke 文件要跨盘复制，迁移期间请等待命令结束。这里没有清空或覆盖目标目录；若目标已经存在，脚本停止，先核对两处内容。若此前缓存放在工程以外（例如 `~/.cache/huggingface` 或另设的 `HF_HUB_CACHE`），它不会被上述命令迁移；需将实际缓存复制到卷上并核对后再切换对应环境变量，避免重复下载。不要修改运行中的 `run.json`、各类 manifest 或原始代码快照。

迁移后设置当前终端环境并检查空间：

```bash
export REPEAT_SFT_ROOT=/data/repeat_sft
export HF_HOME="$REPEAT_SFT_ROOT/.cache/huggingface"
export TMPDIR="$REPEAT_SFT_ROOT/.cache/tmp"
mkdir -p "$HF_HOME" "$TMPDIR"
source .venv/bin/activate
df -h / /data
readlink -f runs/smoke-s42
```

最后一行应显示 `/data/repeat_sft/runs/smoke-s42`。终端新开后如需显式设置这些变量，可以重复上述 `export`。新脚本本身也有相同默认值。

## 2. 准备并冻结数据

```bash
python prepare_data.py --out artifacts/data-s42 --seed 42 --panel-seed 2026
```

第一次会在服务器读取数据和 tokenizer；训练开始时才加载模型权重。每个新输出目录必须不存在，以避免覆盖。`main` 只用于首次解析为实际 commit，之后都记录并使用解析后的 SHA。重做完全相同数据时，可显式传入 `--model-revision`、`--sft-revision`、`--math-revision`、`--gpqa-revision` 的已记录 SHA。也支持原 schema 的 `--sft-jsonl`、`--math-jsonl` 和 `--gpqa-csv`，保存本地源文件 SHA256。

处理规则：

- 数学题取 `question`，完整解答取 **`response`**；`answer` 仅作参考答案。保留原 `message` 提示和原始元数据。来源记录在 `data_source`。这些字段已按[数据集](https://huggingface.co/datasets/jasonrqh/Math-CoT-20k)的公开 schema 核对。
- 保留完整 `<think>...</think>` 和最终答案，交给模型自身的真实聊天模板处理。模板若改变了除空白／思考标记之外的解答内容、或 prompt 不是完整序列的精确 token 前缀，立即报错。
- 完整模板序列 **大于 10,240 token 时整条过滤**，不截断。末尾模板附加的换行也计入长度；损失只覆盖助手起始内容到 `<|im_end|>`，末尾换行不监督。
- 发布者标注该数据为 verified；本代码还用 math-verify 检查最终答案与参考答案等价。无法提取、无法验证、或验证失败的解答会记录并过滤。这使候选总体限定为可验证子集，需在分析时报告，不能把它当作未经额外过滤的原始 20k。
- 同一规范化题目只保留源文件顺序中**第一条满足要求的解答**。先建立唯一合格题池，再按 seed 随机留出 200 题；从其余候选随机取 400 题。重复轮次不会重新选解答。
- 在抽样前与**完整 MATH500、完整所选 GPQA 配置**检查重复并排除。还排除候选与 200 题留出集的近重复。方法为 NFKC＋去空白完全匹配，以及 7 字符 shingles 的 Jaccard ≥ 0.9；保留数字、运算符和大小写。这只能发现文本重复及高度近似，不能排除语义改写或模型预训练污染。
- 固定 128 道 MATH500、64 道 GPQA。GPQA 选项排列按题目身份和 panel seed 固定，正确字母随排列更新。所有组使用同一个 `panel.jsonl`。

`inspection.jsonl` 和终端会完整打印默认 3 条样本的实际输入、被监督的文本、掩码 prompt、末尾 token ID/label。正式训练前请实际查看这些输出，尤其确认推理在监督区、用户输入被屏蔽、`<|im_end|>` 参与监督。Qwen3 Base 的默认 EOS/PAD 为 `<|endoftext|>`，它和聊天结束符不同，见[原始 tokenizer 配置](https://huggingface.co/Qwen/Qwen3-4B-Base/blob/main/tokenizer_config.json)。代码不会因 PAD 与 EOS 同 ID 而错误屏蔽有效结束 token。

准备目录保存 `train.jsonl`、`holdout.jsonl`、`pool.jsonl`、`panel.jsonl`、检查重复所用题目全文、拒收原因、tokenizer、聊天模板、代码快照和带文件哈希的 `manifest.json`。`pool.jsonl` 已排除 holdout；保留它供后续初始损失分组。

## 3. 长训练前的真实烟雾测试

```bash
python smoke_test.py --data artifacts/data-s42 --out runs/smoke-s42
```

该脚本在服务器加载真实 Base，选择 8 条较短的真实训练样本，执行：

1. 连续训练 32 步。
2. 从同一 Base、同一 seed 另起一次，在第 13 步用 `os._exit(86)` 硬退出；恢复点是第 8 步，9–13 步会重放。
3. 启动新进程，从完整恢复点续训至 32 步；逐步比较样本身份、学习率、loss、梯度范数，逐参数比较最终 safetensors 权重，要求逐位相同。
4. 加载第 0 和 4 轮模型，对各 8 条 train/holdout 样本计算 teacher forcing 指标，各生成 2 道数学题和 2 道 GPQA 并判分。

通过才写 `SUCCESS.json`；失败会留下证据，禁止静默放宽误差。默认烟雾生成上限 **512 token**，只用于连通性验证，可能尚未输出最终答案；正式配置仍为 **30,000**。可以传 `--generation-max-new-tokens 30000` 额外验证正式长度预算。烟雾测试全程不要求正确率提升，但要查看 `data/inspection.jsonl`、逐步 loss 和生成输出；其作用是排除管线错误，不是衡量学习效果。

本机只有 CPU；真实 CUDA/BF16/bitsandbytes 烟雾测试尚未在本机执行。CPU 测试验证的是合成 token、小型随机网络上的损失、梯度与保存恢复不变量，不替代上面的真实服务器测试。

### 恢复没有跑完的 smoke test

完成上面的一次迁移后，用同一个输出目录加 `--resume`：

```bash
python smoke_test.py --data artifacts/data-s42 --out runs/smoke-s42 --resume
```

脚本核验冻结数据、配置和代码，检查两个训练分支的完整检查点。已到第 32 步且第 4 轮检查点完整的分支会跳过；未完成的分支从最新完整恢复点继续。针对这次连续分支已完成、恢复分支写盘失败的情况，会跳过 `continuous`，恢复 `resumed`，然后继续比较权重、计算拟合指标、生成和判分。若第 16 步保存失败而第 8 步恢复点完整，则从第 8 步重放；实际位置以终端打印出的检查点路径为准。`.partial-*` 目录不会被选为恢复点，旧日志与旧检查点也不会被覆盖。

**`git pull` 后续跑旧 smoke 会自动使用经哈希校验的原始代码快照。** 旧版本没有顶层 `code/` 时，从 `continuous/attempts/.../code/` 中查找。这样两个分支仍使用同一版训练代码，保持逐位比较的意义；新存储路径通过绝对参数和环境变量传入。旧快照继续保留旧版保存逻辑，新版每次写盘空间预检查只作用于新版代码创建的运行；当前 smoke 入口对新旧运行都会先做剩余空间检查。

原有命令和逐步日志保留；新子进程退出状态另记在 `command_results.jsonl`。已完成的逐题评测会按原协议跳过；如果已存在 `grades/`，重判写入新的 `grades-retry-*`，最终路径记录在 `SUCCESS.json` 的 `grading_output`。只有日志、权重比较和评测判分均通过才写成功标记。

恢复不是把新数据或新配置混入旧运行。若原始快照被改、冻结数据不同，或尚无任何完整检查点，会明确失败并保留现有文件；这时应保留失败目录，另选新输出目录启动。新 smoke 会在开始时冻结顶层配置和代码，后续同样使用 `--resume`。恢复时生成长度必须沿用原配置。默认要求输出文件系统至少有 100 GiB 可用空间，可显式传 `--min-free-gib` 调整；这是保守的启动检查，不是整个实验占用上限。

## 4. 正式随机基线

```bash
python train.py --data artifacts/data-s42 --config configs/baseline.json --run runs/random-s42 --seed 42
```

默认每轮 400 次更新，总计 25,600 次。所有参数可训练；每次一条样本，梯度累积为 1。全参数 BF16、梯度检查点、AdamW8bit，betas `(0.9, 0.999)`、eps `1e-8`、weight decay 0、梯度裁剪 1.0。第 1 次更新使用 `1e-7`，线性预热到第 100 次的 `1e-5`，随后恒定。每轮通过训练 seed 和轮号确定新的完整排列。

`configs/baseline.json` 可复制为新配置修改，配置与代码会完整存档。`--seed` 只改变训练 seed，数据抽样由准备阶段决定。正式组不会继承烟雾模型或其他组的权重。

新版训练默认显示文本进度条，包括完成百分比、全局更新数、当前轮次、当前轮已用时间和预计剩余时间（ETA 是估算，会受样本长度和检查点保存影响）。交互终端动态刷新；重定向到文件时，约每 30 秒及每轮结束／正常暂停输出一行进度，原有逐步 loss 输出继续保留。无需增加依赖或更改启动参数。

计时结果保存在运行目录：

- `epoch_times.json`：每轮 `active_seconds`（秒）和 `active_hms`（时分秒），以及 `epoch_complete`、`timing_complete`。每轮完成、正常暂停和恢复初始化时更新。
- `timing/<attempt>.jsonl`：每次启动的逐步原始计时，保留中断前已写入的记录。
- `progress.json`：原有进度字段保持不变，新增 `timing`，记录实时耗时和预计剩余秒数。

每轮耗时统计训练循环的墙钟时间，包含更新、日志和检查点保存，不包含模型加载及中断后的停机时间。恢复时接续有效步骤的计时；被回滚步骤的耗时不重复计入汇总，但仍保留在原始计时文件中。若在检查点发布之后、对应计时写入之前断电，缺少的计时会标为 `timing_complete=false`，不会伪造完整耗时。新增观测不写入优化器或恢复状态。

保存第 **0、1、2、4、8、16、32、64** 轮完整权重；第 0 轮直接保存未更新的 Base，附带尚未更新的恢复状态。所有关键节点永久保留。每 400 步保存完整恢复点，默认保留最近两个滚动恢复点；如果该步已经是关键节点，则直接使用关键节点恢复状态，避免重复保存。

```bash
# 主动在完成第 1234 次更新后保存并退出
python train.py --run runs/random-s42 --resume --stop-after 1234

# 继续已有运行；数据、tokenizer、配置和代码必须一致
python train.py --run runs/random-s42 --resume
```

`--stop-after` 是绝对更新数，必须大于恢复位置。对运行进程发送 SIGTERM/SIGINT 会在当前更新完成后保存；SIGKILL 或断电只能恢复至最后完整检查点。恢复记录模型、优化器、scheduler、Python/NumPy/CPU/CUDA RNG、轮号、轮内位置和该轮排列。写入临时目录并同步后才原子发布；`.partial-*` 不会被当作完整恢复点。

每次启动是独立 attempt，日志中保留发生中断前尚未提交的更新；`export_curves.py` 根据恢复位置产生无重复的有效时间线，原始日志不会被覆盖。若工程代码后来改变，请用 `run/attempts/<最初 attempt>/code/` 的快照恢复，运行目录传绝对路径；这样代码一致性检查可以通过。不要为了恢复而删除校验。

计时／进度条更新也会改变代码指纹，所以应在正式基线首次启动前更新代码。已用旧代码启动的运行仍按上述方法使用原快照续训，原快照不会自动获得新版计时功能。通常的正常暂停（SIGINT/SIGTERM）会保存当前更新后的恢复点；断电或 SIGKILL 从最近完整恢复点继续，默认可能重放最近一轮中尚未保存的更新。恢复命令不用重新传 `--data`、`--seed` 或 `--subset`。

## 5. 训练结束后串行评测

```bash
python evaluate_loss.py --run runs/random-s42
python generate.py --run runs/random-s42 --config configs/generation.json
python grade.py --responses runs/random-s42/evaluations/generation --out runs/random-s42/grades-v1
```

一次只加载一个检查点，按 `--epochs 0,1,2,4,8,16,32,64` 顺序评测并释放模型，不与训练同时占 GPU。可用 `--epochs 0,1,2` 先评一部分，其余以后补做。每个逐题 JSONL 会及时写盘；相同配置重跑会跳过已经完成的题，末尾不完整写入会先备份再修复。配置、数据或代码变动时，须指定新的 `--out`，避免把不同协议的结果混在一个文件里。

Teacher forcing 针对**原解答前文**，统计所有被监督位置的 next-token NLL 和 argmax 准确率，包含推理、最终答案、思考格式 token 和 `<|im_end|>`。它不等于自由生成正确率。除逐样本指标，还保存 `tokens/<sample_id>.npz`：token 位置、目标 ID、预测 ID、NLL，便于以后拆分推理／答案／结束行为。汇总同时保留逐样本等权和逐 token 加权结果，并单独统计结束 token 准确率。

正式生成参数固定为 temperature 0.6、top-p 0.95、top-k 20、每题一次采样、`max_new_tokens=30000`、无额外重复惩罚。按题目身份派生 seed，各检查点相同；运行中断、跳过已完成题不会改变其他题的随机种子。提示来自冻结面板，数学要求 `\boxed{}`，GPQA 要求 `\boxed{A}` 等字母。未额外启用／禁用 Qwen thinking，沿用已保存的原始模板行为。

停止于 `<|im_end|>` 标记为 `chat_end`；停止于 `<|endoftext|>` 标记为 `base_eos`；用完预算标记为 `length`。`normal_chat_end` 专指前者，`ended_with_eos` 包含两者。同时区分“token 数恰好到上限”和“因上限被截断”，最后一个 token 恰为结束符时不会误记为截断。不会静默缩小生成上限；若 prompt＋30,000 超出模型配置的上下文范围，会明确失败。

每题保存原始回答、去 special tokens 的文字、完整生成 token ID、prompt token ID、长度、seed、停止原因、实际问题和参考答案。OOM 或其他异常记录到 `errors.jsonl` 并停止，不把失败问题偷偷移出分母。

判分仅查看最终答案段：最后一个完整 `</think>` 之后的 `\boxed{}`，或明确 `Answer is/Final answer:` 标记；未闭合 thinking 不把其中的临时答案算成最终答案。数学用 [Math-Verify](https://github.com/huggingface/Math-Verify) 判断等价；GPQA 只抽取明确最终选项，避免把推理中的 A/B/C/D 当作答案。抽取失败计错；判分异常单独记录，汇总分母不缩小，比较脚本会拒绝包含判分异常的结果。

输出正确率、正常结束率、EOS 结束率、长度上限率、答案可抽取率、长度分布，以及按结束行为条件化的正确率。这些条件统计可以帮助发现格式行为与收益的关联，**不构成收益的因果分解**。

修改 `recipe/grading.py` 后直接重判：

```bash
python grade.py --responses runs/random-s42/evaluations/generation --out runs/random-s42/grades-v2
```

不重新生成、不重新训练。每个判分目录保留判分代码和依赖版本。抽取规则比较保守；存在格式不标准但实际正确的回答时，检查逐题证据再改规则。

## 6. 初始损失分组

```bash
python score_pool.py --data artifacts/data-s42 --out artifacts/base-scores-s42
python select_groups.py --data artifacts/data-s42 --scores artifacts/base-scores-s42 --out artifacts/groups-s42 --seed 42

python train.py --data artifacts/data-s42 --subset artifacts/groups-s42/low.jsonl --run runs/low-s42 --seed 42
python train.py --data artifacts/data-s42 --subset artifacts/groups-s42/mid.jsonl --run runs/mid-s42 --seed 42
python train.py --data artifacts/data-s42 --subset artifacts/groups-s42/high.jsonl --run runs/high-s42 --seed 42
```

`score_pool.py` 始终使用准备阶段记录的**原始 Base commit**，计算与训练完全相同监督范围内的平均 token loss；不会使用训练后权重。完成全部候选打分后才允许选择分组。

按全池 loss 排序划分三个三分位区间。每组选择 400 条，在 `source × floor(supervised_tokens / 512)` 层内使用完全相同配额；各组三分位互斥，因此低／中／高组三组互不重叠。随机基线可能与它们有交集，这不是训练／评测泄漏。`selection.json` 保存分位界限、各层配额、抽样 seed、实际长度和 loss 分布，并随训练子集复制进运行目录。

长度按区间匹配，不声称实际连续长度完全相等。公共支持不足 400 时直接报错；可以显式用 `--length-bin 1024` 放宽后另建输出目录，或增加候选池。不要把不匹配的三组直接当成只差 loss 的实验。初始低损失仅表示容易模仿，不预设它的短期／长期下游排序。

对每个训练组执行相同的 `evaluate_loss.py`、`generate.py`、`grade.py`。汇出对照：

```bash
python export_curves.py \
  --run random runs/random-s42 runs/random-s42/grades-v1 \
  --run low runs/low-s42 runs/low-s42/grades-v1 \
  --run mid runs/mid-s42 runs/mid-s42/grades-v1 \
  --run high runs/high-s42 runs/high-s42/grades-v1 \
  --out artifacts/comparison-s42
```

`curves.csv` 合并下游和拟合指标，`training_steps.csv` 提供恢复后的有效训练时间线，`early_late_rankings.csv` 分别用第 1/2/4 轮和第 32/64 轮的平均检查点准确率生成描述性排名，同时保留逐检查点结果。比较脚本要求初始模型、训练配置（除 seed）、面板、生成协议和判分代码一致。

发现交叉后应另建数据抽样 seed 与训练 seed 的运行确认。例如准备 `data-s43` 时保持同一组数据 commit 和 `--panel-seed 2026`，并分别改变数据 seed、训练 seed。改变数据抽样会改变 holdout，跨这些运行的 holdout 拟合曲线不能当成逐题配对结果；下游面板文件相同才可直接配对。128/64 题的一次采样仅用于观察走势，不能依据一次运行认定稳定规律。

## 结果保留与边界

```text
run/
  run.json                        # 实际训练配置、初始化 commit、数据与 tokenizer 哈希
  data/                           # 训练/留出/面板全文、数据版本、实际监督检查
  tokenizer/                      # 冻结 tokenizer 和原始模板
  attempts/<id>/                  # 每次启动的代码快照、环境、GPU 信息
  attempts.jsonl, logs/*.jsonl     # 恢复谱系及每次更新的原始事件
  timing/*.jsonl, epoch_times.json # 原始逐步计时与有效训练路径的每轮耗时
  checkpoints/epoch-000,...,064/   # 完整关键权重及 training_state.pt
  recovery/step-*/                # 最近两个完整恢复点
  evaluations/teacher_forcing/    # 逐样本和逐 token 拟合指标
  evaluations/generation/         # 原始回答、生成 token、停止信息、协议快照
  grades-v1/, grades-v2/           # 可独立重跑的答案抽取和判分
```

训练状态 `.pt` 使用 pickle，恢复函数只用于本工程自己生成的可信检查点。模型权重使用 safetensors。不同 run 必须用不同目录；不要删除关键权重来节约临时空间，否则无法补做未来评测。新版每次保存前按模型和优化器张量估计所需空间，并留出余量；检查通过后仍可能受并发写盘或配额影响。捕获到写入失败时，仅清理本次保存创建的未发布临时目录，保留已有完整恢复点和关键权重。其他历史 `.partial-*` 不自动删除。除此之外，自动删除只包括按配置淘汰的滚动恢复点及 `.test_tmp` 测试临时文件。

对读入格式和数值的关键假设都采用显式检查。当前交付没有多 GPU、vLLM 或自动调参；这些改动应形成有记录的新实验协议。64 轮是观察窗口，不是训练有效性的先验结论，最后一轮也不被视作默认最优。
