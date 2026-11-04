# VM clone 后的操作清单

以下命令在 **Linux VM 的 repo 根目录**执行。默认 RTX A6000 48GB、单 GPU、1000G volume 挂载 `/data`。模型和数据只在 VM 下载。所有新训练默认 BF16 LoRA（Base 不量化）；不要复用旧全参或 QLoRA run 名。

## 1. 确认 GPU 和 volume

- [ ] clone repo 并进入目录；以后更新用 `git pull`。
- [ ] 核对 GPU 型号/显存，确认 `/data` 是挂载点。
- [ ] 为当前账号创建可写目录；不要重新格式化已有 volume。

```bash
nvidia-smi
lsblk -o NAME,SIZE,FSTYPE,MOUNTPOINTS
mountpoint -q /data || { echo '/data 未挂载，停止'; exit 1; }
df -h /data
sudo install -d -o "$(id -u)" -g "$(id -g)" /data/repeat_sft
export REPEAT_SFT_ROOT=/data/repeat_sft
export HF_HOME="$REPEAT_SFT_ROOT/.cache/huggingface"
export TMPDIR="$REPEAT_SFT_ROOT/.cache/tmp"
export PIP_CACHE_DIR="$REPEAT_SFT_ROOT/.cache/pip"
export CUDA_VISIBLE_DEVICES=0
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$HF_HOME" "$TMPDIR" "$PIP_CACHE_DIR" "$REPEAT_SFT_ROOT/logs"
```

新终端重新设置环境变量。程序会尊重已有 HF_HOME/TMPDIR，请确保它们不指向快满的系统盘。

## 2. 安装依赖与取得数据权限

- [ ] 使用 Python 3.11/3.12；安装与 VM 驱动匹配的 CUDA PyTorch。
- [ ] 安装 repo 依赖，确认 BF16、bitsandbytes、PEFT 可用。
- [ ] 取得 GPQA 访问权限，并登录自己的 Hugging Face 账号。

```bash
python3 -m venv .venv
source .venv/bin/activate
# 示例：PyTorch 2.6.0 / CUDA 12.4，驱动必须兼容。
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python -m pip check
python -c 'import torch, peft, bitsandbytes; assert torch.cuda.is_available(); assert torch.cuda.is_bf16_supported(); print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory/1024**3, peft.__version__, bitsandbytes.__version__)'
hf auth login
```

驱动不兼容时按 [PyTorch 官方版本页](https://pytorch.org/get-started/previous-versions/)选择 `2.6 <= torch < 2.10` 的安装方式。不在命令行明文记录 token。GPQA 无权限先申请，或传已有授权文件的 `--gpqa-csv`。

## 3. 准备并检查数据

- [ ] 准备 400 train、200 holdout、128 MATH500、64 GPQA，查看去重/版本记录。
- [ ] 检查完整推理、答案和结束符都参与监督，提示词与填充被屏蔽。

```bash
python -u prepare_data.py --out artifacts/data-s42 --seed 42 --panel-seed 2026
python - <<'PY'
import json, os
from pathlib import Path
p = Path(os.environ["REPEAT_SFT_ROOT"]) / "artifacts/data-s42/inspection.jsonl"
x = json.loads(p.open(encoding="utf-8").readline())
print("被屏蔽的提示词：\n", x["masked_prompt_text"])
print("\n参与监督的内容：\n", x["supervised_text"])
print("\n末尾 token：\n", x["tail_tokens"])
PY
```

已有数据目录可在校验通过后复用，不覆盖重建。默认过滤总长超过 10,240 的整条样本，不截断原解答。`manifest.json` 保存 commit；跨抽样 seed 比较时用 `--model-revision / --sft-revision / --math-revision / --gpqa-revision` 固定同一批 commit，并保持 `--panel-seed 2026`。

## 4. 真实 smoke test

- [ ] 停止同一 GPU 上其他训练/评测。
- [ ] 完成连续/中断恢复两条 32 步轨迹，检查 `SUCCESS.json`。
- [ ] 查看输入、loss、`resume_comparison.json`、生成回答。

```bash
python -u smoke_test.py --data artifacts/data-s42 --out runs/smoke-lora-s42
```

使用 7 条最短样本＋1 条最长训练样本；第 13 步模拟硬退出，从第 8 步恢复，要求连续与恢复结果逐位相同；随后验证拟合、生成、判分。默认 smoke 生成 512 token，只测流程，不能代表正式正确率或 8,192-token 生成显存。

中断后先解决 OOM/空间问题，再恢复：

```bash
python -u smoke_test.py --data artifacts/data-s42 --out runs/smoke-lora-s42 --resume
```

新开一次完整生成长度检查可加 `--generation-max-new-tokens 8192`，使用新的 smoke 目录。已有 smoke 配置不能中途更换。

若最长训练样本 OOM：先检查 VM 实际可用显存并排除其他 GPU 进程，可在新配置/目录测试 FA2 或 `configs/qlora_nf4.json`。若决定改用更短数据，必须显式重新准备数据并对所有对照组采用同一长度规则；这会改变样本范围。不要偷偷截断、跳题或接着原 run 改配置。程序不会自动降低 10,240/8,192 上限。

## 5. 正式随机基线与恢复

- [ ] smoke 通过后从 Base 新建正式 run，不接着 smoke adapter 训练。
- [ ] 保存 PID，检查日志记录的可训练参数仅为 LoRA。

```bash
nohup python -u train.py \
  --data artifacts/data-s42 --config configs/baseline.json \
  --run runs/random-lora-s42 --seed 42 \
  >> /data/repeat_sft/logs/random-lora-s42.train.log 2>&1 < /dev/null &
echo "$!" > /data/repeat_sft/logs/random-lora-s42.train.pid
tail -f /data/repeat_sft/logs/random-lora-s42.train.log
```

400 条/轮、64 轮、25,600 次 adapter 更新。进度条、ETA、每轮耗时见控制台、`progress.json`、`epoch_times.json`。

暂停：`kill -TERM "$(cat /data/repeat_sft/logs/random-lora-s42.train.pid)"`，等待恢复点保存、进程退出。前台可按一次 Ctrl+C。不要 kill -9；退出 tail 只停止查看日志。

代码未变时：

```bash
nohup python -u train.py --run runs/random-lora-s42 --resume \
  >> /data/repeat_sft/logs/random-lora-s42.train.log 2>&1 < /dev/null &
echo "$!" > /data/repeat_sft/logs/random-lora-s42.train.pid
```

git pull 后源代码指纹变化，使用原快照：

```bash
run_dir=/data/repeat_sft/runs/random-lora-s42
code_dir=$(python -c 'import sys; from recipe.resume import original_code; print(original_code(sys.argv[1]))' "$run_dir")
nohup python -u "$code_dir/train.py" --run "$run_dir" --resume \
  >> /data/repeat_sft/logs/random-lora-s42.train.log 2>&1 < /dev/null &
echo "$!" > /data/repeat_sft/logs/random-lora-s42.train.pid
```

恢复使用冻结配置，不传新 seed/data/subset。定期查看 `df -h /data`。第一次初始化失败且没有完整恢复点时，保留失败目录，换新 run 名启动。

## 6. 先拟合评测，再四组生成

- [ ] 确认训练完成并释放 GPU。
- [ ] 完成八个节点的 train/holdout 拟合评测。
- [ ] 再启动四组生成、判分、分析；同一 GPU 串行执行。

本次要用 **vLLM 并发 8** 时：拟合完成后，按 [VLLM_EVALUATION.md](VLLM_EVALUATION.md) 在 `/data/repeat_sft/venvs/vllm-0.11.0` 安装独立环境，并用 `run_grouped_vllm.sh` 启动 `grouped-e4plus-vllm-v1`。如果拟合曾去重，显式指定 `teacher_forcing-dedup-v1`。不要把 vLLM 装进训练 `.venv`，也不要与下面的 Transformers 生成任务同时占用 GPU。下面保留原后端的操作方式。

先运行并等待退出：

```bash
nohup python -u evaluate_loss.py --run runs/random-lora-s42 \
  --epochs 0,1,2,4,8,16,32,64 \
  >> /data/repeat_sft/logs/random-lora-s42.loss.log 2>&1 < /dev/null &
echo "拟合进程 PID=$!"
tail -f /data/repeat_sft/logs/random-lora-s42.loss.log
```

重跑同一命令跳过已保存样本；完成后每个节点的 train/holdout 都有 `summary.json`。恢复期间不要改代码；若已改代码，可从对应 `evaluations/teacher_forcing/epoch-XXX/code/evaluate_loss.py` 快照逐节点恢复，所有节点仍采用同一评测配置。

拟合完整后：

```bash
nohup bash run_grouped_analysis.sh runs/random-lora-s42 grouped-e4plus-8192-v1 \
  --epochs 4,8,16,32,64 \
  >> /data/repeat_sft/logs/random-lora-s42.grouped-e4plus.log 2>&1 < /dev/null &
echo "$!" > /data/repeat_sft/logs/random-lora-s42.grouped-e4plus.pid
tail -f /data/repeat_sft/logs/random-lora-s42.grouped-e4plus.log
```

自动找到本 run 完整拟合结果；**不要把旧全参拟合结果给新 LoRA run**。如果拟合结果曾去重，启动命令必须额外传入 `--teacher-forcing runs/random-lora-s42/evaluations/teacher_forcing-dedup-v1`，完整命令见 [分组评测说明](GROUPED_EVALUATION.md)。每组 48 题按 64→32→16→8→4 完成并保存报告，再进入下一组，共 960 条回答。

每题完成后写盘，当前题可能长时间没有新日志。停止向记录的 PID 发 SIGTERM；等待退出后恢复重跑相同命令，保持相同任务名和拟合路径。已完成题跳过，强制结束时当前未写完的题重跑；不恢复单题内的 token 生成进度。**恢复不加 `--reset-generation`**。旧任务保留冻结的节点列表；用新的 `grouped-e4plus-8192-v1` 任务名开始这次五节点评测。

若需要包含 epoch 0 的八节点完整面板，可改用：

```bash
nohup bash run_analysis.sh runs/random-lora-s42 full-v1 \
  >> /data/repeat_sft/logs/random-lora-s42.full.log 2>&1 < /dev/null &
```

它独立跑拟合和生成，不复用上述分组流程，不要同时启动两者。

## 7. 查看与重新判分

- [ ] 如需五个 epoch × 四种截断/判分情况的案例报告，按 [EXAMPLE_REPORT.md](EXAMPLE_REPORT.md) 运行 `report_examples.py`；下载生成的 `report.html` 查看。只需要 Python 标准库，报告输出使用新目录；某类没有样本会明确留空。

- [ ] 第一组完成可看该组报告；全部完成后看总报告。
- [ ] 同时看正确率、长度、正常结束率、答案提取率，点开具体解答核查。

```text
/data/repeat_sft/runs/random-lora-s42/
  run.json, data/, tokenizer/                  配置、版本、完整题目/原解答
  checkpoints/epoch-000,...,064/              adapter 与恢复状态
  evaluations/teacher_forcing/                逐样本/逐 token 拟合
  postprocess/grouped-e4plus-8192-v1/
    groups/group-001/analysis/report.html      第一组报告
    generation/epoch-*/responses.jsonl        全面板原始回答、token、停止原因
    grades/                                   抽取、判分与汇总
    analysis/report.html                      全部结果和题目证据
    grouped_complete.json                     最终报告与判分的实际路径
```

失败重试可能产生 `analysis-retry-*`，以 `grouped_complete.json` / 每组 `complete.json` 为准。SFTP 下载整个 analysis 目录后本机打开 report.html；只下载 HTML 会丢失证据链接。

规则修改后只重跑 CPU 判分/分析：

```bash
python grade.py \
  --responses runs/random-lora-s42/postprocess/grouped-e4plus-8192-v1/generation \
  --out runs/random-lora-s42/grades-v2
python analyze_run.py --run runs/random-lora-s42 \
  --epochs 0,1,2,4,8,16,32,64 --generation-epochs 4,8,16,32,64 \
  --grades runs/random-lora-s42/grades-v2 --out runs/random-lora-s42/analysis-v2
```

若使用去重后的拟合目录，上面的 `analyze_run.py` 同样要加 `--teacher-forcing runs/random-lora-s42/evaluations/teacher_forcing-dedup-v1`。默认五节点生成没有 1/2 轮结果，不能计算完整的 1—4 轮早期正确率排名；需要此对比时另建包含这些节点的任务。

## 8. 可选：初始 loss 低/中/高组

- [ ] 使用与训练相同的量化、Base 版本和配置打分。
- [ ] 每组从 Base 初始化独立 LoRA；不接着随机基线 adapter 训练。

```bash
python -u score_pool.py --data artifacts/data-s42 --config configs/baseline.json \
  --out artifacts/scores-lora-s42
python select_groups.py --data artifacts/data-s42 --scores artifacts/scores-lora-s42 \
  --out artifacts/groups-lora-s42 --seed 42
for group in low mid high; do
  python -u train.py --data artifacts/data-s42 \
    --subset "artifacts/groups-lora-s42/$group.jsonl" \
    --config configs/baseline.json --run "runs/$group-lora-s42" --seed 42 || break
done
```

这是串行前台执行；需后台时放进自己的 bash 脚本再 nohup。匹配不到 400 条时明确失败，不能把不匹配组当可比对照。对每个 run 重复第 6、7 节，然后比较早期 1/2/4 和后期 32/64：

```bash
python export_curves.py \
  --run random runs/random-lora-s42 runs/random-lora-s42/postprocess/grouped-e4plus-8192-v1/grades \
  --run low runs/low-lora-s42 runs/low-lora-s42/postprocess/grouped-e4plus-8192-v1/grades \
  --run mid runs/mid-lora-s42 runs/mid-lora-s42/postprocess/grouped-e4plus-8192-v1/grades \
  --run high runs/high-lora-s42 runs/high-lora-s42/postprocess/grouped-e4plus-8192-v1/grades \
  --out artifacts/comparison-lora-s42
```

判分有 retry 目录时以对应 `grouped_complete.json` 为准。排名交叉后另建数据抽样 seed / 训练 seed 实验确认。

## 9. 可选：四组测速 / QLoRA

```bash
python -u benchmark_train.py --data artifacts/data-s42 \
  --config configs/baseline.json --out runs/benchmark-lora-s42 --seed 42
# 跳过成功项，失败项从 Base 重跑并保留历史。
python -u benchmark_train.py --out runs/benchmark-lora-s42 --resume
```

只测 batch 1/4 × SDPA/FA2；batch 4 的显存需求取决于实际序列长度，OOM 或 FA2 缺依赖都会记录失败，不自动改配置。FA2 安装和指标解释见 [BENCHMARK.md](BENCHMARK.md)。默认不保存 adapter，首次加 `--save-weights` 可保存。

若希望比较更省显存的 QLoRA，使用 `--config configs/qlora_nf4.json`，另建如 `random-qlora-s42` 的目录，先用同配置跑 smoke。score_pool、benchmark 同样传该文件，evaluate_loss/generate 自动读 run。默认 `configs/baseline.json` 与 `configs/lora_bf16.json` 都是不量化的 BF16 LoRA。不同量化协议不能混成“只差数据”的对照；已经开始的 QLoRA 运行用原配置/代码快照恢复，不直接改成 BF16 LoRA。

## 10. 归档

- [ ] 保留代码 commit、数据 manifest、完整 run、每个 adapter、日志、原始回答与判分。
- [ ] 保留固定 Base commit 的 HF 缓存，或确保以后能下载同版本；adapter 不是独立完整权重。
- [ ] 定期检查 volume，旧全参模型/历史备份不会自动删除。
- [ ] 不编辑冻结输入、代码或 marker；改实验配置就创建新 run。

## 11. 可选：准备 BBH / LiveCodeBench 迁移面板

- [ ] 按 [TRANSFER_DATA.md](TRANSFER_DATA.md) 检查 `/data` 至少有约 20 GiB 空闲，使用实际 run 的训练清单检查重合。
- [ ] 默认新增 BBH 100 道 + LiveCodeBench 100 道，共 200 道；BBH 四类各 25 道。

```bash
python3 -u -B prepare_transfer_data.py \
  --train-jsonl runs/random-lora-s42/data/train.jsonl \
  --out artifacts/transfer-s2026 \
  --bbh-size 100 --lcb-size 100 --seed 2026
# 准备中断：同一命令加 --resume；已完成源文件复用，中断所在文件重读。
```

- [ ] 查看 `summary.json`、`inspection.jsonl`、`selection.jsonl`；保留 `references/` 中原题和测试数据。
- [ ] 此步只准备数据，不替换旧 run 的 `data/panel.jsonl`。生成与判分使用下方第 13 节的独立入口。

## 12. 可选：补存缺失的中间 epoch

- [ ] 使用原训练 `.venv`，确认原 run 保留完整恢复状态和代码快照，`/data` 建议至少 10 GiB 空闲。
- [ ] 按 [REPLAY_CHECKPOINTS.md](REPLAY_CHECKPOINTS.md) 先运行 `--plan-only`，再后台补跑默认 10 个节点。

```bash
python -B replay_train.py --source-run runs/random-lora-s42 \
  --out runs/random-lora-s42-intermediate-v1 --plan-only
mkdir -p /data/repeat_sft/logs
nohup python -B -u replay_train.py --source-run runs/random-lora-s42 \
  --out runs/random-lora-s42-intermediate-v1 \
  >> /data/repeat_sft/logs/random-lora-s42-intermediate-v1.train.log 2>&1 < /dev/null &
echo "补跑 PID=$!"
# 中断恢复：改用 --out runs/random-lora-s42-intermediate-v1 --resume。
```

- [ ] 查看新 run 的 `replay_complete.json`；原 trainer 的 64 轮进度不代表补跑任务进度。
- [ ] 新模型仍用原 192 题面板；文档提供直接 vLLM 生成/判分，以及四组各 48 题的完整路线。
- [ ] 不比较新旧参数逐位一致；结果明确标记为补跑节点，原训练和评测文件保留。

## 13. 可选：vLLM 评测 BBH / LiveCodeBench

- [ ] 按 [TRANSFER_EVALUATION.md](TRANSFER_EVALUATION.md) 准备 Docker，让当前用户可运行 `docker info`。
- [ ] 在 VM 运行 `python3 -B build_lcb_sandbox.py`，完成标准输入输出题、函数题、降权与超时终止自检。
- [ ] 确认第 11 节的面板包含各 100 道题；原 `random-lora-s42` 保留 epoch 4/8/16/32/64。

```bash
mkdir -p /data/repeat_sft/logs
nohup bash run_transfer_vllm.sh \
  --run runs/random-lora-s42 --panel-dir artifacts/transfer-s2026 \
  --job-name transfer-e4plus-vllm-v1 \
  >> /data/repeat_sft/logs/transfer-e4plus-vllm-v1.log 2>&1 < /dev/null &
echo "迁移评测 PID=$!"
```

- [ ] 四组各 50 题，每组完成原五个节点的生成/判分/报告后再进入下一组；不用先跑拟合 loss。
- [ ] 中断后重跑同一命令；不加 reset，不换输出目录，已完成题和组保留。
- [ ] 查看 `runs/random-lora-s42/postprocess/transfer-e4plus-vllm-v1/report/report.html`、`curves.csv` 和逐题 JSON；第一组完成就能先看 `groups/group-001/report/report.html`。
