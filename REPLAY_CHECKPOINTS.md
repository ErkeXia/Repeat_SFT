# 补跑缺失的 epoch：独立保存，可中断恢复

入口：`replay_train.py`。默认补存 **6、7、9、10、12、14、20、22、26、30**，新 run 默认示例为 `/data/repeat_sft/runs/random-lora-s42-intermediate-v1`。

## 如何继续训练

| 原 run 的恢复起点 | 补存的 epoch | 需要训练 |
| --- | --- | --- |
| epoch 4 | 6、7 | 3 轮，1,200 次更新 |
| epoch 8 | 9、10、12、14 | 6 轮，2,400 次更新 |
| epoch 16 | 20、22、26、30 | 14 轮，5,600 次更新 |

共 **23 轮、9,200 次更新**。脚本按实际存在的完整 checkpoint 自动选择每个目标之前最近的起点；如果某个起点只剩权重而没有恢复状态，会退到更早的完整 checkpoint，并在计划里显示额外工作量。如果目标本来已经保留，脚本会提示直接评测原节点。

复制的是完整 LoRA checkpoint，包含 `training_state.pt` 中的优化器、调度器、随机数和数据位置。训练调用**原 run 内保存并校验过的 `train.py` 代码快照**，保留原配置、数据、种子、batch、学习率和绝对步数，不重新预热。到达每个目标后，原 trainer 的 `--stop-after` 保存完整恢复点，补跑脚本再把它原子移动到新 run 的 `checkpoints/epoch-XXX`，作为长期保留的节点。

没有改原始训练循环；没有比较新旧模型是否逐位一致。新模型应标记为原实验的**补跑分支**。仍应使用原训练 `.venv`、GPU 和依赖环境；训练时不要切换到 vLLM 的环境。

原 run 的权重、日志、拟合、生成和判分全部保留。新 run 不拷贝原评测目录，也不把新节点加入原 run。

## 存储预算

当前 rank 16、七种投影层、FP32 adapter，按 [Qwen3-4B-Base 配置](https://huggingface.co/Qwen/Qwen3-4B-Base/blob/main/config.json)计算约 33,030,144 个 LoRA 参数：单份权重约 **126 MiB**，10 份约 **1.23 GiB**。包含 AdamW8bit 状态、tokenizer 后，10 份完整 checkpoint 预计约 **2 GiB**。起点 4/8/16 的独立副本、滚动恢复点、临时保存和代码日志还会占空间，全部预计约 **3–5 GiB**；建议至少 **10 GiB 空闲**。

这些是当前配置的估计。`--plan-only` 按服务器实际起点目录大小计算预算；正式初始化会检查目的分区空闲空间。预算含新节点、起点副本、恢复点、一次临时保存、代码/日志余量和安全余量。已有 Base 缓存共享，不会为每个节点复制 4B Base；如果 Base 缓存丢失、要新建环境或存后续评测结果，需另外留空间。

## 在 VM 上启动

在 clone 的工程目录 `git pull`，然后使用原训练环境：

```bash
source .venv/bin/activate
mountpoint -q /data || { echo '/data 尚未挂载'; exit 1; }
df -h /data

# 只读检查：不会训练、下载模型或创建实验目录。
python -B replay_train.py \
  --source-run runs/random-lora-s42 \
  --out runs/random-lora-s42-intermediate-v1 \
  --plan-only
```

默认就是上述 10 个 epoch，也可显式传 `--epochs 6,7,9,10,12,14,20,22,26,30`。输出目录必须是与原 run 分开的新目录，不允许包含原 run 或放入原 run 内。

```bash
mkdir -p /data/repeat_sft/logs
nohup python -B -u replay_train.py \
  --source-run runs/random-lora-s42 \
  --out runs/random-lora-s42-intermediate-v1 \
  >> /data/repeat_sft/logs/random-lora-s42-intermediate-v1.train.log 2>&1 < /dev/null &
echo "补跑 PID=$!"
tail -f /data/repeat_sft/logs/random-lora-s42-intermediate-v1.train.log
```

`tail -f` 的 Ctrl+C 只退出日志查看。暂停训练用 `kill -TERM <补跑PID>`，等待当前更新和恢复点保存完成。前台运行时 Ctrl+C 同样会通知子训练进程保存。训练进程被强杀/断电时回到最近完整状态，最后尚未保存的更新会重跑；已经保存的指定节点保留。仅 `kill -9` 启动器可能留下继续运行的子进程，因此建议用 TERM。进程锁阻止同一输出目录被重复启动；Linux 上即使启动器被强杀，仍在运行的训练子进程继续持锁。

恢复时使用同一训练环境：

```bash
nohup python -B -u replay_train.py \
  --out runs/random-lora-s42-intermediate-v1 --resume \
  >> /data/repeat_sft/logs/random-lora-s42-intermediate-v1.train.log 2>&1 < /dev/null &
echo "补跑 PID=$!"
```

恢复会调用首次启动时冻结的补跑脚本，跳过已保存的 epoch，并从当前分支最近的完整状态继续。无需再提供原数据或 seed，也不会因 `git pull` 换成新训练循环。初次复制阶段被强杀可能留下同级 `.partial-*` 未发布目录；它不作为有效 run，重启时不会读取它。

## 结果位置和进度

```text
runs/random-lora-s42-intermediate-v1/
  run.json                     原训练配置 + replay 来源/目标说明
  replay_plan.json              三段补跑计划、起点文件哈希、空间估算
  replay_progress.json          补跑整体进度；完成时 status=complete
  replay_complete.json          10 个节点全部保存后出现
  replay_commands.jsonl         每段启动命令、恢复步数、原始起点
  replay_environments.jsonl     每次补跑执行环境
  replay_code/                  冻结的补跑调度代码
  anchors/epoch-004,008,016/     原始起点的完整独立副本
  checkpoints/epoch-006,.../    10 个新节点；adapter + 完整恢复状态
  recovery/                    原配置控制的滚动恢复点
  data/, tokenizer/            冻结的数据与模板副本
  attempts/, logs/, timing/     原 trainer 产生的逐步日志与耗时
```

原 trainer 的进度仍按原始 **25,600 步 / 64 轮**显示，它在每个目标处暂停，所以单独的 `progress.json` 最后可能是 `paused`。**补跑是否完成看 `replay_complete.json` 和 `replay_progress.json`**，不需要继续训练到 64。跳过的原训练区间不补造损失和耗时记录。每个目标会重新加载一次模型，启动耗时通常远小于补跑的几百到上千次更新。

## 测试新节点的正确率

新 run 复制了原来的固定 **128 道 MATH500 + 64 道 GPQA**；使用新 run 路径即可。推理仍为并发 8、最多 8,192 新 token、temperature=0.6、top-p=0.95、top-k=20，前提是使用原来的 `configs/generation.json` / `configs/vllm.json`。

只想先看正确率，不必先算拟合：训练完成后运行以下命令。每次启动前确保没有另一个生成进程在写同一目录。顺序为 30→26→22→20→14→12→10→9→7→6，每个节点评测全部 192 题；共 1,920 条回答。

```bash
nohup /data/repeat_sft/venvs/vllm-0.11.0/bin/python -B -u generate_vllm.py \
  --run runs/random-lora-s42-intermediate-v1 \
  --epochs 6,7,9,10,12,14,20,22,26,30 \
  >> /data/repeat_sft/logs/random-lora-s42-intermediate-v1.generate.log 2>&1 < /dev/null &
echo "推理 PID=$!"
```

中断后重跑同一生成命令，保留已完成题目，最多 8 道未完成题重新生成。生成完成后，在原 `.venv` 中判分和生成报告：

```bash
python -B grade.py \
  --responses runs/random-lora-s42-intermediate-v1/evaluations/generation-vllm \
  --out runs/random-lora-s42-intermediate-v1/grades-v1
python -B analyze_run.py \
  --run runs/random-lora-s42-intermediate-v1 \
  --epochs 6,7,9,10,12,14,20,22,26,30 \
  --grades runs/random-lora-s42-intermediate-v1/grades-v1 \
  --allow-partial \
  --out runs/random-lora-s42-intermediate-v1/analysis-v1
```

正确率和输出行为见 `grades-v1/summary.jsonl`，逐题完整回答见 `grades-v1/epoch-XXX/graded.jsonl`，报告见 `analysis-v1/report.html`。没有做拟合时，`--allow-partial` 会在报告中说明缺失。判分/分析需要新输出目录；再次运行用 `grades-v2` / `analysis-v2`。

如果还要保留四组各 48 题的方式，先用原训练环境对这 10 个新节点补算拟合，再运行已有分组入口。现有脚本已支持识别 replay 的新节点；此路线与上述直接生成二选一，避免重复生成：

```bash
python -B -u evaluate_loss.py --run runs/random-lora-s42-intermediate-v1 \
  --epochs 6,7,9,10,12,14,20,22,26,30
nohup bash run_grouped_vllm.sh runs/random-lora-s42-intermediate-v1 intermediate-vllm-v1 \
  --epochs 6,7,9,10,12,14,20,22,26,30 \
  >> /data/repeat_sft/logs/random-lora-s42-intermediate-v1.grouped.log 2>&1 < /dev/null &
echo "分组评测 PID=$!"
```

每组 48 题，组内完成 10 个节点后进入下一组；同一命令恢复，不加 `--reset-generation`。原 run 的旧拟合/生成结果保持不动。将新结果与原 epoch 8/16/32 等点一起解读时，保持题目、生成配置、后端一致，并注明新增点来自补跑分支。
