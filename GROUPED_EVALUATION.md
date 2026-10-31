# 8,192 token 分组评测

本流程复用已训练好的检查点和已完成的拟合评测，只重新自由生成、判分和分析。

新实验使用 LoRA adapter，自动加载本 run 记录的 Base 版本与量化设置。新建实验先按 [TODO_VM.md](TODO_VM.md) 完成 `evaluate_loss.py`，再运行 `bash run_grouped_analysis.sh runs/random-lora-s42 grouped-8192-v1`，不需要 `--reset-generation`。下面的 `random-s42 / full-context-v2` 是替换历史任务的示例；不能把旧全参拟合结果用于新 LoRA run。

- 生成上限：每题 **8,192 个新 token**，不包含题目。到达上限立即停止；保存原始输出，标记 `stop_reason=length`、`length_truncated=true`。不补写 EOS，不把强制停止算成模型正常结束。若最后一个 token 恰好是模型自己生成的 EOS，则按自然结束记录。
- 检查点顺序：**64、32、16、8、4、2、1**。不生成第 0 轮；已有第 0 轮拟合结果继续保留并进入拟合报告。
- 固定 192 题分成四组，各 **32 道 MATH500＋16 道 GPQA＝48 题**。从保存面板中按各任务的原顺序切分，不重新抽题、不改变提示词、选项、参考答案或题目 seed。
- 执行顺序：第一组 64→1 轮全部生成 → 第一组判分和报告 → 第二组 64→1 轮 → 第二组判分和报告 → …… → 第四组。最后合并完整 192 题结果并生成总报告。
- 温度 0.6、top-p 0.95、top-k 20、每题一次采样，重复惩罚仍为 1.0。

## 在 VM 上替换旧生成并启动

本地修改先由你提交并 push，然后在 VM 工程目录执行。以下假设需要替换的旧任务是 `full-context-v2`；若实际任务名不同，请在命令中替换该名称。

```bash
cd /home/ubuntu/data_recipe/Repeat_SFT
git pull
mkdir -p /data/repeat_sft/logs

nohup bash run_grouped_analysis.sh runs/random-s42 full-context-v2 --reset-generation \
  >> /data/repeat_sft/logs/random-s42-grouped8192.log 2>&1 < /dev/null &

echo "后台 PID：$!"
```

启动前应已停止该旧任务。新入口和原 `run_analysis.sh` 使用同一把 `pipeline.lock`，若同名任务仍运行，将拒绝重置。

`--reset-generation` **只在主动重新开始时使用一次**。所选任务原来的 `generation`、判分、下游报告和相关完成标记会移入 `generation-backups/<唯一编号>/`，随后使用新协议重建。它们不参与本次统计；备份不包含模型权重副本。原 `teacher_forcing`、`state/loss-*.done`、训练日志和所有模型检查点均不改写、不移动。其他任务目录的生成记录也不参与新评测。

拟合来源优先使用所选任务的 `teacher_forcing`，其次查找 run 的独立拟合目录和其他后处理任务。会核对 run、检查点、完整 train/holdout 样本及 token 文件，并保存文件哈希；不会重新跑拟合。也可显式指定：

```bash
nohup bash run_grouped_analysis.sh runs/random-s42 full-context-v2 --reset-generation \
  --teacher-forcing /data/repeat_sft/runs/random-s42/postprocess/full-v1/teacher_forcing \
  >> /data/repeat_sft/logs/random-s42-grouped8192.log 2>&1 < /dev/null &
```

只使用上面两条启动命令中的一条。若拟合数据不完整或不属于该 run，会在替换旧生成或加载模型之前停止，并报告具体路径。

## 查看进度、暂停和恢复

```bash
tail -f /data/repeat_sft/logs/random-s42-grouped8192.log
```

逐题日志例如：

```text
GROUP 1/4: 32 MATH500 + 16 GPQA; epochs 64,32,16,8,4,2,1
group=1/4 epoch=64 1/48 tokens=8192 stop=length
group=1/4 epoch=64 2/48 tokens=1234 stop=chat_end
...
GROUP 1/4 SAVED: .../groups/group-001/analysis/report.html
GROUP 2/4: ...
```

日志在每道题完整生成并写盘后更新；单题生成期间可能没有新行。每题都写盘并 fsync，当前未完成的一题不会保存。

主动停止时，向启动日志中的 `PID` 发送 `kill -TERM PID`，控制器会终止活动的生成／判分／分析子进程。`tail -f` 中的 Ctrl+C 只退出日志查看。

恢复时 **去掉 `--reset-generation`**：

```bash
nohup bash run_grouped_analysis.sh runs/random-s42 full-context-v2 \
  >> /data/repeat_sft/logs/random-s42-grouped8192.log 2>&1 < /dev/null &
```

恢复会使用 `grouped_code` 中的冻结代码，跳过已完成的组和阶段；未完成组内部跳过已保存的题。未完成的判分／报告会另建 `*-retry-*` 目录，成功输出路径记录在 `grouped_state` 和该组 `complete.json`。已完成的回答、seed 和拟合指标不会因中断重排。

## 保存结果的位置

```text
/data/repeat_sft/runs/random-s42/postprocess/full-context-v2/
  teacher_forcing/                         # 原拟合结果，可能是原有链接；始终保留
  grouped_pipeline.json                    # 协议、固定分组、拟合来源及哈希
  grouped_code/, grouped_launcher.sh       # 本次冻结代码
  grouped_progress.json, grouped_state/    # 当前阶段和完成标记
  generation-backups/                      # 被替换的旧生成和下游报告
  groups/
    group-001/
      sample_ids.json, panel.jsonl         # 48 题 ID、完整题目及参考信息
      generation/epoch-064/responses.jsonl # 每题回答、全部 token、长度、停止原因
      generation/epoch-032/...             # 到 epoch-001
      grades/                             # 这 48 题全部七个节点的判分
      analysis/report.html                # 第一组完成即可打开
      analysis/*.csv, summary.json, evidence/
      complete.json
    group-002/...                         # 同上
    group-003/...
    group-004/...
  generation/                             # 四组全部完成后合并的完整 192 题回答
  grades/                                 # 完整面板判分
  analysis/report.html                    # 完整总报告
  grouped_complete.json
```

每组和总报告都包含原拟合的全部检查点（通常 0、1、2、4、8、16、32、64），以及本次自由生成的七个检查点。轨迹按训练轮次从小到大分析，生成顺序从大到小不影响首次观察答对、退步及饱和计算。

本次未生成第 0 轮，报告会明确显示缺少 Base 自由生成对照；相对 Base 的正确率变化和首次高于 Base 的节点留空。第 1 轮答对只表示最早观测节点已经答对，不能据此认定能力由第 1 轮训练产生。

将整个组内 `analysis` 或最终 `analysis` 目录下载到本地，打开 `report.html`。其中保留各题原始回答、完整 token、参考解答、拟合指标及对应证据链接。
