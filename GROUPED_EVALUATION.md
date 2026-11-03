# Epoch 4 及以后的分组自由生成

本流程使用已有 LoRA 检查点和拟合评测，依次完成自由生成、判分和分析。模型自动加载该 run 保存的 Base 版本及对应 adapter；不重新训练。

- 评测面板是固定的 **128 道 MATH500＋64 道 GPQA**，不是用于拟合评测的 200 条 holdout。
- 四组各 **32 道 MATH500＋16 道 GPQA＝48 题**；题目、提示词、选项与每题 seed 保持不变。
- 默认生成节点为 **64→32→16→8→4**。第一组完成五个节点、判分和报告后，才进入第二组；四组共 **192×5＝960 条回答**。
- 每题一次采样：temperature=0.6、top-p=0.95、top-k=20、repetition penalty=1.0；最多 **8,192 个新 token**，不包括 prompt。
- 达到长度上限停止并保留完整截断输出，不补 EOS。若最后一个 token 本身是 EOS，仍按自然结束记录。
- 原来的八个拟合节点 0/1/2/4/8/16/32/64 全部复用并进入报告，不会重跑或改写。

## 启动新任务

将本地修改提交并 push，在 VM 工程目录 `git pull`。确认已有八个节点的完整拟合结果，并释放用于生成的 GPU。

下面的命令适用于本次已经成功去重的拟合目录 `teacher_forcing-dedup-v1`。若你的拟合没有重复、仍在原目录，把 `--teacher-forcing` 的值改为 `runs/random-lora-s42/evaluations/teacher_forcing`。必须选择已通过校验的那份，不能把旧全参 run 的拟合数据用于 LoRA run。

```bash
cd /home/ubuntu/Repeat_SFT
git pull
mkdir -p /data/repeat_sft/logs

nohup bash run_grouped_analysis.sh runs/random-lora-s42 grouped-e4plus-8192-v1 \
  --epochs 4,8,16,32,64 \
  --teacher-forcing runs/random-lora-s42/evaluations/teacher_forcing-dedup-v1 \
  >> /data/repeat_sft/logs/random-lora-s42.grouped-e4plus.log 2>&1 < /dev/null &
echo "$!" > /data/repeat_sft/logs/random-lora-s42.grouped-e4plus.pid
```

新任务保存在 `run/postprocess/grouped-e4plus-8192-v1`，与旧任务分开。不要使用 `--reset-generation`，也不要同时启动多个相同任务。

`--epochs` 可以配置其他已保存节点，输入顺序会规范为降序；新任务不指定时默认 64/32/16/8/4。节点、分组、拟合来源和代码在启动时冻结。已有任务恢复时未指定 `--epochs` 就沿用原列表；显式传入不同列表会拒绝启动。旧七节点任务不会因 git pull 自动变成五节点任务。

## 查看进度与停止

```bash
tail -f /data/repeat_sft/logs/random-lora-s42.grouped-e4plus.log
```

日志示例：

```text
GROUP 1/4: 32 MATH500 + 16 GPQA; epochs 64,32,16,8,4
group=1/4 epoch=64 1/48 tokens=8192 stop=length
group=1/4 epoch=64 2/48 tokens=1234 stop=chat_end
...
GROUP 1/4 SAVED: .../groups/group-001/analysis/report.html
GROUP 2/4: ...
```

每题完成后立即追加写盘并 fsync。单题生成期间可能暂时没有新日志；`tail -f` 的 Ctrl+C 只退出查看。

主动暂停请向此次启动的控制器发 SIGTERM：

```bash
kill -TERM "$(cat /data/repeat_sft/logs/random-lora-s42.grouped-e4plus.pid)"
```

控制器会停止活动子进程。等待退出后再恢复。不要把 `kill -9` 只杀父进程当作停止整个任务的方法：Linux 上存活的 worker 或生成子进程仍会持有任务锁，新启动会报告 `already running`，防止重复写入；需要停止仍在运行的该任务进程后再续跑。VM 重启后进程全部退出，锁会自动释放，无需删锁文件。

## 中断后恢复

重新执行上面的同一条 `nohup` 启动命令及 PID 保存命令，保持相同任务名、节点列表和拟合路径。**不需要额外的 `--resume`，也不加 `--reset-generation`。**

- 使用任务内 `grouped_code` 冻结代码，git pull 不改变已经启动任务的评测协议。
- 跳过已完成的组、阶段和已保存的逐题回答，不覆盖原回答。
- 强制结束或断电时，当前尚未完整写盘的那道题会重新生成；不承诺恢复单题内的生成进度或 KV cache。
- 若最后一行只写了一部分，先保留 `.torn-*` 备份，再从最后完整记录续跑；完整回答保留原样。
- 判分或报告中途失败会使用 `*-retry-*` 新目录，已有生成不必重跑。实际报告路径以组内 `complete.json` 或最终 `grouped_complete.json` 为准。
- 修改同名任务的节点列表或拟合输入会被拒绝。需要不同协议时另建任务；旧输出始终保留。

旧的 `--reset-generation` 仍只用于明确要求归档旧生成并重做的场景，不用于本次启动或恢复。它会备份所选任务的生成及派生报告；拟合结果和模型权重不变。

## 保存结果

```text
/data/repeat_sft/runs/random-lora-s42/postprocess/grouped-e4plus-8192-v1/
  grouped_pipeline.json                    # 固定协议、分组、拟合来源与哈希
  grouped_code/, grouped_launcher.sh       # 冻结的代码
  grouped_progress.json, grouped_state/    # 当前阶段、完成标记
  groups/
    group-001/
      sample_ids.json, panel.jsonl         # 48 题的 ID、题目和参考信息
      generation/epoch-064/responses.jsonl # 原始回答、全部 token、长度与停止原因
      generation/epoch-032/...             # 以及 epoch-016、008、004
      grades/                             # 五个节点的逐题判分
      analysis/report.html                # 第一组完成即可查看
      complete.json                       # 成功判分与报告的实际路径
    group-002/...
    group-003/...
    group-004/...
  generation/                             # 四组完成后合并的完整 192 题回答
  grades/
  analysis/report.html
  grouped_complete.json                    # 最终成功路径
```

报告按训练轮次升序分析，生成的降序执行不会改变轨迹解释。第 4 轮已经答对，只能说明最早观测的节点答对，不能据此断言模型在第 4 轮才学会。缺少 epoch 0 自由生成时，相对 Base 的正确率变化与迁移起点留空；也不能计算完整的 1—4 轮早期正确率排名。

把整个组内或最终 `analysis` 目录下载到本地再打开 `report.html`；包含原题、参考答案、模型完整回答、逐节点判分与证据链接。
