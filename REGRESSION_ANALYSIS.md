# 找出验证集上先答对、后来答错的题

`analyze_regressions.py` 只读已保存的判分结果，支持 MATH500、GPQA、BBH 和 LiveCodeBench。不加载模型、不重新生成、不重新判分，也不需要 Docker 或 GPU。已有训练、拟合和推理结果保持原样。

## 在 VM 上运行

更新代码后，在工程目录执行：

```bash
python -B analyze_regressions.py \
  --run runs/random-lora-s42 \
  --out runs/random-lora-s42/analysis-regressions-v1
```

相对路径沿用 `/data/repeat_sft` 存储根目录。默认自动寻找该 run 的分组评测任务，以及常见位置的独立 `grade.py` 输出。会分别处理每次评测；旧的 Transformers/30,000-token 结果与新的 vLLM/8,192-token 结果不会拼成同一条曲线。

如只关注当前两个 vLLM 任务，可以明确指定（任务名按实际目录替换）：

```bash
python -B analyze_regressions.py \
  --run runs/random-lora-s42 \
  --job runs/random-lora-s42/postprocess/grouped-e4plus-vllm-v1 \
  --job runs/random-lora-s42/postprocess/transfer-e4plus-vllm-v1 \
  --out runs/random-lora-s42/analysis-regressions-v2
```

传入 `--job` 或 `--grades` 后，只读取显式选择的来源。独立判分目录用 `--grades runs/random-lora-s42/grades-early`；可以重复传入多个目录。补跑 run 的结果用它自己的 `--run` 单独生成报告，不会自动与原 run 混合。

默认不写死 epoch 列表，按实际结果与各任务保存的计划显示覆盖情况。可用 `--epochs 4,8,16,32,64` 筛选节点，但通常建议不加，避免漏掉后来补测的节点。

输出目录必须是**新的独立目录**。后续有更多判分结果时改用 `analysis-regressions-v3` 等新名字，保留之前的报告快照。脚本通常只需 CPU 短时间处理，无需停止 GPU 推理。

## 先看什么

打开：

```text
/data/repeat_sft/runs/random-lora-s42/analysis-regressions-v1/report.html
```

先检查顶部的覆盖表，再在退化清单中按数据集、是否恢复或文字搜索筛选。点击题目可查看：

- 题目、数据集、参考答案、实际提示词，以及保存的原题/参考解答。
- 该题在实际节点上的正确性轨迹，缺失和判分异常单独显示。
- 每次“答对→答错”的前后完整回答并排对照，附 token 数、停止原因、提取答案及原文件位置。
- 之后在哪个已测节点再次答对；如果没有，显示“尚未观察到恢复”，不声称永久遗忘。
- 截断、结束符、答案提取、长度、重复文本等变化；LiveCodeBench 还展示代码测试状态、已保存的实际输出和 stderr。

将**整个报告目录**下载后，HTML 之间的链接及逐题 JSON 可以离线打开。只下载 `report.html` 不包含它链接的逐题页面。

| 文件 | 内容 |
| --- | --- |
| `report.html` | 可筛选的退化清单、覆盖情况和诊断线索统计 |
| `coverage.csv` | 每个评测系列/数据集/epoch 的有效数、答对数、答错数、缺失数、判分异常数 |
| `regressions.csv` / `.jsonl` | 每次观测到的退化区间、恢复节点、线索与逐题页面链接 |
| `paired_changes.csv` | 所有已测节点两两比较；双方都有有效判分的同题数，以及 lost / gained / 保持正确 / 保持错误 / 未知 |
| `questions/*.html` | 每道退化题的完整前后回答与全部可用节点回答 |
| `questions/*.json` | 完整题目、全部可用节点的原始判分记录、回答、token IDs、代码测试详情及来源哈希 |
| `manifest.json` | 分析代码哈希、输入文件哈希、来源目录、协议、节点和未读取结果说明 |

LiveCodeBench 的“正确答案”是通过保存的公开/隐藏测试，并没有统一参考解答代码。报告保留已有执行详情和原题的 reference 文件信息，不复制可能很大的隐藏测试文件；原始 reference 仍在评测任务的 `setup/panel/references/`。

## 缺失节点、分组与重复结果

- 某题只有 epoch 8 答对、epoch 32 答错时，记为 **8→32 观测退化区间**。不把未评测的 epoch 16 记为错误，也不能断言退化恰好发生在第 32 轮。
- `correct=null` 或有 `grading_error` 的记录属于未知，不是模型答错。只比较该题相邻的有效观测；中间未知节点会列出。
- 每道题的节点可以不同，各数据集的节点也可以不同。覆盖表明确区分无完成判分与该评测未安排的节点；没有判分并不一定代表从未生成。
- MATH500/GPQA 优先读取分组流程指向的已完成合并判分；否则读取已完成组的判分。`grades-retry-*` 使用流程记录的当前目录，避免混入失败的旧尝试。
- BBH/LiveCodeBench 按各组已写入 `complete.json` 的 epoch 读取。某个 epoch 的判分还没完成就跳过，下一次报告再纳入。不重复统计最后合并出的副本。
- 不读取未判分的原始回答来猜测正确性。重复 ID、身份不符或完成文件校验失败会报错；读取过程中源文件被改写也会报错，不修复或截断原文件。
- 不同 job、不同生成设置/代码/后端或不同判分协议分别形成系列。相同题目在两个系列里出现，会分别列出，不合并或覆盖。
- `run_intermediate_vllm.sh` 产生的两面板任务会自动展开为 `math-gpqa` 与 `bbh-livecodebench` 两个来源，分别读取实际完成的节点；使用补跑 run 的 `--run` 即可。

`paired_changes.csv` 的 `matched` 才是成对比较分母。部分完成节点的题目组成可能不同，不能仅凭 `coverage.csv` 中各自的正确率判定升降。

## “出现了什么问题”的边界

报告自动给出**可核对的输出行为线索**，不会自动认定推理逻辑错误的具体原因：

- 后来才被截断、后来提取失败、后来没有 EOS、think 未闭合。
- 输出至少翻倍/减半，重复长行或答案标记增加。
- 重复连续 16-token 窗口比例达到 0.5 且较前次增加至少 0.2。重复长行至少 40 字符，归一化空白后比较；这是启发式特征，代码和公式也可能自然重复。
- 仍能提取答案但判错，或代码变为 wrong_answer / runtime_error / timeout 等。

指标和完整回答一起保存，便于人工区分输出循环、最终答案变化、抽取问题和具体代码失败。当前每题一次采样，观察到退化不等于已经证明过拟合；优先检查这些案例，再用重复采样或不同训练 seed 确认稳定性。
