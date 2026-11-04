# 用原 checkpoint 评测 BBH 和 LiveCodeBench

入口为 `run_transfer_vllm.sh`，默认使用 **`runs/random-lora-s42` 原 run 的 epoch 64、32、16、8、4**。不使用刚补跑的中间 checkpoint，不重新训练，不替换原来的 MATH500 / GPQA 面板，不需要先跑 teacher-forcing loss。

| 项目 | 配置 |
| --- | --- |
| 数据 | `artifacts/transfer-s2026`：BBH 100 题 + LiveCodeBench 100 题 |
| 分组 | 4 组，每组 25 BBH + 25 LiveCodeBench；BBH 每类在每组有 6/7 题 |
| 顺序 | 第一组 64→32→16→8→4 的生成、判分、报告全部完成，再做第二组 |
| 生成 | vLLM 并发 8；BF16 Base + BF16 adapter；沿用独立 vLLM 环境 |
| 采样 | 每题一次；temperature=0.6、top-p=0.95、top-k=20、repetition penalty=1 |
| 长度 | 最多 8192 个新 token；EOS 或到上限即停止，不伪造正常结束 |
| 总量 | 200 题 × 5 个节点 = 1000 条回答 |
| 新输出 | `/data/repeat_sft/runs/random-lora-s42/postprocess/transfer-e4plus-vllm-v1/` |

提示词使用准备时保存的完整文本，统一使用原 run 的 tokenizer。所有 200 题共同决定 vLLM 上下文预算，各组/节点一致；若提示词 + 8192 超出模型上下文，会报错，不截题、不静默减小生成长度。隐藏测试和参考答案不会进入模型提示词。

## 1. 准备与一次性执行环境

先 `git pull`。数据准备方法见 [TRANSFER_DATA.md](TRANSFER_DATA.md)；已经准备完成的 200 题直接复用，不重新下载。

```bash
mountpoint -q /data || { echo '/data 尚未挂载'; exit 1; }
test -f /data/repeat_sft/artifacts/transfer-s2026/manifest.json
test -x /data/repeat_sft/venvs/vllm-0.11.0/bin/python
```

LiveCodeBench 需要执行生成代码，因此增加 Docker 执行环境。已安装 Docker 且当前用户能执行 `docker info` 时，跳过安装。Ubuntu 新 VM 可用发行版的软件包：

```bash
sudo apt-get update
sudo apt-get install -y docker.io
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER"
# 添加用户组后重新 SSH 登录，再继续下面的命令。
```

也可按 [Docker 官方 Ubuntu 安装说明](https://docs.docker.com/engine/install/ubuntu/)安装。此步骤不改 Python 训练环境。先确认 Docker 可用及其存储位置，再构建小型 Python 执行镜像：

```bash
docker info --format '{{.DockerRootDir}}'
python3 -B build_lcb_sandbox.py
```

构建只在 VM 下载 Python 基础镜像、NumPy 2.2.6 和 sortedcontainers 2.4.0，不下载模型或数据。构建后会实际运行标准输入输出题、函数题，并检查候选进程已降权、无限循环能超时终止；保存镜像 ID、构建说明和结果到 `/data/repeat_sft/artifacts/lcb-sandbox/`。超时自检约需额外 6 秒。Docker 未安装、权限不够、镜像不匹配或自检失败都会提前停止，不先花 GPU 时间生成。

实验结果、冻结的 200 题副本、日志都在 `/data`。Docker 镜像位置由 daemon 的 `DockerRootDir` 决定；本脚本不改系统 Docker 配置。若系统盘紧张，先按 Docker 的存储配置把镜像存储安排到 `/data`。容器内部临时文件用有限大小的内存文件系统；建议为选中题副本、镜像、回答和报告另留 **5–10 GiB**，实际大小取决于隐藏测试。不会复制 Base 权重或完整 LiveCodeBench 候选池。

## 2. 后台启动

```bash
mkdir -p /data/repeat_sft/logs
nohup bash run_transfer_vllm.sh \
  --run runs/random-lora-s42 \
  --panel-dir artifacts/transfer-s2026 \
  --job-name transfer-e4plus-vllm-v1 \
  >> /data/repeat_sft/logs/transfer-e4plus-vllm-v1.log 2>&1 < /dev/null &
echo "迁移评测 PID=$!"
tail -f /data/repeat_sft/logs/transfer-e4plus-vllm-v1.log
```

默认就是原五个节点。`--epochs` 可显式传 `4,8,16,32,64`，执行时仍降序。不要把 `--run` 指向 `random-lora-s42-intermediate-v1`；这次入口会拒绝 replay run，避免混淆。

第一次启动会冻结评测代码、选题、提示词、生成参数、判分配置、Docker 镜像 ID、checkpoint 身份；只复制选中的 200 题和参考测试。旧 MATH500 / GPQA 的生成与拟合目录不变。

## 3. 停止和恢复

- 停止查看 `tail -f` 用 Ctrl+C，不会停止后台任务。
- 停止整个流程用 `kill -TERM <迁移评测PID>`。当前阶段会停止，已完成回答和逐题判分保留。
- 恢复直接**重跑完全相同的 nohup 命令**，不加 reset，不换 job-name。启动器自动调用任务内冻结代码，适用于之后又 `git pull` 的情况。
- 生成每完成一题就写盘；中断时最多 8 道未完成题重新生成，不恢复单题 KV cache。
- 判分每完成一题就写盘；中断题重新运行测试，已保存的判分跳过。Docker/数据损坏等基础设施错误会中止流程，不默默变成错误答案或丢题。
- 每组的全部五个节点生成、判分和报告完成后写 `groups/group-XXX/complete.json`；该组后续直接跳过。原始回答不覆盖，损坏记录报错。
- Linux 进程锁阻止重复启动。仅强杀启动器可能留下子进程，应优先 TERM；容器还有自身的最长运行时间限制。

## 4. 如何判分

**BBH**：使用保存的 `gold_answer`，根据子任务接受 A–E、True/False、Yes/No。只从最后的明确答案标记、完整 boxed 答案或单独答案中抽取；不在推理过程里搜索正确选项。未闭合 `<think>` 不作为最终答案。正确率是固定子面板的精确匹配率。

**LiveCodeBench**：取最后一个完整 Python 代码块；没有代码围栏时接受可解析的纯 Python。代码块有语法错误也保存并记作编译错误。缺少完整代码则记作无法提取。压缩隐藏测试按源格式解码，pickle 只允许数据、不允许加载类或执行函数。必须具备隐藏测试，绝不降级为仅跑公开样例。

- 公开和隐藏测试按原顺序执行，全部通过才算该题正确。每题一次生成，对应这个自定义子面板的 pass@1。
- 标准输入输出题比较去除首尾空白后的逐行文本；数值行用 Decimal 精确比较，不用宽松浮点容差。
- 函数题从 metadata 读取函数名，支持 `Solution` 方法或同名顶层函数，输入按逐行 JSON 参数解析；返回值经 JSON 序列化后比较。
- 默认每个测试 6 秒，整题最多 600 秒；容器 1 CPU、2 GiB 内存，候选子进程 1 GiB 地址空间，每路输出最多 1 MiB。限制可在**新 job**通过 `--judge-config` 改，不在旧任务中混用。
- 遇到第一个失败即停止，记录已测试数量、总测试数量、失败位置和原因；未执行测试不能当作通过。保存每个已执行测试的状态、耗时、实际输出、stderr，以及输入/期望输出的哈希；完整期望值仍在冻结 reference 中。
- 候选程序在禁网、只读根文件系统、限制进程数的容器中以 UID 65534 运行。监督进程读测试后只把单个输入交给候选进程；不把金标准输出、其他题或宿主工程、模型缓存、Docker socket 挂进去。

数据解码和比较规则参考 [LiveCodeBench 数据适配](https://github.com/LiveCodeBench/LiveCodeBench/blob/main/lcb_runner/benchmarks/code_generation.py)和[判分代码](https://github.com/LiveCodeBench/LiveCodeBench/blob/main/lcb_runner/evaluation/testing_util.py)。这里是**项目自定义 Python 执行协议 `transfer-bbh-python-v1`**：每例新进程、Docker 资源上限、代码抽取和运行时与官方 harness 不完全相同。报告明确标注，不能作为官方完整 BBH/LiveCodeBench 排行榜分数。Docker 限制参考[官方资源文档](https://docs.docker.com/engine/containers/resource_constraints/)。

## 5. 看结果与重新判分

```text
transfer-e4plus-vllm-v1/
  setup/plan.json                       选题、节点、镜像身份、配置、自检
  setup/panel/panel.jsonl               200 道实际题目和提示词
  setup/panel/references/*.json         完整原题与公开/隐藏测试
  setup/code/                          冻结代码
  groups/group-001/...                  第一组完成后即可看报告
    generation/epoch-*/responses.jsonl  原始回答、全部 token、长度、停止原因
    grades/epoch-*/graded.jsonl         原回答 + 提取答案/代码 + 判分和测试详情
    report/report.html                 第一组的五节点曲线与逐题全文
    complete.json
  generation/epoch-*/responses.jsonl    最后合并的完整 200 题回答
  grades/epoch-*/graded.jsonl           完整 200 题判分（复用各组，不重复执行）
  report/report.html                   全部结果入口
  report/curves.csv                    BBH、LCB、BBH 子任务的逐 epoch 指标
  report/question_milestones.csv       每题首次/持续答对、退步等观察节点
  report/questions/*.html              同一道题在五个 epoch 的完整回答
  report/questions/*.json              题目、回答、token IDs、逐题判分
  progress.json                       当前组与阶段
  complete.json                       全部 1000 条回答判分完成后出现
```

正确率、平均输出长度、正常结束率（同时保留 chat-end 与任意 EOS）、截断率、答案/代码提取率都在 CSV。报告中的“第一次答对”按 epoch 升序计算，虽然实际推理按降序执行。抽样只跑一次，不能仅凭少数题的波动认定迁移出现。

下载整个 `transfer-e4plus-vllm-v1` 目录能保留原题/测试链接；仅下载 `report/` 仍含完整回答和 token JSON，但原始 reference 链接需要上级 `setup/panel/references`。

生成和判分分开。修改抽取/判分规则后，用**新判分输出目录**，保留原回答与旧判分：

```bash
python3 -B grade_transfer.py \
  --panel-dir runs/random-lora-s42/postprocess/transfer-e4plus-vllm-v1/setup/panel \
  --responses runs/random-lora-s42/postprocess/transfer-e4plus-vllm-v1/generation \
  --out runs/random-lora-s42/postprocess/transfer-e4plus-vllm-v1/grades-v2
python3 -B report_transfer.py \
  --panel-dir runs/random-lora-s42/postprocess/transfer-e4plus-vllm-v1/setup/panel \
  --grades runs/random-lora-s42/postprocess/transfer-e4plus-vllm-v1/grades-v2 \
  --out runs/random-lora-s42/postprocess/transfer-e4plus-vllm-v1/report-v2
```

旧环境中断时保持原镜像不删除，任务记录的 image ID 必须仍在 Docker 中。重新构建镜像产生不同 ID 时，新评测使用新 job-name；已开始任务继续使用记录的旧镜像。
