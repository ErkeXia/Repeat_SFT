# 五种题型各十题：50 条 × 16 轮

入口：`dataset/train_topic_balanced.py`。五类各随机选十道不同题，合并成**一个 50 条训练集**，从原始 `Qwen/Qwen3-4B-Base` 新建 LoRA。每轮重新打乱全部 50 条，每题更新一次；共 800 次更新。保留 **epoch 0 和 1–16 的全部检查点**，包括优化器、学习率、随机数和数据位置。不会接着旧实验的 adapter 训练。

## 题型与选择规则

Math-CoT-20k 没有官方统一题型字段。当前本地版本有 20,480 行，7 种 `data_source` 是来源标签，不能当作 7 种题型。为这次实验使用如下粗分类：

| 题型 | 本次数量 |
| --- | ---: |
| 代数与方程 | 10 |
| 几何 | 10 |
| 数论 | 10 |
| 组合与概率 | 10 |
| 微积分 | 10 |
| 数列与递推 | 不选 |
| 不等式 | 不选 |

分类代码在 `topic_selection.py`。只读取**题目文本**，匹配明确关键词；同时命中多类或无法识别的题目不选。这是可审查的规则标签，并非穷尽的数学分类或人工确认的真值，可能漏选或误分。选题报告列出具体命中词，建议开始训练前抽查。

候选来自原 `artifacts/data-s42/pool.jsonl`，不是局限在旧的随机 400 条或 high/low 100 条里。沿用原正确解答和实际聊天模板，完整保留推理、最终答案和结束符，不编辑解答。保留总长度 ≤10,240 token、已有正确性验证的题目，排除重复、外链图片，以及与原 holdout、完整 MATH500/GPQA 去重参考和固定 BBH/LCB 面板重合的题目。重合判断沿用规范化精确匹配和字符 shingle Jaccard ≥0.9，不保证消除语义改写题。选中样本之间再检查近重复。

每类按稳定排序和 seed 42 随机抽十条。候选不足就报错，不重复抽同题、不悄悄换类别。不按似然度挑选，也不匹配各类的来源和解答长度，因此本实验是混合题型训练，不能独立识别题型的因果效果。

## 在 VM 上运行

先 push / pull 代码，在工程根目录使用**训练环境 `.venv`**，不使用 vLLM 环境。须已有 `artifacts/data-s42` 和 `artifacts/transfer-s2026`。准备阶段只加载已保存的 tokenizer，不下载模型或新数据；实际训练按原流程读取固定版本 Base 缓存（缺失时在 VM 获取）。

可以先准备并阅读 50 道题，不加载模型：

```bash
source .venv/bin/activate
python -B -u dataset/train_topic_balanced.py --prepare-only
```

报告：`/data/repeat_sft/runs/topic-balanced-50x16-s42/bundle/review.html`，可下载到本地浏览。每题包含类别、分类依据、来源、原始行号、token 数、题目、短答案和完整原解答。

启动训练（已经做过 prepare-only 时复用冻结名单）：

```bash
source .venv/bin/activate
mountpoint -q /data || { echo '/data 尚未挂载'; exit 1; }
mkdir -p /data/repeat_sft/logs

nohup python -B -u dataset/train_topic_balanced.py \
  --data artifacts/data-s42 \
  --panel-dir artifacts/transfer-s2026 \
  --out runs/topic-balanced-50x16-s42 \
  --seed 42 \
  >> /data/repeat_sft/logs/topic-balanced-50x16-s42.log 2>&1 < /dev/null &
echo "训练 PID=$!"
tail -n 30 -f /data/repeat_sft/logs/topic-balanced-50x16-s42.log
```

不需要先 prepare-only，也可以直接执行上述命令完成选题和训练。配置沿用 `configs/baseline.json`，默认 BF16 不量化 Base、FP32 LoRA 参数，rank 16 / alpha 32 / dropout 0，七个 attention/MLP 投影模块；SDPA、梯度检查点、AdamW8bit、batch 1、无梯度累积。只监督助手回答及结束符。学习率 `1e-5`，前 100 步预热（本实验恰好前 2 轮），之后恒定；weight decay 0，梯度裁剪 1.0。

不重复保存完整 Base，检查点保存 adapter 和完整训练状态，沿用已有存储空间检查。1000G volume 的实际剩余空间仍取决于旧实验。配置、数据、代码在首次准备时冻结，后续 git pull 不改变该任务的训练实现。需要改变选题或配置时，请指定新的 `--out`。

## 结果、停止和恢复

所有默认相对路径指向 `/data/repeat_sft`。输出结构：

```text
runs/topic-balanced-50x16-s42/
  bundle/
    review.html                 50 道题及完整解答的阅读报告
    selection.json              选题清单、类别、分类统计、哈希
    classification.jsonl        候选池每条的标签、命中词、入选/排除原因
    config.json                 实际训练配置
    prepared/train/              完整 50 条、200 holdout、原评测面板、去重参考、tokenizer
      inspection.jsonl          50 条实际输入/监督文本/结束 token
    code/                       冻结代码
    plan.json                   版本、来源、配置、文件哈希
  train/
    run.json
    checkpoints/epoch-000/      Base 起点的零增量 adapter + 完整恢复状态
    checkpoints/epoch-001/      第 50 次更新
    ...
    checkpoints/epoch-016/      第 800 次更新
    recovery/                   每 25 步滚动恢复点，保留最近 2 份
    logs/                       每步样本身份、loss、学习率等
    progress.json               进度和耗时
  commands.jsonl
  complete.json                 全部 17 个节点核验通过后产生
```

前台 Ctrl+C 或 `kill -TERM <训练PID>` 会在当前更新完成后保存。**等待退出，重复同一条启动命令即可恢复，不需要加 `--resume`。** 已完成检查点和日志保留；已完成全部训练时再次运行只核验后退出。每轮检查点永久保留，只有 `recovery/` 的滚动恢复点淘汰。断电/kill -9 从最近完整状态恢复；未保存区间最多重跑 24 次更新。单任务文件锁阻止对同一目录重复启动。

可先在启动参数中加 `--stop-after 27` 验证暂停；去掉该参数重跑会从第 27 步接着训练。此参数是累计更新数。

拟合评测沿用原脚本，`--run` 指向 `train/`：

```bash
python -B evaluate_loss.py \
  --run runs/topic-balanced-50x16-s42/train \
  --epochs 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16
```

它统计选中的 50 条及原 200 条 holdout 的逐样本 loss / token 准确率，不是四个 benchmark 的自由生成。当前新入口只自动执行选题和训练。

四个 benchmark 的自由生成使用 [run_topic_benchmarks.sh](TOPIC_BENCHMARKS.md)：先测 1、2、3、4、8、12、16，再升序补齐其他轮次；vLLM 并发 8，每轮全部四任务完成后立即出报告，支持逐题续跑。

以后继续增加 16 轮、仍每轮保存，可使用已有续训入口，原目录完整保留：

```bash
python -B -u dataset/continue_training.py \
  --source-run runs/topic-balanced-50x16-s42/train \
  --out runs/topic-balanced-50x32-s42/train \
  --additional-epochs 16 --checkpoint-every 1
```
