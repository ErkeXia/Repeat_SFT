# 人工 10 条回答：Base、epoch 64、epoch 128 似然度

入口 `dataset/run_curated_likelihood.sh` 自动使用现有独立 vLLM 0.11.0 环境。它读取 `curated-10x128-s42/bundle` 中当时训练使用的 **10 条冻结回答**，对 epoch 64、128 各评分一次；已有 Base 分数只读取并复制到新报告，不重算、不覆盖。

## 在 VM 上执行

本地代码 push、VM `git pull` 后，在工程根目录运行：

```bash
mkdir -p /data/repeat_sft/logs
nohup bash dataset/run_curated_likelihood.sh \
  --experiment runs/curated-10x128-s42 \
  --epochs 64,128 \
  --job-name e64-e128-v1 \
  >> /data/repeat_sft/logs/curated-likelihood-e64-e128-v1.log 2>&1 < /dev/null &
echo "似然度评测 PID=$!"
tail -n 30 -f /data/repeat_sft/logs/curated-likelihood-e64-e128-v1.log
```

注意 `--experiment` 是实验父目录，**不带 `/train`**。默认存储根目录仍为 `/data/repeat_sft`，要求 `/data` 已挂载。脚本调用 `/data/repeat_sft/venvs/vllm-0.11.0/bin/python`，不需要手动切换当前 `.venv`；非默认路径通过 `VLLM_PYTHON_BIN` 指定。

只加载一次固定版本的 Base，然后切换两个 LoRA；每次最多并发 8 条。新增工作量是 **20 条固定回答评分**，不运行四个 benchmark，也不做长回答生成。

先检查输入和冻结计划可加 `--prepare-only`；它不加载 GPU 模型。中断后等待进程退出，**重复同一命令**，逐题保留已提交的结果并补完剩余样本。`kill -TERM <PID>` 或前台 Ctrl+C 停止计算；`tail -f` 的 Ctrl+C 只停止看日志。已完成时重跑不再加载 GPU。`--report-only` 可根据保存的分数重新生成汇总，未完成节点标为 pending。

## 计算方式

- 给定同一问题、同一聊天模板和原人工回答前文，求回答中每个参考 token 的概率。
- 只统计助手推理、最终答案及 `<|im_end|>`；用户 prompt、padding 和结束后的换行不计入。
- 沿用 Base 的 `prompt_logprobs=0`、关闭 prefix caching、完整输入，不截断解答。请求附带的 1 个生成 token 仅用于完成 vLLM 请求，丢弃不计分；自由生成的 8,192-token 限制不用于这里。
- Base BF16，LoRA 推理 BF16，与已有 vLLM adapter 评测一致。原训练保存的 FP32 adapter 文件不修改。此结果不要求与 Transformers/FP32 adapter 路径逐位相同。
- 新任务要求 vLLM、torch、transformers、numpy 版本与原 Base 评分记录一致；不一致会提前列出版本差异。原输入或检查点变更也会报错，避免混用。

实现使用 vLLM 的 [AsyncLLM.generate 的 lora_request 参数](https://docs.vllm.ai/en/v0.11.0/api/vllm/v1/engine/async_llm.html) 切换检查点；adapter 的名字、整数 ID 和路径遵循 [LoRARequest 官方接口](https://docs.vllm.ai/en/v0.11.0/features/lora.html)。

## 看哪些结果

报告路径：

```text
/data/repeat_sft/runs/curated-10x128-s42/checkpoint_likelihood/e64-e128-v1/
  report/
    report.html       Base / 64 / 128 整体对比、逐题对比、完整题目及人工回答
    summary.json      整体和历史 high/low 分组统计、完成状态
    curves.csv        每个模型一行的整体指标
    samples.csv       每个模型每题一行，含相对 Base 的变化
    samples.jsonl     全部指标、完整文本、逐 token 文件路径
  epochs/
    epoch-064/        单独的 report.html、summary.json、scores.csv、samples.jsonl
      records/        每条已提交的分数及身份
      tokens/         输入 token IDs、助手范围、逐 token logprob（NPZ）
      complete.json
    epoch-128/        相同结构
  setup/
    base/             原 Base 分数的完整副本
    samples.jsonl     当时训练使用的 10 条人工回答
    index.jsonl       原评分 token 身份与长度
    tokenizer/、code/、plan.json   冻结输入、代码、版本、检查点身份
  complete.json       两个检查点全部完成后生成
```

每完成一个 epoch 就更新总报告。最终共有 30 条逐题记录：10 条原 Base + 10 条 epoch 64 + 10 条 epoch 128。HTML 可以单独下载阅读；若要继续分析逐 token 概率，保留整个任务目录。

**平均 NLL 越低、归一化似然越高，表示模型越容易拟合这些固定回答。** `normalized_likelihood = exp(mean_logprob)`，不是整条解答的联合概率。报告同时提供逐样本平均和按 token 数加权的平均；`delta_nll_vs_base < 0` 表示比 Base 更易拟合。High/Low 仍是修改前的历史分组。这些指标衡量训练回答的拟合程度，不直接代表自由解题正确率或泛化。

离线测试使用真实 10 条文本、合成 tokenizer/logprob 和模拟 adapter 引擎，覆盖监督范围、adapter 切换、Base 复用、逐题续跑、完整报告及损坏结果拒绝。实际 GPU/vLLM 计算在 VM 上执行，本地不下载模型或数据。
