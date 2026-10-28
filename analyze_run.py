"""CPU-only analysis of saved checkpoint metrics and graded answers, with portable evidence."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import html
import json
import math
from pathlib import Path
import shutil

from grade import summarize as summarize_grades
from recipe.analysis import (correctness_milestones, fit_milestones, fit_summary, paired_changes,
                             quantile, task_milestones, transition)
from recipe.common import (digest, environment, file_hash, fresh_dir, iter_jsonl, now, read_json,
                           snapshot_code, stable_seed, write_json, write_jsonl)
from recipe.paths import artifact_path

GRADE_FIELDS = {"extracted_answer", "extraction_method", "extractable", "correct", "grading_error", "grader_version"}
STYLE = """body{font:16px system-ui,sans-serif;max-width:1500px;margin:32px auto;padding:0 24px;color:#202b3c}
table{border-collapse:collapse;width:100%;margin:18px 0;font-size:14px}th,td{border:1px solid #dce2eb;padding:8px;text-align:left;vertical-align:top}
th{background:#edf2f8;position:sticky;top:0}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f7fa;padding:16px}
a{color:#195cb2}details{margin:12px 0}input{padding:10px;width:60%;font:inherit}.scroll{overflow:auto}small{color:#526172}"""


def csv_write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list, tuple)) else v
                             for k, v in r.items()})


def escaped(value):
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.6g}"
    return html.escape(str(value), quote=True)


def table_html(rows, columns, link=None):
    headings = "".join(f"<th>{escaped(label)}</th>" for _, label in columns)
    body = []
    for row in rows:
        cells = []
        for key, _ in columns:
            content = escaped(row.get(key))
            if link and key == link[0]:
                content = f'<a href="{escaped(row[link[1]])}">{content}</a>'
            cells.append(f"<td>{content}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f'<div class="scroll"><table><thead><tr>{headings}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def page(path, title, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'<!doctype html><html lang="zh"><meta charset="utf-8"><title>{escaped(title)}</title>'
                    f'<style>{STYLE}</style><body><h1>{escaped(title)}</h1>{body}</body></html>', encoding="utf-8")


class Analysis:
    def __init__(self, args):
        self.args = args
        self.run = artifact_path(args.run)
        self.grades = artifact_path(args.grades) if args.grades else self.run / "grades-v1"
        self.fit_root = artifact_path(args.teacher_forcing) if args.teacher_forcing else self.run / "evaluations" / "teacher_forcing"
        self.inputs, self.issues = {}, []
        self.metadata = self.json(self.run / "run.json")
        self.epochs = sorted(set(int(e) for e in args.epochs.split(","))) if args.epochs else sorted(set(self.metadata["config"]["checkpoint_epochs"]))
        if not self.epochs or any(e < 0 for e in self.epochs):
            raise ValueError("Epochs must be nonnegative")
        self.samples = {}
        for split in ("train", "holdout", "panel"):
            path = self.run / "data" / f"{split}.jsonl"
            self.remember(path)
            if self.metadata["data_files"][path.name] != file_hash(path):
                raise ValueError(f"Frozen run data changed: {path}")
            rows = list(iter_jsonl(path))
            ids = [r["sample_id"] for r in rows]
            if not rows or len(set(ids)) != len(rows):
                raise ValueError(f"Empty/duplicate sample IDs: {path}")
            self.samples[split] = {r["sample_id"]: r for r in rows}
        self.out = fresh_dir(artifact_path(args.out))
        write_json(self.out / "analysis_status.json", {"status": "running", "started_at": now()})
        for split in self.samples:
            write_jsonl(self.out / "inputs" / f"{split}.jsonl", self.samples[split].values())
        write_json(self.out / "inputs" / "run.json", self.metadata)
        if (self.run / "tokenizer").is_dir():
            for path in (self.run / "tokenizer").rglob("*"):
                if path.is_file():
                    self.remember(path)
                    expected = self.metadata.get("tokenizer_files", {}).get(path.relative_to(self.run / "tokenizer").as_posix())
                    if expected is not None and file_hash(path) != expected:
                        raise ValueError(f"Frozen tokenizer changed: {path}")
            shutil.copytree(self.run / "tokenizer", self.out / "inputs" / "tokenizer")
        self.fit = {s: {} for s in ("train", "holdout")}
        self.eval = {}

    def remember(self, path):
        path = Path(path).resolve()
        self.inputs[str(path)] = file_hash(path)

    def json(self, path):
        self.remember(path)
        return read_json(path)

    def absent(self, message):
        if not self.args.allow_partial:
            raise ValueError(message + "; finish evaluation first, select --epochs explicitly, or use --allow-partial for inspection")
        self.issues.append(message)

    def check_identity(self, identity, epoch):
        if identity["run_sha256"] != file_hash(self.run / "run.json") or identity["epoch"] != epoch:
            raise ValueError(f"Evaluation belongs to a different run/epoch: {epoch}")
        checkpoint = identity["checkpoint"]
        if checkpoint["config_sha256"] != self.metadata["config_sha256"]:
            raise ValueError(f"Checkpoint identity mismatch: epoch {epoch}")
        if checkpoint["step"] != epoch * len(self.samples["train"]):
            raise ValueError(f"Unexpected checkpoint update count: epoch {epoch}")

    def load_fit(self):
        signatures = set()
        for epoch in self.epochs:
            directory = self.fit_root / f"epoch-{epoch:03d}"
            if not (directory / "identity.json").exists():
                self.absent(f"Missing teacher-forcing epoch {epoch}: {directory}")
                continue
            identity = self.json(directory / "identity.json")
            self.check_identity(identity, epoch)
            write_json(self.out / "inputs" / "teacher_forcing" / directory.name / "identity.json", identity)
            signatures.add(digest({k: identity[k] for k in ("chunk_size", "data_sha256", "selected_ids")}))
            for split in self.fit:
                expected = self.samples[split]
                if identity["data_sha256"][split] != file_hash(self.run / "data" / f"{split}.jsonl"):
                    raise ValueError(f"Teacher-forcing data mismatch: {split}")
                selected = identity["selected_ids"][split]
                if len(set(selected)) != len(selected) or not set(selected).issubset(expected):
                    raise ValueError("Duplicate/unexpected teacher-forcing selected IDs")
                path = directory / split / "samples.jsonl"
                if not path.exists():
                    self.absent(f"Missing teacher-forcing samples: {path}")
                    continue
                self.remember(path)
                found = {}
                for row in iter_jsonl(path):
                    sid = row["sample_id"]
                    if sid not in selected or sid in found or row["question_id"] != expected[sid]["question_id"]:
                        raise ValueError(f"Duplicate/unexpected teacher-forcing sample: {sid}")
                    count = row["n_tokens"]
                    if (type(count) is not int or count < 1 or type(row["correct_tokens"]) is not int
                            or not 0 <= row["correct_tokens"] <= count):
                        raise ValueError(f"Invalid teacher-forcing token counts: {sid}")
                    if (not math.isfinite(row["mean_loss"]) or row["mean_loss"] < 0
                            or not math.isclose(row["mean_loss"], row["nll_sum"] / count, rel_tol=1e-5, abs_tol=1e-6)
                            or not math.isclose(row["token_accuracy"], row["correct_tokens"] / count, abs_tol=1e-6)):
                        raise ValueError(f"Inconsistent teacher-forcing metrics: {sid}")
                    if "n_supervised" in expected[sid] and count != expected[sid]["n_supervised"]:
                        raise ValueError(f"Teacher-forcing supervision length changed: {sid}")
                    row = dict(row, epoch=epoch, split=split, source_file=str(path))
                    if row.get("token_file"):
                        token = (path.parent / row["token_file"]).resolve()
                        if not token.is_relative_to(path.parent.resolve()):
                            raise ValueError("Token file escapes teacher-forcing directory")
                        if token.exists():
                            self.remember(token)
                            target = self.out / "evidence" / "tokens" / directory.name / split / (digest(sid) + ".npz")
                            target.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(token, target)
                            row["token_file"] = target.relative_to(self.out).as_posix()
                        else:
                            self.absent(f"Missing token evidence: {token}")
                            row["token_file"] = None
                    found[sid] = row
                if set(found) != set(expected):
                    self.absent(f"Teacher-forcing epoch {epoch} {split}: {len(found)}/{len(expected)} samples")
                self.fit[split][epoch] = found
        if len(signatures) > 1:
            raise ValueError("Teacher-forcing epochs used different samples/data/chunk settings")

    def load_grades(self):
        path = self.grades / "grading_manifest.json"
        if not path.exists():
            self.absent(f"Missing grading manifest: {path}")
            return
        manifest = self.json(path)
        write_json(self.out / "inputs" / "grading_manifest.json", manifest)
        sources = {}
        for source in manifest["sources"]:
            epoch = source["identity"]["epoch"]
            if epoch in sources:
                raise ValueError(f"Duplicate graded epoch: {epoch}")
            sources[epoch] = source
        signatures = set()
        panel = self.samples["panel"]
        for epoch in self.epochs:
            if epoch not in sources:
                self.absent(f"No graded responses for epoch {epoch}")
                continue
            source = sources[epoch]
            identity = source["identity"]
            self.check_identity(identity, epoch)
            write_json(self.out / "inputs" / "generation" / f"epoch-{epoch:03d}" / "identity.json", identity)
            if identity["panel_sha256"] != file_hash(self.run / "data" / "panel.jsonl"):
                raise ValueError("Generation used a different evaluation panel")
            signatures.add(digest({k: identity.get(k) for k in ("settings", "generation_config", "selected_ids", "seed_policy")}))
            selected = identity["selected_ids"]
            if len(set(selected)) != len(selected) or not set(selected).issubset(panel):
                raise ValueError("Duplicate/unexpected generation selected IDs")
            path = self.grades / f"epoch-{epoch:03d}" / "graded.jsonl"
            if not path.exists():
                self.absent(f"Missing graded answers: {path}")
                continue
            self.remember(path)
            raw_hashes = None
            # A moved run can be analyzed from graded.jsonl alone: it already contains
            # all raw responses and token IDs. If originals exist, verify them as well.
            candidates = [self.run / "evaluations" / "generation" / f"epoch-{epoch:03d}" / "responses.jsonl",
                          Path(source["responses"])]
            raw = next((p for p in candidates if p.is_file() and file_hash(p) == source["sha256"]), None)
            if raw is not None:
                self.remember(raw)
                raw_hashes = {r["sample_id"]: digest(r) for r in iter_jsonl(raw)}
            elif any(p.is_file() for p in candidates):
                raise ValueError(f"Original responses hash differs from the grading manifest: epoch {epoch}")
            else:
                self.issues.append(f"Epoch {epoch}: original responses file unavailable; using embedded full graded responses, hash recorded")
            found = {}
            for row in iter_jsonl(path):
                sid = row["sample_id"]
                if sid not in selected or sid in found:
                    raise ValueError(f"Duplicate/unexpected graded sample: {sid}")
                question = panel[sid]
                if (row["epoch"] != epoch or row["question_id"] != question["question_id"]
                        or row["task"] != question["task"] or row["messages"] != question["messages"]
                        or row["gold_answer"] != question["gold_answer"]):
                    raise ValueError(f"Graded question/reference/prompt mismatch: {sid}")
                if row["grader_version"] != manifest["grader_version"]:
                    raise ValueError(f"Mixed grading rules: {sid}")
                if raw_hashes is not None and raw_hashes.get(sid) != digest({k: v for k, v in row.items() if k not in GRADE_FIELDS}):
                    raise ValueError(f"Graded raw response differs from original: {sid}")
                if row["generated_tokens"] != len(row["generated_token_ids"]):
                    raise ValueError(f"Generated token count mismatch: {sid}")
                if row["seed"] != stable_seed(identity["settings"]["seed"], sid):
                    raise ValueError(f"Generation seed policy mismatch: {sid}")
                if row["extractable"] != (row["extracted_answer"] is not None):
                    raise ValueError(f"Inconsistent extraction status: {sid}")
                if row["correct"] is not True and row["correct"] is not False and row["correct"] is not None:
                    raise ValueError(f"Invalid correctness value: {sid}")
                if row["grading_error"] is not None:
                    self.absent(f"Grading error at epoch {epoch}, sample {sid}: {row['grading_error']}")
                elif row["correct"] is None:
                    raise ValueError(f"Missing correctness without a grading error: {sid}")
                if row["correct"] is True and (not row["extractable"] or row["grading_error"]):
                    raise ValueError(f"Inconsistent correctness: {sid}")
                response = self.out / "evidence" / "responses" / f"epoch-{epoch:03d}" / (digest(sid) + ".json")
                write_json(response, row)
                # Keep only compact metrics in RAM; full answers and tokens live in evidence.
                found[sid] = {k: v for k, v in row.items() if k not in
                              {"raw_response", "response_text", "generated_token_ids", "prompt_token_ids", "messages"}}
                found[sid].update(response_file=response.relative_to(self.out).as_posix(), source_file=str(path))
            if set(found) != set(panel):
                self.absent(f"Graded epoch {epoch}: {len(found)}/{len(panel)} panel questions")
            self.eval[epoch] = found
        if len(signatures) > 1:
            raise ValueError("Generation epochs used different prompts/panels/settings/seed policies")

    def export_fit(self):
        curves, samples, milestones, overall = [], [], [], []
        for split, by_epoch in self.fit.items():
            expected = self.samples[split]
            split_milestones = []
            for sid, question in expected.items():
                observations = [by_epoch.get(e, {}).get(sid) for e in self.epochs]
                evidence = Path("evidence") / "fit" / split / (digest(sid) + ".json")
                report = Path("questions") / f"fit-{split}-{digest(sid)}.html"
                row = {"split": split, "sample_id": sid, "question_id": question["question_id"],
                       "source": question["source"], "question": question["question"],
                       **fit_milestones(self.epochs, observations, **self.fit_options()),
                       "evidence_file": evidence.as_posix(), "report_file": report.as_posix()}
                split_milestones.append(row)
                trajectory = []
                for i, (epoch, observation) in enumerate(zip(self.epochs, observations)):
                    entry = {"epoch": epoch, "split": split, "sample_id": sid, "available": observation is not None,
                             "evidence_file": evidence.as_posix(), "report_file": report.as_posix()}
                    if observation:
                        entry.update(observation)
                        previous = observations[i - 1] if i else None
                        if previous:
                            gap = epoch - self.epochs[i - 1]
                            entry.update(previous_epoch=self.epochs[i - 1],
                                         loss_change=observation["mean_loss"] - previous["mean_loss"],
                                         token_accuracy_change=observation["token_accuracy"] - previous["token_accuracy"],
                                         loss_change_per_epoch=(observation["mean_loss"] - previous["mean_loss"]) / gap,
                                         token_accuracy_change_per_epoch=(observation["token_accuracy"] - previous["token_accuracy"]) / gap)
                    samples.append(entry)
                    trajectory.append(entry)
                write_json(self.out / evidence, {"sample": question, "milestones": row, "trajectory": trajectory})
                body = '<p><a href="../report.html">返回汇总</a> · ' + f'<a href="../{evidence.as_posix()}">完整 JSON</a></p>'
                body += f'<h2>题目</h2><pre>{escaped(question["question"])}</pre><h2>原始完整解答</h2><pre>{escaped(question["response"])}</pre>'
                body += f'<p>参考答案：{escaped(question.get("answer"))}；损失稳定节点：{escaped(row["loss_tail_stable_epoch"])}；'
                body += f'token 准确率稳定节点：{escaped(row["token_accuracy_tail_stable_epoch"])}</p>'
                body += table_html(trajectory, [("epoch", "Epoch"), ("available", "有结果"), ("mean_loss", "平均损失"),
                    ("token_accuracy", "Token 准确率"), ("end_token_correct", "结束符预测正确"),
                    ("loss_change_per_epoch", "每轮损失变化"), ("token_accuracy_change_per_epoch", "每轮准确率变化")])
                for entry in trajectory:
                    if entry.get("token_file"):
                        body += f'<p>Epoch {entry["epoch"]}: <a href="../{escaped(entry["token_file"])}">逐 token NPZ</a></p>'
                page(self.out / report, f"{split}: {sid}", body)
            milestones.extend(split_milestones)
            split_curves = []
            for epoch in self.epochs:
                curve = {"epoch": epoch, "split": split, **fit_summary(list(by_epoch.get(epoch, {}).values()), len(expected))}
                for key in ("loss_tail_stable_epoch", "token_accuracy_tail_stable_epoch", "joint_tail_stable_epoch"):
                    curve[key + "_reached_n"] = sum(r[key] is not None and r[key] <= epoch for r in split_milestones)
                curve["saturation_denominator"] = len(expected)
                split_curves.append(curve)
            curves.extend(split_curves)
            aggregate = [{"mean_loss": c["sample_mean_loss"], "token_accuracy": c["sample_mean_token_accuracy"]}
                         if c["complete"] else None for c in split_curves]
            overall.append({"split": split, **fit_milestones(self.epochs, aggregate, **self.fit_options())})
        csv_write(self.out / "fit_curves.csv", curves)
        csv_write(self.out / "fit_samples.csv", samples)
        csv_write(self.out / "fit_saturation.csv", milestones)
        csv_write(self.out / "fit_summary_milestones.csv", overall)
        return curves, milestones, overall

    def fit_options(self):
        return {"loss_tolerance": self.args.loss_tolerance, "accuracy_tolerance": self.args.accuracy_tolerance,
                "min_points": self.args.saturation_min_points, "accuracy_target": self.args.token_accuracy_target}

    def export_downstream(self):
        curves, questions, long_rows, transitions, pairs, task_events = [], [], [], [], [], []
        tasks = defaultdict(list)
        for sid, question in self.samples["panel"].items():
            tasks[question["task"]].append(sid)
            observations = [self.eval.get(e, {}).get(sid) for e in self.epochs]
            evidence = Path("evidence") / "questions" / (digest(sid) + ".json")
            report = Path("questions") / (digest(sid) + ".html")
            row = {"sample_id": sid, "question_id": question["question_id"], "task": question["task"],
                   "question": question["question"],
                   **correctness_milestones(self.epochs, observations, self.args.correct_min_points),
                   "evidence_file": evidence.as_posix(), "report_file": report.as_posix()}
            questions.append(row)
            trajectory = []
            details = []
            for i, (epoch, observation) in enumerate(zip(self.epochs, observations)):
                entry = {"epoch": epoch, "sample_id": sid, "task": question["task"], "available": observation is not None,
                         "evidence_file": evidence.as_posix(), "report_file": report.as_posix()}
                if observation:
                    entry.update(observation)
                    original = read_json(self.out / observation["response_file"])
                    details.append(f'<details><summary>Epoch {epoch} · correct={escaped(observation["correct"])} · '
                                   f'{observation["generated_tokens"]} tokens · {escaped(observation["stop_reason"])}</summary>'
                                   f'<p>提取答案：{escaped(observation["extracted_answer"])}</p>'
                                   f'<p><a href="../{escaped(observation["response_file"])}">原始回答与全部 token ID（JSON）</a></p>'
                                   f'<pre>{escaped(original["raw_response"])}</pre></details>')
                trajectory.append(entry)
                long_rows.append(entry)
                if i:
                    previous = observations[i - 1]
                    transitions.append({"task": question["task"], "sample_id": sid, "from_epoch": self.epochs[i - 1],
                        "to_epoch": epoch, "transition": transition(previous, observation),
                        "from_response_file": previous.get("response_file") if previous else None,
                        "to_response_file": observation.get("response_file") if observation else None,
                        "report_file": report.as_posix()})
            write_json(self.out / evidence, {"sample": question, "milestones": row, "trajectory": trajectory})
            body = '<p><a href="../report.html">返回汇总</a> · ' + f'<a href="../{evidence.as_posix()}">题目轨迹 JSON</a></p>'
            body += f'<h2>题目与实际提示词</h2><pre>{escaped(question["question"])}</pre>'
            body += f'<pre>{escaped(question["messages"])}</pre><p>参考答案：{escaped(question["gold_answer"])}</p>'
            # MATH500's original row includes the reference solution; GPQA usually has no reasoning reference.
            reference_solution = question.get("original", {}).get("solution")
            if reference_solution is not None:
                body += f'<details><summary>数据集参考解答</summary><pre>{escaped(reference_solution)}</pre></details>'
            body += f'<p>首次观察答对：{escaped(row["first_observed_correct_epoch"])}；此前最后观察答错：'
            body += f'{escaped(row["first_correct_after_last_observed_wrong_epoch"])}；持续答对节点：{escaped(row["sustained_correct_epoch"])}</p>'
            body += table_html(trajectory, [("epoch", "Epoch"), ("available", "有结果"), ("correct", "正确"),
                ("generated_tokens", "长度"), ("normal_chat_end", "正常结束"), ("length_truncated", "截断"),
                ("extractable", "答案可提取"), ("extracted_answer", "提取答案")]) + "".join(details)
            page(self.out / report, f"{question['task']}: {sid}", body)
        for task, ids in tasks.items():
            task_curves = []
            for epoch in self.epochs:
                rows = [self.eval.get(epoch, {})[sid] for sid in ids if sid in self.eval.get(epoch, {})]
                complete = len(rows) == len(ids) and all(r["grading_error"] is None and r["correct"] is not None for r in rows)
                stats = summarize_grades(rows) if rows else {}
                curve = {"epoch": epoch, "task": task, **stats, "expected_n": len(ids), "observed_n": len(rows),
                         "complete": complete, "observed_accuracy": stats.get("accuracy"),
                         "accuracy": stats.get("accuracy") if complete else None,
                         "generated_tokens_p90": quantile([r["generated_tokens"] for r in rows], .9),
                         "generated_tokens_p95": quantile([r["generated_tokens"] for r in rows], .95),
                         "generated_tokens_max": max((r["generated_tokens"] for r in rows), default=None)}
                task_curves.append(curve)
            baseline = next((c["accuracy"] for c in task_curves if c["epoch"] == 0), None)
            for c in task_curves:
                c["accuracy_change_vs_base"] = c["accuracy"] - baseline if c["accuracy"] is not None and baseline is not None else None
            curves.extend(task_curves)
            task_events.append({"task": task, **task_milestones(self.epochs, task_curves, self.args.correct_min_points)})
            comparisons = [("adjacent", a, b) for a, b in zip(self.epochs, self.epochs[1:])]
            if 0 in self.epochs:
                comparisons += [("vs_base", 0, e) for e in self.epochs if e != 0]
            for kind, a, b in comparisons:
                pair = {"task": task, "comparison": kind, "from_epoch": a, "to_epoch": b,
                        **paired_changes(ids, self.eval.get(a, {}), self.eval.get(b, {}))}
                target = Path("evidence") / "changes" / f"{digest(task)[:16]}-{kind}-{a:03d}-{b:03d}.json"
                write_json(self.out / target, {**pair, "question_reports": {sid: "questions/" + digest(sid) + ".html" for sid in ids}})
                change_report = Path("changes") / (target.stem + ".html")
                body = '<p><a href="../report.html">返回汇总</a> · ' + f'<a href="../{target.as_posix()}">对应题目清单 JSON</a></p>'
                body += '<p>结束行为与答案提取的分组可能重叠；这些对应关系用于排查输出行为，不代表因果分解。</p>'
                labels = {"gained": "由错变对", "lost": "由对变错", "still_correct": "两次均正确", "still_incorrect": "两次均错误",
                          "unknown": "缺失或判分异常", "gained_with_normal_end_improvement": "由错变对，且正常结束改善",
                          "gained_with_extraction_improvement": "由错变对，且答案提取改善",
                          "gained_with_unchanged_end_and_extraction": "由错变对，结束与提取状态相同"}
                for group, members in pair["members"].items():
                    body += f'<h2>{escaped(labels[group])}（{len(members)}）</h2><ul>'
                    for sid in members:
                        body += f'<li><a href="../questions/{digest(sid)}.html">{escaped(self.samples["panel"][sid]["question"])}</a></li>'
                    body += '</ul>'
                page(self.out / change_report, f"{task}: {a} → {b} ({kind})", body)
                pairs.append({k: v for k, v in pair.items() if k != "members"} |
                             {"evidence_file": target.as_posix(), "report_file": change_report.as_posix()})
        csv_write(self.out / "downstream_curves.csv", curves)
        csv_write(self.out / "question_milestones.csv", questions)
        csv_write(self.out / "question_checkpoints.csv", long_rows)
        csv_write(self.out / "question_transitions.csv", transitions)
        csv_write(self.out / "paired_changes.csv", pairs)
        csv_write(self.out / "task_milestones.csv", task_events)
        return curves, questions, task_events, pairs

    def report(self, fit_curves, fit_samples, overall, curves, questions, task_events, pairs):
        body = f'<p>检查点：{escaped(self.epochs)}。损失/准确率稳定是可调阈值的描述性判断；只覆盖已评测检查点，不能证明永久饱和。</p>'
        body += '<p>首次答对是首次观察到答对；单次采样、稀疏检查点不能精确定位能力出现的训练轮次。正常结束和正确率的共同变化不构成因果分解。</p>'
        if self.issues:
            body += '<h2>覆盖与证据提示</h2><ul>' + ''.join(f'<li>{escaped(x)}</li>' for x in self.issues) + '</ul>'
        body += '<p>汇总：<a href="summary.json">summary.json</a> · <a href="analysis_manifest.json">方法与输入版本</a></p>'
        body += '<h2>训练集与 holdout 拟合</h2>' + table_html(fit_curves, [("split", "集合"), ("epoch", "Epoch"),
            ("complete", "完整"), ("sample_mean_loss", "逐题平均损失"), ("token_weighted_loss", "Token 加权损失"),
            ("sample_mean_token_accuracy", "逐题 Token 准确率"), ("end_token_accuracy", "结束符准确率"),
            ("joint_tail_stable_epoch_reached_n", "已进入稳定区间题数")])
        body += '<h3>整体拟合稳定节点</h3>' + table_html(overall, [("split", "集合"),
            ("loss_tail_stable_epoch", "逐题平均损失稳定"), ("token_accuracy_tail_stable_epoch", "逐题平均准确率稳定"),
            ("first_accuracy_target_epoch", "首次达到准确率阈值"), ("last_token_accuracy", "最终平均 Token 准确率")])
        body += '<h2>MATH500 泛化与 GPQA 迁移</h2>' + table_html(curves, [("task", "任务"), ("epoch", "Epoch"),
            ("complete", "完整"), ("accuracy", "正确率"), ("accuracy_change_vs_base", "相对 Base 变化"),
            ("mean_generated_tokens", "平均长度"), ("normal_chat_end_rate", "正常结束率"),
            ("length_truncated_rate", "截断率"), ("extractable_rate", "答案提取率")])
        body += '<h2>任务改善节点（描述性）</h2>' + table_html(task_events, [("task", "任务"),
            ("first_observed_above_base_epoch", "首次高于 Base"), ("sustained_above_base_epoch", "此后持续高于 Base"),
            ("best_epochs", "最高正确率节点"), ("best_accuracy", "最高正确率")])
        body += '<h2>改善和退步分别来自哪些题</h2><p>点击任务名查看对应题目；结束改善和提取改善计数可能重叠。</p>'
        body += table_html(pairs, [("task", "任务与题目清单"), ("comparison", "比较"), ("from_epoch", "起点"),
            ("to_epoch", "终点"), ("gained_n", "由错变对"), ("lost_n", "由对变错"),
            ("gained_with_normal_end_improvement_n", "答对且结束改善"),
            ("gained_with_extraction_improvement_n", "答对且提取改善")], link=("task", "report_file"))
        body += '<h2>按题检索</h2><p>输入题目、sample_id、集合或任务名称；点击题目查看参考解答与各检查点回答。</p>'
        body += '<input id="search" placeholder="搜索题目 / ID / task" oninput="filterRows()">'
        body += '<div id="question-tables"><h3>评测题</h3>' + table_html(questions, [("task", "任务"), ("sample_id", "ID"),
            ("question", "题目"), ("first_observed_correct_epoch", "首次答对"), ("sustained_correct_epoch", "持续答对"),
            ("regression_intervals", "答对后退步区间")], link=("question", "report_file"))
        body += '<h3>训练与 holdout 原解答</h3>' + table_html(fit_samples, [("split", "集合"), ("sample_id", "ID"),
            ("question", "题目"), ("loss_tail_stable_epoch", "损失稳定"), ("token_accuracy_tail_stable_epoch", "准确率稳定"),
            ("last_token_accuracy", "最终 Token 准确率")], link=("question", "report_file")) + '</div>'
        body += "<script>function filterRows(){const q=document.getElementById('search').value.toLowerCase();document.querySelectorAll('#question-tables tbody tr').forEach(r=>{r.hidden=!r.textContent.toLowerCase().includes(q);});}</script>"
        page(self.out / "report.html", "重复 SFT 检查点分析", body)
        write_json(self.out / "summary.json", {"epochs": self.epochs, "fit_curves": fit_curves,
                   "fit_summary_milestones": overall, "downstream_curves": curves, "task_milestones": task_events,
                   "issues": self.issues, "question_details": "question_milestones.csv",
                   "sample_saturation": "fit_saturation.csv", "paired_behavior_changes": "paired_changes.csv"})

    def execute(self):
        try:
            self.load_fit()
            self.load_grades()
            fit_curves, fit_samples, overall = self.export_fit()
            curves, questions, events, pairs = self.export_downstream()
            self.report(fit_curves, fit_samples, overall, curves, questions, events, pairs)
            code = snapshot_code(self.out / "code")
            write_json(self.out / "analysis_manifest.json", {"created_at": now(), "code_sha256": code,
                "arguments": vars(self.args), "epochs": self.epochs, "inputs_sha256": self.inputs,
                "environment": environment(), "issues": self.issues,
                "methods": {"saturation": "Earliest suffix of requested checkpoints with no missing values, at least min_points observations, and total max-min range <= tolerance. This is retrospective and not necessarily low loss/high accuracy.",
                    "saturation_thresholds": self.fit_options(), "correct_min_points": self.args.correct_min_points,
                    "first_correct": "First OBSERVED correct checkpoint; previous observed incorrect checkpoint is an open left bound, not an exact onset time.",
                    "sustained_correct": "All requested checkpoints from candidate through the last must be correct; at least correct_min_points observations.",
                    "transfer": "Descriptive accuracy improvements vs epoch 0, not causal or statistically significant onset estimates.",
                    "missing": "Unknown, never counted as an incorrect answer; incomplete task accuracy is null and observed_accuracy is separately labeled.",
                    "roles": "Train and holdout: teacher forcing on original solutions. MATH500: in-domain generation; GPQA: transfer generation."}})
            write_json(self.out / "analysis_status.json", {"status": "complete", "finished_at": now(), "issues": self.issues})
        except Exception as exc:
            write_json(self.out / "analysis_status.json", {"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--grades", help="Grading directory; default RUN/grades-v1")
    p.add_argument("--teacher-forcing", help="Teacher-forcing root; default RUN/evaluations/teacher_forcing")
    p.add_argument("--out", required=True, help="NEW output directory; contains self-contained report and evidence")
    p.add_argument("--epochs", help="Comma-separated epochs; default all configured milestone epochs")
    p.add_argument("--allow-partial", action="store_true", help="Keep missing results explicitly unknown; default requires complete evaluation")
    p.add_argument("--loss-tolerance", type=float, default=.01, help="Maximum suffix NLL range (nats per token)")
    p.add_argument("--accuracy-tolerance", type=float, default=.005, help="Maximum suffix token-accuracy range (0.005 = 0.5 percentage points)")
    p.add_argument("--saturation-min-points", type=int, default=3, help="Minimum observations in a stable suffix")
    p.add_argument("--token-accuracy-target", type=float, default=.99)
    p.add_argument("--correct-min-points", type=int, default=2, help="Minimum observations for sustained correctness/improvement")
    args = p.parse_args()
    if (not math.isfinite(args.loss_tolerance) or args.loss_tolerance < 0
            or not math.isfinite(args.accuracy_tolerance) or not 0 <= args.accuracy_tolerance <= 1
            or not 0 <= args.token_accuracy_target <= 1 or args.saturation_min_points < 2 or args.correct_min_points < 2):
        p.error("Invalid saturation/accuracy settings; minimum point counts must be >= 2")
    analysis = Analysis(args)
    analysis.execute()
    print(f"Analysis saved: {analysis.out}\nOpen {analysis.out / 'report.html'}; tables and all question/response evidence are local.")


if __name__ == "__main__":
    main()
