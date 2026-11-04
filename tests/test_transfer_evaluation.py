"""Synthetic transfer panels, extraction, execution protocol and group resume. No downloads."""
from __future__ import annotations

import base64
from copy import deepcopy
import json
import os
from pathlib import Path
import pickle
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zlib

import grade_transfer
import transfer_evaluation as pipeline
from generate import stop_details
from recipe.common import ROOT, append_jsonl, file_hash, iter_jsonl, read_json, write_json, write_jsonl
from recipe.lcb_docker import DockerJudge, docker_command, validate_config
from recipe.lcb_worker import PROTOCOL, stdio_equal, supervise
from recipe.pipeline import Pipeline
from recipe.transfer_data import BBH_TASKS, adapt_bbh, adapt_lcb
from recipe.transfer_eval import load_panel, partition
from recipe.transfer_grading import decode_private, extract_bbh, extract_code, test_payload

TEST_TMP = ROOT / ".test_tmp"
TEST_TMP.mkdir(exist_ok=True)


def panel_fixture(directory, training_hash):
    panels = []
    for task in BBH_TASKS:
        for i in range(25):
            target = "True" if task == "boolean_expressions" else "Yes" if task == "web_of_lies" else "(E)"
            panels.append(adapt_bbh({"input": f"Synthetic {task} reasoning item {i}?", "target": target}, task, {}))
    for i in range(100):
        functional = bool(i % 2)
        kind = "functional" if functional else "stdin"
        raw = {"question_title": f"Add one {i}", "question_content": f"Synthetic program item {i}. Add one to n.",
               "platform": "leetcode" if functional else "atcoder", "question_id": str(i),
               "contest_date": "2025-01-01T00:00:00", "difficulty": "easy",
               "starter_code": "class Solution:\n    def solve(self, n): pass" if functional else "",
               "public_test_cases": json.dumps([{"input": "1", "output": "2", "testtype": kind}]),
               "private_test_cases": json.dumps([{"input": "2", "output": "3", "testtype": kind}]),
               "metadata": json.dumps({"func_name": "solve"} if functional else {})}
        panels.append(adapt_lcb(raw, {}))
    rows, files = [], {}
    for item in panels:
        row = item["panel"]
        name = f"references/{row['sample_id']}.json"
        write_json(directory / name, item["original"])
        files[name] = file_hash(directory / name)
        rows.append({**row, "reference_file": name, "reference_sha256": files[name]})
    write_jsonl(directory / "panel.jsonl", rows)
    write_json(directory / "plan.json", {"training_sha256": training_hash})
    files.update({n: file_hash(directory / n) for n in ("panel.jsonl", "plan.json")})
    write_json(directory / "manifest.json", {"format_version": "transfer-panel-v1", "files": files})
    return rows


class FakeJudge:
    image = {"id": "sha256:synthetic-image", "labels": {"protocol": "synthetic"}}
    def __init__(self, *_):
        self.config = read_json(ROOT / "configs" / "lcb_grading.json")
        self.calls = 0
    def preflight(self):
        return {"image": self.image, "config": self.config}
    def run(self, code, tests):
        self.calls += 1
        return {"protocol": PROTOCOL, "correct": True, "status": "passed", "total_tests": len(tests["cases"]),
                "tested": len(tests["cases"]), "passed": len(tests["cases"]), "test_results": []}


def synthetic_responses(job, group, epochs, fail=False):
    plan = read_json(job / "setup" / "plan.json")
    panel, _ = load_panel(job / "setup" / "panel")
    questions = {p["sample_id"]: p for p in panel}
    for epoch in epochs:
        out = job / "groups" / group["name"] / "generation" / f"epoch-{epoch:03d}"
        identity = {"run_sha256": plan["run_sha256"], "epoch": epoch, "panel_sha256": plan["panel_sha256"],
                    "transfer_manifest_sha256": file_hash(job / "setup" / "panel" / "manifest.json"),
                    "backend": "vllm", "code_sha256": plan["code_sha256"], "settings": plan["settings"],
                    "selected_ids": group["selected_ids"], "checkpoint": read_json(Path(plan["run"]) / "checkpoints" / f"epoch-{epoch:03d}" / "complete.json")}
        write_json(out / "identity.json", identity)
        log = out / "responses.jsonl"
        done = {r["sample_id"] for r in iter_jsonl(log)} if log.exists() else set()
        for sid in group["selected_ids"]:
            if sid in done:
                continue
            q = questions[sid]
            text = (f"Final answer: {q['gold_answer']}" if q["task"] == "bbh" else
                    "```python\nclass Solution:\n    def solve(self,n): return n+1\n```" if q["interface"] == "functional" else
                    "```python\nprint(int(input())+1)\n```")
            append_jsonl(log, {k: q[k] for k in ("sample_id", "question_id", "task", "messages", "gold_answer")} |
                         {"epoch": epoch, "backend": "vllm", "raw_response": text + "<|im_end|>",
                          "generated_token_ids": [7, 2], "generated_tokens": 2, **stop_details([7, 2], 3, 2, 8192)})
            done.add(sid)
            if fail and len(done) == 7:
                raise RuntimeError("synthetic generation interruption")
        write_json(out / "complete.json", {"samples": len(done), "responses_sha256": file_hash(log)})


class ExtractionTests(unittest.TestCase):
    def test_bbh_letters_boolean_yesno_and_no_reasoning_search(self):
        for text, task, expected in (("Final answer: (E)", BBH_TASKS[0], "E"),
             ("\\boxed{False}", BBH_TASKS[2], "False"), ("**Final answer:** Yes.", BBH_TASKS[3], "Yes"),
             ("I considered Yes but No also appears", BBH_TASKS[3], None),
             ("Final answer: Yes\nFinal answer: uncertain", BBH_TASKS[3], None),
             ("<think>Final answer: E", BBH_TASKS[0], None),
             ("Final answer: A\n\\boxed{E}", BBH_TASKS[0], "E")):
            self.assertEqual(extract_bbh(text, task)[0], expected)

    def test_code_extraction_last_complete_block_bare_and_unfinished(self):
        text = "```python\nprint(1)\n```\nCheck again\n```python\nprint(2)\n```\n```python\nbroken"
        self.assertEqual(extract_code(text)[0], "print(2)")
        self.assertEqual(extract_code("print(2)")[0], "print(2)")
        self.assertIsNone(extract_code("```python\nprint(2)")[0])
        self.assertIsNone(extract_code("<think>```python\nprint(2)\n```")[0])

    def test_decode_hidden_json_and_upstream_pickle_without_executing_globals(self):
        cases = [{"input": "1", "output": "2", "testtype": "stdin"}]
        encoded = base64.b64encode(zlib.compress(pickle.dumps(json.dumps(cases)))).decode()
        self.assertEqual(decode_private(encoded, 10000), cases)
        self.assertEqual(decode_private(json.dumps(cases), 10000), cases)
        malicious = base64.b64encode(zlib.compress(b"cos\nsystem\n(S'never-run'\ntR.")).decode()
        with self.assertRaisesRegex(ValueError, "globals"):
            decode_private(malicious, 10000)
        oversized = base64.b64encode(zlib.compress(pickle.dumps("x" * 1000))).decode()
        with self.assertRaises(ValueError):
            decode_private(oversized, 100)

    def test_stdio_exact_decimal_without_float_tolerance(self):
        self.assertTrue(stdio_equal(" 1.00  2\n", "1 2.0"))
        self.assertFalse(stdio_equal("50000000000000000", "50000000000000001"))
        self.assertFalse(stdio_equal("1.000001", "1"))
        self.assertFalse(stdio_equal("1\n2", "1 2"))

    def test_docker_constraints_and_config_validation(self):
        cfg = read_json(ROOT / "configs" / "lcb_grading.json")
        cmd = docker_command("sha256:pinned", "repeat-sft-lcb-fixture", cfg)
        for flag in ("--network=none", "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges", "--pull=never"):
            self.assertIn(flag, cmd)
        # Root supervisor must be able to kill its demoted UID 65534 child.
        self.assertIn("--cap-add=KILL", cmd)
        self.assertNotIn("--privileged", cmd)
        self.assertNotIn("--volume", cmd)
        self.assertNotIn("--gpus", cmd)
        with self.assertRaises(ValueError):
            validate_config({**cfg, "pids_limit": 1.2})

    def test_preflight_checks_permission_drop_and_timeout_cleanup(self):
        judge = DockerJudge.__new__(DockerJudge)
        judge.config = read_json(ROOT / "configs" / "lcb_grading.json")
        judge.image = FakeJudge.image
        with patch.object(judge, "run", side_effect=[{"correct": True}, {"correct": True},
                                                     {"correct": False, "status": "timeout"}]) as run:
            self.assertEqual(judge.preflight()["timeout_and_unprivileged_execution_self_tests"], "passed")
            self.assertIn("geteuid()==65534", run.call_args_list[0].args[0])
            self.assertEqual(run.call_args_list[2].args[0], "while True: pass")
        with patch.object(judge, "run", side_effect=[{"correct": True}, {"correct": True},
                                                     {"correct": False, "status": "runtime_error"}]):
            with self.assertRaisesRegex(RuntimeError, "timeout/child cleanup"):
                judge.preflight()

    def test_real_candidate_process_stdio_and_function_without_gold(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as temp:
            path = Path(temp) / "solution.py"
            for code, fn in (("print(int(input())+1)", ""), ("class Solution:\n def solve(self,x): return x+1", "solve")):
                path.write_text(code, encoding="utf-8")
                result = subprocess.run([sys.executable, "-I", "-B", str(ROOT / "recipe" / "lcb_worker.py"),
                                         "--candidate", str(path), fn], input="2", text=True, capture_output=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.strip(), "3")
            path.write_text("print(random.randrange(3,4)); assert hasattr(datetime, 'datetime')", encoding="utf-8")
            result = subprocess.run([sys.executable, "-I", "-B", str(ROOT / "recipe" / "lcb_worker.py"),
                                     "--candidate", str(path), ""], input="", text=True, capture_output=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), "3")

    def test_real_function_inputs_with_blank_lines_preserve_json_values(self):
        cases = (
            ("2\n3", [2, 3]),
            ("2\n3\n", [2, 3]),  # Exact function self-test input from DockerJudge.preflight.
            ("2\r\n3\r\n", [2, 3]),
            ("\n \t\n2\n\n3\n\n", [2, 3]),
            ('""\n"   "\n0\nfalse\nnull\n[]\n{}\n{"n":[1,2]}\n',
             ["", "   ", 0, False, None, [], {}, {"n": [1, 2]}]),
            ("", []),
        )
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as temp:
            path = Path(temp) / "solution.py"
            path.write_text("class Solution:\n def solve(self, *args): return list(args)", encoding="utf-8")
            for raw, expected in cases:
                with self.subTest(input=raw):
                    result = subprocess.run([sys.executable, "-I", "-B", str(ROOT / "recipe" / "lcb_worker.py"),
                                             "--candidate", str(path), "solve"],
                                            input=raw, text=True, capture_output=True, timeout=10)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(json.loads(result.stdout), expected)

    def test_real_function_inputs_reject_malformed_nonempty_lines(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as temp:
            path = Path(temp) / "solution.py"
            path.write_text("class Solution:\n def solve(self, *args): return list(args)", encoding="utf-8")
            result = subprocess.run([sys.executable, "-I", "-B", str(ROOT / "recipe" / "lcb_worker.py"),
                                     "--candidate", str(path), "solve"],
                                    input="2\nnot-json\n3\n", text=True, capture_output=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("JSONDecodeError", result.stderr)
            self.assertEqual(result.stdout, "")

    def test_supervisor_requires_all_tests_and_records_first_failure(self):
        cfg = read_json(ROOT / "configs" / "lcb_grading.json")
        cases = [{"split": "public" if i == 0 else "private", "index": i, "input": str(i), "output": str(i+1)} for i in range(3)]
        calls = []
        def execute(path, case, fn, config):
            calls.append(case["index"])
            return {"status": "passed" if case["index"] == 0 else "wrong_answer", "actual_output": "1", "stderr": ""}
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as tmp:
            result = supervise({"code": "print(1)", "cases": cases, "config": cfg, "fn_name": None}, execute, tmp)
            self.assertFalse(result["correct"])
            self.assertEqual((result["total_tests"], result["tested"], result["passed"]), (3, 2, 1))
            self.assertEqual(calls, [0, 1])
            self.assertEqual(result["test_results"][-1]["split"], "private")
            for file in Path(tmp).rglob("*"):
                file.chmod(0o700)
        result = supervise({"code": "syntax error !", "cases": cases, "config": cfg, "fn_name": None}, execute)
        self.assertEqual(result["status"], "compile_error")
        self.assertEqual(result["tested"], 0)


class TransferPipelineTests(unittest.TestCase):
    def setUp(self):
        from test_replay import fixture
        self.temp = tempfile.TemporaryDirectory(dir=TEST_TMP)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.run = fixture(self.root)
        self.panel_dir = self.root / "panel"
        self.panel = panel_fixture(self.panel_dir, file_hash(self.run / "data" / "train.jsonl"))
        self.job = self.run / "postprocess" / "transfer"
        self.job.mkdir(parents=True)
        self.args = SimpleNamespace(epochs=None, image=None, judge_config=None)
        self.judge = FakeJudge()

    def prepare(self):
        with patch.object(pipeline, "DockerJudge", return_value=self.judge):
            return pipeline.prepare(self.run, self.panel_dir, self.job, self.args)

    def test_four_balanced_groups_and_manifest_checks(self):
        groups = partition(self.panel)
        self.assertEqual(len(groups), 4)
        self.assertEqual(len({sid for g in groups for sid in g["selected_ids"]}), 200)
        for g in groups:
            self.assertEqual(g["task_counts"], {"bbh": 25, "livecodebench": 25})
            self.assertEqual(sorted(g["bbh_subtasks"].values()), [6, 6, 6, 7])
        with self.assertRaisesRegex(ValueError, "different training"):
            load_panel(self.panel_dir, "wrong")
        row = self.panel[0]
        (self.panel_dir / row["reference_file"]).write_text("{}")
        with self.assertRaisesRegex(ValueError, "reference changed"):
            load_panel(self.panel_dir)

    def test_old_epochs_preserved_and_job_conflicts_rejected(self):
        original = file_hash(self.run / "run.json")
        plan = self.prepare()
        self.assertEqual(plan["generation_epochs"], [64, 32, 16, 8, 4])
        self.assertEqual(file_hash(self.run / "run.json"), original)
        self.assertEqual(self.prepare(), plan)
        self.args.epochs = "6"
        with self.assertRaisesRegex(ValueError, "inputs differ"):
            self.prepare()

    def test_group_barrier_resume_raw_answers_grades_and_full_report(self):
        from recipe.replay import tree_hashes
        before_data = tree_hashes(self.run / "data")
        before_checkpoints = tree_hashes(self.run / "checkpoints")
        plan = self.prepare()
        calls, interrupt = [], [True]
        def invoke(command, cwd):
            script = Path(command[3]).name
            out = Path(command[command.index("--out") + 1])
            group = next(g for g in plan["groups"] if g["name"] == out.parent.name)
            calls.append((script, group["name"]))
            if script == "generate_vllm.py":
                fail = group["name"] == "group-002" and interrupt[0]
                if fail:
                    interrupt[0] = False
                synthetic_responses(self.job, group, plan["generation_epochs"], fail)
            else:
                grade_transfer.grade_directory(out.parent / "generation", self.job / "setup" / "panel", out, self.judge)
        fake = SimpleNamespace(check_stop=lambda: None, invoke=invoke)
        with self.assertRaisesRegex(RuntimeError, "interruption"):
            pipeline.execute(self.job, fake)
        self.assertTrue((self.job / "groups" / "group-001" / "complete.json").exists())
        self.assertFalse((self.job / "groups" / "group-003").exists())
        first = tree_hashes(self.job / "groups" / "group-001")
        prefix = (self.job / "groups" / "group-002" / "generation" / "epoch-064" / "responses.jsonl").read_bytes()
        pipeline.execute(self.job, fake)
        self.assertEqual(tree_hashes(self.job / "groups" / "group-001"), first)
        self.assertTrue((self.job / "groups" / "group-002" / "generation" / "epoch-064" / "responses.jsonl").read_bytes().startswith(prefix))
        self.assertEqual(self.judge.calls, 500)
        self.assertEqual(read_json(self.job / "complete.json")["responses"], 1000)
        self.assertEqual(len(list((self.job / "report" / "questions").glob("*.json"))), 200)
        count = len(calls)
        pipeline.execute(self.job, fake)
        self.assertEqual(len(calls), count)
        self.assertEqual(tree_hashes(self.run / "data"), before_data)
        self.assertEqual(tree_hashes(self.run / "checkpoints"), before_checkpoints)

    def test_grading_interrupt_resume_preserves_completed_rows_and_hidden_tests(self):
        plan = self.prepare()
        group = plan["groups"][0]
        synthetic_responses(self.job, group, [64])
        folder = self.job / "groups" / group["name"]
        actual = self.judge.run
        def fail(code, tests):
            if self.judge.calls == 2:
                raise RuntimeError("Docker unavailable")
            return actual(code, tests)
        with patch.object(self.judge, "run", side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, "Docker unavailable"):
                grade_transfer.grade_directory(folder / "generation", self.job / "setup" / "panel", folder / "grades", self.judge)
        log = folder / "grades" / "epoch-064" / "graded.jsonl"
        prefix = log.read_bytes()
        grade_transfer.grade_directory(folder / "generation", self.job / "setup" / "panel", folder / "grades", self.judge)
        self.assertTrue(log.read_bytes().startswith(prefix))
        self.assertEqual(self.judge.calls, 25)
        self.assertTrue(all(r["hidden_tests"] == 1 for r in iter_jsonl(log) if r["task"] == "livecodebench"))

    def test_pipeline_lock_rejects_second_writer(self):
        with Pipeline(self.job):
            with self.assertRaisesRegex(RuntimeError, "already running"):
                with Pipeline(self.job):
                    self.fail("Second writer")

    @unittest.skipUnless(os.name == "posix", "Linux signal forwarding and inherited lock")
    def test_pipeline_sigterm_waits_for_child_and_releases_lock(self):
        ready, saved = self.root / "ready", self.root / "saved"
        child = ("from pathlib import Path; import signal,time\nstop=False\n"
                 "def done(s,f):\n global stop\n stop=True\n"
                 "signal.signal(signal.SIGTERM,done)\n"
                 f"Path({str(ready)!r}).touch()\n"
                 "deadline=time.monotonic()+15\nwhile not stop and time.monotonic()<deadline: time.sleep(.02)\n"
                 f"Path({str(saved)!r}).write_text(str(stop))\n")
        parent_code = ("from pathlib import Path; import sys; from recipe.pipeline import Pipeline\n"
                       f"with Pipeline(Path({str(self.job)!r})) as p:\n"
                       f" p.invoke([sys.executable,'-B','-c',{child!r}],Path({str(ROOT)!r}))\n")
        process = subprocess.Popen([sys.executable, "-B", "-c", parent_code], cwd=ROOT, start_new_session=True)
        try:
            deadline = time.monotonic() + 15
            while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
                time.sleep(.05)
            self.assertTrue(ready.exists())
            process.terminate()
            self.assertEqual(process.wait(timeout=10), 143)
            self.assertEqual(saved.read_text(), "True")
            with Pipeline(self.job):
                pass
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)


if __name__ == "__main__":
    unittest.main()
