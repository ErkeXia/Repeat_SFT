"""Offline missing-epoch planning, full-state copies, interruptions and evaluation."""
from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import replay_train
from recipe.common import ROOT, digest, file_hash, iter_jsonl, read_json, write_json, write_jsonl
from recipe.replay import (DEFAULT_EPOCHS, IDENTITY_KEYS, continuation, evaluation_epochs,
                           initialize, parse_epochs, plan_replay, publish_target, tree_hashes, verify_replay)

TEST_TMP = ROOT / ".test_tmp"
TEST_TMP.mkdir(exist_ok=True)

# Executed by real subprocesses to test the launcher/snapshot/CLI boundary;
# synthetic full states and adapter bytes, never a pretrained model or GPU.
FAKE_TRAIN = '''import argparse, json, shutil
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('--run'); p.add_argument('--resume'); p.add_argument('--stop-after', type=int)
a = p.parse_args(); run = Path(a.run); src = Path(a.resume)
old = json.loads((src / 'complete.json').read_text())
assert old['step'] < a.stop_after
assert src.is_relative_to(run)
dst = run / 'recovery' / f'step-{a.stop_after:08d}'
shutil.copytree(src, dst)
old['step'] = a.stop_after
(dst / 'complete.json').write_text(json.dumps(old))
(dst / 'training_state.pt').write_bytes(b'updated synthetic full state')
with (run / 'synthetic_updates.jsonl').open('a') as f:
    f.write(json.dumps({'start': json.loads((src / 'complete.json').read_text())['step'], 'end': a.stop_after}) + '\\n')
'''


def make_checkpoint(path, metadata, epoch):
    path.mkdir(parents=True)
    ft = metadata["config"]["finetuning"]
    write_json(path / "recipe_model.json", {"format_version": 1, "compute_dtype": "bfloat16",
               "finetuning": ft, "base_model": metadata["model"]})
    write_json(path / "adapter_config.json", {"peft_type": "LORA", "task_type": "CAUSAL_LM",
               "r": ft["rank"], "lora_alpha": ft["alpha"], "lora_dropout": ft["dropout"],
               "target_modules": ft["target_modules"], "bias": "none", "modules_to_save": None,
               "base_model_name_or_path": metadata["model"]["id"], "revision": metadata["model"]["revision"]})
    (path / "adapter_model.safetensors").write_bytes(f"synthetic adapter {epoch}".encode())
    write_json(path / "tokenizer.json", {"synthetic": True})
    files = tree_hashes(path)
    (path / "training_state.pt").write_bytes(f"optimizer/scheduler/RNG/cursor {epoch}".encode())
    write_json(path / "complete.json", {"step": epoch * metadata["config"]["expected_train_size"],
               "config_sha256": metadata["config_sha256"], "has_training_state": True,
               "weights_format": "lora_adapter", "model_files": files})


def fixture(root):
    source = root / "original"
    config = read_json(ROOT / "configs" / "baseline.json")
    write_jsonl(source / "data" / "train.jsonl", [{"sample_id": str(i)} for i in range(400)])
    write_jsonl(source / "data" / "holdout.jsonl", [{"sample_id": "heldout"}])
    write_jsonl(source / "data" / "panel.jsonl", [{"sample_id": "evaluation"}])
    write_json(source / "tokenizer" / "tokenizer.json", {"synthetic": True})
    code = source / "attempts" / "original" / "code"
    code.mkdir(parents=True)
    (code / "train.py").write_text(FAKE_TRAIN, encoding="utf-8")
    files = tree_hashes(code)
    write_json(code / "manifest.json", {"files": files, "code_sha256": digest(files)})
    metadata = {"config": config, "model": {"id": "synthetic-base", "revision": "pinned"},
                "semantics": {"adapter_dtype": "float32", "batch_size": 1},
                "data_files": tree_hashes(source / "data"), "tokenizer_files": tree_hashes(source / "tokenizer"),
                "code_sha256": digest(files), "max_length": 10240}
    metadata["config_sha256"] = digest({key: metadata[key] for key in IDENTITY_KEYS})
    write_json(source / "run.json", metadata)
    for epoch in config["checkpoint_epochs"]:
        make_checkpoint(source / "checkpoints" / f"epoch-{epoch:03d}", metadata, epoch)
    write_json(source / "evaluations" / "old_results.json", {"must": "remain untouched"})
    return source


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=TEST_TMP)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.source = fixture(self.root)
        self.out = self.root / "replay"
        self.before = tree_hashes(self.source)

    def prepare(self, epochs=None):
        plan = plan_replay(self.source, self.out, epochs or DEFAULT_EPOCHS)
        initialize(plan)
        return verify_replay(self.out)

    def simulate(self, command, cwd, pause_at=None):
        src = Path(command[command.index("--resume") + 1])
        end = int(command[command.index("--stop-after") + 1])
        self.assertEqual(Path(cwd), self.out / "attempts" / "replay-bootstrap" / "code")
        self.assertEqual(Path(command[3]).parent, Path(cwd))
        marker = read_json(src / "complete.json")
        self.assertLess(marker["step"], end)
        end = pause_at or end
        target = self.out / "recovery" / f"step-{end:08d}"
        shutil.copytree(src, target)
        marker["step"] = end
        write_json(target / "complete.json", marker)

    def test_exact_targets_nearest_anchors_and_disk_budget(self):
        plan = plan_replay(self.source, self.out, DEFAULT_EPOCHS)
        self.assertEqual(plan["segments"], [{"anchor_epoch": 4, "target_epochs": [6, 7]},
                         {"anchor_epoch": 8, "target_epochs": [9, 10, 12, 14]},
                         {"anchor_epoch": 16, "target_epochs": [20, 22, 26, 30]}])
        self.assertEqual(plan["training_updates"], 9200)
        self.assertEqual(plan["storage"]["peak_checkpoint_copies_budget"], 16)
        self.assertFalse(self.out.exists())
        self.assertEqual(tree_hashes(self.source), self.before)
        for bad in ("", "6,6", "0,6", "-1", "x"):
            with self.assertRaises(ValueError):
                parse_epochs(bad)
        for destination in (self.source, self.source / "new", self.root):
            with self.assertRaisesRegex(ValueError, "separate"):
                plan_replay(self.source, destination, [6])
        with self.assertRaisesRegex(ValueError, "already saved"):
            plan_replay(self.source, self.out, [8])

    def test_no_state_falls_back_to_earlier_anchor_and_bad_source_fails(self):
        (self.source / "checkpoints" / "epoch-004" / "training_state.pt").unlink()
        plan = plan_replay(self.source, self.out, [6])
        self.assertEqual(plan["segments"][0]["anchor_epoch"], 2)
        (self.source / "data" / "train.jsonl").write_text("modified")
        with self.assertRaisesRegex(ValueError, "Frozen run input"):
            plan_replay(self.source, self.out, [6])

    def test_independent_copy_keeps_full_state_bytes_identity_and_original_code(self):
        plan, metadata = self.prepare()
        original = read_json(self.source / "run.json")
        self.assertEqual(metadata["config"], original["config"])
        self.assertEqual(metadata["config_sha256"], original["config_sha256"])
        self.assertEqual(metadata["code_sha256"], original["code_sha256"])
        self.assertEqual(evaluation_epochs(metadata), DEFAULT_EPOCHS)
        self.assertEqual(evaluation_epochs(original), original["config"]["checkpoint_epochs"])
        for epoch in (4, 8, 16):
            a = self.out / "anchors" / f"epoch-{epoch:03d}"
            b = self.source / "checkpoints" / a.name
            self.assertEqual(tree_hashes(a), tree_hashes(b))
            self.assertNotEqual(a.resolve(), b.resolve())
        self.assertFalse((self.out / "evaluations").exists())
        with self.assertRaises(FileExistsError):
            initialize(plan)
        self.assertEqual(tree_hashes(self.source), self.before)

    def test_disk_full_rejected_before_any_output_or_source_write(self):
        plan = plan_replay(self.source, self.out, [6])
        with patch("recipe.replay.require_free_space", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                initialize(plan)
        self.assertFalse(self.out.exists())
        self.assertEqual(tree_hashes(self.source), self.before)

    def test_all_nodes_run_from_expected_branches_and_repeat_is_noop(self):
        self.prepare()
        starts = []
        def invoke(command, cwd):
            starts.append(read_json(Path(command[command.index("--resume") + 1]) / "complete.json")["step"] // 400)
            self.simulate(command, cwd)
        with patch.object(replay_train, "invoke", side_effect=invoke):
            self.assertTrue(replay_train.execute(self.out))
        self.assertEqual(starts, [4, 6, 8, 9, 10, 12, 16, 20, 22, 26])
        self.assertEqual(sorted(int(p.name.split("-")[1]) for p in (self.out / "checkpoints").iterdir()), DEFAULT_EPOCHS)
        saved = tree_hashes(self.out / "checkpoints")
        with patch.object(replay_train, "invoke") as invoke:
            self.assertTrue(replay_train.execute(self.out))
            invoke.assert_not_called()
        self.assertEqual(tree_hashes(self.out / "checkpoints"), saved)
        self.assertEqual(tree_hashes(self.source), self.before)
        self.assertFalse(read_json(self.out / "replay_complete.json")["trajectory_comparison_performed"])

    def test_interrupted_update_resumes_from_latest_state_and_preserves_completed_epoch(self):
        self.prepare([6, 7])
        calls = []
        def interrupted(command, cwd):
            calls.append(command)
            self.simulate(command, cwd, pause_at=2533 if len(calls) == 2 else None)
            if len(calls) == 2:
                raise RuntimeError("simulated power loss")
        with patch.object(replay_train, "invoke", side_effect=interrupted):
            with self.assertRaisesRegex(RuntimeError, "power loss"):
                replay_train.execute(self.out)
        epoch6 = tree_hashes(self.out / "checkpoints" / "epoch-006")
        partial = self.out / "recovery" / "step-99999999.partial-aborted"
        partial.mkdir()
        write_json(partial / "complete.json", {"step": 99999999})
        starts = []
        def resumed(command, cwd):
            starts.append(read_json(Path(command[command.index("--resume") + 1]) / "complete.json")["step"])
            self.simulate(command, cwd)
        with patch.object(replay_train, "invoke", side_effect=resumed):
            self.assertTrue(replay_train.execute(self.out))
        self.assertEqual(starts, [2533])
        self.assertEqual(tree_hashes(self.out / "checkpoints" / "epoch-006"), epoch6)
        self.assertEqual(tree_hashes(self.source), self.before)

    def test_graceful_child_pause_is_not_automatically_restarted(self):
        self.prepare([6])
        with patch.object(replay_train, "invoke", side_effect=lambda c, w: self.simulate(c, w, 1777)) as invoke:
            self.assertFalse(replay_train.execute(self.out))
        self.assertEqual(invoke.call_count, 1)
        self.assertEqual(read_json(self.out / "replay_progress.json")["status"], "paused")
        self.assertFalse((self.out / "replay_complete.json").exists())

    def test_crash_after_checkpoint_write_before_promotion_needs_no_training(self):
        plan, metadata = self.prepare([6])
        src = self.out / "anchors" / "epoch-004"
        dst = self.out / "recovery" / "step-00002400"
        shutil.copytree(src, dst)
        marker = read_json(dst / "complete.json")
        marker["step"] = 2400
        write_json(dst / "complete.json", marker)
        with patch.object(replay_train, "invoke") as invoke:
            self.assertTrue(replay_train.execute(self.out))
            invoke.assert_not_called()
        self.assertFalse(dst.exists())
        self.assertTrue((self.out / "checkpoints" / "epoch-006" / "training_state.pt").exists())
        with self.assertRaisesRegex(ValueError, "incomplete epoch"):
            publish_target(self.out, metadata, 6, src, 1600)

    def test_corrupted_anchor_and_plan_are_rejected(self):
        self.prepare([6])
        anchor = self.out / "anchors" / "epoch-004" / "training_state.pt"
        old = anchor.read_bytes()
        anchor.write_bytes(b"broken")
        with self.assertRaisesRegex(ValueError, "anchor changed"):
            verify_replay(self.out)
        anchor.write_bytes(old)
        plan = read_json(self.out / "replay_plan.json")
        plan["target_epochs"] = [7]
        write_json(self.out / "replay_plan.json", plan)
        with self.assertRaisesRegex(ValueError, "plan changed"):
            verify_replay(self.out)

    def test_cli_plan_and_frozen_worker_run_without_torch_or_gpu(self):
        env = {**os.environ, "REPEAT_SFT_ROOT": str(self.root), "HF_HOME": str(self.root / "cache"),
               "TMPDIR": str(self.root / "tmp"), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
        command = [sys.executable, "-B", str(ROOT / "replay_train.py"), "--source-run", str(self.source),
                   "--out", str(self.out), "--epochs", "6,7"]
        plan = subprocess.run(command + ["--plan-only"], capture_output=True, text=True, env=env, timeout=30)
        self.assertEqual(plan.returncode, 0, plan.stderr)
        self.assertIn("1,200 updates", plan.stdout)
        self.assertFalse(self.out.exists())
        result = subprocess.run(command, capture_output=True, text=True, env=env, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(list(iter_jsonl(self.out / "synthetic_updates.jsonl")),
                         [{"start": 1600, "end": 2400}, {"start": 2400, "end": 2800}])
        resume = subprocess.run([sys.executable, "-B", str(ROOT / "replay_train.py"),
                                 "--out", str(self.out), "--resume"], capture_output=True, text=True, env=env, timeout=60)
        self.assertEqual(resume.returncode, 0, resume.stdout + resume.stderr)
        self.assertEqual(len(list(iter_jsonl(self.out / "synthetic_updates.jsonl"))), 2)
        self.assertEqual(tree_hashes(self.source), self.before)

    def test_lock_prevents_duplicate_writers(self):
        with replay_train.replay_lock(self.out):
            with self.assertRaisesRegex(RuntimeError, "already running"):
                with replay_train.replay_lock(self.out):
                    self.fail("Concurrent writer acquired lock")

    @unittest.skipUnless(os.name == "posix", "POSIX graceful signal forwarding requires Linux")
    def test_sigterm_saves_child_state_and_frozen_worker_resumes(self):
        code = self.source / "attempts" / "original" / "code"
        wait_for_signal = '''import signal, time
if not (run / 'allow_finish').exists():
    stopped = False
    def pause(signum, frame):
        global stopped
        stopped = True
    signal.signal(signal.SIGTERM, pause)
    (run / 'worker_ready').touch()
    deadline = time.monotonic() + 30
    while not stopped and time.monotonic() < deadline: time.sleep(.02)
    assert stopped, 'test did not deliver SIGTERM'
    a.stop_after = old['step'] + 17
'''
        (code / "train.py").write_text(FAKE_TRAIN.replace("dst = run /", wait_for_signal + "dst = run /"), encoding="utf-8")
        files = {"train.py": file_hash(code / "train.py")}
        write_json(code / "manifest.json", {"files": files, "code_sha256": digest(files)})
        metadata = read_json(self.source / "run.json")
        metadata["code_sha256"] = digest(files)
        write_json(self.source / "run.json", metadata)
        before = tree_hashes(self.source)
        env = {**os.environ, "REPEAT_SFT_ROOT": str(self.root), "HF_HOME": str(self.root / "cache"),
               "TMPDIR": str(self.root / "tmp")}
        command = [sys.executable, "-B", str(ROOT / "replay_train.py"), "--source-run", str(self.source),
                   "--out", str(self.out), "--epochs", "6"]
        parent = subprocess.Popen(command, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                  start_new_session=True)
        try:
            deadline = time.monotonic() + 40
            while not (self.out / "worker_ready").exists() and parent.poll() is None and time.monotonic() < deadline:
                time.sleep(.05)
            self.assertTrue((self.out / "worker_ready").exists(), "Training child never became ready")
            parent.terminate()
            _, errors = parent.communicate(timeout=20)
            self.assertEqual(parent.returncode, 143, errors)
        finally:
            if parent.poll() is None:
                # A failed signal test must not leak its synthetic process tree.
                os.killpg(parent.pid, signal.SIGKILL)
                parent.communicate(timeout=10)
        self.assertEqual(read_json(self.out / "replay_progress.json")["status"], "paused")
        self.assertTrue((self.out / "recovery" / "step-00001617" / "training_state.pt").exists())
        (self.out / "allow_finish").touch()
        result = subprocess.run([sys.executable, "-B", str(ROOT / "replay_train.py"), "--out", str(self.out),
                                 "--resume"], env=env, capture_output=True, text=True, timeout=40)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(list(iter_jsonl(self.out / "synthetic_updates.jsonl")),
                         [{"start": 1600, "end": 1617}, {"start": 1617, "end": 2400}])
        self.assertEqual(tree_hashes(self.source), before)

    @unittest.skipUnless(os.name == "posix", "POSIX inherited flock requires Linux")
    def test_force_killed_launcher_keeps_child_lock(self):
        ready, release = self.root / "ready", self.root / "release"
        child = ("from pathlib import Path; import time; "
                 f"Path({str(ready)!r}).touch(); deadline=time.monotonic()+15\n"
                 f"while not Path({str(release)!r}).exists() and time.monotonic()<deadline: time.sleep(.05)\n")
        parent_script = ("from pathlib import Path; import sys; import replay_train as r\n"
                         f"with r.replay_lock(Path({str(self.out)!r})):\n"
                         f"    r.invoke([sys.executable, '-B', '-c', {child!r}], Path({str(ROOT)!r}))\n")
        parent = subprocess.Popen([sys.executable, "-B", "-c", parent_script], cwd=ROOT)
        try:
            deadline = time.monotonic() + 10
            while not ready.exists() and parent.poll() is None and time.monotonic() < deadline:
                time.sleep(.05)
            self.assertTrue(ready.exists())
            parent.kill()
            parent.wait(timeout=5)
            with self.assertRaisesRegex(RuntimeError, "already running"):
                with replay_train.replay_lock(self.out):
                    self.fail("Child lost lock")
        finally:
            release.touch()
            if parent.poll() is None:
                parent.kill()
                parent.wait(timeout=5)
        deadline = time.monotonic() + 5
        while True:
            try:
                with replay_train.replay_lock(self.out):
                    break
            except RuntimeError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(.05)


if __name__ == "__main__":
    unittest.main()
