"""No-download tests for volume paths, atomic saves, and legacy smoke recovery."""
from __future__ import annotations

import errno
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from recipe.common import ROOT, append_jsonl, digest, file_hash, iter_jsonl, read_json, snapshot_code, write_json, write_jsonl
from recipe.checkpoints import latest_recovery, prune_recovery, save_checkpoint
from recipe.paths import artifact_path, configure_runtime, storage_root
from recipe.storage import require_free_space, tensor_storage_bytes
import smoke_test

TEST_TMP = ROOT / ".test_tmp"
TEST_TMP.mkdir(exist_ok=True)


class PathTests(unittest.TestCase):
    def test_relative_results_are_on_volume_and_resolution_does_not_write(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as tmp, patch.dict(os.environ, {"REPEAT_SFT_ROOT": tmp}):
            self.assertEqual(artifact_path("runs/random-s42"), Path(tmp).resolve() / "runs" / "random-s42")
            self.assertEqual(artifact_path("artifacts/data-s42"), Path(tmp).resolve() / "artifacts" / "data-s42")
            self.assertFalse((Path(tmp) / "runs").exists())

    def test_absolute_override_and_relative_escape(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as tmp, patch.dict(os.environ, {"REPEAT_SFT_ROOT": tmp}):
            absolute = TEST_TMP.resolve() / "legacy"
            self.assertEqual(artifact_path(absolute), absolute)
            with self.assertRaisesRegex(ValueError, "escapes"):
                artifact_path("../outside")

    def test_relative_storage_root_rejected(self):
        with patch.dict(os.environ, {"REPEAT_SFT_ROOT": "relative-folder"}):
            with self.assertRaisesRegex(ValueError, "absolute"):
                storage_root()

    @unittest.skipUnless(os.name == "posix", "Linux mount guard; Windows does not have /data mounts")
    def test_missing_data_mount_is_rejected_without_writes(self):
        with patch.dict(os.environ, {"REPEAT_SFT_ROOT": "/data/repeat_sft"}), patch.object(Path, "is_mount", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "not mounted"):
                storage_root()

    def test_caches_default_to_volume_and_explicit_environment_is_preserved(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as tmp:
            root = Path(tmp).resolve()
            with patch.dict(os.environ, {"REPEAT_SFT_ROOT": str(root)}, clear=True):
                configure_runtime()
                self.assertEqual(Path(os.environ["HF_HOME"]), root / ".cache" / "huggingface")
                self.assertTrue(Path(os.environ["TMPDIR"]).is_dir())
            with patch.dict(os.environ, {"REPEAT_SFT_ROOT": str(root), "HF_HOME": str(root / "existing-cache"),
                                         "TMPDIR": str(root / "existing-tmp")}, clear=True):
                configure_runtime()
                self.assertEqual(Path(os.environ["HF_HOME"]), root / "existing-cache")


def completed(path, step, identity="experiment"):
    path.mkdir(parents=True, exist_ok=True)
    (path / "training_state.pt").write_bytes(b"fixture state")
    write_json(path / "complete.json", {"step": step, "config_sha256": identity})


class CheckpointStorageTests(unittest.TestCase):
    def test_no_space_fails_before_creating_partial(self):
        import torch
        model = SimpleNamespace(state_dict=lambda: {"weight": torch.zeros(4)})
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as tmp:
            with patch("recipe.storage.shutil.disk_usage", return_value=SimpleNamespace(free=0)):
                with self.assertRaisesRegex(OSError, "Insufficient disk space"):
                    save_checkpoint(Path(tmp) / "step-2", model, None, 2, "test")
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_space_check_uses_nearest_existing_parent(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as tmp:
            root = Path(tmp).resolve()
            with patch("recipe.storage.shutil.disk_usage", return_value=SimpleNamespace(free=1024)) as usage:
                require_free_space(root / "new" / "run", 512)
                usage.assert_called_once_with(root)
                self.assertFalse((root / "new").exists())

    def test_shared_storages_counted_once(self):
        import torch
        tensor = torch.zeros(16, dtype=torch.float32)
        self.assertEqual(tensor_storage_bytes({"one": tensor, "alias": [tensor, tensor[:2]]}), 64)

    def test_failed_weight_write_preserves_old_checkpoints_and_rng(self):
        import torch
        tensor = torch.zeros(4)
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as tmp:
            root = Path(tmp)
            old = root / "step-1"
            completed(old, 1)
            stale = root / "step-2.partial-other-attempt"
            stale.mkdir()
            (stale / "evidence").write_bytes(b"keep")

            def fail(path, **kwargs):
                (path / "broken.safetensors").write_bytes(b"incomplete")
                random.random()
                torch.rand(1)
                raise RuntimeError("I/O error: No space left on device (os error 28)")

            model = SimpleNamespace(state_dict=lambda: {"weight": tensor}, save_pretrained=fail)
            before_python, before_torch = random.getstate(), torch.get_rng_state().clone()
            with patch("recipe.checkpoints.require_free_space"):
                with self.assertRaises(OSError) as caught:
                    save_checkpoint(root / "step-2", model, None, 2, "test")
            self.assertEqual(caught.exception.errno, errno.ENOSPC)
            self.assertEqual(set(root.iterdir()), {old, stale})
            self.assertEqual((old / "training_state.pt").read_bytes(), b"fixture state")
            self.assertEqual(before_python, random.getstate())
            self.assertTrue(torch.equal(before_torch, torch.get_rng_state()))

    def test_optimizer_save_failure_does_not_publish(self):
        import torch
        model = SimpleNamespace(state_dict=lambda: {"weight": torch.zeros(4)},
                                save_pretrained=lambda p, **kw: (p / "weights").write_bytes(b"weights"))
        tokenizer = SimpleNamespace(save_pretrained=lambda p: (p / "tokenizer.json").write_text("{}"))
        optimizer = SimpleNamespace(state_dict=lambda: {"moment": torch.zeros(4)})
        scheduler = SimpleNamespace(state_dict=lambda: {})
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as tmp:
            with patch("recipe.checkpoints.require_free_space"), patch("torch.save", side_effect=OSError(errno.ENOSPC, "disk full")):
                with self.assertRaises(OSError):
                    save_checkpoint(Path(tmp) / "step-2", model, tokenizer, 2, "test", optimizer, scheduler, {})
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_unpublished_partial_never_selected_or_pruned(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as tmp:
            root = Path(tmp)
            old, latest = root / "recovery" / "step-1", root / "recovery" / "step-2"
            partial = root / "recovery" / "step-999.partial-unpublished"
            milestone = root / "checkpoints" / "epoch-000"
            for path, step in ((old, 1), (latest, 2), (partial, 999), (milestone, 0)):
                completed(path, step)
            self.assertEqual(latest_recovery(root), latest)
            prune_recovery(root, 1)
            self.assertFalse(old.exists())
            self.assertTrue(latest.exists() and partial.exists() and milestone.exists())


def record(step):
    return {"step": step, "sample_id": str(step), "learning_rate": 1e-5,
            "loss": 1 / step, "grad_norm_before_clip": 1.0}


def legacy_fixture(root, continuous_step=32, resumed=True):
    data, out = root / "data", root / "smoke"
    data.mkdir()
    write_json(data / "manifest.json", {"prepared": "same"})
    config = {"epochs": 4, "expected_train_size": 8, "checkpoint_epochs": [0, 4], "recovery_every_steps": 8}
    write_json(out / "smoke_train.json", config)
    write_json(out / "smoke_generation.json", {"max_new_tokens": 512})
    write_jsonl(out / "subset.jsonl", [{"sample_id": "fixture"}])
    for name in (["continuous", "resumed"] if resumed else ["continuous"]):
        run = out / name
        code = run / "attempts" / "initial" / "code"
        code.mkdir(parents=True)
        for entry in ("train.py", "evaluate_loss.py", "generate.py", "grade.py"):
            (code / entry).write_text("# Original frozen source\n", encoding="utf-8")
        files = {p.name: file_hash(p) for p in code.iterdir()}
        code_hash = digest(files)
        write_json(code / "manifest.json", {"files": files, "code_sha256": code_hash})
        (run / "data").mkdir()
        (run / "data" / "train.jsonl").write_bytes((out / "subset.jsonl").read_bytes())
        write_json(run / "tokenizer" / "tokenizer.json", {})
        write_json(run / "run.json", {"config": config, "config_sha256": "experiment", "code_sha256": code_hash,
                   "prepared_manifest_sha256": file_hash(data / "manifest.json"),
                   "data_files": {"train.jsonl": file_hash(run / "data" / "train.jsonl")},
                   "tokenizer_files": {"tokenizer.json": file_hash(run / "tokenizer" / "tokenizer.json")}})
        write_jsonl(run / "attempts.jsonl", [{"attempt": "initial", "start_step": 0}])
        end = continuous_step if name == "continuous" else 13
        write_jsonl(run / "logs" / "initial.jsonl", [record(i) for i in range(1, end + 1)])
        if name == "continuous" and continuous_step == 32:
            completed(run / "checkpoints" / "epoch-004", 32)
        else:
            completed(run / "recovery" / "step-00000008", 8)
    commands = []
    if resumed:
        commands.append(["python", "train.py", "--run", "/old/project/runs/smoke/resumed", "--simulate-crash-at", "13"])
        append_jsonl(out / "resumed" / "attempts.jsonl", {"attempt": "interrupted-continuation", "start_step": 8})
        write_jsonl(out / "resumed" / "logs" / "interrupted-continuation.jsonl", [record(i) for i in range(9, 17)])
        completed(out / "resumed" / "recovery" / "step-00000016.partial-failed", 16)
    write_json(out / "commands.json", commands)
    return out, data


class SmokeResumeTests(unittest.TestCase):
    def test_new_snapshot_preserves_training_fingerprint_and_frozen_config(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as tmp:
            out, data = legacy_fixture(Path(tmp))
            code_hash = snapshot_code(out / "code")
            write_json(out / "smoke_manifest.json", {
                "code_sha256": code_hash, "prepared_manifest_sha256": file_hash(data / "manifest.json"),
                "files": {name: file_hash(out / name) for name in
                          ("subset.jsonl", "smoke_train.json", "smoke_generation.json")}})
            for name in ("continuous", "resumed"):
                path = out / name / "run.json"
                metadata = read_json(path)
                metadata["code_sha256"] = code_hash
                write_json(path, metadata)
            source = smoke_test.frozen_smoke(out, data)
            self.assertEqual(source, (out / "code").resolve())
            # Training recomputes its fingerprint from inside the selected snapshot.
            result = subprocess.run([sys.executable, "-B", "-c",
                                     "from recipe.common import code_fingerprint; print(code_fingerprint())"],
                                    cwd=source, capture_output=True, text=True, check=True)
            self.assertEqual(result.stdout.strip(), code_hash)
            write_json(out / "smoke_generation.json", {"max_new_tokens": 123})
            with self.assertRaisesRegex(ValueError, "Frozen smoke input changed"):
                smoke_test.frozen_smoke(out, data)

    def test_legacy_original_code_and_crash_evidence(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as tmp:
            out, data = legacy_fixture(Path(tmp))
            self.assertEqual(smoke_test.frozen_smoke(out, data), (out / "continuous" / "attempts" / "initial" / "code").resolve())
            self.assertEqual(smoke_test.crash_evidence(out)["restored_step"], 8)
            self.assertEqual(smoke_test.continuation(out / "resumed")[1], 8)

    def test_snapshot_tampering_rejected(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as tmp:
            out, data = legacy_fixture(Path(tmp))
            (out / "continuous" / "attempts" / "initial" / "code" / "train.py").write_text("# changed")
            with self.assertRaisesRegex(ValueError, "No intact original"):
                smoke_test.frozen_smoke(out, data)

    def test_modified_frozen_data_rejected(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as tmp:
            out, data = legacy_fixture(Path(tmp))
            (out / "resumed" / "data" / "train.jsonl").write_text("{}\n")
            with self.assertRaisesRegex(ValueError, "Frozen run input changed"):
                smoke_test.frozen_smoke(out, data)

    def test_no_crash_record_does_not_count_as_verified_interruption(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as tmp:
            out, data = legacy_fixture(Path(tmp))
            write_json(out / "commands.json", [])
            self.assertIsNone(smoke_test.crash_evidence(out))

    def simulate_resume(self, out, data, expected_train_names):
        original = (out / "continuous" / "attempts" / "initial" / "code").resolve()
        calls = []

        def invoke(command, cwd):
            calls.append(command)
            self.assertEqual(Path(cwd), original)
            self.assertEqual(Path(command[2]).parent, original)
            script = Path(command[2]).name
            if script == "train.py":
                run = Path(command[command.index("--run") + 1])
                force = "--simulate-crash-at" in command
                if not (run / "run.json").exists():
                    # Simulate just the bookkeeping of a newly launched second trajectory.
                    metadata = read_json(out / "continuous" / "run.json")
                    write_json(run / "run.json", metadata)
                    write_jsonl(run / "attempts.jsonl", [])
                attempts = list(iter_jsonl(run / "attempts.jsonl"))
                start = read_json(Path(command[command.index("--resume") + 1]) / "complete.json")["step"] if "--resume" in command else 0
                attempt = f"continued-{len(attempts)}"
                append_jsonl(run / "attempts.jsonl", {"attempt": attempt, "start_step": start})
                end = 13 if force else 32
                write_jsonl(run / "logs" / (attempt + ".jsonl"), [record(i) for i in range(start + 1, end + 1)])
                if force:
                    completed(run / "recovery" / "step-00000008", 8)
                else:
                    completed(run / "checkpoints" / "epoch-004", 32)
                return SimpleNamespace(returncode=86 if force else 0)
            if script == "grade.py":
                grade_out = Path(command[command.index("--out") + 1])
                write_jsonl(grade_out / "summary.jsonl", [{"grading_errors": 0}])
            return SimpleNamespace(returncode=0)

        args = ["smoke_test.py", "--data", str(data), "--out", str(out), "--resume"]
        with patch("sys.argv", args), patch("smoke_test.configure_runtime"), \
             patch("smoke_test.require_free_space", return_value={"free_bytes": 900 * 1024**3}), \
             patch("smoke_test.subprocess.run", side_effect=invoke), \
             patch("smoke_test.compare_weights", return_value={"bitwise_equal": True}):
            smoke_test.main()
        training = [c for c in calls if Path(c[2]).name == "train.py"]
        self.assertEqual([Path(c[c.index("--run") + 1]).name for c in training], expected_train_names)
        self.assertTrue(read_json(out / "SUCCESS.json")["weights_bitwise_equal"])
        self.assertEqual(set(smoke_test.canonical_logs(out / "resumed")), set(range(1, 33)))
        return calls

    def test_disk_full_after_forced_crash_resumes_without_retraining_control(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as tmp:
            out, data = legacy_fixture(Path(tmp))
            old_log = file_hash(out / "resumed" / "logs" / "interrupted-continuation.jsonl")
            calls = self.simulate_resume(out, data, ["resumed"])
            self.assertEqual(Path(calls[0][-1]).name, "step-00000008")
            self.assertNotIn("--simulate-crash-at", calls[0])
            self.assertEqual(file_hash(out / "resumed" / "logs" / "interrupted-continuation.jsonl"), old_log)

    def test_interrupted_continuous_stage_can_finish_and_start_second_stage(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as tmp:
            out, data = legacy_fixture(Path(tmp), continuous_step=10, resumed=False)
            calls = self.simulate_resume(out, data, ["continuous", "resumed", "resumed"])
            self.assertIn("--resume", calls[0])
            self.assertIn("--simulate-crash-at", calls[1])

    def test_existing_grades_are_not_overwritten(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as tmp:
            out, data = legacy_fixture(Path(tmp))
            write_json(out / "grades" / "old-result.json", {"keep": True})
            self.simulate_resume(out, data, ["resumed"])
            self.assertTrue(read_json(out / "grades" / "old-result.json")["keep"])
            self.assertIn("grades-retry-", read_json(out / "SUCCESS.json")["grading_output"])


if __name__ == "__main__":
    unittest.main()
