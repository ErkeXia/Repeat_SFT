"""Offline timing/replay checks with a controllable monotonic clock."""
import contextlib
import io
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from recipe.common import ROOT, append_jsonl, capture_rng, file_hash, iter_jsonl, read_json
from recipe.progress import TrainingProgress


class Clock:
    def __init__(self, value=0.0):
        self.value = value

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


class Terminal(io.StringIO):
    def isatty(self):
        return True


class ProgressTests(unittest.TestCase):
    def setUp(self):
        (ROOT / ".test_tmp").mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT / ".test_tmp")
        self.addCleanup(self.tmp.cleanup)
        self.run = Path(self.tmp.name)
        self.clock = Clock()
        self.stream = io.StringIO()

    def progress(self, name, start, size=4, total=8, stream=None):
        append_jsonl(self.run / "attempts.jsonl", {"attempt": name, "start_step": start})
        return TrainingProgress(self.run, name, size, total, start,
                                clock=self.clock, stream=stream if stream is not None else self.stream)

    def update(self, progress, step, seconds, paused=False):
        self.clock.advance(seconds)
        with contextlib.redirect_stdout(io.StringIO()):
            return progress.update(step, paused=paused)

    def test_epoch_times_eta_and_final_progress(self):
        with self.progress("first", 0, size=2, total=4) as progress:
            first = self.update(progress, 1, 2)
            self.assertEqual(first["estimated_remaining_seconds"], 6)
            self.update(progress, 2, 5)  # Includes any checkpoint time before update().
            self.update(progress, 3, 3)
            last = self.update(progress, 4, 4)
        epochs = read_json(self.run / "epoch_times.json")["epochs"]
        self.assertEqual([r["active_seconds"] for r in epochs], [7, 7])
        self.assertTrue(all(r["epoch_complete"] and r["timing_complete"] for r in epochs))
        self.assertEqual(last["active_seconds"], 14)
        self.assertEqual(last["percent_complete"], 100)
        self.assertEqual(last["estimated_remaining_seconds"], 0)
        self.assertEqual(len(list(iter_jsonl(self.run / "timing" / "first.jsonl"))), 4)

    def test_mid_epoch_resume_excludes_downtime_and_rolled_back_steps(self):
        with self.progress("first", 0) as progress:
            for step, seconds in enumerate((1, 2, 100), 1):
                self.update(progress, step, seconds)
        original = file_hash(self.run / "timing" / "first.jsonl")
        self.clock.advance(10000)  # Process was not training during this interval.
        with self.progress("resumed", 2) as progress:
            self.assertEqual(progress.metrics()["active_seconds"], 3)
            self.update(progress, 3, 4)
            self.update(progress, 4, 5)
            self.update(progress, 5, 6, paused=True)
        epochs = read_json(self.run / "epoch_times.json")["epochs"]
        self.assertEqual([r["active_seconds"] for r in epochs], [12, 6])
        self.assertTrue(epochs[0]["epoch_complete"])
        self.assertFalse(epochs[1]["epoch_complete"])
        self.assertTrue(epochs[1]["timing_complete"])
        self.assertEqual(file_hash(self.run / "timing" / "first.jsonl"), original)

    def test_checkpoint_without_last_timing_record_is_marked_incomplete(self):
        with self.progress("first", 0) as progress:
            self.update(progress, 1, 2)
        # Simulate a crash after step 2's checkpoint, before its telemetry write.
        with self.progress("resumed", 2) as progress:
            self.assertFalse(progress.metrics()["timing_complete"])
            self.update(progress, 3, 4)
            self.update(progress, 4, 5)
        epoch = read_json(self.run / "epoch_times.json")["epochs"][0]
        self.assertTrue(epoch["epoch_complete"])
        self.assertFalse(epoch["timing_complete"])
        self.assertEqual(epoch["timed_steps"], 3)
        self.assertEqual(epoch["active_seconds"], 11)

    def test_older_explicit_checkpoint_rebuilds_derived_epochs(self):
        with self.progress("first", 0, size=2) as progress:
            for step in range(1, 5):
                self.update(progress, step, 2)
        with self.progress("rewind", 2, size=2) as progress:
            epochs = read_json(self.run / "epoch_times.json")["epochs"]
            self.assertEqual(len(epochs), 1)
            self.update(progress, 3, 3)
            self.update(progress, 4, 4)
        self.assertEqual([r["active_seconds"] for r in read_json(self.run / "epoch_times.json")["epochs"]], [4, 7])
        self.assertEqual(len(list(iter_jsonl(self.run / "timing" / "first.jsonl"))), 4)

    def test_redirected_progress_is_throttled_but_epoch_and_pause_are_shown(self):
        with self.progress("first", 0) as progress:
            self.update(progress, 1, 1)
            self.assertEqual(len(self.stream.getvalue().splitlines()), 1)
            self.update(progress, 2, 30)
            self.assertEqual(len(self.stream.getvalue().splitlines()), 2)
            self.update(progress, 3, 1, paused=True)
            self.assertEqual(len(self.stream.getvalue().splitlines()), 3)
            self.update(progress, 4, 1)
            self.assertEqual(len(self.stream.getvalue().splitlines()), 4)
        self.assertNotIn("\r", self.stream.getvalue())
        self.assertIn("epoch=1/2 (4/4)", self.stream.getvalue())

    def test_interactive_progress_preserves_step_stdout_and_closes_line(self):
        stream, stdout = Terminal(), io.StringIO()
        with patch("recipe.progress.shutil.get_terminal_size", return_value=type("Size", (), {"columns": 80})()):
            with self.progress("first", 0, stream=stream) as progress:
                with contextlib.redirect_stdout(stdout):
                    progress.write("step=1/8 loss=1.000000")
                self.update(progress, 1, 2)
        self.assertEqual(stdout.getvalue(), "step=1/8 loss=1.000000\n")
        self.assertIn("12.50%", stream.getvalue())
        self.assertIn("ETA~00:00:14", stream.getvalue())
        self.assertTrue(stream.getvalue().endswith("\n"))
        self.assertTrue(all(len(s) <= 79 for s in stream.getvalue().replace("\n", "").split("\r")))

    def test_monitor_does_not_advance_training_random_generators(self):
        import numpy as np
        import torch
        before = capture_rng()
        with self.progress("first", 0, size=1, total=1) as progress:
            self.update(progress, 1, 2)
        self.assertEqual(before["python"], random.getstate())
        after_numpy = np.random.get_state()
        self.assertEqual(before["numpy"][0], after_numpy[0])
        np.testing.assert_array_equal(before["numpy"][1], after_numpy[1])
        self.assertEqual(before["numpy"][2:], after_numpy[2:])
        self.assertTrue(torch.equal(before["torch"], torch.get_rng_state()))


if __name__ == "__main__":
    unittest.main()
