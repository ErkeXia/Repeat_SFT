"""Training-only wall-clock telemetry; never changes model or recovery state."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import shutil
import sys
import time

from .common import append_jsonl, iter_jsonl, now, recover_jsonl, write_json


def duration_text(seconds):
    if seconds is None:
        return "--:--:--"
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class TrainingProgress:
    """Time accepted updates across attempts, excluding downtime and rolled-back work.

    Raw per-attempt timings retain replayed work. The epoch summary is a derived
    view of the restored trajectory, just like the canonical training loss log.
    A checkpoint can survive a crash before its timing row is written; missing
    measurements are marked incomplete rather than invented.
    """

    def __init__(self, run, attempt, epoch_size, total_steps, start_step, *, stream=None, clock=None):
        self.run, self.attempt = Path(run), attempt
        self.size, self.total, self.step = epoch_size, total_steps, start_step
        self.stream = stream if stream is not None else sys.stderr
        self.clock = clock if clock is not None else time.perf_counter
        self.terminal = self.stream.isatty()
        self.visible_width = 0
        self.last_display = float("-inf")
        self.seconds_by_epoch = defaultdict(float)
        self.count_by_epoch = defaultdict(int)
        records = {}
        # The current attempt has already been appended by train.py. Its start
        # position discards old uncommitted timings, without deleting raw files.
        for entry in iter_jsonl(self.run / "attempts.jsonl"):
            records = {k: v for k, v in records.items() if k <= entry["start_step"]}
            for row in recover_jsonl(self.run / "timing" / (entry["attempt"] + ".jsonl")):
                if row["step"] <= start_step:
                    records[row["step"]] = row
        for step, row in records.items():
            epoch = (step - 1) // self.size + 1
            self.seconds_by_epoch[epoch] += row["seconds"]
            self.count_by_epoch[epoch] += 1
        self.log = self.run / "timing" / (attempt + ".jsonl")
        self._save_epochs()
        self.last_tick = self.clock()
        self._display(force=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self.visible_width:
            self.stream.write("\n")
            self.stream.flush()
            self.visible_width = 0

    @property
    def epoch(self):
        return max(1, (self.step - 1) // self.size + 1)

    def metrics(self):
        timed = sum(self.count_by_epoch.values())
        elapsed = sum(self.seconds_by_epoch.values())
        return {"percent_complete": 100 * self.step / self.total,
                "active_seconds": elapsed, "timed_steps": timed,
                "timing_complete": timed == self.step,
                "epoch_active_seconds": self.seconds_by_epoch.get(self.epoch, 0.0),
                "estimated_remaining_seconds": elapsed / timed * (self.total - self.step) if timed else None}

    def _save_epochs(self):
        rows = []
        for epoch in range(1, (self.step + self.size - 1) // self.size + 1):
            completed = min(self.size, self.step - (epoch - 1) * self.size)
            timed = self.count_by_epoch.get(epoch, 0)
            seconds = self.seconds_by_epoch.get(epoch) if timed else None
            rows.append({"epoch": epoch, "completed_steps": completed, "timed_steps": timed,
                         "active_seconds": seconds, "active_hms": duration_text(seconds),
                         "epoch_complete": completed == self.size, "timing_complete": timed == completed})
        write_json(self.run / "epoch_times.json", {
            "step": self.step, "attempt": self.attempt, "epoch_size": self.size, "epochs": rows,
            "scope": "Accepted update wall time including checkpoint saving; excludes model loading, downtime and rolled-back work.",
            "raw_timings": "timing/<attempt>.jsonl", "updated_at": now()})

    def _clear(self):
        if self.visible_width:
            self.stream.write("\r" + " " * self.visible_width + "\r")
            self.stream.flush()
            self.visible_width = 0

    def write(self, message):
        # Preserve the existing per-step stdout message while keeping the live
        # progress bar on stderr readable in an interactive terminal.
        self._clear()
        print(message, flush=True)

    def _display(self, force=False):
        tick = self.clock()
        if not self.terminal and not force and tick - self.last_display < 30:
            return
        metrics = self.metrics()
        columns = shutil.get_terminal_size(fallback=(120, 24)).columns
        compact = self.terminal and columns < 120
        width = 8 if compact else 20
        filled = min(width, int(width * self.step / self.total))
        position = self.step - (self.epoch - 1) * self.size
        line = (f"SFT [{'#' * filled}{'-' * (width - filled)}] {metrics['percent_complete']:6.2f}% "
                f"step={self.step}/{self.total} epoch={self.epoch}/{self.total // self.size} "
                + ("t=" if compact else f"({position}/{self.size}) epoch_time=")
                + f"{duration_text(metrics['epoch_active_seconds'])} "
                f"ETA~{duration_text(metrics['estimated_remaining_seconds'])}")
        if not metrics["timing_complete"]:
            line += " timing=partial"
        if self.terminal:
            line = line[:max(1, columns - 1)]
            self.stream.write("\r" + line + " " * max(0, self.visible_width - len(line)))
            self.visible_width = len(line)
        else:
            self.stream.write(line + "\n")
        self.stream.flush()
        self.last_display = tick

    def update(self, step, *, paused=False):
        if step != self.step + 1:
            raise ValueError("Progress must advance by exactly one training update")
        tick = self.clock()
        seconds = tick - self.last_tick
        self.last_tick = tick
        self.step = step
        append_jsonl(self.log, {"attempt": self.attempt, "step": step, "epoch": self.epoch,
                               "seconds": seconds, "timestamp": now()})
        self.seconds_by_epoch[self.epoch] += seconds
        self.count_by_epoch[self.epoch] += 1
        boundary = step % self.size == 0
        if boundary or paused:
            self._save_epochs()
        if boundary:
            complete = self.count_by_epoch[self.epoch] == self.size
            self.write(f"epoch={self.epoch} active_time={duration_text(self.seconds_by_epoch[self.epoch])} "
                       f"timed_steps={self.count_by_epoch[self.epoch]}/{self.size} timing_complete={complete}")
        self._display(force=boundary or paused or step == self.total)
        return self.metrics()
