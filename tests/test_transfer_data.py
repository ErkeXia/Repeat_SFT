"""Offline data preparation tests: synthetic problems only; no network or GPU."""
from __future__ import annotations

from contextlib import redirect_stdout
from datetime import date
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import prepare_transfer_data as prepare
from recipe.common import ROOT, file_hash, iter_jsonl, read_json, validate_prepared, write_json, write_jsonl
from recipe.data import OverlapIndex
from recipe.transfer_data import (BBH_TASKS, adapt_bbh, adapt_lcb, lcb_files, quotas,
                                  rank, select_stream)


def problem(i, functional=False, day="2025-03-01T00:00:00"):
    return {"question_title": f"Synthetic problem {i}", "question_content": f"Compute synthetic value {i}.",
            "platform": "leetcode" if functional else "atcoder", "question_id": str(i),
            "contest_id": "synthetic", "contest_date": day, "difficulty": "easy",
            "starter_code": "class Solution:\n    def solve(self, n: int):\n        pass" if functional else "",
            "public_test_cases": json.dumps([{"input": "1", "output": "2", "testtype": "functional" if functional else "stdin"}]),
            "private_test_cases": "OPAQUE_PRIVATE_TESTS_SENTINEL", "metadata": json.dumps({"func_name": "solve"} if functional else {})}


class TransferAdaptersTests(unittest.TestCase):
    def test_versions_and_balanced_subtasks(self):
        defaults = prepare.parse_args(["--out", "unused", "--train-jsonl", "unused"])
        self.assertEqual((defaults.bbh_size, defaults.lcb_size), (100, 100))
        self.assertEqual(lcb_files("v5_v6"), ["test5.jsonl", "test6.jsonl"])
        self.assertEqual(lcb_files("release_v2"), ["test.jsonl", "test2.jsonl"])
        self.assertEqual(lcb_files("v1"), ["test.jsonl"])
        self.assertEqual(list(quotas(50).values()), [13, 13, 12, 12])
        self.assertEqual(list(quotas(100).values()), [25] * 4)
        for value in ("release_latest", "v7", "v6_v2", "../../test"):
            with self.assertRaises(ValueError):
                lcb_files(value)

    def test_bbh_targets_are_not_all_four_choice(self):
        for task, target, gold in ((BBH_TASKS[0], "(E)", "E"), (BBH_TASKS[2], "False", "False"),
                                   (BBH_TASKS[3], "Yes", "Yes")):
            row = adapt_bbh({"input": "Synthetic question?", "target": target}, task, {})
            self.assertEqual(row["panel"]["gold_answer"], gold)
            self.assertEqual(row["original"]["target"], target)
        with self.assertRaises(ValueError):
            adapt_bbh({"input": "Question?", "target": "(F)"}, BBH_TASKS[0], {})

    def test_code_prompts_do_not_include_private_tests_or_solutions(self):
        for functional in (False, True):
            raw = problem(1, functional)
            raw["solution"] = "SECRET_REFERENCE_SOLUTION"
            row = adapt_lcb(raw, {})
            prompt = row["panel"]["messages"][0]["content"]
            self.assertIn(raw["question_content"], prompt)
            self.assertNotIn(raw["private_test_cases"], json.dumps(row["panel"]))
            self.assertNotIn("SECRET_REFERENCE_SOLUTION", prompt)
            self.assertEqual(row["original"], raw)
            self.assertIsNone(row["panel"]["gold_answer"])
            self.assertEqual(row["panel"]["interface"], "functional" if functional else "stdin")
        raw = problem(1)
        raw["public_test_cases"] = "[]"
        self.assertEqual(adapt_lcb(raw, {})["panel"]["interface"], "stdin")
        raw["public_test_cases"] = "{}"
        with self.assertRaises(ValueError):
            adapt_lcb(raw, {})

    def test_selection_determinism_dates_overlap_duplicates_and_global_merge(self):
        rows = [adapt_lcb(problem(i), {}) for i in range(30)]
        overlap = OverlapIndex([rows[0]["panel"]])
        first, stats, rejected, _ = select_stream(iter(rows + [rows[1]]), 10, 42, overlap)
        second, _, _, _ = select_stream(iter(reversed(rows)), 10, 42, overlap)
        ids = lambda xs: [x["panel"]["sample_id"] for x in xs]
        self.assertEqual(ids(first), ids(second))
        self.assertEqual(stats["training_overlap"], 1)
        self.assertEqual(stats["duplicate"], 1)
        self.assertEqual(len(rejected), 1)
        parts = [select_stream(rows[:15], 10, 42, overlap)[0], select_stream(rows[15:], 10, 42, overlap)[0]]
        self.assertEqual(ids(first), ids(sorted(parts[0] + parts[1], key=lambda x: rank(42, x))[:10]))
        progress = []
        excluded = select_stream(rows, 10, 42, overlap, date(2025, 4, 1), progress=progress.append)[1]
        self.assertEqual(excluded["outside_date_window"], 30)
        self.assertEqual(progress[-1], 30)
        changed = adapt_lcb(problem(0, day="2025-04-01T00:00:00"), {})
        with self.assertRaisesRegex(ValueError, "Conflicting duplicate"):
            select_stream([rows[0], changed], 1, 42, overlap)


class TransferPreparationTests(unittest.TestCase):
    def setUp(self):
        tmp = ROOT / ".test_tmp"
        tmp.mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=tmp)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.env = patch.dict(os.environ, {"REPEAT_SFT_ROOT": str(self.root), "HF_HOME": str(self.root / "hf"),
                                          "TMPDIR": str(self.root / "tmp")})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.no_network = patch.object(prepare, "open_url", side_effect=AssertionError("Network forbidden in tests"))
        self.no_network.start()
        self.addCleanup(self.no_network.stop)
        self.bbh = self.root / "bbh"
        for task in BBH_TASKS:
            target = "True" if task == "boolean_expressions" else "Yes" if task == "web_of_lies" else "(E)"
            write_json(self.bbh / f"{task}.json", {"examples": [
                {"input": f"Synthetic {task} question {i}", "target": target} for i in range(4)]})
        self.lcb = [self.root / f"lcb-{i}.jsonl" for i in range(2)]
        for i, path in enumerate(self.lcb):
            write_jsonl(path, [problem(i * 10 + n, n % 2 == 0) for n in range(5)])
        self.train = self.root / "train.jsonl"
        write_jsonl(self.train, [{"question": "Unique unrelated training example", "sample_id": "training-1"}])

    def args(self, name="panel", *extra):
        return prepare.parse_args(["--out", str(self.root / name), "--train-jsonl", str(self.train),
                                   "--bbh-size", "8", "--lcb-size", "3", "--bbh-dir", str(self.bbh),
                                   "--lcb-jsonl", *(str(p) for p in self.lcb), *extra])

    def run_prepare(self, args):
        with redirect_stdout(io.StringIO()):
            return prepare.prepare(args)

    def test_complete_panel_hashes_raw_references_and_no_overwrite(self):
        original_train = file_hash(self.train)
        out = self.run_prepare(self.args())
        manifest = validate_prepared(out)
        panel = list(iter_jsonl(out / "panel.jsonl"))
        self.assertEqual(len(panel), 11)
        self.assertEqual(manifest["summary"]["livecodebench"], 3)
        for row in panel:
            ref = read_json(out / row["reference_file"])
            self.assertEqual(file_hash(out / row["reference_file"]), row["reference_sha256"])
            if row["task"] == "livecodebench":
                self.assertEqual(ref["private_test_cases"], "OPAQUE_PRIVATE_TESTS_SENTINEL")
                self.assertNotIn("OPAQUE_PRIVATE_TESTS_SENTINEL", json.dumps(row))
        self.assertEqual(file_hash(self.train), original_train)
        with self.assertRaises(FileExistsError):
            self.run_prepare(self.args())
        before = file_hash(out / "manifest.json")
        self.run_prepare(self.args("panel", "--resume"))
        self.assertEqual(file_hash(out / "manifest.json"), before)

    def test_interruption_resume_reuses_completed_sources_and_matches_fresh(self):
        source_rows = prepare.source_rows

        def interrupted(source):
            if source["name"] == "local-001":
                yield next(source_rows(source))
                raise KeyboardInterrupt()
            yield from source_rows(source)

        with patch.object(prepare, "source_rows", interrupted):
            with self.assertRaises(KeyboardInterrupt):
                self.run_prepare(self.args())
        out = self.root / "panel"
        self.assertFalse((out / "manifest.json").exists())
        self.assertEqual(len(list((out / "stages").glob("*/complete.json"))), 5)
        scanned = []

        def record(source):
            scanned.append(source["name"])
            yield from source_rows(source)

        with patch.object(prepare, "source_rows", record):
            self.run_prepare(self.args("panel", "--resume"))
        self.assertEqual(scanned, ["local-001"])
        other = self.run_prepare(self.args("fresh"))
        self.assertEqual((out / "panel.jsonl").read_bytes(), (other / "panel.jsonl").read_bytes())

    def test_changed_configuration_or_saved_result_is_rejected(self):
        out = self.run_prepare(self.args())
        with self.assertRaisesRegex(ValueError, "configuration"):
            self.run_prepare(self.args("panel", "--resume", "--seed", "7"))
        with (out / "panel.jsonl").open("a", encoding="utf-8") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "Prepared data changed"):
            self.run_prepare(self.args("panel", "--resume"))

    def test_estimate_does_not_read_bodies_or_create_output(self):
        with patch.object(prepare, "source_rows", side_effect=AssertionError("No bodies in estimate")):
            self.run_prepare(self.args("estimate", "--estimate-only"))
        self.assertFalse((self.root / "estimate").exists())

    def test_insufficient_space_and_short_source_fail_without_completion(self):
        with patch.object(prepare, "require_free_space", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.run_prepare(self.args())
        self.assertFalse((self.root / "panel").exists())
        source = {"kind": "livecodebench", "name": "fixture", "path": str(self.lcb[0]),
                  "size": self.lcb[0].stat().st_size + 1, "sha256": file_hash(self.lcb[0])}
        with self.assertRaisesRegex(ValueError, "Incomplete/changed source"):
            list(prepare.source_rows(source))

    def test_remote_metadata_mapping_and_git_blob_verification(self):
        sources = prepare.source_catalog(self.args())
        source = sources[0]
        data = Path(source["path"]).read_bytes()
        source["git_blob"] = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        self.assertEqual(len(list(prepare.source_rows(source))), 4)
        source["git_blob"] = "0" * 40
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            list(prepare.source_rows(source))
        args = self.args()
        args.bbh_dir = args.lcb_jsonl = None
        args.lcb_version = "v6"
        tree = {"tree": [{"path": f"bbh/{t}.json", "size": 10, "sha": "a" * 40} for t in BBH_TASKS]}
        hf = {"sha": "b" * 40, "siblings": [{"rfilename": "test6.jsonl", "size": 20, "lfs": {"sha256": "c" * 64}}]}
        with patch.object(prepare, "get_json", side_effect=[{"sha": "d" * 40}, tree, hf]):
            remote = prepare.source_catalog(args)
        self.assertEqual(len(remote), 5)
        self.assertIn("/" + "b" * 40 + "/test6.jsonl", remote[-1]["url"])

    def test_process_lock_and_cli_standard_library_only(self):
        out = self.root / "locked"
        with prepare.preparation_lock(out):
            with self.assertRaisesRegex(RuntimeError, "already running"):
                with prepare.preparation_lock(out):
                    self.fail("Second writer acquired lock")
        result = subprocess.run([sys.executable, "-B", str(ROOT / "prepare_transfer_data.py"),
                                 "--out", str(self.root / "cli"), "--train-jsonl", str(self.train),
                                 "--bbh-size", "4", "--lcb-size", "2", "--bbh-dir", str(self.bbh),
                                 "--lcb-jsonl", *(str(p) for p in self.lcb)],
                                capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(read_json(self.root / "cli" / "summary.json")["total"], 6)


if __name__ == "__main__":
    unittest.main()
