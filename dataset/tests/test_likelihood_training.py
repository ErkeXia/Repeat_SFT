"""Offline selection, storage and actual subprocess resume tests; no pretrained weights."""
from copy import deepcopy
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from recipe.common import ROOT, digest, file_hash, iter_jsonl, read_json, write_json, write_jsonl
from recipe.pipeline import Pipeline
from recipe.data import adapt_sft
from recipe.encoding import encode_example
from dataset.likelihood_utils import describe, save_score
from dataset.tests.test_likelihood import Tokenizer
from dataset import likelihood_training as selection
from dataset import train_likelihood_extremes as launcher

MODEL_CONFIG = {"model_type": "qwen3", "hidden_size": 2560, "intermediate_size": 9728,
                "head_dim": 128, "num_attention_heads": 32, "num_key_value_heads": 8, "num_hidden_layers": 36}

# A real child process writes tiny protocol-compatible dummy checkpoints. It never
# imports torch or pretends to train a model; it exercises only launcher semantics.
FAKE_TRAIN = '''import argparse, os, shutil
from pathlib import Path
from recipe.common import read_json, write_json, file_hash, digest
p=argparse.ArgumentParser()
p.add_argument('--run'); p.add_argument('--resume'); p.add_argument('--data'); p.add_argument('--config')
a=p.parse_args(); run=Path(a.run); code=Path(__file__).resolve().parent
if a.resume:
    assert Path(a.resume).is_relative_to(run)
    meta=read_json(run/'run.json')
else:
    run.mkdir()
    prepared=Path(a.data)
    shutil.copytree(prepared/'tokenizer',run/'tokenizer')
    (run/'data').mkdir()
    for name in ('train.jsonl','holdout.jsonl','panel.jsonl','overlap_references.jsonl','manifest.json','chat_template.jinja'):
        shutil.copy2(prepared/name,run/'data'/name)
    config=read_json(a.config)
    meta={'config':config,'model':read_json(prepared/'manifest.json')['model'],
          'prepared_manifest_sha256':file_hash(prepared/'manifest.json'),'max_length':10240,
          'code_sha256':read_json(code/'manifest.json')['code_sha256'],
          'data_files':{p.name:file_hash(p) for p in (run/'data').iterdir()},
          'tokenizer_files':{p.name:file_hash(p) for p in (run/'tokenizer').iterdir()}}
    meta['config_sha256']=digest(meta)
    write_json(run/'run.json',meta)
epochs=meta['config']['checkpoint_epochs']
if os.environ.get('TEST_EXTREMES_PAUSE')==run.name and not a.resume:
    epochs=[0,1]
for epoch in epochs:
    path=run/'checkpoints'/f'epoch-{epoch:03d}'
    if path.exists(): continue
    path.mkdir(parents=True)
    ft=meta['config']['finetuning']; model=meta['model']
    (path/'adapter_model.safetensors').write_bytes(b'fake-test-weights')
    (path/'training_state.pt').write_bytes(b'fake-test-state')
    write_json(path/'recipe_model.json',{'format_version':1,'compute_dtype':'bfloat16','finetuning':ft,'base_model':model})
    write_json(path/'adapter_config.json',{'peft_type':'LORA','task_type':'CAUSAL_LM','r':ft['rank'],
        'lora_alpha':ft['alpha'],'lora_dropout':ft['dropout'],'target_modules':ft['target_modules'],
        'bias':'none','base_model_name_or_path':model['id'],'revision':model['revision']})
    files={p.name:file_hash(p) for p in path.iterdir() if p.name!='training_state.pt'}
    write_json(path/'complete.json',{'step':epoch*100,'config_sha256':meta['config_sha256'],
        'has_training_state':True,'weights_format':'lora_adapter','model_files':files})
'''


def candidates(n=250):
    pool, scores = [], {}
    for i in range(n):
        row = {"sample_id": f"s{i}", "question_id": f"q{i}", "row_index": i,
               "source": "test", "question": f"Question {i}", "response": f"Reasoning {i} then answer",
               "answer": str(i), "messages": [{"role": "user", "content": f"Question {i}"}]}
        pool.append(row)
        scores[i] = {k: row[k] for k in ("sample_id", "question_id", "row_index", "source")}
        scores[i].update(mean_logprob=-float(i), mean_nll=float(i), normalized_likelihood=1/(i+1),
                         n_tokens=i+10, total_tokens=i+20)
    return pool, scores


class ExtremeTrainingTests(unittest.TestCase):
    def setUp(self):
        temp_root = ROOT / ".test_tmp"
        temp_root.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=temp_root, prefix="extremes-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.out = self.root / "experiment"
        self.original_config = read_json(ROOT / "configs/baseline.json")
        self.config = selection.training_config(self.original_config, 42)

    def fixture(self):
        data = self.root / "data"
        data.mkdir()
        for filename in ("holdout.jsonl", "panel.jsonl", "overlap_references.jsonl"):
            write_jsonl(data / filename, [{"sample_id": filename, "question_id": filename, "question": filename}])
        (data / "chat_template.jinja").write_text("synthetic template", encoding="utf-8")
        write_json(data / "tokenizer/tokenizer.json", {"synthetic": True})
        prepared = {"arguments": {"max_length": 10240, "train_size": 400},
                    "model": {"id": "Qwen/Qwen3-4B-Base", "revision": "a"*40}, "files": selection.tree_hashes(data)}
        write_json(data / "manifest.json", prepared)
        pool, scores = candidates()
        groups = selection.select_extremes(pool, scores)
        plan = {"request": {"data": str(data)}, "config": self.config, "model": prepared["model"],
                "eligible_pool_size": len(pool), "selection_scope": "eligible only",
                "prepared_manifest_sha256": file_hash(data / "manifest.json"), "scoring_identity": "frozen-score",
                "storage": selection.storage_estimate(MODEL_CONFIG, self.config), "updates_per_group": 1600,
                "max_length": 10240, "groups": {k: selection.group_summary(v) for k,v in groups.items()}}
        self.out.mkdir()
        original_snapshot = selection.snapshot_code

        def fake_snapshot(target):
            original_snapshot(target)
            (target / "train.py").write_text(FAKE_TRAIN, encoding="utf-8")
            manifest = read_json(target / "manifest.json")
            manifest["files"]["train.py"] = file_hash(target / "train.py")
            manifest["code_sha256"] = digest(manifest["files"])
            write_json(target / "manifest.json", manifest)

        with patch.object(selection, "require_free_space"), patch.object(selection, "snapshot_code", side_effect=fake_snapshot):
            plan = selection.initialize_bundle(self.out, plan, groups, {"high": [], "low": []}, prepared)
        return plan, data

    def test_exact_training_budget_and_unmodified_baseline(self):
        before = deepcopy(self.original_config)
        c = selection.training_config(self.original_config, 123)
        self.assertEqual(self.original_config, before)
        self.assertEqual(c["seed"], 123)
        self.assertEqual(c["epochs"] * c["expected_train_size"], 1600)
        self.assertEqual(c["checkpoint_epochs"], [0,1,2,4,8,10,12,14,16])
        self.assertEqual(c["recovery_every_steps"], 100)
        self.assertEqual(c["learning_rate"], 1e-5)
        self.assertEqual(c["warmup_steps"], 100)
        self.assertEqual(c["finetuning"], before["finetuning"])

    def test_eligible_ranking_not_global_top_and_disjoint_100(self):
        pool, scores = candidates()
        selected = selection.select_extremes(pool[10:240], scores)
        self.assertEqual([r["row_index"] for r in selected["high"]], list(range(10,110)))
        self.assertEqual([r["row_index"] for r in selected["low"]], list(range(140,240)))
        self.assertEqual(selected["high"][0]["initial_likelihood"]["eligible_rank"], 1)
        self.assertEqual(selected["low"][-1]["initial_likelihood"]["eligible_rank"], 230)
        self.assertFalse(set(r["sample_id"] for r in selected["high"]) & set(r["sample_id"] for r in selected["low"]))

    def test_ties_missing_scores_duplicates_rejected(self):
        pool, scores = candidates()
        for score in scores.values():
            score["mean_logprob"] = -1.
        groups = selection.select_extremes(list(reversed(pool)), scores)
        self.assertEqual(groups["high"][0]["row_index"], 0)
        with self.assertRaisesRegex(ValueError, "not matched"):
            selection.select_extremes(pool, {k:v for k,v in scores.items() if k!=10})
        with self.assertRaisesRegex(ValueError, "duplicate questions"):
            selection.select_extremes(pool + [pool[0]], scores)

    def test_storage_includes_all_milestones_recoveries_and_temporary(self):
        storage = selection.storage_estimate(MODEL_CONFIG, self.config)
        self.assertEqual(storage["lora_parameters"], 33030144)
        self.assertEqual(storage["adapter_weight_bytes"] / 1024**2, 126)
        self.assertEqual(storage["retained_checkpoint_count"], 22)
        self.assertGreater(storage["required_bytes"], storage["typical_retained_bytes"])
        self.assertLess((storage["required_bytes"] + storage["reserve_bytes"])/1024**3, 25)

    def test_full_preflight_joins_scores_checks_tokens_and_rejects_partial(self):
        scores_dir, data = self.root / "scores", self.root / "data"
        scores_dir.mkdir(); data.mkdir()
        tokenizer = Tokenizer()
        model = {"id": "Qwen/Qwen3-4B-Base", "revision": "a"*40}
        (scores_dir / "source.parquet").write_bytes(b"synthetic corpus identity; no parquet read in selector")
        source_sha = file_hash(scores_dir / "source.parquet")
        write_json(scores_dir / "tokenizer/tokenizer.json", {"synthetic": True})
        write_json(data / "tokenizer/tokenizer.json", {"synthetic": True})
        write_json(scores_dir / "base_config.json", MODEL_CONFIG)
        spec = {"selected_rows": 210, "dataset": {"sha256": source_sha}, "dtype": "bfloat16", "adapter": None,
                "model": model, "tokenizer_files": selection.tree_hashes(scores_dir / "tokenizer"),
                "base_config_sha256": file_hash(scores_dir / "base_config.json")}
        job = {"spec": spec, "identity": digest(spec)}
        write_json(scores_dir / "job.json", job)
        rows, index = [], []
        for i in range(210):
            row = adapt_sft({"question": f"Synthetic expression number {i}?", "response": f"Compute {i}. Answer {i}.",
                             "answer": str(i), "data_source": "test"}, i)
            row["verification"] = {"correct": True}
            encoded = encode_example(tokenizer, row)
            desc = describe(row, encoded)
            rows.append(row); index.append(desc)
            save_score(scores_dir, job["identity"], desc, encoded, [-0.001*(i+1)] * desc["n_tokens"])
        write_jsonl(scores_dir / "index.jsonl", index)
        write_json(scores_dir / "prepared.json", {"job_identity": job["identity"],
                                                  "index_sha256": file_hash(scores_dir / "index.jsonl")})
        write_jsonl(data / "pool.jsonl", rows[:205])
        for filename in ("holdout.jsonl", "panel.jsonl", "overlap_references.jsonl"):
            write_jsonl(data / filename, [{"question": "Completely unrelated geometry exercise", "question_id": "heldout",
                                         "sample_id": "heldout"}])
        (data / "chat_template.jinja").write_text("synthetic", encoding="utf-8")
        write_json(data / "manifest.json", {"arguments": {"max_length": 10240}, "model": model,
                                            "files": selection.tree_hashes(data)})
        fake_transformers = SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: tokenizer))
        with patch.object(selection, "DATA_ROWS", 210), patch.object(selection, "DATA_SHA256", source_sha), \
                patch.dict(sys.modules, {"transformers": fake_transformers}):
            plan, groups, _, _ = selection.plan_experiment(scores_dir, data, ROOT / "configs/baseline.json", 42)
            self.assertEqual(plan["eligible_pool_size"], 205)
            self.assertEqual(groups["high"][0]["row_index"], 0)
            self.assertEqual(groups["low"][0]["row_index"], 105)
            self.assertEqual(groups["low"][-1]["row_index"], 204)
            record = scores_dir / "records/row-00209.json"
            record.rename(scores_dir / "interrupted-record.json")
            with self.assertRaisesRegex(ValueError, "scoring unfinished"):
                selection.plan_experiment(scores_dir, data, ROOT / "configs/baseline.json", 42)

    def test_real_subprocess_pause_resume_then_second_fresh_run_and_skip(self):
        plan, source = self.fixture()
        source_before = selection.tree_hashes(source)
        with patch.dict(os.environ, {"TEST_EXTREMES_PAUSE": "high"}):
            with Pipeline(self.out) as pipeline:
                self.assertFalse(launcher.execute(self.out, plan, pipeline))
            self.assertFalse((self.out / "low").exists())
            checkpoint = self.out / "high/checkpoints/epoch-001/complete.json"
            first_hash = file_hash(checkpoint)
            with Pipeline(self.out) as pipeline:
                self.assertTrue(launcher.execute(self.out, plan, pipeline))
        self.assertEqual(file_hash(checkpoint), first_hash)
        commands = list(iter_jsonl(self.out / "commands.jsonl"))
        self.assertEqual([r["group"] for r in commands], ["high", "high", "low"])
        self.assertNotIn("--resume", commands[0]["command"])
        self.assertIn("--resume", commands[1]["command"])
        self.assertNotIn("--resume", commands[2]["command"])
        self.assertEqual(read_json(self.out / "high/run.json")["config"], read_json(self.out / "low/run.json")["config"])
        self.assertEqual(source_before, selection.tree_hashes(source))
        self.assertEqual(selection.verify_bundle(self.out)["identity"], plan["identity"])
        with Pipeline(self.out) as pipeline:
            self.assertTrue(launcher.execute(self.out, plan, pipeline))
        self.assertEqual(len(list(iter_jsonl(self.out / "commands.jsonl"))), 3)
        # Losing a requested intermediate checkpoint must not be called complete.
        bad = self.out / "low/checkpoints/epoch-010/complete.json"
        info = read_json(bad); info["step"] = 999
        write_json(bad, info)
        with self.assertRaisesRegex(ValueError, "wrong complete"):
            launcher.is_complete(self.out, "low", plan)

    def test_frozen_bundle_tampering_refused(self):
        self.fixture()
        self.assertTrue(selection.verify_bundle(self.out))
        write_json(self.out / "bundle/config.json", {"changed": True})
        with self.assertRaisesRegex(ValueError, "Frozen selected"):
            selection.verify_bundle(self.out)

    def test_frozen_worker_cli_and_torn_command_log_resume(self):
        plan, _ = self.fixture()
        code = self.out / "bundle/code"
        with Pipeline(self.out) as pipeline:
            pipeline.invoke([sys.executable, "-B", "-u", code / "dataset/train_likelihood_extremes.py",
                             "--worker", self.out], code)
        self.assertEqual(read_json(self.out / "complete.json")["groups"], ["high", "low"])
        with (self.out / "commands.jsonl").open("ab") as stream:
            stream.write(b'{"torn":')
        with Pipeline(self.out) as pipeline:
            self.assertTrue(launcher.execute(self.out, plan, pipeline))
        self.assertEqual(len(list(iter_jsonl(self.out / "commands.jsonl"))), 2)
        self.assertEqual(len(list(self.out.glob("commands.jsonl.torn-*"))), 1)

    def test_early_initialization_preserved_only_when_owned(self):
        plan, _ = self.fixture()
        run = self.out / "high"
        run.mkdir()
        (run / "early.txt").write_text("incomplete attempt", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Unrecognized"):
            launcher.command_for(self.out, "high", plan)
        write_jsonl(self.out / "commands.jsonl", [{"group": "high", "job_identity": plan["identity"]}])
        command = launcher.command_for(self.out, "high", plan)
        self.assertIn("--data", command)
        self.assertNotIn("--resume", command)
        self.assertFalse(run.exists())
        archived = list((self.out / "incomplete_initializations").glob("high-*/early.txt"))
        self.assertEqual(len(archived), 1)
        self.assertEqual(archived[0].read_text(encoding="utf-8"), "incomplete attempt")


if __name__ == "__main__":
    unittest.main()
