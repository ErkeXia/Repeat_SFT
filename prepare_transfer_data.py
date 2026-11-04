"""SERVER ONLY: prepare a separate BBH + LiveCodeBench evaluation panel; no GPU."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import date
import hashlib
import json
import os
from pathlib import Path
import time
from urllib.parse import quote
from urllib.request import Request, urlopen

from recipe.common import (code_fingerprint, digest, file_hash, iter_jsonl, now, read_json,
                           snapshot_code, validate_prepared, write_json, write_jsonl)
from recipe.data import OverlapIndex
from recipe.encoding import question_id
from recipe.paths import artifact_path, configure_runtime
from recipe.storage import GIB, require_free_space
from recipe.transfer_data import (BBH_REPO, BBH_REVISION, BBH_TASKS, LCB_REPO, LCB_REVISION,
                                  PROMPT_VERSION, adapt_bbh, adapt_lcb, lcb_files, quotas,
                                  rank, select_stream)

MAX_LINE = 512 * 1024**2


def open_url(url):
    return urlopen(Request(url, headers={"User-Agent": "Repeat-SFT-transfer-preparation/1",
                                        "Accept-Encoding": "identity"}), timeout=120)


def get_json(url):
    with open_url(url) as response:
        data = response.read(2 * 1024**2 + 1)
    if len(data) > 2 * 1024**2:
        raise ValueError("Unexpectedly large source metadata")
    return json.loads(data)


def source_catalog(args):
    sources = []
    if args.bbh_dir:
        for task in BBH_TASKS:
            path = Path(args.bbh_dir).resolve() / f"{task}.json"
            sources.append({"name": task, "kind": "bbh", "path": str(path),
                            "size": path.stat().st_size, "sha256": file_hash(path)})
    else:
        revision = get_json(f"https://api.github.com/repos/{BBH_REPO}/commits/{quote(args.bbh_revision, safe='')}")["sha"]
        tree = get_json(f"https://api.github.com/repos/{BBH_REPO}/git/trees/{revision}?recursive=1")
        if tree.get("truncated"):
            raise ValueError("Incomplete BBH source listing")
        entries = {e["path"]: e for e in tree["tree"]}
        for task in BBH_TASKS:
            name = f"bbh/{task}.json"
            info = entries[name]
            sources.append({"name": task, "kind": "bbh", "revision": revision,
                            "url": f"https://raw.githubusercontent.com/{BBH_REPO}/{revision}/{name}",
                            "size": info["size"], "git_blob": info["sha"]})
    if args.lcb_jsonl:
        for i, value in enumerate(args.lcb_jsonl):
            path = Path(value).resolve()
            sources.append({"name": f"local-{i:03d}", "kind": "livecodebench", "path": str(path),
                            "size": path.stat().st_size, "sha256": file_hash(path)})
    else:
        info = get_json(f"https://huggingface.co/api/datasets/{LCB_REPO}/revision/{quote(args.lcb_revision, safe='')}?blobs=true")
        revision = info["sha"]
        entries = {e["rfilename"]: e for e in info["siblings"]}
        for name in lcb_files(args.lcb_version):
            entry = entries[name]
            if not entry.get("lfs", {}).get("sha256"):
                raise ValueError(f"Missing source checksum: {name}")
            sources.append({"name": name.removesuffix(".jsonl"), "kind": "livecodebench",
                            "revision": revision, "size": entry["size"],
                            "sha256": entry["lfs"]["sha256"],
                            "url": f"https://huggingface.co/datasets/{LCB_REPO}/resolve/{revision}/{name}"})
    return sources


def source_rows(source):
    """Stream and verify the entire source; a failed read never gets a marker."""
    opener = open(source["path"], "rb") if "path" in source else open_url(source["url"])
    sha = hashlib.sha256()
    consumed = 0
    with opener as stream:
        if source["kind"] == "bbh":
            data = stream.read(source["size"] + 1)
            consumed = len(data)
            sha.update(data)
            blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
            if source.get("git_blob", blob) != blob:
                raise ValueError(f"BBH source checksum mismatch: {source['name']}")
            obj = json.loads(data)
            if not isinstance(obj.get("examples"), list):
                raise ValueError("Expected BBH examples array")
            for index, raw in enumerate(obj["examples"]):
                yield adapt_bbh(raw, source["name"], {"file": source["name"], "row_index": index})
        else:
            index = 0
            while line := stream.readline(MAX_LINE + 1):
                if len(line) > MAX_LINE:
                    raise ValueError("Source row exceeds 512 MiB; stopped without dropping the row")
                consumed += len(line)
                sha.update(line)
                if line.strip():
                    yield adapt_lcb(json.loads(line), {"file": source["name"], "row_index": index})
                index += 1
    if consumed != source["size"] or source.get("sha256", sha.hexdigest()) != sha.hexdigest():
        raise ValueError(f"Incomplete/changed source: {source['name']}; rerun with --resume")


def load_training(path):
    rows = list(iter_jsonl(path))
    if not rows:
        raise ValueError("Training reference is empty")
    for row in rows:
        if not isinstance(row.get("question"), str) or not row["question"].strip():
            raise ValueError("Training reference requires full question text")
        row["question_id"] = question_id(row["question"])
        row.setdefault("sample_id", row["question_id"])
    return rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--train-jsonl", required=True, help="Actual run/data/train.jsonl for overlap exclusion")
    parser.add_argument("--bbh-size", type=int, default=100)
    parser.add_argument("--lcb-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--bbh-revision", default=BBH_REVISION)
    parser.add_argument("--lcb-revision", default=LCB_REVISION)
    parser.add_argument("--lcb-version", default="release_v6")
    parser.add_argument("--lcb-start-date", type=date.fromisoformat)
    parser.add_argument("--lcb-end-date", type=date.fromisoformat)
    parser.add_argument("--bbh-dir", help="Optional existing directory with the four upstream JSON files")
    parser.add_argument("--lcb-jsonl", nargs="+", help="Optional existing upstream JSONL files; no remote data reads")
    parser.add_argument("--estimate-only", action="store_true", help="Read metadata and report free-space budget; no dataset bodies")
    parser.add_argument("--resume", action="store_true", help="Reuse verified completed source files")
    args = parser.parse_args(argv)
    if args.bbh_size < len(BBH_TASKS) or args.lcb_size < 1:
        parser.error("Need BBH >= 4 and LiveCodeBench >= 1")
    if bool(args.bbh_dir) != bool(args.lcb_jsonl):
        parser.error("Local/offline input requires both --bbh-dir and --lcb-jsonl")
    if args.lcb_start_date and args.lcb_end_date and args.lcb_start_date > args.lcb_end_date:
        parser.error("Start date is after end date")
    if args.resume and args.estimate_only:
        parser.error("--resume and --estimate-only cannot be combined")
    lcb_files(args.lcb_version)
    return args


@contextmanager
def preparation_lock(out):
    out.parent.mkdir(parents=True, exist_ok=True)
    with (out.parent / f".{out.name}.prepare.lock").open("a+b") as lock:
        try:
            if os.name == "posix":
                import fcntl
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            else:
                import msvcrt
                if lock.seek(0, 2) == 0:
                    lock.write(b"\0")
                    lock.flush()
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise RuntimeError(f"Preparation is already running for {out}") from exc
        try:
            yield
        finally:
            if os.name != "posix":
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


def prepare(args):
    configure_runtime()
    out = artifact_path(args.out)
    with preparation_lock(out):
        return prepare_locked(args)


def prepare_locked(args):
    out, training_path = artifact_path(args.out), artifact_path(args.train_jsonl)
    if training_path.is_relative_to(out):
        raise ValueError("Output directory must not contain the training input")
    settings = {k: str(v) if isinstance(v, date) else v for k, v in vars(args).items()
                if k not in {"out", "resume", "estimate_only"}}
    settings["train_jsonl"] = str(training_path)
    training_hash = file_hash(training_path)
    training = load_training(training_path)
    if args.resume:
        plan = read_json(out / "plan.json")
        if plan["settings"] != settings or plan["training_sha256"] != training_hash:
            raise ValueError("Preparation configuration/training input changed; use a new output directory")
        if plan["code_sha256"] != code_fingerprint():
            raise ValueError("Code changed; resume with this output's code/prepare_transfer_data.py snapshot")
        if (out / "manifest.json").exists():
            validate_prepared(out)
            print(f"Already complete and verified: {out}", flush=True)
            return out
        sources = plan["sources"]
        for source in sources:
            if "path" in source and file_hash(source["path"]) != source["sha256"]:
                raise ValueError(f"Local source changed: {source['path']}")
    else:
        if out.exists():
            raise FileExistsError(f"Output exists: {out}; use --resume or a new directory")
        sources = source_catalog(args)
    source_bytes = sum(s["size"] for s in sources)
    # Selected rows, reusable per-file candidates, and atomic-write headroom.
    budget = {"source_bytes_to_scan": source_bytes,
              "preparation_budget_bytes": 3 * source_bytes + 2 * GIB,
              "note": "Conservative budget; no full dataset cache, model, or GPU required. "
                      "Future execution-grader temporary files are not included."}
    disk = require_free_space(out, budget["preparation_budget_bytes"])
    print(json.dumps({**budget, **disk}, indent=2), flush=True)
    if args.estimate_only:
        return None
    if not args.resume:
        out.mkdir(parents=True, exist_ok=False)
        fingerprint = snapshot_code(out / "code")
        plan = {"format_version": "transfer-preparation-v1", "created_at": now(),
                "settings": settings, "training_sha256": training_hash, "sources": sources,
                "code_sha256": fingerprint, "storage": budget, "prompt_version": PROMPT_VERSION}
        write_json(out / "plan.json", plan)
    overlap = OverlapIndex(training)
    allocation = quotas(args.bbh_size)
    bbh, lcb, stage_stats, all_ids = [], [], {}, set()
    for number, source in enumerate(sources):
        stage = out / "stages" / f"{number:02d}-{source['name']}"
        marker = stage / "complete.json"
        count = allocation[source["name"]] if source["kind"] == "bbh" else args.lcb_size
        if marker.exists():
            info = read_json(marker)
            if info["source"] != source or info["plan_sha256"] != digest(plan):
                raise ValueError(f"Stage identity mismatch: {stage}")
            for name, expected in info["files"].items():
                if file_hash(stage / name) != expected:
                    raise ValueError(f"Saved preparation stage changed: {stage / name}")
            candidates = list(iter_jsonl(stage / "candidates.jsonl"))
            print(f"Reusing complete source: {source['name']}", flush=True)
        else:
            print(f"Scanning {source['kind']}/{source['name']} ({source['size'] / 1024**2:.1f} MiB)", flush=True)
            last = [time.monotonic()]

            def progress(n):
                if time.monotonic() - last[0] >= 15:
                    print(f"  {source['name']}: scanned {n} rows", flush=True)
                    last[0] = time.monotonic()

            candidates, stats, rejected, identities = select_stream(
                source_rows(source), count, args.seed, overlap,
                args.lcb_start_date, args.lcb_end_date, progress)
            if source["kind"] == "bbh" and len(candidates) < count:
                raise ValueError(f"Too few eligible questions for {source['name']}: {len(candidates)} < {count}")
            stage.mkdir(parents=True, exist_ok=True)
            write_jsonl(stage / "candidates.jsonl", candidates)
            write_jsonl(stage / "overlap_rejections.jsonl", rejected)
            info = {"source": source, "plan_sha256": digest(plan), "stats": stats,
                    "sample_ids": list(identities),
                    "files": {name: file_hash(stage / name) for name in ("candidates.jsonl", "overlap_rejections.jsonl")}}
            write_json(marker, info)
            print(f"Finished {source['name']}: {stats}; retained {len(candidates)} candidates", flush=True)
        # Repeated IDs across LCB releases would bias sampling; require inspection.
        duplicate = all_ids.intersection(info["sample_ids"])
        if duplicate:
            raise ValueError(f"Duplicate IDs across source files: {sorted(duplicate)[:3]}")
        all_ids.update(info["sample_ids"])
        stage_stats[source["name"]] = info["stats"]
        (bbh if source["kind"] == "bbh" else lcb).extend(candidates)
    lcb = sorted(lcb, key=lambda row: rank(args.seed, row))[:args.lcb_size]
    if len(lcb) < args.lcb_size:
        raise ValueError(f"Only {len(lcb)} eligible LiveCodeBench questions; requested {args.lcb_size}")
    selected = bbh + lcb
    if len({r["panel"]["question_id"] for r in selected}) != len(selected):
        raise ValueError("Duplicate selected question text across sources; inspect before changing the panel")
    if file_hash(training_path) != training_hash:
        raise ValueError("Training input changed while preparing the panel")
    panels, selections = [], []
    for row in selected:
        panel = dict(row["panel"])
        rel = f"references/{panel['sample_id']}.json"
        write_json(out / rel, row["original"])
        panel["reference_file"], panel["reference_sha256"] = rel, file_hash(out / rel)
        panels.append(panel)
        selections.append({k: panel.get(k) for k in ("sample_id", "question_id", "task", "subtask",
                                                    "platform", "difficulty", "contest_date", "source_location")})
    write_jsonl(out / "panel.jsonl", panels)
    write_jsonl(out / "selection.jsonl", selections)
    write_jsonl(out / "inspection.jsonl", [{"sample_id": p["sample_id"], "task": p["task"],
                                            "messages": p["messages"], "reference_file": p["reference_file"]}
                                           for p in panels if p["sample_id"] in {bbh[0]["panel"]["sample_id"], lcb[0]["panel"]["sample_id"]}])
    summary = {"total": len(panels), "bbh": len(bbh), "livecodebench": len(lcb),
               "bbh_subtasks": allocation,
               "lcb_difficulty": dict(Counter(p["difficulty"] for p in panels if p["task"] == "livecodebench")),
               "lcb_platform": dict(Counter(p["platform"] for p in panels if p["task"] == "livecodebench")),
               "stages": stage_stats, "training_overlap_check": "NFKC/whitespace exact and char7 Jaccard >= 0.9",
               "semantic_or_pretraining_overlap_ruled_out": False,
               "prompt_tokens_checked": False, "hidden_tests_decoded_or_executed": False,
               "generation_and_grading_status": "prepared_only; existing MATH500/GPQA pipeline is unchanged"}
    write_json(out / "summary.json", summary)
    files = {p.relative_to(out).as_posix(): file_hash(p) for p in out.rglob("*")
             if p.is_file() and not p.is_relative_to(out / "code") and ".tmp-" not in p.name}
    write_json(out / "manifest.json", {"format_version": "transfer-panel-v1", "created_at": now(),
                                        "files": files, "code_sha256": plan["code_sha256"],
                                        "sources": sources, "settings": settings, "summary": summary,
                                        "selected_and_stage_bytes": sum((out / name).stat().st_size for name in files)})
    print(f"Prepared {len(panels)} evaluation questions at {out}. No inference or code execution was run.", flush=True)
    return out


def main():
    prepare(parse_args())


if __name__ == "__main__":
    main()
