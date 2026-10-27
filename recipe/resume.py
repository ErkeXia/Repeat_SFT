"""Verify and locate original source snapshots for trusted existing runs."""
from pathlib import Path

from .common import digest, file_hash, read_json


def verified_code(code, expected):
    code = Path(code).resolve()
    manifest = read_json(code / "manifest.json")
    if manifest["code_sha256"] != expected or digest(manifest["files"]) != expected:
        raise ValueError(f"Code manifest mismatch: {code}")
    for name, checksum in manifest["files"].items():
        file = code / name
        if not file.resolve().is_relative_to(code) or not file.is_file() or file_hash(file) != checksum:
            raise ValueError(f"Source snapshot changed: {file}")
    if not (code / "train.py").is_file():
        raise ValueError(f"Missing training entrypoint: {code}")
    return code


def original_code(run):
    run = Path(run)
    expected = read_json(run / "run.json")["code_sha256"]
    for manifest_path in sorted((run / "attempts").glob("*/code/manifest.json")):
        try:
            return verified_code(manifest_path.parent, expected)
        except (ValueError, KeyError, FileNotFoundError):
            continue
    raise ValueError(f"No intact original code snapshot for {run}. Code checks were not bypassed.")


def verify_run_inputs(run):
    run = Path(run)
    metadata = read_json(run / "run.json")
    for directory, field in (("data", "data_files"), ("tokenizer", "tokenizer_files")):
        for name, expected in metadata[field].items():
            path = run / directory / name
            if not path.resolve().is_relative_to((run / directory).resolve()) or file_hash(path) != expected:
                raise ValueError(f"Frozen run input changed: {path}")
    return metadata
