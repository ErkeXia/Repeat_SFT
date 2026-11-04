"""Container-only test supervisor; candidate subprocesses never receive gold outputs."""
from __future__ import annotations

import ast
import contextlib
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

PROTOCOL = "repeat-sft-lcb-container-v1"
MODULES = ("string", "re", "datetime", "collections", "heapq", "bisect", "copy", "math", "random",
           "statistics", "itertools", "functools", "operator", "io", "sys", "json", "typing")


def stdio_equal(predicted, expected):
    left = [s.strip() for s in predicted.strip().split("\n")]
    right = [s.strip() for s in expected.strip().split("\n")]
    if len(left) != len(right):
        return False
    for a, b in zip(left, right):
        if a == b:
            continue
        try:
            x, y = [Decimal(v) for v in a.split()], [Decimal(v) for v in b.split()]
            if x != y or any(not v.is_finite() for v in x + y):
                return False
        except InvalidOperation:
            return False
    return True


def candidate(path, fn_name):
    """Only the generated code and one case's inputs enter this unprivileged process."""
    code = Path(path).read_text(encoding="utf-8")
    namespace = {"__name__": "__main__"}
    prelude = "\n".join([f"from {m} import *" for m in MODULES] + [f"import {m}" for m in MODULES])
    sys.setrecursionlimit(50000)
    if hasattr(sys, "set_int_max_str_digits"):
        sys.set_int_max_str_digits(50000)
    if fn_name:
        inputs = [json.loads(line) for line in sys.stdin.read().split("\n")]
        with contextlib.redirect_stdout(sys.stderr):
            exec(prelude, namespace)
            exec(compile(code, "solution.py", "exec"), namespace)
            owner = namespace["Solution"]() if "Solution" in namespace else None
            function = getattr(owner, fn_name) if owner is not None else namespace[fn_name]
            result = function(*inputs)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    else:
        exec(prelude, namespace)
        exec(compile(code, "solution.py", "exec"), namespace)


def demote(config):
    import resource
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (config["max_output_bytes"], config["max_output_bytes"]))
    memory = config["candidate_memory_mb"] * 1024**2
    resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
    cpu = math.ceil(config["case_timeout_seconds"]) + 1
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
    os.setgroups([])
    os.setgid(65534)
    os.setuid(65534)


def run_case(code_path, case, fn_name, config):
    start = time.monotonic()
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        process = subprocess.Popen([sys.executable, "-I", "-B", __file__, "--candidate", str(code_path), fn_name or ""],
                   stdin=subprocess.PIPE, stdout=stdout, stderr=stderr, cwd="/tmp",
                   env={"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/tmp", "LANG": "C.UTF-8",
                        "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"},
                   start_new_session=True, preexec_fn=lambda: demote(config))
        status = "finished"
        try:
            process.communicate(case["input"].encode(), timeout=config["case_timeout_seconds"])
        except subprocess.TimeoutExpired:
            status = "timeout"
        finally:
            # Remove descendants too, including children that outlive a successful parent.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        sizes = [os.fstat(f.fileno()).st_size for f in (stdout, stderr)]
        stdout.seek(0)
        stderr.seek(0)
        output = stdout.read(config["max_output_bytes"]).decode("utf-8", errors="replace")
        error = stderr.read(config["max_output_bytes"]).decode("utf-8", errors="replace")
    if status == "finished":
        if max(sizes) >= config["max_output_bytes"]:
            status = "output_limit"
        elif process.returncode:
            status = "runtime_error"
        else:
            try:
                correct = json.loads(output) == json.loads(case["output"]) if fn_name else stdio_equal(output, case["output"])
                status = "passed" if correct else "wrong_answer"
            except (ValueError, TypeError):
                status = "invalid_function_output"
    return {"status": status, "seconds": time.monotonic() - start, "exit_code": process.returncode,
            "actual_output": output, "stderr": error}


def supervise(payload, case_runner=run_case, work_root="/tmp"):
    config, cases, code = payload["config"], payload["cases"], payload["code"]
    if not cases:
        raise ValueError("No test cases")
    try:
        ast.parse(code)
    except SyntaxError as exc:
        return {"protocol": PROTOCOL, "correct": False, "status": "compile_error", "error": str(exc),
                "total_tests": len(cases), "tested": 0, "passed": 0, "test_results": []}
    # This directory is root-owned; candidate may read code but cannot modify it.
    directory = Path(tempfile.mkdtemp(prefix="solution-", dir=work_root))
    directory.chmod(0o755)
    path = directory / "solution.py"
    path.write_text(code, encoding="utf-8")
    path.chmod(0o444)
    results = []
    for case in cases:
        result = case_runner(path, case, payload["fn_name"], config)
        results.append({**result, "split": case["split"], "index": case["index"],
                        "input_sha256": hashlib.sha256(case["input"].encode()).hexdigest(),
                        "expected_sha256": hashlib.sha256(case["output"].encode()).hexdigest()})
        if result["status"] != "passed":
            break
    correct = len(results) == len(cases) and all(r["status"] == "passed" for r in results)
    return {"protocol": PROTOCOL, "correct": correct, "status": "passed" if correct else results[-1]["status"],
            "total_tests": len(cases), "tested": len(results), "passed": sum(r["status"] == "passed" for r in results),
            "test_results": results}


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--candidate":
        candidate(sys.argv[2], sys.argv[3])
        return
    if os.environ.get("REPEAT_SFT_LCB_CONTAINER") != PROTOCOL or os.geteuid() != 0:
        raise RuntimeError("Run this supervisor only through the project's Docker judge")
    payload = json.load(sys.stdin)
    # A host kill -9 can orphan Docker; the container still has its own lifetime bound.
    signal.signal(signal.SIGALRM, lambda *_: os._exit(124))
    signal.alarm(math.ceil(payload["config"]["problem_timeout_seconds"]))
    result = supervise(payload)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
