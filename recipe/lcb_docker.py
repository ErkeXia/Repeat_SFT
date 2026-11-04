"""Run untrusted candidate solutions in a resource-limited, offline Docker container."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import uuid

from .common import ROOT, file_hash
from .lcb_worker import PROTOCOL

DEFAULT_IMAGE = "repeat-sft-lcb:py311-v1"


def validate_config(config):
    keys = {"case_timeout_seconds", "problem_timeout_seconds", "memory_mb", "candidate_memory_mb", "cpus",
            "pids_limit", "tmpfs_mb", "max_output_bytes", "max_decoded_test_bytes"}
    if set(config) != keys or any(type(config[k]) not in (int, float) or not math.isfinite(config[k]) or config[k] <= 0 for k in keys):
        raise ValueError("Invalid/missing LiveCodeBench resource configuration")
    if any(type(config[k]) is not int for k in keys - {"case_timeout_seconds", "problem_timeout_seconds", "cpus"}):
        raise ValueError("Memory, process, and byte limits must be integers")
    if config["candidate_memory_mb"] >= config["memory_mb"] or config["case_timeout_seconds"] >= config["problem_timeout_seconds"]:
        raise ValueError("Candidate limits must leave room for the supervisor")
    return config


def inspect_image(image):
    result = subprocess.run(["docker", "image", "inspect", image], capture_output=True, text=True, check=True)
    spec = json.loads(result.stdout)[0]
    labels = spec["Config"].get("Labels") or {}
    if (labels.get("repeat_sft.protocol") != PROTOCOL
            or labels.get("repeat_sft.worker_sha256") != file_hash(ROOT / "recipe" / "lcb_worker.py")):
        raise ValueError("LCB image does not match this frozen worker. Build it with build_lcb_sandbox.py")
    if spec.get("Os") != "linux":
        raise ValueError("The execution image must use Linux")
    return {"id": spec["Id"], "labels": labels, "architecture": spec.get("Architecture"), "os": spec["Os"]}


def docker_command(image_id, name, config):
    return ["docker", "run", "--rm", "--pull=never", "--name", name, "--log-driver=none", "-i",
            "--network=none", "--read-only", "--cap-drop=ALL", "--cap-add=SETUID", "--cap-add=SETGID", "--cap-add=KILL",
            "--security-opt=no-new-privileges", "--user=0:0", "--pids-limit", str(config["pids_limit"]),
            "--memory", f"{config['memory_mb']}m", "--memory-swap", f"{config['memory_mb']}m",
            "--cpus", str(config["cpus"]), "--tmpfs", f"/tmp:rw,nosuid,nodev,size={config['tmpfs_mb']}m,mode=1777", image_id]


class DockerJudge:
    def __init__(self, image, config):
        self.config = validate_config(config)
        self.image = inspect_image(image)

    def run(self, code, tests):
        name = "repeat-sft-lcb-" + uuid.uuid4().hex
        payload = {"code": code, **tests, "config": self.config}
        command = docker_command(self.image["id"], name, self.config)
        options = {}
        if os.name == "posix" and os.environ.get("REPEAT_SFT_PIPELINE_LOCK_FD"):
            options["pass_fds"] = (int(os.environ["REPEAT_SFT_PIPELINE_LOCK_FD"]),)
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **options)
        try:
            output, errors = process.communicate(json.dumps(payload).encode(), timeout=self.config["problem_timeout_seconds"] + 30)
            if process.returncode == 124:
                return {"protocol": PROTOCOL, "correct": False, "status": "problem_timeout",
                        "total_tests": len(tests["cases"]), "tested": None, "passed": None, "test_results": []}
            if process.returncode:
                # Docker/worker/OOM failures are not quietly counted as wrong answers.
                raise RuntimeError(f"LCB container failed ({process.returncode}); saved answers retained: {errors.decode(errors='replace')[-4000:]}")
            result = json.loads(output)
            if result.get("protocol") != PROTOCOL or type(result.get("correct")) is not bool:
                raise ValueError("Invalid container grading response")
            if result["total_tests"] != len(tests["cases"]):
                raise ValueError("Container did not receive the full public + hidden test set")
            return result
        finally:
            # No host input/output/cache directories are mounted into this container.
            subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=20)
            if process.poll() is None:
                process.kill()
            process.communicate()

    def preflight(self):
        config = self.config
        cases = [{"split": "public", "index": 0, "input": "2\n3\n", "output": "5\n"}]
        for code, fn in (("import os\nassert os.geteuid()==65534\na=int(input()); b=int(input()); print(a+b)", None),
                         ("class Solution:\n    def add(self, a, b): return a+b", "add")):
            result = self.run(code, {"fn_name": fn, "cases": cases})
            if not result["correct"]:
                raise RuntimeError(f"Container self-test failed: {result}")
        timeout = self.run("while True: pass", {"fn_name": None, "cases": cases})
        if timeout["correct"] or timeout["status"] != "timeout":
            raise RuntimeError(f"Container timeout/child cleanup self-test failed: {timeout}")
        return {"image": self.image, "config": config, "stdin_and_function_self_tests": "passed",
                "timeout_and_unprivileged_execution_self_tests": "passed"}
