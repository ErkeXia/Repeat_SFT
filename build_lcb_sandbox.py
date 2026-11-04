"""VM ONLY: build the small Python execution image; no model or dataset downloads."""
import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

from recipe.common import ROOT, file_hash, now, write_json
from recipe.lcb_docker import DEFAULT_IMAGE, DockerJudge
from recipe.lcb_worker import PROTOCOL
from recipe.common import read_json
from recipe.paths import configure_runtime


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", default=DEFAULT_IMAGE)
    p.add_argument("--base-image", default="python:3.11-slim-bookworm")
    args = p.parse_args()
    if any(c.isspace() for c in args.base_image + args.tag):
        p.error("Image references may not contain whitespace")
    root = configure_runtime()
    worker = ROOT / "recipe" / "lcb_worker.py"
    build_root = root / ".cache" / "lcb-build"
    build_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="build-", dir=build_root) as temp:
        temp = Path(temp)
        shutil.copy2(worker, temp / "worker.py")
        dockerfile = (f"FROM {args.base_image}\n"
                      "RUN python -m pip install --no-cache-dir numpy==2.2.6 sortedcontainers==2.4.0\n"
                      "COPY worker.py /opt/worker.py\n"
                      f'LABEL repeat_sft.protocol="{PROTOCOL}" repeat_sft.worker_sha256="{file_hash(worker)}"\n'
                      f"ENV REPEAT_SFT_LCB_CONTAINER={PROTOCOL}\n"
                      'ENTRYPOINT ["python", "-I", "-B", "/opt/worker.py"]\n')
        (temp / "Dockerfile").write_text(dockerfile, encoding="utf-8")
        subprocess.run(["docker", "build", "--tag", args.tag, str(temp)], check=True)
    judge = DockerJudge(args.tag, read_json(ROOT / "configs" / "lcb_grading.json"))
    result = judge.preflight()
    record = {"created_at": now(), "base_image_requested": args.base_image, "dockerfile": dockerfile, **result}
    out = root / "artifacts" / "lcb-sandbox" / (judge.image["id"].replace(":", "-") + ".json")
    write_json(out, record)
    print(f"Execution image ready: {judge.image['id']}\nBuild/self-test record: {out}")


if __name__ == "__main__":
    main()
