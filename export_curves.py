"""Export curve tables and early/late rankings from completed saved results."""
import argparse
import csv
from pathlib import Path
import statistics

from recipe.common import digest, file_hash, fresh_dir, iter_jsonl, read_json, write_json
from smoke_test import canonical_logs


def csv_write(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", action="append", nargs=3, metavar=("NAME", "RUN_DIR", "GRADE_DIR"), required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    out = fresh_dir(args.out)
    curves, training, provenance, windows = [], [], [], []
    panel_hashes = set()
    comparison_settings = set()
    for name, run_name, grade_name in args.run:
        run, grades = Path(run_name), Path(grade_name)
        metadata = read_json(run / "run.json")
        grading_manifest = read_json(grades / "grading_manifest.json")
        for source in grading_manifest["sources"]:
            identity = source["identity"]
            if identity["run_sha256"] != file_hash(run / "run.json"):
                raise ValueError("Grade directory belongs to a different training run")
            comparison_settings.add(digest({"generation": identity["settings"],
                                            "selected_ids": identity["selected_ids"],
                                            "grader": grading_manifest["grader_version"],
                                            "grader_code": grading_manifest["code_sha256"],
                                            "model": metadata["model"],
                                            "training": {k: v for k, v in metadata["config"].items() if k != "seed"}}))
        panel_hashes.add(file_hash(run / "data" / "panel.jsonl"))
        grouped = {}
        for row in iter_jsonl(grades / "summary.jsonl"):
            if not row["complete_panel"] or row["grading_errors"]:
                raise ValueError("Only complete, error-free grading may enter curve comparisons")
            curve = {"group": name, "training_seed": metadata["config"]["seed"], **row}
            for split in ("train", "holdout"):
                loss_file = run / "evaluations" / "teacher_forcing" / f"epoch-{row['epoch']:03d}" / split / "summary.json"
                if loss_file.exists():
                    curve.update({f"{split}_{k}": v for k, v in read_json(loss_file).items()})
            curves.append(curve)
            grouped.setdefault(row["task"], {})[row["epoch"]] = row["accuracy"]
        for task, values in grouped.items():
            for window, epochs in (("early", [1, 2, 4]), ("late", [32, 64])):
                if all(e in values for e in epochs):
                    windows.append({"group": name, "task": task, "window": window,
                                    "mean_checkpoint_accuracy": statistics.mean(values[e] for e in epochs),
                                    "epochs": ",".join(map(str, epochs))})
        training.extend({"group": name, **r} for r in canonical_logs(run).values())
        provenance.append({"group": name, "run": str(run.resolve()), "grades": str(grades.resolve()),
                           "run_sha256": file_hash(run / "run.json"),
                           "grade_manifest_sha256": file_hash(grades / "grading_manifest.json")})
    if len(panel_hashes) != 1:
        raise ValueError("Runs use different evaluation panels; do not rank them as matched experiments")
    if len(comparison_settings) != 1:
        raise ValueError("Runs differ in initial model, training settings, generation, questions or grading code")
    for task in {x["task"] for x in windows}:
        for window in ("early", "late"):
            members = sorted([x for x in windows if x["task"] == task and x["window"] == window],
                             key=lambda x: -x["mean_checkpoint_accuracy"])
            for row in members:
                row["rank"] = 1 + sum(x["mean_checkpoint_accuracy"] > row["mean_checkpoint_accuracy"] for x in members)
    csv_write(out / "curves.csv", curves)
    csv_write(out / "training_steps.csv", training)
    csv_write(out / "early_late_rankings.csv", windows)
    write_json(out / "provenance.json", {"inputs": provenance,
                                         "note": "Descriptive rankings only; confirm crossings with new data and training seeds. "
                                                 "Conditional end-rate accuracy is not a causal decomposition."})


if __name__ == "__main__":
    main()
