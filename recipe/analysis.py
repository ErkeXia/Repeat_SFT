"""Descriptive checkpoint trajectories; no model loading or statistical significance claims."""
from __future__ import annotations

import math
import statistics


def quantile(values, q):
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * q
    left, right = math.floor(position), math.ceil(position)
    return values[left] + (values[right] - values[left]) * (position - left)


def tail_stable_epoch(epochs, values, tolerance, min_points):
    """Earliest observed suffix whose TOTAL range is small; never extrapolate past it.

    Missing checkpoints invalidate a suffix. Requiring multiple future observations
    prevents the final checkpoint (or many tiny but cumulative changes) from being
    declared a plateau automatically. This is a configurable descriptive heuristic.
    """
    for i, epoch in enumerate(epochs):
        tail = values[i:]
        if len(tail) >= min_points and all(v is not None for v in tail) and max(tail) - min(tail) <= tolerance:
            return epoch
    return None


def sustained_epoch(epochs, states, min_points):
    for i, epoch in enumerate(epochs):
        if len(states[i:]) >= min_points and all(s is True for s in states[i:]):
            return epoch
    return None


def fit_milestones(epochs, observations, loss_tolerance=.01, accuracy_tolerance=.005,
                   min_points=3, accuracy_target=.99):
    losses = [r["mean_loss"] if r else None for r in observations]
    accuracies = [r["token_accuracy"] if r else None for r in observations]
    joint = None
    for i, epoch in enumerate(epochs):
        if (len(epochs[i:]) >= min_points and all(r is not None for r in observations[i:])
                and max(losses[i:]) - min(losses[i:]) <= loss_tolerance
                and max(accuracies[i:]) - min(accuracies[i:]) <= accuracy_tolerance):
            joint = epoch
            break
    high_accuracy = [v >= accuracy_target if v is not None else None for v in accuracies]
    return {"loss_tail_stable_epoch": tail_stable_epoch(epochs, losses, loss_tolerance, min_points),
            "token_accuracy_tail_stable_epoch": tail_stable_epoch(epochs, accuracies, accuracy_tolerance, min_points),
            "joint_tail_stable_epoch": joint,
            "first_accuracy_target_epoch": next((e for e, ok in zip(epochs, high_accuracy) if ok is True), None),
            "sustained_accuracy_target_epoch": sustained_epoch(epochs, high_accuracy, min_points),
            "observed_checkpoints": sum(r is not None for r in observations),
            "complete_trajectory": all(r is not None for r in observations),
            "last_mean_loss": losses[-1], "last_token_accuracy": accuracies[-1]}


def correctness_milestones(epochs, observations, min_points=2):
    states = [r["correct"] if r and r.get("grading_error") is None else None for r in observations]
    first = next((i for i, state in enumerate(states) if state is True), None)
    first_epoch = epochs[first] if first is not None else None
    previous_wrong = next((epochs[i] for i in range(first - 1, -1, -1) if states[i] is False), None) if first is not None else None
    regressions = [(epochs[i - 1], epochs[i]) for i in range(1, len(epochs))
                   if states[i - 1] is True and states[i] is False]
    recovered = [(epochs[i - 1], epochs[i]) for i in range(1, len(epochs))
                 if states[i - 1] is False and states[i] is True]
    baseline = states[epochs.index(0)] if 0 in epochs else None
    complete = all(s is not None for s in states)
    return {"baseline_correct": baseline, "first_observed_correct_epoch": first_epoch,
            "first_correct_after_last_observed_wrong_epoch": previous_wrong,
            "first_correct_interval_right_epoch": first_epoch,
            "first_correct_is_baseline": first_epoch == 0,
            "sustained_correct_epoch": sustained_epoch(epochs, states, min_points),
            "last_observed_correct_epoch": next((e for e, s in reversed(list(zip(epochs, states))) if s is True), None),
            "correct_epochs": [e for e, s in zip(epochs, states) if s is True],
            "incorrect_epochs": [e for e, s in zip(epochs, states) if s is False],
            "unknown_epochs": [e for e, s in zip(epochs, states) if s is None],
            "regression_intervals": regressions, "improvement_intervals": recovered,
            "never_correct_on_evaluated_checkpoints": first is None if complete else None,
            "complete_trajectory": complete}


def transition(previous, current):
    if not previous or not current or previous.get("grading_error") or current.get("grading_error"):
        return "unknown"
    a, b = previous["correct"], current["correct"]
    if a is None or b is None:
        return "unknown"
    return {(False, False): "still_incorrect", (False, True): "gained",
            (True, False): "lost", (True, True): "still_correct"}[(a, b)]


def paired_changes(ids, before, after):
    groups = {k: [] for k in ("gained", "lost", "still_correct", "still_incorrect", "unknown")}
    gained_end, gained_extract, gained_stable_behavior = [], [], []
    for sid in ids:
        a, b = before.get(sid), after.get(sid)
        kind = transition(a, b)
        groups[kind].append(sid)
        if kind == "gained":
            if not a["normal_chat_end"] and b["normal_chat_end"]:
                gained_end.append(sid)
            if not a["extractable"] and b["extractable"]:
                gained_extract.append(sid)
            if a["normal_chat_end"] == b["normal_chat_end"] and a["extractable"] == b["extractable"]:
                gained_stable_behavior.append(sid)
    matched = len(ids) - len(groups["unknown"])
    return {"expected_n": len(ids), "matched_n": matched,
            "complete_pair": matched == len(ids),
            **{k + "_n": len(v) for k, v in groups.items()},
            "net_correct_change": len(groups["gained"]) - len(groups["lost"]),
            "accuracy_change": (len(groups["gained"]) - len(groups["lost"])) / len(ids) if matched == len(ids) else None,
            "gained_with_normal_end_improvement_n": len(gained_end),
            "gained_with_extraction_improvement_n": len(gained_extract),
            "gained_with_unchanged_end_and_extraction_n": len(gained_stable_behavior),
            "members": {**groups, "gained_with_normal_end_improvement": gained_end,
                        "gained_with_extraction_improvement": gained_extract,
                        "gained_with_unchanged_end_and_extraction": gained_stable_behavior}}


def fit_summary(rows, expected_n):
    n, tokens = len(rows), sum(r["n_tokens"] for r in rows)
    losses = [r["mean_loss"] for r in rows]
    accuracies = [r["token_accuracy"] for r in rows]
    return {"expected_n": expected_n, "observed_n": n, "complete": n == expected_n,
            "supervised_tokens": tokens,
            "sample_mean_loss": statistics.mean(losses) if n else None,
            "token_weighted_loss": sum(r["nll_sum"] for r in rows) / tokens if tokens else None,
            "sample_mean_token_accuracy": statistics.mean(accuracies) if n else None,
            "token_weighted_accuracy": sum(r["correct_tokens"] for r in rows) / tokens if tokens else None,
            "end_token_accuracy": sum(r["end_token_correct"] for r in rows) / n if n else None,
            **{f"{metric}_p{int(q * 100):02d}": quantile(values, q)
               for metric, values in (("loss", losses), ("token_accuracy", accuracies)) for q in (.1, .5, .9)}}


def task_milestones(epochs, curves, min_points=2):
    accuracy = {r["epoch"]: r["accuracy"] for r in curves if r["complete"]}
    baseline = accuracy.get(0)
    states = [accuracy[e] > baseline if baseline is not None and e in accuracy else None for e in epochs]
    best = max(accuracy.values()) if accuracy else None
    return {"baseline_accuracy": baseline,
            "first_observed_above_base_epoch": next((e for e, ok in zip(epochs, states) if ok is True), None),
            "sustained_above_base_epoch": sustained_epoch(epochs, states, min_points),
            "best_accuracy": best, "best_epochs": [e for e in epochs if best is not None and accuracy.get(e) == best],
            "final_accuracy": accuracy.get(epochs[-1]),
            "interpretation": "Descriptive observed improvement only; not proof of transfer onset or statistical significance"}
