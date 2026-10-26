"""Versioned final-answer extraction, independent of GPU generation."""
from __future__ import annotations

import re

GRADER_VERSION = "boxed-final-v1"


def last_boxed(text):
    matches = list(re.finditer(r"\\(?:boxed|fbox)\s*\{", text))
    for match in reversed(matches):
        start, level = match.end(), 1
        for i in range(start, len(text)):
            if text[i] == "{" and (i == 0 or text[i - 1] != "\\"):
                level += 1
            elif text[i] == "}" and (i == 0 or text[i - 1] != "\\"):
                level -= 1
                if level == 0:
                    return text[start:i].strip()
    return None


def extract_answer(text, task):
    # Do not count an intermediate boxed guess inside an unfinished thought.
    if "<think>" in text and "</think>" not in text:
        return None, "unfinished_think"
    final = text.rsplit("</think>", 1)[-1]
    for token in ("<|im_end|>", "<|endoftext|>"):
        final = final.split(token)[0]
    boxed = last_boxed(final)
    if task == "math500" or task == "math":
        if boxed is not None:
            return boxed, "boxed"
        matches = re.findall(r"(?:final\s+answer|answer)\s*(?:is|:)\s*([^\n]+)", final, re.I)
        if matches:
            return matches[-1].strip().rstrip(".").strip("$ "), "answer_marker"
        return None, "no_final_answer"
    if task != "gpqa":
        raise ValueError(task)
    if boxed:
        cleaned = re.sub(r"\\(?:text|mathrm)\{([^{}]+)\}", r"\1", boxed).strip("() $.")
        if re.fullmatch("[A-D]", cleaned):
            return cleaned, "boxed_letter"
    matches = re.findall(r"(?:final\s+answer|answer)\s*(?:is|:)\s*\(?([A-D])\)?(?:\s|[.,:]|$)", final, re.I)
    if matches:
        return matches[-1].upper(), "answer_marker"
    bare = final.strip().strip("(). $")
    if re.fullmatch("[A-D]", bare):
        return bare, "bare_letter"
    return None, "no_final_answer"


def math_equal(gold, predicted):
    from math_verify import LatexExtractionConfig, parse, verify
    # Both are final-answer strings, never search the reasoning for a gold match.
    cfg = [LatexExtractionConfig()]
    gold_parsed = parse("$" + str(gold).strip("$") + "$", extraction_config=cfg,
                        fallback_mode="no_fallback", parsing_timeout=5)
    if not gold_parsed:
        raise ValueError(f"Unparseable reference answer: {gold!r}")
    pred_parsed = parse("$" + str(predicted).strip("$") + "$", extraction_config=cfg,
                        fallback_mode="no_fallback", parsing_timeout=5)
    return bool(pred_parsed and verify(gold_parsed, pred_parsed, timeout_seconds=5))


def grade_one(text, gold, task):
    extracted, method = extract_answer(text, task)
    result = {"extracted_answer": extracted, "extraction_method": method,
              "extractable": extracted is not None, "correct": False, "grading_error": None}
    if extracted is not None:
        try:
            result["correct"] = extracted == gold if task == "gpqa" else math_equal(gold, extracted)
        except Exception as exc:
            result["grading_error"] = f"{type(exc).__name__}: {exc}"
            result["correct"] = None
    return result
