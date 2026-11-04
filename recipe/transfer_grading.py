"""Explicit answer extraction and bounded, non-executable hidden-test decoding."""
from __future__ import annotations

import ast
import base64
import io
import json
import pickle
import re
import zlib

from .grading import last_boxed

GRADER_VERSION = "transfer-bbh-python-v1"


def final_text(text):
    if "<think>" in text and "</think>" not in text:
        return None
    text = text.rsplit("</think>", 1)[-1]
    return re.split(r"<\|im_end\|>|<\|endoftext\|>", text)[0].strip()


def extract_bbh(text, subtask):
    final = final_text(text)
    if final is None:
        return None, "unfinished_think"
    markers = list(re.finditer(r"(?:final\s+answer|answer)\s*\**\s*(?:is|:)\s*([^\n]+)", final, re.I))
    boxed = last_boxed(final)
    if markers and (boxed is None or markers[-1].start() > max(final.rfind("\\boxed"), final.rfind("\\fbox"))):
        value, method = markers[-1][1], "final_marker"
    elif boxed is not None:
        value, method = boxed, "boxed"
    else:
        value, method = final, "bare_answer"
    value = re.sub(r"\\(?:text|mathrm)\{([^{}]+)\}", r"\1", value)
    value = value.strip(" *`$().:{}")
    allowed = ({"true": "True", "false": "False"} if subtask == "boolean_expressions" else
               {"yes": "Yes", "no": "No"} if subtask == "web_of_lies" else
               {c.lower(): c for c in "ABCDE"})
    answer = allowed.get(value.lower())
    return answer, method if answer is not None else "no_unambiguous_final_answer"


def extract_code(text):
    final = final_text(text)
    if final is None:
        return None, "unfinished_think"
    blocks = list(re.finditer(r"^\s*```([^\n`]*)\n(.*?)^\s*```\s*$", final, re.M | re.S))
    eligible = [m[2].strip() for m in blocks if m[1].strip().lower() in ("", "python", "python3", "py")]
    if eligible:
        return eligible[-1] or None, "last_complete_python_fence"
    if "```" in final:
        return None, "no_complete_python_fence"
    try:
        tree = ast.parse(final)
        if not tree.body:
            return None, "empty_code"
    except SyntaxError:
        return None, "no_python_code"
    return final, "bare_python"


class DataOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise ValueError("Hidden tests contain executable pickle globals")

    def persistent_load(self, pid):
        raise ValueError("Hidden tests contain pickle persistent references")


def decode_private(value, max_bytes):
    if isinstance(value, list):
        return value
    if not isinstance(value, str):
        raise ValueError("Unexpected private test encoding")
    if len(value.encode()) > max_bytes * 2:
        raise ValueError("Encoded private tests exceed configured limit")
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        compressed = base64.b64decode(value, validate=True)
        decoder = zlib.decompressobj()
        raw = decoder.decompress(compressed, max_bytes + 1)
        if len(raw) > max_bytes or not decoder.eof or decoder.unused_data:
            raise ValueError("Private tests exceed decode limit or have invalid compressed framing")
        decoded = DataOnlyUnpickler(io.BytesIO(raw)).load()
        # The upstream encoding pickles a JSON string, never a Python callable.
        if not isinstance(decoded, (str, bytes)) or len(decoded) > max_bytes:
            raise ValueError("Expected a bounded pickled JSON string")
        return json.loads(decoded)


def test_payload(reference, row, config):
    limit = config["max_decoded_test_bytes"]
    public = reference["public_test_cases"]
    public = json.loads(public) if isinstance(public, str) else public
    private = decode_private(reference["private_test_cases"], limit)
    if not isinstance(public, list) or not isinstance(private, list) or not private:
        raise ValueError("Missing/invalid public or hidden tests; never grade on public tests alone")
    metadata = reference["metadata"]
    metadata = json.loads(metadata) if isinstance(metadata, str) else metadata
    fn = metadata.get("func_name")
    expected_type = "functional" if fn else "stdin"
    if row["interface"] != expected_type:
        raise ValueError("Function interface and metadata.func_name disagree")
    cases = []
    for split, tests in (("public", public), ("private", private)):
        for index, test in enumerate(tests):
            if (test.get("testtype") != expected_type or not isinstance(test.get("input"), str)
                    or not isinstance(test.get("output"), str)):
                raise ValueError("Invalid LiveCodeBench test case")
            if fn:
                [json.loads(line) for line in test["input"].split("\n")]
                json.loads(test["output"])
            cases.append({"split": split, "index": index, "input": test["input"], "output": test["output"]})
    if len(json.dumps(cases).encode()) > limit:
        raise ValueError("Decoded tests exceed configured size; increase the limit in a NEW grading job")
    return {"fn_name": fn, "cases": cases}


