"""Dataset schema adapters, question overlap checks, and matched selection."""
from __future__ import annotations

from collections import Counter, defaultdict
import random

from .common import digest, stable_seed
from .encoding import normalize_question, question_id

MATH_SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
GPQA_SUFFIX = "\nReason through the question, then put only the final option letter (A, B, C, or D) within \\boxed{}."


def adapt_sft(row, index):
    # 'answer' is the gold short answer. Never use it as the training response.
    question, response = row["question"], row["response"]
    messages = row.get("message") or [{"role": "user", "content": question + MATH_SUFFIX}]
    if [m["role"] for m in messages] not in (["user"], ["system", "user"]):
        raise ValueError(f"Unexpected training conversation at row {index}")
    if normalize_question(question) not in normalize_question(messages[-1]["content"]):
        raise ValueError(f"Question / prompt mismatch at row {index}")
    qid = question_id(question)
    return {"sample_id": digest([qid, response]), "question_id": qid, "row_index": index,
            "question": question, "response": response, "answer": row["answer"],
            "source": str(row.get("data_source", "unknown")), "messages": messages,
            "original_metadata": {k: v for k, v in row.items()
                                  if k not in {"question", "response", "answer", "message"}}}


def adapt_math(row, index):
    question = row["problem"]
    return {"sample_id": "math500-" + question_id(question), "question_id": question_id(question),
            "task": "math500", "question": question, "gold_answer": row["answer"],
            "messages": [{"role": "user", "content": question + MATH_SUFFIX}],
            "original_row_index": index, "original": row}


def adapt_gpqa(row, index, seed):
    question = row["Question"]
    choices = [row["Correct Answer"]] + [row[f"Incorrect Answer {i}"] for i in (1, 2, 3)]
    permutation = list(range(4))
    random.Random(stable_seed(seed, question_id(question))).shuffle(permutation)
    shuffled = [choices[i] for i in permutation]
    prompt = question + "\n\n" + "\n".join(f"{letter}. {value}" for letter, value in zip("ABCD", shuffled)) + GPQA_SUFFIX
    return {"sample_id": "gpqa-" + question_id(question), "question_id": question_id(question),
            "task": "gpqa", "question": question, "gold_answer": "ABCD"[permutation.index(0)],
            "messages": [{"role": "user", "content": prompt}], "choices": shuffled,
            "choice_permutation": permutation, "original_row_index": index, "original": row}


def unique_questions(rows):
    found = {}
    for row in rows:
        found.setdefault(row["question_id"], row)
    return list(found.values())


def shingles(text):
    text = normalize_question(text)
    return {text[i:i + 7] for i in range(max(0, len(text) - 6))}


class OverlapIndex:
    """Exact normalized identity + conservative 7-character Jaccard >= 0.9.

    Not a semantic contamination proof. The index only narrows comparisons;
    every true Jaccard match must share a shingle and is therefore considered.
    """
    def __init__(self, rows, threshold=0.9):
        self.rows = rows
        self.threshold = threshold
        self.exact = {r["question_id"]: r["sample_id"] for r in rows}
        self.grams = [shingles(r["question"]) for r in rows]
        self.index = defaultdict(list)
        for i, grams in enumerate(self.grams):
            for gram in grams:
                self.index[gram].append(i)

    def match(self, row):
        if row["question_id"] in self.exact:
            return {"other_id": self.exact[row["question_id"]], "method": "normalized_exact", "similarity": 1.0}
        grams = shingles(row["question"])
        if len(normalize_question(row["question"])) < 30:
            return None
        counts = Counter(i for gram in grams for i in self.index.get(gram, ()))
        for i, intersection in counts.items():
            score = intersection / (len(grams) + len(self.grams[i]) - intersection)
            if score >= self.threshold:
                return {"other_id": self.rows[i]["sample_id"], "method": "char7_jaccard", "similarity": score}
        return None


def matched_loss_groups(rows, scores, size=400, seed=42, length_bin=512):
    """Global loss tertiles, exact source x answer-length-bin quotas per group.

    Never relax matching silently. Insufficient common support raises an error.
    """
    ranked = sorted(rows, key=lambda r: (scores[r["sample_id"]], r["sample_id"]))
    groups = {"low": ranked[:len(ranked) // 3],
              "mid": ranked[len(ranked) // 3:2 * len(ranked) // 3],
              "high": ranked[2 * len(ranked) // 3:]}
    strata = {}
    for name, members in groups.items():
        strata[name] = defaultdict(list)
        for row in members:
            strata[name][(row["source"], row["n_supervised"] // length_bin)].append(row)
    keys = sorted(set.intersection(*(set(x) for x in strata.values())))
    capacities = {k: min(len(strata[g][k]) for g in groups) for k in keys}
    if sum(capacities.values()) < size:
        raise ValueError(f"Only {sum(capacities.values())} matched slots; need {size}. "
                         "Increase --length-bin explicitly or enlarge candidate pool.")
    # Allocate common quotas proportional to the available matched support.
    total = sum(capacities.values())
    quotas = {k: int(size * capacities[k] / total) for k in keys}
    remainder = size - sum(quotas.values())
    for k in sorted(keys, key=lambda k: (-(size * capacities[k] / total - quotas[k]), k))[:remainder]:
        quotas[k] += 1
    selected = {}
    for name in groups:
        rng = random.Random(stable_seed(seed, name))
        selected[name] = []
        for k in keys:
            selected[name].extend(rng.sample(strata[name][k], quotas[k]))
        rng.shuffle(selected[name])
    return selected, {"definition": "global loss tertiles; identical source x supervised-token-bin counts",
                      "length_bin": length_bin, "quotas": [{"source": k[0], "length_bin_index": k[1], "count": quotas[k]} for k in keys],
                      "tertile_maxima": {k: max(scores[r["sample_id"]] for r in v) for k, v in groups.items()}}
