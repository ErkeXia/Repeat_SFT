"""Conservative question-text tags, not an official Math-CoT-20k taxonomy."""
from __future__ import annotations

from collections import Counter
import random
import re
import unicodedata

from recipe.common import digest, stable_seed
from recipe.data import OverlapIndex
from recipe.encoding import question_id

VERSION = "question-topic-rules-v1"
TOPICS = {
    "algebra": "代数与方程", "geometry": "几何", "number_theory": "数论",
    "combinatorics_probability": "组合与概率", "calculus": "微积分",
    "sequences": "数列与递推", "inequalities": "不等式",
}
DEFAULT_TOPICS = tuple(TOPICS)[:5]
# Strong topic vocabulary only. Generic words such as integer, function, solution,
# and number are deliberately insufficient. Multiple matches are left ambiguous.
RULES = {
    "algebra": [r"\b(?:polynomials?|quadratic|cubic|quartic|equations?|complex numbers?|matri(?:x|ces)|determinants?|logarithms?)\b",
                r"\bsolve\s+(?:for\s+)?[xyz]\b", r"代数|方程|多项式|复数|矩阵|对数"],
    "geometry": [r"\b(?:triangles?|circles?|circumcircles?|incircles?|quadrilaterals?|rectangles?|parallelograms?|trapez(?:oids?|iums?)|rhomb(?:us|i)|polygons?|tetrahedr(?:on|a)|cuboids?|spheres?|cylinders?|cones?|parabolas?|ellipses?|hyperbolas?|inscribed|circumscribed|perpendicular|collinear|concyclic)\b",
                 r"\b(?:planes?|chords?|diameters?|radius|radii|incenter|circumcenter|orthocenter)\b",
                 r"三角形|圆周|圆心|内切圆|外接圆|四边形|长方形|平行四边形|梯形|菱形|多边形|四面体|长方体|球体|圆柱|圆锥|垂直|共线|共圆"],
    "number_theory": [r"\b(?:primes?|divisors?|divisible|divisibility|remainders?|modulo|coprime|relatively prime|congruences?|diophantine|gcd|lcm)\b",
                      r"\b(?:greatest common divisor|least common multiple|units digit|last\s+(?:\d+|two|three)\s+digits)\b",
                      r"\\(?:pmod|bmod)\b", r"素数|质数|整除|约数|余数|同余|互质|最大公约数|最小公倍数"],
    "combinatorics_probability": [r"\b(?:probabilit(?:y|ies)|randomly|at random|permutations?|combinations?|combinatorial|binomial coefficients?|expected value|expectation)\b",
                                  r"\b(?:how many|number of)\s+(?:different\s+|distinct\s+)?ways\b",
                                  r"概率|随机|排列|组合|期望|多少种|几种方法"],
    "calculus": [r"\b(?:derivatives?|differentiat(?:e|ion|ing|ed)|integrat(?:e|ion|ing|ed)|antiderivatives?|differentiable|calculus)\b",
                 r"\bintegrals?\b(?!\s+(?:solutions?|values?|roots?|points?|coordinates?|parts?|numbers?|sides?|lengths?|coefficients?)\b)",
                 r"\b(?:evaluate|compute|find|determine)\s+(?:the\s+)?(?:following\s+)?limit\b",
                 r"\\(?:int|iint|iiint|oint|lim)\b", r"导数|微分|积分|求极限"],
    "sequences": [r"\b(?:sequences?|recurrence|recursive|recursively|fibonacci|arithmetic progression|geometric progression)\b",
                  r"数列|递推|斐波那契|等差|等比"],
    "inequalities": [r"\b(?:inequalit(?:y|ies)|am-gm|cauchy.schwarz)\b", r"不等式"],
}


def classify(question):
    text = unicodedata.normalize("NFKC", question)
    matches = {topic: sorted({m.group(0) for rule in rules for m in re.finditer(rule, text, re.I)})
               for topic, rules in RULES.items()}
    matches = {topic: words for topic, words in matches.items() if words}
    return {"topic": next(iter(matches)) if len(matches) == 1 else None,
            "status": "single_topic" if len(matches) == 1 else "ambiguous" if matches else "unclassified",
            "matches": matches}


def select_rows(rows, references, seed, check_stop=lambda: None):
    """Ten per default topic; original answers untouched, no quota relaxation."""
    overlap = OverlapIndex(references)
    candidates = {topic: [] for topic in DEFAULT_TOPICS}
    audit, seen_questions, seen_samples = [], set(), set()
    for row in rows:
        check_stop()
        sid, qid = row["sample_id"], row["question_id"]
        if qid != question_id(row["question"]) or sid != digest([qid, row["response"]]):
            raise ValueError(f"Pool sample identity changed: {sid}")
        tag = classify(row["question"])
        record = {k: row[k] for k in ("row_index", "sample_id", "question_id", "source")}
        record.update(tag)
        record["disposition"] = tag["status"]
        if sid in seen_samples or qid in seen_questions:
            record["disposition"] = "duplicate"
        elif (not 0 < row.get("n_supervised", 0) <= row.get("n_tokens", 0) <= 10240
              or row.get("verification", {}).get("correct") is not True):
            record["disposition"] = "ineligible_length_or_verification"
        elif re.search(r"!\[[^\]]*\]\(|<img\b|\\includegraphics\b|https?://\S+\.(?:png|jpe?g|gif|svg)", row["question"], re.I):
            record["disposition"] = "external_image"
        elif tag["topic"] and tag["topic"] not in candidates:
            record["disposition"] = "outside_requested_topics"
        elif tag["topic"]:
            match = overlap.match(row)
            if match:
                record.update(disposition="evaluation_overlap", overlap=match)
            else:
                record["disposition"] = "candidate"
                candidates[tag["topic"]].append((row, record))
        seen_questions.add(qid)
        seen_samples.add(sid)
        audit.append(record)
    selected = []
    for topic in DEFAULT_TOPICS:
        ordered = sorted(candidates[topic], key=lambda pair: pair[0]["sample_id"])
        random.Random(stable_seed(seed, topic)).shuffle(ordered)
        count = 0
        for row, record in ordered:
            check_stop()
            match = OverlapIndex(selected).match(row)
            if match:
                record.update(disposition="selected_question_overlap", overlap=match)
                continue
            record["disposition"] = "selected"
            selected.append({**row, "topic_selection": {"version": VERSION, "topic": topic,
                              "label": TOPICS[topic], "matches": record["matches"][topic]}})
            count += 1
            if count == 10:
                break
        if count != 10:
            raise ValueError(f"Topic {topic}: only {count} distinct eligible questions, need 10; "
                             "no automatic substitution or repeated questions")
    random.Random(stable_seed(seed, "mixed-topic-training-order")).shuffle(selected)
    stats = {"pool_rows": len(audit), "classification_counts": dict(Counter(r["topic"] or r["status"] for r in audit)),
             "candidate_counts": {k: len(v) for k, v in candidates.items()},
             "dispositions": dict(Counter(r["disposition"] for r in audit)),
             "selected_counts": dict(Counter(r["topic_selection"]["topic"] for r in selected))}
    return selected, audit, stats
