"""Black-box audit of the authors' answer extraction, grading and pass@k code.

Feeds hand-written responses to the SAME functions the evaluation uses
(examples/math_eval: parser.run_execute -> grader.math_equal, evaluate.estimate_pass_at_k,
evaluate.evaluate) and reports what they do, so the behaviour can be checked without
reading the code. No GPU or model is needed.

Usage (from the math/ directory, with the evaluation environment active):
    python audit_grader.py                     # prints tables, writes audit_out/grader_audit.md
    python audit_grader.py --out other.md

Sections:
  1. Grading cases: for each response, the answer the original pipeline extracts, its verdict,
     whether that matches a human judgement, and (for comparison) math-verify run on the full
     response, i.e. the "revised grader" idea from our plan.
  2. pass@k estimator: compared against the exact formula 1 - C(n-c,k)/C(n,k).
  3. Padding: what evaluate.evaluate() does when problems have different numbers of samples.
"""
import argparse
import json
import os
import re
import sys
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.join(HERE, "examples", "math_eval")
sys.path.insert(0, EVAL_DIR)

from parser import parse_ground_truth, run_execute  # noqa: E402  (authors' code)
from grader import math_equal  # noqa: E402
from evaluate import estimate_pass_at_k, evaluate  # noqa: E402

DATA_NAME = "math500"
PROMPT_TYPE = "qwen-boxed"


def load_problems():
    with open(os.path.join(EVAL_DIR, "data", DATA_NAME, "test.jsonl")) as f:
        return [json.loads(line) for line in f]


def qwen_stop_words():
    """Stop words math_eval.py uses for the qwen-boxed prompt, read from its source."""
    src = open(os.path.join(EVAL_DIR, "math_eval.py")).read()
    words = ["</s>", "<|im_end|>", "<|endoftext|>"]
    m = re.search(r'elif "qwen" in args\.prompt_type:\s*\n\s*stop_words\.extend\((\[.*?\])\)', src)
    if m:
        words += json.loads(m.group(1))
    return words


def apply_stop_words(text, stop_words):
    """Same post-processing as math_eval.py: cut the response at the first stop word."""
    cut = None
    for w in stop_words:
        if w in text:
            text = text.split(w)[0].strip()
            cut = cut or w
    return text, cut


def math_verify_full(response, gt):
    try:
        from math_verify import parse, verify
    except ImportError:
        return "n/a"
    try:
        return bool(verify(parse("$" + gt + "$"), parse(response)))
    except Exception as e:  # math-verify can raise on odd input
        return f"error: {type(e).__name__}"


# (case name, MATH500 index, response text, human judgement of correctness, note)
CASES = [
    ("boxed, correct", 2,
     "Adding the two areas gives $\\frac{14}{3}$. So the answer is $\\boxed{\\frac{14}{3}}$.",
     True, "baseline"),
    ("boxed, wrong (control)", 2,
     "Adding the two areas gives $\\boxed{\\frac{13}{3}}$.",
     False, "baseline"),
    ("boxed, \\dfrac instead of \\frac", 2,
     "The area is $\\boxed{\\dfrac{14}{3}}$.",
     True, "equivalent notation"),
    ("boxed, plain-text fraction", 11,
     "The probability is $\\boxed{3/56}$.",
     True, "equivalent notation"),
    ("boxed, degrees without symbol", 7,
     "So the angle is $\\boxed{90}$.",
     True, "equivalent notation"),
    ("boxed, answer without 'x ='", 23,
     "Solving the equation gives $\\boxed{5}$.",
     True, "equivalent notation"),
    ("boxed, text answer", 4,
     "Evelyn has the greatest average speed, so the answer is $\\boxed{\\text{Evelyn}}$.",
     True, "text answer"),
    ("no box, 'final answer is' phrase", 2,
     "Therefore, the final answer is $\\frac{14}{3}$.",
     True, "format: no \\boxed{}"),
    ("no box, answer then Python code (smoke-test case)", 0,
     "The point is on the positive y-axis, so $r = 3$ and $\\theta = \\frac{\\pi}{2}$. "
     "The polar coordinates are $(3, \\frac{\\pi}{2})$.\n\n```python\nimport math\n"
     "r = math.sqrt(0**2 + 3**2)\ntheta = math.atan2(3, 0)\nprint((r, theta))\n```",
     True, "format: no \\boxed{}"),
    ("no box, answer is the last number", 3,
     "Counting the cases one by one, there are 9 in total.",
     True, "format: no \\boxed{}"),
    ("no box, another number after the answer", 3,
     "Counting the cases one by one, there are 9 in total. I checked this with 2 methods.",
     True, "format: no \\boxed{}"),
    ("two boxes, corrected to the right answer", 3,
     "First I get $\\boxed{8}$. Wait, I missed one case, so the answer is $\\boxed{9}$.",
     True, "multiple answers"),
    ("two boxes, changed to a wrong answer", 3,
     "First I get $\\boxed{9}$. Wait, I double-counted, so the answer is $\\boxed{8}$.",
     False, "multiple answers"),
    ("truncated before the answer", 2,
     "First, split the region into a triangle and a rectangle. The triangle has base 4 and",
     False, "truncation"),
    ("contains the stop word 'user'", 2,
     "As the user might expect, adding the two areas gives $\\boxed{\\frac{14}{3}}$.",
     True, "stop word"),
    ("contains the stop word '_start'", 3,
     "Let $t_start = 0$. Counting from $t_start$, there are $\\boxed{9}$ cases.",
     True, "stop word"),
]


def grading_section(problems, stop_words):
    rows = []
    for name, idx, response, human, note in CASES:
        _, gt = parse_ground_truth(problems[idx], DATA_NAME)
        saved, cut = apply_stop_words(response, stop_words)
        pred, _ = run_execute(None, saved, PROMPT_TYPE, DATA_NAME)
        try:
            verdict = bool(math_equal(pred, gt, timeout=True))
        except Exception as e:
            verdict = f"error: {type(e).__name__}"
        rows.append({
            "case": name, "note": note, "idx": idx, "gt": gt, "human": human,
            "cut_by": cut, "pred": pred, "original": verdict,
            # math-verify sees the saved text too: in a real run, vLLM stops generating at the
            # stop word, so the rest of the response never exists.
            "agrees": verdict == human, "math_verify_full": math_verify_full(saved, gt),
        })
    return rows


def passk_section():
    rows = []
    for n, c, k in [(128, 0, 1), (128, 1, 1), (128, 1, 128), (128, 5, 8), (128, 64, 16),
                    (128, 127, 2), (16, 3, 4), (2, 1, 2)]:
        est = float(estimate_pass_at_k(n, [c], k)[0])
        exact = 1 - comb(n - c, k) / comb(n, k)
        rows.append({"n": n, "c": c, "k": k, "authors": est, "exact": exact,
                     "match": abs(est - exact) < 1e-9})
    return rows


def padding_section(problems):
    """Problem A has 2 samples (wrong, right), problem B has 4 (all right)."""
    def sample(idx, preds):
        p = dict(problems[idx], idx=idx)
        p["pred"] = preds
        return p
    gt2 = parse_ground_truth(problems[2], DATA_NAME)[1]
    gt3 = parse_ground_truth(problems[3], DATA_NAME)[1]
    samples = [sample(2, ["13/3", gt2]), sample(3, [gt3] * 4)]
    _, result = evaluate(DATA_NAME, PROMPT_TYPE, samples=samples)
    reported = result["pass@k"][1] / 100
    true_pass1 = (1 / 2 + 4 / 4) / 2  # each problem's own fraction correct, averaged
    padded_pass1 = (3 / 4 + 4 / 4) / 2  # A padded with copies of its last score (correct)
    return {"reported_pass@1": reported, "unpadded_pass@1": true_pass1,
            "padded_pass@1": padded_pass1}


def fmt(v):
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.4f}"
    return "" if v is None else str(v).replace("|", "\\|").replace("\n", " ")


def table(rows, cols, headers):
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(fmt(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "audit_out", "grader_audit.md"))
    args = ap.parse_args()

    problems = load_problems()
    stop_words = qwen_stop_words()

    grading = grading_section(problems, stop_words)
    passk = passk_section()
    padding = padding_section(problems)

    md = ["# Grader audit (authors' evaluation code)", "",
          f"Benchmark `{DATA_NAME}`, prompt `{PROMPT_TYPE}`. Stop words used by math_eval.py: "
          + ", ".join(f"`{w}`" for w in stop_words), "",
          "## 1. Grading cases", "",
          "`extracted` is what the original pipeline takes as the answer (after its stop-word "
          "cut); `original` is its verdict; `agrees` compares that with the human judgement; "
          "`math-verify (full)` runs math-verify on the whole saved response, for comparison. "
          "A response cut at a stop word stays cut for math-verify as well, because in a real "
          "run vLLM stops generating there.", "",
          table(grading,
                ["case", "note", "idx", "gt", "human", "cut_by", "pred", "original", "agrees",
                 "math_verify_full"],
                ["case", "type", "MATH500 idx", "ground truth", "human: correct?",
                 "cut at stop word", "extracted", "original: correct?", "agrees",
                 "math-verify (full)"]), "",
          f"Original grader disagrees with the human judgement on "
          f"{sum(not r['agrees'] for r in grading)} of {len(grading)} cases.", "",
          "## 2. pass@k estimator", "",
          table(passk, ["n", "c", "k", "authors", "exact", "match"],
                ["n", "c", "k", "authors' estimate", "exact 1-C(n-c,k)/C(n,k)", "match"]), "",
          "## 3. Unequal numbers of samples", "",
          "Problem A has 2 samples (1 correct), problem B has 4 (all correct). evaluate() pads "
          "shorter score lists with copies of their last score.", "",
          f"- pass@1 reported by evaluate(): {padding['reported_pass@1']:.4f}",
          f"- pass@1 without padding: {padding['unpadded_pass@1']:.4f}",
          f"- pass@1 if A is padded with its last score: {padding['padded_pass@1']:.4f}", ""]
    text = "\n".join(md)
    print(text)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(text)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
