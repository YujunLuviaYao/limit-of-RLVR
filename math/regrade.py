"""Re-grade saved responses with a revised grader (Phase 2, RQ2/RQ3).

The revised grader only tries to RESCUE responses the original grader marked wrong (false
negatives). A response the original grader accepted stays correct, so any change in pass@k
comes from fixing false negatives and nothing else. Rules, tried in order on an originally
wrong response:

    skip  finish_reason == "length" (truncated) or long repetition anywhere in the response
          (team rule: a response that loops counts as wrong even if it stated an answer)
    R1    last NON-EMPTY \\boxed{...}. The original parser takes the last "boxed" in the text,
          so a correct box followed by an echoed instruction ("put your final answer within
          \\boxed{}") is read as the empty answer "".
    R2    no non-empty box: the answer stated in the concluding sentence of the text before
          the first code block (and before any echoed instruction), e.g. "Therefore, the
          answer is $x = 3$." -> "3". Only sentences with a conclusion cue (therefore, thus,
          so, hence, answer, ...) are used; no last-number fallback.
    R3    normalisation applied to R1/R2 answers before math_equal: \\textbf/\\text/\\mathrm
          removed; unicode (π, ×, subscript digits) mapped to LaTeX; right-hand side of an
          equation ("5 x 6 x 8 = 240" -> "240"); base subscript dropped when the reference
          has one (gt "52_8" accepts "52" and "52₈"); comma lists compared as unordered.
Equivalence is decided by the authors' grader.math_equal on the authors' strip_string
output, as in the original evaluation, with a time limit per comparison.

Usage (math/ directory, evaluation environment active, CPU only):
  1) check the revised grader against the human labels of batch 1 (A/D = correct):
     python regrade.py --validate annotation/labels/b1_final.csv \\
         --items $SCRATCH/rlvr/annotation_b1/items_full.jsonl \\
         --key $SCRATCH/rlvr/annotation_b1/key.csv
  2) re-grade every response and write jsonl files in the format of math_eval.py:
     python regrade.py 'Qwen2.5-7B=$SCRATCH/rlvr/main_out*/Qwen2.5-7B/chunk_*/math500/*.jsonl' \\
         'SimpleRL-Zoo-7B=$SCRATCH/rlvr/main_out*/SimpleRL-Zoo-7B/chunk_*/math500/*.jsonl' \\
         --out_dir $SCRATCH/rlvr/regraded --workers 16
     then corrected pass@k curves:
     python passk_minimal.py --bootstrap 2000 --quiet --plot fig_passk_regraded.png \\
         'Base=$SCRATCH/rlvr/regraded/Qwen2.5-7B/math500/*.jsonl' \\
         'RL=$SCRATCH/rlvr/regraded/SimpleRL-Zoo-7B/math500/*.jsonl'

Each output line keeps every field of the input and replaces `score` with the revised
score; the original is kept in `score_orig` and the rule that fired in `revised_rule`
("orig" = already correct, "R1"/"R2" = rescued, "" = still wrong).
"""
import argparse
import csv
import json
import os
import re
import signal
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "examples", "math_eval"))
sys.path.insert(0, HERE)

CODE_FENCE = "```"
ECHO_RE = re.compile(r"(please reason step by step|put your final answer|final answer (is )?within"
                     r"|within \\boxed\{\})", re.I)
CUE_RE = re.compile(r"\b(therefore|thus|hence|so|answer|final|conclusion|in summary|result|"
                    r"we (get|find|have))\b", re.I)
VERB_RE = re.compile(r"(\bis\b|\bare\b|\bequals?\b|\bbe\b|=|:)", re.I)
MATH_RE = re.compile(r"(\$\$.+?\$\$|\\\[.+?\\\]|\\\(.+?\\\)|\$.+?\$)", re.S)
SUBSCRIPT = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")


# ---------------------------------------------------------------- helpers
def boxes(text):
    """Contents of every \\boxed{...} (balanced braces), in order."""
    out, i = [], 0
    while True:
        j = text.find("boxed", i)
        if j < 0:
            return out
        k = j + 5
        if k < len(text) and text[k] == "{":
            depth, start, k = 1, k + 1, k + 1
            while k < len(text) and depth:
                depth += (text[k] == "{") - (text[k] == "}")
                k += 1
            out.append(text[start:k - 1] if depth == 0 else text[start:k])
        i = j + 5


def repetitive(text, span=30, times=5, window=1500):
    """Some 30-character string recurs >= 5 times within a 1500-character window."""
    for end in range(len(text), window - 1, -window // 2) if len(text) > window else [len(text)]:
        tail = text[max(0, end - window):end]
        if len(tail) < span * times:
            continue
        c = Counter(tail[i:i + span] for i in range(0, len(tail) - span, 10))
        if c.most_common(1)[0][1] >= times:
            return True
    return False


def strip_math(s):
    s = s.strip()
    for a, b in (("$$", "$$"), ("\\[", "\\]"), ("\\(", "\\)"), ("$", "$")):
        if s.startswith(a) and s.endswith(b) and len(s) >= len(a) + len(b):
            return s[len(a):len(s) - len(b)].strip()
    return s


def unwrap(s, cmd):
    """\\cmd{X} -> X (balanced)."""
    key = "\\" + cmd + "{"
    while key in s:
        i = s.find(key)
        k, depth = i + len(key), 1
        while k < len(s) and depth:
            depth += (s[k] == "{") - (s[k] == "}")
            k += 1
        s = s[:i] + s[i + len(key):k - 1] + s[k:]
    return s


def normalise(ans, gt):
    """R3: candidate strings for an extracted answer (first = plain cleanup)."""
    a = ans.strip().rstrip(".").strip()
    a = a.replace("π", "\\pi ").replace("×", "\\times ").replace("·", "\\cdot ").replace("−", "-")
    a = re.sub(r"([0-9A-Za-z])([₀-₉]+)", lambda m: m.group(1) + "_" + m.group(2).translate(SUBSCRIPT), a)
    for cmd in ("textbf", "mathbf", "text", "mathrm", "boxed"):
        a = unwrap(a, cmd)
    cands = [a]
    if "=" in a and not re.search(r"[<>]|\\le(q|\b)|\\ge(q|\b)|\\ne(q|\b)", a):
        cands.append(a.rsplit("=", 1)[1].strip())
    m = re.fullmatch(r"(\w+)_\{?(\d+)\}?", gt.strip())
    if m:  # base subscript in the reference: compare digits, with or without the subscript
        for c in list(cands):
            mc = re.fullmatch(r"\(?(\w+)\)?(_\{?\d+\}?)?", c.replace(" ", ""))
            if mc and mc.group(1) == m.group(1):
                cands.append(gt.strip())
    return cands


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


def equal(pred, gt, seconds=5):
    """Authors' math_equal on strip_string output, with a time limit (main thread only)."""
    from grader import math_equal
    from parser import strip_string
    old = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(seconds)
    try:
        return bool(math_equal(strip_string(pred), gt))
    except Exception:
        return False
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def list_equal(pred, gt):
    """Unordered comparison of comma lists when the reference is a bare list ("1,-2")."""
    if "," not in gt or re.search(r"[\(\)\[\]\{\}]|\\cup|pmatrix", gt):
        return False
    p = [x.strip() for x in pred.split(",") if x.strip()]
    g = [x.strip() for x in gt.split(",") if x.strip()]
    if len(p) != len(g):
        return False
    used = [False] * len(g)
    for x in p:
        for i, y in enumerate(g):
            if not used[i] and equal(x, y):
                used[i] = True
                break
        else:
            return False
    return True


def judge(ans, gt):
    for c in normalise(ans, gt):
        if c and (equal(c, gt) or list_equal(c, gt)):
            return True
    return False


# ---------------------------------------------------------------- R2 extraction
def body_before_code(text):
    t = text.split(CODE_FENCE)[0]
    m = ECHO_RE.search(t)
    if m and m.start() > 0:
        t = t[:m.start()]
    return t


def sentences(t):
    """Split into sentences without breaking inside math segments."""
    segs = MATH_RE.split(t)
    out, cur = [], ""
    for k, s in enumerate(segs):
        if k % 2 == 1:  # math
            cur += s
            continue
        parts = re.split(r"((?<=[.!?])\s+|\n\s*\n)", s)
        for p in parts:
            if p is None:
                continue
            if re.fullmatch(r"(?<=[.!?])\s+|\n\s*\n|\s+", p) and cur.strip():
                out.append(cur)
                cur = ""
            else:
                cur += p
    if cur.strip():
        out.append(cur)
    return [x.strip() for x in out if x.strip()]


def stated_answer(text, gt="", last=4):
    """R2: answer in the last conclusion sentence of the text before code / echo, or None."""
    sents = sentences(body_before_code(text))[-last:]
    word_gt = bool(re.fullmatch(r"[A-Za-z][A-Za-z ]{2,}", gt.strip()))
    for pos, s in enumerate(reversed(sents)):
        if re.search(r"\b(python|code)\b", s, re.I):
            continue
        if word_gt:  # answers that are names/words ("Navin"): the final sentence names it
            return gt.strip() if re.search(r"\b" + re.escape(gt.strip()) + r"\b", s) else None
        if not CUE_RE.search(s) and not (pos == 0 and re.search(
                r"\b(is|are|equals)\b:?\s*(\$|\\\[|\\\()", s)):
            continue  # no conclusion cue (the final sentence may still be "X is $...$")
        maths = [m.group(0) for m in MATH_RE.finditer(s)]
        # text after the last verb outside math
        plain = MATH_RE.sub(lambda m: "\x00" * len(m.group(0)), s)
        verbs = [m.end() for m in VERB_RE.finditer(plain)]
        if verbs:
            rest = s[verbs[-1]:]
            m = MATH_RE.search(rest)
            if m and len(rest[:m.start()].split()) <= 3:  # "is $x$", "is approximately $x$"
                return strip_math(m.group(0))
            num = re.match(r"\s*(-?\d[\d,]*\.?\d*(/\d+)?)", rest)
            if num:
                return num.group(1).rstrip(".,")
        if maths:
            return strip_math(maths[-1])
        nums = re.findall(r"-?\d+\.?\d*(?:/\d+)?", s)
        if nums:
            return nums[-1]
    return None


def revise(text, gt, score, finish_reason=None):
    """Return (correct, rule, extracted)."""
    if score:
        return True, "orig", None
    if finish_reason == "length" or repetitive(text):
        return False, "", None
    nonempty = [b for b in boxes(text) if b.strip()]
    if nonempty:
        a = nonempty[-1]
        return judge(a, gt), ("R1" if judge(a, gt) else ""), a
    a = stated_answer(text, gt)
    if a is None:
        return False, "", None
    ok = judge(a, gt)
    return ok, ("R2" if ok else ""), a


# ---------------------------------------------------------------- modes
def _work(args):
    text, gt, s, fr = args
    return revise(text, gt, s, fr)


def regrade_file(problems, workers):
    jobs = []
    for d in problems:
        frs = d.get("finish_reason") or [None] * len(d["code"])
        for t, s, fr in zip(d["code"], d["score"], frs):
            jobs.append((t, d["gt"], bool(s), fr))
    if workers > 1:
        from multiprocessing import Pool
        with Pool(workers) as pool:
            res = pool.map(_work, jobs, chunksize=64)
    else:
        res = [_work(j) for j in jobs]
    k = 0
    for d in problems:
        n = len(d["code"])
        r = res[k:k + n]
        k += n
        d["score_orig"] = [bool(s) for s in d["score"]]
        d["score"] = [x[0] for x in r]
        d["revised_rule"] = [x[1] for x in r]
        d["revised_pred"] = [x[2] for x in r]
    return problems


def run_regrade(args):
    from analyze_responses import load
    summary = []
    for run in args.runs:
        name, pattern = run.split("=", 1)
        problems = load(pattern)
        print(f"{name}: {len(problems)} problems, re-grading ...", flush=True)
        problems = regrade_file(problems, args.workers)
        out = os.path.join(args.out_dir, name, "math500")
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "regraded.jsonl"), "w") as f:
            for d in problems:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        rules = Counter(r for d in problems for r in d["revised_rule"])
        n = sum(len(d["score"]) for d in problems)
        p_old = sum(sum(d["score_orig"]) for d in problems) / n
        p_new = sum(sum(d["score"]) for d in problems) / n
        echo = sum(any(not b.strip() for b in boxes(t)) for d in problems for t in d["code"])
        summary.append((name, len(problems), n, p_old, p_new, rules["R1"], rules["R2"], echo))
    lines = ["| model | problems | responses | pass@1 orig | pass@1 revised | rescued R1 | rescued R2 | responses with an empty box |",
             "|---|---|---|---|---|---|---|---|"]
    for name, P, n, a, b, r1, r2, e in summary:
        lines.append(f"| {name} | {P} | {n} | {100*a:.2f} | {100*b:.2f} | {r1} ({100*r1/n:.2f}%) | "
                     f"{r2} ({100*r2/n:.2f}%) | {e} ({100*e/n:.2f}%) |")
    txt = "\n".join(lines)
    print("\n" + txt)
    with open(os.path.join(args.out_dir, "regrade_summary.md"), "w") as f:
        f.write(txt + "\n")
    print(f"\nWrote {args.out_dir}/<model>/math500/regraded.jsonl and regrade_summary.md")


def run_validate(args):
    labels = {r["item_id"]: r["label"] for r in csv.DictReader(open(args.validate, encoding="utf-8-sig"))}
    items = {}
    for line in open(args.items, encoding="utf-8"):
        if line.strip():
            d = json.loads(line)
            items[d["item_id"]] = d
    key = {}
    if args.key and os.path.exists(args.key):
        key = {r["item_id"]: r for r in csv.DictReader(open(args.key, encoding="utf-8"))}
    rows = []
    for iid, lab in sorted(labels.items()):
        it = items[iid]
        ok, rule, a = revise(it["response"], it["ground_truth"], False)
        rows.append((iid, lab, ok, rule, a, key.get(iid, {}).get("model", "?"),
                     key.get(iid, {}).get("source", "?")))
    tp = sum(r[2] and r[1] in "AD" for r in rows)
    fp = sum(r[2] and r[1] in "BC" for r in rows)
    fn = sum((not r[2]) and r[1] in "AD" for r in rows)
    tn = sum((not r[2]) and r[1] in "BC" for r in rows)
    out = ["# Revised grader vs human labels (A/D = correct, B/C = wrong)", "",
           f"items: {len(rows)}   TP {tp}  FP {fp}  FN {fn}  TN {tn}",
           f"precision {tp/max(tp+fp,1):.3f}   recall {tp/max(tp+fn,1):.3f}   "
           f"accuracy {(tp+tn)/len(rows):.3f}", "",
           "| label | n | revised says correct | via R1 | via R2 |", "|---|---|---|---|---|"]
    for L in "ABCD":
        rs = [r for r in rows if r[1] == L]
        out.append(f"| {L} | {len(rs)} | {sum(r[2] for r in rs)} | {sum(r[3]=='R1' for r in rs)} | "
                   f"{sum(r[3]=='R2' for r in rs)} |")
    if key:
        out += ["", "| model | source | n | label A/D | revised correct | agree |", "|---|---|---|---|---|---|"]
        g = defaultdict(list)
        for r in rows:
            g[(r[5], r[6])].append(r)
        for (m, s), rs in sorted(g.items()):
            out.append(f"| {m} | {s} | {len(rs)} | {sum(r[1] in 'AD' for r in rs)} | "
                       f"{sum(r[2] for r in rs)} | {sum(r[2] == (r[1] in 'AD') for r in rs)} |")
    out += ["", "## Disagreements", "", "| item | label | revised | rule | extracted |", "|---|---|---|---|---|"]
    for r in rows:
        if r[2] != (r[1] in "AD"):
            a = (r[4] or "").replace("|", "\\|").replace("\n", " ")[:80]
            out.append(f"| {r[0]} | {r[1]} | {'correct' if r[2] else 'wrong'} | {r[3]} | `{a}` |")
    txt = "\n".join(out)
    print(txt)
    if args.report:
        with open(args.report, "w") as f:
            f.write(txt + "\n")
        print(f"\nSaved to {args.report}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", help="NAME=glob of math_eval.py result files (quote it)")
    ap.add_argument("--out_dir", default="regraded_out")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--validate", default=None, help="final label CSV (item_id,label,...)")
    ap.add_argument("--items", default=None, help="items_full.jsonl of the same batch")
    ap.add_argument("--key", default=None, help="key.csv of the same batch (optional, per-model table)")
    ap.add_argument("--report", default=None, help="save the validation report (.md)")
    args = ap.parse_args()
    if args.validate:
        if not args.items:
            raise SystemExit("--validate needs --items")
        run_validate(args)
    elif args.runs:
        run_regrade(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
