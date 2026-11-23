"""Run the Python code inside saved responses and grade its output (Phase 2, RQ4).

The base model often writes "Let's verify with Python" and a ```python block, then stops or
makes up an output, instead of stating a final answer. The original grader sees no answer.
This script asks: if the code had actually been run (as in a tool-use setting), would its
printed output be correct?

For every response that contains a ```python block:
    1. all python blocks of the response are joined in order (later blocks may use names
       defined earlier) and run in a separate process, in an empty temporary directory,
       with a time limit (--timeout, default 10 s), a memory limit (2 GB) and no stdin;
    2. the answer is the last non-empty line of stdout (if the code prints nothing, but its
       last line is a bare expression, that expression's value);
    3. it is graded against the reference with the same equivalence as regrade.py.
code_result per response: no_code / correct / wrong / error (exception, timeout, no output).

Conditions written (score lists in the format of math_eval.py, for passk_minimal.py):
    tool      revised score (regrade.py) OR, when the response states no answer of its own
              (no non-empty box and no stated answer before the code), code_result==correct.
              A response that states a wrong answer is NOT rescued by its code.
    code_any  revised score OR code_result==correct (upper bound, for reference)

The code comes from the model: run this only on a machine where that is acceptable (e.g. a
compute node in your own account). It runs as your user; the limits above are not a
security sandbox.

Usage (math/ directory, evaluation environment active, CPU only):
  check on the labelled batch-1 items (per-label table, no result files needed):
    python run_code.py --items $SCRATCH/rlvr/annotation_b1/items_full.jsonl \\
        --labels annotation/labels/b1_final.csv --key $SCRATCH/rlvr/annotation_b1/key.csv
  all responses (input = the output of regrade.py):
    python run_code.py 'Qwen2.5-7B=$SCRATCH/rlvr/regraded/Qwen2.5-7B/math500/*.jsonl' \\
        'SimpleRL-Zoo-7B=$SCRATCH/rlvr/regraded/SimpleRL-Zoo-7B/math500/*.jsonl' \\
        --out_dir $SCRATCH/rlvr/code_exec --workers 16
"""
import argparse
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from regrade import boxes, judge, stated_answer  # noqa: E402

BLOCK_RE = re.compile(r"```python\s*\n(.*?)(?:```|$)", re.S)
OUTPUT_RE = re.compile(r"```output")

RUNNER = r'''
import sys, ast, io, contextlib
src = sys.stdin.read()
sys.stdin = io.StringIO("")
tree = ast.parse(src)
last = None
if tree.body and isinstance(tree.body[-1], ast.Expr):
    last = ast.Expression(tree.body.pop().value)
g = {"__name__": "__main__"}
exec(compile(tree, "<response>", "exec"), g)
if last is not None:
    v = eval(compile(last, "<response>", "eval"), g)
    if v is not None:
        print(v)
'''


def code_of(text):
    """Python blocks written by the model (stop at a fabricated ```output block)."""
    return [b for b in BLOCK_RE.findall(text) if b.strip()]


def _limits():
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
    resource.setrlimit(resource.RLIMIT_FSIZE, (10 << 20, 10 << 20))
    os.setsid()


# The child sees exactly the packages of this interpreter (sympy needs mpmath, which may live
# outside the env's site-packages), and nothing from the response's working directory.
CHILD_ENV = {"PATH": "/usr/bin:/bin", "OMP_NUM_THREADS": "1", "MPLBACKEND": "Agg",
             "PYTHONPATH": os.pathsep.join(p for p in sys.path if p and os.path.isdir(p)
                                           and os.path.abspath(p) != HERE)}


def execute(code, timeout):
    with tempfile.TemporaryDirectory() as d:
        env = dict(CHILD_ENV, HOME=d)
        try:
            p = subprocess.run([sys.executable, "-c", RUNNER], input=code, cwd=d,
                               capture_output=True, text=True, timeout=timeout,
                               preexec_fn=_limits, env=env)
        except subprocess.TimeoutExpired:
            return "timeout", ""
    lines = [l.strip() for l in p.stdout.splitlines() if l.strip()]
    if p.returncode != 0 and not lines:
        err = (p.stderr.strip().splitlines() or ["error"])[-1]
        return "error", err[:200]
    if not lines:
        return "no_output", ""
    return "ok", lines[-1]


def answer_from_output(line):
    """Last printed line -> answer string ("The answer is 12" -> "12")."""
    s = line.strip()
    if ":" in s and not s.startswith("{"):
        s = s.rsplit(":", 1)[1].strip()
    m = re.search(r"\b(is|=)\s*(.+)$", s)
    if m:
        s = m.group(2).strip()
    return s.rstrip(".")


def sympy_forms(a, seconds=5):
    """Python/sympy printouts ("6.0*sqrt(2)", "1/tan(x)", "[5]") as LaTeX candidates."""
    import signal
    from regrade import _alarm
    old = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(seconds)
    try:
        import sympy
        e = sympy.sympify(a)
        if isinstance(e, (list, tuple)) and len(e) == 1:
            e = e[0]
        forms = [sympy.latex(e), sympy.latex(sympy.nsimplify(e))]
        forms.append(sympy.latex(sympy.simplify(e.rewrite(sympy.cot))))
        return forms
    except Exception:
        return []
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def run_one(args):
    text, gt, timeout = args
    blocks = code_of(text)
    if not blocks:
        return "no_code", None
    status, out = execute("\n\n".join(blocks), timeout)
    if status != "ok":
        return "error", f"{status}: {out}" if out else status
    a = answer_from_output(out)
    if any(judge(c, gt) for c in [a, out] + sympy_forms(a)):
        return "correct", out[:200]
    return "wrong", out[:200]


def states_answer(text, gt):
    return any(b.strip() for b in boxes(text)) or stated_answer(text, gt) is not None


def pool_map(fn, jobs, workers):
    from regrade import progress_map
    return progress_map(fn, jobs, workers, every=2000, task_timeout=120,
                        on_timeout=("error", "timeout (grading)"))


def run_items(args):
    items = [json.loads(l) for l in open(args.items, encoding="utf-8") if l.strip()]
    labels = {r["item_id"]: r["label"] for r in csv.DictReader(open(args.labels, encoding="utf-8-sig"))}
    key = {}
    if args.key and os.path.exists(args.key):
        key = {r["item_id"]: r for r in csv.DictReader(open(args.key, encoding="utf-8"))}
    items = [it for it in items if it["item_id"] in labels]
    res = pool_map(run_one, [(it["response"], it["ground_truth"], args.timeout) for it in items],
                   args.workers)
    rows = []
    for it, (cr, out) in zip(items, res):
        rows.append({"item_id": it["item_id"], "label": labels[it["item_id"]],
                     "model": key.get(it["item_id"], {}).get("model", "?"),
                     "states_answer": states_answer(it["response"], it["ground_truth"]),
                     "fabricated_output": bool(OUTPUT_RE.search(it["response"])),
                     "code_result": cr, "output": out})
    cols = ["no_code", "correct", "wrong", "error"]
    lines = ["# Code execution on labelled items", "",
             "| label | n | " + " | ".join(cols) + " | fabricated ```output |",
             "|---|---|" + "---|" * (len(cols) + 1)]
    for L in "ABCD":
        rs = [r for r in rows if r["label"] == L]
        c = Counter(r["code_result"] for r in rs)
        lines.append(f"| {L} | {len(rs)} | " + " | ".join(str(c[x]) for x in cols)
                     + f" | {sum(r['fabricated_output'] for r in rs)} |")
    if key:
        lines += ["", "B items by model:", "", "| model | B | code correct | wrong | error | no code |",
                  "|---|---|---|---|---|---|"]
        for m in sorted({r["model"] for r in rows}):
            rs = [r for r in rows if r["label"] == "B" and r["model"] == m]
            c = Counter(r["code_result"] for r in rs)
            lines.append(f"| {m} | {len(rs)} | {c['correct']} | {c['wrong']} | {c['error']} | {c['no_code']} |")
    lines += ["", "| item | label | code | output |", "|---|---|---|---|"]
    for r in rows:
        if r["code_result"] != "no_code":
            o = str(r["output"] or "").replace("|", "\\|")[:80]
            lines.append(f"| {r['item_id']} | {r['label']} | {r['code_result']} | `{o}` |")
    txt = "\n".join(lines)
    print(txt)
    if args.report:
        with open(args.report, "w") as f:
            f.write(txt + "\n")
        print(f"\nSaved to {args.report}")


def run_all(args):
    from analyze_responses import load
    os.makedirs(args.out_dir, exist_ok=True)
    table = []
    for run in args.runs:
        name, pattern = run.split("=", 1)
        problems = load(pattern)
        jobs, where = [], []
        cache = {}
        for d in problems:
            for j, t in enumerate(d["code"]):
                blocks = code_of(t)
                if blocks:
                    jobs.append((t, d["gt"], args.timeout))
                    where.append((d["idx"], j))
        print(f"{name}: {len(problems)} problems, {len(jobs)} responses with code; running ...",
              flush=True)
        res = pool_map(run_one, jobs, args.workers)
        for w, r in zip(where, res):
            cache[w] = r
        stats = Counter()
        for d in problems:
            base = d["score"]  # revised score if the input came from regrade.py
            cr, outs, tool, anyc = [], [], [], []
            for j, t in enumerate(d["code"]):
                c, o = cache.get((d["idx"], j), ("no_code", None))
                cr.append(c)
                outs.append(o)
                stated = states_answer(t, d["gt"])
                tool.append(bool(base[j]) or (c == "correct" and not stated))
                anyc.append(bool(base[j]) or c == "correct")
                stats[c] += 1
                stats["wrong_no_answer_code_correct"] += (not base[j]) and (not stated) and c == "correct"
            d["code_result"], d["code_output"] = cr, outs
            d["score_tool"], d["score_code_any"] = tool, anyc
        for cond, field in (("tool", "score_tool"), ("code_any", "score_code_any")):
            out = os.path.join(args.out_dir, cond, name, "math500")
            os.makedirs(out, exist_ok=True)
            with open(os.path.join(out, "scored.jsonl"), "w") as f:
                for d in problems:
                    f.write(json.dumps(dict(d, score=d[field]), ensure_ascii=False) + "\n")
        n = sum(len(d["code"]) for d in problems)
        p = {k: sum(sum(d[f]) for d in problems) / n
             for k, f in (("input", "score"), ("tool", "score_tool"), ("code_any", "score_code_any"))}
        table.append((name, n, stats, p))
    lines = ["| model | responses | with code | code correct | code wrong | code error | "
             "rescued (no stated answer, code correct) | pass@1 input | pass@1 tool | pass@1 code_any |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name, n, s, p in table:
        wc = n - s["no_code"]
        lines.append(f"| {name} | {n} | {wc} ({100*wc/n:.1f}%) | {s['correct']} | {s['wrong']} | "
                     f"{s['error']} | {s['wrong_no_answer_code_correct']} | {100*p['input']:.2f} | "
                     f"{100*p['tool']:.2f} | {100*p['code_any']:.2f} |")
    txt = "\n".join(lines)
    print("\n" + txt)
    with open(os.path.join(args.out_dir, "code_summary.md"), "w") as f:
        f.write(txt + "\n")
    print(f"\nWrote {args.out_dir}/{{tool,code_any}}/<model>/math500/scored.jsonl and code_summary.md")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", help="NAME=glob of (re-graded) result files")
    ap.add_argument("--out_dir", default="code_out")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=10)
    ap.add_argument("--items", default=None, help="items_full.jsonl (labelled-item mode)")
    ap.add_argument("--labels", default=None, help="final label CSV for --items")
    ap.add_argument("--key", default=None)
    ap.add_argument("--report", default=None)
    args = ap.parse_args()
    if args.items:
        if not args.labels:
            raise SystemExit("--items needs --labels")
        run_items(args)
    elif args.runs:
        run_all(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
