"""Describe the saved responses of one or more models (format, truncation, extraction path).

First step of Phase 2: before any re-grading, measure how responses that the original
grader marked wrong are distributed over format-level categories, per model and per
difficulty level, and export samples for manual reading.

Usage (from math/, evaluation environment active, no GPU needed):
    python analyze_responses.py \
        'SimpleRL-Zoo-7B=$SCRATCH/rlvr/main_out/SimpleRL-Zoo-7B/chunk_*/math500/*.jsonl' \
        'Qwen2.5-7B=$SCRATCH/rlvr/main_out/Qwen2.5-7B/chunk_*/math500/*.jsonl' \
        --out_dir analysis_out

Writes to --out_dir:
    summary.md            tables printed to the screen
    responses.csv         one row per response (for later joins with manual labels)
    samples_<model>.md    random incorrect responses per category, for reading

How each response is classified
-------------------------------
extraction path: which rule of the authors' parser.extract_answer produced the answer,
checked in the same order as that function:
    minerva_phrase  "final answer is $ ... $. I hope"
    boxed           the text contains "boxed" anywhere (the LAST occurrence is used)
    answer_is       "he answer is"
    final_answer_is "final answer is"
    zh_answer       "答案是"
    last_number     none of the above: the last number in the text
    none            none of the above and no number at all

category of an INCORRECT response (score False), first match wins:
    truncated    finish_reason == "length" (hit the token limit)
    no_box       not truncated, and the answer did not come from a \\boxed{}
                 (candidate format failure: may be a false negative)
    boxed_wrong  not truncated, answer taken from \\boxed{}, still marked wrong
                 (wrong answer, OR an equivalent answer the grader failed to match)
These are format-level categories, not the final RQ1 labels: telling an extraction
failure from a reasoning error inside no_box and boxed_wrong needs the revised grader
and manual checks. Responses cut early by a stop word cannot be detected from the
saved files (vLLM reports them as an ordinary "stop").
"""
import argparse
import csv
import glob
import json
import os
import random
import re
import statistics
from collections import Counter, defaultdict

CATEGORIES = ["truncated", "no_box", "boxed_wrong"]


def extraction_path(text):
    """Mirror the branch order of parser.extract_answer (examples/math_eval/parser.py)."""
    if "final answer is $" in text and "$. I hope" in text:
        return "minerva_phrase"
    if "boxed" in text:
        return "boxed"
    if "he answer is" in text:
        return "answer_is"
    if "final answer is" in text:
        return "final_answer_is"
    if "答案是" in text:
        return "zh_answer"
    if re.findall(r"-?\d*\.?\d+", text.replace(",", "")):
        return "last_number"
    return "none"


def category(score, finish_reason, path):
    if score:
        return "correct"
    if finish_reason == "length":
        return "truncated"
    if path != "boxed":
        return "no_box"
    return "boxed_wrong"


def load(pattern):
    paths = sorted(glob.glob(os.path.expandvars(pattern)))
    if not paths:
        raise SystemExit(f"No files match {pattern}")
    problems = {}
    for p in paths:
        with open(p) as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    problems[d["idx"]] = d
    return [problems[i] for i in sorted(problems)]


def analyze(name, problems):
    rows = []
    dup = []
    for d in problems:
        codes, scores = d["code"], d["score"]
        reasons = d.get("finish_reason") or [None] * len(codes)
        preds = d.get("pred") or [None] * len(codes)
        dup.append(len(set(codes)) / len(codes))
        for j, (text, s, fr, pred) in enumerate(zip(codes, scores, reasons, preds)):
            path = extraction_path(text)
            rows.append({
                "model": name, "idx": d["idx"], "sample": j, "level": d.get("level", ""),
                "score": bool(s), "finish_reason": fr, "path": path,
                "category": category(bool(s), fr, path), "chars": len(text),
                "pred": pred, "gt": d.get("gt", ""),
            })
    return rows, dup


def pct(a, b):
    return f"{100 * a / b:.1f}%" if b else "-"


def summary_tables(all_rows, dups):
    models = list(dict.fromkeys(r["model"] for r in all_rows))
    by_model = {m: [r for r in all_rows if r["model"] == m] for m in models}
    out = []

    head = ["metric"] + models
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]

    def add(label, fn):
        lines.append("| " + " | ".join([label] + [fn(by_model[m], m) for m in models]) + " |")

    add("problems", lambda rs, m: str(len({r["idx"] for r in rs})))
    add("responses", lambda rs, m: str(len(rs)))
    add("accuracy (pass@1)", lambda rs, m: pct(sum(r["score"] for r in rs), len(rs)))
    add("contains \\boxed", lambda rs, m: pct(sum(r["path"] == "boxed" for r in rs), len(rs)))
    add("truncated (all responses)",
        lambda rs, m: pct(sum(r["finish_reason"] == "length" for r in rs), len(rs)))
    add("median length (chars)", lambda rs, m: f"{statistics.median(r['chars'] for r in rs):.0f}")
    add("unique responses per problem (mean)", lambda rs, m: f"{100 * statistics.mean(dups[m]):.1f}%")
    add("**incorrect responses**", lambda rs, m: str(sum(not r["score"] for r in rs)))
    for c in CATEGORIES:
        add(f"  {c} (share of incorrect)",
            lambda rs, m, c=c: pct(sum(r["category"] == c for r in rs),
                                   sum(not r["score"] for r in rs)))
    add("correct via last-number fallback",
        lambda rs, m: str(sum(r["score"] and r["path"] == "last_number" for r in rs)))
    out += ["## Overview", "", *lines, ""]

    paths = ["boxed", "answer_is", "final_answer_is", "minerva_phrase", "zh_answer",
             "last_number", "none"]
    out += ["## Extraction path of incorrect, non-truncated responses", ""]
    lines = ["| path | " + " | ".join(models) + " |", "|" + "---|" * (len(models) + 1)]
    for p in paths:
        cells = []
        for m in models:
            rs = [r for r in by_model[m] if not r["score"] and r["finish_reason"] != "length"]
            cells.append(f"{sum(r['path'] == p for r in rs)} ({pct(sum(r['path'] == p for r in rs), len(rs))})")
        lines.append(f"| {p} | " + " | ".join(cells) + " |")
    out += [*lines, ""]

    out += ["## By difficulty level", ""]
    lines = ["| model | level | problems | accuracy | truncated | no_box (of incorrect) | "
             "boxed_wrong (of incorrect) |", "|---|---|---|---|---|---|---|"]
    for m in models:
        lv = defaultdict(list)
        for r in by_model[m]:
            lv[r["level"]].append(r)
        for level in sorted(lv, key=str):
            rs = lv[level]
            wrong = sum(not r["score"] for r in rs)
            lines.append(
                f"| {m} | {level} | {len({r['idx'] for r in rs})} | "
                f"{pct(sum(r['score'] for r in rs), len(rs))} | "
                f"{pct(sum(r['finish_reason'] == 'length' for r in rs), len(rs))} | "
                f"{pct(sum(r['category'] == 'no_box' for r in rs), wrong)} | "
                f"{pct(sum(r['category'] == 'boxed_wrong' for r in rs), wrong)} |")
    out += [*lines, ""]
    return "\n".join(out)


def write_samples(path, name, problems, rows, k, seed, tail_chars):
    rng = random.Random(seed)
    by_idx = {d["idx"]: d for d in problems}
    out = [f"# Incorrect responses of {name}: random samples", "",
           f"Up to {k} per category, seed {seed}. Each shows the last {tail_chars} characters.", ""]
    for c in CATEGORIES:
        pool = [r for r in rows if r["category"] == c]
        pick = rng.sample(pool, min(k, len(pool)))
        out += [f"## {c} ({len(pool)} responses in total)", ""]
        for r in pick:
            d = by_idx[r["idx"]]
            text = d["code"][r["sample"]]
            out += [f"### idx {r['idx']}, sample {r['sample']}, level {r['level']}",
                    f"- ground truth: `{r['gt']}`",
                    f"- extracted: `{r['pred']}` (path: {r['path']}, finish: {r['finish_reason']})",
                    f"- question: {d.get('question', '')[:300]}", "", "```",
                    ("..." if len(text) > tail_chars else "") + text[-tail_chars:], "```", ""]
    with open(path, "w") as f:
        f.write("\n".join(out))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="NAME=glob of result .jsonl files (quote it)")
    ap.add_argument("--out_dir", default="analysis_out")
    ap.add_argument("--samples", type=int, default=20, help="samples per category per model")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tail_chars", type=int, default=1200)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    all_rows, dups = [], {}
    for run in args.runs:
        name, pattern = run.split("=", 1)
        problems = load(pattern)
        rows, dup = analyze(name, problems)
        all_rows += rows
        dups[name] = dup
        write_samples(os.path.join(args.out_dir, f"samples_{name}.md"), name, problems, rows,
                      args.samples, args.seed, args.tail_chars)

    text = summary_tables(all_rows, dups)
    print(text)
    with open(os.path.join(args.out_dir, "summary.md"), "w") as f:
        f.write(text)
    with open(os.path.join(args.out_dir, "responses.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nWrote summary.md, responses.csv and samples_*.md to {args.out_dir}")


if __name__ == "__main__":
    main()
