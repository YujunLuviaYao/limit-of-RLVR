"""Draw the responses to annotate by hand and write them as spreadsheets (Phase 2, RQ1).

Only responses the ORIGINAL grader marked incorrect are sampled. Per model:
    no_box         --no_box (default 100) random responses whose answer did not come from
                   \\boxed{} (see analyze_responses.py for the categories)
    boxed_random   --boxed_random (default 50) random boxed_wrong responses
    boxed_mv       up to --boxed_mv (default 100) boxed_wrong responses that math-verify
                   judges CORRECT on the saved text: candidates for "equivalent answer the
                   original grader failed to match"
Random draws take at most --per_problem (default 2) responses from any one problem, because
responses to the same problem tend to fail the same way.

Each item gets a SUGGESTED label (prelabel) from simple rules, to speed up annotation. It is a
hint only: annotators must check every item, and the report should say how often the final
label differs from the suggestion.
    A?  the ground truth appears in the text before the first code block (string match or
        math-verify on that text)
    B?  contains a Python code block and no A? evidence
    C?  long repetition in the tail, or nothing above
    D?  (boxed items) math-verify accepts the boxed answer
Label definitions: annotation/GUIDELINE.md.

Usage (math/ directory, evaluation environment active, no GPU needed):
    python make_annotation_sheet.py \\
        'Qwen2.5-7B=$SCRATCH/rlvr/main_out*/Qwen2.5-7B/chunk_*/math500/*.jsonl' \\
        'SimpleRL-Zoo-7B=$SCRATCH/rlvr/main_out*/SimpleRL-Zoo-7B/chunk_*/math500/*.jsonl' \\
        --out_dir $SCRATCH/rlvr/annotation

Batches: you can annotate in two batches, e.g. first from the 200-problem results and later
from the remaining 300, with proportional sizes (40% / 60%), different --id_prefix and seed,
and --shared 0 for the second batch. Problems in each batch keep their MATH500 idx.

Writes to --out_dir:
    sheet_all.csv            every item (the 'assignee' column says who labels it)
    sheet_<annotator>.csv    the shared items plus that annotator's own items
    key.csv                  item_id -> model and sampling source (kept out of the sheets so
                             annotators do not see which model wrote a response)
    items_full.jsonl         full response text per item, for items whose tail is not enough
The CSVs are UTF-8 with BOM so Excel shows the math symbols correctly; Google Sheets opens
them directly (File > Import).
"""
import argparse
import csv
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from analyze_responses import category, extraction_path, load  # noqa: E402

CODE_FENCE = "```"


def norm(s):
    s = s or ""
    for t in ["\\left", "\\right", "\\!", "\\,", "\\;", "$", " ", "\n", "\\displaystyle"]:
        s = s.replace(t, "")
    return s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")


def mv_verify(gt, text):
    try:
        from math_verify import parse, verify
        return bool(verify(parse("$" + gt + "$"), parse(text)))
    except Exception:
        return False


def repetitive(text, span=30, times=5):
    tail = text[-1500:]
    if len(tail) < span * times:
        return False
    counts = Counter(tail[i:i + span] for i in range(0, len(tail) - span, 10))
    return counts.most_common(1)[0][1] >= times


def gt_in(gt, text):
    """Ground truth occurs in text as a whole token (so '10' does not match '100' or 'x_{10}')."""
    g, t = norm(gt), norm(text)
    if not g:
        return False
    pat = re.escape(g)
    if re.fullmatch(r"-?[\d.]+", g):
        pat = r"(?<![\d._{^])" + pat + r"(?![\d.]*\d)"
    return re.search(pat, t) is not None


def prelabel_no_box(text, gt):
    before = text.split(CODE_FENCE)[0]
    tail = before[-400:]  # where a conclusion before the code would be
    if gt and (gt_in(gt, tail) or mv_verify(gt, tail)):
        return "A?", "ground truth appears near the end of the text before the first code block"
    if "```python" in text:
        return "B?", "has a Python code block, no final answer found before it"
    if repetitive(text):
        return "C?", "long repetition in the tail"
    return "C?", "no ground truth found, no code"


def capped_sample(pool, n, cap, rng):
    pool = pool[:]
    rng.shuffle(pool)
    taken, per = [], Counter()
    for r in pool:
        if per[r["idx"]] < cap:
            taken.append(r)
            per[r["idx"]] += 1
            if len(taken) == n:
                break
    return taken


def responses(name, problems):
    out = []
    for d in problems:
        reasons = d.get("finish_reason") or [None] * len(d["code"])
        preds = d.get("pred") or [None] * len(d["code"])
        for j, (text, s, fr, pred) in enumerate(zip(d["code"], d["score"], reasons, preds)):
            cat = category(bool(s), fr, extraction_path(text))
            if cat in ("no_box", "boxed_wrong"):
                out.append({"model": name, "idx": d["idx"], "sample": j,
                            "level": d.get("level", ""), "question": d.get("question", ""),
                            "gt": d.get("gt", ""), "pred": pred, "text": text,
                            "finish_reason": fr, "category": cat})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="NAME=glob of result .jsonl files (quote it)")
    ap.add_argument("--out_dir", default="annotation_out")
    ap.add_argument("--no_box", type=int, default=100)
    ap.add_argument("--boxed_random", type=int, default=50)
    ap.add_argument("--boxed_mv", type=int, default=100)
    ap.add_argument("--per_problem", type=int, default=2)
    ap.add_argument("--annotators", default="annotator1,annotator2,annotator3")
    ap.add_argument("--shared", type=int, default=20, help="items every annotator labels")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tail_chars", type=int, default=1500)
    ap.add_argument("--id_prefix", default="item",
                    help="prefix of item ids; use a different one per batch (e.g. b1, b2)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    items = []
    for run in args.runs:
        name, pattern = run.split("=", 1)
        rs = responses(name, load(pattern))
        no_box = [r for r in rs if r["category"] == "no_box"]
        boxed = [r for r in rs if r["category"] == "boxed_wrong"]
        print(f"{name}: {len(no_box)} no_box, {len(boxed)} boxed_wrong responses")

        for r in capped_sample(no_box, args.no_box, args.per_problem, rng):
            items.append(dict(r, source="no_box"))
        rand = capped_sample(boxed, args.boxed_random, args.per_problem, rng)
        for r in rand:
            items.append(dict(r, source="boxed_random"))

        print(f"  running math-verify on {len(boxed)} boxed_wrong responses ...", flush=True)
        cands = []
        for i, r in enumerate(boxed):
            if mv_verify(r["gt"], r["text"]):
                cands.append(r)
            if (i + 1) % 2000 == 0:
                print(f"    {i + 1}/{len(boxed)}", flush=True)
        seen = {(r["idx"], r["sample"]) for r in rand}
        cands = [r for r in cands if (r["idx"], r["sample"]) not in seen]
        print(f"  {len(cands)} boxed_wrong responses accepted by math-verify (candidates)")
        for r in capped_sample(cands, args.boxed_mv, max(args.per_problem, 3), rng):
            items.append(dict(r, source="boxed_mv"))

    # suggested labels
    for it in items:
        if it["source"] == "no_box":
            it["prelabel"], it["reason"] = prelabel_no_box(it["text"], it["gt"])
        elif mv_verify(it["gt"], it["text"]):
            it["prelabel"], it["reason"] = "D?", "math-verify accepts the boxed answer"
        else:
            it["prelabel"], it["reason"] = "C?", "boxed answer differs from the ground truth"

    # blind order, ids, assignment
    rng.shuffle(items)
    for k, it in enumerate(items, 1):
        it["item_id"] = f"{args.id_prefix}{k:04d}"
    annotators = [a.strip() for a in args.annotators.split(",") if a.strip()]
    by_source = defaultdict(list)
    for it in items:
        by_source[it["source"]].append(it)
    shared = []
    while len(shared) < min(args.shared, len(items)):  # spread shared items over sources
        for src in sorted(by_source):
            if by_source[src] and len(shared) < args.shared:
                shared.append(by_source[src].pop())
    shared_ids = {it["item_id"] for it in shared}
    rest = [it for it in items if it["item_id"] not in shared_ids]
    for it in shared:
        it["assignee"] = "ALL"
    for k, it in enumerate(rest):
        it["assignee"] = annotators[k % len(annotators)]

    os.makedirs(args.out_dir, exist_ok=True)
    cols = ["item_id", "assignee", "idx", "level", "question", "ground_truth", "extracted",
            "before_first_code_block", "response_tail", "prelabel", "prelabel_reason",
            "label", "annotator", "notes"]

    def row(it):
        before = it["text"].split(CODE_FENCE)[0]
        tail = it["text"][-args.tail_chars:]
        return {"item_id": it["item_id"], "assignee": it["assignee"], "idx": it["idx"],
                "level": it["level"], "question": it["question"], "ground_truth": it["gt"],
                "extracted": it["pred"],
                "before_first_code_block": ("..." if len(before) > 800 else "") + before[-800:],
                "response_tail": ("..." if len(it["text"]) > args.tail_chars else "") + tail,
                "prelabel": it["prelabel"], "prelabel_reason": it["reason"],
                "label": "", "annotator": "", "notes": ""}

    def write(path, its):
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for it in sorted(its, key=lambda x: x["item_id"]):
                w.writerow(row(it))

    write(os.path.join(args.out_dir, "sheet_all.csv"), items)
    for a in annotators:
        write(os.path.join(args.out_dir, f"sheet_{a}.csv"),
              [it for it in items if it["assignee"] in ("ALL", a)])
    with open(os.path.join(args.out_dir, "key.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "model", "source", "idx", "sample", "category", "prelabel"])
        for it in sorted(items, key=lambda x: x["item_id"]):
            w.writerow([it["item_id"], it["model"], it["source"], it["idx"], it["sample"],
                        it["category"], it["prelabel"]])
    with open(os.path.join(args.out_dir, "items_full.jsonl"), "w", encoding="utf-8") as f:
        for it in sorted(items, key=lambda x: x["item_id"]):
            f.write(json.dumps({"item_id": it["item_id"], "question": it["question"],
                                "ground_truth": it["gt"], "response": it["text"]},
                               ensure_ascii=False) + "\n")

    print(f"\n{len(items)} items ({len(shared)} shared by everyone). By model and source:")
    for (m, s), c in sorted(Counter((it["model"], it["source"]) for it in items).items()):
        print(f"  {m:20s} {s:13s} {c}")
    print("Suggested labels:", dict(Counter(it["prelabel"] for it in items)))
    print("Per annotator:", {a: sum(it["assignee"] in ("ALL", a) for it in items) for a in annotators})
    print(f"Wrote sheet_all.csv, sheet_<annotator>.csv, key.csv, items_full.jsonl to {args.out_dir}")


if __name__ == "__main__":
    main()
