"""Draw annotation batch 2: an independent check of the revised grader and of code execution.

Batch 1 was used to write the revised grader's rules, so its 98% agreement is optimistic.
Batch 2 is drawn ONLY from the 300 problems not used in batch 1 (subsets/math500_rest300)
and is stratified by what the revised grader / code execution decided, so that we can
measure precision (are rescued responses really correct?) and recall (how many false
negatives are still missed?) per model. Only responses the ORIGINAL grader marked wrong
are drawn; truncated responses are skipped.

Strata (per response, first match wins):
    R1           rescued by the revised grader from the last non-empty \\boxed{}
    R2           rescued by the revised grader from the stated answer before the code
    tool         still wrong after re-grading, no answer of its own, but its code prints the
                 correct answer ("tool" condition of run_code.py)
    code_only    states a WRONG answer (often a made-up code output) while its code, when run,
                 prints the correct one ("code_any" minus "tool")
    wrong_nobox  wrong under every condition, no non-empty box
    wrong_boxed  wrong under every condition, has a non-empty box
Sizes per model and stratum: --sizes (defaults below); at most --per_problem items per
problem. The stratum sizes in the population are saved (strata.csv) so precision/recall
can be weighted back to all responses.

Annotators see no prelabel and nothing about the grader's decision (to keep the check
independent). For responses with Python code they also see the real output of running it,
and answer one extra question (code_ok, see GUIDELINE.md).

Usage (math/ directory, after slurm/phase2_analysis.sbatch has finished):
    python make_batch2.py --out_dir $SCRATCH/rlvr/annotation_b2 \\
        'Qwen2.5-7B=$SCRATCH/rlvr/phase2/code_exec/tool/Qwen2.5-7B/math500/*.jsonl' \\
        'SimpleRL-Zoo-7B=$SCRATCH/rlvr/phase2/code_exec/tool/SimpleRL-Zoo-7B/math500/*.jsonl'
    python make_annotation_html.py $SCRATCH/rlvr/annotation_b2
Writes the same files as make_annotation_sheet.py (sheet_*.csv, key.csv, items_full.jsonl)
plus strata.csv.
"""
import argparse
import csv
import json
import os
import random
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from analyze_responses import load  # noqa: E402
from make_annotation_sheet import CODE_FENCE, capped_sample  # noqa: E402
from regrade import boxes  # noqa: E402

STRATA = ["R1", "R2", "tool", "code_only", "wrong_nobox", "wrong_boxed"]
DEFAULT_SIZES = ("Qwen2.5-7B:R1=20,R2=30,tool=25,code_only=15,wrong_nobox=25,wrong_boxed=15;"
                 "SimpleRL-Zoo-7B:R1=15,wrong_nobox=5,wrong_boxed=15")


def parse_sizes(s):
    out = defaultdict(dict)
    for part in s.split(";"):
        if part.strip():
            model, spec = part.split(":", 1)
            for kv in spec.split(","):
                k, v = kv.split("=")
                out[model.strip()][k.strip()] = int(v)
    return out


def stratum(d, j):
    if d["score_orig"][j]:
        return None
    if (d.get("finish_reason") or [None] * len(d["code"]))[j] == "length":
        return None
    rule = d["revised_rule"][j]
    if rule in ("R1", "R2"):
        return rule
    if d["score_tool"][j]:
        return "tool"
    if d["score_code_any"][j]:
        return "code_only"
    return "wrong_boxed" if any(b.strip() for b in boxes(d["code"][j])) else "wrong_nobox"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="NAME=glob of code_exec/tool/<model>/math500/*.jsonl")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--subset", default=os.path.join(HERE, "subsets/math500_rest300/math500/test.jsonl"),
                    help="only problems in this file are used (default: the 300 not in batch 1)")
    ap.add_argument("--sizes", default=DEFAULT_SIZES)
    ap.add_argument("--per_problem", type=int, default=2)
    ap.add_argument("--annotators", default="yujun,ganlin,haotong")
    ap.add_argument("--shared", type=int, default=15)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--tail_chars", type=int, default=1500)
    ap.add_argument("--id_prefix", default="b2_")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    keep = {json.loads(l)["idx"] for l in open(args.subset) if l.strip()}
    sizes = parse_sizes(args.sizes)
    items, pop = [], []
    for run in args.runs:
        name, pattern = run.split("=", 1)
        probs = [d for d in load(pattern) if d["idx"] in keep]
        for need in ("score_orig", "revised_rule", "score_tool", "score_code_any", "code_result"):
            if need not in probs[0]:
                raise SystemExit(f"{name}: field {need} missing; pass the output of run_code.py "
                                 "(code_exec/tool/<model>/math500/*.jsonl)")
        pools = defaultdict(list)
        for d in probs:
            for j, t in enumerate(d["code"]):
                s = stratum(d, j)
                if s:
                    pools[s].append({"model": name, "idx": d["idx"], "sample": j, "stratum": s,
                                     "level": d.get("level", ""), "question": d.get("question", ""),
                                     "gt": d.get("gt", ""), "pred": (d.get("pred") or [""] * len(d["code"]))[j], "text": t,
                                     "code_result": d["code_result"][j],
                                     "code_output": d["code_output"][j]})
        print(f"{name}: {len(probs)} problems; pool sizes: "
              + ", ".join(f"{s}={len(pools[s])}" for s in STRATA))
        for s in STRATA:
            want = sizes.get(name, {}).get(s, 0)
            got = capped_sample(pools[s], want, args.per_problem, rng) if want else []
            items += got
            pop.append((name, s, len(pools[s]), len(got), len({r["idx"] for r in pools[s]})))
            if want and len(got) < want:
                print(f"  note: {name}/{s}: only {len(got)} of {want} available")

    rng.shuffle(items)
    for k, it in enumerate(items, 1):
        it["item_id"] = f"{args.id_prefix}{k:04d}"
    annotators = [a.strip() for a in args.annotators.split(",") if a.strip()]
    by = defaultdict(list)
    for it in items:
        by[it["stratum"]].append(it)
    shared = []
    while len(shared) < min(args.shared, len(items)):
        for s in STRATA:
            if by[s] and len(shared) < args.shared:
                shared.append(by[s].pop())
    sid = {it["item_id"] for it in shared}
    for it in shared:
        it["assignee"] = "ALL"
    for k, it in enumerate(sorted((i for i in items if i["item_id"] not in sid),
                                  key=lambda x: x["item_id"])):
        it["assignee"] = annotators[k % len(annotators)]

    os.makedirs(args.out_dir, exist_ok=True)
    cols = ["item_id", "assignee", "idx", "level", "question", "ground_truth", "extracted",
            "before_first_code_block", "response_tail", "prelabel", "prelabel_reason",
            "code_run", "label", "code_ok", "annotator", "notes"]

    def code_run(it):
        if it["code_result"] == "no_code":
            return ""
        out = it["code_output"] or ""
        return out if it["code_result"] in ("correct", "wrong") else f"[运行失败] {out}"

    def row(it):
        before = it["text"].split(CODE_FENCE)[0]
        tail = it["text"][-args.tail_chars:]
        return {"item_id": it["item_id"], "assignee": it["assignee"], "idx": it["idx"],
                "level": it["level"], "question": it["question"], "ground_truth": it["gt"],
                "extracted": it["pred"],
                "before_first_code_block": ("..." if len(before) > 800 else "") + before[-800:],
                "response_tail": ("..." if len(it["text"]) > args.tail_chars else "") + tail,
                "prelabel": "", "prelabel_reason": "", "code_run": code_run(it),
                "label": "", "code_ok": "", "annotator": "", "notes": ""}

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
        w.writerow(["item_id", "model", "source", "idx", "sample", "category", "prelabel",
                    "code_result"])
        for it in sorted(items, key=lambda x: x["item_id"]):
            w.writerow([it["item_id"], it["model"], it["stratum"], it["idx"], it["sample"],
                        it["stratum"], "", it["code_result"]])
    with open(os.path.join(args.out_dir, "strata.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "stratum", "responses_in_population", "sampled", "problems_in_population"])
        w.writerows(pop)
    with open(os.path.join(args.out_dir, "items_full.jsonl"), "w", encoding="utf-8") as f:
        for it in sorted(items, key=lambda x: x["item_id"]):
            f.write(json.dumps({"item_id": it["item_id"], "question": it["question"],
                                "ground_truth": it["gt"], "response": it["text"],
                                "code_run": code_run(it)}, ensure_ascii=False) + "\n")

    print(f"\n{len(items)} items ({len(shared)} shared). By model and stratum:")
    for (m, s), c in sorted(Counter((it["model"], it["stratum"]) for it in items).items()):
        print(f"  {m:18s} {s:12s} {c}")
    print("Items with code output shown:", sum(bool(code_run(it)) for it in items))
    print("Per annotator:", {a: sum(it["assignee"] in ("ALL", a) for it in items) for a in annotators})
    print(f"Wrote sheet_*.csv, key.csv, strata.csv, items_full.jsonl to {args.out_dir}")


if __name__ == "__main__":
    main()
