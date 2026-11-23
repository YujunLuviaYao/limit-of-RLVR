"""Summarise the manual labels (Phase 2, RQ1): error composition per model, with intervals.

Inputs
    --labels      final labels (item_id,label,...), e.g. annotation/labels/b1_final.csv
    --key         key.csv of the same batch (item_id -> model, source, idx, sample, prelabel)
    --raw         per-annotator label files (for agreement on the shared items), optional
    runs          NAME=glob of the result files the batch was drawn from (for pool sizes);
                  NAME must match the model names in key.csv

What is reported
    1. counts and shares of A/B/C/D per model x sampling source, with Wilson 95% intervals
       (unweighted: share among the sampled items);
    2. population estimate per model for the RANDOM strata (no_box, boxed_random): each item
       is weighted by (#responses of that category for that problem) / (#items sampled from
       that problem), because at most --per_problem items were drawn per problem; the
       weighted share is scaled to the stratum size. The 95% interval resamples problems
       (cluster bootstrap). boxed_mv is a targeted sample and is reported only in table 1;
    3. the implied error composition of ALL originally-wrong responses: truncated (exact
       count) + no_box x shares + boxed_wrong x shares, and the implied pass@1 gain if
       every A/D response were counted correct;
    4. agreement on the items every annotator labelled (percent and Fleiss' kappa);
    5. how often the final label differs from the suggested prelabel.

Usage (math/ directory, evaluation environment active):
    python summarize_annotation.py --labels annotation/labels/b1_final.csv \\
        --key $SCRATCH/rlvr/annotation_b1/key.csv \\
        --raw annotation/labels/b1_yujun.csv annotation/labels/b1_ganlin.csv annotation/labels/b1_haotong.csv \\
        --out annotation/labels/b1_summary.md \\
        'Qwen2.5-7B=$SCRATCH/rlvr/main_out/Qwen2.5-7B/chunk_*/math500/*.jsonl' \\
        'SimpleRL-Zoo-7B=$SCRATCH/rlvr/main_out/SimpleRL-Zoo-7B/chunk_*/math500/*.jsonl'
(batch 1 was drawn from the 200-problem results in main_out/, not main_out_rest300/.)
"""
import argparse
import csv
import math
import os
import sys
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

LABELS = "ABCD"
RANDOM_STRATA = {"no_box": "no_box", "boxed_random": "boxed_wrong"}  # source -> category


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0, c - h), min(1, c + h))


def fleiss(rows, cats=LABELS):
    """rows: list of label lists (same number of raters per item)."""
    n = len(rows[0])
    N = len(rows)
    P = [(sum(r.count(c) ** 2 for c in cats) - n) / (n * (n - 1)) for r in rows]
    pj = [sum(r.count(c) for r in rows) / (N * n) for c in cats]
    Pbar, Pe = sum(P) / N, sum(p * p for p in pj)
    return (Pbar - Pe) / (1 - Pe) if Pe < 1 else 1.0


def pct(x):
    return f"{100 * x:.1f}%"


def read_csv(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def pools(runs):
    """Per model: Counter((category, idx)), total responses, #problems, n per problem."""
    from analyze_responses import category, extraction_path, load
    out = {}
    for run in runs:
        name, pattern = run.split("=", 1)
        probs = load(pattern)
        c, total = Counter(), 0
        for d in probs:
            frs = d.get("finish_reason") or [None] * len(d["code"])
            for t, s, fr in zip(d["code"], d["score"], frs):
                c[(category(bool(s), fr, extraction_path(t)), d["idx"])] += 1
                total += 1
        out[name] = (c, total, len(probs))
    return out


def weighted_share(items, pool_c, cat, B, rng):
    """Hajek estimator of label shares in a stratum + cluster bootstrap over problems."""
    per_prob = defaultdict(list)
    for it in items:
        per_prob[it["idx"]].append(it["label"])
    probs = sorted(per_prob)

    def est(sel):
        w = Counter()
        for p in sel:
            labs = per_prob[p]
            wt = pool_c[(cat, p)] / len(labs) if pool_c[(cat, p)] else 1.0
            for L in labs:
                w[L] += wt
        tot = sum(w.values())
        return np.array([w[L] / tot if tot else np.nan for L in LABELS])

    point = est(probs)
    if B and len(probs) > 1:
        bs = np.array([est([probs[i] for i in rng.integers(0, len(probs), len(probs))])
                       for _ in range(B)])
        lo, hi = np.nanpercentile(bs, 2.5, axis=0), np.nanpercentile(bs, 97.5, axis=0)
    else:
        lo = hi = point * np.nan
    return point, lo, hi


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", help="NAME=glob of the results the batch was drawn from")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--raw", nargs="*", default=[])
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    final = {r["item_id"]: r for r in read_csv(args.labels)}
    key = {r["item_id"]: r for r in read_csv(args.key)}
    missing = [i for i in final if i not in key]
    if missing:
        raise SystemExit(f"{len(missing)} labelled items are not in key.csv, e.g. {missing[:3]}")
    items = [dict(key[i], label=final[i]["label"], idx=int(key[i]["idx"])) for i in final]
    models = sorted({it["model"] for it in items})
    out = []

    # 1. unweighted
    out += ["## 1. Labels per model and sampling source (share of sampled items, Wilson 95% CI)", "",
            "| model | source | n | " + " | ".join(LABELS) + " |", "|---|---|---|" + "---|" * 4]
    for m in models:
        for src in ("no_box", "boxed_random", "boxed_mv", "ALL"):
            rs = [it for it in items if it["model"] == m and (src == "ALL" or it["source"] == src)]
            if not rs:
                continue
            c = Counter(it["label"] for it in rs)
            cells = []
            for L in LABELS:
                lo, hi = wilson(c[L], len(rs))
                cells.append(f"{c[L]} ({pct(c[L]/len(rs))}, {pct(lo)}–{pct(hi)})")
            out.append(f"| {m} | {src} | {len(rs)} | " + " | ".join(cells) + " |")
    out.append("")

    # 2./3. weighted population estimates
    if args.runs:
        P = pools(args.runs)
        unknown = [m for m in models if m not in P]
        if unknown:
            raise SystemExit(f"model names {unknown} from key.csv have no NAME=glob argument")
        out += ["## 2. Population estimate, random strata (weighted by problem, 95% cluster-bootstrap CI)", "",
                "| model | stratum | responses in stratum | " + " | ".join(LABELS) + " |",
                "|---|---|---|" + "---|" * 4]
        comp = {}
        for m in models:
            pool_c, total, nprob = P[m]
            comp[m] = {"total": total, "nprob": nprob,
                       "truncated": sum(v for (c, _), v in pool_c.items() if c == "truncated"),
                       "correct": sum(v for (c, _), v in pool_c.items() if c == "correct")}
            for src, cat in RANDOM_STRATA.items():
                rs = [it for it in items if it["model"] == m and it["source"] == src]
                size = sum(v for (c, _), v in pool_c.items() if c == cat)
                if not rs:
                    continue
                p, lo, hi = weighted_share(rs, pool_c, cat, args.bootstrap, rng)
                comp[m][cat] = (size, p, lo, hi)
                out.append(f"| {m} | {cat} | {size} | " + " | ".join(
                    f"{pct(a)} ({pct(b)}–{pct(c)})" for a, b, c in zip(p, lo, hi)) + " |")
        out += ["", "## 3. Implied composition of all originally-wrong responses", "",
                "Estimated number of responses (share of all responses). A+D = estimated false "
                "negatives of the original grader.", "",
                "| model | responses | orig. correct | truncated | A | B | C | D | A+D (FN) | pass@1 orig → if A+D counted |",
                "|---|---|---|---|---|---|---|---|---|---|"]
        for m in models:
            c = comp[m]
            est = np.zeros(4)
            for cat in RANDOM_STRATA.values():
                if cat in c:
                    est += c[cat][0] * c[cat][1]
            T = c["total"]
            fn = est[0] + est[3]
            out.append(f"| {m} | {T} | {c['correct']} ({pct(c['correct']/T)}) | "
                       f"{c['truncated']} ({pct(c['truncated']/T)}) | " +
                       " | ".join(f"{e:.0f} ({pct(e/T)})" for e in est) +
                       f" | {fn:.0f} ({pct(fn/T)}) | {pct(c['correct']/T)} → {pct((c['correct']+fn)/T)} |")
        out += ["", "Pass@1 here is the share of correct responses over all problems in the batch "
                "(the same as mean pass@1 because n is equal for every problem).", ""]

    # 4. agreement
    if args.raw:
        raws = [{r["item_id"]: r["label"] for r in read_csv(p)} for p in args.raw]
        shared = sorted(set.intersection(*[set(r) for r in raws]))
        if shared:
            rows = [[r[i] for r in raws] for i in shared]
            agree_all = sum(len(set(r)) == 1 for r in rows)
            pair = []
            for a in range(len(raws)):
                for b in range(a + 1, len(raws)):
                    pair.append(sum(raws[a][i] == raws[b][i] for i in shared) / len(shared))
            out += ["## 4. Agreement on shared items", "",
                    f"{len(shared)} items labelled by all {len(raws)} annotators: all agree on "
                    f"{agree_all} ({pct(agree_all/len(shared))}); mean pairwise agreement "
                    f"{pct(sum(pair)/len(pair))}; Fleiss' kappa {fleiss(rows):.3f}.", ""]
            dis = [(i, r) for i, r in zip(shared, rows) if len(set(r)) > 1]
            for i, r in dis:
                out.append(f"- {i}: {'/'.join(r)} → final {final[i]['label']}")
            out.append("")

    # 5. prelabel
    if all("prelabel" in it for it in items):
        ch = Counter()
        for it in items:
            pre = it["prelabel"].rstrip("?")
            ch[(pre, it["label"])] += 1
        n_diff = sum(v for (a, b), v in ch.items() if a != b)
        out += ["## 5. Suggested prelabel vs final label", "",
                f"final label differs from the prelabel for {n_diff}/{len(items)} items "
                f"({pct(n_diff/len(items))}).", "",
                "| prelabel \\ final | " + " | ".join(LABELS) + " |", "|---|" + "---|" * 4]
        for a in LABELS:
            if any(k[0] == a for k in ch):
                out.append(f"| {a}? | " + " | ".join(str(ch[(a, b)]) for b in LABELS) + " |")
        out.append("")

    txt = "\n".join(out)
    print(txt)
    if args.out:
        with open(args.out, "w") as f:
            f.write("# Annotation summary\n\n" + txt + "\n")
        print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
