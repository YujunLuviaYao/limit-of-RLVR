"""Draw a difficulty-stratified subset of MATH500 for the main experiment.

Problems are sampled from each difficulty level (1-5) in proportion to the level's
share of MATH500 (largest-remainder rounding), with a fixed seed. Each sampled problem
keeps its original MATH500 index as `idx`, so results can be traced back to the full set.

Usage:
    python make_subset.py                       # 200 problems, seed 0
    python make_subset.py --size 100 --seed 1
    python make_subset.py --rest_of subsets/math500_strat200_seed0   # every problem NOT in it

Writes <out_root>/math500/test.jsonl, which math_eval.py reads via --data_dir <out_root>.
"""
import argparse
import collections
import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "examples/math_eval/data/math500/test.jsonl")


def allocate(counts, size):
    total = sum(counts.values())
    quotas = {k: size * v / total for k, v in counts.items()}
    alloc = {k: int(q) for k, q in quotas.items()}
    leftover = size - sum(alloc.values())
    for k in sorted(quotas, key=lambda k: quotas[k] - alloc[k], reverse=True)[:leftover]:
        alloc[k] += 1
    return alloc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_root", default=None)
    ap.add_argument("--rest_of", default=None,
                    help="subset dir: write all MATH500 problems not in it (no sampling)")
    args = ap.parse_args()

    with open(SRC) as f:
        problems = [dict(json.loads(line), idx=i) for i, line in enumerate(f)]

    if args.rest_of:
        with open(os.path.join(args.rest_of, "math500", "test.jsonl")) as f:
            taken = {json.loads(line)["idx"] for line in f if line.strip()}
        rest = [p for p in problems if p["idx"] not in taken]
        out_root = args.out_root or os.path.join(HERE, "subsets", f"math500_rest{len(rest)}")
        os.makedirs(os.path.join(out_root, "math500"), exist_ok=True)
        out_file = os.path.join(out_root, "math500", "test.jsonl")
        with open(out_file, "w") as f:
            for p in rest:
                f.write(json.dumps(p) + "\n")
        levels = collections.Counter(p["level"] for p in rest)
        print(f"Wrote {len(rest)} problems (all MATH500 problems not in {args.rest_of}) to {out_file}")
        for lvl in sorted(levels):
            print(f"  level {lvl}: {levels[lvl]}")
        return

    by_level = collections.defaultdict(list)
    for p in problems:
        by_level[p["level"]].append(p)
    alloc = allocate({lvl: len(ps) for lvl, ps in by_level.items()}, args.size)

    rng = random.Random(args.seed)
    subset = []
    for lvl in sorted(by_level):
        subset += rng.sample(by_level[lvl], alloc[lvl])
    subset.sort(key=lambda p: p["idx"])

    out_root = args.out_root or os.path.join(
        HERE, "subsets", f"math500_strat{args.size}_seed{args.seed}"
    )
    os.makedirs(os.path.join(out_root, "math500"), exist_ok=True)
    out_file = os.path.join(out_root, "math500", "test.jsonl")
    with open(out_file, "w") as f:
        for p in subset:
            f.write(json.dumps(p) + "\n")

    print(f"Wrote {len(subset)} problems to {out_file}")
    for lvl in sorted(by_level):
        print(f"  level {lvl}: {alloc[lvl]} of {len(by_level[lvl])}")


if __name__ == "__main__":
    main()
