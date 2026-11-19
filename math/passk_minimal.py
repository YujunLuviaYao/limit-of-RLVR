"""Compute (and optionally plot) unbiased pass@k from math_eval.py output files.

Usage:
    python passk_minimal.py [--plot out.png] NAME=path/to/results.jsonl [NAME2=...]

A path may be a glob (quote it), e.g. 'NAME=main_out/Qwen2.5-7B/chunk_*/math500/*.jsonl',
to combine the chunk files written by run_main.sh. Problems are de-duplicated by `idx`.

Each jsonl line produced by math_eval.py has a `score` field: a list of n bools,
one per sampled response. pass@k uses the unbiased estimator from Chen et al. (2021):
    pass@k = 1 - C(n-c, k) / C(n, k)
"""
import argparse
import glob
import json

import numpy as np


def pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def load_samples(pattern):
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"No files match {pattern}")
    samples = {}
    for path in paths:
        with open(path) as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    samples[d["idx"]] = d
    return [samples[i] for i in sorted(samples)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="NAME=path/to/results.jsonl")
    ap.add_argument("--plot", default=None, help="save a pass@k plot to this path")
    args = ap.parse_args()

    curves = {}
    for run in args.runs:
        name, path = run.split("=", 1)
        samples = load_samples(path)
        scores = [[bool(x) for x in d["score"]] for d in samples]
        n = min(len(s) for s in scores)
        ks = [k for k in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024) if k <= n]
        correct = [sum(s[:n]) for s in scores]
        curve = {k: float(np.mean([pass_at_k(n, c, k) for c in correct])) for k in ks}
        curves[name] = curve

        print(f"\n== {name}  ({len(scores)} problems, n={n}) ==")
        print("  " + "  ".join(f"pass@{k}={v:.3f}" for k, v in curve.items()))
        print(f"  #correct per problem: {correct}")
        print(f"  never solved: {sum(c == 0 for c in correct)}/{len(correct)}")
        reasons = [r for d in samples for r in d.get("finish_reason", [])]
        if reasons:
            n_len = sum(r == "length" for r in reasons)
            print(f"  truncated (finish_reason=length): {n_len}/{len(reasons)} responses")

    if args.plot:
        try:
            import matplotlib
        except ImportError:
            print("\nmatplotlib not installed; skipping plot (pip install matplotlib)")
            return
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 3.5))
        for name, curve in curves.items():
            ax.plot(list(curve), list(curve.values()), marker="o", label=name)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("k")
        ax.set_ylabel("pass@k")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        print(f"\nPlot saved to {args.plot}")


if __name__ == "__main__":
    main()
