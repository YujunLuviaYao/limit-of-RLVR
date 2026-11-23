"""Compute (and optionally plot) unbiased pass@k from math_eval.py output files.

Usage:
    python passk_minimal.py [--plot out.png] [--bootstrap 2000] NAME=glob [NAME2=glob ...]

A path may be a glob (quote it), e.g. 'NAME=main_out/Qwen2.5-7B/chunk_*/math500/*.jsonl',
to combine the chunk files written by run_main.sh. Problems are de-duplicated by `idx`.
NAME is used as the legend label and may contain spaces, e.g. "Base (Qwen2.5-7B)=...".

Each jsonl line produced by math_eval.py has a `score` field: a list of n bools,
one per sampled response. pass@k uses the unbiased estimator from Chen et al. (2021):
    pass@k = 1 - C(n-c, k) / C(n, k)

--bootstrap B resamples PROBLEMS with replacement B times and reports percentile
confidence intervals (--ci, default 95%) for each model's pass@k, drawn as shaded bands.
With exactly two models evaluated on the same problems it also reports a paired interval
for the difference (first model minus second) at each k, using the same resampled problems
for both models; a difference whose interval excludes 0 is marked with *.
The interval covers sampling of problems only; the randomness of the n responses per
problem is already averaged by the unbiased estimator and is not added on top.
"""
import argparse
import csv
import glob
import json
import os

import numpy as np

KS_ALL = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)


def pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def load_samples(pattern):
    paths = sorted(glob.glob(os.path.expandvars(pattern)))
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
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="NAME=glob of result .jsonl files")
    ap.add_argument("--plot", default=None, help="save a pass@k plot (.png or .pdf)")
    ap.add_argument("--bootstrap", type=int, default=0, help="bootstrap resamples (0 = off)")
    ap.add_argument("--ci", type=float, default=95.0, help="confidence level in percent")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--title", default=None, help="plot title")
    ap.add_argument("--csv", default=None, help="write pass@k (and intervals) to this CSV")
    ap.add_argument("--quiet", action="store_true", help="do not print per-problem counts")
    args = ap.parse_args()

    models = []  # (name, idx list, per-problem pass@k matrix [problems x ks], ks)
    for run in args.runs:
        name, path = run.split("=", 1)
        samples = load_samples(path)
        scores = [[bool(x) for x in d["score"]] for d in samples]
        n = min(len(s) for s in scores)
        ks = [k for k in KS_ALL if k <= n]
        correct = [sum(s[:n]) for s in scores]
        mat = np.array([[pass_at_k(n, c, k) for k in ks] for c in correct])
        models.append((name, [d["idx"] for d in samples], mat, ks))

        print(f"\n== {name}  ({len(scores)} problems, n={n}) ==")
        print("  " + "  ".join(f"pass@{k}={v:.3f}" for k, v in zip(ks, mat.mean(0))))
        if not args.quiet:
            print(f"  #correct per problem: {correct}")
        print(f"  never solved: {sum(c == 0 for c in correct)}/{len(correct)}")
        reasons = [r for d in samples for r in d.get("finish_reason", [])]
        if reasons:
            n_len = sum(r == "length" for r in reasons)
            print(f"  truncated (finish_reason=length): {n_len}/{len(reasons)} responses")

    ks = min((m[3] for m in models), key=len)
    nk = len(ks)
    lo_q, hi_q = (100 - args.ci) / 2, 100 - (100 - args.ci) / 2
    bands, diff = {}, None
    if args.bootstrap > 0:
        rng = np.random.default_rng(args.seed)
        same = len(models) == 2 and models[0][1] == models[1][1]
        n_prob = len(models[0][1])
        print(f"\nBootstrap: {args.bootstrap} resamples of problems, {args.ci:g}% percentile intervals")
        if same:  # paired: resample the same problems for both models
            idx = rng.integers(0, n_prob, size=(args.bootstrap, n_prob))
            boots = [m[2][:, :nk][idx].mean(1) for m in models]
        else:
            boots = []
            for m in models:
                i = rng.integers(0, len(m[1]), size=(args.bootstrap, len(m[1])))
                boots.append(m[2][:, :nk][i].mean(1))
        for m, b in zip(models, boots):
            bands[m[0]] = (np.percentile(b, lo_q, axis=0), np.percentile(b, hi_q, axis=0))
            print(f"  {m[0]}: " + "  ".join(
                f"pass@{k}={v:.3f} [{l:.3f},{h:.3f}]"
                for k, v, l, h in zip(ks, m[2][:, :nk].mean(0), *bands[m[0]])))
        if same:
            d = boots[0] - boots[1]
            point = models[0][2][:, :nk].mean(0) - models[1][2][:, :nk].mean(0)
            dl, dh = np.percentile(d, lo_q, axis=0), np.percentile(d, hi_q, axis=0)
            diff = (point, dl, dh)
            print(f"\n  Paired difference ({models[0][0]} minus {models[1][0]}), "
                  f"* = interval excludes 0:")
            for k, p, l, h in zip(ks, point, dl, dh):
                star = "*" if (l > 0 or h < 0) else " "
                print(f"    k={k:<4d} {p:+.3f} [{l:+.3f}, {h:+.3f}] {star}")
        elif len(models) == 2:
            print("  (models cover different problems: no paired difference)")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            head = ["k"]
            for m in models:
                head += [m[0]] + ([f"{m[0]} lo", f"{m[0]} hi"] if bands else [])
            if diff:
                head += ["diff", "diff lo", "diff hi"]
            w.writerow(head)
            for j, k in enumerate(ks):
                row = [k]
                for m in models:
                    row.append(round(float(m[2][:, j].mean()), 4))
                    if bands:
                        row += [round(float(bands[m[0]][0][j]), 4), round(float(bands[m[0]][1][j]), 4)]
                if diff:
                    row += [round(float(x[j]), 4) for x in diff]
                w.writerow(row)
        print(f"\nTable saved to {args.csv}")

    if args.plot:
        try:
            import matplotlib
        except ImportError:
            print("\nmatplotlib not installed; skipping plot (pip install matplotlib)")
            return
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams.update({"font.size": 12, "axes.labelsize": 13, "legend.fontsize": 11})
        fig, ax = plt.subplots(figsize=(6, 4.2))
        markers = ["o", "s", "^", "D", "v"]
        for i, m in enumerate(models):
            y = m[2][:, :nk].mean(0)
            line, = ax.plot(ks, y, marker=markers[i % len(markers)], linewidth=2, markersize=6,
                            label=m[0])
            if m[0] in bands:
                ax.fill_between(ks, bands[m[0]][0], bands[m[0]][1], color=line.get_color(),
                                alpha=0.2, linewidth=0)
        ax.set_xscale("log", base=2)
        ax.set_xticks(ks)
        ax.set_xticklabels([str(k) for k in ks])
        ax.set_xlabel("k (number of samples)")
        ax.set_ylabel("pass@k")
        if args.title:
            ax.set_title(args.title)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right")
        if bands:
            ax.text(0.02, 0.97, f"shaded: {args.ci:g}% bootstrap CI over problems",
                    transform=ax.transAxes, va="top", fontsize=9, color="0.35")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=300 if args.plot.endswith(".png") else None)
        print(f"\nPlot saved to {args.plot}")


if __name__ == "__main__":
    main()
