"""Task 8: how much can a 296-row leaderboard tell us? Uses run-1 OOF predictions only, never test.

  python leaderboard_noise.py
  python leaderboard_noise.py --draws 5000 --lb 41.93
  python leaderboard_noise.py --compare submissions/v1/submission_v1.csv submissions/v2/submission_v2.csv

Score = 50 x clarity macro-F1 + 50 x evasion macro-F1 (sklearn semantics: macro over the classes
present in the drawn gold labels or the predictions). Each draw is a 296-row bootstrap sample of the
train rows with the train class mix (stratified by evasion label, with replacement).
1. Distribution of v1's score: is the leaderboard 41.93 plausible as sampling noise?
2. Paired differences between v1 and other ensembles on the same draws: what gap is real?
3. Which classes drive the variance, and how often a class's F1 is exactly 0.
"""
import argparse, glob, os, re
from collections import defaultdict
import numpy as np
from common import load, EVASION_LABELS, CLARITY_LABELS, EVASION_TO_CLARITY

ap = argparse.ArgumentParser()
ap.add_argument("--runs", default="results/run1")
ap.add_argument("--draws", type=int, default=5000)
ap.add_argument("--n", type=int, default=296, help="rows per draw (test size)")
ap.add_argument("--lb", type=float, default=41.93, help="observed leaderboard score of v1")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--compare", nargs=2, metavar=("A.csv", "B.csv"),
                help="only report how often two submission files agree (predicted labels only), then exit")
args = ap.parse_args()

if args.compare:  # compares two of our own prediction files; no test answers involved
    import pandas as pd
    a, b = (pd.read_csv(f).set_index(["subtaskID", "datapointID"]).answer for f in args.compare)
    assert a.index.sort_values().equals(b.index.sort_values()), "the files cover different rows"
    for t, name in [(1, "clarity"), (2, "evasion")]:
        print(f"{name}: the two files agree on {(a.loc[t] == b.loc[t].reindex(a.loc[t].index)).mean():.1%} of rows")
    raise SystemExit

tr, _ = load()
y_ev = np.array([EVASION_LABELS.index(l) for l in tr.evasion_label])
EV2CL = np.array([CLARITY_LABELS.index(EVASION_TO_CLARITY[l]) for l in EVASION_LABELS])
y_cl = EV2CL[y_ev]


def per_class_f1(y, p, k):
    """Per-class F1 and which classes count for sklearn's macro average (present in y or p)."""
    cm = np.bincount(y * k + p, minlength=k * k).reshape(k, k)
    d = cm.sum(0) + cm.sum(1)
    return np.where(d > 0, 2 * np.diag(cm) / np.maximum(d, 1), 0.0), d > 0


def macro(y, p, k):
    f, present = per_class_f1(y, p, k)
    return f[present].mean()


def clarity_combined(pe, pc):  # evaluate.py's "combined" clarity
    return ((np.stack([pe[:, EV2CL == c].sum(1) for c in range(3)], 1) + pc) / 2).argmax(1)


def f1sum(pe, pc):
    cl, ev = clarity_combined(pe, pc), pe.argmax(1)
    return macro(y_cl, cl, 3) + macro(y_ev, ev, 9)


# ---- best-epoch OOF probabilities per config, as in evaluate.py
groups = defaultdict(list)
for f in glob.glob(os.path.join(args.runs, "*_f[0-9]_s*.npz")):
    groups[re.sub(r"_f\d_s\d+\.npz$", "", os.path.basename(f))].append(f)
P = {}
for c, files in sorted(groups.items()):
    runs = [np.load(f) for f in sorted(files)]
    best = None
    for e in range(sum(k.startswith("val_pe") for k in runs[0].files)):
        pe, pc = np.zeros((len(tr), 9)), np.zeros((len(tr), 3))
        for r in runs:
            pe[r["val_idx"]], pc[r["val_idx"]] = r[f"val_pe{e}"], r[f"val_pc{e}"]
        s = max(macro(y_cl, clarity_combined(pe, pc), 3), macro(y_cl, EV2CL[pe.argmax(1)], 3), macro(y_cl, pc.argmax(1), 3)) + macro(y_ev, pe.argmax(1), 9)
        if best is None or s > best[0]:
            best = (s, pe, pc)
    P[c] = best[1:]


def ens(weights):
    tot = sum(weights.values())
    return (sum(w * P[c][0] for c, w in weights.items()) / tot, sum(w * P[c][1] for c, w in weights.items()) / tot)


X, R, B = "xlm-roberta-large_p0", "RoBERT-large_p0", "bert-base-romanian-cased-v1_p0"
ENS = {  # predicted labels (clarity, evasion) for every train row
    "v1": ens({X: 1, R: 1, B: 1}),
    "greedy (XLM-R x3 + RoBERT + RO-BERT)": ens({X: 3, R: 1, B: 1}),
    "all 4 run-1 configs": ens({c: 1 for c in P}),
    "XLM-R large alone": ens({X: 1}),
    "RO-BERT base alone": ens({B: 1}),
}
PRED = {k: (clarity_combined(pe, pc), pe.argmax(1)) for k, (pe, pc) in ENS.items()}

# ---- stratified bootstrap draws with the train class mix
rng = np.random.default_rng(args.seed)
by_class = [np.flatnonzero(y_ev == k) for k in range(9)]
quota = np.round(np.bincount(y_ev, minlength=9) / len(tr) * args.n).astype(int)
quota[0] += args.n - quota.sum()  # rounding remainder goes to Explicit
draws = [np.concatenate([rng.choice(ix, q, replace=True) for ix, q in zip(by_class, quota)]) for _ in range(args.draws)]

scores = {k: [] for k in PRED}
cls_f1 = np.zeros((args.draws, 9))
cls_zero = np.zeros((args.draws, 9), bool)
cl_f1, ev_f1 = [], []
for i, d in enumerate(draws):
    for k, (pcl, pev) in PRED.items():
        scores[k].append(50 * macro(y_cl[d], pcl[d], 3) + 50 * macro(y_ev[d], pev[d], 9))
    f, present = per_class_f1(y_ev[d], PRED["v1"][1][d], 9)
    cls_f1[i], cls_zero[i] = f, present & (f == 0)
    cl_f1.append(macro(y_cl[d], PRED["v1"][0][d], 3)); ev_f1.append(macro(y_ev[d], PRED["v1"][1][d], 9))
scores = {k: np.array(v) for k, v in scores.items()}
v1 = scores["v1"]
full = 50 * macro(y_cl, PRED["v1"][0], 3) + 50 * macro(y_ev, PRED["v1"][1], 9)
pct = lambda a, q: np.percentile(a, q)

print(f"{args.draws} stratified bootstrap draws of {args.n} rows (per-class quota {dict(zip(EVASION_LABELS, quota.tolist()))})\n")
print("## 1. v1's score on a 296-row sample")
print(f"  full OOF score: {full:.2f}")
print(f"  draws: mean {v1.mean():.2f}, sd {v1.std():.2f}, 5th-95th {pct(v1, 5):.2f} - {pct(v1, 95):.2f}, 1st {pct(v1, 1):.2f}, min {v1.min():.2f}")
print(f"  clarity part: mean {50 * np.mean(cl_f1):.2f} sd {50 * np.std(cl_f1):.2f} | evasion part: mean {50 * np.mean(ev_f1):.2f} sd {50 * np.std(ev_f1):.2f}")
z = (args.lb - v1.mean()) / v1.std()
print(f"  leaderboard {args.lb}: z = {z:.1f}, share of draws at or below it = {np.mean(v1 <= args.lb):.4f}")

print("\n## 1b. Which explanations make the leaderboard score plausible? (train-only scenarios)")
acc = lambda y, p: np.mean(y == p)
formulas = {
    "50 x clarity F1 + 50 x evasion F1 (assumed)": lambda d: 50 * macro(y_cl[d], PRED["v1"][0][d], 3) + 50 * macro(y_ev[d], PRED["v1"][1][d], 9),
    "100 x evasion macro-F1 only": lambda d: 100 * macro(y_ev[d], PRED["v1"][1][d], 9),
    "100 x clarity macro-F1 only": lambda d: 100 * macro(y_cl[d], PRED["v1"][0][d], 3),
    "50 x clarity mapped F1 + 50 x evasion F1": lambda d: 50 * macro(y_cl[d], EV2CL[PRED["v1"][1][d]], 3) + 50 * macro(y_ev[d], PRED["v1"][1][d], 9),
    "50 x evasion F1 + 50 x evasion accuracy": lambda d: 50 * macro(y_ev[d], PRED["v1"][1][d], 9) + 50 * acc(y_ev[d], PRED["v1"][1][d]),
}
for name, fn in formulas.items():
    s = np.array([fn(d) for d in draws[:2000]])
    print(f"  {name:46s} mean {s.mean():6.2f}  5th-95th {pct(s, 5):6.2f} - {pct(s, 95):6.2f}  share <= {args.lb}: {np.mean(s <= args.lb):.3f}")
pres = tr.president.values
for name, keep in [("only Iohannis + Nicusor Dan + Bolojan rows", np.isin(pres, ["Klaus Iohannis", "Nicusor Dan", "Ilie Bolojan"])),
                   ("only Nicusor Dan + Bolojan rows", np.isin(pres, ["Nicusor Dan", "Ilie Bolojan"]))]:
    ix = np.flatnonzero(keep)
    s = []
    for _ in range(2000):
        d = rng.choice(ix, args.n, replace=True)
        s.append(50 * macro(y_cl[d], PRED["v1"][0][d], 3) + 50 * macro(y_ev[d], PRED["v1"][1][d], 9))
    s = np.array(s)
    print(f"  50/50 score, {name:42s} (n={len(ix)}) mean {s.mean():6.2f}  5th-95th {pct(s, 5):6.2f} - {pct(s, 95):6.2f}  share <= {args.lb}: {np.mean(s <= args.lb):.3f}")

print("\n## 2. Paired difference to v1 on the same draws (other - v1), in leaderboard points")
ev_only = {k: np.array([100 * macro(y_ev[d], pev[d], 9) for d in draws]) for k, (_, pev) in PRED.items()}
print(f"  {'ensemble':38s} {'agree':>6s} | 50/50 score: full-OOF diff, draws mean, sd, 2.5th..97.5th | evasion-only score: same")
for k in PRED:
    if k == "v1":
        continue
    a, b = scores[k] - v1, ev_only[k] - ev_only["v1"]
    fa = 50 * (f1sum(*ENS[k]) - f1sum(*ENS["v1"]))
    fb = 100 * (macro(y_ev, PRED[k][1], 9) - macro(y_ev, PRED["v1"][1], 9))
    agree = np.mean(PRED[k][1] == PRED["v1"][1])
    print(f"  {k:38s} {agree:6.1%} | {fa:+5.2f} {a.mean():+5.2f} {a.std():4.2f} {pct(a, 2.5):+6.2f}..{pct(a, 97.5):+5.2f}"
          f" | {fb:+5.2f} {b.mean():+5.2f} {b.std():4.2f} {pct(b, 2.5):+6.2f}..{pct(b, 97.5):+5.2f}")

print("\n## 3. Per-class variance (v1, evasion)")
print("  class                 quota  mean F1  sd F1  sd in score points (x50/9)  F1 = 0 in % of draws")
for k, l in enumerate(EVASION_LABELS):
    print(f"  {l:22s} {quota[k]:5d}  {cls_f1[:, k].mean():.3f}   {cls_f1[:, k].std():.3f}   {50 / 9 * cls_f1[:, k].std():.2f}"
          f"                        {100 * cls_zero[:, k].mean():.1f}")
