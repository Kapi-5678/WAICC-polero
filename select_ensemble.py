"""Task 7: greedy forward ensemble selection with replacement (Caruana et al.), with a nested check.

  python select_ensemble.py --runs results/run1
  python select_ensemble.py --runs results/run1 results/run2
  python select_ensemble.py --runs results/run1 results/run2 --submission submissions/v2/submission_v2.csv
  python make_submission_py.py submissions/v2/submission_v2.csv

Every complete 5-fold config in the run folders is taken at its best epoch (as evaluate.py
chooses it) and named <run folder>/<config>. Starting from an empty bag, the config whose
addition most improves the objective of the probability average is added, repeatedly (a config
may be added more than once); selection stops when nothing improves.
Objective (default --objective evasion, Rares's rule): maximise OOF evasion macro-F1 while
clarity stays within --max_clarity_drop (0.01) of the reference, the v1 trio (or, without run 1,
the best single config). --objective sum maximises clarity + evasion instead.
Clarity is the best of combined / mapped / head, as in evaluate.py.
Nested check: the selection is rerun on 4 folds' OOF rows (common.folds) and scored on the 5th.
"""
import argparse, glob, os, re
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from common import load, folds, EVASION_LABELS, CLARITY_LABELS, EVASION_TO_CLARITY

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", default=["results/run1"])
ap.add_argument("--max_steps", type=int, default=30)
ap.add_argument("--objective", choices=["evasion", "sum"], default="evasion")
ap.add_argument("--max_clarity_drop", type=float, default=0.01)
ap.add_argument("--submission", help="write the selected ensemble's test predictions to this csv")
args = ap.parse_args()

tr, te = load()
y_ev = np.array([EVASION_LABELS.index(l) for l in tr.evasion_label])
y_cl = np.array([CLARITY_LABELS.index(l) for l in tr.clarity_label])
EV2CL = np.array([CLARITY_LABELS.index(EVASION_TO_CLARITY[l]) for l in EVASION_LABELS])
METHODS = ["combined", "mapped", "head"]


def mf(y, p, k):  # macro-F1 over k classes (equals sklearn's when every class occurs in y)
    cm = np.bincount(y * k + p, minlength=k * k).reshape(k, k)
    d = cm.sum(0) + cm.sum(1)
    return np.mean(np.where(d > 0, 2 * np.diag(cm) / np.maximum(d, 1), 0.0))


def clarity_pred(pe, pc, method):
    if method == "mapped":
        return EV2CL[pe.argmax(1)]
    if method == "head":
        return pc.argmax(1)
    agg = np.stack([pe[:, EV2CL == c].sum(1) for c in range(3)], 1)
    return ((agg + pc) / 2).argmax(1)  # combined, as in evaluate.py


def score(pe, pc, rows=None, method=None):
    """(clarity, evasion, clarity method) on the given rows; the method is chosen unless given."""
    r = slice(None) if rows is None else rows
    ev = mf(y_ev[r], pe[r].argmax(1), 9)
    cands = [method] if method else METHODS
    cl, m = max((mf(y_cl[r], clarity_pred(pe[r], pc[r], m), 3), m) for m in cands)
    return cl, ev, m


# ---- load every complete config at its best epoch
groups = defaultdict(list)
for d in args.runs:
    for f in glob.glob(os.path.join(d, "*_f[0-9]_s*.npz")):
        groups[f"{os.path.basename(os.path.normpath(d))}/{re.sub(r'_f\d_s\d+\.npz$', '', os.path.basename(f))}"].append(f)
names, PE, PC, TPE, TPC = [], [], [], [], []
for c in sorted(groups):
    runs = [np.load(f) for f in sorted(groups[c])]
    if len(runs) != 5 or len(np.unique(np.concatenate([r["val_idx"] for r in runs]))) != len(tr):
        print(f"{c}: skipped, {len(runs)} folds")
        continue
    assert all((r["test_ids"] == te.datapointID.values).all() for r in runs), c
    best = None
    for e in range(sum(k.startswith("val_pe") for k in runs[0].files)):
        pe, pc = np.zeros((len(tr), 9)), np.zeros((len(tr), 3))
        for r in runs:
            pe[r["val_idx"]], pc[r["val_idx"]] = r[f"val_pe{e}"], r[f"val_pc{e}"]
        cl, ev, _ = score(pe, pc)
        if best is None or cl + ev > best[0]:
            best = (cl + ev, e, pe, pc, np.mean([r[f"test_pe{e}"] for r in runs], 0), np.mean([r[f"test_pc{e}"] for r in runs], 0))
    names.append(c); PE.append(best[2]); PC.append(best[3]); TPE.append(best[4]); TPC.append(best[5])
    print(f"{c}: best epoch {best[1]}")
PE, PC, TPE, TPC = map(np.array, (PE, PC, TPE, TPC))
K = len(names)


def blend(w, P):
    return np.tensordot(w / w.sum(), P, axes=1)


V1 = [f"run1/{c}" for c in ["xlm-roberta-large_p0", "RoBERT-large_p0", "bert-base-romanian-cased-v1_p0"]]
W_V1 = np.array([float(n in V1) for n in names]) if set(V1) <= set(names) else None


def ref_clarity(rows):
    """Clarity the selection must stay close to: v1's, else the best single config's."""
    if W_V1 is not None:
        return score(blend(W_V1, PE), blend(W_V1, PC), rows)[0]
    return max(score(PE[k], PC[k], rows)[0] for k in range(K))


def objective(cl, ev, ref):
    if args.objective == "sum":
        return (cl + ev,)
    return (cl >= ref - args.max_clarity_drop - 1e-12, ev, cl)  # feasible first, then evasion, then clarity


def greedy(rows):
    """Forward selection with replacement on the given rows; returns the count vector."""
    ref = ref_clarity(rows)
    w, best = np.zeros(K), None
    for _ in range(args.max_steps):
        trials = []
        for k in range(K):
            w2 = w.copy()
            w2[k] += 1
            cl, ev, _ = score(blend(w2, PE), blend(w2, PC), rows)
            trials.append((objective(cl, ev, ref), -w2.sum(), k))
        o, _, k = max(trials)
        if best is not None and o <= best:  # no strict improvement
            break
        best, w[k] = o, w[k] + 1
    return w


def row(label, w):
    cl, ev, m = score(blend(w, PE), blend(w, PC))
    return (label, cl, ev, cl + ev, m)


table = []
if W_V1 is not None:
    table.append(row("v1 trio (submission v1)", W_V1))
table.append(row(f"all {K} configs averaged", np.ones(K)))
w_sel = greedy(slice(None))
table.append(row("greedy selection, in-sample", w_sel))

# nested: select on 4 folds, apply the bag (and its clarity method) to the 5th
splits = folds(tr.evasion_label.values)
p_ev, p_cl, bags = np.empty(len(tr), int), np.empty(len(tr), int), []
for trn, val in splits:
    w = greedy(trn)
    pe, pc = blend(w, PE), blend(w, PC)
    m = score(pe, pc, trn)[2]
    p_ev[val], p_cl[val] = pe[val].argmax(1), clarity_pred(pe[val], pc[val], m)
    bags.append(w)
n_cl, n_ev = mf(y_cl, p_cl, 3), mf(y_ev, p_ev, 9)
table.append(("greedy selection, nested", n_cl, n_ev, n_cl + n_ev, "per fold"))
for k, n in enumerate(names):
    table.append(row(n, np.eye(K)[k]))

goal = ("evasion first, clarity within " + f"{args.max_clarity_drop:g} of " + ("v1" if W_V1 is not None else "the best single config")) if args.objective == "evasion" else "clarity + evasion"
print(f"\n{K} configs from {', '.join(args.runs)}; selection objective: {goal}. OOF macro-F1 over {len(tr)} train rows:\n")
print(f"| {'ensemble':52s} | clarity | evasion | sum    | clarity method |")
print(f"|{'-' * 54}|---------|---------|--------|----------------|")
for label, cl, ev, s, m in table:
    print(f"| {label:52s} | {cl:.4f}  | {ev:.4f}  | {s:.4f} | {m:14s} |")
ins = [t for t in table if t[0].startswith("greedy selection, in")][0][3]
print(f"\nin-sample selection overstates the nested score by {ins - n_cl - n_ev:+.4f} (sum)")
print("selected bag (in-sample):", {names[k]: int(c) for k, c in enumerate(w_sel) if c})
for i, b in enumerate(bags):
    print(f"  nested fold {i} bag:", {names[k]: int(c) for k, c in enumerate(b) if c})
if W_V1 is not None:
    _, v1_cl, v1_ev, v1_sum, _ = table[0]
    print(f"nested greedy vs v1: evasion {n_ev - v1_ev:+.4f}, clarity {n_cl - v1_cl:+.4f}, sum {n_cl + n_ev - v1_sum:+.4f}")
    if args.objective == "evasion":
        ok = n_ev - v1_ev >= 0.02 and n_cl - v1_cl >= -args.max_clarity_drop
        print(f"bar for a new submission: nested evasion +0.02 with clarity drop <= {args.max_clarity_drop:g} -> {'MET' if ok else 'not met'}")
    else:
        print(f"bar for a new submission: nested sum +0.02 -> {'MET' if n_cl + n_ev - v1_sum >= 0.02 else 'not met'}")

if args.submission:
    tpe, tpc = blend(w_sel, TPE), blend(w_sel, TPC)
    m = score(blend(w_sel, PE), blend(w_sel, PC))[2]
    clar = dict(zip(te.datapointID, [CLARITY_LABELS[j] for j in clarity_pred(tpe, tpc, m)]))
    evas = dict(zip(te.datapointID, [EVASION_LABELS[j] for j in tpe.argmax(1)]))
    sub = pd.read_csv("data/sample_output.csv")
    sub["answer"] = [clar[d] if t == 1 else evas[d] for t, d in zip(sub.subtaskID, sub.datapointID)]
    assert len(sub) == 592 and set(sub.datapointID) == set(te.datapointID)
    assert sub[sub.subtaskID == 1].answer.isin(CLARITY_LABELS).all() and sub[sub.subtaskID == 2].answer.isin(EVASION_LABELS).all()
    os.makedirs(os.path.dirname(args.submission) or ".", exist_ok=True)
    sub.to_csv(args.submission, index=False)
    print("wrote", args.submission, f"(clarity method: {m})")
    for t in (1, 2):
        print(" ", sub[sub.subtaskID == t].answer.value_counts().to_dict())
