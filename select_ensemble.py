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
Fixed recipes (--recipe, default RECIPES below) average every config whose name matches a pattern,
with equal weights and no selection, so their OOF score needs no nesting. They are defined before
runs 3-6 land; matching configs from later runs join automatically.
--combine (default plain): how member probabilities are averaged.
  plain  arithmetic mean of probabilities (as evaluate.py)
  geo    geometric mean: mean of log-probabilities, renormalised
  temp   per-config temperature first, p ~ p_raw^(1/T), T fitted by OOF log-loss on a grid
         (separately for the evasion and clarity heads), nested: each fold's OOF rows use the T
         fitted on the other 4 folds; test uses the T fitted on all rows. Then the plain mean.
"""
import argparse, fnmatch, glob, os, re
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from common import load, folds, EVASION_LABELS, CLARITY_LABELS, EVASION_TO_CLARITY

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", default=["results/run1"])
ap.add_argument("--max_steps", type=int, default=30)
ap.add_argument("--objective", choices=["sum", "clarity", "evasion_only", "evasion"], default="sum",
                help="sum: clarity + evasion (default, Rares after the v3 probe); clarity / evasion_only: choose a bag "
                     "for one subtask only (combine two with split_submission.py); evasion: evasion first with "
                     "clarity within --max_clarity_drop of v1 (the earlier rule)")
ap.add_argument("--max_clarity_drop", type=float, default=0.01)
ap.add_argument("--submission", help="write the selected ensemble's test predictions to this csv")
ap.add_argument("--recipe", action="append", metavar="LABEL=PATTERN[,PATTERN]",
                help="fixed equal-weight average of the configs matching the patterns (repeatable); "
                     "default: the RECIPES below")
ap.add_argument("--combine", choices=["plain", "geo", "temp"], default="plain")
ap.add_argument("--dump", metavar="NPZ", help="save every fixed recipe's OOF and test probabilities (for pick_recipes.py)")
ap.add_argument("--submit_recipe", metavar="LABEL", help="with --submission: write this recipe's predictions instead of the greedy bag")
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
TGRID = np.round(np.arange(0.5, 5.01, 0.1), 2)
PE_RAW, PC_RAW = PE.copy(), PC.copy()  # the submitted v1 is a plain average of these


def temper(P, T):
    lp = np.log(np.clip(P, 1e-12, 1)) / T
    lp -= lp.max(-1, keepdims=True)
    q = np.exp(lp)
    return q / q.sum(-1, keepdims=True)


def fit_T(P, y):
    return min(TGRID, key=lambda T: -np.log(np.clip(temper(P, T)[np.arange(len(y)), y], 1e-12, 1)).mean())


if args.combine == "temp":  # nested temperature per config and head
    fold_rows = folds(tr.evasion_label.values)
    Ts = {}
    for k in range(K):
        for P, TP, y, head in ((PE, TPE, y_ev, "ev"), (PC, TPC, y_cl, "cl")):
            Pk = P[k].copy()
            for trn, val in fold_rows:
                Pk[val] = temper(P[k][val], fit_T(P[k][trn], y[trn]))
            T_all = fit_T(P[k], y)
            P[k], TP[k] = Pk, temper(TP[k], T_all)
            Ts[(names[k], head)] = T_all
    print("temperatures fitted on all rows (evasion / clarity):")
    for n in names:
        print(f"  {n}: {Ts[(n, 'ev')]:.1f} / {Ts[(n, 'cl')]:.1f}")


def blend(w, P):
    if args.combine == "geo":
        lp = np.tensordot(w / w.sum(), np.log(np.clip(P, 1e-12, 1)), axes=1)
        q = np.exp(lp - lp.max(-1, keepdims=True))
        return q / q.sum(-1, keepdims=True)
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
    if args.objective == "clarity":
        return (cl, ev)
    if args.objective == "evasion_only":
        return (ev, cl)
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


# fixed recipes, defined before runs 3-6 (large = XLM-R large / RoBERT-large at lr 1e-5, any seed or run)
LARGE = ["run1/xlm-roberta-large_p0", "run1/RoBERT-large_p0", "*/r[0-9]-xlmr-large-lr1e-5*", "*/r[0-9]-robert-large-lr1e-5*"]
RECIPES = {
    "R1 all large lr1e-5 models": LARGE,
    "R2 R1 + v1's RO-BERT base": LARGE + ["run1/bert-base-romanian-cased-v1_p0"],
    "R3 R1 + run-3 headtail variants": LARGE + ["*/r[0-9]-robert-large-headtail*"],
    "R4 all XLM-R large models": ["run1/xlm-roberta-large_p0", "*/r[0-9]-xlmr-large-lr1e-5*"],
    "T2 run-2 trio": ["run2/r2-xlmr-large-lr1e-5_p0", "run2/r2-robert-large-lr1e-5_p0", "run2/r2-robert-base-lr2e-5_p0"],
    "T12 v1 trio + run-2 trio": ["run1/xlm-roberta-large_p0", "run1/RoBERT-large_p0", "run1/bert-base-romanian-cased-v1_p0",
                                 "run2/r2-xlmr-large-lr1e-5_p0", "run2/r2-robert-large-lr1e-5_p0", "run2/r2-robert-base-lr2e-5_p0"],
    # pre-set before runs 4-6 land (21 Nov ~18:00): v2's six models plus every later large lr 1e-5 seed; use with --combine geo
    "V4 T12 + new large seeds": ["run1/xlm-roberta-large_p0", "run1/RoBERT-large_p0", "run1/bert-base-romanian-cased-v1_p0",
                                 "run2/r2-xlmr-large-lr1e-5_p0", "run2/r2-robert-large-lr1e-5_p0", "run2/r2-robert-base-lr2e-5_p0",
                                 "*/r[4-9]-xlmr-large-lr1e-5*", "*/r[4-9]-robert-large-lr1e-5*"],
}
if args.recipe:
    RECIPES = {r.split("=", 1)[0]: r.split("=", 1)[1].split(",") for r in args.recipe}
W_REC = {}
for label, pats in RECIPES.items():
    w = np.array([float(any(fnmatch.fnmatch(n, p) for p in pats)) for n in names])
    if w.sum():
        W_REC[label] = w

table = []
if W_V1 is not None:
    table.append(row("v1 trio (submission v1)", W_V1))
for label, w in W_REC.items():
    table.append(row(f"{label} ({int(w.sum())})", w))
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
print(f"\n{K} configs from {', '.join(args.runs)}; combine: {args.combine}; selection objective: {goal}. OOF macro-F1 over {len(tr)} train rows:\n")
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
    w1 = W_V1 / W_V1.sum()  # submitted v1: plain mean of the raw probabilities, whatever --combine is
    v1_cl, v1_ev, _ = score(np.tensordot(w1, PE_RAW, axes=1), np.tensordot(w1, PC_RAW, axes=1))
    v1_sum = v1_cl + v1_ev
    print(f"reference = submitted v1 (plain average): clarity {v1_cl:.4f}, evasion {v1_ev:.4f}")
    print(f"nested greedy vs v1: evasion {n_ev - v1_ev:+.4f}, clarity {n_cl - v1_cl:+.4f}, sum {n_cl + n_ev - v1_sum:+.4f}")
    if args.objective == "evasion":
        ok = n_ev - v1_ev >= 0.02 and n_cl - v1_cl >= -args.max_clarity_drop
        print(f"bar for a new submission: nested evasion +0.02 with clarity drop <= {args.max_clarity_drop:g} -> {'MET' if ok else 'not met'}")
    elif args.objective in ("clarity", "evasion_only"):
        part, gain = ("clarity", n_cl - v1_cl) if args.objective == "clarity" else ("evasion", n_ev - v1_ev)
        print(f"objective {args.objective}: nested {part} {gain:+.4f} vs v1. Use this bag only for the {part} rows; "
              f"a split submission's nested sum = nested clarity of the clarity bag + nested evasion of the evasion bag (bar +0.02)")
    else:
        print(f"bar for a new submission: nested sum +0.02 -> {'MET' if n_cl + n_ev - v1_sum >= 0.02 else 'not met'}")

if W_REC:
    print("\nfixed recipes (no selection, so OOF needs no nesting):")
    for label, w in W_REC.items():
        cl, ev, _ = score(blend(w, PE), blend(w, PC))
        vs = f"  vs v1: evasion {ev - v1_ev:+.4f}, clarity {cl - v1_cl:+.4f}" if W_V1 is not None else ""
        print(f"  {label}: {[n for n, x in zip(names, w) if x]}{vs}")

if args.dump:
    out = {"y_ev": y_ev, "y_cl": y_cl, "test_ids": te.datapointID.values, "labels": np.array(list(W_REC))}
    for i, (label, w) in enumerate(W_REC.items()):
        out[f"pe{i}"], out[f"pc{i}"] = blend(w, PE), blend(w, PC)
        out[f"tpe{i}"], out[f"tpc{i}"] = blend(w, TPE), blend(w, TPC)
    if W_V1 is not None:  # the submitted v1, always as a plain average of the raw probabilities
        w1 = W_V1 / W_V1.sum()
        out["v1_pe"], out["v1_pc"] = np.tensordot(w1, PE_RAW, axes=1), np.tensordot(w1, PC_RAW, axes=1)
    np.savez(args.dump, combine=args.combine, **out)
    print("wrote", args.dump)

if args.submission:
    w_out = W_REC[args.submit_recipe] if args.submit_recipe else w_sel
    tpe, tpc = blend(w_out, TPE), blend(w_out, TPC)
    m = score(blend(w_out, PE), blend(w_out, PC))[2]
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
