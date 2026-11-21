"""Task 7: choose the clarity recipe and the evasion recipe separately, with a nested check.

  python select_ensemble.py --runs results/run1 results/run2 --combine plain --dump d_plain.npz
  python select_ensemble.py --runs results/run1 results/run2 --combine geo   --dump d_geo.npz
  python select_ensemble.py --runs results/run1 results/run2 --combine temp  --dump d_temp.npz
  python pick_recipes.py d_plain.npz d_geo.npz d_temp.npz [--submission submissions/vN/submission_vN.csv]

Candidates are every fixed recipe under every --combine mode (and, for clarity, every clarity
method: combined / mapped / head). The clarity candidate and the evasion candidate are each
picked on their own OOF macro-F1. Nested check: pick on 4 folds' OOF rows (common.folds), score
the 5th, rotate; the nested sum against the submitted v1 decides (bar +0.02).
"""
import argparse, os
import numpy as np
import pandas as pd
from common import load, folds, EVASION_LABELS, CLARITY_LABELS, EVASION_TO_CLARITY

ap = argparse.ArgumentParser()
ap.add_argument("dumps", nargs="+")
ap.add_argument("--submission")
args = ap.parse_args()

tr, te = load()
EV2CL = np.array([CLARITY_LABELS.index(EVASION_TO_CLARITY[l]) for l in EVASION_LABELS])


def mf(y, p, k):
    cm = np.bincount(y * k + p, minlength=k * k).reshape(k, k)
    d = cm.sum(0) + cm.sum(1)
    return np.mean(np.where(d > 0, 2 * np.diag(cm) / np.maximum(d, 1), 0.0))


def clar(pe, pc, m):
    if m == "mapped":
        return EV2CL[pe.argmax(1)]
    if m == "head":
        return pc.argmax(1)
    return ((np.stack([pe[:, EV2CL == c].sum(1) for c in range(3)], 1) + pc) / 2).argmax(1)


ev_c, cl_c = {}, {}  # name -> (OOF labels, test labels)
for f in args.dumps:
    z = np.load(f, allow_pickle=False)
    y_ev, y_cl, mode = z["y_ev"], z["y_cl"], str(z["combine"])
    assert (z["test_ids"] == te.datapointID.values).all()
    for i, label in enumerate(z["labels"]):
        name = f"{label} [{mode}]"
        ev_c[name] = (z[f"pe{i}"].argmax(1), z[f"tpe{i}"].argmax(1))
        for m in ("combined", "mapped", "head"):
            cl_c[f"{name} {m}"] = (clar(z[f"pe{i}"], z[f"pc{i}"], m), clar(z[f"tpe{i}"], z[f"tpc{i}"], m))
    if "v1_pe" in z.files:
        v1_ev, v1_cl = z["v1_pe"].argmax(1), clar(z["v1_pe"], z["v1_pc"], "combined")

base_ev, base_cl = mf(y_ev, v1_ev, 9), mf(y_cl, v1_cl, 3)
splits = folds(tr.evasion_label.values)


def pick(cands, y, k, rows):
    return max(cands, key=lambda n: mf(y[rows], cands[n][0][rows], k))


def nested(cands, y, k):
    pred, chosen = np.empty(len(y), int), []
    for trn, val in splits:
        n = pick(cands, y, k, trn)
        pred[val] = cands[n][0][val]
        chosen.append(n)
    return mf(y, pred, k), chosen


all_rows = np.arange(len(y_ev))
print(f"{len(cl_c)} clarity candidates, {len(ev_c)} evasion candidates. Submitted v1: clarity {base_cl:.4f}, evasion {base_ev:.4f}\n")
for part, cands, y, k, base in (("clarity", cl_c, y_cl, 3, base_cl), ("evasion", ev_c, y_ev, 9, base_ev)):
    ranked = sorted(cands, key=lambda n: -mf(y, cands[n][0], k))
    print(f"top {part} candidates (in-sample OOF):")
    for n in ranked[:5]:
        print(f"  {mf(y, cands[n][0], k):.4f} ({mf(y, cands[n][0], k) - base:+.4f})  {n}")
n_cl, ch_cl = nested(cl_c, y_cl, 3)
n_ev, ch_ev = nested(ev_c, y_ev, 9)
best_cl, best_ev = pick(cl_c, y_cl, 3, all_rows), pick(ev_c, y_ev, 9, all_rows)
in_cl, in_ev = mf(y_cl, cl_c[best_cl][0], 3), mf(y_ev, ev_c[best_ev][0], 9)
print(f"\nin-sample pick: clarity {in_cl:.4f} + evasion {in_ev:.4f} = {in_cl + in_ev:.4f} ({in_cl + in_ev - base_cl - base_ev:+.4f} vs v1)")
print(f"nested pick:    clarity {n_cl:.4f} + evasion {n_ev:.4f} = {n_cl + n_ev:.4f} ({n_cl + n_ev - base_cl - base_ev:+.4f} vs v1)")
print("  clarity picks per fold:", ch_cl)
print("  evasion picks per fold:", ch_ev)
gain = n_cl + n_ev - base_cl - base_ev
print(f"bar for a new submission: nested sum +0.02 -> {'MET' if gain >= 0.02 else 'not met'} ({gain:+.4f})")

if args.submission:
    c = dict(zip(te.datapointID, [CLARITY_LABELS[j] for j in cl_c[best_cl][1]]))
    e = dict(zip(te.datapointID, [EVASION_LABELS[j] for j in ev_c[best_ev][1]]))
    sub = pd.read_csv("data/sample_output.csv")
    sub["answer"] = [c[d] if t == 1 else e[d] for t, d in zip(sub.subtaskID, sub.datapointID)]
    assert len(sub) == 592 and set(sub.datapointID) == set(te.datapointID)
    assert sub[sub.subtaskID == 1].answer.isin(CLARITY_LABELS).all() and sub[sub.subtaskID == 2].answer.isin(EVASION_LABELS).all()
    os.makedirs(os.path.dirname(args.submission) or ".", exist_ok=True)
    sub.to_csv(args.submission, index=False)
    print(f"wrote {args.submission}: clarity from '{best_cl}', evasion from '{best_ev}'")
