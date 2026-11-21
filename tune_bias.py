"""Task 5: per-class additive log-prob offsets that maximise macro-F1, honestly scored.

  python tune_bias.py                                   # the run-1 ensemble behind submission_v1
  python tune_bias.py --use xlm-roberta-large_p0 RoBERT-large_p0
  python tune_bias.py --submission submissions/submission_bias.csv

Offsets are found by coordinate ascent on the OOF ensemble probabilities, separately for
evasion (9) and clarity (3). The reported gain is nested: offsets are fitted on the OOF rows
of 4 folds (common.folds) and scored on the 5th, rotating through all 5. The final offsets are
then refitted on all rows and saved.
"""
import argparse, glob, json, os, re
from collections import defaultdict
import numpy as np
from common import load, folds, write_submission, EVASION_LABELS, CLARITY_LABELS, EVASION_TO_CLARITY

ap = argparse.ArgumentParser()
ap.add_argument("--runs", default="results/run1")
ap.add_argument("--use", nargs="*", default=["xlm-roberta-large_p0", "RoBERT-large_p0", "bert-base-romanian-cased-v1_p0"])
ap.add_argument("--max_offset", type=float, default=3.0)
ap.add_argument("--step", type=float, default=0.1)
ap.add_argument("--out", default="bias/offsets.json")
ap.add_argument("--submission", help="also write a submission with the offsets applied")
args = ap.parse_args()

tr, te = load()
y_ev = np.array([EVASION_LABELS.index(l) for l in tr.evasion_label])
y_cl = np.array([CLARITY_LABELS.index(l) for l in tr.clarity_label])
EV2CL = np.array([CLARITY_LABELS.index(EVASION_TO_CLARITY[l]) for l in EVASION_LABELS])
GRID = sorted(np.arange(-args.max_offset, args.max_offset + 1e-9, args.step), key=abs)  # ties keep small offsets


def macro_f1(y, p, k):
    cm = np.bincount(y * k + p, minlength=k * k).reshape(k, k)
    denom = cm.sum(0) + cm.sum(1)
    return np.mean(np.where(denom > 0, 2 * np.diag(cm) / np.maximum(denom, 1), 0.0))


def clarity_probs(pe, pc):  # same as evaluate.py: clarity head averaged with evasion mapped upward
    agg = np.zeros_like(pc)
    for k in range(9):
        agg[:, EV2CL[k]] += pe[:, k]
    return (agg + pc) / 2


def best_epoch(cfg, files):
    """OOF and test probabilities of the config's best epoch, chosen as evaluate.py does."""
    runs = [np.load(f) for f in sorted(files)]
    assert len(runs) == 5 and len(np.unique(np.concatenate([r["val_idx"] for r in runs]))) == len(tr), cfg
    best = None
    for e in range(sum(k.startswith("val_pe") for k in runs[0].files)):
        pe, pc = np.zeros((len(tr), 9)), np.zeros((len(tr), 3))
        for r in runs:
            pe[r["val_idx"]], pc[r["val_idx"]] = r[f"val_pe{e}"], r[f"val_pc{e}"]
        clar = max(macro_f1(y_cl, clarity_probs(pe, pc).argmax(1), 3), macro_f1(y_cl, EV2CL[pe.argmax(1)], 3),
                   macro_f1(y_cl, pc.argmax(1), 3))
        s = clar + macro_f1(y_ev, pe.argmax(1), 9)
        if best is None or s > best[0]:
            best = (s, e, pe, pc, np.mean([r[f"test_pe{e}"] for r in runs], 0), np.mean([r[f"test_pc{e}"] for r in runs], 0))
    print(f"{cfg}: best epoch {best[1]}")
    return best[2:]


def fit(logp, y, k):
    """Coordinate ascent: for each class in turn, try every offset on the grid; repeat until stable."""
    b, best = np.zeros(k), macro_f1(y, logp.argmax(1), k)
    for _ in range(10):
        improved = False
        for c in range(k):
            for v in GRID:
                bb = b.copy()
                bb[c] = v
                s = macro_f1(y, (logp + bb).argmax(1), k)
                if s > best + 1e-12:
                    b, best, improved = bb, s, True
        if not improved:
            break
    return b - b.mean()  # only differences between classes matter


def nested(logp, y, k, splits):
    pred, gains = np.empty(len(y), int), []
    for trn, val in splits:
        pred[val] = (logp[val] + fit(logp[trn], y[trn], k)).argmax(1)
        gains.append(macro_f1(y[val], pred[val], k) - macro_f1(y[val], logp[val].argmax(1), k))
    return pred, np.array(gains)


groups = defaultdict(list)
for f in glob.glob(os.path.join(args.runs, "*_f[0-9]_s*.npz")):
    groups[re.sub(r"_f\d_s\d+\.npz$", "", os.path.basename(f))].append(f)
parts = [best_epoch(c, groups[c]) for c in args.use]
pe, pc, tpe, tpc = (np.mean([p[i] for p in parts], 0) for i in range(4))
lp_ev, lp_cl = np.log(pe + 1e-9), np.log(clarity_probs(pe, pc) + 1e-9)
splits = folds(tr.evasion_label.values)

ev_pred, ev_gain = nested(lp_ev, y_ev, 9, splits)
cl_pred, cl_gain = nested(lp_cl, y_cl, 3, splits)
rows = [
    ("evasion", macro_f1(y_ev, pe.argmax(1), 9), macro_f1(y_ev, ev_pred, 9), ev_gain),
    ("clarity (combined probs, own offsets)", macro_f1(y_cl, lp_cl.argmax(1), 3), macro_f1(y_cl, cl_pred, 3), cl_gain),
    ("clarity (mapped from evasion)", macro_f1(y_cl, EV2CL[pe.argmax(1)], 3), macro_f1(y_cl, EV2CL[ev_pred], 3), None),
]
print(f"\nensemble of {args.use}, nested 5-fold OOF macro-F1 (offsets fitted on 4 folds, scored on the 5th):")
for name, base, tuned, g in rows:
    extra = f"  per-fold gain {g.mean():+.4f} ± {g.std():.4f}" if g is not None else ""
    print(f"  {name:40s} argmax {base:.4f}  tuned {tuned:.4f}  gain {tuned - base:+.4f}{extra}")
print("\nper-class evasion F1 (argmax -> tuned):")
f1c = lambda p, c: 2 * ((p == c) & (y_ev == c)).sum() / max((p == c).sum() + (y_ev == c).sum(), 1)
for c, l in enumerate(EVASION_LABELS):
    print(f"  {l:22s} n={int((y_ev == c).sum()):4d}  {f1c(pe.argmax(1), c):.3f} -> {f1c(ev_pred, c):.3f}")

b_ev, b_cl = fit(lp_ev, y_ev, 9), fit(lp_cl, y_cl, 3)
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
json.dump({"use": args.use, "evasion": dict(zip(EVASION_LABELS, b_ev.round(3).tolist())),
           "clarity": dict(zip(CLARITY_LABELS, b_cl.round(3).tolist())),
           "nested_gain": {"evasion": round(rows[0][2] - rows[0][1], 4), "clarity": round(rows[1][2] - rows[1][1], 4)}},
          open(args.out, "w"), indent=1, ensure_ascii=False)
print(f"\noffsets fitted on all rows -> {args.out}")
print("  evasion:", {l: round(float(v), 2) for l, v in zip(EVASION_LABELS, b_ev)})
print("  clarity:", {l: round(float(v), 2) for l, v in zip(CLARITY_LABELS, b_cl)})

t_ev = (np.log(tpe + 1e-9) + b_ev).argmax(1)
t_cl = (np.log(clarity_probs(tpe, tpc) + 1e-9) + b_cl).argmax(1)
print("test prediction counts, evasion argmax -> tuned:")
for c, l in enumerate(EVASION_LABELS):
    print(f"  {l:22s} {(tpe.argmax(1) == c).sum():4d} -> {(t_ev == c).sum():4d}")
if args.submission:
    ids = np.load(groups[args.use[0]][0])["test_ids"]
    assert (ids == te.datapointID.values).all()
    write_submission(te, [CLARITY_LABELS[j] for j in t_cl], [EVASION_LABELS[j] for j in t_ev], args.submission)
    print("wrote", args.submission)
