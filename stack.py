"""Task 6: learn the model combination with a logistic-regression stacker instead of averaging.

  python stack.py                                        # run-1 configs behind submission v1
  python stack.py --runs results/run1 results/run2 --use run1/xlm-roberta-large_p0 run2/<config>
  python stack.py --submission submissions/v3/submission_v3.csv

Features: the concatenated OOF probabilities (9 evasion + 3 clarity) of each chosen config, at its
best epoch (chosen as in evaluate.py). One multinomial logistic regression is fitted for evasion
and one for clarity. Scores are nested: the stacker is fitted on the OOF rows of 4 folds
(common.folds) and scored on the 5th, rotating through all 5. The baseline is evaluate.py's
plain average (evasion argmax, clarity = clarity head averaged with evasion mapped up).
"""
import argparse, glob, os, re
from collections import defaultdict
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from common import load, folds, write_submission, EVASION_LABELS, CLARITY_LABELS, EVASION_TO_CLARITY

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", default=["results/run1"])
ap.add_argument("--use", nargs="*", help="configs as <run dir name>/<config>; default: the v1 trio for results/run1 "
                "alone, otherwise every complete 5-fold config found")
ap.add_argument("--features", choices=["probs", "logprobs"], default="probs")
ap.add_argument("--C", type=float, nargs="+", default=[0.1, 1.0])
ap.add_argument("--submission", help="write test predictions of the best stacker setting to this path")
args = ap.parse_args()

tr, te = load()
y_ev = np.array([EVASION_LABELS.index(l) for l in tr.evasion_label])
y_cl = np.array([CLARITY_LABELS.index(l) for l in tr.clarity_label])
EV2CL = np.array([CLARITY_LABELS.index(EVASION_TO_CLARITY[l]) for l in EVASION_LABELS])
f1 = lambda y, p: f1_score(y, p, average="macro")


def clarity_probs(pe, pc):  # as in evaluate.py
    return (np.stack([pe[:, EV2CL == c].sum(1) for c in range(3)], 1) + pc) / 2


def best_epoch(files):
    runs = [np.load(f) for f in sorted(files)]
    assert len(runs) == 5 and len(np.unique(np.concatenate([r["val_idx"] for r in runs]))) == len(tr), files[0]
    best = None
    for e in range(sum(k.startswith("val_pe") for k in runs[0].files)):
        pe, pc = np.zeros((len(tr), 9)), np.zeros((len(tr), 3))
        for r in runs:
            pe[r["val_idx"]], pc[r["val_idx"]] = r[f"val_pe{e}"], r[f"val_pc{e}"]
        s = max(f1(y_cl, clarity_probs(pe, pc).argmax(1)), f1(y_cl, EV2CL[pe.argmax(1)]), f1(y_cl, pc.argmax(1))) + f1(y_ev, pe.argmax(1))
        if best is None or s > best[0]:
            tpe, tpc = np.mean([r[f"test_pe{e}"] for r in runs], 0), np.mean([r[f"test_pc{e}"] for r in runs], 0)
            best = (s, e, pe, pc, tpe, tpc, runs[0]["test_ids"])
    return best[1:]


groups = defaultdict(list)
for d in args.runs:
    for f in glob.glob(os.path.join(d, "*_f[0-9]_s*.npz")):
        groups[f"{os.path.basename(os.path.normpath(d))}/{re.sub(r'_f\d_s\d+\.npz$', '', os.path.basename(f))}"].append(f)
V1 = [f"run1/{c}" for c in ["xlm-roberta-large_p0", "RoBERT-large_p0", "bert-base-romanian-cased-v1_p0"]]
complete = sorted(c for c, fs in groups.items() if len(fs) == 5)
use = args.use or (V1 if len(args.runs) == 1 and set(V1) <= set(complete) else complete)
models = {}
for c in use:
    models[c] = best_epoch(groups[c])
    print(f"{c}: best epoch {models[c][0]}")
ids = models[use[0]][5]
assert all((m[5] == ids).all() for m in models.values()) and (ids == te.datapointID.values).all()

feat = (lambda p: np.log(p + 1e-6)) if args.features == "logprobs" else (lambda p: p)
X = np.hstack([feat(np.hstack([m[1], m[2]])) for m in models.values()])
Xt = np.hstack([feat(np.hstack([m[3], m[4]])) for m in models.values()])
pe_avg, pc_avg = np.mean([m[1] for m in models.values()], 0), np.mean([m[2] for m in models.values()], 0)
base_ev, base_cl = f1(y_ev, pe_avg.argmax(1)), f1(y_cl, clarity_probs(pe_avg, pc_avg).argmax(1))
splits = folds(tr.evasion_label.values)


def stacker(C, cw):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight=cw, max_iter=3000))


def nested(y, C, cw):
    proba = np.zeros((len(y), y.max() + 1))
    for trn, val in splits:
        proba[val] = stacker(C, cw).fit(X[trn], y[trn]).predict_proba(X[val])
    return proba


print(f"\n{len(use)} configs, {X.shape[1]} features ({args.features}); nested 5-fold OOF macro-F1")
print(f"  plain average (evaluate.py):            evasion {base_ev:.4f}  clarity {base_cl:.4f}")
results = []
for cw in [None, "balanced"]:
    for C in args.C:
        p_ev, p_cl = nested(y_ev, C, cw), nested(y_cl, C, cw)
        ev, cl = f1(y_ev, p_ev.argmax(1)), f1(y_cl, p_cl.argmax(1))
        cl_map = f1(y_cl, EV2CL[p_ev.argmax(1)])
        fold_gain = [f1(y_ev[v], p_ev[v].argmax(1)) - f1(y_ev[v], pe_avg[v].argmax(1)) for _, v in splits]
        results.append((ev + max(cl, cl_map), cw, C, ev, cl, cl_map))
        print(f"  stack class_weight={str(cw):8s} C={C:<5g}  evasion {ev:.4f} ({ev - base_ev:+.4f}, per fold {np.mean(fold_gain):+.4f} ± {np.std(fold_gain):.4f})"
              f"  clarity {cl:.4f} ({cl - base_cl:+.4f})  clarity mapped {cl_map:.4f}")

_, cw, C, ev, cl, cl_map = max(results, key=lambda r: r[0])
print(f"\nbest: class_weight={cw} C={C}  gain over plain average: evasion {ev - base_ev:+.4f}, clarity {max(cl, cl_map) - base_cl:+.4f}"
      f" ({'mapped from evasion' if cl_map > cl else 'own stacker'})")
if args.submission:
    m_ev, m_cl = stacker(C, cw).fit(X, y_ev), stacker(C, cw).fit(X, y_cl)
    t_ev = m_ev.predict(Xt)
    t_cl = EV2CL[t_ev] if cl_map > cl else m_cl.predict(Xt)
    os.makedirs(os.path.dirname(args.submission) or ".", exist_ok=True)
    write_submission(te, [CLARITY_LABELS[j] for j in t_cl], [EVASION_LABELS[j] for j in t_ev], args.submission)
    print("wrote", args.submission, "| test evasion counts:", {EVASION_LABELS[k]: int((t_ev == k).sum()) for k in range(9)})
