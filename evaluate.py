"""Score 5-fold CV runs, pick each config's best epoch, ensemble configs, write submission.

  python evaluate.py runs/                      # score every config found in runs/
  python evaluate.py runs/ --use A B --out submission.csv
A config is the run tag without the _f<fold>_s<seed> suffix, e.g. xlm-roberta-large_p0.
"""
import argparse, glob, os, re
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from common import load, EVASION_LABELS, CLARITY_LABELS, EVASION_TO_CLARITY

ap = argparse.ArgumentParser()
ap.add_argument("runs")
ap.add_argument("--use", nargs="*", help="configs to ensemble (default: all)")
ap.add_argument("--out", default="submission.csv")
args = ap.parse_args()

tr, te = load()
y_ev = np.array([EVASION_LABELS.index(l) for l in tr.evasion_label])
y_cl = np.array([CLARITY_LABELS.index(l) for l in tr.clarity_label])
EV2CL = np.array([CLARITY_LABELS.index(EVASION_TO_CLARITY[l]) for l in EVASION_LABELS])


def clarity_probs(pe, pc):
    agg = np.zeros_like(pc)
    for k in range(9):
        agg[:, EV2CL[k]] += pe[:, k]
    return (agg + pc) / 2  # clarity head averaged with evasion mapped upward


def score(pe, pc):
    return {
        "clar_combined": f1_score(y_cl, clarity_probs(pe, pc).argmax(1), average="macro"),
        "clar_mapped": f1_score(y_cl, EV2CL[pe.argmax(1)], average="macro"),
        "clar_head": f1_score(y_cl, pc.argmax(1), average="macro"),
        "evasion": f1_score(y_ev, pe.argmax(1), average="macro"),
    }


groups = defaultdict(list)
for f in glob.glob(os.path.join(args.runs, "*_f[0-9]_s*.npz")):
    groups[re.sub(r"_f\d_s\d+\.npz$", "", os.path.basename(f))].append(f)

best = {}
for cfg, files in sorted(groups.items()):
    runs = [np.load(f) for f in files]
    covered = np.concatenate([r["val_idx"] for r in runs])
    if len(files) != 5 or len(set(covered)) != len(tr):
        print(f"{cfg}: skipped, {len(files)} folds found")
        continue
    n_ep = sum(k.startswith("val_pe") for k in runs[0].files)
    per_ep = []
    for e in range(n_ep):
        pe, pc = np.zeros((len(tr), 9)), np.zeros((len(tr), 3))
        for r in runs:
            pe[r["val_idx"]], pc[r["val_idx"]] = r[f"val_pe{e}"], r[f"val_pc{e}"]
        tpe = np.mean([r[f"test_pe{e}"] for r in runs], 0)
        tpc = np.mean([r[f"test_pc{e}"] for r in runs], 0)
        s = score(pe, pc)
        per_ep.append((max(s["clar_combined"], s["clar_mapped"], s["clar_head"]) + s["evasion"], e, s, pe, pc, tpe, tpc))
        print(f"{cfg} epoch {e}: " + " ".join(f"{k}={v:.4f}" for k, v in s.items()))
    best[cfg] = max(per_ep, key=lambda t: t[0])
    print(f"{cfg}: best epoch {best[cfg][1]}\n")

use = args.use or list(best)
if not use:
    raise SystemExit("no complete 5-fold config found in " + args.runs)
pe = np.mean([best[c][3] for c in use], 0)
pc = np.mean([best[c][4] for c in use], 0)
s = score(pe, pc)
print(f"ensemble of {use}: " + " ".join(f"{k}={v:.4f}" for k, v in s.items()))
clar_method = max(["clar_combined", "clar_mapped", "clar_head"], key=s.get)
print("clarity method:", clar_method)


def clarity_pred(pe, pc):
    if clar_method == "clar_mapped":
        return EV2CL[pe.argmax(1)]
    return (pc if clar_method == "clar_head" else clarity_probs(pe, pc)).argmax(1)

tpe = np.mean([best[c][5] for c in use], 0)
tpc = np.mean([best[c][6] for c in use], 0)
ids = np.load(groups[use[0]][0])["test_ids"]
assert (ids == te.datapointID.values).all()
clar = dict(zip(ids, [CLARITY_LABELS[j] for j in clarity_pred(tpe, tpc)]))
evas = dict(zip(ids, [EVASION_LABELS[j] for j in tpe.argmax(1)]))

sub = pd.read_csv("data/sample_output.csv")
sub["answer"] = [clar[d] if t == 1 else evas[d] for t, d in zip(sub.subtaskID, sub.datapointID)]
assert len(sub) == 592 and set(sub.datapointID) == set(te.datapointID)
assert sub[sub.subtaskID == 1].answer.isin(CLARITY_LABELS).all() and sub[sub.subtaskID == 2].answer.isin(EVASION_LABELS).all()
sub.to_csv(args.out, index=False)
print("wrote", args.out)
for t in (1, 2):
    print(sub[sub.subtaskID == t].answer.value_counts().to_dict())
