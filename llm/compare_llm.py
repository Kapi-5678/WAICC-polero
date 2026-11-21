"""Compare an LLM's fold-0 evasion predictions with the v1 ensemble's OOF predictions on the same rows.

Usage (repo root): python llm/compare_llm.py path/to/llm_fold0_<model>.csv
The CSV comes from colab/polero_llm_test.ipynb (columns row_idx, gold, pred). Train data only.
"""
import glob, os, re, sys
from collections import defaultdict
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
sys.path.insert(0, os.getcwd())
from common import load, EVASION_LABELS, CLARITY_LABELS, EVASION_TO_CLARITY

V1 = ["xlm-roberta-large_p0", "RoBERT-large_p0", "bert-base-romanian-cased-v1_p0"]
tr, _ = load()
y_ev = np.array([EVASION_LABELS.index(l) for l in tr.evasion_label])
y_cl = np.array([CLARITY_LABELS.index(l) for l in tr.clarity_label])
EV2CL = np.array([CLARITY_LABELS.index(EVASION_TO_CLARITY[l]) for l in EVASION_LABELS])

# v1 ensemble OOF evasion probabilities, each config at its best epoch (same rule as evaluate.py)
groups = defaultdict(list)
for f in glob.glob("results/run1/*_f[0-9]_s*.npz"):
    groups[re.sub(r"_f\d_s\d+\.npz$", "", os.path.basename(f))].append(f)
best = []
for cfg in V1:
    runs = [np.load(f) for f in groups[cfg]]
    n_ep = sum(k.startswith("val_pe") for k in runs[0].files)
    cands = []
    for e in range(n_ep):
        pe, pc = np.zeros((len(tr), 9)), np.zeros((len(tr), 3))
        for r in runs:
            pe[r["val_idx"]], pc[r["val_idx"]] = r[f"val_pe{e}"], r[f"val_pc{e}"]
        agg = np.zeros_like(pc)
        for k in range(9):
            agg[:, EV2CL[k]] += pe[:, k]
        s_cl = max(f1_score(y_cl, ((agg + pc) / 2).argmax(1), average="macro"),
                   f1_score(y_cl, EV2CL[pe.argmax(1)], average="macro"), f1_score(y_cl, pc.argmax(1), average="macro"))
        cands.append((s_cl + f1_score(y_ev, pe.argmax(1), average="macro"), pe))
    best.append(max(cands, key=lambda t: t[0])[1])
ens = np.mean(best, 0).argmax(1)

llm = pd.read_csv(sys.argv[1])
idx = llm.row_idx.values
assert (tr.evasion_label.values[idx] == llm.gold.values).all(), "row_idx does not match train order"
gold = llm.gold.tolist()
preds = {"v1 ensemble (OOF)": [EVASION_LABELS[k] for k in ens[idx]], "LLM": llm.pred.tolist()}

def clar(xs):
    return [EVASION_TO_CLARITY[x] for x in xs]

print(f"{len(idx)} fold-0 train rows\n")
print(f"{'':20s} {'evasion F1':>10s} {'clarity F1':>10s}")
for name, p in preds.items():
    print(f"{name:20s} {f1_score(gold, p, average='macro'):10.4f} {f1_score(clar(gold), clar(p), average='macro'):10.4f}")
print("\nPer-class evasion F1 (n = gold count):")
print(f"{'class':22s} {'n':>4s} " + " ".join(f"{k[:10]:>10s}" for k in ["v1 ens", "LLM"]))
for lab in EVASION_LABELS:
    f = [f1_score([g == lab for g in gold], [x == lab for x in p], zero_division=0) for p in preds.values()]
    print(f"{lab:22s} {gold.count(lab):4d} " + " ".join(f"{v:10.3f}" for v in f))
