"""Task 4a: export the OOF errors of the run-1 ensemble for reading, plus a 9x9 confusion matrix.

  python analysis/export_errors.py
  python analysis/export_errors.py --use xlm-roberta-large_p0      # one config instead of the ensemble

Writes analysis/errors.csv (git-ignored: it holds train text; focus classes first) and analysis/confusion.md.
The ensemble and best-epoch choice are the same as evaluate.py (the default --use is submission_v1).
"""
import argparse, glob, os, re, sys
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import load, EVASION_LABELS, CLARITY_LABELS, EVASION_TO_CLARITY

ap = argparse.ArgumentParser()
ap.add_argument("--runs", default="results/run1")
ap.add_argument("--use", nargs="*", default=["xlm-roberta-large_p0", "RoBERT-large_p0", "bert-base-romanian-cased-v1_p0"])
ap.add_argument("--out", default="analysis")
ap.add_argument("--focus", nargs="*", default=["Partial/half-answer", "Deflection", "Implicit", "General"],
                help="gold classes listed first in errors.csv, in this order")
args = ap.parse_args()

tr, _ = load()
y_ev = np.array([EVASION_LABELS.index(l) for l in tr.evasion_label])
y_cl = np.array([CLARITY_LABELS.index(l) for l in tr.clarity_label])
EV2CL = np.array([CLARITY_LABELS.index(EVASION_TO_CLARITY[l]) for l in EVASION_LABELS])
f1 = lambda y, p: f1_score(y, p, average="macro")


def clarity_probs(pe, pc):
    return (np.stack([pe[:, EV2CL == c].sum(1) for c in range(3)], 1) + pc) / 2


groups = defaultdict(list)
for f in glob.glob(os.path.join(args.runs, "*_f[0-9]_s*.npz")):
    groups[re.sub(r"_f\d_s\d+\.npz$", "", os.path.basename(f))].append(f)
pes, pcs = [], []
for cfg in args.use:
    runs = [np.load(f) for f in sorted(groups[cfg])]
    best = None
    for e in range(sum(k.startswith("val_pe") for k in runs[0].files)):
        pe, pc = np.zeros((len(tr), 9)), np.zeros((len(tr), 3))
        for r in runs:
            pe[r["val_idx"]], pc[r["val_idx"]] = r[f"val_pe{e}"], r[f"val_pc{e}"]
        clar = max(f1(y_cl, clarity_probs(pe, pc).argmax(1)), f1(y_cl, EV2CL[pe.argmax(1)]), f1(y_cl, pc.argmax(1)))
        s = clar + f1(y_ev, pe.argmax(1))
        if best is None or s > best[0]:
            best = (s, pe, pc)
    pes.append(best[1]); pcs.append(best[2])
pe, pc = np.mean(pes, 0), np.mean(pcs, 0)
pred = pe.argmax(1)
print(f"ensemble {args.use}: evasion macro-F1 {f1(y_ev, pred):.4f}, clarity (combined) {f1(y_cl, clarity_probs(pe, pc).argmax(1)):.4f}")

cm = pd.crosstab(pd.Categorical(tr.evasion_label, EVASION_LABELS), pd.Categorical([EVASION_LABELS[i] for i in pred], EVASION_LABELS),
                 rownames=["gold"], colnames=["pred"], dropna=False)
tp, n_pred, n_gold = np.diag(cm.values), cm.values.sum(0), cm.values.sum(1)
prec, rec = tp / np.maximum(n_pred, 1), tp / np.maximum(n_gold, 1)
f1c = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
short = ["Expl", "Impl", "Gen", "Part", "Dodg", "Defl", "Decl", "Ign", "Clar"]
md = [f"# Run-1 OOF confusion matrix\n",
      f"Ensemble {', '.join(args.use)} (best epochs, as in `evaluate.py`), out-of-fold over all {len(tr)} train rows.",
      f"Evasion macro-F1 {f1(y_ev, pred):.4f}, clarity macro-F1 (combined probs) {f1(y_cl, clarity_probs(pe, pc).argmax(1)):.4f}.",
      f"Rebuild: `.venv\\Scripts\\python analysis/export_errors.py`\n",
      "Rows are gold, columns predicted. " + ", ".join(f"{a} = {b}" for a, b in zip(short, EVASION_LABELS)) + ".\n",
      "| gold \\ pred | " + " | ".join(short) + " | n |", "|---" * 11 + "|"]
for i, l in enumerate(EVASION_LABELS):
    md.append(f"| {l} | " + " | ".join(f"**{v}**" if j == i else str(v) for j, v in enumerate(cm.values[i])) + f" | {n_gold[i]} |")
md += ["| predicted | " + " | ".join(map(str, n_pred)) + f" | {len(tr)} |", "",
       "## Per class\n", "| class | n | predicted | precision | recall | F1 |", "|---|---|---|---|---|---|"]
for i, l in enumerate(EVASION_LABELS):
    md.append(f"| {l} | {n_gold[i]} | {n_pred[i]} | {prec[i]:.3f} | {rec[i]:.3f} | {f1c[i]:.3f} |")
md.append(f"| **macro** | | | {prec.mean():.3f} | {rec.mean():.3f} | {f1c.mean():.3f} |")
open(os.path.join(args.out, "confusion.md"), "w").write("\n".join(md) + "\n")
print("\n".join(md[-12:]))

wrong = pred != y_ev
order = {l: k for k, l in enumerate(args.focus)}
err = pd.DataFrame({
    "datapointID": tr.datapointID, "president": tr.president, "gold": tr.evasion_label,
    "pred": [EVASION_LABELS[i] for i in pred], "p_pred": pe.max(1).round(3), "p_gold": pe[np.arange(len(tr)), y_ev].round(3),
    "second": [EVASION_LABELS[i] for i in np.argsort(-pe, 1)[:, 1]], "answer_chars": tr.interview_answer.str.len(),
    "prev_question": tr.prev_question, "question": tr.interview_question, "answer": tr.interview_answer,
})[wrong]
err["_rank"] = err.gold.map(order).fillna(len(order))
err = err.sort_values(["_rank", "gold", "pred", "p_pred"], ascending=[True, True, True, False]).drop(columns="_rank")
err.to_csv(os.path.join(args.out, "errors.csv"), index=False)
print(f"{wrong.sum()} errors of {len(tr)} -> {args.out}/errors.csv (focus first: {', '.join(args.focus)})")
pairs = err.groupby(["gold", "pred"]).size().sort_values(ascending=False)
print("biggest confusions (gold -> pred):\n" + pairs.head(12).to_string())
