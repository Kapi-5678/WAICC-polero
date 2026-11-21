"""Combine two submission files: clarity rows (subtask 1) from one, evasion rows (subtask 2) from the other.

  python split_submission.py --clarity A.csv --evasion B.csv --out submissions/v4/submission_v4.csv
  python make_submission_py.py submissions/v4/submission_v4.csv

Each subtask is scored on its own rows, so the clarity answers may come from a different ensemble
than the evasion answers (as submission v3 did). Choose each part on its own OOF score, e.g. with
select_ensemble.py --objective clarity / --objective evasion_only, or fixed recipes. The output keeps
data/sample_output.csv's row order and gets the same checks as evaluate.py.
"""
import argparse, os
import pandas as pd
from common import load, CLARITY_LABELS, EVASION_LABELS

ap = argparse.ArgumentParser()
ap.add_argument("--clarity", required=True, help="submission whose subtaskID=1 rows are kept")
ap.add_argument("--evasion", required=True, help="submission whose subtaskID=2 rows are kept")
ap.add_argument("--out", required=True)
args = ap.parse_args()

_, te = load()
key = ["subtaskID", "datapointID"]
a, b = pd.read_csv(args.clarity), pd.read_csv(args.evasion)
for name, d in (("--clarity", a), ("--evasion", b)):
    assert list(d.columns) == ["subtaskID", "datapointID", "answer"] and len(d) == 592, f"{name}: not a 592-row submission"
ans = {**a[a.subtaskID == 1].set_index(key).answer.to_dict(), **b[b.subtaskID == 2].set_index(key).answer.to_dict()}

sub = pd.read_csv("data/sample_output.csv")
sub["answer"] = [ans[(t, d)] for t, d in zip(sub.subtaskID, sub.datapointID)]
assert len(sub) == 592 and set(sub.datapointID) == set(te.datapointID)
assert sub[sub.subtaskID == 1].answer.isin(CLARITY_LABELS).all() and sub[sub.subtaskID == 2].answer.isin(EVASION_LABELS).all()
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
sub.to_csv(args.out, index=False)
print("wrote", args.out, f"(clarity from {args.clarity}, evasion from {args.evasion})")
for t in (1, 2):
    print(" ", sub[sub.subtaskID == t].answer.value_counts().to_dict())
