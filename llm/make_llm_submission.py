"""Turn the Kaggle LLM labels into a 592-row submission in data/sample_output.csv order.

  python llm/make_llm_submission.py LABELS_CSV OUT_CSV [--hybrid] [--v6 PATH]

Rows the LLM left empty (or labelled with something outside the 9 labels) take both answers from v6.
Default: clarity = EVASION_TO_CLARITY[LLM evasion]. --hybrid: LLM evasion, v6 clarity for every row.
"""
import argparse, os, sys
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from common import EVASION_LABELS, EVASION_TO_CLARITY

ap = argparse.ArgumentParser()
ap.add_argument("labels")
ap.add_argument("out")
ap.add_argument("--hybrid", action="store_true")
ap.add_argument("--v6", default=os.path.join(ROOT, "submissions", "v6", "submission_v6.csv"))
a = ap.parse_args()
if os.path.abspath(a.out).lower().startswith(os.path.join(ROOT, "submissions").lower()):
    sys.exit("refusing to write into submissions/")

sample = pd.read_csv(os.path.join(ROOT, "data", "sample_output.csv"))
v6 = pd.read_csv(a.v6).set_index(["subtaskID", "datapointID"]).answer
llm = pd.read_csv(a.labels, keep_default_na=False).set_index("datapointID").evasion_label
llm = llm[llm.isin(EVASION_LABELS)]

ev, cl, n_llm = {}, {}, 0
for d in sample.datapointID.unique():
    if d in llm.index:
        ev[d], n_llm = llm[d], n_llm + 1
        cl[d] = v6[(1, d)] if a.hybrid else EVASION_TO_CLARITY[llm[d]]
    else:
        ev[d], cl[d] = v6[(2, d)], v6[(1, d)]
out = sample[["subtaskID", "datapointID"]].copy()
out["answer"] = [cl[d] if s == 1 else ev[d] for s, d in zip(out.subtaskID, out.datapointID)]
assert len(out) == 592 and out.answer.notna().all()
out.to_csv(a.out, index=False)
n = out.datapointID.nunique()
print(f"wrote {a.out}: {len(out)} rows, LLM evasion on {n_llm}/{n} rows, v6 fill on {n - n_llm}, "
      f"clarity {'v6 (hybrid)' if a.hybrid else 'mapped from evasion'}")
print("evasion:", out[out.subtaskID == 2].answer.value_counts().to_dict())
print("clarity:", out[out.subtaskID == 1].answer.value_counts().to_dict())
