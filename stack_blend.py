"""Balanced-LR stacker on [recipe log-probs + metadata + text stats], geometric 50/50 blend with the recipe.

  python select_ensemble.py --runs ... --combine geo --dump DUMP.npz --max_steps 1
  python stack_blend.py --dump DUMP.npz --recipe R3 [--out submission.csv]

OOF score is nested (stacker fitted on 4 folds, scored on the 5th), repeated over 5 CV seeds.
Clarity = blended evasion mapped up. Test: stacker refitted on all train rows.
"""
import argparse, sys, numpy as np, pandas as pd
sys.path.insert(0, ".")
from common import load, folds, EVASION_LABELS, CLARITY_LABELS, EVASION_TO_CLARITY
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

ap = argparse.ArgumentParser()
ap.add_argument("--dump", required=True)
ap.add_argument("--recipe", default="R3", help="label prefix in the dump")
ap.add_argument("--w", type=float, default=0.5)
ap.add_argument("--out")
a = ap.parse_args()
d = np.load(a.dump); tr, te = load()
i = [k for k, l in enumerate(d["labels"]) if l.startswith(a.recipe)][0]
pe, pc, tpe, tpc = d[f"pe{i}"], d[f"pc{i}"], d[f"tpe{i}"], d[f"tpc{i}"]
y, ycl = d["y_ev"], d["y_cl"]
EV2CL = np.array([CLARITY_LABELS.index(EVASION_TO_CLARITY[l]) for l in EVASION_LABELS])


def mf(y, p, k):
    cm = np.bincount(y * k + p, minlength=k * k).reshape(k, k); dd = cm.sum(0) + cm.sum(1)
    return np.mean(np.where(dd > 0, 2 * np.diag(cm) / np.maximum(dd, 1), 0.0))


def feats(df):
    q, ans = df.interview_question.fillna(""), df.interview_answer.fillna("")
    f = pd.DataFrame({"loga": np.log1p(ans.str.len()), "logq": np.log1p(q.str.len()), "nq": q.str.count(r"\?"),
                      "a_q": ans.str.strip().str.endswith("?").astype(int), "na": ans.str.count(r"\?"),
                      "multi": q.str.contains(r"două întrebări|a doua întrebare|și dacă|doua intrebari", case=False).astype(int),
                      "nustiu": ans.str.lower().str.contains(r"^\W*nu (?:știu|stiu)").astype(int),
                      "qo": df.question_order, "year": pd.to_datetime(df.date, format="mixed", dayfirst=True).dt.year})
    for p in ["Ion Iliescu", "Traian Basescu", "Klaus Iohannis", "Nicusor Dan", "Ilie Bolojan"]:
        f["p_" + p] = (df.president == p).astype(int)
    for c in ["Interview", "Declaration", "Press Conference"]:
        f["c_" + c] = (df.context == c).astype(int)
    return f.values.astype(float)


X = np.hstack([np.log(pe + 1e-9), np.log(pc + 1e-9), feats(tr)])
XT = np.hstack([np.log(tpe + 1e-9), np.log(tpc + 1e-9), feats(te)])
fit = lambda Xa, ya: (lambda s: (s, LogisticRegression(C=0.05, max_iter=3000, class_weight="balanced").fit(s.transform(Xa), ya)))(StandardScaler().fit(Xa))
blend = lambda P, base: a.w * np.log(P + 1e-9) + (1 - a.w) * np.log(base + 1e-9)
score = lambda e: (mf(ycl, EV2CL[e], 3), mf(y, e, 9))

b = score(pe.argmax(1)); print(f"{d['labels'][i]}  base OOF: clarity {b[0]:.4f} evasion {b[1]:.4f} sum {sum(b):.4f}")
gains = []
for seed in (42, 1, 2, 3, 4):
    fr = folds(tr.evasion_label.values) if seed == 42 else list(StratifiedKFold(5, shuffle=True, random_state=seed).split(y, y))
    P = np.zeros((len(y), 9))
    for trn, val in fr:
        s, m = fit(X[trn], y[trn]); P[val] = m.predict_proba(s.transform(X[val]))
    r = score(blend(P, pe).argmax(1)); gains.append(sum(r) - sum(b))
    print(f"  seed {seed:2d}: clarity {r[0]:.4f} evasion {r[1]:.4f} sum {sum(r):.4f} ({gains[-1]:+.4f})")
print(f"  mean gain {np.mean(gains):+.4f}")

s, m = fit(X, y)
e = blend(m.predict_proba(s.transform(XT)), tpe).argmax(1)
print(f"test: evasion agreement with base {(e == tpe.argmax(1)).mean():.3f}, clarity {(EV2CL[e] == EV2CL[tpe.argmax(1)]).mean():.3f}")
if a.out:
    ev = dict(zip(te.datapointID, [EVASION_LABELS[k] for k in e])); cl = dict(zip(te.datapointID, [CLARITY_LABELS[EV2CL[k]] for k in e]))
    sub = pd.read_csv("data/sample_output.csv")
    sub["answer"] = [cl[x] if t == 1 else ev[x] for t, x in zip(sub.subtaskID, sub.datapointID)]
    assert len(sub) == 592 and set(sub.datapointID) == set(te.datapointID)
    sub.to_csv(a.out, index=False); print("wrote", a.out, sub[sub.subtaskID == 2].answer.value_counts().to_dict())
