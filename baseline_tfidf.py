"""TF-IDF + logistic regression baseline, 5-fold CV on train."""
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from common import load, folds, report, EVASION_TO_CLARITY

tr, te = load()
y_ev = tr.evasion_label.values
y_cl = tr.clarity_label.values


def feats(fit_df, dfs):
    vecs = [
        ("interview_answer", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, max_features=200000)),
        ("interview_answer", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("interview_question", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
    ]
    for col, v in vecs:
        v.fit(fit_df[col])
    return [hstack([v.transform(d[col]) for col, v in vecs]).tocsr() for d in dfs]


p_ev = np.empty(len(tr), dtype=object)
p_cl = np.empty(len(tr), dtype=object)
for trn, val in folds(y_ev):
    Xa, Xb = feats(tr.iloc[trn], [tr.iloc[trn], tr.iloc[val]])
    m = LogisticRegression(C=10, max_iter=3000, class_weight="balanced").fit(Xa, y_ev[trn])
    p_ev[val] = m.predict(Xb)
    m2 = LogisticRegression(C=10, max_iter=3000, class_weight="balanced").fit(Xa, y_cl[trn])
    p_cl[val] = m2.predict(Xb)

report("tfidf-LR (clarity mapped from evasion)", y_ev, p_ev)
print("tfidf-LR direct clarity macroF1=%.4f" % f1_score(y_cl, p_cl, average="macro"))
