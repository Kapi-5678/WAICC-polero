"""Shared data loading, label taxonomy and CV helpers."""
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold

DATA = "data"

EVASION_TO_CLARITY = {
    "Explicit": "Clear Reply",
    "Implicit": "Ambivalent",
    "General": "Ambivalent",
    "Partial/half-answer": "Ambivalent",
    "Dodging": "Ambivalent",
    "Deflection": "Ambivalent",
    "Declining to answer": "Clear Non-Reply",
    "Claims ignorance": "Clear Non-Reply",
    "Clarification": "Clear Non-Reply",
}
EVASION_LABELS = list(EVASION_TO_CLARITY)
CLARITY_LABELS = ["Clear Reply", "Ambivalent", "Clear Non-Reply"]


def add_context(df):
    """Add the previous question in the same transcript (helps with follow-ups)."""
    df = df.sort_values(["url", "question_order"])
    df["prev_question"] = df.groupby("url")["interview_question"].shift(1).fillna("")
    return df.sort_index()


def load():
    tr = pd.read_csv(f"{DATA}/train_data.csv")
    te = pd.read_csv(f"{DATA}/test_data.csv")
    # context is built over train+test together: test turns sit between train turns
    allx = pd.concat([tr.assign(_split="train"), te.assign(_split="test")], ignore_index=True)
    allx = add_context(allx)
    tr = allx[allx._split == "train"].drop(columns="_split").reset_index(drop=True)
    te = allx[allx._split == "test"].drop(columns="_split").reset_index(drop=True)
    return tr, te


def folds(y, n=5, seed=42):
    return list(StratifiedKFold(n, shuffle=True, random_state=seed).split(y, y))


def report(name, y_ev, p_ev):
    """Scores for both subtasks; clarity is mapped up from evasion predictions."""
    y_cl = [EVASION_TO_CLARITY[v] for v in y_ev]
    p_cl = [EVASION_TO_CLARITY[v] for v in p_ev]
    s = {
        "clarity_macroF1": f1_score(y_cl, p_cl, average="macro"),
        "clarity_acc": accuracy_score(y_cl, p_cl),
        "evasion_macroF1": f1_score(y_ev, p_ev, average="macro"),
        "evasion_acc": accuracy_score(y_ev, p_ev),
    }
    print(name, " ".join(f"{k}={v:.4f}" for k, v in s.items()))
    return s


def write_submission(test, p_cl, p_ev, path):
    rows = [(1, i, c) for i, c in zip(test.datapointID, p_cl)]
    rows += [(2, i, e) for i, e in zip(test.datapointID, p_ev)]
    out = pd.DataFrame(rows, columns=["subtaskID", "datapointID", "answer"])
    out.to_csv(path, index=False)
    return out
