"""PolERo submission: clarity (subtask 1) and evasion (subtask 2) predictions.

  python polero_v7.py           -> writes submission.csv (592 rows) and prints it
  python polero_v7.py unpack    -> writes the training/evaluation source files used to produce it

Standard library only. The pipeline itself: 5-fold fine-tuning of XLM-R large, RoBERT-large and
Romanian BERT with an evasion head and a clarity head, then a probability-averaged ensemble.
"""
import os, sys

PREDICTIONS = r'''subtaskID,datapointID,answer
1,18,Clear Reply
1,23,Clear Reply
1,63,Clear Reply
1,72,Clear Reply
1,89,Ambivalent
1,91,Ambivalent
1,213,Ambivalent
1,226,Clear Reply
1,267,Ambivalent
1,299,Clear Non-Reply
1,332,Clear Reply
1,366,Ambivalent
1,370,Ambivalent
1,386,Ambivalent
1,411,Clear Reply
1,415,Clear Reply
1,449,Clear Reply
1,454,Clear Reply
1,465,Clear Reply
1,488,Clear Reply
1,545,Clear Reply
1,550,Ambivalent
1,608,Clear Non-Reply
1,610,Ambivalent
1,687,Ambivalent
1,701,Clear Reply
1,702,Ambivalent
1,718,Ambivalent
1,733,Clear Reply
1,742,Clear Reply
1,748,Clear Non-Reply
1,752,Clear Reply
1,770,Ambivalent
1,797,Ambivalent
1,822,Clear Reply
1,847,Clear Reply
1,884,Ambivalent
1,887,Clear Reply
1,924,Clear Reply
1,939,Clear Reply
1,1010,Ambivalent
1,1044,Clear Reply
1,1055,Clear Reply
1,1086,Ambivalent
1,1087,Ambivalent
1,1100,Clear Reply
1,1126,Ambivalent
1,1164,Clear Reply
1,1186,Ambivalent
1,1196,Ambivalent
1,1202,Ambivalent
1,1212,Clear Non-Reply
1,1253,Clear Reply
1,1268,Ambivalent
1,1311,Ambivalent
1,1336,Ambivalent
1,1404,Clear Non-Reply
1,1425,Ambivalent
1,1442,Ambivalent
1,1491,Ambivalent
1,1503,Clear Reply
1,1576,Ambivalent
1,1578,Ambivalent
1,1629,Ambivalent
1,1631,Ambivalent
1,1721,Ambivalent
1,1724,Clear Reply
1,1733,Clear Reply
1,1737,Clear Reply
1,1776,Ambivalent
1,1786,Clear Reply
1,1797,Ambivalent
1,1822,Clear Reply
1,1922,Clear Non-Reply
1,2004,Ambivalent
1,2086,Clear Reply
1,2095,Clear Reply
1,2121,Ambivalent
1,2153,Ambivalent
1,2160,Clear Reply
1,2391,Ambivalent
1,2496,Ambivalent
1,2563,Clear Reply
1,2621,Clear Reply
1,2623,Clear Reply
1,2718,Clear Non-Reply
1,2753,Clear Reply
1,2755,Ambivalent
1,2773,Clear Reply
1,2782,Clear Reply
1,2808,Ambivalent
1,2859,Clear Reply
1,2926,Ambivalent
1,2968,Clear Reply
1,3019,Clear Reply
1,3022,Clear Reply
1,3044,Clear Reply
1,3055,Ambivalent
1,3089,Ambivalent
1,3105,Ambivalent
1,3168,Clear Reply
1,3196,Clear Reply
1,3239,Clear Reply
1,3256,Clear Reply
1,3262,Clear Reply
1,3309,Ambivalent
1,3333,Ambivalent
1,3349,Clear Reply
1,3444,Clear Reply
1,3455,Ambivalent
1,3475,Ambivalent
1,3481,Clear Non-Reply
1,3522,Ambivalent
1,3533,Ambivalent
1,3548,Clear Reply
1,3560,Ambivalent
1,3565,Ambivalent
1,3567,Clear Reply
1,3570,Clear Reply
1,3601,Clear Reply
1,3609,Ambivalent
1,3643,Clear Non-Reply
1,3650,Ambivalent
1,3655,Ambivalent
1,3663,Clear Reply
1,3665,Clear Reply
1,3667,Clear Reply
1,3731,Ambivalent
1,3739,Clear Reply
1,3741,Clear Reply
1,3764,Ambivalent
1,3793,Ambivalent
1,3796,Clear Reply
1,3813,Clear Non-Reply
1,3842,Ambivalent
1,3863,Clear Reply
1,3905,Ambivalent
1,3937,Clear Reply
1,3943,Ambivalent
1,3947,Clear Reply
1,3953,Ambivalent
1,3959,Ambivalent
1,3965,Ambivalent
1,4008,Ambivalent
1,4022,Ambivalent
1,4071,Clear Reply
1,4081,Ambivalent
1,4144,Ambivalent
1,4227,Clear Reply
1,4358,Clear Reply
1,4461,Clear Reply
1,4520,Ambivalent
1,4542,Ambivalent
1,4585,Clear Reply
1,4643,Clear Reply
1,4697,Clear Reply
1,4737,Ambivalent
1,4770,Ambivalent
1,4773,Ambivalent
1,4827,Ambivalent
1,4891,Ambivalent
1,4942,Clear Reply
1,4951,Ambivalent
1,4956,Ambivalent
1,4966,Clear Reply
1,4967,Ambivalent
1,4997,Clear Non-Reply
1,5049,Clear Non-Reply
1,5081,Ambivalent
1,5093,Ambivalent
1,5101,Clear Reply
1,5178,Clear Reply
1,5182,Ambivalent
1,5206,Ambivalent
1,5227,Ambivalent
1,5234,Clear Reply
1,5244,Clear Non-Reply
1,5247,Clear Non-Reply
1,5275,Ambivalent
1,5305,Ambivalent
1,5359,Clear Non-Reply
1,5378,Clear Non-Reply
1,5379,Ambivalent
1,5408,Ambivalent
1,5441,Clear Reply
1,5501,Clear Reply
1,5504,Ambivalent
1,5507,Ambivalent
1,5519,Clear Reply
1,5530,Clear Reply
1,5538,Clear Non-Reply
1,5558,Ambivalent
1,5565,Ambivalent
1,5585,Ambivalent
1,5599,Clear Reply
1,5608,Clear Reply
1,5610,Clear Reply
1,5643,Clear Reply
1,5678,Clear Reply
1,5683,Ambivalent
1,5736,Clear Reply
1,5757,Ambivalent
1,5813,Ambivalent
1,5833,Clear Reply
1,5853,Ambivalent
1,5864,Ambivalent
1,5871,Ambivalent
1,5883,Clear Reply
1,5900,Ambivalent
1,5910,Ambivalent
1,5932,Ambivalent
1,5942,Clear Reply
1,5966,Ambivalent
1,5990,Ambivalent
1,5991,Clear Reply
1,6005,Ambivalent
1,6013,Ambivalent
1,6028,Ambivalent
1,6076,Clear Reply
1,6127,Clear Reply
1,6142,Ambivalent
1,6143,Ambivalent
1,6153,Ambivalent
1,6155,Clear Reply
1,6167,Ambivalent
1,6178,Ambivalent
1,6182,Ambivalent
1,6195,Clear Non-Reply
1,6238,Ambivalent
1,6276,Ambivalent
1,6292,Clear Reply
1,6324,Ambivalent
1,6365,Ambivalent
1,6374,Clear Reply
1,6375,Clear Reply
1,6419,Clear Reply
1,6432,Ambivalent
1,6567,Ambivalent
1,6576,Clear Reply
1,6643,Ambivalent
1,6645,Clear Reply
1,6652,Clear Non-Reply
1,6678,Ambivalent
1,6684,Clear Reply
1,6718,Clear Non-Reply
1,6727,Ambivalent
1,6755,Clear Reply
1,6762,Clear Reply
1,6773,Ambivalent
1,6784,Ambivalent
1,6795,Clear Reply
1,6800,Clear Reply
1,6812,Ambivalent
1,6875,Clear Reply
1,6901,Ambivalent
1,6912,Ambivalent
1,6933,Ambivalent
1,6952,Ambivalent
1,6980,Ambivalent
1,6992,Ambivalent
1,6999,Ambivalent
1,7006,Ambivalent
1,7015,Ambivalent
1,7016,Ambivalent
1,7039,Ambivalent
1,7049,Ambivalent
1,7076,Clear Reply
1,7082,Ambivalent
1,7094,Ambivalent
1,7154,Clear Reply
1,7202,Ambivalent
1,7208,Ambivalent
1,7220,Ambivalent
1,7248,Ambivalent
1,7272,Ambivalent
1,7276,Ambivalent
1,7312,Ambivalent
1,7325,Ambivalent
1,7328,Ambivalent
1,7357,Clear Reply
1,7365,Clear Reply
1,7377,Clear Non-Reply
1,7390,Ambivalent
1,7407,Clear Reply
1,7422,Clear Reply
1,7500,Ambivalent
1,7506,Clear Reply
1,7535,Clear Reply
1,7572,Clear Non-Reply
1,7612,Ambivalent
1,7622,Ambivalent
1,7678,Ambivalent
1,3617,Ambivalent
1,4444,Ambivalent
1,5937,Ambivalent
1,7545,Ambivalent
2,18,Explicit
2,23,Explicit
2,63,Explicit
2,72,Explicit
2,89,Implicit
2,91,General
2,213,General
2,226,Explicit
2,267,Deflection
2,299,Claims ignorance
2,332,Explicit
2,366,Deflection
2,370,General
2,386,Deflection
2,411,Explicit
2,415,Explicit
2,449,Explicit
2,454,Explicit
2,465,Explicit
2,488,Explicit
2,545,Explicit
2,550,Dodging
2,608,Claims ignorance
2,610,General
2,687,Deflection
2,701,Explicit
2,702,General
2,718,Dodging
2,733,Explicit
2,742,Explicit
2,748,Clarification
2,752,Explicit
2,770,Dodging
2,797,Deflection
2,822,Explicit
2,847,Explicit
2,884,Dodging
2,887,Explicit
2,924,Explicit
2,939,Explicit
2,1010,General
2,1044,Explicit
2,1055,Explicit
2,1086,General
2,1087,Implicit
2,1100,Explicit
2,1126,Deflection
2,1164,Explicit
2,1186,Deflection
2,1196,Dodging
2,1202,Dodging
2,1212,Claims ignorance
2,1253,Explicit
2,1268,Dodging
2,1311,Deflection
2,1336,Deflection
2,1404,Claims ignorance
2,1425,General
2,1442,Deflection
2,1491,Deflection
2,1503,Explicit
2,1576,Implicit
2,1578,Deflection
2,1629,Dodging
2,1631,Dodging
2,1721,Deflection
2,1724,Explicit
2,1733,Explicit
2,1737,Explicit
2,1776,Deflection
2,1786,Explicit
2,1797,Deflection
2,1822,Explicit
2,1922,Claims ignorance
2,2004,General
2,2086,Explicit
2,2095,Explicit
2,2121,General
2,2153,General
2,2160,Explicit
2,2391,Deflection
2,2496,Deflection
2,2563,Explicit
2,2621,Explicit
2,2623,Explicit
2,2718,Claims ignorance
2,2753,Explicit
2,2755,Deflection
2,2773,Explicit
2,2782,Explicit
2,2808,Deflection
2,2859,Explicit
2,2926,Dodging
2,2968,Explicit
2,3019,Explicit
2,3022,Explicit
2,3044,Explicit
2,3055,Deflection
2,3089,Dodging
2,3105,Deflection
2,3168,Explicit
2,3196,Explicit
2,3239,Explicit
2,3256,Explicit
2,3262,Explicit
2,3309,Partial/half-answer
2,3333,Deflection
2,3349,Explicit
2,3444,Explicit
2,3455,Dodging
2,3475,Dodging
2,3481,Declining to answer
2,3522,Dodging
2,3533,Deflection
2,3548,Explicit
2,3560,General
2,3565,Dodging
2,3567,Explicit
2,3570,Explicit
2,3601,Explicit
2,3609,Implicit
2,3643,Declining to answer
2,3650,Implicit
2,3655,General
2,3663,Explicit
2,3665,Explicit
2,3667,Explicit
2,3731,Deflection
2,3739,Explicit
2,3741,Explicit
2,3764,General
2,3793,Deflection
2,3796,Explicit
2,3813,Clarification
2,3842,Deflection
2,3863,Explicit
2,3905,Deflection
2,3937,Explicit
2,3943,Dodging
2,3947,Explicit
2,3953,Deflection
2,3959,Deflection
2,3965,Implicit
2,4008,Deflection
2,4022,Deflection
2,4071,Explicit
2,4081,Deflection
2,4144,General
2,4227,Explicit
2,4358,Explicit
2,4461,Explicit
2,4520,General
2,4542,Deflection
2,4585,Explicit
2,4643,Explicit
2,4697,Explicit
2,4737,Deflection
2,4770,Deflection
2,4773,Deflection
2,4827,General
2,4891,Dodging
2,4942,Explicit
2,4951,Deflection
2,4956,Implicit
2,4966,Explicit
2,4967,Deflection
2,4997,Declining to answer
2,5049,Declining to answer
2,5081,Implicit
2,5093,Partial/half-answer
2,5101,Explicit
2,5178,Explicit
2,5182,Deflection
2,5206,Deflection
2,5227,General
2,5234,Explicit
2,5244,Declining to answer
2,5247,Declining to answer
2,5275,General
2,5305,Dodging
2,5359,Declining to answer
2,5378,Declining to answer
2,5379,Deflection
2,5408,Deflection
2,5441,Explicit
2,5501,Explicit
2,5504,General
2,5507,Dodging
2,5519,Explicit
2,5530,Explicit
2,5538,Declining to answer
2,5558,General
2,5565,General
2,5585,Deflection
2,5599,Explicit
2,5608,Explicit
2,5610,Explicit
2,5643,Explicit
2,5678,Explicit
2,5683,General
2,5736,Explicit
2,5757,Partial/half-answer
2,5813,Deflection
2,5833,Explicit
2,5853,Dodging
2,5864,Dodging
2,5871,Deflection
2,5883,Explicit
2,5900,Implicit
2,5910,Deflection
2,5932,General
2,5942,Explicit
2,5966,General
2,5990,General
2,5991,Explicit
2,6005,Deflection
2,6013,Deflection
2,6028,Deflection
2,6076,Explicit
2,6127,Explicit
2,6142,General
2,6143,Deflection
2,6153,Deflection
2,6155,Explicit
2,6167,Deflection
2,6178,Deflection
2,6182,Dodging
2,6195,Claims ignorance
2,6238,Deflection
2,6276,Deflection
2,6292,Explicit
2,6324,Deflection
2,6365,Deflection
2,6374,Explicit
2,6375,Explicit
2,6419,Explicit
2,6432,General
2,6567,Dodging
2,6576,Explicit
2,6643,Deflection
2,6645,Explicit
2,6652,Declining to answer
2,6678,Deflection
2,6684,Explicit
2,6718,Claims ignorance
2,6727,General
2,6755,Explicit
2,6762,Explicit
2,6773,General
2,6784,Implicit
2,6795,Explicit
2,6800,Explicit
2,6812,General
2,6875,Explicit
2,6901,Deflection
2,6912,Deflection
2,6933,General
2,6952,Deflection
2,6980,General
2,6992,Dodging
2,6999,Deflection
2,7006,General
2,7015,General
2,7016,Deflection
2,7039,General
2,7049,General
2,7076,Explicit
2,7082,Deflection
2,7094,Deflection
2,7154,Explicit
2,7202,Deflection
2,7208,Partial/half-answer
2,7220,Dodging
2,7248,Dodging
2,7272,Implicit
2,7276,Dodging
2,7312,Dodging
2,7325,General
2,7328,Deflection
2,7357,Explicit
2,7365,Explicit
2,7377,Claims ignorance
2,7390,General
2,7407,Explicit
2,7422,Explicit
2,7500,Deflection
2,7506,Explicit
2,7535,Explicit
2,7572,Claims ignorance
2,7612,Deflection
2,7622,General
2,7678,Deflection
2,3617,Deflection
2,4444,Dodging
2,5937,Dodging
2,7545,Deflection
'''

SOURCES = {
    'common.py': r'''"""Shared data loading, label taxonomy and CV helpers."""
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
''',
    'finetune.py': r'''"""Fine-tune a shared encoder with two heads (evasion 9-way, clarity 3-way).

Input: [prev question SEP] question SEP answer; long answers keep head + tail.
Each run trains one CV fold (or all data with --fold -1) and saves, for every epoch,
the out-of-fold and test probabilities, so the epoch count can be chosen afterwards.

  python finetune.py --model FacebookAI/xlm-roberta-large --fold 0
"""
import argparse, os, time
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

from common import load, folds, EVASION_LABELS, CLARITY_LABELS

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="dumitrescustefan/bert-base-romanian-cased-v1")
ap.add_argument("--fold", type=int, default=0, help="0..4, or -1 to train on all data")
ap.add_argument("--epochs", type=int, default=4)
ap.add_argument("--lr", type=float, default=3e-5)
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--max_len", type=int, default=320)
ap.add_argument("--q_len", type=int, default=80)
ap.add_argument("--q_mode", choices=["head", "tail", "headtail"], default="head",
                help="which part of a long question to keep: head, tail, or the first 32 + last q_len-32 tokens")
ap.add_argument("--president", action="store_true", help="prepend the president's name to the question")
ap.add_argument("--prev_len", type=int, default=0, help="tokens of the previous question to prepend")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--cl_weight", type=float, default=0.5, help="weight of the clarity loss next to the evasion loss")
ap.add_argument("--out", default="runs")
ap.add_argument("--name", default=None, help="config name used in output files (default: model name)")
args = ap.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"
if dev == "cpu":
    torch.set_num_threads(os.cpu_count())

tr, te = load()
ev_id = {l: i for i, l in enumerate(EVASION_LABELS)}
cl_id = {l: i for i, l in enumerate(CLARITY_LABELS)}
tok = AutoTokenizer.from_pretrained(args.model)
CLS, SEP = tok.cls_token_id, tok.sep_token_id


def cut_question(q):
    """Journalists often ask at the end of a long preamble, so 'tail' and 'headtail' keep the end."""
    if len(q) <= args.q_len:
        return q
    if args.q_mode == "tail":
        return q[-args.q_len :]
    if args.q_mode == "headtail":
        return q[:32] + q[-(args.q_len - 32) :]
    return q[: args.q_len]


def encode(df):
    ids = []
    for pres, p, q, a in zip(df.president, df.prev_question, df.interview_question, df.interview_answer):
        q = cut_question(tok.encode(str(q), add_special_tokens=False))
        if args.prev_len:
            q = tok.encode(str(p), add_special_tokens=False)[-args.prev_len :] + [SEP] + q
        if args.president:
            q = tok.encode(str(pres), add_special_tokens=False) + [SEP] + q
        a = tok.encode(str(a), add_special_tokens=False)
        room = args.max_len - 3 - len(q)
        if len(a) > room:  # keep the start and the end of the answer
            head = room * 2 // 3
            a = a[:head] + a[-(room - head) :]
        ids.append([CLS] + q + [SEP] + a + [SEP])
    return ids


def batches(ids, idx, bs):
    for s in range(0, len(idx), bs):
        b = idx[s : s + bs]
        L = max(len(ids[i]) for i in b)
        x = torch.full((len(b), L), tok.pad_token_id)
        m = torch.zeros((len(b), L), dtype=torch.long)
        for j, i in enumerate(b):
            x[j, : len(ids[i])] = torch.tensor(ids[i])
            m[j, : len(ids[i])] = 1
        yield b, x.to(dev), m.to(dev)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = AutoModel.from_pretrained(args.model)
        h = self.enc.config.hidden_size
        self.drop = nn.Dropout(0.1)
        self.ev = nn.Linear(h, len(EVASION_LABELS))
        self.cl = nn.Linear(h, len(CLARITY_LABELS))

    def forward(self, x, m):
        hs = self.enc(input_ids=x, attention_mask=m).last_hidden_state
        mf = m.unsqueeze(-1).to(hs.dtype)
        pooled = self.drop((hs * mf).sum(1) / mf.sum(1))  # mean pooling
        return self.ev(pooled), self.cl(pooled)


def class_weights(labels, n):
    c = np.bincount(labels, minlength=n).astype(float)
    w = (c.sum() / np.maximum(c, 1)) ** 0.5  # sqrt inverse frequency
    return torch.tensor(w / w.mean(), dtype=torch.float, device=dev)


def predict(net, ids, idx):
    net.eval()
    pe, pc = np.zeros((len(idx), 9)), np.zeros((len(idx), 3))
    pos = {i: k for k, i in enumerate(idx)}
    with torch.no_grad(), torch.autocast(dev, enabled=dev == "cuda"):
        for b, x, m in batches(ids, idx, 64):
            le, lc = net(x, m)
            k = [pos[i] for i in b]
            pe[k], pc[k] = le.float().softmax(-1).cpu().numpy(), lc.float().softmax(-1).cpu().numpy()
    return pe, pc


tr_ids, te_ids = encode(tr), encode(te)
yev = np.array([ev_id[l] for l in tr.evasion_label])
ycl = np.array([cl_id[l] for l in tr.clarity_label])
if args.fold >= 0:
    trn, val = folds(tr.evasion_label.values)[args.fold]
else:
    trn, val = np.arange(len(tr)), None
te_idx = np.arange(len(te))

net = Net().to(dev)
loss_ev = nn.CrossEntropyLoss(weight=class_weights(yev[trn], 9))
loss_cl = nn.CrossEntropyLoss(weight=class_weights(ycl[trn], 3))
opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)
steps = args.epochs * ((len(trn) + args.bs - 1) // args.bs)
sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
scaler = torch.amp.GradScaler(enabled=dev == "cuda")

tag = f"{args.name or args.model.split('/')[-1]}_p{args.prev_len}_f{args.fold}_s{args.seed}"
os.makedirs(args.out, exist_ok=True)
res = {"val_idx": val if val is not None else np.array([], int), "test_ids": te.datapointID.values}
for ep in range(args.epochs):
    net.train()
    rng = np.random.default_rng(args.seed + ep)
    order = rng.permutation(trn)
    # sort shuffled chunks by length to cut padding, then shuffle the batches
    order = np.concatenate([sorted(order[i : i + args.bs * 20], key=lambda i: len(tr_ids[i])) for i in range(0, len(order), args.bs * 20)])
    bl = list(batches(tr_ids, order, args.bs))
    rng.shuffle(bl)
    t0, tot = time.time(), 0.0
    for b, x, m in bl:
        with torch.autocast(dev, enabled=dev == "cuda"):
            le, lc = net(x, m)
            loss = loss_ev(le.float(), torch.tensor(yev[b], device=dev)) + args.cl_weight * loss_cl(lc.float(), torch.tensor(ycl[b], device=dev))
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad()
        tot += loss.item()
    msg = f"{tag} epoch {ep} loss {tot/len(bl):.3f} {time.time()-t0:.0f}s"
    if val is not None:
        res[f"val_pe{ep}"], res[f"val_pc{ep}"] = predict(net, tr_ids, val)
        from sklearn.metrics import f1_score
        msg += " val evasion macroF1 %.4f" % f1_score(yev[val], res[f"val_pe{ep}"].argmax(1), average="macro")
    res[f"test_pe{ep}"], res[f"test_pc{ep}"] = predict(net, te_ids, te_idx)
    print(msg, flush=True)

np.savez(f"{args.out}/{tag}.npz", **res)
print("wrote", f"{args.out}/{tag}.npz", flush=True)
''',
    'evaluate.py': r'''"""Score 5-fold CV runs, pick each config's best epoch, ensemble configs, write submission.

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
''',
    'select_ensemble.py': r'''"""Task 7: greedy forward ensemble selection with replacement (Caruana et al.), with a nested check.

  python select_ensemble.py --runs results/run1
  python select_ensemble.py --runs results/run1 results/run2
  python select_ensemble.py --runs results/run1 results/run2 --submission submissions/v2/submission_v2.csv
  python make_submission_py.py submissions/v2/submission_v2.csv

Every complete 5-fold config in the run folders is taken at its best epoch (as evaluate.py
chooses it) and named <run folder>/<config>. Starting from an empty bag, the config whose
addition most improves the objective of the probability average is added, repeatedly (a config
may be added more than once); selection stops when nothing improves.
Objective (default --objective evasion, Rares's rule): maximise OOF evasion macro-F1 while
clarity stays within --max_clarity_drop (0.01) of the reference, the v1 trio (or, without run 1,
the best single config). --objective sum maximises clarity + evasion instead.
Clarity is the best of combined / mapped / head, as in evaluate.py.
Nested check: the selection is rerun on 4 folds' OOF rows (common.folds) and scored on the 5th.
Fixed recipes (--recipe, default RECIPES below) average every config whose name matches a pattern,
with equal weights and no selection, so their OOF score needs no nesting. They are defined before
runs 3-6 land; matching configs from later runs join automatically.
--combine (default plain): how member probabilities are averaged.
  plain  arithmetic mean of probabilities (as evaluate.py)
  geo    geometric mean: mean of log-probabilities, renormalised
  temp   per-config temperature first, p ~ p_raw^(1/T), T fitted by OOF log-loss on a grid
         (separately for the evasion and clarity heads), nested: each fold's OOF rows use the T
         fitted on the other 4 folds; test uses the T fitted on all rows. Then the plain mean.
"""
import argparse, fnmatch, glob, os, re
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from common import load, folds, EVASION_LABELS, CLARITY_LABELS, EVASION_TO_CLARITY

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", default=["results/run1"])
ap.add_argument("--max_steps", type=int, default=30)
ap.add_argument("--objective", choices=["sum", "clarity", "evasion_only", "evasion"], default="sum",
                help="sum: clarity + evasion (default, Rares after the v3 probe); clarity / evasion_only: choose a bag "
                     "for one subtask only (combine two with split_submission.py); evasion: evasion first with "
                     "clarity within --max_clarity_drop of v1 (the earlier rule)")
ap.add_argument("--max_clarity_drop", type=float, default=0.01)
ap.add_argument("--submission", help="write the selected ensemble's test predictions to this csv")
ap.add_argument("--recipe", action="append", metavar="LABEL=PATTERN[,PATTERN]",
                help="fixed equal-weight average of the configs matching the patterns (repeatable); "
                     "default: the RECIPES below")
ap.add_argument("--combine", choices=["plain", "geo", "temp"], default="plain")
ap.add_argument("--epoch_avg", type=float, default=0, help="0: each config at its best epoch (default). "
                "f > 0: average the probabilities of its last max(2, round(f * n_epochs)) epochs instead (fixed rule, no epoch pick)")
ap.add_argument("--dump", metavar="NPZ", help="save every fixed recipe's OOF and test probabilities (for pick_recipes.py)")
ap.add_argument("--submit_recipe", metavar="LABEL", help="with --submission: write this recipe's predictions instead of the greedy bag")
args = ap.parse_args()

tr, te = load()
y_ev = np.array([EVASION_LABELS.index(l) for l in tr.evasion_label])
y_cl = np.array([CLARITY_LABELS.index(l) for l in tr.clarity_label])
EV2CL = np.array([CLARITY_LABELS.index(EVASION_TO_CLARITY[l]) for l in EVASION_LABELS])
METHODS = ["combined", "mapped", "head"]


def mf(y, p, k):  # macro-F1 over k classes (equals sklearn's when every class occurs in y)
    cm = np.bincount(y * k + p, minlength=k * k).reshape(k, k)
    d = cm.sum(0) + cm.sum(1)
    return np.mean(np.where(d > 0, 2 * np.diag(cm) / np.maximum(d, 1), 0.0))


def clarity_pred(pe, pc, method):
    if method == "mapped":
        return EV2CL[pe.argmax(1)]
    if method == "head":
        return pc.argmax(1)
    agg = np.stack([pe[:, EV2CL == c].sum(1) for c in range(3)], 1)
    return ((agg + pc) / 2).argmax(1)  # combined, as in evaluate.py


def score(pe, pc, rows=None, method=None):
    """(clarity, evasion, clarity method) on the given rows; the method is chosen unless given."""
    r = slice(None) if rows is None else rows
    ev = mf(y_ev[r], pe[r].argmax(1), 9)
    cands = [method] if method else METHODS
    cl, m = max((mf(y_cl[r], clarity_pred(pe[r], pc[r], m), 3), m) for m in cands)
    return cl, ev, m


# ---- load every complete config at its best epoch
groups = defaultdict(list)
for d in args.runs:
    for f in glob.glob(os.path.join(d, "*_f[0-9]_s*.npz")):
        groups[f"{os.path.basename(os.path.normpath(d))}/{re.sub(r'_f\d_s\d+\.npz$', '', os.path.basename(f))}"].append(f)
names, PE, PC, TPE, TPC = [], [], [], [], []
for c in sorted(groups):
    runs = [np.load(f) for f in sorted(groups[c])]
    if len(runs) != 5 or len(np.unique(np.concatenate([r["val_idx"] for r in runs]))) != len(tr):
        print(f"{c}: skipped, {len(runs)} folds")
        continue
    assert all((r["test_ids"] == te.datapointID.values).all() for r in runs), c
    n_ep = sum(k.startswith("val_pe") for k in runs[0].files)
    if args.epoch_avg:
        eps = range(n_ep - max(2, round(args.epoch_avg * n_ep)), n_ep)
        pe, pc = np.zeros((len(tr), 9)), np.zeros((len(tr), 3))
        for r in runs:
            pe[r["val_idx"]] = np.mean([r[f"val_pe{e}"] for e in eps], 0)
            pc[r["val_idx"]] = np.mean([r[f"val_pc{e}"] for e in eps], 0)
        names.append(c); PE.append(pe); PC.append(pc)
        TPE.append(np.mean([r[f"test_pe{e}"] for r in runs for e in eps], 0)); TPC.append(np.mean([r[f"test_pc{e}"] for r in runs for e in eps], 0))
        print(f"{c}: epochs {eps.start}-{eps.stop - 1} of {n_ep} averaged")
        continue
    best = None
    for e in range(n_ep):
        pe, pc = np.zeros((len(tr), 9)), np.zeros((len(tr), 3))
        for r in runs:
            pe[r["val_idx"]], pc[r["val_idx"]] = r[f"val_pe{e}"], r[f"val_pc{e}"]
        cl, ev, _ = score(pe, pc)
        if best is None or cl + ev > best[0]:
            best = (cl + ev, e, pe, pc, np.mean([r[f"test_pe{e}"] for r in runs], 0), np.mean([r[f"test_pc{e}"] for r in runs], 0))
    names.append(c); PE.append(best[2]); PC.append(best[3]); TPE.append(best[4]); TPC.append(best[5])
    print(f"{c}: best epoch {best[1]}")
PE, PC, TPE, TPC = map(np.array, (PE, PC, TPE, TPC))
K = len(names)
TGRID = np.round(np.arange(0.5, 5.01, 0.1), 2)
PE_RAW, PC_RAW = PE.copy(), PC.copy()  # the submitted v1 is a plain average of these


def temper(P, T):
    lp = np.log(np.clip(P, 1e-12, 1)) / T
    lp -= lp.max(-1, keepdims=True)
    q = np.exp(lp)
    return q / q.sum(-1, keepdims=True)


def fit_T(P, y):
    return min(TGRID, key=lambda T: -np.log(np.clip(temper(P, T)[np.arange(len(y)), y], 1e-12, 1)).mean())


if args.combine == "temp":  # nested temperature per config and head
    fold_rows = folds(tr.evasion_label.values)
    Ts = {}
    for k in range(K):
        for P, TP, y, head in ((PE, TPE, y_ev, "ev"), (PC, TPC, y_cl, "cl")):
            Pk = P[k].copy()
            for trn, val in fold_rows:
                Pk[val] = temper(P[k][val], fit_T(P[k][trn], y[trn]))
            T_all = fit_T(P[k], y)
            P[k], TP[k] = Pk, temper(TP[k], T_all)
            Ts[(names[k], head)] = T_all
    print("temperatures fitted on all rows (evasion / clarity):")
    for n in names:
        print(f"  {n}: {Ts[(n, 'ev')]:.1f} / {Ts[(n, 'cl')]:.1f}")


def blend(w, P):
    if args.combine == "geo":
        lp = np.tensordot(w / w.sum(), np.log(np.clip(P, 1e-12, 1)), axes=1)
        q = np.exp(lp - lp.max(-1, keepdims=True))
        return q / q.sum(-1, keepdims=True)
    return np.tensordot(w / w.sum(), P, axes=1)


V1 = [f"run1/{c}" for c in ["xlm-roberta-large_p0", "RoBERT-large_p0", "bert-base-romanian-cased-v1_p0"]]
W_V1 = np.array([float(n in V1) for n in names]) if set(V1) <= set(names) else None


def ref_clarity(rows):
    """Clarity the selection must stay close to: v1's, else the best single config's."""
    if W_V1 is not None:
        return score(blend(W_V1, PE), blend(W_V1, PC), rows)[0]
    return max(score(PE[k], PC[k], rows)[0] for k in range(K))


def objective(cl, ev, ref):
    if args.objective == "sum":
        return (cl + ev,)
    if args.objective == "clarity":
        return (cl, ev)
    if args.objective == "evasion_only":
        return (ev, cl)
    return (cl >= ref - args.max_clarity_drop - 1e-12, ev, cl)  # feasible first, then evasion, then clarity


def greedy(rows):
    """Forward selection with replacement on the given rows; returns the count vector."""
    ref = ref_clarity(rows)
    w, best = np.zeros(K), None
    for _ in range(args.max_steps):
        trials = []
        for k in range(K):
            w2 = w.copy()
            w2[k] += 1
            cl, ev, _ = score(blend(w2, PE), blend(w2, PC), rows)
            trials.append((objective(cl, ev, ref), -w2.sum(), k))
        o, _, k = max(trials)
        if best is not None and o <= best:  # no strict improvement
            break
        best, w[k] = o, w[k] + 1
    return w


def row(label, w):
    cl, ev, m = score(blend(w, PE), blend(w, PC))
    return (label, cl, ev, cl + ev, m)


# fixed recipes, defined before runs 3-6 (large = XLM-R large / RoBERT-large at lr 1e-5, any seed or run)
LARGE = ["run1/xlm-roberta-large_p0", "run1/RoBERT-large_p0", "*/r[0-9]-xlmr-large-lr1e-5*", "*/r[0-9]-robert-large-lr1e-5*"]
RECIPES = {
    "R1 all large lr1e-5 models": LARGE,
    "R2 R1 + v1's RO-BERT base": LARGE + ["run1/bert-base-romanian-cased-v1_p0"],
    "R3 R1 + run-3 headtail variants": LARGE + ["*/r[0-9]-robert-large-headtail*"],
    "R4 all XLM-R large models": ["run1/xlm-roberta-large_p0", "*/r[0-9]-xlmr-large-lr1e-5*"],
    "T2 run-2 trio": ["run2/r2-xlmr-large-lr1e-5_p0", "run2/r2-robert-large-lr1e-5_p0", "run2/r2-robert-base-lr2e-5_p0"],
    "T12 v1 trio + run-2 trio": ["run1/xlm-roberta-large_p0", "run1/RoBERT-large_p0", "run1/bert-base-romanian-cased-v1_p0",
                                 "run2/r2-xlmr-large-lr1e-5_p0", "run2/r2-robert-large-lr1e-5_p0", "run2/r2-robert-base-lr2e-5_p0"],
    # pre-set before runs 4-6 land (21 Nov ~18:00): v2's six models plus every later large lr 1e-5 seed; use with --combine geo
    "V4 T12 + new large seeds": ["run1/xlm-roberta-large_p0", "run1/RoBERT-large_p0", "run1/bert-base-romanian-cased-v1_p0",
                                 "run2/r2-xlmr-large-lr1e-5_p0", "run2/r2-robert-large-lr1e-5_p0", "run2/r2-robert-base-lr2e-5_p0",
                                 "*/r[4-9]-xlmr-large-lr1e-5*", "*/r[4-9]-robert-large-lr1e-5*"],
}
if args.recipe:
    RECIPES = {r.split("=", 1)[0]: r.split("=", 1)[1].split(",") for r in args.recipe}
W_REC = {}
for label, pats in RECIPES.items():
    w = np.array([float(any(fnmatch.fnmatch(n, p) for p in pats)) for n in names])
    if w.sum():
        W_REC[label] = w

table = []
if W_V1 is not None:
    table.append(row("v1 trio (submission v1)", W_V1))
for label, w in W_REC.items():
    table.append(row(f"{label} ({int(w.sum())})", w))
table.append(row(f"all {K} configs averaged", np.ones(K)))
w_sel = greedy(slice(None))
table.append(row("greedy selection, in-sample", w_sel))

# nested: select on 4 folds, apply the bag (and its clarity method) to the 5th
splits = folds(tr.evasion_label.values)
p_ev, p_cl, bags = np.empty(len(tr), int), np.empty(len(tr), int), []
for trn, val in splits:
    w = greedy(trn)
    pe, pc = blend(w, PE), blend(w, PC)
    m = score(pe, pc, trn)[2]
    p_ev[val], p_cl[val] = pe[val].argmax(1), clarity_pred(pe[val], pc[val], m)
    bags.append(w)
n_cl, n_ev = mf(y_cl, p_cl, 3), mf(y_ev, p_ev, 9)
table.append(("greedy selection, nested", n_cl, n_ev, n_cl + n_ev, "per fold"))
for k, n in enumerate(names):
    table.append(row(n, np.eye(K)[k]))

goal = ("evasion first, clarity within " + f"{args.max_clarity_drop:g} of " + ("v1" if W_V1 is not None else "the best single config")) if args.objective == "evasion" else "clarity + evasion"
print(f"\n{K} configs from {', '.join(args.runs)}; combine: {args.combine}; selection objective: {goal}. OOF macro-F1 over {len(tr)} train rows:\n")
print(f"| {'ensemble':52s} | clarity | evasion | sum    | clarity method |")
print(f"|{'-' * 54}|---------|---------|--------|----------------|")
for label, cl, ev, s, m in table:
    print(f"| {label:52s} | {cl:.4f}  | {ev:.4f}  | {s:.4f} | {m:14s} |")
ins = [t for t in table if t[0].startswith("greedy selection, in")][0][3]
print(f"\nin-sample selection overstates the nested score by {ins - n_cl - n_ev:+.4f} (sum)")
print("selected bag (in-sample):", {names[k]: int(c) for k, c in enumerate(w_sel) if c})
for i, b in enumerate(bags):
    print(f"  nested fold {i} bag:", {names[k]: int(c) for k, c in enumerate(b) if c})
if W_V1 is not None:
    w1 = W_V1 / W_V1.sum()  # submitted v1: plain mean of the raw probabilities, whatever --combine is
    v1_cl, v1_ev, _ = score(np.tensordot(w1, PE_RAW, axes=1), np.tensordot(w1, PC_RAW, axes=1))
    v1_sum = v1_cl + v1_ev
    print(f"reference = submitted v1 (plain average): clarity {v1_cl:.4f}, evasion {v1_ev:.4f}")
    print(f"nested greedy vs v1: evasion {n_ev - v1_ev:+.4f}, clarity {n_cl - v1_cl:+.4f}, sum {n_cl + n_ev - v1_sum:+.4f}")
    if args.objective == "evasion":
        ok = n_ev - v1_ev >= 0.02 and n_cl - v1_cl >= -args.max_clarity_drop
        print(f"bar for a new submission: nested evasion +0.02 with clarity drop <= {args.max_clarity_drop:g} -> {'MET' if ok else 'not met'}")
    elif args.objective in ("clarity", "evasion_only"):
        part, gain = ("clarity", n_cl - v1_cl) if args.objective == "clarity" else ("evasion", n_ev - v1_ev)
        print(f"objective {args.objective}: nested {part} {gain:+.4f} vs v1. Use this bag only for the {part} rows; "
              f"a split submission's nested sum = nested clarity of the clarity bag + nested evasion of the evasion bag (bar +0.02)")
    else:
        print(f"bar for a new submission: nested sum +0.02 -> {'MET' if n_cl + n_ev - v1_sum >= 0.02 else 'not met'}")

if W_REC:
    print("\nfixed recipes (no selection, so OOF needs no nesting):")
    for label, w in W_REC.items():
        cl, ev, _ = score(blend(w, PE), blend(w, PC))
        vs = f"  vs v1: evasion {ev - v1_ev:+.4f}, clarity {cl - v1_cl:+.4f}" if W_V1 is not None else ""
        print(f"  {label}: {[n for n, x in zip(names, w) if x]}{vs}")

if args.dump:
    out = {"y_ev": y_ev, "y_cl": y_cl, "test_ids": te.datapointID.values, "labels": np.array(list(W_REC))}
    for i, (label, w) in enumerate(W_REC.items()):
        out[f"pe{i}"], out[f"pc{i}"] = blend(w, PE), blend(w, PC)
        out[f"tpe{i}"], out[f"tpc{i}"] = blend(w, TPE), blend(w, TPC)
    if W_V1 is not None:  # the submitted v1, always as a plain average of the raw probabilities
        w1 = W_V1 / W_V1.sum()
        out["v1_pe"], out["v1_pc"] = np.tensordot(w1, PE_RAW, axes=1), np.tensordot(w1, PC_RAW, axes=1)
    np.savez(args.dump, combine=args.combine, **out)
    print("wrote", args.dump)

if args.submission:
    w_out = W_REC[args.submit_recipe] if args.submit_recipe else w_sel
    tpe, tpc = blend(w_out, TPE), blend(w_out, TPC)
    m = score(blend(w_out, PE), blend(w_out, PC))[2]
    clar = dict(zip(te.datapointID, [CLARITY_LABELS[j] for j in clarity_pred(tpe, tpc, m)]))
    evas = dict(zip(te.datapointID, [EVASION_LABELS[j] for j in tpe.argmax(1)]))
    sub = pd.read_csv("data/sample_output.csv")
    sub["answer"] = [clar[d] if t == 1 else evas[d] for t, d in zip(sub.subtaskID, sub.datapointID)]
    assert len(sub) == 592 and set(sub.datapointID) == set(te.datapointID)
    assert sub[sub.subtaskID == 1].answer.isin(CLARITY_LABELS).all() and sub[sub.subtaskID == 2].answer.isin(EVASION_LABELS).all()
    os.makedirs(os.path.dirname(args.submission) or ".", exist_ok=True)
    sub.to_csv(args.submission, index=False)
    print("wrote", args.submission, f"(clarity method: {m})")
    for t in (1, 2):
        print(" ", sub[sub.subtaskID == t].answer.value_counts().to_dict())
''',
    'stack_blend.py': r'''"""Balanced-LR stacker on [recipe log-probs + metadata + text stats], geometric 50/50 blend with the recipe.

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
''',
    'kaggle/run.py': r'''"""Kaggle entry point: run every (config, fold) job, spread over the available GPUs.

The attached dataset holds the code files and the competition CSVs.
Outputs land in /kaggle/working/runs/*.npz plus one log per job.
"""
import glob, os, shutil, subprocess, sys, time
import torch

src = glob.glob("/kaggle/input/**/finetune.py", recursive=True)[0].rsplit("/", 1)[0]
os.chdir("/kaggle/working")
os.makedirs("data", exist_ok=True)
os.makedirs("runs", exist_ok=True)
for f in glob.glob(f"{src}/*.py"):
    shutil.copy(f, ".")
for f in glob.glob(f"{src}/*.csv"):
    shutil.copy(f, "data/")

# Run 1's configs are in results/run1/SCORES.md. RUN picks which list this notebook trains.
LONG = ["--epochs", "10", "--max_len", "512", "--q_len", "128", "--bs", "8"]
ROBERT = ["--model", "readerbench/RoBERT-large", "--lr", "1e-5", *LONG]
RUNS = {
    # Run 2: the paper's settings (longer training, 512 tokens, batch 8)
    "run2": [
        ["--name", "r2-robert-large-lr1e-5", *ROBERT],
        ["--name", "r2-xlmr-large-lr1e-5", "--model", "FacebookAI/xlm-roberta-large", "--lr", "1e-5", *LONG],
        ["--name", "r2-robert-large-lr2e-5", "--model", "readerbench/RoBERT-large", "--lr", "2e-5", *LONG],
        ["--name", "r2-robert-base-lr2e-5", "--model", "dumitrescustefan/bert-base-romanian-cased-v1", "--lr", "2e-5", *LONG],
    ],
    # Run 3: r2-robert-large-lr1e-5 plus task 4's fixes; compare each against it (same folds and seed)
    "run3": [
        ["--name", "r3-robert-large-headtail", *ROBERT, "--q_mode", "headtail"],
        ["--name", "r3-robert-large-headtail-pres", *ROBERT, "--q_mode", "headtail", "--president"],
    ],
    # Run 4 (Colleague 2's Kaggle account): run 2's two large lr1e-5 configs with a second seed, for the ensemble
    "run4": [
        ["--name", "r4-xlmr-large-lr1e-5-s43", "--model", "FacebookAI/xlm-roberta-large", "--lr", "1e-5", *LONG, "--seed", "43"],
        ["--name", "r4-robert-large-lr1e-5-s43", *ROBERT, "--seed", "43"],
    ],
    # Run 5 (Colleague 1's Kaggle account): the same two configs with a third seed
    "run5": [
        ["--name", "r5-xlmr-large-lr1e-5-s44", "--model", "FacebookAI/xlm-roberta-large", "--lr", "1e-5", *LONG, "--seed", "44"],
        ["--name", "r5-robert-large-lr1e-5-s44", *ROBERT, "--seed", "44"],
    ],
    # Run 7 (Colleague 2's Kaggle, started 18:29 by Neagu): two more large lr 1e-5 seeds, they join recipe V4
    "run7": [
        ["--name", "r7-robert-large-lr1e-5-s45", *ROBERT, "--seed", "45"],
        ["--name", "r7-xlmr-large-lr1e-5-s46", "--model", "FacebookAI/xlm-roberta-large", "--lr", "1e-5", *LONG, "--seed", "46"],
    ],
    # Run 8 (Colleague 1's Kaggle): clarity loss weighted 1.0 instead of 0.5, seed 42, so each config
    # compares directly with its run-2 twin (r2-xlmr-large-lr1e-5 / r2-robert-large-lr1e-5)
    "run8": [
        ["--name", "r8-xlmr-large-cl1", "--model", "FacebookAI/xlm-roberta-large", "--lr", "1e-5", *LONG, "--cl_weight", "1.0"],
        ["--name", "r8-robert-large-cl1", *ROBERT, "--cl_weight", "1.0"],
    ],
    # Run 8b (our Kaggle, ~00:00): run 8 again (its results are stuck in Neagu's account), 7 epochs to save an hour.
    # Every large model peaked around epoch 7 of 10. The later --epochs overrides LONG's.
    "run8b": [
        ["--name", "r8b-xlmr-large-cl1-e7", "--model", "FacebookAI/xlm-roberta-large", "--lr", "1e-5", *LONG, "--cl_weight", "1.0", "--epochs", "7"],
        ["--name", "r8b-robert-large-cl1-e7", *ROBERT, "--cl_weight", "1.0", "--epochs", "7"],
    ],
}
RUN = "run8b"
CONFIGS = RUNS[RUN]
jobs = [cfg + ["--fold", str(f)] for cfg in CONFIGS for f in range(5)]
n_gpu = max(torch.cuda.device_count(), 1)
print(f"{len(jobs)} jobs on {n_gpu} GPU(s)", flush=True)

running = {}  # gpu -> (proc, job, start)
t0 = time.time()
while jobs or running:
    for g in range(n_gpu):
        if g not in running and jobs:
            job = jobs.pop(0)
            name = f"{job[job.index('--name') + 1]}_f{job[job.index('--fold') + 1]}"  # flags like --president take no value
            log = open(f"runs/{name}.log", "w")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), TOKENIZERS_PARALLELISM="false")
            p = subprocess.Popen([sys.executable, "finetune.py", *job], stdout=log, stderr=subprocess.STDOUT, env=env)
            running[g] = (p, name, time.time())
    for g, (p, name, ts) in list(running.items()):
        if p.poll() is not None:
            status = "ok" if p.returncode == 0 else f"FAILED ({p.returncode})"
            print(f"[{(time.time()-t0)/60:.0f} min] {name}: {status} in {(time.time()-ts)/60:.1f} min", flush=True)
            del running[g]
    time.sleep(10)

subprocess.run([sys.executable, "evaluate.py", "runs", "--out", "submission.csv"])
''',
    'llm/qwen36_test_pass.py': r'''"""PolERo LLM pass on Kaggle T4 x2: few-shot Qwen3.6-35B-A3B (GGUF Q4_K_M) on llama-server, reasoning on.

Mirrors the paper's few-shot setup (Appendix D.2) as far as it is described: Romanian prompt, task description,
the 9 evasion label names with one-line definitions (Table 10), one labelled train example per label, reasoning on,
evasion only (clarity is mapped up locally). Shots come only from train rows outside fold 0, so the same prompt
serves the fold-0 sanity pass and the test pass.

Steps: copy code + CSVs, start the GGUF download and an sm75 llama.cpp build in the background (a prebuilt
binary is tried first when the driver is new enough), start llama-server over both GPUs, run the 296 test rows,
then up to SANITY_N fold-0 train rows with the time left. Every result is appended to a jsonl as it arrives.

Outputs in /kaggle/working: llm_test_raw.jsonl, llm_test_labels.csv, sanity_raw.jsonl, sanity.csv, server.log.
Launch: .venv/Scripts/python -m kaggle kernels push -p kaggle_build/llm_test --accelerator NvidiaTeslaT4
"""
import glob, json, os, re, shutil, subprocess, sys, tarfile, time, unicodedata, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

T0 = time.time()
SANITY_N = 120                 # fold-0 train rows to score AFTER the test pass (0 = skip)
SANITY_MAX_MIN = 25            # sanity stops starting new rows after this many minutes of its own
SANITY_END_MIN = 200           # ... or at this many minutes since start, whichever comes first
THINK_UNTIL_MIN = 170          # test rows started after this many minutes since start run with thinking off
TEST_STOP_MIN = 225            # test rows not started by then stay empty (v6 fills them), so the kernel ends in time
NP = 12                        # parallel slots on the server = client threads
CTX = 98304                    # total KV cells, shared by the slots (--kv-unified)
THINK_MAX_TOKENS = 2048        # reasoning + answer
REASONING_BUDGET = 1536
MAX_ITEM_WORDS = 1200          # only a few train answers are longer; test max is 633 words
SHOT_SEED = 7
MODELS = [  # (repo, quant tag, sampling with thinking on); the second is a fallback if the first will not run
    ("ggml-org/Qwen3.6-35B-A3B-GGUF", "Q4_K_M", dict(temperature=1.0, top_p=0.95, top_k=20, presence_penalty=1.5)),
    ("unsloth/Qwen3-30B-A3B-Thinking-2507-GGUF", "Q4_K_M", dict(temperature=0.6, top_p=0.95, top_k=20)),
]
OFF_SAMPLING = dict(temperature=0.7, top_p=0.8, top_k=20)
LLAMA_TAG = "b12359"
URL = "http://127.0.0.1:8080"
W = "/kaggle/working"

DEFS = {
    "Explicit": "informația cerută este dată direct, în forma cerută, fără ambiguitate și fără să fie nevoie de deducții.",
    "Implicit": "informația cerută este transmisă, dar nu în forma așteptată; ascultătorul trebuie să deducă singur concluzia.",
    "General": "răspunsul rămâne la subiect, dar este prea larg, abstract sau vag pentru a oferi detaliile concrete cerute.",
    "Partial/half-answer": "acoperă concret doar o componentă a informației cerute și o omite pe restul.",
    "Dodging": "nu se angajează cu substanța întrebării; subiectul întrebării nu este nici recunoscut, nici atins.",
    "Deflection": "recunoaște subiectul întrebării, dar mută argumentul și atenția spre altceva decât ce s-a întrebat.",
    "Declining to answer": "un refuz verbal, asumat și explicit de a oferi informația cerută.",
    "Claims ignorance": "vorbitorul admite explicit că nu deține în acest moment informația cerută.",
    "Clarification": "nu oferă informația cerută și cere în schimb lămuriri despre la ce se referă întrebarea.",
}
TASK = ("Analizezi interviuri și conferințe de presă ale președinților României. Pentru perechea întrebare-răspuns "
        "de la final, stabilește cum răspunde vorbitorul la întrebarea jurnalistului și alege exact o etichetă "
        "de evaziune din lista de mai jos.")
NOTES = ("Distincții importante:\n"
         "- Dodging vs Deflection: la Deflection vorbitorul recunoaște subiectul întrebării și apoi deviază spre altceva; "
         "la Dodging subiectul nu este recunoscut deloc.\n"
         "- General vs Deflection: un răspuns General rămâne la subiect, doar că e vag; Deflection mută discuția în altă parte.\n"
         "- Explicit vs Implicit vs General: Explicit dă direct ce s-a cerut; la Implicit răspunsul se poate deduce, "
         "dar nu este spus în forma cerută; la General nu se poate extrage răspunsul concret cerut.\n"
         "- Declining to answer, Claims ignorance și Clarification sunt non-răspunsuri clare: vorbitorul refuză, "
         "spune că nu știe sau cere lămuriri.")
ANSWER_FMT = "Răspunde doar cu un singur rând de forma:\nEtichetă: <numele etichetei, exact ca în listă>"


# ---------------- prompt ----------------
def cut(text, n=MAX_ITEM_WORDS):
    w = str(text).split()
    return str(text) if len(w) <= n else " ".join(w[: n // 2]) + " [...] " + " ".join(w[-n // 2:])


def qa(q, a):
    return f"Întrebare: {cut(q)}\nRăspuns: {cut(a)}"


def pick_shots(tr, exclude, labels, seed=SHOT_SEED):
    """One fixed train example per evasion label, short-to-medium length, from rows outside `exclude`."""
    pool = tr.drop(index=exclude)
    aw, qw = pool.interview_answer.str.split().str.len(), pool.interview_question.str.split().str.len()
    cutoff = (pool.interview_question + pool.interview_answer).str.contains(r"\.\.\.|…")
    ok = pool[aw.between(12, 90) & qw.between(4, 50) & ~cutoff]
    rng = np.random.RandomState(seed)
    shots = []
    for lab in labels:
        cand = ok[ok.evasion_label == lab]
        cand = cand if len(cand) else pool[pool.evasion_label == lab]
        shots.append(cand.loc[rng.choice(cand.index)])
    return [shots[i] for i in rng.permutation(len(shots))]


def build_prompt(shots, labels, q, a):
    defs = "\n".join(f"{i + 1}. {lab}: {DEFS[lab]}" for i, lab in enumerate(labels))
    ex = "\n\n".join(f"Exemplul {i + 1}\n{qa(s.interview_question, s.interview_answer)}\nEtichetă: {s.evasion_label}"
                     for i, s in enumerate(shots))
    return (f"{TASK}\n\nEtichete:\n{defs}\n\n{NOTES}\n\nExemple (câte unul pentru fiecare etichetă):\n\n{ex}\n\n"
            f"Acum etichetează perechea următoare.\n\n{qa(q, a)}\n\n{ANSWER_FMT}")


# ---------------- parsing ----------------
def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9/:=\n]+", " ", s)


ALIASES = [  # (normalized alias, label); matched as whole words
    ("partial/half answer", "Partial/half-answer"), ("partial half answer", "Partial/half-answer"),
    ("half answer", "Partial/half-answer"), ("raspuns partial", "Partial/half-answer"), ("partial", "Partial/half-answer"),
    ("declining to answer", "Declining to answer"), ("declining", "Declining to answer"), ("refuz", "Declining to answer"),
    ("claims ignorance", "Claims ignorance"), ("claim ignorance", "Claims ignorance"), ("ignorance", "Claims ignorance"),
    ("clarification", "Clarification"), ("clarificare", "Clarification"),
    ("explicit", "Explicit"), ("explicita", "Explicit"), ("implicit", "Implicit"), ("implicita", "Implicit"),
    ("general", "General"), ("generala", "General"), ("dodging", "Dodging"), ("eschivare", "Dodging"),
    ("deflection", "Deflection"), ("deviere", "Deflection"),
]
ALIAS_RE = re.compile(r"\b(" + "|".join(re.escape(a) for a, _ in ALIASES) + r")\b")
ALIAS_MAP = dict(ALIASES)
MARKER_RE = re.compile(r"(?:eticheta|label|categoria|categorie|raspuns final|final answer|answer)\s*(?:finala|final)?\s*[:=]\s*([^\n]*)")


def parse_label(content, labels):
    """Final evasion label from the answer text (reasoning already split off), or None."""
    text = str(content or "")
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    elif "<think>" in text:
        return None  # reasoning was cut off before the answer
    t = norm(text).strip()
    if not t:
        return None
    marks = MARKER_RE.findall(t)
    if marks:  # "Etichetă: X": take the first label named after the last marker
        seg = marks[-1].strip()
        m = re.match(r"([1-9])\b", seg)
        hits = ALIAS_RE.findall(seg)
        if hits or m:
            return ALIAS_MAP[hits[0]] if hits else labels[int(m[1]) - 1]
    if re.fullmatch(r"[1-9]", t):
        return labels[int(t) - 1]
    hits = ALIAS_RE.findall(t)  # no marker: the last label named is usually the verdict
    return ALIAS_MAP[hits[-1]] if hits else None


# ---------------- server ----------------
def sh(cmd, **kw):
    print("+", cmd, flush=True)
    return subprocess.run(cmd, shell=True, **kw)


def chat(prompt, think, sampling, max_tokens, timeout=1500):
    body = dict(messages=[{"role": "user", "content": prompt}], max_tokens=max_tokens,
                chat_template_kwargs={"enable_thinking": think}, **(sampling if think else OFF_SAMPLING))
    req = urllib.request.Request(f"{URL}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    m = d["choices"][0]["message"]
    return m.get("content") or "", m.get("reasoning_content") or "", d.get("usage", {})


def start_server(binary, gguf):
    """Start llama-server; drop optional flags one by one if it rejects them. Returns the process or None."""
    base = [binary, "-m", gguf, "-ngl", "999", "--split-mode", "layer", "--tensor-split", "1,1", "-c", str(CTX),
            "-np", str(NP), "-b", "2048", "-ub", "512", "--jinja", "--reasoning-format", "deepseek",
            "--host", "127.0.0.1", "--port", "8080"]
    optional = [["--reasoning-budget", str(REASONING_BUDGET)], ["--kv-unified"], ["-fa", "auto"]]
    env = dict(os.environ, LD_LIBRARY_PATH=os.path.dirname(binary) + ":" + os.environ.get("LD_LIBRARY_PATH", ""),
               CUDA_CACHE_MAXSIZE="4294967296")
    for k in range(len(optional) + 1):
        cmd = base + sum(optional[k:], [])
        print("server:", " ".join(cmd), flush=True)
        log = open(f"{W}/server.log", "a")
        p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
        t = time.time()
        while time.time() - t < 25 * 60 and p.poll() is None:
            try:
                if urllib.request.urlopen(f"{URL}/health", timeout=5).status == 200:
                    break
            except Exception:
                pass
            time.sleep(5)
        if p.poll() is None:
            try:
                t1 = time.time()
                out = chat("Răspunde doar cu: OK", False, None, 8, timeout=900)
                print(f"server up after {(time.time() - t) / 60:.1f} min, test reply {out[0]!r} in {time.time() - t1:.0f} s", flush=True)
                used = subprocess.run("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits", shell=True,
                                      capture_output=True, text=True).stdout.split()
                print("GPU MiB used:", used, flush=True)
                if sum(int(x) for x in used if x.isdigit()) > 12000:  # weights really sit on the GPUs
                    return p
                print("model is not on the GPUs (CUDA backend missing?): rejecting this binary", flush=True)
            except Exception as e:
                print("test request failed:", e, flush=True)
            p.kill(); p.wait()
            return None
        tail = open(f"{W}/server.log", errors="ignore").read()[-3000:]
        print("server exited:\n" + tail, flush=True)
        if not re.search(r"error: invalid argument: -|error while handling argument|unknown argument|unrecognized|usage:", tail):
            return None  # a load/runtime failure, not a rejected flag
    return None


def start_download(repo, quant, dest):
    code = (f"from huggingface_hub import list_repo_files, hf_hub_download as d\n"
            f"fs = sorted(f for f in list_repo_files('{repo}') if f.endswith('.gguf') and '{quant}' in f and 'mmproj' not in f.lower())\n"
            f"fs = [f for f in fs if 'UD-' not in f] or fs\n"
            f"print('files', fs, flush=True)\n"
            f"[print(d('{repo}', f, local_dir='{dest}'), flush=True) for f in fs]\n")
    env = dict(os.environ, HF_XET_CHUNK_CACHE_SIZE_BYTES="0")  # no second copy of the GGUF in ~/.cache
    return subprocess.Popen([sys.executable, "-c", code], stdout=open(f"{W}/download.log", "a"), stderr=subprocess.STDOUT, env=env)


def gguf_path(dest):
    fs = sorted(glob.glob(f"{dest}/**/*.gguf", recursive=True))
    return fs[0] if fs else None


def try_prebuilt():
    smi = subprocess.run("nvidia-smi", shell=True, capture_output=True, text=True).stdout
    m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", smi)
    if not m or (int(m[1]), int(m[2])) < (12, 8):
        print("driver CUDA < 12.8 or unknown: skipping prebuilt binary", flush=True)
        return None
    u = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_TAG}"
    try:
        for name in (f"llama-{LLAMA_TAG}-bin-ubuntu-cuda-12.8-x64.tar.gz", f"cudart-llama-{LLAMA_TAG}-bin-ubuntu-cuda-12.8-x64.tar.gz"):
            urllib.request.urlretrieve(f"{u}/{name}", f"/tmp/{name}")
            with tarfile.open(f"/tmp/{name}") as tf:
                tf.extractall("/tmp/llb")
        server = sorted(glob.glob("/tmp/llb/**/llama-server", recursive=True))[0]
        for so in glob.glob("/tmp/llb/**/*.so*", recursive=True):  # all libs next to the binary ($ORIGIN rpath)
            if os.path.dirname(so) != os.path.dirname(server):
                shutil.copy(so, os.path.dirname(server))
        os.chmod(server, 0o755)
        return server
    except Exception as e:
        print("prebuilt failed:", e, flush=True)
        return None


def start_build():
    os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ["PATH"]
    cm = "cmake" if shutil.which("cmake") else f"{sys.executable} -m pip install -q cmake && cmake"
    cmd = (f"(git clone -q --depth 1 --branch {LLAMA_TAG} https://github.com/ggml-org/llama.cpp /tmp/llama.cpp || "
           f"git clone -q --depth 1 https://github.com/ggml-org/llama.cpp /tmp/llama.cpp) && {cm} -S /tmp/llama.cpp "
           f"-B /tmp/llama.cpp/build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 "
           f"-DGGML_NATIVE=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_OPENSSL=OFF -DBUILD_SHARED_LIBS=OFF "
           f"&& cmake --build /tmp/llama.cpp/build --target llama-server -j {os.cpu_count()}")
    return subprocess.Popen(cmd, shell=True, stdout=open(f"{W}/build.log", "w"), stderr=subprocess.STDOUT)


# ---------------- labelling ----------------
def run_pass(rows, prompts, labels, sampling, stop_at, think_until, raw_path, on_done):
    """Label rows in parallel; append each result to raw_path; call on_done(results) every 10 rows."""
    def one(i):
        if time.time() > stop_at:
            return None
        think = time.time() < think_until
        rec = dict(datapointID=int(rows[i].datapointID), mode="think" if think else "off")
        try:
            c, r, u = chat(prompts[i], think, sampling, THINK_MAX_TOKENS if think else 48)
        except Exception as e:
            c, r, u = "", f"ERROR {e}", {}
        lab = parse_label(c, labels)
        if lab is None and think:  # reasoning ran out or the answer did not parse: ask once more without thinking
            rec["think_content"], rec["think_tokens"] = c, u.get("completion_tokens")
            rec["mode"] = "think+off"
            try:
                c2, _, u2 = chat(prompts[i], False, sampling, 48)
                c, lab = c2, parse_label(c2, labels)
            except Exception as e:
                c = c + f" | ERROR {e}"
        rec.update(label=lab, content=c, reasoning=r, prompt_tokens=u.get("prompt_tokens"),
                   completion_tokens=u.get("completion_tokens"))
        return rec

    res, t = {}, time.time()
    with open(raw_path, "a", encoding="utf-8") as f, ThreadPoolExecutor(NP) as ex:
        for fut in as_completed([ex.submit(one, i) for i in range(len(rows))]):
            rec = fut.result()
            if rec is None:
                continue
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
            res[rec["datapointID"]] = rec
            if len(res) % 10 == 0 or len(res) == len(rows):
                ok = [r for r in res.values() if r["label"]]
                ct = [r["completion_tokens"] or 0 for r in res.values()]
                el = (time.time() - t) / 60
                print(f"  {len(res)}/{len(rows)} rows, {el:.1f} min, {len(res) / max(el, 1e-6):.1f} rows/min, "
                      f"parsed {len(ok)}, mean completion tokens {np.mean(ct):.0f}, "
                      f"modes {dict(zip(*np.unique([r['mode'] for r in res.values()], return_counts=True)))}", flush=True)
                on_done(res)
    on_done(res)
    return res


def sanity_rows(tr, n, seed=0):
    from common import folds
    val = folds(tr.evasion_label.values)[0][1]
    pick = np.sort(np.random.RandomState(seed).choice(val, min(n, len(val)), replace=False)) if n else val[:0]
    return val, pick


def main():
    src = glob.glob("/kaggle/input/**/finetune.py", recursive=True)[0].rsplit("/", 1)[0]
    os.chdir(W)
    os.makedirs("data", exist_ok=True)
    for f in glob.glob(f"{src}/*.py"):
        shutil.copy(f, ".")
    for f in glob.glob(f"{src}/*.csv"):
        shutil.copy(f, "data/")
    sys.path.insert(0, W)
    import pandas as pd
    from sklearn.metrics import f1_score, classification_report
    from common import load, EVASION_LABELS, EVASION_TO_CLARITY
    sh("nvidia-smi; nproc; free -g; df -h /tmp /kaggle/working; which nvcc cmake; nvcc --version | tail -2")
    sh(f"{sys.executable} -m pip install -q -U 'huggingface_hub[hf_xet]'")

    tr, te = load()
    val, san = sanity_rows(tr, SANITY_N)
    shots = pick_shots(tr, val, EVASION_LABELS)
    print("shots:", [(int(s.datapointID), s.evasion_label) for s in shots], flush=True)
    print("example prompt:\n" + build_prompt(shots, EVASION_LABELS, te.interview_question[0], te.interview_answer[0]), flush=True)

    dl = start_download(MODELS[0][0], MODELS[0][1], "/tmp/models/m0")
    build = start_build()
    pre = try_prebuilt()
    server, sampling = None, None
    for mi, (repo, quant, samp) in enumerate(MODELS):
        dest = f"/tmp/models/m{mi}"
        if mi:
            shutil.rmtree("/tmp/models", ignore_errors=True)
            dl = start_download(repo, quant, dest)
        dl.wait()
        gguf = gguf_path(dest)
        print(f"{repo}: download done at {(time.time() - T0) / 60:.1f} min -> {gguf}", flush=True)
        print(open(f"{W}/download.log").read()[-1500:], flush=True)
        if not gguf:
            continue
        for binary in ([pre] if pre else []) + ["BUILD"]:
            if binary == "BUILD":
                rc = build.wait()
                print(f"build exit {rc} at {(time.time() - T0) / 60:.1f} min\n" + open(f"{W}/build.log", errors="ignore").read()[-2500:], flush=True)
                binary = "/tmp/llama.cpp/build/bin/llama-server"
                if not os.path.exists(binary):
                    continue
            server = start_server(binary, gguf)
            if server:
                pre = binary  # reuse the binary that worked if a fallback model is needed
                break
        if server:
            sampling = samp
            print(f"using {repo} at {(time.time() - T0) / 60:.1f} min", flush=True)
            break
    if build.poll() is None:
        build.kill()
    if not server:
        raise SystemExit("no model could be served; see server.log and build.log")

    def prompts_for(df):
        return [build_prompt(shots, EVASION_LABELS, r.interview_question, r.interview_answer) for r in df]

    rows = [te.loc[i] for i in range(len(te))]

    def save_test(res):
        pd.DataFrame({"datapointID": te.datapointID,
                      "evasion_label": [(res.get(int(d)) or {}).get("label") or "" for d in te.datapointID]}) \
            .to_csv("llm_test_labels.csv", index=False)
    save_test({})
    print(f"TEST: {len(rows)} rows at {(time.time() - T0) / 60:.1f} min", flush=True)
    res = run_pass(rows, prompts_for(rows), EVASION_LABELS, sampling, T0 + TEST_STOP_MIN * 60,
                   T0 + THINK_UNTIL_MIN * 60, "llm_test_raw.jsonl", save_test)
    labs = [v["label"] for v in res.values()]
    print(f"test: {len(res)} rows at {(time.time() - T0) / 60:.1f} min, parsed {sum(map(bool, labs))}, label counts "
          f"{pd.Series([l for l in labs if l]).value_counts().to_dict()}", flush=True)

    sanity_stop = min(time.time() + SANITY_MAX_MIN * 60, T0 + SANITY_END_MIN * 60)
    if len(san) and time.time() < sanity_stop:
        rows = [tr.loc[i] for i in san]
        gold = {int(r.datapointID): r.evasion_label for r in rows}

        def save_sanity(res):
            pd.DataFrame([dict(datapointID=k, gold=gold[k], pred=v["label"] or "", mode=v["mode"]) for k, v in res.items()]) \
                .to_csv("sanity.csv", index=False)
        print(f"SANITY: {len(rows)} fold-0 train rows, thinking on, until minute {(sanity_stop - T0) / 60:.0f}", flush=True)
        res = run_pass(rows, prompts_for(rows), EVASION_LABELS, sampling, sanity_stop, float("inf"),
                       "sanity_raw.jsonl", save_sanity)
        ok = {k: v["label"] for k, v in res.items() if v["label"]}
        print(f"sanity: {len(res)} rows done, parse failures {1 - len(ok) / max(len(res), 1):.1%}", flush=True)
        if ok:
            g, p = [gold[k] for k in ok], list(ok.values())
            print("sanity evasion macro-F1 (parsed rows, single-annotator train labels):",
                  round(f1_score(g, p, average="macro"), 4))
            print("sanity mapped clarity macro-F1:",
                  round(f1_score([EVASION_TO_CLARITY[x] for x in g], [EVASION_TO_CLARITY[x] for x in p], average="macro"), 4))
            print(classification_report(g, p, labels=EVASION_LABELS, zero_division=0), flush=True)
    server.kill()
    print(f"done in {(time.time() - T0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
''',
    'llm/make_llm_submission.py': r'''"""Turn the Kaggle LLM labels into a 592-row submission in data/sample_output.csv order.

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
'''
}

if __name__ == "__main__":
    if sys.argv[1:] == ["unpack"]:
        for path, code in SOURCES.items():
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            open(path, "w", encoding="utf-8").write(code)
            print("wrote", path)
    else:
        open("submission.csv", "w", encoding="utf-8", newline="").write(PREDICTIONS)
        print(PREDICTIONS, end="")
