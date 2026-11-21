"""PolERo submission: clarity (subtask 1) and evasion (subtask 2) predictions.

  python polero_v1.py           -> writes submission.csv (592 rows) and prints it
  python polero_v1.py unpack    -> writes the training/evaluation source files used to produce it

Standard library only. The pipeline itself: 5-fold fine-tuning of XLM-R large, RoBERT-large and
Romanian BERT with an evasion head and a clarity head, then a probability-averaged ensemble.
"""
import os, sys

PREDICTIONS = r'''subtaskID,datapointID,answer
1,18,Ambivalent
1,23,Clear Reply
1,63,Ambivalent
1,72,Clear Reply
1,89,Clear Reply
1,91,Clear Reply
1,213,Ambivalent
1,226,Clear Reply
1,267,Ambivalent
1,299,Clear Reply
1,332,Clear Reply
1,366,Clear Reply
1,370,Clear Reply
1,386,Clear Reply
1,411,Clear Reply
1,415,Ambivalent
1,449,Ambivalent
1,454,Clear Reply
1,465,Clear Reply
1,488,Clear Reply
1,545,Clear Reply
1,550,Clear Reply
1,608,Clear Reply
1,610,Clear Reply
1,687,Ambivalent
1,701,Ambivalent
1,702,Clear Reply
1,718,Ambivalent
1,733,Ambivalent
1,742,Clear Reply
1,748,Ambivalent
1,752,Clear Reply
1,770,Ambivalent
1,797,Clear Reply
1,822,Clear Reply
1,847,Clear Reply
1,884,Ambivalent
1,887,Ambivalent
1,924,Clear Reply
1,939,Clear Reply
1,1010,Clear Reply
1,1044,Clear Reply
1,1055,Clear Reply
1,1086,Clear Reply
1,1087,Ambivalent
1,1100,Clear Reply
1,1126,Clear Reply
1,1164,Clear Reply
1,1186,Clear Non-Reply
1,1196,Clear Reply
1,1202,Ambivalent
1,1212,Ambivalent
1,1253,Clear Reply
1,1268,Clear Reply
1,1311,Clear Reply
1,1336,Ambivalent
1,1404,Ambivalent
1,1425,Clear Reply
1,1442,Clear Reply
1,1491,Clear Reply
1,1503,Clear Reply
1,1576,Ambivalent
1,1578,Ambivalent
1,1629,Clear Reply
1,1631,Clear Reply
1,1721,Clear Reply
1,1724,Clear Reply
1,1733,Ambivalent
1,1737,Clear Reply
1,1776,Clear Reply
1,1786,Clear Reply
1,1797,Clear Reply
1,1822,Clear Reply
1,1922,Clear Reply
1,2004,Ambivalent
1,2086,Clear Reply
1,2095,Clear Reply
1,2121,Ambivalent
1,2153,Clear Reply
1,2160,Clear Reply
1,2391,Ambivalent
1,2496,Clear Reply
1,2563,Clear Reply
1,2621,Clear Reply
1,2623,Clear Reply
1,2718,Ambivalent
1,2753,Clear Reply
1,2755,Ambivalent
1,2773,Clear Reply
1,2782,Ambivalent
1,2808,Ambivalent
1,2859,Clear Reply
1,2926,Clear Reply
1,2968,Clear Reply
1,3019,Clear Reply
1,3022,Clear Reply
1,3044,Clear Reply
1,3055,Ambivalent
1,3089,Ambivalent
1,3105,Ambivalent
1,3168,Clear Reply
1,3196,Ambivalent
1,3239,Clear Reply
1,3256,Ambivalent
1,3262,Clear Reply
1,3309,Ambivalent
1,3333,Clear Reply
1,3349,Clear Reply
1,3444,Clear Reply
1,3455,Ambivalent
1,3475,Ambivalent
1,3481,Clear Reply
1,3522,Ambivalent
1,3533,Clear Reply
1,3548,Clear Reply
1,3560,Ambivalent
1,3565,Ambivalent
1,3567,Clear Reply
1,3570,Clear Reply
1,3601,Clear Reply
1,3609,Clear Reply
1,3643,Clear Non-Reply
1,3650,Ambivalent
1,3655,Ambivalent
1,3663,Clear Reply
1,3665,Clear Reply
1,3667,Clear Reply
1,3731,Ambivalent
1,3739,Clear Reply
1,3741,Ambivalent
1,3764,Clear Reply
1,3793,Clear Reply
1,3796,Ambivalent
1,3813,Ambivalent
1,3842,Ambivalent
1,3863,Clear Reply
1,3905,Ambivalent
1,3937,Clear Reply
1,3943,Clear Reply
1,3947,Ambivalent
1,3953,Clear Reply
1,3959,Clear Reply
1,3965,Clear Reply
1,4008,Clear Reply
1,4022,Clear Reply
1,4071,Clear Reply
1,4081,Ambivalent
1,4144,Clear Reply
1,4227,Ambivalent
1,4358,Clear Reply
1,4461,Clear Reply
1,4520,Clear Reply
1,4542,Clear Reply
1,4585,Ambivalent
1,4643,Clear Reply
1,4697,Clear Reply
1,4737,Clear Non-Reply
1,4770,Ambivalent
1,4773,Clear Reply
1,4827,Ambivalent
1,4891,Ambivalent
1,4942,Ambivalent
1,4951,Ambivalent
1,4956,Ambivalent
1,4966,Clear Reply
1,4967,Ambivalent
1,4997,Clear Non-Reply
1,5049,Clear Non-Reply
1,5081,Clear Reply
1,5093,Ambivalent
1,5101,Clear Reply
1,5178,Clear Reply
1,5182,Ambivalent
1,5206,Clear Reply
1,5227,Ambivalent
1,5234,Ambivalent
1,5244,Clear Non-Reply
1,5247,Clear Non-Reply
1,5275,Ambivalent
1,5305,Ambivalent
1,5359,Clear Non-Reply
1,5378,Clear Non-Reply
1,5379,Ambivalent
1,5408,Clear Reply
1,5441,Clear Reply
1,5501,Clear Reply
1,5504,Ambivalent
1,5507,Ambivalent
1,5519,Clear Reply
1,5530,Clear Reply
1,5538,Clear Non-Reply
1,5558,Clear Reply
1,5565,Ambivalent
1,5585,Ambivalent
1,5599,Ambivalent
1,5608,Ambivalent
1,5610,Clear Reply
1,5643,Clear Reply
1,5678,Clear Reply
1,5683,Clear Reply
1,5736,Clear Reply
1,5757,Clear Reply
1,5813,Ambivalent
1,5833,Clear Reply
1,5853,Ambivalent
1,5864,Ambivalent
1,5871,Ambivalent
1,5883,Ambivalent
1,5900,Ambivalent
1,5910,Ambivalent
1,5932,Ambivalent
1,5942,Clear Reply
1,5966,Ambivalent
1,5990,Clear Reply
1,5991,Clear Reply
1,6005,Clear Reply
1,6013,Ambivalent
1,6028,Clear Reply
1,6076,Ambivalent
1,6127,Ambivalent
1,6142,Ambivalent
1,6143,Clear Reply
1,6153,Clear Non-Reply
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
1,6375,Ambivalent
1,6419,Clear Reply
1,6432,Clear Non-Reply
1,6567,Clear Reply
1,6576,Ambivalent
1,6643,Ambivalent
1,6645,Ambivalent
1,6652,Ambivalent
1,6678,Ambivalent
1,6684,Ambivalent
1,6718,Ambivalent
1,6727,Ambivalent
1,6755,Ambivalent
1,6762,Ambivalent
1,6773,Ambivalent
1,6784,Ambivalent
1,6795,Clear Reply
1,6800,Clear Reply
1,6812,Ambivalent
1,6875,Clear Reply
1,6901,Clear Reply
1,6912,Clear Non-Reply
1,6933,Ambivalent
1,6952,Ambivalent
1,6980,Ambivalent
1,6992,Ambivalent
1,6999,Clear Non-Reply
1,7006,Ambivalent
1,7015,Ambivalent
1,7016,Ambivalent
1,7039,Clear Reply
1,7049,Ambivalent
1,7076,Clear Reply
1,7082,Ambivalent
1,7094,Ambivalent
1,7154,Clear Reply
1,7202,Ambivalent
1,7208,Clear Reply
1,7220,Clear Reply
1,7248,Ambivalent
1,7272,Clear Reply
1,7276,Ambivalent
1,7312,Ambivalent
1,7325,Ambivalent
1,7328,Ambivalent
1,7357,Clear Reply
1,7365,Clear Reply
1,7377,Clear Non-Reply
1,7390,Ambivalent
1,7407,Ambivalent
1,7422,Clear Reply
1,7500,Clear Reply
1,7506,Ambivalent
1,7535,Ambivalent
1,7572,Clear Non-Reply
1,7612,Ambivalent
1,7622,Clear Reply
1,7678,Ambivalent
1,3617,Ambivalent
1,4444,Clear Reply
1,5937,Ambivalent
1,7545,Clear Reply
2,18,General
2,23,Explicit
2,63,Explicit
2,72,Explicit
2,89,Explicit
2,91,Explicit
2,213,General
2,226,Explicit
2,267,General
2,299,Explicit
2,332,Explicit
2,366,Explicit
2,370,Explicit
2,386,Explicit
2,411,Explicit
2,415,Explicit
2,449,Explicit
2,454,Explicit
2,465,Explicit
2,488,Explicit
2,545,Explicit
2,550,Explicit
2,608,Explicit
2,610,Explicit
2,687,Explicit
2,701,General
2,702,Explicit
2,718,Implicit
2,733,Explicit
2,742,Explicit
2,748,Clarification
2,752,Explicit
2,770,General
2,797,Explicit
2,822,Explicit
2,847,Explicit
2,884,Dodging
2,887,Dodging
2,924,Explicit
2,939,Explicit
2,1010,Explicit
2,1044,Explicit
2,1055,Explicit
2,1086,Explicit
2,1087,Explicit
2,1100,Explicit
2,1126,Explicit
2,1164,Explicit
2,1186,Declining to answer
2,1196,Explicit
2,1202,General
2,1212,Dodging
2,1253,Explicit
2,1268,Explicit
2,1311,Explicit
2,1336,Dodging
2,1404,Deflection
2,1425,Explicit
2,1442,Explicit
2,1491,Explicit
2,1503,Explicit
2,1576,Implicit
2,1578,Implicit
2,1629,Explicit
2,1631,Explicit
2,1721,Explicit
2,1724,Explicit
2,1733,Dodging
2,1737,Explicit
2,1776,Explicit
2,1786,Explicit
2,1797,Explicit
2,1822,Explicit
2,1922,Explicit
2,2004,Dodging
2,2086,Explicit
2,2095,Explicit
2,2121,General
2,2153,Explicit
2,2160,Explicit
2,2391,General
2,2496,Explicit
2,2563,Explicit
2,2621,Explicit
2,2623,Explicit
2,2718,Claims ignorance
2,2753,Explicit
2,2755,Implicit
2,2773,Explicit
2,2782,Explicit
2,2808,Dodging
2,2859,Explicit
2,2926,Explicit
2,2968,Explicit
2,3019,Explicit
2,3022,Explicit
2,3044,Explicit
2,3055,General
2,3089,Dodging
2,3105,Explicit
2,3168,Explicit
2,3196,Dodging
2,3239,Explicit
2,3256,Dodging
2,3262,Explicit
2,3309,General
2,3333,Explicit
2,3349,Explicit
2,3444,Explicit
2,3455,Dodging
2,3475,Dodging
2,3481,Explicit
2,3522,General
2,3533,Explicit
2,3548,Explicit
2,3560,General
2,3565,Implicit
2,3567,Explicit
2,3570,Explicit
2,3601,Explicit
2,3609,Explicit
2,3643,Declining to answer
2,3650,Implicit
2,3655,Explicit
2,3663,Explicit
2,3665,Explicit
2,3667,Explicit
2,3731,General
2,3739,Explicit
2,3741,General
2,3764,Explicit
2,3793,Explicit
2,3796,General
2,3813,Clarification
2,3842,Explicit
2,3863,Explicit
2,3905,General
2,3937,Explicit
2,3943,Explicit
2,3947,Implicit
2,3953,Explicit
2,3959,Explicit
2,3965,Explicit
2,4008,Explicit
2,4022,Explicit
2,4071,Explicit
2,4081,Explicit
2,4144,Explicit
2,4227,Explicit
2,4358,Explicit
2,4461,Explicit
2,4520,Explicit
2,4542,Explicit
2,4585,Explicit
2,4643,Explicit
2,4697,Explicit
2,4737,Declining to answer
2,4770,Dodging
2,4773,Explicit
2,4827,Implicit
2,4891,Implicit
2,4942,General
2,4951,Explicit
2,4956,Implicit
2,4966,Explicit
2,4967,General
2,4997,Declining to answer
2,5049,Claims ignorance
2,5081,Explicit
2,5093,Deflection
2,5101,Explicit
2,5178,Explicit
2,5182,General
2,5206,Explicit
2,5227,Explicit
2,5234,Dodging
2,5244,Declining to answer
2,5247,Declining to answer
2,5275,Implicit
2,5305,Dodging
2,5359,Declining to answer
2,5378,Declining to answer
2,5379,Explicit
2,5408,Explicit
2,5441,Explicit
2,5501,Explicit
2,5504,Dodging
2,5507,General
2,5519,Explicit
2,5530,Explicit
2,5538,Declining to answer
2,5558,Explicit
2,5565,Implicit
2,5585,Implicit
2,5599,General
2,5608,Explicit
2,5610,Explicit
2,5643,Explicit
2,5678,Explicit
2,5683,Explicit
2,5736,Explicit
2,5757,Explicit
2,5813,Deflection
2,5833,Explicit
2,5853,Implicit
2,5864,Implicit
2,5871,Implicit
2,5883,Implicit
2,5900,Implicit
2,5910,General
2,5932,General
2,5942,Explicit
2,5966,General
2,5990,Explicit
2,5991,Explicit
2,6005,Explicit
2,6013,General
2,6028,Explicit
2,6076,General
2,6127,Explicit
2,6142,Deflection
2,6143,Explicit
2,6153,Claims ignorance
2,6155,Explicit
2,6167,Implicit
2,6178,Deflection
2,6182,Dodging
2,6195,Claims ignorance
2,6238,Implicit
2,6276,Implicit
2,6292,Explicit
2,6324,Explicit
2,6365,General
2,6374,Explicit
2,6375,Explicit
2,6419,Explicit
2,6432,Declining to answer
2,6567,Explicit
2,6576,Deflection
2,6643,Explicit
2,6645,Explicit
2,6652,Dodging
2,6678,Declining to answer
2,6684,Implicit
2,6718,General
2,6727,General
2,6755,Explicit
2,6762,Dodging
2,6773,Deflection
2,6784,Implicit
2,6795,Explicit
2,6800,Explicit
2,6812,General
2,6875,Explicit
2,6901,Explicit
2,6912,Declining to answer
2,6933,General
2,6952,Explicit
2,6980,Deflection
2,6992,General
2,6999,Declining to answer
2,7006,General
2,7015,General
2,7016,General
2,7039,Explicit
2,7049,Implicit
2,7076,Explicit
2,7082,General
2,7094,Explicit
2,7154,Explicit
2,7202,General
2,7208,Explicit
2,7220,Explicit
2,7248,Explicit
2,7272,Explicit
2,7276,General
2,7312,Clarification
2,7325,General
2,7328,Deflection
2,7357,Explicit
2,7365,Explicit
2,7377,Claims ignorance
2,7390,Implicit
2,7407,Deflection
2,7422,Explicit
2,7500,Explicit
2,7506,General
2,7535,Explicit
2,7572,Claims ignorance
2,7612,Dodging
2,7622,Explicit
2,7678,Explicit
2,3617,Dodging
2,4444,Explicit
2,5937,Implicit
2,7545,Explicit
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
ap.add_argument("--prev_len", type=int, default=0, help="tokens of the previous question to prepend")
ap.add_argument("--seed", type=int, default=42)
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


def encode(df):
    ids = []
    for p, q, a in zip(df.prev_question, df.interview_question, df.interview_answer):
        q = tok.encode(str(q), add_special_tokens=False)[: args.q_len]
        if args.prev_len:
            q = tok.encode(str(p), add_special_tokens=False)[-args.prev_len :] + [SEP] + q
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
            loss = loss_ev(le.float(), torch.tensor(yev[b], device=dev)) + 0.5 * loss_cl(lc.float(), torch.tensor(ycl[b], device=dev))
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

# Run 2: the paper's settings (longer training, 512 tokens, batch 8). Run 1's configs are in results/run1/SCORES.md.
LONG = ["--epochs", "10", "--max_len", "512", "--q_len", "128", "--bs", "8"]
CONFIGS = [
    ["--name", "r2-robert-large-lr1e-5", "--model", "readerbench/RoBERT-large", "--lr", "1e-5", *LONG],
    ["--name", "r2-xlmr-large-lr1e-5", "--model", "FacebookAI/xlm-roberta-large", "--lr", "1e-5", *LONG],
    ["--name", "r2-robert-large-lr2e-5", "--model", "readerbench/RoBERT-large", "--lr", "2e-5", *LONG],
    ["--name", "r2-robert-base-lr2e-5", "--model", "dumitrescustefan/bert-base-romanian-cased-v1", "--lr", "2e-5", *LONG],
]
jobs = [cfg + ["--fold", str(f)] for cfg in CONFIGS for f in range(5)]
n_gpu = max(torch.cuda.device_count(), 1)
print(f"{len(jobs)} jobs on {n_gpu} GPU(s)", flush=True)

running = {}  # gpu -> (proc, job, start)
t0 = time.time()
while jobs or running:
    for g in range(n_gpu):
        if g not in running and jobs:
            job = jobs.pop(0)
            name = "_".join(a.split("/")[-1] for a in job[1::2])
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
