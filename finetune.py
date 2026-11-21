"""Fine-tune a shared encoder with two heads (evasion 9-way, clarity 3-way).

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

tag = f"{args.model.split('/')[-1]}_p{args.prev_len}_f{args.fold}_s{args.seed}"
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
