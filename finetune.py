"""Fine-tune a shared encoder with two heads (evasion 9-way, clarity 3-way).

Input: question [SEP] answer, answer truncated head+tail so the end of long answers survives.
Usage:
  python finetune.py --model <hf id> --mode holdout   # 85/15 split, prints val scores per epoch
  python finetune.py --model <hf id> --mode full      # train on all train, write test probs
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

from common import load, report, EVASION_LABELS, CLARITY_LABELS, EVASION_TO_CLARITY

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="dumitrescustefan/bert-base-romanian-cased-v1")
ap.add_argument("--mode", choices=["holdout", "full"], default="holdout")
ap.add_argument("--epochs", type=int, default=4)
ap.add_argument("--lr", type=float, default=3e-5)
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--max_len", type=int, default=320)
ap.add_argument("--q_len", type=int, default=80)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--out", default="runs")
args = ap.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.set_num_threads(os.cpu_count())

tr, te = load()
ev_id = {l: i for i, l in enumerate(EVASION_LABELS)}
cl_id = {l: i for i, l in enumerate(CLARITY_LABELS)}
# evasion -> clarity index, used to sum evasion probs into clarity probs
EV2CL = torch.tensor([cl_id[EVASION_TO_CLARITY[l]] for l in EVASION_LABELS])

tok = AutoTokenizer.from_pretrained(args.model)


def encode(df):
    ids = []
    a_len = args.max_len - 3
    for q, a in zip(df.interview_question, df.interview_answer):
        q = tok.encode(str(q), add_special_tokens=False)[: args.q_len]
        a = tok.encode(str(a), add_special_tokens=False)
        room = a_len - len(q)
        if len(a) > room:  # keep the start and the end of the answer
            head = room * 2 // 3
            a = a[:head] + a[-(room - head):]
        ids.append(tok.build_inputs_with_special_tokens(q, a))
    return ids


def batches(ids, idx, bs):
    for s in range(0, len(idx), bs):
        b = [ids[i] for i in idx[s : s + bs]]
        L = max(map(len, b))
        x = torch.full((len(b), L), tok.pad_token_id)
        m = torch.zeros((len(b), L), dtype=torch.long)
        for j, seq in enumerate(b):
            x[j, : len(seq)] = torch.tensor(seq)
            m[j, : len(seq)] = 1
        yield idx[s : s + bs], x, m


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
        pooled = (hs * m.unsqueeze(-1)).sum(1) / m.sum(1, keepdim=True)  # mean pooling
        pooled = self.drop(pooled)
        return self.ev(pooled), self.cl(pooled)


def class_weights(labels, n):
    c = np.bincount(labels, minlength=n).astype(float)
    w = (c.sum() / np.maximum(c, 1)) ** 0.5  # sqrt inverse frequency
    return torch.tensor(w / w.mean(), dtype=torch.float)


def predict(net, ids, idx):
    net.eval()
    pe, pc = np.zeros((len(ids), 9)), np.zeros((len(ids), 3))
    with torch.no_grad():
        for b, x, m in batches(ids, idx, 32):
            le, lc = net(x, m)
            pe[b], pc[b] = le.softmax(-1).numpy(), lc.softmax(-1).numpy()
    return pe, pc


def combine_clarity(pe, pc):
    """Average the clarity head with evasion probs summed into clarity groups."""
    agg = np.zeros_like(pc)
    for k in range(9):
        agg[:, EV2CL[k]] += pe[:, k]
    return (agg + pc) / 2


def scores(y_ev, pe, pc):
    from sklearn.metrics import f1_score
    p_ev = [EVASION_LABELS[i] for i in pe.argmax(1)]
    s = report("  mapped-from-evasion", list(y_ev), p_ev)
    y_cl = [EVASION_TO_CLARITY[v] for v in y_ev]
    s["clarity_head_macroF1"] = f1_score(y_cl, [CLARITY_LABELS[i] for i in pc.argmax(1)], average="macro")
    s["clarity_combined_macroF1"] = f1_score(y_cl, [CLARITY_LABELS[i] for i in combine_clarity(pe, pc).argmax(1)], average="macro")
    print("  clarity head macroF1=%.4f  combined macroF1=%.4f" % (s["clarity_head_macroF1"], s["clarity_combined_macroF1"]))
    return s


tr_ids = encode(tr)
yev = np.array([ev_id[l] for l in tr.evasion_label])
ycl = np.array([cl_id[l] for l in tr.clarity_label])
all_idx = np.arange(len(tr))
if args.mode == "holdout":
    trn, val = train_test_split(all_idx, test_size=0.15, stratify=yev, random_state=args.seed)
else:
    trn, val = all_idx, None

net = Net()
loss_ev = nn.CrossEntropyLoss(weight=class_weights(yev[trn], 9))
loss_cl = nn.CrossEntropyLoss(weight=class_weights(ycl[trn], 3))
opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)
steps = args.epochs * ((len(trn) + args.bs - 1) // args.bs)
sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)

tag = f"{args.model.split('/')[-1]}_{args.mode}_s{args.seed}"
os.makedirs(args.out, exist_ok=True)
log = []
for ep in range(args.epochs):
    net.train()
    rng = np.random.default_rng(args.seed + ep)
    # sort shuffled chunks by length to cut padding
    order = rng.permutation(trn)
    chunks = [sorted(order[i : i + args.bs * 20], key=lambda i: len(tr_ids[i])) for i in range(0, len(order), args.bs * 20)]
    order = np.concatenate(chunks)
    t0, tot = time.time(), 0.0
    bl = list(batches(tr_ids, order, args.bs))
    rng.shuffle(bl)
    for n, (b, x, m) in enumerate(bl):
        le, lc = net(x, m)
        loss = loss_ev(le, torch.tensor(yev[b])) + 0.5 * loss_cl(lc, torch.tensor(ycl[b]))
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad()
        tot += loss.item()
        if n % 20 == 0:
            print(f"ep{ep} step {n}/{len(bl)} loss {tot/(n+1):.3f} {time.time()-t0:.0f}s", flush=True)
    print(f"epoch {ep} done, loss {tot/len(bl):.3f}, {time.time()-t0:.0f}s", flush=True)
    if val is not None:
        pe, pc = predict(net, tr_ids, val)
        s = scores(tr.evasion_label.values[val], pe[val], pc[val])
        log.append({"epoch": ep, **s})
        np.savez(f"{args.out}/{tag}_val_ep{ep}.npz", idx=val, pe=pe[val], pc=pc[val])

if val is not None:
    json.dump({"args": vars(args), "log": log}, open(f"{args.out}/{tag}.json", "w"), indent=1)

te_ids = encode(te)
pe, pc = predict(net, te_ids, np.arange(len(te)))
np.savez(f"{args.out}/{tag}_test.npz", ids=te.datapointID.values, pe=pe, pc=pc)
print("wrote", f"{args.out}/{tag}_test.npz")
