"""LLM full pass on Kaggle T4 x2: few-shot Qwen3-14B (AWQ) as a 9-way probability model for the ensemble.

Same model and prompt as colab/polero_llm_test.ipynb, except the labels are numbered 1-9 and the model
answers with one digit, so each row gets a probability over the 9 labels (softmax of the digit logprobs).
Out-of-fold: fold k's train rows (and, in "full" mode, all test rows) are labelled with few-shot examples
drawn only from the other 4 folds. One process per GPU, each on half the rows.

  MODE = "check": fold 0 train rows only. Prints macro-F1s, writes check_fold0.csv for llm/compare_llm.py.
  MODE = "full":  all 5 folds + test. Writes runs/llm-qwen3-14b_p0_f{k}_s42.npz in finetune.py's format
                  (val_idx, test_ids, val_pe0, val_pc0, test_pe0, test_pc0), readable by evaluate.py.

Launch: .venv/Scripts/python -m kaggle kernels push -p kaggle_build/llm_full --accelerator NvidiaTeslaT4
"""
import glob, json, os, shutil, subprocess, sys, time
import numpy as np

MODE = "check"                 # "check" (fold 0 train rows only) or "full" (5 folds + test)
MODEL = "Qwen/Qwen3-14B-AWQ"
SHOTS_PER_CLASS = 2
MAX_ANSWER_WORDS = 350
TAG = "llm-qwen3-14b"

DESC = {
    "Explicit": "the requested information is given directly and explicitly.",
    "Implicit": "the information is given, but only indirectly; it has to be inferred.",
    "General": "the answer stays vague or too general and avoids the specifics asked for.",
    "Partial/half-answer": "the question asks several things (or one compound thing) and only part of it is answered.",
    "Dodging": "the answer ignores the question and talks about something else.",
    "Deflection": "the answer stays near the topic but shifts the focus elsewhere (to another issue, actor, or a criticism) instead of answering.",
    "Declining to answer": "the speaker openly refuses or says they will not answer.",
    "Claims ignorance": "the speaker says they do not know or lack the information.",
    "Clarification": "the speaker asks for clarification or does not understand the question instead of answering.",
}


def defs(labels):
    lines = [f"{i + 1}. {lab}: {DESC[lab]}" for i, lab in enumerate(labels)]
    return ("You label how a Romanian president answers a journalist's question. Pick exactly ONE label:\n"
            + "\n".join(lines) + "\nReply with the label's number only (one digit).")


def cut(text, n=MAX_ANSWER_WORDS):
    w = str(text).split()
    return text if len(w) <= n else " ".join(w[: n // 2]) + " [...] " + " ".join(w[-n // 2:])


def user_msg(q, a):
    return f"Question: {cut(q, 120)}\nAnswer: {cut(a)}\nLabel number:"


def digit_probs(top):
    """top: {decoded token text: logprob} from the first generated token -> probabilities over labels 1..9."""
    lp = {}
    for t, v in top.items():
        t = str(t).strip()
        if t in {str(d) for d in range(1, 10)}:
            lp[int(t)] = max(lp.get(int(t), -np.inf), v)
    floor = (min(lp.values()) if lp else 0.0) - 5.0
    x = np.array([lp.get(d, floor) for d in range(1, 10)], dtype=np.float64)
    x = np.exp(x - x.max())
    return x / x.sum()


def ev_to_cl_matrix(ev_labels, ev_to_cl, cl_labels):
    m = np.zeros((len(ev_labels), len(cl_labels)))
    for i, lab in enumerate(ev_labels):
        m[i, cl_labels.index(ev_to_cl[lab])] = 1.0
    return m


def shot_rows(tr, val, ev_labels, seed=1):
    """Few-shot examples from the rows outside `val`, drawn as in colab/polero_llm_test.ipynb."""
    pool = tr.drop(index=val)
    rng = np.random.RandomState(seed)
    shots = []
    for lab in ev_labels:
        cand = pool[pool.evasion_label == lab]
        shots += [cand.loc[i] for i in rng.choice(cand.index, min(SHOTS_PER_CLASS, len(cand)), replace=False)]
    rng.shuffle(shots)
    return shots


def messages(shots, ev_labels, q, a):
    m = [{"role": "system", "content": defs(ev_labels)}]
    for s in shots:
        m += [{"role": "user", "content": user_msg(s.interview_question, s.interview_answer)},
              {"role": "assistant", "content": str(ev_labels.index(s.evasion_label) + 1)}]
    return m + [{"role": "user", "content": user_msg(q, a)}]


def worker(gpu, n_gpu, mode):
    """Label this GPU's share (every n_gpu-th row) of each fold's rows; write parts/f{k}_g{gpu}.npz."""
    from common import load, folds, EVASION_LABELS
    from vllm import LLM, SamplingParams
    tr, te = load()
    assert (tr.index == np.arange(len(tr))).all()
    fs = folds(tr.evasion_label.values)
    llm = LLM(MODEL, dtype="float16", tensor_parallel_size=1, max_model_len=12288, enforce_eager=True,
              gpu_memory_utilization=0.92, enable_prefix_caching=True, max_logprobs=20)
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0, max_tokens=1, logprobs=20)
    for k in ([0] if mode == "check" else range(5)):
        t0 = time.time()
        val = fs[k][1]
        shots = shot_rows(tr, val, EVASION_LABELS)
        rows_tr = val[gpu::n_gpu]
        rows_te = np.arange(len(te))[gpu::n_gpu] if mode == "full" else np.array([], dtype=int)
        df = [tr.loc[i] for i in rows_tr] + [te.loc[i] for i in rows_te]
        prompts = [tok.apply_chat_template(messages(shots, EVASION_LABELS, r.interview_question, r.interview_answer),
                                           tokenize=False, add_generation_prompt=True, enable_thinking=False) for r in df]
        outs = llm.generate(prompts, sp)
        probs = []
        for o in outs:
            first = o.outputs[0].logprobs[0] if o.outputs[0].logprobs else {}
            probs.append(digit_probs({lp.decoded_token: lp.logprob for lp in first.values()}))
        probs = np.array(probs, dtype=np.float32).reshape(-1, 9)
        os.makedirs("parts", exist_ok=True)
        np.savez(f"parts/f{k}_g{gpu}.npz", rows_tr=rows_tr, rows_te=rows_te,
                 p_tr=probs[: len(rows_tr)], p_te=probs[len(rows_tr):])
        print(f"[gpu {gpu}] fold {k}: {len(rows_tr)} train + {len(rows_te)} test rows in {(time.time() - t0) / 60:.1f} min", flush=True)


def merge(mode, n_gpu):
    from sklearn.metrics import f1_score, classification_report
    from common import load, folds, EVASION_LABELS, CLARITY_LABELS, EVASION_TO_CLARITY
    tr, te = load()
    fs = folds(tr.evasion_label.values)
    ev2cl = ev_to_cl_matrix(EVASION_LABELS, EVASION_TO_CLARITY, CLARITY_LABELS)
    os.makedirs("runs", exist_ok=True)
    for k in ([0] if mode == "check" else range(5)):
        val = fs[k][1]
        pe, tpe = np.zeros((len(val), 9), np.float32), np.zeros((len(te), 9), np.float32)
        pos = {r: i for i, r in enumerate(val)}
        for g in range(n_gpu):
            d = np.load(f"parts/f{k}_g{g}.npz")
            pe[[pos[r] for r in d["rows_tr"]]] = d["p_tr"]
            tpe[d["rows_te"]] = d["p_te"]
        assert np.allclose(pe.sum(1), 1, atol=1e-3), f"fold {k}: missing train rows"
        if mode == "check":
            gold = tr.evasion_label.values[val]
            pred = [EVASION_LABELS[i] for i in pe.argmax(1)]
            print("evasion macro-F1:", round(f1_score(gold, pred, average="macro"), 4))
            print("clarity macro-F1 (mapped):", round(f1_score([EVASION_TO_CLARITY[g] for g in gold],
                                                                 [EVASION_TO_CLARITY[p] for p in pred], average="macro"), 4))
            print(classification_report(gold, pred, labels=EVASION_LABELS, zero_division=0), flush=True)
            import pandas as pd
            pd.DataFrame({"row_idx": val, "datapointID": tr.datapointID.values[val], "gold": gold, "pred": pred}) \
                .to_csv("check_fold0.csv", index=False)
            np.savez("check_fold0.npz", val_idx=val, val_pe0=pe, val_pc0=(pe @ ev2cl).astype(np.float32))
        else:
            assert np.allclose(tpe.sum(1), 1, atol=1e-3), f"fold {k}: missing test rows"
            np.savez(f"runs/{TAG}_p0_f{k}_s42.npz", val_idx=val, test_ids=te.datapointID.values,
                     val_pe0=pe, val_pc0=(pe @ ev2cl).astype(np.float32),
                     test_pe0=tpe, test_pc0=(tpe @ ev2cl).astype(np.float32))
            print("wrote", f"runs/{TAG}_p0_f{k}_s42.npz", flush=True)


def main():
    src = glob.glob("/kaggle/input/**/finetune.py", recursive=True)[0].rsplit("/", 1)[0]
    os.chdir("/kaggle/working")
    os.makedirs("data", exist_ok=True)
    for f in glob.glob(f"{src}/*.py"):
        shutil.copy(f, ".")
    for f in glob.glob(f"{src}/*.csv"):
        shutil.copy(f, "data/")
    me = os.path.abspath(globals().get("__file__") or sys.argv[0])
    if me != os.path.abspath("full_pass.py"):
        shutil.copy(me, "full_pass.py")  # workers re-run this same file
    t0 = time.time()
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm", "nvidia-cutlass-dsl"], check=True)
    print(f"pip install done in {(time.time() - t0) / 60:.1f} min", flush=True)
    import torch
    n_gpu = max(torch.cuda.device_count(), 1)
    print(f"MODE={MODE}, {n_gpu} GPU(s)", flush=True)
    procs = []
    for g in range(n_gpu):  # fresh processes see the packages pip just installed
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
        log = open(f"worker{g}.log", "w")
        procs.append((subprocess.Popen([sys.executable, "full_pass.py", "worker", str(g), str(n_gpu), MODE],
                                       stdout=log, stderr=subprocess.STDOUT, env=env), g))
    for p, g in procs:
        rc = p.wait()
        print(f"worker {g}: {'ok' if rc == 0 else f'FAILED ({rc})'} after {(time.time() - t0) / 60:.1f} min", flush=True)
        print(open(f"worker{g}.log").read()[-3000:], flush=True)
        if rc:
            raise SystemExit(f"worker {g} failed")
    merge(MODE, n_gpu)
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    if sys.argv[1:2] == ["worker"]:
        worker(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
    else:
        main()
