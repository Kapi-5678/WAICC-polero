"""Kaggle entry point: run every (config, fold) job, spread over the available GPUs.

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
}
RUN = "run4"  # Colleague 1 (TreiPaDus) changes this to "run5"
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
