"""PolERo LLM pass on Kaggle T4 x2: few-shot Qwen3.6-35B-A3B (GGUF Q4_K_M) on llama-server, reasoning on.

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
