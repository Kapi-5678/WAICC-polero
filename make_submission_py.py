"""Bundle a submission CSV and the pipeline source into one standalone .py file.

  python make_submission_py.py submissions/submission_v1.csv submissions/polero_v1.py
"""
import sys

SOURCES = ["common.py", "finetune.py", "evaluate.py", "kaggle/run.py"]
TEMPLATE = '''"""PolERo submission: clarity (subtask 1) and evasion (subtask 2) predictions.

  python {name}           -> writes submission.csv (592 rows) and prints it
  python {name} unpack    -> writes the training/evaluation source files used to produce it

Standard library only. The pipeline itself: 5-fold fine-tuning of XLM-R large, RoBERT-large and
Romanian BERT with an evasion head and a clarity head, then a probability-averaged ensemble.
"""
import os, sys

PREDICTIONS = r\'\'\'{csv}\'\'\'

SOURCES = {{
{sources}
}}

if __name__ == "__main__":
    if sys.argv[1:] == ["unpack"]:
        for path, code in SOURCES.items():
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            open(path, "w", encoding="utf-8").write(code)
            print("wrote", path)
    else:
        open("submission.csv", "w", encoding="utf-8", newline="").write(PREDICTIONS)
        print(PREDICTIONS, end="")
'''


def literal(text, what):
    assert "\'\'\'" not in text and not text.endswith("\\"), f"{what} can't go in a raw triple-quoted string"
    return "r\'\'\'" + text + "\'\'\'"


csv_path, out = sys.argv[1], sys.argv[2]
csv = open(csv_path, encoding="utf-8").read().replace("\r\n", "\n")
assert csv.splitlines()[0] == "subtaskID,datapointID,answer" and len(csv.strip().splitlines()) == 593
srcs = ",\n".join(f"    {p!r}: {literal(open(p, encoding='utf-8').read(), p)}" for p in SOURCES)
open(out, "w", encoding="utf-8", newline="\n").write(
    TEMPLATE.format(name=out.replace("\\", "/").split("/")[-1], csv=csv, sources=srcs)
)
print("wrote", out)
