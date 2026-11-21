# PolERo: Political Evasion & Clarity

Models for the PolERo shared task: classifying Romanian presidential interview answers by clarity (3 classes, Subtask 1) and evasion strategy (9 classes, Subtask 2). Who does what, and by when, is in [TEAM_TASKS.md](TEAM_TASKS.md).

## Setup
```powershell
git clone https://github.com/Kapi-5678/polero.git
cd polero
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```
Put `train_data.csv`, `test_data.csv` and `sample_output.csv` from the competition page in `data/`. They are git-ignored.

Check that it works (prints 5-fold CV scores in about a minute):
```powershell
.venv\Scripts\python baseline_tfidf.py
```

## Files
- `common.py`: data loading, label taxonomy, CV folds (fixed seed, shared by everyone)
- `baseline_tfidf.py`: TF-IDF + logistic regression baseline
- `finetune.py`: two-head transformer fine-tuning, one CV fold per run
- `evaluate.py`: scores the CV runs out-of-fold, ensembles them, writes the submission
- `kaggle/`: notebook that runs all folds on Kaggle GPUs
