# PolERo project rules

Shared task: classify Romanian presidential interview answers by evasion strategy (9 labels) and clarity (3 labels, derived from evasion). The task list, owners and deadlines are in `TEAM_TASKS.md`. Read it before starting. The final deadline is Sun 22 Nov 09:00, and experiments stop Sat 23:30.

## Rules
- Work only in your task's folder and on your own branch `<name>/<task>`. Never commit to `main`. Open a PR when done.
- Don't edit `common.py`, `finetune.py`, `evaluate.py` or `kaggle/` unless your task says so. They run the main Kaggle pipeline.
- Use `common.load()` for the data and `common.folds()` for CV splits (fixed seed). That keeps every result comparable.
- Label strings must match `common.EVASION_LABELS` / `common.CLARITY_LABELS` exactly. Clarity is derived through `common.EVASION_TO_CLARITY`.
- Never read, inspect or hand-label `data/test_data.csv` answers. Test data is only passed through a finished model.
- Report results as macro-F1 for both clarity and evasion, out-of-fold over the full train set (or on the fold-0 sample for the LLM task), together with the command that produced them.
- The data CSVs in `data/` are git-ignored. Never commit them or any API key.
- Keep scripts short and runnable from the repo root, using `.venv\Scripts\python`.
