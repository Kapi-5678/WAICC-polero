# Kaggle run 4 (5-fold CV, T4 x2, ~2 h 6 min)

Run 2's two large lr 1e-5 configs with a second seed (43): 10 epochs, 512 tokens, question cut to 128 tokens, batch 8. Same folds as runs 1 and 2. Out-of-fold macro-F1 over all 3,278 train rows, best epoch per config (chosen on OOF, so slightly optimistic). Kaggle notebook `snowy3/polero-run4`, `kaggle/run.py` from main with `RUN = "run4"`; all 10 jobs `ok` (`kaggle.log`).

| Config | Best epoch | Clarity (best method) | Evasion | Same config, seed 42 (run 2) |
|---|---|---|---|---|
| xlm-roberta-large, lr 1e-5, seed 43 | 10 of 10 | 0.668 (mapped) | **0.391** | 0.663 / 0.379 |
| RoBERT-large, lr 1e-5, seed 43 | 6 of 10 | 0.657 (mapped) | 0.376 | 0.657 / 0.375 |
| Ensemble of the two | | 0.675 (head) / 0.673 (combined) | 0.398 | v1 trio: 0.663 / 0.395 |

`python select_ensemble.py --runs results/run1 results/run2 results/run4` (10 configs):
- Greedy, nested against v1: evasion +0.002, clarity +0.006. Bar not met.
- **Fixed recipe R1, all six large lr 1e-5 models** (runs 1, 2 and 4), no selection so no nesting needed: **evasion +0.007, clarity +0.018** against v1. Under the evasion-first rule that's still below +0.02 evasion, but it's the best fixed recipe so far. Runs 5 and 6 (seeds 44 and 45) add two more large models to it.

Rebuild with `python evaluate.py results/run4`. File layout is as in `results/run1/SCORES.md`.
