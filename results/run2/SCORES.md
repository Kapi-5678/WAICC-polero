# Kaggle run 2 (5-fold CV, T4 x2, ~3.5 h)

Run 1's configs with the paper's settings: 10 epochs, 512 tokens, question cut to 128 tokens, batch 8. Same folds as run 1. Out-of-fold macro-F1 over all 3,278 train rows, best epoch per config (chosen on OOF, so slightly optimistic).

| Config | Best epoch | Clarity | Evasion | Run 1 equivalent (clarity / evasion) |
|---|---|---|---|---|
| xlm-roberta-large, lr 1e-5 | 7 of 10 | 0.663 | 0.379 | 0.672 / 0.376 |
| RoBERT-large, lr 1e-5 | 7 of 10 | 0.657 | 0.375 | 0.631 / 0.361 |
| RoBERT-large, lr 2e-5 | 7 of 10 | 0.645 | 0.361 | |
| bert-base-romanian-cased-v1, lr 2e-5 | 10 of 10 | 0.634 | 0.365 | 0.635 / 0.367 (lr 3e-5) |
| Ensemble of the four | | 0.663 | 0.386 | v1 trio: 0.663 / 0.395 |

`python select_ensemble.py --runs results/run1 results/run2` (8 configs):
- All 8 averaged: 0.663 / 0.394, the same as v1.
- Greedy selection: 0.674 / 0.405 in-sample, but 0.645 / 0.374 nested. The selected bag changes from fold to fold, so the in-sample gain is overfitting.
- Bar for a new submission (nested evasion +0.02, clarity drop at most 0.01): not met. **v1 stays.**

Notes:
- Longer training helped RoBERT-large (+0.014 evasion, +0.026 clarity) and did nothing for XLM-R large or the base model. All configs peak around epoch 7, then flatten.
- lr 2e-5 is worse than 1e-5 for RoBERT-large.
- Single configs vary by about ±0.01 between runs, so the reliable lever is averaging more independent models. Runs 4 and 5 (seeds 43 and 44 of both large lr 1e-5 configs) do that.

Rebuild with `python evaluate.py results/run2`. File layout is as in `results/run1/SCORES.md`.
