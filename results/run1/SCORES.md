# Kaggle run 1 (5-fold CV, T4 x2, ~65 min)

Out-of-fold macro-F1 over all 3,278 train rows, best epoch per config.

| Config | Best epoch | Clarity (combined) | Evasion |
|---|---|---|---|
| xlm-roberta-large, lr 1e-5 | 5 of 5 | 0.672 | 0.376 |
| bert-base-romanian-cased-v1, lr 3e-5 | 4 of 4 | 0.635 | 0.367 |
| RoBERT-large, lr 1e-5 | 5 of 5 | 0.631 | 0.361 |
| bert-base-romanian + previous question (32 tok) | 4 of 4 | 0.615 | 0.338 |
| **Ensemble: XLM-R L + RoBERT-L + RO-BERT base** (`submissions/v1/`) | | **0.663** | **0.395** |

For reference, TF-IDF + LR scores 0.49 clarity / 0.24 evasion.

**Leaderboard (v1, submitted Sat 21 Nov):** 41.93 / 100. If the score is 50 × clarity macro-F1 + 50 × evasion macro-F1, OOF predicted about 53, so test macro-F1s sum to about 0.84 against 1.06 OOF. Chance level with misaligned rows would be about 22, so the file is correct.

Notes:
- The previous-question context hurt, so it's dropped.
- Both large models were still improving at their last epoch, so run 2 could use more epochs.
- On test, the ensemble never predicts Partial/half-answer and predicts Clarification only 3 times. That's the target of task 5 (bias tuning).

Files: `<config>_f<fold>_s42.npz` hold `val_idx` (row positions in `common.load()` train), `val_pe<e>`/`val_pc<e>` (OOF evasion/clarity probabilities after epoch e), and `test_pe<e>`/`test_pc<e>`. Rebuild everything with `python evaluate.py results/run1`.
