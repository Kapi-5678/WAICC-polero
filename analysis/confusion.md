# Run-1 OOF confusion matrix

Ensemble xlm-roberta-large_p0, RoBERT-large_p0, bert-base-romanian-cased-v1_p0 (best epochs, as in `evaluate.py`), out-of-fold over all 3278 train rows.
Evasion macro-F1 0.3950, clarity macro-F1 (combined probs) 0.6626.
Rebuild: `.venv\Scripts\python analysis/export_errors.py`

Rows are gold, columns predicted. Expl = Explicit, Impl = Implicit, Gen = General, Part = Partial/half-answer, Dodg = Dodging, Defl = Deflection, Decl = Declining to answer, Ign = Claims ignorance, Clar = Clarification.

| gold \ pred | Expl | Impl | Gen | Part | Dodg | Defl | Decl | Ign | Clar | n |
|---|---|---|---|---|---|---|---|---|---|---|
| Explicit | **1408** | 81 | 174 | 8 | 56 | 39 | 20 | 15 | 0 | 1801 |
| Implicit | 106 | **71** | 29 | 4 | 34 | 21 | 7 | 3 | 1 | 276 |
| General | 173 | 41 | **141** | 1 | 40 | 23 | 16 | 3 | 0 | 438 |
| Partial/half-answer | 26 | 5 | 10 | **0** | 3 | 6 | 4 | 2 | 0 | 56 |
| Dodging | 69 | 36 | 22 | 1 | **114** | 10 | 22 | 6 | 6 | 286 |
| Deflection | 61 | 23 | 44 | 2 | 20 | **21** | 7 | 1 | 0 | 179 |
| Declining to answer | 11 | 4 | 3 | 1 | 24 | 8 | **95** | 2 | 0 | 148 |
| Claims ignorance | 8 | 0 | 2 | 1 | 7 | 2 | 4 | **33** | 1 | 58 |
| Clarification | 4 | 2 | 1 | 0 | 11 | 0 | 0 | 1 | **17** | 36 |
| predicted | 1866 | 263 | 426 | 18 | 309 | 130 | 175 | 66 | 25 | 3278 |

## Per class

| class | n | predicted | precision | recall | F1 |
|---|---|---|---|---|---|
| Explicit | 1801 | 1866 | 0.755 | 0.782 | 0.768 |
| Implicit | 276 | 263 | 0.270 | 0.257 | 0.263 |
| General | 438 | 426 | 0.331 | 0.322 | 0.326 |
| Partial/half-answer | 56 | 18 | 0.000 | 0.000 | 0.000 |
| Dodging | 286 | 309 | 0.369 | 0.399 | 0.383 |
| Deflection | 179 | 130 | 0.162 | 0.117 | 0.136 |
| Declining to answer | 148 | 175 | 0.543 | 0.642 | 0.588 |
| Claims ignorance | 58 | 66 | 0.500 | 0.569 | 0.532 |
| Clarification | 36 | 25 | 0.680 | 0.472 | 0.557 |
| **macro** | | | 0.401 | 0.396 | 0.395 |
