# Task 5: class-bias tuning result

**Verdict: don't apply the offsets.** The gain is under 0.02 for evasion and negative for clarity.

Command (repo root, run-1 ensemble = the configs behind `submission_v1.csv`):
```
.venv\Scripts\python tune_bias.py
```

Method: one additive log-probability offset per class, found by coordinate ascent on macro-F1 (grid ±3, step 0.1). The scores below are nested: offsets are fitted on the OOF rows of 4 folds (`common.folds`) and scored on the 5th, rotating through all 5.

| | argmax | tuned (nested) | gain | per-fold gain |
|---|---|---|---|---|
| Evasion macro-F1 | 0.3950 | 0.4004 | +0.0055 | +0.004 ± 0.017 |
| Clarity, combined probs (what `submission_v1` uses) with its own 3 offsets | 0.6626 | 0.6516 | −0.0110 | −0.012 ± 0.015 |
| Clarity, mapped from tuned evasion | 0.6589 | 0.6654 | +0.0064 | — |

The baseline matches `results/run1/SCORES.md` (0.663 / 0.395). `evaluate.py results/run1 --use xlm-roberta-large_p0 RoBERT-large_p0 bert-base-romanian-cased-v1_p0` reproduces `submission_v1.csv` exactly.

## Robustness
None of the variations change the verdict:

| Variant | Evasion gain | Clarity (combined) gain |
|---|---|---|
| `--max_offset 1 --step 0.05` | +0.008 | −0.006 |
| `--max_offset 2 --step 0.25` | +0.011 | −0.005 |
| `--max_offset 5 --step 0.2` | +0.005 | −0.009 |
| XLM-R large alone | +0.003 | −0.003 |

## Why it doesn't help
Per-class evasion F1, argmax → tuned:

| Class | n | argmax | tuned |
|---|---|---|---|
| Explicit | 1801 | .768 | .761 |
| Implicit | 276 | .263 | .276 |
| General | 438 | .326 | .359 |
| Partial/half-answer | 56 | .000 | .014 |
| Dodging | 286 | .383 | .383 |
| Deflection | 179 | .136 | .126 |
| Declining to answer | 148 | .588 | .602 |
| Claims ignorance | 58 | .532 | .545 |
| Clarification | 36 | .557 | .537 |

Partial/half-answer gets the largest offset (+0.69), but its F1 only moves from .000 to .014. The models don't rank Partial answers above the others, so shifting the threshold just adds false positives. The small classes that already work (Clarification, Claims ignorance, Declining) are not under-predicted, so there's nothing for an offset to fix there. The weak spots (Partial, Deflection, Implicit) need better probabilities, not a different cut-off.

Better levers for run 2: a class-weighted loss (in the paper's loss ablation, class weighting lifts the minority classes), longer training and longer inputs (see `notes/reference_scores.md`).

`bias/offsets.json` holds the offsets refitted on all rows, kept for the record. Re-run the script on the run-2 ensemble; if the gain there reaches 0.02, `--submission <path>` writes a submission with the offsets applied.
