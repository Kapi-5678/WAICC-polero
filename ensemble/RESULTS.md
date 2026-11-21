# Task 7: ensemble selection

## Run 1 only
```
.venv\Scripts\python select_ensemble.py --runs results/run1
```
Out-of-fold macro-F1 over all 3,278 train rows.

| ensemble | clarity | evasion | sum |
|---|---|---|---|
| v1 trio (submission v1) | 0.6626 | 0.3950 | 1.0576 |
| all 4 configs averaged | 0.6553 | 0.3958 | 1.0511 |
| greedy selection, in-sample | 0.6733 | 0.4004 | 1.0737 |
| **greedy selection, nested** | 0.6542 | 0.3931 | **1.0473** |
| run1/xlm-roberta-large_p0 | 0.6717 | 0.3764 | 1.0481 |
| run1/bert-base-romanian-cased-v1_p0 | 0.6346 | 0.3666 | 1.0012 |
| run1/RoBERT-large_p0 | 0.6308 | 0.3607 | 0.9916 |
| run1/bert-base-romanian-cased-v1_p32 | 0.6146 | 0.3377 | 0.9524 |

- **The in-sample bag** is XLM-R ×3, RoBERT-large ×1 and Romanian BERT ×1. It beats v1 by +0.016, but in-sample selection overstates by **+0.027**.
- **Nested, greedy selection is −0.010 against v1.** The 5 nested folds pick 2 different bags (XLM-R + RoBERT, or the trio with XLM-R weighted 2–3×).
- **Verdict for run 1:** no new submission; v1 stays.
- **Caveat:** v1 itself was picked by looking at run-1 out-of-fold scores (the p32 config was dropped), so it's not a nested number either. The comparison slightly favours v1.

## Run 1, evasion-first objective (Rares's rule, now the default)
`select_ensemble.py` now maximises OOF **evasion** macro-F1 while clarity stays within 0.01 of v1 (`--objective evasion --max_clarity_drop 0.01`). `--objective sum` gives the table above.

| ensemble | clarity | evasion | sum |
|---|---|---|---|
| v1 trio | 0.6626 | 0.3950 | 1.0576 |
| greedy, in-sample (XLM-R ×2 + RoBERT + RO-BERT) | 0.6688 | 0.4041 | 1.0729 |
| **greedy, nested** | 0.6490 | **0.3806** | 1.0296 |

- **Nested against v1:** evasion −0.014, clarity −0.013. The bar (nested evasion +0.02 with clarity drop ≤ 0.01) is **not met**, so v1 stays.
- **It overfits more than the sum objective:** in-sample overstates the nested score by +0.043, against +0.027. Evasion macro-F1 depends on a handful of rare-class rows, so the 5 nested folds pick 4 different bags.

## Runs 1 + 2 (8 configs)
```
.venv\Scripts\python select_ensemble.py --runs results/run1 results/run2
.venv\Scripts\python stack.py --runs results/run1 results/run2 --C 0.1 1
```

| ensemble | clarity | evasion | sum |
|---|---|---|---|
| v1 trio | 0.6626 | 0.3950 | 1.0576 |
| all 8 averaged | 0.6634 | 0.3942 | 1.0576 |
| greedy, in-sample (XLM-R run 1 ×3 + XLM-R run 2) | 0.6743 | 0.4045 | 1.0787 |
| **greedy, nested** | 0.6448 | **0.3744** | 1.0192 |
| stack, class_weight None, C=0.1 (nested) | 0.6681 | 0.3775 | 1.0456 |
| **R1: all large lr 1e-5 models, fixed (4)** | **0.6742** | **0.3954** | **1.0696** |
| R2: R1 + v1's RO-BERT base (5) | 0.6740 | 0.3963 | 1.0703 |
| R4: XLM-R large only (2) | 0.6690 | 0.3899 | 1.0589 |

- **Greedy selection overfits badly now.** In-sample overstates the nested score by **+0.060**, and the 5 nested folds pick 5 unrelated bags. Nested it's −0.021 evasion against v1.
- **Stacking still loses on evasion:** −0.017 against the plain average in its best setting.
- **The bar** (nested evasion +0.02 with clarity drop ≤ 0.01) is not met. **v1 stays.**

### Fixed recipes: the plan for runs 3–6
Learned selection can't beat the noise with 3,278 rows. The recipes in `select_ensemble.py` (`RECIPES`) are fixed equal-weight averages, defined now, before runs 3–6 land. Configs from later runs join automatically by name pattern, and with no selection step their OOF score needs no nesting.
- **R1 = every XLM-R large and RoBERT-large model at lr 1e-5, any run or seed.** Today that's 4 models: evasion level with v1 (+0.0004) and **clarity +0.012**. Runs 4, 5 and 6 add seeds 43, 44 and 45 of exactly these, up to 9 models. More seeds is the one lever Rares's run-2 notes call reliable.
- **On test, R1 and v1 agree** on 85.1% of evasion and 91.2% of clarity labels (`leaderboard_noise.py --compare`). That's the 80–90% range task 8 suggested for a hedged pair of finals.

**Proposed rule (Rares decides):**
- when runs 4–6 are in, score R1 again;
- if R1's evasion ≥ v1 + 0.02 with clarity drop ≤ 0.01, **v2 = R1**;
- if not, keep v1 as the main final and consider R1 as the second, hedging final. It's at least as good on OOF, it's a different set of models, and the visible leaderboard can't tell them apart anyway (task 9).

To write it:
```
.venv\Scripts\python select_ensemble.py --runs results/run1 results/run2 results/run4 results/run5 results/run6 --submission submissions/v2/submission_v2.csv --submit_recipe "R1 all large lr1e-5 models"
.venv\Scripts\python make_submission_py.py submissions/v2/submission_v2.csv
```

## Softening the probabilities before averaging (Rares, issue #2)
```
.venv\Scripts\python select_ensemble.py --runs results/run1 results/run2 --combine plain|geo|temp
```
- `geo`: mean of log-probabilities, renormalised (a geometric mean). No fitting.
- `temp`: one temperature per config and head, `p ∝ p^(1/T)`, T on a 0.5–5 grid by out-of-fold (OOF) log-loss. It's **nested**: each fold's OOF rows use the T fitted on the other 4 folds, and test uses the T fitted on all rows. The plain mean is applied after that.

**The temperatures confirm the diagnosis.** Run-1 models need T = 1.2–1.8; run-2 models need **T = 2.5–4.9** (evasion / clarity):

| model | run 1 T | | model | run 2 T |
|---|---|---|---|---|
| XLM-R large | 1.4 / 1.7 | | XLM-R large lr 1e-5 | 2.5 / 3.1 |
| RoBERT-large | 1.2 / 1.5 | | RoBERT-large lr 1e-5 | 3.0 / 3.7 |
| RO-BERT base | 1.4 / 1.8 | | RO-BERT base | 3.5 / 4.8 |

In the table below, each cell is clarity / evasion, OOF over 3,278 rows. Fixed sets need no nesting; greedy is nested. **Δ is against the submitted v1** (plain average): 0.6626 / 0.3950.

| set | plain | geo | temp |
|---|---|---|---|
| v1 trio | 0.6626 / 0.3950 | 0.6667 / 0.3968 | 0.6595 / 0.3968 |
| run-2 trio (T2) | 0.6689 / **0.4019** (Δev +0.007) | 0.6697 / 0.4000 (+0.005) | 0.6701 / 0.3975 (+0.003) |
| both trios, 6 (T12) | 0.6709 / 0.3938 (−0.001) | **0.6755 / 0.4033** (**+0.008**, clarity +0.013) | 0.6716 / 0.3953 (+0.000) |
| R1: large lr 1e-5, 4 | 0.6742 / 0.3954 (+0.000) | 0.6731 / 0.3921 (−0.003) | **0.6773 / 0.3991** (+0.004, clarity +0.015) |
| greedy, in-sample | 0.6743 / 0.4045 | 0.6812 / 0.4064 | 0.6837 / 0.4050 |
| **greedy, nested** | 0.6448 / 0.3744 (**−0.021**) | 0.6689 / 0.3928 (−0.002) | 0.6747 / 0.3922 (−0.003) |

**Findings:**
- **Softening fixes the problem Rares spotted.** Plain-averaging both trios is *worse* than run 2's trio alone (0.394 vs 0.402 evasion), because the over-confident run-2 members dominate. With the geometric mean, the 6-model average becomes the best fixed set: 0.6755 / 0.4033.
- **It also makes greedy selection much more stable.** Nested evasion goes from −0.021 to about −0.002 against v1. The in-sample overstatement shrinks from +0.060 (plain) to +0.026 (geo) and +0.022 (temp).
- **Nothing clears the bar.** The best evasion gain over v1 is **+0.008** (geo, both trios), far from +0.02, though clarity gains are consistent at +0.01 to +0.015. **v1 stays.**
- **For runs 4–6:** use `--combine geo` or `temp` for seed ensembles. Run 4/5/6 models come from run 2's settings, so they'll be just as over-confident, and a plain average would let them dominate the run-1 models.
- **Caveat:** the clarity method (combined / mapped / head) is still picked per set on OOF, a small in-sample choice.

## After the v3 probe: choose on the sum; clarity and evasion separately (Rares, issue #2)
The 50/50 formula is confirmed (`submissions/LEADERBOARD.md`). `select_ensemble.py` now defaults to `--objective sum`, with the bar at a nested **sum** gain of +0.02 over the submitted v1 (0.6626 + 0.3950 = 1.0576). New tools:
- **`--objective clarity` / `--objective evasion_only`:** greedy selection for one subtask.
- **`split_submission.py --clarity A.csv --evasion B.csv --out ...`:** clarity rows from one file, evasion rows from another, with the same checks as `evaluate.py`. It **reproduces `submissions/v3` byte for byte** from v2 (clarity) + v1 (evasion).
- **`select_ensemble.py --dump` + `pick_recipes.py`:** picks the clarity recipe (any fixed recipe × combine mode × clarity method, 54 candidates) and the evasion recipe (18 candidates) separately, **with a nested check**.

### Runs 1 + 2
Nested greedy against v1, as Δclarity / Δevasion / Δsum:

| combine | `--objective sum` | `clarity` bag + `evasion_only` bag (split) |
|---|---|---|
| plain | +0.010 / −0.001 / **+0.009** | +0.003 clarity, −0.020 evasion → −0.016 |
| geo | +0.005 / −0.001 / +0.004 | +0.000, −0.006 → −0.005 |
| temp | +0.002 / −0.007 / −0.005 | +0.002, −0.003 → −0.001 |

Fixed recipes picked per subtask (`pick_recipes.py`):

| | clarity | evasion | sum vs v1 |
|---|---|---|---|
| best clarity: R1 [temp] head; best evasion: T12 [geo], in-sample | 0.6773 | 0.4033 | +0.023 |
| **same choice, nested** (pick on 4 folds, score on the 5th) | 0.6654 | 0.3835 | **−0.009** |
| **v2 as submitted** (T12 [geo] for both, one pre-set choice) | 0.6755 | 0.4033 | **+0.021** |

**Findings:**
- **Picking the best recipe per subtask is noise.** The 5 nested folds pick 3 different clarity recipes and 4 different evasion recipes, and the +0.023 in-sample gain turns into −0.009.
- **Nested greedy gets no closer:** at most +0.009 on the sum.
- **The best-supported candidate is v2 as Rares submitted it:** both trios with the geometric mean, for both subtasks. It's a single choice made from the run-2 confidence diagnosis, with no search over recipes. It gives **+0.021 sum** over v1 (clarity +0.013, evasion +0.008), the only candidate at the bar without a search. Strictly it isn't nested: the clarity method, *mapped*, is still picked from 3.
- **For tonight's seeds (runs 4–6):** keep **one pre-set recipe**, the geometric mean of every model, and add seeds rather than choose between recipes.
  - In `RECIPES`, R1 (large lr 1e-5 models) is pattern-based and picks new seeds up automatically. Both trios (T12) is a fixed list of names.
  - **Proposal:** the v4 candidate is the geometric mean of the T12 models plus every new large lr 1e-5 seed.
  - Split clarity / evasion only if a *pre-set* split beats it by more than noise.
