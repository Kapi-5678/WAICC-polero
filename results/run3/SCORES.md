# Kaggle run 3 (5-fold CV, T4 x2, ~2 h 10 min, started 18:17 after a stuck queue)

RoBERT-large, lr 1e-5, run 2's long settings (10 epochs, 512 tokens, q_len 128, batch 8, seed 42), with task 4's input fixes. Each config compares directly with its run-2 twin `r2-robert-large-lr1e-5`. Out-of-fold macro-F1 over all 3,278 train rows, best epoch per config.

| Config | Best epoch | Clarity | Evasion | Sum | vs run-2 twin (sum) |
|---|---|---|---|---|---|
| run-2 twin: question head only | 7 of 10 | 0.657 | 0.375 | 1.032 | |
| `--q_mode headtail` (first 32 + last 96 question tokens) | 7 of 10 | 0.663 | 0.378 | 1.041 | +0.009 |
| `--q_mode headtail --president` | 6 of 10 | 0.656 | 0.385 | 1.041 | +0.009 |

- Both fixes are a little better than the twin, but +0.009 from one seed is inside the ±0.01 run-to-run noise. Only 3.4% of questions are longer than 128 tokens, so a small effect is expected.
- **In the ensemble they help:** recipe R3 (every large lr 1e-5 model + these two, 9 models, `--combine geo`) scores **0.685 / 0.405, sum 1.090**. That's the best so far: +0.026 over v1, +0.004 over submitted v4. R3 was defined at 15:24, before run 3 existed, so there's no selection effect.

All 10 jobs ok. `kaggle.log` is the notebook log. Rebuild with `python evaluate.py results/run3`.
