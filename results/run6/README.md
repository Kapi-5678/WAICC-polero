# Run 6: XLM-R large, lr 1e-5, seed 45 (`r6-xlmr-large-lr1e-5-s45`)

Trained on Colab free T4s with `colab/polero_colab_run6.ipynb` (PR #13), same settings as runs 2/4/5 (10 epochs, 512 tokens, q_len 128, batch 8).

| Folds | Where | Status |
|---|---|---|
| 0, 1, 2 | Neagu's Colab, 17:30–18:53 | done, `ok` (28.3 / 27.6 / 27.6 min), files in this folder |
| 3, 4 | Rares's Colab | to be added |

`val_idx` of folds 0–2 checked against `common.folds()`. Last-epoch fold evasion macro-F1: 0.342 / 0.387 / 0.406. Once folds 3 and 4 are in, `python evaluate.py results/run6` scores the config.
