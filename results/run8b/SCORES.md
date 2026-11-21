# Run 8b: clarity loss weight 1.0, 7 epochs (Kaggle lucazanotti/polero-cv-r8b)

5-fold OOF, best epoch per config, geo combine.

| config | clarity | evasion | sum |
|---|---|---|---|
| r8b-robert-large-cl1-e7 | 0.6618 | 0.3836 | 1.0455 |
| r8b-xlmr-large-cl1-e7 | 0.6662 | 0.3750 | 1.0412 |
| R3 + both (11 models), vs v1 | +0.0204 | +0.0075 | +0.0279 |
| R3 (9 models), vs v1 | +0.0219 | +0.0101 | +0.0320 |

Adding run 8b lowers R3. No v7 from it. Raising the clarity loss weight does not help.
