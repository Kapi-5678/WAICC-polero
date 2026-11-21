# Task 4: error analysis of run 1

Ensemble analysed: `submission_v1`, i.e. XLM-R large + RoBERT-large + Romanian BERT base, best epochs. Out-of-fold over all 3,278 train rows: evasion macro-F1 0.3950, clarity 0.6626.

```
.venv\Scripts\python analysis/export_errors.py
```
This writes `analysis/errors.csv` (1,378 errors, git-ignored because it holds train text) and `analysis/confusion.csv`.

## Confusion matrix (gold rows, predicted columns)

| gold \ pred | Expl | Impl | Gen | Part | Dodg | Defl | Decl | Ign | Clar |
|---|---|---|---|---|---|---|---|---|---|
| Explicit (1801) | **1408** | 81 | 174 | 8 | 56 | 39 | 20 | 15 | 0 |
| Implicit (276) | 106 | **71** | 29 | 4 | 34 | 21 | 7 | 3 | 1 |
| General (438) | 173 | 41 | **141** | 1 | 40 | 23 | 16 | 3 | 0 |
| Partial (56) | 26 | 5 | 10 | **0** | 3 | 6 | 4 | 2 | 0 |
| Dodging (286) | 69 | 36 | 22 | 1 | **114** | 10 | 22 | 6 | 6 |
| Deflection (179) | 61 | 23 | 44 | 2 | 20 | **21** | 7 | 1 | 0 |
| Declining (148) | 11 | 4 | 3 | 1 | 24 | 8 | **95** | 2 | 0 |
| Claims ignorance (58) | 8 | 0 | 2 | 1 | 7 | 2 | 4 | **33** | 1 |
| Clarification (36) | 4 | 2 | 1 | 0 | 11 | 0 | 0 | 1 | **17** |

Biggest confusions: Explicit ↔ General (174 / 173), Implicit → Explicit (106), Explicit → Implicit (81). Almost every class leaks into Explicit. **Partial/half-answer is never predicted correctly** (0 of 56), and **Deflection is mostly predicted as something else** (21 of 179 right).

## What I read
I sampled 100 errors across the main confusion pairs (seed 0) and tagged each one. Tags are in `analysis/tagged_sample.csv`, with a short note per row.

| Tag | Count |
|---|---|
| Genuinely ambiguous | 59 |
| Model clearly wrong | 29 |
| Gold label looks wrong | 12 |

- **Mostly ambiguous:** Explicit / General / Implicit, Dodging ↔ Deflection, and Deflection → General. This matches the paper's annotator agreement, which is lowest for Deflection (κ=.36) and General (κ=.46).
- **The model's clear mistakes cluster** in Declining → Dodging (6 of 8), Clarification → Dodging (5 of 6), and the two Claims-ignorance pairs.
- **Gold labels that look wrong** are mostly short direct answers labelled Deflection or Implicit, such as "Nu cred că e o monedă de schimb" or "Eu nu fac înțelegeri cu nimeni". One row is a parsing error: another journalist's question appears inside the answer.

## Patterns worth fixing
Measured over the full train set, not just the sample.

### 1. Long questions lose their actual question (fix in `finetune.py`, the biggest lever)
`finetune.py` keeps the **first** `q_len` tokens of the question (`q[: args.q_len]`). Run 1 used 80 tokens and run 2 uses 128. Romanian journalists build a long preamble and ask at the end: in long questions the last "?" sits at the very end of the text, at a median of 100% of its length. A quarter of questions are over 250 characters, and 10% are over 416 characters (about 90+ tokens), so the question itself gets cut off.

| Question length | n | OOF accuracy |
|---|---|---|
| < 250 chars | 2454 | 0.61 |
| 250–400 | 475 | 0.49 |
| 400–600 | 225 | 0.48 |
| 600+ | 124 | 0.56 |

Long questions are harder anyway, so not all of the drop is truncation. Still, the fix is cheap and safe: **keep the end of the question**, for example `q[-q_len:]`, or the first 32 + last 96 tokens. It's worth trying in the next Kaggle run. It needs a change to `finetune.py`, which is Rares's call.

### 2. Short counter-questions: Clarification vs Dodging
58 answers are ≤ 120 characters and end in "?". The gold labels split 27 Clarification / 20 Dodging, and the model gets 62% right. The labels are actually consistent by meaning:
- **Genuine requests for information are Clarification:** "Care?", "Cui?", "Unde?", "Poftim?", "De Parlamentul nostru sau cel european?", "Dați-mi un exemplu."
- **Rhetorical pushback is Dodging:** "Păi de ce?", "De la ce?", "Ce motiv am ca să-mi dau demisia?", "De unde ați tras concluzia…?"

A simple post-rule (answer ≤ 60 chars, ends in "?", doesn't start with păi/dar/de unde and contains no "de ce" → Clarification) hits 38 rows: 24 Clarification, 10 Dodging. It lifts Clarification F1 from .557 to .641 and **evasion macro-F1 by +0.009** (0.3950 → 0.4037), and clarity by +0.003. That's below our 0.02 bar, and it was measured on train after reading errors, so it's optimistic. I don't recommend it alone. It could be bundled with other small rules if someone wants to try.

### 3. Formulaic refusals and "I don't know" are already mostly learned; their gold labels are mixed
- **Postponement / refusal phrases.** "veți afla", "o să aflați", "vom discuta", "am mai răspuns", "v-am răspuns": 40 short answers, with gold split 16 Declining / 10 Dodging / 5 Explicit / 5 General. The model gets 45% right.
- **"Nu știu…" openers.** 53 answers, gold 29 Claims ignorance / 10 Dodging / 8 Explicit. The model already predicts Claims ignorance 32 times. "Nu știu" is often a hedge inside a real answer, e.g. "Nu știu, să văd. M-aș duce cu plăcere…" is gold Explicit.
- **Passing the question to someone else.** "întrebați-l pe…", "e treaba Guvernului", "mingea e în curtea PSD", "INSP vă poate răspunde": 31 answers, labelled **five different ways** (Explicit 10, Dodging 6, Deflection 4, Implicit 4, Declining 3). That's annotation noise.

The "Nu știu → Claims ignorance" rule suggested in the task would hurt: about half of those answers aren't Claims ignorance. Don't add keyword rules here.

### 4. Era / president shifts the base rate
| President | n | Explicit share | OOF accuracy |
|---|---|---|---|
| Ion Iliescu | 756 | 0.62 | 0.63 |
| Traian Băsescu | 1120 | 0.62 | 0.64 |
| Klaus Iohannis | 997 | 0.46 | 0.52 |
| Nicușor Dan | 356 | 0.46 | 0.48 |
| Ilie Bolojan | 49 | 0.37 | 0.33 |

The model's errors lean towards Explicit, and that hurts most for the recent presidents, where evasive answers are more common. A cheap feature is to **prepend the president's name** (or the year) to the input, so the model can learn the different base rates. It needs a `finetune.py` change. Check whether the test set has the same president mix before relying on this.

### 5. Partial/half-answer is invisible to the model
All 56 Partial answers are predicted as something else: 26 Explicit, 10 General, 6 Deflection. Partial is 4.7× more common when the question has two or more "?" (6.1% vs 1.3%). Typical case: "will you go to France *before the end of the year*?" answered with "I'll go with pleasure", leaving out the "when". Bias offsets didn't fix this (task 5). Partial needs the whole multi-part question, so pattern 1's fix also helps here. A count of the question's parts as a feature could help a little. With 56 examples and the classes weighted equally, it's one ninth of evasion macro-F1, so even a small improvement matters.

## Data quality notes
- **About 43 answers contain transcript artefacts.** These are "/.../" gaps, "/rp/" tags, or an embedded speaker turn ("Președintele României, domnul Klaus Iohannis: …", or another journalist's question inside the answer). Accuracy on them is 0.58, about average, so it isn't worth cleaning now.
- **331 answers trail off with "..."** (interrupted). Accuracy is 0.59, about average.
- **Answer length isn't the problem.** Accuracy is flat at 0.55–0.65 across answer lengths, so `max_len` 320 → 512 (run 2) should help less than fixing how the question is cut (pattern 1).
