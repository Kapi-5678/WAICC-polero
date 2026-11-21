# Reference scores from the PolERo paper

Source: Ștefan & Nisioi, *PolERo: Studying Political Evasion in Romanian*, EMNLP 2026 (2026.emnlp-main.2688, arXiv 2609.02391). Numbers are copied from Table 1 and Table 11.

## How to read them
- All scores are **macro-F1 on the PolERo test split (n=296)**. That is the same size as our `test_data.csv`, so the paper's test split is most likely our test set.
- Our numbers are 5-fold OOF on train, so they won't match exactly. Use these as a sense of what "good" looks like, not as a target to hit to the third decimal.
- **C** = direct 3-class clarity, **E** = direct 9-class evasion, **M** = clarity mapped up from the predicted evasion label.
- **Our pipeline's `clarity_macroF1` is the paper's M column**: `common.report` maps evasion to clarity. So compare our clarity to M, and our evasion to E.
- SH = single-head, MH = multi-head (shared encoder, a clarity head and an evasion head, overlapping chunks with max-pooling, 7-fold CV with probability-averaged fold ensemble).

## Best per method (Romanian)

| Method | Model | Clarity M | Clarity C | Evasion E |
|---|---|---|---|---|
| TF-IDF + LR | — | .472 | .453 | .312 |
| Romanian encoder, single-head | RoBERT-large | .701 | .688 | .504 |
| Romanian encoder, single-head | Romanian BERT (base) | .634 | .684 | .470 |
| Multilingual encoder, single-head | XLM-R large | .686 | .680 | .365 |
| Multi-head + sliding window | **RoBERT-large** | **.763** | **.729** | .545 |
| Multi-head + sliding window | XLM-R large | .706 | .694 | .508 |
| EN+RO joint training (MH) | XLM-R large | .717 | .723 | **.570** (best fine-tuned evasion) |
| EN+RO joint training (MH) | RoBERTa-large (English) | .681 | .712 | .463 |
| LLM zero-shot | GPT-5.4 | .744 | .687 | .649 |
| LLM zero-shot | Qwen3.6-35B-A3B | .738 | .684 | .631 |
| LLM few-shot | **GPT-5.4** | **.763** | **.763** | .666 |
| LLM few-shot | **Qwen3.6-35B-A3B** | .760 | .746 | **.682** (best overall evasion) |

**Bottom line:** clarity (mapped) tops out at about **.76** (MH RoBERT-large ties few-shot GPT-5.4). Evasion tops out at about **.68** with LLMs and **.57** with fine-tuned encoders (EN+RO joint XLM-R).

## Full Romanian results (Table 11)

| Model | C | E | M |
|---|---|---|---|
| TF-IDF + Logistic Regression | .453 | .312 | .472 |
| *Single-head fine-tuning* | | | |
| mBERT-base | .601 | .328 | .502 |
| BERT-large-cased | .475 | .282 | .501 |
| ELECTRA-large | .500 | .084 | .241 |
| ModernBERT-large | .440 | .246 | .443 |
| RoBERTa-large | .646 | .428 | .613 |
| XLM-RoBERTa-large | .680 | .365 | .686 |
| Romanian BERT | .684 | .470 | .634 |
| RoBERT-large | .688 | .504 | .701 |
| *Multi-head fine-tuning* | | | |
| RoBERTa-large | .696 | .501 | .652 |
| XLM-RoBERTa-large | .694 | .508 | .706 |
| RoBERT-large | .729 | .545 | .763 |
| *Cross-lingual, XLM-R large MH* | | | |
| EN → RO | .635 | .399 | .655 |
| EN2RO → RO | .665 | .412 | .706 |
| RO → RO2EN | .681 | .494 | .702 |
| RO + RO2EN → RO | .708 | .492 | .681 |
| RO + EN2RO → RO | .718 | .489 | .684 |
| EN + RO → RO (joint) | .723 | .570 | .717 |
| *Cross-lingual, RoBERTa-large MH* | | | |
| EN + RO → RO (joint) | .712 | .463 | .681 |
| *LLM prompting (no fine-tuning)* | | | |
| Llama-3.3-70B zero-shot / few-shot | .479 / .555 | .354 / .312 | .446 / .411 |
| Gemma-4-31B-it zero-shot / few-shot | .657 / .672 | .560 / .619 | .713 / .738 |
| GPT-OSS-120B zero-shot / few-shot | .575 / .553 | .544 / .580 | .740 / .729 |
| DeepSeek-V4-Pro zero-shot / few-shot | .604 / .627 | .556 / .594 | .696 / .751 |
| Qwen3.6-35B-A3B zero-shot / few-shot | .684 / .746 | .631 / .682 | .738 / .760 |
| GPT-5.4 zero-shot / few-shot | .687 / .763 | .649 / .666 | .744 / .763 |

EN2RO / RO2EN = training data machine-translated with NLLB-200-distilled-600M.

## Ablation: what the multi-head pieces add (Table 2, RoBERT-large, Romanian, mean ± std over 3 seeds)

| Step | Clarity (C) | Evasion (E) |
|---|---|---|
| S0 single-head | .678 ± .022 | .511 ± .015 |
| S1 + multi-head (joint clarity + evasion loss) | .701 ± .022 | .531 ± .030 |
| S2 + sliding window | .705 ± .017 | .515 ± .019 |
| S3 + cross-validation | .707 ± .015 | .516 ± .046 |
| S4 + fold ensembling | .721 ± .011 | .537 ± .021 |
| Gain over S0 | +.043 | +.026 |

Seed std is about ±.02 on evasion, which supports our 0.02 keep threshold.

## Takeaways for our plan
- **Backbone:** RoBERT-large beats XLM-R large and Romanian BERT on Romanian in every setup except EN+RO joint training. Our run-1 configs already cover all three.
- **Predict evasion and map up.** For RoBERT-large on Romanian, mapped clarity beats the direct clarity head (.763 vs .729). That's what our pipeline already does.
- **English QEvasion (task 2):** joint EN+RO training raised XLM-R evasion by .062 (.508 → .570) and clarity M by .011. It *hurt* RoBERTa-large on Romanian evasion. Expect the gain on XLM-R, not on RoBERT.
- **LLMs (task 3):** strongest on evasion (+.11 over the best encoder). Few-shot mostly helps the Dodging/Deflection and Implicit/General boundaries. Llama-3.3-70B collapses to Deflection. Reasoning models (GPT-5.4, Qwen3.6, Gemma-4) did best.
- **Hardest classes:** annotator agreement is lowest for Deflection (κ=.364) and General (κ=.464), and highest for Clarification (κ=1.0) and Claims ignorance (κ=.881). Overall test Fleiss κ is .843 for clarity and .678 for evasion. Treat that as a rough ceiling for evasion.

## Warning: test labels are public
The paper releases the dataset, code and prompts under CC BY 4.0 at `github.com/gabriel-stefan/polero`. That release very likely contains the gold labels for our 296 test rows. **Don't download or look at that repo's data files**, and don't use them for training or checking. That would break our "never read test answers" rule. The paper's own contamination check (CoDeC) found 37–45% contamination scores, below its 80% threshold. Still, LLMs may have seen this public text, so take LLM scores on test with some suspicion.
