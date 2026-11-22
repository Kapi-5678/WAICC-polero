# LLM test pass (v7)

Qwen3.6-35B-A3B Q4_K_M (ggml-org GGUF) on llama.cpp, 2x T4 on Kaggle, few-shot with thinking on. Code: `llm/qwen36_test_pass.py`; submission builder: `llm/make_llm_submission.py`.

- `llm_test_labels.csv`: evasion label for all 296 test rows (0 parse failures). Became v7; clarity is mapped from evasion.
- `sanity_preds.csv`: predictions on 102 fold-0 train rows (join to train data on datapointID for gold). Clarity / evasion macro-F1: LLM 0.684 / 0.332, v6 on the same rows 0.658 / 0.362.
- `kernel_progress.log`: progress and score lines from the Kaggle log (interview text removed).

Board: v7 61.99, v8 (v6 evasion + LLM clarity) 59.80, v6 59.66.
