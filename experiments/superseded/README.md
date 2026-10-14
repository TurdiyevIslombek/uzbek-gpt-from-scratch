# Superseded experiments

These are the original evaluation scripts. They produced the figures in the first version of the paper and model card:

| | Original figure |
|---|---|
| uzbek-gpt-103m | 1.105 bits/byte |
| mGPT-1.3B zero-shot | 1.163 |
| mGPT-1.3B + QLoRA | 1.147, later 1.121 |
| Tokenizer ablation (232M) | 1.158 |
| Bootstrap vs zero-shot | 0.058 [0.037, 0.078] |
| Adaptation sweep 0 / 1M / 10M | 1.163 / 1.121 / 1.120 |

**Do not use these numbers.** The held-out set these scripts evaluate on (`uz_heldout.txt`) is the opening of `train.bin`, so the from-scratch model had trained on it. The QLoRA adaptation pool was also drawn from the start of the same stream, so the fine-tuned baseline had seen much of it too. The bootstrap resampled the two models independently rather than in pairs.

The scripts are kept unchanged so the correction can be checked. Re-scoring `uz_heldout.txt` with the new evaluation code reproduces 1.1050, 1.1628 and 1.1579 exactly.

## The first tokenizer ablation

The first ablation model (`ablation_best.pt`, 232M parameters, mGPT's tokenizer) was also flawed, and its scores in `../audit-2026-09/` runs 03 and 05 (1.0492 normalized, 1.0477 web) should not be used:

- Its data cap was `CORPUS_WORDS = 955_000_000`, meant to match the 955M-**token** train split. The corpus has only ~574M words, so it read the whole FineWeb-2 `uzn_Latn` stream, including the validation region the test documents come from.
- It stopped at step 10,000 of a planned 18,693 (about one pass, learning rate not yet decayed).

It was retrained on exactly `train.bin`'s documents for two full passes by `../audit-2026-09/08k_retrain_ablation_kaggle.py`: 1.0310 web, 1.0330 normalized.

Current results and the audit are in [`../audit-2026-09/`](../audit-2026-09/).
