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

Current results and the audit are in [`../audit-2026-09/`](../audit-2026-09/).
