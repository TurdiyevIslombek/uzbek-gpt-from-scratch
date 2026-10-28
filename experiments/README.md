# Experiments

```
experiments/
├── README.md                     ← this file
├── audit-2026-09/                ← all current results
│   ├── 01_contamination_check.py
│   ├── 02_clean_eval_and_score.py
│   ├── 03_score_ablation.py
│   ├── 04_adaptation_sweep.py
│   ├── 05_eval_text_check.py     ← headline results (web spelling)
│   ├── 06_sweep_raw.py           ← final Figure 2
│   ├── 07_final_checks.py        ← Table 1, current models, samples
│   ├── 08k_retrain_ablation_kaggle.py  ← clean retrain of the tokenizer ablation (Table 4)
│   ├── ablation-1-prepare.ipynb  ┐
│   ├── ablation-2-train.ipynb    ├ the same script as three Kaggle notebooks
│   ├── ablation-3-score.ipynb    ┘
│   ├── 09_document_bootstrap.py  ← confidence intervals, resampling documents (CPU, seconds)
│   ├── build_heldout.py          ← rebuilds the 474-document test set anywhere
│   ├── left_quote_check.py       ← cost of the tokenizer's left-quote gap (CPU, seconds)
│   ├── audit_A_contamination_eval_ablation.ipynb
│   ├── audit_B_adaptation_sweep.ipynb
│   ├── logs/                     ← console output of every run
│   └── results/                  ← JSON results, per-span scores, figures, test set
├── alphabet-2026/                ← side study: the Uzbek alphabet amendment
└── superseded/                   ← the original experiments, kept for comparison
```

## The runs, in order

| Script | Question | Result |
|---|---|---|
| `01_contamination_check.py` | Is the original test set really unseen? | No: 11 of 12 passages are in `train.bin` |
| `02_clean_eval_and_score.py` | Build a clean test set; re-score every model | 474 documents, 0 of 16 passages in training data |
| `03_score_ablation.py` | What does the tokenizer alone contribute? | Ablation re-scored on the clean set |
| `04_adaptation_sweep.py` | Does more fine-tuning close the gap? | Sweep at 0 / 1M / 10M tokens |
| `05_eval_text_check.py` | Does the test text's spelling favour my model? | Yes; web spelling becomes the headline |
| `06_sweep_raw.py` | The sweep, re-scored in web spelling | Final Figure 2 |
| `07_final_checks.py` | Current models? Documented fertility sample? What does the model write? | Qwen much worse; Table 1 on the test set; samples |
| `08k_retrain_ablation_kaggle.py` | Does the tokenizer matter when the training documents are exactly the same? | Not for quality: ablation 1.0310 vs mine 1.0380 (web), 1.0330 vs 1.0281 (normalized) |
| `09_document_bootstrap.py` | Are the intervals too narrow because spans from one document are correlated? | Resampling documents widens them 1.3–2.8×; every difference still excludes zero |
| `left_quote_check.py` | How much does the tokenizer's one known gap (the left quote `‘`) cost? | 18.7% of the marks in the test documents; normalizing it too would save 2.3% of tokens (1.749 → 1.709 per word) |

Runs 02–04 score the test set in my tokenizer's normalized spelling; 05–08 use the original web spelling, which is what the paper reports. Both are kept so the effect of spelling can be seen.

**The ablation numbers in runs 03 and 05 come from the first, flawed ablation run** (`ablation_best.pt`, step 10,000): its data cap counted words instead of tokens, so it trained on the whole corpus including the test documents' region, and it stopped at step 10,000 of 18,693. Run 08 replaces it. The old scores are kept so the change can be checked.

## Headline results (web spelling)

Bits-per-byte, lower is better; 95% CIs from a paired bootstrap over the 474 documents, 10,000 resamples (`09`; scripts 02–08k print span-level intervals, which are narrower).

| Model | Size | Bits/byte | Gap to mine [95% CI] | Source |
|---|---|---|---|---|
| **uzbek-gpt-103m** | 0.10B | **1.0380** | — | 05 |
| Ablation (mGPT tokenizer), clean retrain | 0.23B | 1.0310 | −0.0070 [−0.0086, −0.0054] | 08 |
| mGPT-1.3B + QLoRA 1M | 1.3B | 1.0772 | +0.0392 [0.0306, 0.0486] | 05 |
| mGPT-1.3B zero-shot | 1.3B | 1.0801 | +0.0421 [0.0336, 0.0514] | 05 |
| mGPT-1.3B + QLoRA 10M | 1.3B | 1.0639 | +0.0258 [0.0178, 0.0340] | 06 |
| Qwen3-1.7B-Base | 1.72B | 1.5315 | +0.4935 [0.4699, 0.5164] | 07 |
| Qwen3-0.6B-Base | 0.60B | 1.9052 | +0.8672 [0.8402, 0.8936] | 07 |
| Qwen2.5-1.5B | 1.54B | 1.9475 | +0.9094 [0.8826, 0.9355] | 07 |
| Qwen2.5-0.5B | 0.49B | 2.3565 | +1.3185 [1.2893, 1.3468] | 07 |

Same documents in normalized spelling: 1.0281 (mine), 1.0330 (ablation, +0.0049 [+0.0024, +0.0076]), 1.0767 (mGPT + QLoRA 1M), 1.0774 (mGPT zero-shot). Bits per word — comparable across the two spellings — is 8.428 for my model in both; the web-trained models, including the ablation (8.371 web, 8.468 normalized), do about 1% better on web spelling. The first, flawed ablation scored 1.0477 (web) and 1.0492 (normalized).

## Reproducibility

- **Deterministic results reproduce exactly** across sessions (from-scratch, zero-shot mGPT, ablation, Qwen).
- **QLoRA training is not bit-for-bit deterministic.** Four sessions of the 1M configuration gave 1.0764–1.0767 on normalized text.
- **The correction is checkable.** Running the corrected code on the original test set gives 1.1078 / 1.1653 / 1.1604, within 0.003 of the originally reported 1.1050 / 1.1628 / 1.1579. The original code divided by the byte count of the whole file rather than the bytes it scored; `superseded/check_old_denominator.py` shows this from the original per-chunk scores.
- **`build_heldout.py`** rebuilds the test set from `train.bin` / `val.bin` and refuses to continue unless it matches exactly (474 documents, 200,818 words, 1,646,167 bytes).
- **`results/uz_heldout_dedup_raw.jsonl`** holds the recovered web text; normalizing it with `uzbek-bpe-16k` reproduces `uz_heldout_dedup.jsonl` character for character.
- **Seed 20260913** is used everywhere.

## How to run

Everything runs on a free Kaggle GPU (T4) with internet on.

| Notebook inputs | Run |
|---|---|
| dataset `uzbek-fineweb2-tokens-16k`, dataset `ablation-best` | 01, 02, 03 (notebook A), then 05 |
| dataset `uzbek-fineweb2-tokens-16k` | 04 (notebook B) |
| output of the 05 notebook (+ output of notebook B, to reuse adapters) | 06 |
| output of the 05 notebook | 07 |
| dataset `uzbek-fineweb2-tokens-16k`, output of the 05 notebook, Kaggle secret `HF_TOKEN` | 08k: `ablation-1-prepare` (CPU), then `ablation-2-train` (GPU T4 ×2, repeat until it says `TRAINING COMPLETE`, about 51 hours in total on the two GPUs at the logged ~11 s per step; see `logs/08k_ablation_train_tail.log`), then `ablation-3-score` with the train notebook's output attached |

Kaggle gives two T4s; scripts 01–07 pin one, because a 4-bit model can't be split across GPUs. 08k trains on both with `torchrun` and saves checkpoints to a private Hugging Face repo between sessions; it uses fp16 with loss scaling, since T4s have no bf16.

## Files in `results/`

| File | What it is | From |
|---|---|---|
| `results_v3.json`, `ablation_results.json` | Normalized-spelling results | 02, 03 |
| `figure2_clean.png`, `figure2_clean.json` | Sweep, normalized spelling | 04 |
| `rawcheck_results.json`, `rawcheck_*.csv` | All four models × both spellings, per-span scores | 05 |
| `uz_heldout_dedup.jsonl`, `uz_heldout_dedup_raw.jsonl` | The test set, normalized and web spelling | 05 |
| `sweep_raw_results.json`, `sweep_*.csv`, `figure2_final.png` | Final Figure 2 | 06 |
| `final_checks.json`, `table1_web.json`, `samples.json`, `modern_*_raw.csv` | Table 1, current models, samples | 07 |
| `retrain_results.json`, `rawcheck_ablation_clean_raw.csv`, `rawcheck_ablation_clean_clean.csv`, `ablation_clean_config.json` | Clean ablation, both spellings, per-span scores (`_clean_raw` = web spelling, `_clean_clean` = normalized spelling) | 08 |
| `document_bootstrap.json` | Document-level intervals for every reported gap | 09 |

**A note on the file names.** In the `rawcheck_*` and `sweep_*` files, a final `_clean` means the normalized spelling and `_raw` the web spelling. So `rawcheck_ablation_clean.csv` (1.0492) and `rawcheck_ablation_raw.csv` (1.0477) are the *first, flawed* ablation from run 05, while `rawcheck_ablation_clean_clean.csv` (1.0330) and `rawcheck_ablation_clean_raw.csv` (1.0310) are the *clean retrain* from run 08, normalized and web spelling.

The test-set text comes from FineWeb-2 and is shared under its ODC-By 1.0 licence.
