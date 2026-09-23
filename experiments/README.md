# Experiments

```
experiments/
├── README.md                      ← this file
├── audit-2026-09/                 ← current results: use these
│   ├── audit_A_contamination_eval_ablation.ipynb
│   ├── audit_B_adaptation_sweep.ipynb
│   ├── 01_contamination_check.py
│   ├── 02_clean_eval_and_score.py
│   ├── 03_score_ablation.py
│   ├── 04_adaptation_sweep.py
│   ├── results/                   ← JSON results, Figure 2, per-span losses
│   └── logs/                      ← console output of every run
└── superseded/                    ← the original experiments, kept for comparison
```

## What happened

The original results were measured on a held-out set that turned out to be the opening of `train.bin`. The held-out text had been built by streaming FineWeb-2 from its beginning, and `tokenize_data.py` splits train and validation by position, so the first documents of the stream are training data. The model had trained on its own test set for about two epochs.

A second held-out set drawn from `val.bin` was also rejected: about 39% of validation documents appear elsewhere in the training data, because the web corpus contains duplicate pages.

The audit in `audit-2026-09/` builds a deduplicated held-out set, proves it shares nothing with the training data, and re-measures every model. Re-scoring the old held-out set with the same code reproduces the old published figures exactly (1.1050 / 1.1628 / 1.1579), so the correction can be checked.

## Results

Bits-per-byte on the deduplicated held-out set: 474 validation documents with no 24-token passage in common with `train.bin`. Protocol B: every model scores the same ~800-byte spans with the same preceding context. Lower is better. Intervals are from a paired bootstrap, 10,000 resamples.

| Model | Params | Bits/byte | Gap to from-scratch [95% CI] |
|---|---|---|---|
| uzbek-gpt-103m (from scratch) | 103M | **1.0281** | — |
| Ablation: same architecture, mGPT's tokenizer | 232M | 1.0492 | +0.0211 [0.0195, 0.0227] |
| mGPT-1.3B + QLoRA, 1M tokens | 1.3B | 1.0764 | +0.0483 [0.0444, 0.0520] |
| mGPT-1.3B, zero-shot | 1.3B | 1.0774 | +0.0493 [0.0455, 0.0530] |

Adaptation budget sweep (`audit_B`):

| Adaptation tokens | 0 | 1M | 10M |
|---|---|---|---|
| mGPT-1.3B + QLoRA, bits/byte | 1.0774 | 1.0757 | 1.0591 |

## Run-to-run reproducibility

The evaluation script was run in three separate Kaggle sessions with identical code and seed. Figures in the paper and model card come from session 1.

| Measurement | Session 1 | Session 2 | Session 3 |
|---|---|---|---|
| from-scratch, Protocol B | 1.0281 | 1.0281 | 1.0281 |
| mGPT zero-shot, Protocol B | 1.0774 | 1.0774 | 1.0774 |
| mGPT + QLoRA 1M, Protocol B | 1.0764 | 1.0766 | 1.0766 |
| bootstrap vs QLoRA | +0.0483 [0.0444, 0.0520] | +0.0485 [0.0447, 0.0522] | +0.0485 [0.0446, 0.0522] |
| ablation, Protocol B | 1.0492 | 1.0492 | — |

Everything that involves no training reproduces exactly. The QLoRA baseline moves by 0.0002, because 4-bit GPU training is not bit-for-bit deterministic. The sweep's 1M point (1.0757, in two sessions) trains on a slightly different slice of the same documents, so it differs a little more. All of this is far below the effects reported.

The session 1 notebook outputs were overwritten by later re-runs; `logs/02_clean_eval_session1.log` is rebuilt from that session's saved console output and checked against `results/results_v3.json`.

## How to reproduce

Everything runs on a free Kaggle GPU (T4) with internet enabled.

1. Open `audit_A_contamination_eval_ablation.ipynb` on Kaggle and add two inputs:
   - [`uzbek-fineweb2-tokens-16k`](https://www.kaggle.com/datasets/islombekturdiyev/uzbek-fineweb2-tokens-16k): `train.bin` and `val.bin`, as produced by `tokenize_data.py`
   - [`ablation-best`](https://www.kaggle.com/datasets/islombekturdiyev/ablation-best): `ablation_best.pt`, the 232M tokenizer-ablation checkpoint
2. Run all cells. About 60–90 minutes.
3. Open `audit_B_adaptation_sweep.ipynb` with the first input and run all cells. About 5 hours, mostly the 10M-token training run. Each stage caches its result, so a timeout does not lose finished work.

The `.py` files are the same code as the notebook cells, for running outside Kaggle. They must run in order in one session, because each step reads the files the previous one wrote to `/kaggle/working`.

## Files in `results/`

| File | What it is |
|---|---|
| `results_v3.json` | Every figure from `02_clean_eval_and_score.py`, session 1. Source of the published numbers. |
| `ablation_results.json` | Ablation scores, bootstrap, and the eval-set text profile (Cyrillic share etc.). |
| `figure2_clean.json` | Sweep results at 0 / 1M / 10M tokens. |
| `figure2_clean.png` | Figure 2 of the paper. |
| `dedup_chunks_*.csv` | Per-span losses for each model (span id, nats, bytes). Recompute any bootstrap on a CPU with no GPU. The QLoRA file is from session 2. |

## Notes on method

**Bits-per-byte** is total negative log-likelihood divided by (ln 2 × total bytes), computed as an aggregate ratio rather than an average over chunks. It is used instead of perplexity because it does not depend on the tokenizer.

**Protocol A** scores non-overlapping 512-token chunks per model, the original design. Because the Uzbek tokenizer packs more text into each token, a 512-token chunk covers about 2,250 bytes for it and about 1,300 for mGPT, which gives it extra context, and the chunks cannot be paired across models. **Protocol B** fixes both problems and is the headline.

**The "memorisation effect" line** in `02_clean_eval_and_score.py`'s output is not a memorisation measurement. The old and clean held-out sets are different text; mGPT, which trained on neither, also scores much better on the clean set. `03_score_ablation.py` uses mGPT as a control and traces the difference to Cyrillic text (0.51% of the old set, 0.20% of the clean one).

**Limitation of the clean set.** It keeps only documents that appear nowhere else in the corpus, so it leans away from boilerplate and syndicated news.
