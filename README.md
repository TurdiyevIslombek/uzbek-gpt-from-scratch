# Uzbek GPT — a language model from scratch

A ~103M-parameter, decoder-only GPT-style language model pretrained from scratch on Uzbek (Latin script), with the transformer architecture implemented by hand (RMSNorm + RoPE + SwiGLU + multi-head causal attention), a custom Uzbek tokenizer, and a single-GPU pretraining run.

Scored by bits-per-byte on held-out Uzbek text, this 103M model beats mGPT-1.3B, a multilingual model about 13× its size, both zero-shot and fine-tuned. A controlled ablation splits the advantage into two measurable parts: the tokenizer, and pretraining on in-domain data.

**Model weights:** [IslombekT/uzbek-gpt-103m](https://huggingface.co/IslombekT/uzbek-gpt-103m) on Hugging Face
**Tokenizer:** [IslombekT/uzbek-bpe-16k](https://huggingface.co/IslombekT/uzbek-bpe-16k) (16,384-vocab byte-level BPE)

This repository is the code: the architecture, the data pipeline, the training loop, a generation script, and the evaluation experiments. It is meant to be readable end to end.

## Try it (no training, no GPU)

The trained weights are published on Hugging Face. To generate Uzbek text you only need this repo's code plus the weights file:

```bash
git clone https://github.com/TurdiyevIslombek/uzbek-gpt-from-scratch.git
cd uzbek-gpt-from-scratch
pip install -r requirements.txt

# download the trained weights (~425 MB) from the Hugging Face model
wget https://huggingface.co/IslombekT/uzbek-gpt-103m/resolve/main/model.safetensors

# generate
python generate.py --ckpt model.safetensors --prompt "Oʻzbekiston "
```

`generate.py` loads `model.safetensors` directly, with no checkpoint of your own and no training required. The tokenizer downloads automatically from the Hub. A one-click runnable version is also available as a Kaggle notebook.

## Why

Uzbek is a low-resource language with little dedicated open tooling. The goal was to build a usable Uzbek base model the hard way, understanding every component rather than fine-tuning an existing model, and to release it openly as a foundation for further Uzbek NLP work.

## What's here

| File | What it does |
|---|---|
| `model.py` | The from-scratch transformer: RoPE, RMSNorm, SwiGLU, fused causal attention, the GPT module. |
| `tokenize_data.py` | Tokenizes the FineWeb-2 `uzn_Latn` split into `train.bin` / `val.bin` (uint16, `<\|endoftext\|>`-separated, 90/10 split by position). |
| `train.py` | The pretraining loop: bf16 mixed precision, gradient accumulation, cosine LR schedule, gradient clipping, `torch.compile`, checkpointing. |
| `generate.py` | Loads a checkpoint (`.pt` or `.safetensors`) and samples Uzbek text. |
| `experiments/` | The evaluation: tokenizer benchmark, bits-per-byte comparison, the controlled experiments, and the contamination audit. |

## Architecture

| Component | Choice |
|---|---|
| Normalization | RMSNorm (pre-norm) |
| Positional encoding | Rotary Position Embeddings (RoPE) |
| Feed-forward | SwiGLU |
| Attention | Multi-head causal self-attention (scaled dot-product) |
| Layers / dim / heads | 12 / 768 / 12 |
| Vocab / context | 16,384 / 1,024 |
| Parameters | 103.06M |

## Training

- **Data:** Uzbek Latin split of FineWeb-2: ~1.23M documents → ~1.06B tokens (955M train / 106M val).
- **Hardware:** 1× RTX 4090 (24 GB).
- **Setup:** AdamW (0.9, 0.95; wd 0.1), warmup 400 → cosine decay (3e-4 → 3e-5), effective batch 192 sequences (~197K tokens/step), 9,700 steps (~2 epochs).
- **Result:** ~3.4 h wall-clock, ~$3.60, best validation loss 3.059. Full log in `training_log.txt`.

Note on the validation loss: `val.bin` was later found to be about 39% duplicated from `train.bin`, because the web corpus contains duplicate pages. The 3.059 figure is therefore somewhat optimistic. The bits-per-byte results below use a deduplicated held-out set instead.

## Results

Full details and scripts are in `experiments/`. Lower is better throughout.

### Tokenizer efficiency

`uzbek-bpe-16k` has the lowest fertility (tokens per word) of ten benchmarked tokenizers:

| Tokenizer | Vocab | Fertility (tok/word) ↓ |
|---|---|---|
| uzbek-bpe-16k (this repo) | 16,384 | **1.839** |
| XLM-RoBERTa | 250,002 | 2.334 |
| GPT-4o `o200k_base` | 200,019 | 2.724 |
| mBERT | 119,547 | 2.906 |
| GPT-2 | 50,257 | 3.584 |

It beats XLM-R, whose vocabulary is 15× larger: efficiency comes from language-specific design, not vocabulary size. (Full 10-tokenizer table in `experiments/`.)

### Model comparison

All models are scored by **bits-per-byte**, which measures how well a model predicts raw text independent of its tokenizer. The held-out set is 474 validation documents that share no 24-token passage with the training data, and every model scores the same ~800-byte spans with the same preceding context.

| Model | Params | Bits/byte ↓ | Gap to from-scratch [95% CI] |
|---|---|---|---|
| **uzbek-gpt-103m** (from scratch) | 103M | **1.028** | — |
| Ablation: same architecture, mGPT's tokenizer | 232M | 1.049 | +0.021 [0.020, 0.023] |
| mGPT-1.3B + QLoRA (1M adaptation tokens) | 1.3B | 1.076 | +0.048 [0.044, 0.052] |
| mGPT-1.3B (base, zero-shot) | 1.3B | 1.077 | +0.049 [0.046, 0.053] |

Intervals are from a paired bootstrap over the evaluation spans (10,000 resamples).

### Two controlled experiments

**Tokenizer ablation.** The same architecture, trained on the same text for the same steps, with mGPT's 100k tokenizer instead of the 16k Uzbek one. The bigger vocabulary makes it a 232M-parameter model, more than twice the size, and it still scores 0.021 bits/byte worse. Because data and training are held constant, that 0.021 is the tokenizer's contribution. The remaining gap to the adapted baseline comes from pretraining on in-domain data: 0.027 against the 1M-token baseline.

**Adaptation budget.** mGPT+QLoRA was re-trained with more Uzbek data:

| Adaptation tokens | 0 | 1M | 10M |
|---|---|---|---|
| Bits/byte | 1.077 | 1.076 | 1.059 |

Adaptation keeps improving, about 0.017 per tenfold increase, but at 10M tokens it is still 0.031 behind. Against that better baseline the pretraining share shrinks to 0.010 while the tokenizer's 0.021 stays fixed. A rough extrapolation from two points puts the budget needed to close the gap near 700M adaptation tokens, the same order as this model's own corpus.

**Takeaway:** the tokenizer is the cheapest high-leverage investment for a low-resource language. It is built in hours on ordinary hardware, and its 0.021 bits/byte contribution does not shrink when the baseline gets more data. Pretraining on in-domain text accounts for the rest of the advantage, but that part costs a billion-token corpus and GPU time.

## Evaluation correction (September 2026)

An earlier version of this README reported 1.105 / 1.163 / 1.147 bits/byte, a bootstrap interval of 0.058 [0.037, 0.078], and an adaptation curve that plateaued near 1.120. It concluded that the advantage came entirely from the tokenizer.

An audit found that the held-out set used for those numbers was the opening of `train.bin`: it was built by streaming the corpus from the start, and `tokenize_data.py` splits train and validation by position, so the first documents of the stream are training data. The model had trained on its own test set. A second set drawn from `val.bin` was also rejected, because about 39% of validation documents appear elsewhere in the training data.

The results above use a deduplicated set: every 24-token window of `train.bin` was fingerprinted, and only validation documents with zero overlap were kept (474 of 780). Exact search confirms none of the sampled passages appears in the training data. Re-scoring the old set with the new code reproduces the old figures exactly (1.1050 / 1.1628 / 1.1579), so the correction can be checked.

What changed:
- The advantage over the fine-tuned baseline grew, from 0.016 to 0.048, because the baseline's fine-tuning data had also overlapped the old test set.
- The claim that the tokenizer explains the whole advantage is withdrawn. On clean data it explains 0.021 of it.
- The claim that adaptation plateaus is withdrawn. It keeps improving.

The audit and re-measurement scripts are in `experiments/`. If you evaluate a model on a web-scraped corpus, check the held-out set against the training data directly. It takes minutes, and neither problem here was visible without it.

## Limitations

A small base model on ~1B tokens: it produces fluent, grammatical Uzbek but is not an instruction/chat model, is not factually reliable, and can repeat without a repetition penalty. It covers Latin-script Uzbek only, not Cyrillic. Evaluation is intrinsic only (bits-per-byte), with no downstream task benchmarks. The deduplicated held-out set keeps only documents that appear nowhere else in the corpus, so it leans away from boilerplate and syndicated news. See the model card for details and sample outputs.

## License

Apache-2.0.

## Acknowledgements

Built on the cleaned FineWeb-2 corpus (Hugging Face). Architecture and training informed by the open from-scratch LM literature (GPT-2; the RoPE / RMSNorm / SwiGLU lines of work; and public educational implementations of transformer training).
