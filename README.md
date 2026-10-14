# Uzbek GPT — a language model from scratch

A 103M-parameter, decoder-only language model pretrained from scratch on Uzbek (Latin script), with a hand-written transformer (RMSNorm, RoPE, SwiGLU, multi-head causal attention), a custom Uzbek tokenizer, and a single-GPU training run.

On held-out Uzbek text it predicts better than mGPT-1.3B (13× its size, used as-is or fine-tuned on Uzbek) and than four current Qwen models of up to 1.7B parameters. A controlled ablation shows where that advantage comes from: training on Uzbek text. The tokenizer doesn't make the model more accurate; it makes it smaller and cheaper.

- **Paper:** [`paper/Research_Islombek_Turdiyev_v18.pdf`](paper/Research_Islombek_Turdiyev_v18.pdf) · one-page summary: [`paper/one_page_abstract.pdf`](paper/one_page_abstract.pdf)
- **Model weights:** [IslombekT/uzbek-gpt-103m](https://huggingface.co/IslombekT/uzbek-gpt-103m)
- **Tokenizer:** [IslombekT/uzbek-bpe-16k](https://huggingface.co/IslombekT/uzbek-bpe-16k) (16,384-token byte-level BPE)

## Try it (no training, no GPU)

```bash
git clone https://github.com/TurdiyevIslombek/uzbek-gpt-from-scratch.git
cd uzbek-gpt-from-scratch
pip install -r requirements.txt

# download the trained weights (~425 MB)
curl -LO https://huggingface.co/IslombekT/uzbek-gpt-103m/resolve/main/model.safetensors

# generate
python generate.py --ckpt model.safetensors --prompt "Oʻzbekiston "
```

The tokenizer downloads automatically from the Hub.

## Why

Language models split Uzbek into far more tokens than English, which makes them slower and more expensive for Uzbek speakers. I wanted to know what matters more for a small language: a tokenizer built for it, or training a model on it from scratch. So I built both, openly, as a foundation for further Uzbek NLP work.

## What's here

| Path | What it is |
|---|---|
| `model.py` | The transformer: RoPE, RMSNorm, SwiGLU, fused causal attention (no output projection after the heads). |
| `tokenize_data.py` | Tokenizes FineWeb-2 `uzn_Latn` into `train.bin` / `val.bin` (uint16, `<\|endoftext\|>`-separated, 90/10 split by position). |
| `train.py` | Pretraining loop: bf16, gradient accumulation, cosine LR, gradient clipping, `torch.compile`, checkpointing. |
| `generate.py` | Loads a checkpoint (`.pt` or `.safetensors`) and samples Uzbek text. |
| `experiments/` | Every evaluation, the audit of the evaluation, and the alphabet-reform study. See [`experiments/README.md`](experiments/README.md). |
| `paper/` | The paper and the one-page summary. |

## Architecture and training

| | |
|---|---|
| Layers / width / heads | 12 / 768 / 12 |
| Vocabulary / context | 16,384 / 1,024 |
| Parameters | 103.06M |
| Data | FineWeb-2 `uzn_Latn`: ~1.23M documents → ~1.06B tokens (955M train / 106M val) |
| Optimizer | AdamW (0.9, 0.95; weight decay 0.1), 400 warmup steps, cosine 3e-4 → 3e-5 |
| Batch | 192 sequences × 1,024 tokens (~197K tokens/step), 9,700 steps (~2 epochs) |
| Hardware / time / cost | 1× RTX 4090, ~3.4 hours, ~$3.60 |

The best validation loss was 3.059, but `val.bin` turned out to be about 39% duplicated from `train.bin` (web pages repeat), so that figure is optimistic. Use the bits-per-byte results below instead. Full log: `training_log.txt`.

## Results

All models are scored by **bits-per-byte**: how many bits a model needs per byte of text. Unlike perplexity it doesn't depend on the tokenizer, so it compares models with different tokenizers fairly. Lower is better.

The test set is 474 validation documents that share no 24-token passage with the training data, in their **original web spelling**. Every model scores the same ~800-byte spans with the same preceding context. Confidence intervals come from a paired bootstrap that resamples the 474 documents (10,000 resamples).

### Model comparison

| Model | Size | Bits/byte ↓ | Gap to mine [95% CI] |
|---|---|---|---|
| **uzbek-gpt-103m** (from scratch) | 0.10B | **1.038** | — |
| mGPT-1.3B + QLoRA (1M Uzbek tokens) | 1.3B | 1.077 | +0.039 [0.031, 0.049] |
| mGPT-1.3B, zero-shot | 1.3B | 1.080 | +0.042 [0.034, 0.051] |
| Qwen3-1.7B-Base, zero-shot | 1.72B | 1.532 | +0.494 [0.470, 0.516] |
| Qwen3-0.6B-Base, zero-shot | 0.60B | 1.905 | +0.867 [0.840, 0.894] |
| Qwen2.5-1.5B, zero-shot | 1.54B | 1.948 | +0.909 [0.883, 0.935] |
| Qwen2.5-0.5B, zero-shot | 0.49B | 2.357 | +1.319 [1.289, 1.347] |

My model needs about 4% fewer bits than fine-tuned mGPT and 32% fewer than the best Qwen model, which is 17× larger. The Qwen models were not fine-tuned on Uzbek and may have seen some of these web pages in training; both would favour them.

### Tokenizer or training data?

To separate the two, I trained a second model with the same architecture and settings on **exactly the same training documents**, but with mGPT's 100k tokenizer instead of mine.

| | uzbek-gpt-103m | Ablation (mGPT's tokenizer) |
|---|---|---|
| Parameters | 103M | 232M |
| Training steps (two passes) | ~9,700 | 16,822 |
| Training compute (estimate) | 1× | ~2.8× |
| Bits/byte, web spelling | 1.038 | **1.031** |
| Bits/byte, normalized spelling | **1.028** | 1.033 |

Ablation minus mine, 95% CI: web −0.007 [−0.009, −0.005]; normalized +0.005 [+0.002, +0.008].

The two models reach **comparable quality**. Each wins on the spelling it trained on: my tokenizer normalizes apostrophes, so my model learned the normalized spelling; mGPT's tokenizer doesn't, so the ablation learned web spelling. Both are far ahead of mGPT, which shares the ablation's tokenizer, so **the advantage over mGPT comes from pretraining on a billion tokens of Uzbek, not from the tokenizer.**

What the tokenizer buys is **efficiency**. My model reached comparable quality with less than half the parameters and about a third of the training compute, and it uses fewer tokens for every text:

| Tokenizer | Vocab | Tokens per word ↓ |
|---|---|---|
| **uzbek-bpe-16k** | 16,384 | **1.749** |
| XLM-RoBERTa | 250,002 | 2.218 |
| GPT-4o `o200k_base` | 200,019 | 2.578 |
| mGPT | 100,000 | 3.159 |
| GPT-3.5/4 `cl100k_base` | 100,277 | 3.187 |
| GPT-2 | 50,257 | 3.426 |

About half the tokens of GPT-2, 45% fewer than mGPT's tokenizer, 21% fewer than XLM-R — roughly half the cost and twice the usable context. (All 11 tokenizers are in the paper, measured on the same 474 documents.)

The ablation held the training **data** fixed, not the compute, which favours the ablation. It was trained on two free Kaggle T4 GPUs in fp16 (the original ran on one RTX 4090 in bf16) with [`08k_retrain_ablation_kaggle.py`](experiments/audit-2026-09/08k_retrain_ablation_kaggle.py).

### Does fine-tuning catch up?

| Adaptation tokens | 0 | 1M | 10M |
|---|---|---|---|
| mGPT-1.3B + QLoRA, bits/byte | 1.080 | 1.077 | 1.064 |

It keeps improving (0.013 per tenfold increase) but is still 0.026 behind at 10M tokens. A rough extrapolation from these two points suggests catching up could take on the order of 1 billion adaptation tokens — about as much Uzbek text as training from scratch. The ablation shows why: mGPT's tokenizer isn't the problem (it reaches 1.031 with a billion Uzbek tokens); mGPT lacks Uzbek text.

### What it writes

One continuation per prompt, fixed seed, not selected (first sentence):

> **Toshkent shahrida** ochilgan filialda faoliyat yuritib kelayotgan yoshlar bilan uchrashdi.
> *(He) met with young people working at a branch that opened in Tashkent.*

The output is grammatical and reads like Uzbek news and legal text, but it has no understanding behind it: longer outputs invent names and products, and one of four samples doesn't make sense. All samples are in `experiments/audit-2026-09/logs/07_final_checks.log`.

## Evaluation audit (September 2026)

Earlier versions of this README reported 1.105 bits/byte and claimed the whole advantage came from the tokenizer. An audit of my own work found four problems, each of which made my results look better than they were:

1. **The test set was inside the training data.** It was built from the start of the corpus, and `tokenize_data.py` splits train/validation by position. 11 of 12 test passages were found in `train.bin`.
2. **The validation split was ~39% duplicated** from training data. Fixed by fingerprinting every 24-token window of `train.bin` and keeping only validation documents with zero overlap (474 of 780).
3. **The test text was in my tokenizer's spelling.** `val.bin` stores text after my tokenizer's apostrophe normalization, which favoured my model. Fixed by recovering the original web text of all 474 documents from FineWeb-2 (exact match) and re-scoring every model on it.
4. **The first tokenizer ablation didn't run as described.** Its data cap counted words instead of tokens, so it read the whole corpus, including the validation documents the test set comes from; and it stopped at step 10,000 of 18,693, before its learning rate decayed. It scored 1.048, which made the tokenizer look responsible for a quarter of the advantage. Retrained from scratch on exactly my model's training documents (verified against `train.bin`, 0 of 16 test passages in its data), it scores 1.031.

Running the corrected code on the old test set reproduces the old numbers exactly (1.1050 / 1.1628 / 1.1579), so every change can be checked. Earlier versions also resampled the 2,058 scoring spans one by one; the intervals above resample whole documents instead, because spans from one document are correlated. That makes them up to about three times wider, and every difference still excludes zero ([`09_document_bootstrap.py`](experiments/audit-2026-09/09_document_bootstrap.py)). The tokenizer's share of the quality advantage fell from "all of it" to none; the corrected numbers are the ones above. If you evaluate a model on web data, check your test set against your training data, against how the text is really written, and check that each experiment actually ran the way you describe it.

## Limitations

A small base model: fluent Uzbek, but not a chat model, not factually reliable, and it can repeat without a repetition penalty. Latin-script Uzbek only. Evaluation is intrinsic (bits-per-byte) with no downstream tasks. One training run per model, so the small (0.005–0.007) differences between my model and the ablation haven't been checked across seeds. The ablation matched training data, not compute. The tokenizer's normalization misses the left quote `‘`, the most common way people type the mark in *oʻ* (about 34% of cases). The test set keeps only non-duplicated documents, so it leans away from boilerplate.

## Acknowledgements

Built on FineWeb-2 (Hugging Face). Architecture informed by GPT-2 and the RoPE / RMSNorm / SwiGLU literature. I used Claude (Anthropic) to learn the methods, write and debug code, and edit writing. It also flagged two of the four problems in the evaluation audit (the test-set overlap and the mismatch in the first ablation run), which I confirmed. I ran every check, trained every model and ran every experiment myself.

## License

Apache-2.0. The evaluation text in `experiments/audit-2026-09/results/` comes from FineWeb-2 (ODC-By 1.0).
