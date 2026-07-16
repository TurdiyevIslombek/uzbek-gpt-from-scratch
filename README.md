# Uzbek GPT — a language model from scratch

A ~103M-parameter, decoder-only GPT-style language model **pretrained from scratch on Uzbek (Latin script)** — with the transformer architecture implemented by hand (RMSNorm + RoPE + SwiGLU + multi-head causal attention), a custom Uzbek tokenizer, and a single-GPU pretraining run.

Scored by bits-per-byte on held-out Uzbek text, this 103M model beats a 1.3B multilingual model (mGPT) — about 13× its size — and two controlled experiments show the advantage comes from the **tokenizer**, not from training from scratch.

**Model weights:** [`IslombekT/uzbek-gpt-103m`](https://huggingface.co/IslombekT/uzbek-gpt-103m) on Hugging Face
**Tokenizer:** [`IslombekT/uzbek-bpe-16k`](https://huggingface.co/IslombekT/uzbek-bpe-16k) (16,384-vocab byte-level BPE)

This repository is the **code**: the architecture, the data pipeline, the training loop, a generation script, and the evaluation experiments. It is meant to be readable end to end.

## Try it (no training, no GPU)

The trained weights are published on Hugging Face. To generate Uzbek text from them you only need this repo's code plus the weights file:

```bash
git clone https://github.com/TurdiyevIslombek/uzbek-gpt-from-scratch.git
cd uzbek-gpt-from-scratch
pip install -r requirements.txt

# download the trained weights (~425 MB) from the Hugging Face model
wget https://huggingface.co/IslombekT/uzbek-gpt-103m/resolve/main/model.safetensors

# generate
python generate.py --ckpt model.safetensors --prompt "Oʻzbekiston "
```

`generate.py` loads `model.safetensors` directly — no checkpoint of your own and no training required. The tokenizer downloads automatically from the Hub. A one-click runnable version is also available as a [Kaggle notebook](#).

## Why

Uzbek is a low-resource language with little dedicated open tooling. The goal was to build a usable Uzbek base model the hard way — understanding every component rather than fine-tuning an existing model — and release it openly as a foundation for further Uzbek NLP work.

## What's here

| File | What it does |
|---|---|
| `model.py` | The from-scratch transformer: RoPE, RMSNorm, SwiGLU, fused causal attention, the GPT module. |
| `tokenize_data.py` | Tokenizes the FineWeb-2 `uzn_Latn` split into `train.bin` / `val.bin` (uint16, `<|endoftext|>`-separated, 90/10 split). |
| `train.py` | The pretraining loop: bf16 mixed precision, gradient accumulation, cosine LR schedule, gradient clipping, `torch.compile`, checkpointing. |
| `generate.py` | Loads a checkpoint (`.pt` or `.safetensors`) and samples Uzbek text. |
| `experiments/` | The evaluation: tokenizer benchmark, bits-per-byte comparison, and the two controlled experiments. |

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

- **Data:** Uzbek Latin split of [FineWeb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) — ~1.23M documents → ~1.06B tokens (955M train / 106M val).
- **Hardware:** 1× RTX 4090 (24 GB).
- **Setup:** AdamW (0.9, 0.95; wd 0.1), warmup 400 → cosine decay (3e-4 → 3e-5), effective batch 192 sequences (~197K tokens/step), 9,700 steps (~2 epochs).
- **Result:** ~3.4 h wall-clock, final validation loss **3.059**. Full log in [`training_log.txt`](training_log.txt).

## Results

Full details and scripts in [`experiments/`](experiments/). All models scored by **bits-per-byte** (how well a model predicts raw text, independent of its tokenizer) on the same held-out Uzbek set — lower is better.

**Tokenizer efficiency.** `uzbek-bpe-16k` has the lowest fertility (tokens per word) of ten benchmarked tokenizers:

| Tokenizer | Vocab | Fertility (tok/word) ↓ |
|---|---:|---:|
| **uzbek-bpe-16k (this repo)** | 16,384 | **1.839** |
| XLM-RoBERTa | 250,002 | 2.334 |
| GPT-4o `o200k_base` | 200,019 | 2.724 |
| mBERT | 119,547 | 2.906 |
| GPT-2 | 50,257 | 3.584 |

It beats XLM-R, whose vocabulary is 15× larger — efficiency comes from language-specific design, not vocabulary size. (Full 10-tokenizer table in `experiments/`.)

**Model comparison.** The 103M from-scratch model beats mGPT-1.3B on bits-per-byte:

| Model | Params | bits/byte ↓ |
|---|---:|---:|
| **uzbek-gpt-103m (from scratch)** | 103M | **1.105** |
| mGPT-1.3B (base, zero-shot) | 1.3B | 1.163 |
| mGPT-1.3B + QLoRA | 1.3B | 1.147 |

**Two controlled experiments** test whether this is fair and what causes it:

- **Fair-budget test.** Re-training mGPT+QLoRA on 10× more Uzbek data (1M → 10M tokens) improves it by only 0.002 bpb; it plateaus at ~1.120, still above 1.105. A bootstrap gives the from-scratch advantage over the zero-shot baseline as 0.058 bpb (95% CI [0.037, 0.078]) — a resolvable gap, not noise. More data does not close it.
- **Tokenizer ablation.** The same architecture trained from scratch on the same text but with mGPT's tokenizer — a *larger*, 232M-parameter model — scores 1.158 bpb, worse than this 103M model. With everything else held constant, the tokenizer is the deciding factor.

**Takeaway:** for a low-resource language, the tokenizer is the highest-leverage investment — it outweighs both model size and the choice to train from scratch.

## Limitations

A small base model on ~1B tokens: it produces fluent, grammatical Uzbek but is **not** an instruction/chat model, is not factually reliable, and can repeat without a repetition penalty. It covers Latin-script Uzbek only, not Cyrillic. See the [model card](https://huggingface.co/IslombekT/uzbek-gpt-103m) for details and sample outputs.

## License

Apache-2.0.

## Acknowledgements

Built on the cleaned **FineWeb-2** corpus (Hugging Face). Architecture and training informed by the open from-scratch LM literature (GPT-2; the RoPE / RMSNorm / SwiGLU lines of work; and public educational implementations of transformer training).
